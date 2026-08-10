from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from tabula import logger

PopulationCenter = Literal["mean", "median"]
CalibrationSpace = Literal["input", "output"]
DiscriminatorInputTransform = Literal["none", "cohort_mean_fold_change"]


class DiscriminatorMLP(nn.Sequential):
    """Predict normalized age and identity scores from a cell expression profile.

    This lineage-specific discriminator follows the architecture described in
    the rejuvenation-factor analysis: a linear layer, a Tanh activation, and a
    second linear layer with two outputs corresponding to the predicted age
    normalized age score and normalized identity score. The sequential layout preserves the state-dict
    keys used by the trained discriminator checkpoints.

    Args:
        in_features: Number of input genes.
        hidden_features: Number of hidden units. The manuscript setting is 64.
    """

    def __init__(self, in_features: int, hidden_features: int = 64) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive.")
        super().__init__(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 2),
        )


def compute_cohort_mean_fold_change(
    expression: Tensor,
    cohort_mean: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Compute cohort-mean fold-change values for discriminator scoring.

    Each gene value is divided by its mean across the matched young and aged
    fibroblast cohort. Only the denominator is protected near zero; the input
    expression matrix itself is not clamped.

    Args:
        expression: Cell-by-gene expression matrix with shape
            ``(n_cells, n_genes)``.
        cohort_mean: Per-gene cohort mean with shape ``(n_genes,)`` or
            ``(1, n_genes)``.
        epsilon: Minimum absolute denominator value.

    Returns:
        Cohort-mean fold-change matrix with the same shape as ``expression``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional tensor.")
    if cohort_mean.ndim not in (1, 2):
        raise ValueError("cohort_mean must be one- or two-dimensional.")
    if cohort_mean.numel() != expression.shape[1]:
        raise ValueError(
            "cohort_mean must contain one value for each expression gene."
        )

    cohort_mean = cohort_mean.reshape(1, -1).to(
        device=expression.device,
        dtype=expression.dtype,
    )
    denominator = torch.sign(cohort_mean) * torch.clamp(
        torch.abs(cohort_mean),
        min=epsilon,
    )
    denominator = torch.where(
        denominator == 0,
        torch.full_like(denominator, epsilon),
        denominator,
    )
    return expression / denominator


def compute_lower_bound_age_score(
    young_age_score: float,
    aged_age_score: float,
    young_age: float,
    aged_age: float,
    lower_bound_age: float,
) -> float:
    """Compute the lower-bound age score by chronological-age calibration.

    The target is obtained by linear interpolation or extrapolation between
    the population-level normalized age-score centers of the matched young and aged
    fibroblast groups.

    Args:
        young_age_score: Population center of the young-group normalized age scores.
        aged_age_score: Population center of the aged-group normalized age scores.
        young_age: Chronological age of the young fibroblast group.
        aged_age: Chronological age of the aged fibroblast group.
        lower_bound_age: Chronological age represented by the rejuvenation
            lower boundary.

    Returns:
        Normalized age-score value corresponding to ``lower_bound_age``.
    """
    if aged_age == young_age:
        return float(young_age_score)
    slope = (aged_age_score - young_age_score) / (aged_age - young_age)
    return float(
        young_age_score + slope * (lower_bound_age - young_age)
    )


def predict_rejuvenation_trajectory(
    aged_expression: Tensor,
    young_expression: Tensor,
    gene_ids: Tensor,
    reconstruction_model: nn.Module,
    discriminator: nn.Module,
    young_age: float,
    aged_age: float,
    lower_bound_age: float,
    *,
    identity_preservation_weight: float = 5.0,
    learning_rate: float = 1e-4,
    n_steps: int = 3000,
    population_center: PopulationCenter = "median",
    target_calibration_space: CalibrationSpace = "output",
    discriminator_input_transform: DiscriminatorInputTransform = (
        "cohort_mean_fold_change"
    ),
    fold_change_epsilon: float = 1e-8,
    reconstruction_batch_size: Optional[int] = 64,
    log_every: int = 0,
) -> Dict[str, Tensor]:
    """Predict a Tabula-guided rejuvenation trajectory for aged fibroblasts.

    The aged-cell expression matrix is treated as the optimization variable,
    while both the Tabula reconstruction model and the lineage-specific
    discriminator MLP remain frozen. At each rejuvenation step, the current
    input-space matrix is reconstructed by Tabula, and the output-space matrix
    is scored by the discriminator. The loss moves the predicted normalized age score
    toward the lower-bound normalized age score while preserving fibroblast identity by
    moving the normalized identity score toward the young-group center.

    In this API, ``input space`` denotes the Tabula-guided expression matrix
    that is directly optimized. ``Output space`` denotes the expression matrix
    reconstructed by Tabula from that input. Gradients are computed from the
    output-space score objective and propagated through the frozen
    reconstruction model to the input-space matrix.

    Gene-wise trajectory statistics are derived from the mean stepwise change
    across cells between consecutive rejuvenation steps. ``pos_path`` and
    ``neg_path`` accumulate the magnitudes of positive and negative stepwise
    changes, whereas ``n_up`` and ``n_down`` count how consistently a gene
    increases or decreases. All statistics are computed separately in input
    space and output space.

    Args:
        aged_expression: Initial aged-cell expression matrix with shape
            ``(n_aged_cells, n_genes)``. This matrix defines the initial input
            space and is copied before optimization.
        young_expression: Matched young-cell expression matrix with shape
            ``(n_young_cells, n_genes)``.
        gene_ids: Vocabulary IDs for the input genes, ordered consistently with
            the expression columns.
        reconstruction_model: Frozen Tabula model returning a
            ``reconstruction`` item from
            ``forward(..., head="reconstruction")``.
        discriminator: Frozen lineage-specific MLP returning predicted normalized
            age and identity scores in columns zero and one, respectively.
        young_age: Chronological age of the matched young fibroblast group.
        aged_age: Chronological age of the matched aged fibroblast group.
        lower_bound_age: Chronological age used to define the lower-bound age
            normalized age score for rejuvenation calibration.
        identity_preservation_weight: Weight assigned to the
            identity-preservation loss term. The manuscript setting is 5.0.
        learning_rate: Adam learning rate for the input-space expression
            matrix.
        n_steps: Number of predicted rejuvenation steps.
        population_center: Population-level summary statistic used to define
            score-distribution centers. The manuscript implementation uses the
            median.
        target_calibration_space: Space used to evaluate matched young and aged
            groups when defining target scores. ``"output"`` corresponds to
            Tabula-reconstructed expression and is the manuscript default.
        discriminator_input_transform: Representation passed to the
            discriminator. ``"cohort_mean_fold_change"`` uses gene-wise means
            computed from the matched young and aged cohort in the relevant
            space.
        fold_change_epsilon: Minimum absolute cohort-mean denominator.
        reconstruction_batch_size: Batch size used to reconstruct the matched
            young and aged groups during target calibration. Set to ``None`` to
            reconstruct each matrix in one call.
        log_every: Log progress every this many steps. Zero disables step-wise
            logging.

    Returns:
        Dictionary containing the optimized input- and output-space matrices,
        the lower-bound normalized age and young normalized identity targets, score histories, net
        change histories, and trajectory-derived gene statistics in both
        spaces.
    """
    _validate_trajectory_inputs(
        aged_expression=aged_expression,
        young_expression=young_expression,
        gene_ids=gene_ids,
        identity_preservation_weight=identity_preservation_weight,
        learning_rate=learning_rate,
        n_steps=n_steps,
        population_center=population_center,
        target_calibration_space=target_calibration_space,
        discriminator_input_transform=discriminator_input_transform,
        fold_change_epsilon=fold_change_epsilon,
        reconstruction_batch_size=reconstruction_batch_size,
        log_every=log_every,
    )

    device = aged_expression.device
    aged_expression = aged_expression.detach().to(
        device=device,
        dtype=torch.float32,
    )
    young_expression = young_expression.detach().to(
        device=device,
        dtype=torch.float32,
    )
    gene_ids = gene_ids.detach().to(device=device, dtype=torch.int64)

    reconstruction_model = reconstruction_model.to(device)
    discriminator = discriminator.to(device)
    reconstruction_model.eval()
    discriminator.eval()
    reconstruction_model.requires_grad_(False)
    discriminator.requires_grad_(False)

    with torch.no_grad():
        aged_output_expression = _reconstruct_output_expression_batched(
            reconstruction_model,
            aged_expression,
            gene_ids,
            reconstruction_batch_size,
        )
        young_output_expression = _reconstruct_output_expression_batched(
            reconstruction_model,
            young_expression,
            gene_ids,
            reconstruction_batch_size,
        )

    input_cohort_mean: Optional[Tensor] = None
    output_cohort_mean: Optional[Tensor] = None
    if discriminator_input_transform == "cohort_mean_fold_change":
        input_cohort_mean = torch.cat(
            [aged_expression, young_expression],
            dim=0,
        ).mean(dim=0, keepdim=True)
        output_cohort_mean = torch.cat(
            [aged_output_expression, young_output_expression],
            dim=0,
        ).mean(dim=0, keepdim=True)

    with torch.no_grad():
        aged_input_scores = _predict_age_identity_scores(
            discriminator,
            _prepare_discriminator_input(
                aged_expression,
                discriminator_input_transform,
                input_cohort_mean,
                fold_change_epsilon,
            ),
        )
        young_input_scores = _predict_age_identity_scores(
            discriminator,
            _prepare_discriminator_input(
                young_expression,
                discriminator_input_transform,
                input_cohort_mean,
                fold_change_epsilon,
            ),
        )
        aged_output_scores = _predict_age_identity_scores(
            discriminator,
            _prepare_discriminator_input(
                aged_output_expression,
                discriminator_input_transform,
                output_cohort_mean,
                fold_change_epsilon,
            ),
        )
        young_output_scores = _predict_age_identity_scores(
            discriminator,
            _prepare_discriminator_input(
                young_output_expression,
                discriminator_input_transform,
                output_cohort_mean,
                fold_change_epsilon,
            ),
        )

    aged_input_center = _population_score_center(
        aged_input_scores,
        population_center,
    )
    young_input_center = _population_score_center(
        young_input_scores,
        population_center,
    )
    aged_output_center = _population_score_center(
        aged_output_scores,
        population_center,
    )
    young_output_center = _population_score_center(
        young_output_scores,
        population_center,
    )

    if target_calibration_space == "input":
        young_calibration_center = young_input_center
        aged_calibration_center = aged_input_center
    else:
        young_calibration_center = young_output_center
        aged_calibration_center = aged_output_center

    lower_bound_age_score = compute_lower_bound_age_score(
        young_age_score=float(young_calibration_center[0].item()),
        aged_age_score=float(aged_calibration_center[0].item()),
        young_age=young_age,
        aged_age=aged_age,
        lower_bound_age=lower_bound_age,
    )
    young_identity_score = float(young_calibration_center[1].item())

    n_genes = aged_expression.shape[1]
    optimized_input_expression = nn.Parameter(aged_expression.clone())
    optimizer = Adam([optimized_input_expression], lr=learning_rate)

    pos_path_input = torch.zeros(n_genes, device=device)
    neg_path_input = torch.zeros(n_genes, device=device)
    pos_path_output = torch.zeros(n_genes, device=device)
    neg_path_output = torch.zeros(n_genes, device=device)
    n_up_input = torch.zeros(n_genes, dtype=torch.int32, device=device)
    n_down_input = torch.zeros(n_genes, dtype=torch.int32, device=device)
    n_up_output = torch.zeros(n_genes, dtype=torch.int32, device=device)
    n_down_output = torch.zeros(n_genes, dtype=torch.int32, device=device)

    previous_output_expression = aged_output_expression.detach().clone()
    loss_history = []
    input_score_center_history = [aged_input_center]
    output_score_center_history = [aged_output_center]
    net_change_input_history = [torch.zeros(n_genes, device=device)]
    net_change_output_history = [torch.zeros(n_genes, device=device)]

    logger.info(
        "Starting Tabula-guided rejuvenation trajectory prediction for %d steps.",
        n_steps,
    )

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        previous_input_expression = optimized_input_expression.detach().clone()

        output_expression_for_loss = _reconstruct_output_expression(
            reconstruction_model,
            optimized_input_expression,
            gene_ids,
        )
        predicted_scores = _predict_age_identity_scores(
            discriminator,
            _prepare_discriminator_input(
                output_expression_for_loss,
                discriminator_input_transform,
                output_cohort_mean,
                fold_change_epsilon,
            ),
        )
        age_loss = (
            predicted_scores[:, 0] - lower_bound_age_score
        ).pow(2).mean()
        identity_loss = (
            predicted_scores[:, 1] - young_identity_score
        ).pow(2).mean()
        loss = age_loss + identity_preservation_weight * identity_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_output_expression = _reconstruct_output_expression(
                reconstruction_model,
                optimized_input_expression,
                gene_ids,
            )
            current_input_scores = _predict_age_identity_scores(
                discriminator,
                _prepare_discriminator_input(
                    optimized_input_expression,
                    discriminator_input_transform,
                    input_cohort_mean,
                    fold_change_epsilon,
                ),
            )
            current_output_scores = _predict_age_identity_scores(
                discriminator,
                _prepare_discriminator_input(
                    current_output_expression,
                    discriminator_input_transform,
                    output_cohort_mean,
                    fold_change_epsilon,
                ),
            )

            stepwise_change_input = (
                optimized_input_expression - previous_input_expression
            ).mean(dim=0)
            stepwise_change_output = (
                current_output_expression - previous_output_expression
            ).mean(dim=0)

            pos_path_input += torch.clamp(stepwise_change_input, min=0.0)
            neg_path_input += torch.clamp(-stepwise_change_input, min=0.0)
            pos_path_output += torch.clamp(stepwise_change_output, min=0.0)
            neg_path_output += torch.clamp(-stepwise_change_output, min=0.0)
            n_up_input += (stepwise_change_input > 0).to(torch.int32)
            n_down_input += (stepwise_change_input < 0).to(torch.int32)
            n_up_output += (stepwise_change_output > 0).to(torch.int32)
            n_down_output += (stepwise_change_output < 0).to(torch.int32)

            net_change_input_history.append(
                (optimized_input_expression - aged_expression).mean(dim=0)
            )
            net_change_output_history.append(
                (current_output_expression - aged_output_expression).mean(dim=0)
            )
            input_score_center_history.append(
                _population_score_center(
                    current_input_scores,
                    population_center,
                )
            )
            output_score_center_history.append(
                _population_score_center(
                    current_output_scores,
                    population_center,
                )
            )
            previous_output_expression = current_output_expression.detach().clone()
            loss_history.append(loss.detach())

        if log_every > 0 and step % log_every == 0:
            logger.info(
                "Rejuvenation trajectory step %d: loss=%.6f.",
                step,
                float(loss.item()),
            )

    logger.info("Tabula-guided rejuvenation trajectory prediction finished.")

    input_score_center_history = torch.stack(input_score_center_history)
    output_score_center_history = torch.stack(output_score_center_history)
    net_change_input_history = torch.stack(net_change_input_history)
    net_change_output_history = torch.stack(net_change_output_history)

    return {
        "optimized_input_expression": optimized_input_expression.detach().clone(),
        "optimized_output_expression": previous_output_expression.detach().clone(),
        "lower_bound_age_score": torch.tensor(
            lower_bound_age_score,
            device=device,
        ),
        "young_identity_score": torch.tensor(
            young_identity_score,
            device=device,
        ),
        "steps": torch.arange(n_steps + 1, device=device),
        "loss_history": torch.stack(loss_history),
        "center_age_score_input": input_score_center_history[:, 0],
        "center_identity_score_input": input_score_center_history[:, 1],
        "center_age_score_output": output_score_center_history[:, 0],
        "center_identity_score_output": output_score_center_history[:, 1],
        "net_change_input": net_change_input_history[-1],
        "net_change_output": net_change_output_history[-1],
        "net_change_input_history": net_change_input_history,
        "net_change_output_history": net_change_output_history,
        "pos_path_input": pos_path_input,
        "neg_path_input": neg_path_input,
        "pos_path_output": pos_path_output,
        "neg_path_output": neg_path_output,
        "n_up_input": n_up_input,
        "n_down_input": n_down_input,
        "n_up_output": n_up_output,
        "n_down_output": n_down_output,
    }


def _validate_trajectory_inputs(
    aged_expression: Tensor,
    young_expression: Tensor,
    gene_ids: Tensor,
    identity_preservation_weight: float,
    learning_rate: float,
    n_steps: int,
    population_center: str,
    target_calibration_space: str,
    discriminator_input_transform: str,
    fold_change_epsilon: float,
    reconstruction_batch_size: Optional[int],
    log_every: int,
) -> None:
    if aged_expression.ndim != 2 or young_expression.ndim != 2:
        raise ValueError(
            "aged_expression and young_expression must be two-dimensional."
        )
    if aged_expression.shape[1] != young_expression.shape[1]:
        raise ValueError(
            "Aged and young expression matrices must use the same genes."
        )
    if aged_expression.shape[0] == 0 or young_expression.shape[0] == 0:
        raise ValueError(
            "Aged and young expression matrices must contain at least one cell."
        )
    if gene_ids.ndim != 1 or gene_ids.numel() != aged_expression.shape[1]:
        raise ValueError(
            "gene_ids must contain one ID for each expression gene."
        )
    if identity_preservation_weight < 0:
        raise ValueError(
            "identity_preservation_weight must be nonnegative."
        )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if population_center not in ("mean", "median"):
        raise ValueError("population_center must be 'mean' or 'median'.")
    if target_calibration_space not in ("input", "output"):
        raise ValueError(
            "target_calibration_space must be 'input' or 'output'."
        )
    if discriminator_input_transform not in (
        "none",
        "cohort_mean_fold_change",
    ):
        raise ValueError(
            "discriminator_input_transform must be 'none' or "
            "'cohort_mean_fold_change'."
        )
    if fold_change_epsilon <= 0:
        raise ValueError("fold_change_epsilon must be positive.")
    if reconstruction_batch_size is not None and reconstruction_batch_size <= 0:
        raise ValueError(
            "reconstruction_batch_size must be positive or None."
        )
    if log_every < 0:
        raise ValueError("log_every must be nonnegative.")


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def _reconstruct_output_expression(
    model: nn.Module,
    input_expression: Tensor,
    gene_ids: Tensor,
) -> Tensor:
    model_dtype = _module_dtype(model, input_expression.dtype)
    output = model(
        genes=gene_ids,
        values=input_expression.to(dtype=model_dtype).contiguous(),
        head="reconstruction",
    )
    if not isinstance(output, dict) or "reconstruction" not in output:
        raise ValueError(
            "reconstruction_model must return a dictionary containing "
            "'reconstruction'."
        )
    output_expression = output["reconstruction"]
    if output_expression.shape != input_expression.shape:
        raise ValueError(
            "Output-space expression must have the same shape as input-space "
            "expression."
        )
    return output_expression.float()


def _reconstruct_output_expression_batched(
    model: nn.Module,
    input_expression: Tensor,
    gene_ids: Tensor,
    batch_size: Optional[int],
) -> Tensor:
    if batch_size is None or input_expression.shape[0] <= batch_size:
        return _reconstruct_output_expression(model, input_expression, gene_ids)
    output_batches = []
    for start in range(0, input_expression.shape[0], batch_size):
        output_batches.append(
            _reconstruct_output_expression(
                model,
                input_expression[start : start + batch_size],
                gene_ids,
            )
        )
    return torch.cat(output_batches, dim=0)


def _prepare_discriminator_input(
    expression: Tensor,
    transform: DiscriminatorInputTransform,
    cohort_mean: Optional[Tensor],
    epsilon: float,
) -> Tensor:
    if transform == "none":
        return expression
    if cohort_mean is None:
        raise ValueError(
            "cohort_mean is required for cohort-mean fold-change scoring."
        )
    return compute_cohort_mean_fold_change(
        expression,
        cohort_mean,
        epsilon,
    )


def _predict_age_identity_scores(
    discriminator: nn.Module,
    expression: Tensor,
) -> Tensor:
    discriminator_dtype = _module_dtype(discriminator, expression.dtype)
    scores = discriminator(expression.to(dtype=discriminator_dtype))
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(
            "discriminator must return a tensor with shape (n_cells, 2)."
        )
    return scores.float()


def _population_score_center(
    scores: Tensor,
    mode: PopulationCenter,
) -> Tensor:
    if mode == "mean":
        return scores.mean(dim=0)
    return scores.median(dim=0).values
