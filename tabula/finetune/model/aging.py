"""Frozen-Tabula-guided optimization of aging expression trajectories."""

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from tabula import logger

CenterMode = Literal["mean", "median"]
CalibrationSpace = Literal["raw", "recon"]
ScoreInputTransform = Literal["none", "fc"]


class AgingScoreRegressor(nn.Sequential):
    """Predict age and identity scores from expression profiles.

    The layer layout matches the regressors used in the aging experiments, so
    their existing state dictionaries can be loaded without renaming keys.

    Args:
        in_features: Number of input genes.
        hidden_features: Number of hidden units.
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


def compute_fold_change(
    expression: Tensor,
    reference_mean: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Transform expression values into fold-change values.

    Only the denominator is protected from values close to zero. The input
    expression is not clamped, and the sign of a nonzero reference mean is
    preserved.

    Args:
        expression: Expression matrix with shape ``(n_cells, n_genes)``.
        reference_mean: Per-gene reference mean with shape ``(n_genes,)`` or
            ``(1, n_genes)``.
        epsilon: Minimum absolute denominator value.

    Returns:
        Fold-change-transformed expression with the same shape as
        ``expression``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional tensor.")
    if reference_mean.ndim not in (1, 2):
        raise ValueError("reference_mean must be one- or two-dimensional.")
    if reference_mean.numel() != expression.shape[1]:
        raise ValueError(
            "reference_mean must contain one value for each expression gene."
        )

    reference_mean = reference_mean.reshape(1, -1).to(
        device=expression.device,
        dtype=expression.dtype,
    )
    denominator = torch.sign(reference_mean) * torch.clamp(
        torch.abs(reference_mean),
        min=epsilon,
    )
    denominator = torch.where(
        denominator == 0,
        torch.full_like(denominator, epsilon),
        denominator,
    )
    return expression / denominator


def calibrate_age_score(
    young_score: float,
    old_score: float,
    young_age: float,
    old_age: float,
    target_age: float,
) -> float:
    """Calibrate an age-score target by linear age extrapolation.

    Args:
        young_score: Center age score of the young reference cells.
        old_score: Center age score of the old reference cells.
        young_age: Chronological age of the young reference group.
        old_age: Chronological age of the old reference group.
        target_age: Chronological age represented by the desired target score.

    Returns:
        The calibrated age score at ``target_age``.
    """
    if old_age == young_age:
        return float(young_score)
    slope = (old_score - young_score) / (old_age - young_age)
    intercept = young_score - slope * young_age
    return float(slope * target_age + intercept)


def optimize_aging_trajectory(
    old_expression: Tensor,
    young_expression: Tensor,
    gene_ids: Tensor,
    reconstruction_model: nn.Module,
    score_model: nn.Module,
    young_age: float,
    old_age: float,
    target_age: float,
    *,
    identity_weight: float = 1.0,
    learning_rate: float = 1e-4,
    max_iterations: int = 3000,
    center: CenterMode = "median",
    calibration_space: CalibrationSpace = "recon",
    score_input_transform: ScoreInputTransform = "fc",
    fold_change_epsilon: float = 1e-8,
    reconstruction_batch_size: Optional[int] = 64,
    log_every: int = 0,
) -> Dict[str, Tensor]:
    """Optimize input expression through frozen reconstruction and score models.

    The reconstruction model and score model are frozen. Gradients pass through
    both models and update only a copy of ``old_expression``. The loss is the
    cell-wise mean squared error to scalar age and identity targets. Trajectory
    statistics are accumulated from the mean per-gene change across cells at
    every optimization step.

    Args:
        old_expression: Old-cell expression matrix with shape
            ``(n_old_cells, n_genes)``.
        young_expression: Young-cell expression matrix with shape
            ``(n_young_cells, n_genes)``.
        gene_ids: Vocabulary IDs for the expression genes, in the same order as
            the expression columns.
        reconstruction_model: Tabula model returning a ``reconstruction`` item
            from ``forward(..., head="reconstruction")``.
        score_model: Frozen model returning age and identity scores in columns
            zero and one, respectively.
        young_age: Chronological age of the young reference group.
        old_age: Chronological age of the old reference group.
        target_age: Desired chronological age used to calibrate the age target.
        identity_weight: Weight of the identity loss relative to the age loss.
        learning_rate: Adam learning rate for the expression matrix.
        max_iterations: Number of expression-optimization steps.
        center: Statistic used to summarize group scores for calibration and
            history reporting.
        calibration_space: Use raw expression or reconstructed expression to
            derive the two scalar targets.
        score_input_transform: Pass expression directly to the score model or
            first transform it to cohort-relative fold change.
        fold_change_epsilon: Minimum absolute fold-change denominator.
        reconstruction_batch_size: Batch size used for baseline reconstruction.
            Set to ``None`` to reconstruct each reference matrix in one call.
        log_every: Log progress every this many iterations. Zero disables
            iteration logging.

    Returns:
        A dictionary containing final expressions, calibrated targets, score and
        delta histories, and the eight accumulated trajectory statistics.
    """
    _validate_optimization_inputs(
        old_expression=old_expression,
        young_expression=young_expression,
        gene_ids=gene_ids,
        identity_weight=identity_weight,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        center=center,
        calibration_space=calibration_space,
        score_input_transform=score_input_transform,
        fold_change_epsilon=fold_change_epsilon,
        reconstruction_batch_size=reconstruction_batch_size,
        log_every=log_every,
    )

    device = old_expression.device
    old_expression = old_expression.detach().to(device=device, dtype=torch.float32)
    young_expression = young_expression.detach().to(
        device=device,
        dtype=torch.float32,
    )
    gene_ids = gene_ids.detach().to(device=device, dtype=torch.int64)

    reconstruction_model = reconstruction_model.to(device)
    score_model = score_model.to(device)
    reconstruction_model.eval()
    score_model.eval()
    reconstruction_model.requires_grad_(False)
    score_model.requires_grad_(False)

    with torch.no_grad():
        old_reconstruction = _reconstruct_batched(
            reconstruction_model,
            old_expression,
            gene_ids,
            reconstruction_batch_size,
        )
        young_reconstruction = _reconstruct_batched(
            reconstruction_model,
            young_expression,
            gene_ids,
            reconstruction_batch_size,
        )

    raw_reference_mean: Optional[Tensor] = None
    recon_reference_mean: Optional[Tensor] = None
    if score_input_transform == "fc":
        raw_reference_mean = torch.cat(
            [old_expression, young_expression],
            dim=0,
        ).mean(dim=0, keepdim=True)
        recon_reference_mean = torch.cat(
            [old_reconstruction, young_reconstruction],
            dim=0,
        ).mean(dim=0, keepdim=True)

    with torch.no_grad():
        old_raw_scores = _score(
            score_model,
            _transform_score_input(
                old_expression,
                score_input_transform,
                raw_reference_mean,
                fold_change_epsilon,
            ),
        )
        young_raw_scores = _score(
            score_model,
            _transform_score_input(
                young_expression,
                score_input_transform,
                raw_reference_mean,
                fold_change_epsilon,
            ),
        )
        old_recon_scores = _score(
            score_model,
            _transform_score_input(
                old_reconstruction,
                score_input_transform,
                recon_reference_mean,
                fold_change_epsilon,
            ),
        )
        young_recon_scores = _score(
            score_model,
            _transform_score_input(
                young_reconstruction,
                score_input_transform,
                recon_reference_mean,
                fold_change_epsilon,
            ),
        )

    old_raw_center = _score_center(old_raw_scores, center)
    young_raw_center = _score_center(young_raw_scores, center)
    old_recon_center = _score_center(old_recon_scores, center)
    young_recon_center = _score_center(young_recon_scores, center)

    if calibration_space == "raw":
        calibration_young = young_raw_center
        calibration_old = old_raw_center
    else:
        calibration_young = young_recon_center
        calibration_old = old_recon_center

    age_target = calibrate_age_score(
        young_score=float(calibration_young[0].item()),
        old_score=float(calibration_old[0].item()),
        young_age=young_age,
        old_age=old_age,
        target_age=target_age,
    )
    identity_target = float(calibration_young[1].item())

    n_genes = old_expression.shape[1]
    optimized_expression = nn.Parameter(old_expression.clone())
    optimizer = Adam([optimized_expression], lr=learning_rate)

    positive_path_input = torch.zeros(n_genes, device=device)
    negative_path_input = torch.zeros(n_genes, device=device)
    positive_path_recon = torch.zeros(n_genes, device=device)
    negative_path_recon = torch.zeros(n_genes, device=device)
    up_count_input = torch.zeros(n_genes, dtype=torch.int32, device=device)
    down_count_input = torch.zeros(n_genes, dtype=torch.int32, device=device)
    up_count_recon = torch.zeros(n_genes, dtype=torch.int32, device=device)
    down_count_recon = torch.zeros(n_genes, dtype=torch.int32, device=device)

    previous_reconstruction = old_reconstruction.detach().clone()
    loss_history = []
    raw_score_history = [old_raw_center]
    recon_score_history = [old_recon_center]
    input_delta_history = [torch.zeros(n_genes, device=device)]
    recon_delta_history = [torch.zeros(n_genes, device=device)]

    logger.info(
        "Starting aging trajectory optimization for %d iterations.",
        max_iterations,
    )

    for iteration in range(1, max_iterations + 1):
        optimizer.zero_grad()
        previous_expression = optimized_expression.detach().clone()

        reconstruction_for_loss = _reconstruct(
            reconstruction_model,
            optimized_expression,
            gene_ids,
        )
        predicted_scores = _score(
            score_model,
            _transform_score_input(
                reconstruction_for_loss,
                score_input_transform,
                recon_reference_mean,
                fold_change_epsilon,
            ),
        )
        age_loss = (predicted_scores[:, 0] - age_target).pow(2).mean()
        identity_loss = (
            predicted_scores[:, 1] - identity_target
        ).pow(2).mean()
        loss = age_loss + identity_weight * identity_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_reconstruction = _reconstruct(
                reconstruction_model,
                optimized_expression,
                gene_ids,
            )
            current_raw_scores = _score(
                score_model,
                _transform_score_input(
                    optimized_expression,
                    score_input_transform,
                    raw_reference_mean,
                    fold_change_epsilon,
                ),
            )
            current_recon_scores = _score(
                score_model,
                _transform_score_input(
                    current_reconstruction,
                    score_input_transform,
                    recon_reference_mean,
                    fold_change_epsilon,
                ),
            )

            input_step = (optimized_expression - previous_expression).mean(dim=0)
            recon_step = (
                current_reconstruction - previous_reconstruction
            ).mean(dim=0)

            positive_path_input += torch.clamp(input_step, min=0.0)
            negative_path_input += torch.clamp(-input_step, min=0.0)
            positive_path_recon += torch.clamp(recon_step, min=0.0)
            negative_path_recon += torch.clamp(-recon_step, min=0.0)
            up_count_input += (input_step > 0).to(torch.int32)
            down_count_input += (input_step < 0).to(torch.int32)
            up_count_recon += (recon_step > 0).to(torch.int32)
            down_count_recon += (recon_step < 0).to(torch.int32)

            input_delta_history.append(
                (optimized_expression - old_expression).mean(dim=0)
            )
            recon_delta_history.append(
                (current_reconstruction - old_reconstruction).mean(dim=0)
            )
            raw_score_history.append(_score_center(current_raw_scores, center))
            recon_score_history.append(_score_center(current_recon_scores, center))
            previous_reconstruction = current_reconstruction.detach().clone()
            loss_history.append(loss.detach())

        if log_every > 0 and iteration % log_every == 0:
            logger.info(
                "Aging optimization iteration %d: loss=%.6f.",
                iteration,
                float(loss.item()),
            )

    final_reconstruction = previous_reconstruction
    logger.info("Aging trajectory optimization finished.")

    raw_score_history_tensor = torch.stack(raw_score_history)
    recon_score_history_tensor = torch.stack(recon_score_history)

    return {
        "optimized_expression": optimized_expression.detach().clone(),
        "reconstructed_expression": final_reconstruction.detach().clone(),
        "age_target": torch.tensor(age_target, device=device),
        "identity_target": torch.tensor(identity_target, device=device),
        "steps": torch.arange(max_iterations + 1, device=device),
        "loss_history": torch.stack(loss_history),
        "center_age_raw": raw_score_history_tensor[:, 0],
        "center_identity_raw": raw_score_history_tensor[:, 1],
        "center_age_recon": recon_score_history_tensor[:, 0],
        "center_identity_recon": recon_score_history_tensor[:, 1],
        "delta_X": torch.stack(input_delta_history),
        "delta_recon": torch.stack(recon_delta_history),
        "pos_path_X": positive_path_input,
        "neg_path_X": negative_path_input,
        "pos_path_recon": positive_path_recon,
        "neg_path_recon": negative_path_recon,
        "n_up_X": up_count_input,
        "n_down_X": down_count_input,
        "n_up_recon": up_count_recon,
        "n_down_recon": down_count_recon,
    }


def _validate_optimization_inputs(
    old_expression: Tensor,
    young_expression: Tensor,
    gene_ids: Tensor,
    identity_weight: float,
    learning_rate: float,
    max_iterations: int,
    center: str,
    calibration_space: str,
    score_input_transform: str,
    fold_change_epsilon: float,
    reconstruction_batch_size: Optional[int],
    log_every: int,
) -> None:
    if old_expression.ndim != 2 or young_expression.ndim != 2:
        raise ValueError("old_expression and young_expression must be two-dimensional.")
    if old_expression.shape[1] != young_expression.shape[1]:
        raise ValueError("Old and young expression must use the same genes.")
    if old_expression.shape[0] == 0 or young_expression.shape[0] == 0:
        raise ValueError("Old and young expression must contain at least one cell.")
    if gene_ids.ndim != 1 or gene_ids.numel() != old_expression.shape[1]:
        raise ValueError("gene_ids must contain one ID for each expression gene.")
    if identity_weight < 0:
        raise ValueError("identity_weight must be nonnegative.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if center not in ("mean", "median"):
        raise ValueError("center must be 'mean' or 'median'.")
    if calibration_space not in ("raw", "recon"):
        raise ValueError("calibration_space must be 'raw' or 'recon'.")
    if score_input_transform not in ("none", "fc"):
        raise ValueError("score_input_transform must be 'none' or 'fc'.")
    if fold_change_epsilon <= 0:
        raise ValueError("fold_change_epsilon must be positive.")
    if reconstruction_batch_size is not None and reconstruction_batch_size <= 0:
        raise ValueError("reconstruction_batch_size must be positive or None.")
    if log_every < 0:
        raise ValueError("log_every must be nonnegative.")


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def _reconstruct(
    model: nn.Module,
    expression: Tensor,
    gene_ids: Tensor,
) -> Tensor:
    model_dtype = _module_dtype(model, expression.dtype)
    output = model(
        genes=gene_ids,
        values=expression.to(dtype=model_dtype).contiguous(),
        head="reconstruction",
    )
    if not isinstance(output, dict) or "reconstruction" not in output:
        raise ValueError(
            "reconstruction_model must return a dictionary containing "
            "'reconstruction'."
        )
    reconstruction = output["reconstruction"]
    if reconstruction.shape != expression.shape:
        raise ValueError(
            "Reconstructed expression must have the same shape as the input."
        )
    return reconstruction.float()


def _reconstruct_batched(
    model: nn.Module,
    expression: Tensor,
    gene_ids: Tensor,
    batch_size: Optional[int],
) -> Tensor:
    if batch_size is None or expression.shape[0] <= batch_size:
        return _reconstruct(model, expression, gene_ids)
    reconstructions = []
    for start in range(0, expression.shape[0], batch_size):
        reconstructions.append(
            _reconstruct(model, expression[start : start + batch_size], gene_ids)
        )
    return torch.cat(reconstructions, dim=0)


def _transform_score_input(
    expression: Tensor,
    transform: ScoreInputTransform,
    reference_mean: Optional[Tensor],
    epsilon: float,
) -> Tensor:
    if transform == "none":
        return expression
    if reference_mean is None:
        raise ValueError("reference_mean is required for fold-change input.")
    return compute_fold_change(expression, reference_mean, epsilon)


def _score(model: nn.Module, expression: Tensor) -> Tensor:
    model_dtype = _module_dtype(model, expression.dtype)
    scores = model(expression.to(dtype=model_dtype))
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(
            "score_model must return a tensor with shape (n_cells, 2)."
        )
    return scores.float()


def _score_center(scores: Tensor, mode: CenterMode) -> Tensor:
    if mode == "mean":
        return scores.mean(dim=0)
    return scores.median(dim=0).values
