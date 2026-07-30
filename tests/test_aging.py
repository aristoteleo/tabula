import copy

import torch
import torch.nn as nn

from tabula.finetune.model.aging import (
    DiscriminatorMLP,
    compute_cohort_mean_fold_change,
    compute_lower_bound_age_score,
    predict_rejuvenation_trajectory,
)


class ToyReconstructionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.5, -0.5]))
        self.bias = nn.Parameter(torch.tensor([0.2, 0.1]))

    def forward(self, genes, values, head=None):
        del genes
        assert head == "reconstruction"
        return {"reconstruction": values * self.scale + self.bias}


class ToyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.4],
                        [-0.3, 0.8],
                    ]
                )
            )

    def forward(self, expression):
        return self.linear(expression)


def test_discriminator_mlp_keeps_trained_checkpoint_state_dict_keys():
    model = DiscriminatorMLP(in_features=4, hidden_features=3)
    assert set(model.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    }
    assert model(torch.ones(2, 4)).shape == (2, 2)


def test_cohort_mean_fold_change_does_not_clamp_input_expression():
    expression = torch.tensor([[2.0, -3.0, -4.0]])
    cohort_mean = torch.tensor([0.0, -1e-12, 2.0])

    result = compute_cohort_mean_fold_change(
        expression,
        cohort_mean,
        epsilon=1e-2,
    )

    expected = torch.tensor([[200.0, 300.0, -2.0]])
    torch.testing.assert_close(result, expected)


def test_lower_bound_age_score_uses_linear_extrapolation():
    result = compute_lower_bound_age_score(
        young_age_score=1.0,
        aged_age_score=3.0,
        young_age=20.0,
        aged_age=40.0,
        lower_bound_age=10.0,
    )
    assert result == 0.0


def test_rejuvenation_trajectory_freezes_models_and_matches_reference_statistics():
    aged_expression = torch.tensor([[2.0, 1.0], [1.5, 0.5]])
    young_expression = torch.tensor([[0.5, 1.5], [1.0, 2.0]])
    gene_ids = torch.tensor([10, 11])

    reconstruction_model = ToyReconstructionModel()
    discriminator = ToyDiscriminator()
    reconstruction_before = copy.deepcopy(reconstruction_model.state_dict())
    discriminator_before = copy.deepcopy(discriminator.state_dict())

    result = predict_rejuvenation_trajectory(
        aged_expression=aged_expression,
        young_expression=young_expression,
        gene_ids=gene_ids,
        reconstruction_model=reconstruction_model,
        discriminator=discriminator,
        young_age=20.0,
        aged_age=40.0,
        lower_bound_age=10.0,
        identity_preservation_weight=0.7,
        learning_rate=0.05,
        n_steps=4,
        population_center="median",
        target_calibration_space="output",
        discriminator_input_transform="cohort_mean_fold_change",
        reconstruction_batch_size=1,
    )

    for name, value in reconstruction_model.state_dict().items():
        torch.testing.assert_close(value, reconstruction_before[name])
    for name, value in discriminator.state_dict().items():
        torch.testing.assert_close(value, discriminator_before[name])
    assert all(
        not parameter.requires_grad
        for parameter in reconstruction_model.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in discriminator.parameters()
    )

    reference = _run_reference_rejuvenation_trajectory(
        aged_expression=aged_expression,
        young_expression=young_expression,
        reconstruction_model=ToyReconstructionModel(),
        discriminator=ToyDiscriminator(),
        young_age=20.0,
        aged_age=40.0,
        lower_bound_age=10.0,
        identity_preservation_weight=0.7,
        learning_rate=0.05,
        n_steps=4,
    )

    keys = [
        "optimized_input_expression",
        "optimized_output_expression",
        "pos_path_input",
        "neg_path_input",
        "pos_path_output",
        "neg_path_output",
        "n_up_input",
        "n_down_input",
        "n_up_output",
        "n_down_output",
        "net_change_input",
        "net_change_output",
        "net_change_input_history",
        "net_change_output_history",
    ]
    for key in keys:
        torch.testing.assert_close(result[key], reference[key])

    assert result["loss_history"].shape == (4,)
    assert result["steps"].tolist() == [0, 1, 2, 3, 4]


def _run_reference_rejuvenation_trajectory(
    aged_expression,
    young_expression,
    reconstruction_model,
    discriminator,
    young_age,
    aged_age,
    lower_bound_age,
    identity_preservation_weight,
    learning_rate,
    n_steps,
):
    reconstruction_model.eval().requires_grad_(False)
    discriminator.eval().requires_grad_(False)
    gene_ids = torch.tensor([10, 11])

    with torch.no_grad():
        aged_output = reconstruction_model(
            genes=gene_ids,
            values=aged_expression,
            head="reconstruction",
        )["reconstruction"]
        young_output = reconstruction_model(
            genes=gene_ids,
            values=young_expression,
            head="reconstruction",
        )["reconstruction"]
        output_cohort_mean = torch.cat(
            [aged_output, young_output],
            dim=0,
        ).mean(dim=0, keepdim=True)
        aged_scores = discriminator(
            compute_cohort_mean_fold_change(
                aged_output,
                output_cohort_mean,
            )
        )
        young_scores = discriminator(
            compute_cohort_mean_fold_change(
                young_output,
                output_cohort_mean,
            )
        )
        lower_bound_age_score = compute_lower_bound_age_score(
            young_age_score=float(young_scores[:, 0].median()),
            aged_age_score=float(aged_scores[:, 0].median()),
            young_age=young_age,
            aged_age=aged_age,
            lower_bound_age=lower_bound_age,
        )
        young_identity_score = float(young_scores[:, 1].median())

    input_expression = nn.Parameter(aged_expression.clone())
    optimizer = torch.optim.Adam([input_expression], lr=learning_rate)
    previous_output = aged_output.clone()

    pos_path_input = torch.zeros(2)
    neg_path_input = torch.zeros(2)
    pos_path_output = torch.zeros(2)
    neg_path_output = torch.zeros(2)
    n_up_input = torch.zeros(2, dtype=torch.int32)
    n_down_input = torch.zeros(2, dtype=torch.int32)
    n_up_output = torch.zeros(2, dtype=torch.int32)
    n_down_output = torch.zeros(2, dtype=torch.int32)
    net_change_input_history = [torch.zeros(2)]
    net_change_output_history = [torch.zeros(2)]

    for _ in range(n_steps):
        optimizer.zero_grad()
        previous_input = input_expression.detach().clone()
        output_expression = reconstruction_model(
            genes=gene_ids,
            values=input_expression,
            head="reconstruction",
        )["reconstruction"]
        scores = discriminator(
            compute_cohort_mean_fold_change(
                output_expression,
                output_cohort_mean,
            )
        )
        loss = (scores[:, 0] - lower_bound_age_score).pow(2).mean()
        loss += identity_preservation_weight * (
            scores[:, 1] - young_identity_score
        ).pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_output = reconstruction_model(
                genes=gene_ids,
                values=input_expression,
                head="reconstruction",
            )["reconstruction"]
            stepwise_change_input = (
                input_expression - previous_input
            ).mean(dim=0)
            stepwise_change_output = (
                current_output - previous_output
            ).mean(dim=0)
            pos_path_input += torch.clamp(
                stepwise_change_input,
                min=0.0,
            )
            neg_path_input += torch.clamp(
                -stepwise_change_input,
                min=0.0,
            )
            pos_path_output += torch.clamp(
                stepwise_change_output,
                min=0.0,
            )
            neg_path_output += torch.clamp(
                -stepwise_change_output,
                min=0.0,
            )
            n_up_input += (stepwise_change_input > 0).to(torch.int32)
            n_down_input += (stepwise_change_input < 0).to(torch.int32)
            n_up_output += (stepwise_change_output > 0).to(torch.int32)
            n_down_output += (stepwise_change_output < 0).to(torch.int32)
            net_change_input_history.append(
                (input_expression - aged_expression).mean(dim=0)
            )
            net_change_output_history.append(
                (current_output - aged_output).mean(dim=0)
            )
            previous_output = current_output.clone()

    net_change_input_history = torch.stack(net_change_input_history)
    net_change_output_history = torch.stack(net_change_output_history)
    return {
        "optimized_input_expression": input_expression.detach(),
        "optimized_output_expression": previous_output,
        "pos_path_input": pos_path_input,
        "neg_path_input": neg_path_input,
        "pos_path_output": pos_path_output,
        "neg_path_output": neg_path_output,
        "n_up_input": n_up_input,
        "n_down_input": n_down_input,
        "n_up_output": n_up_output,
        "n_down_output": n_down_output,
        "net_change_input": net_change_input_history[-1],
        "net_change_output": net_change_output_history[-1],
        "net_change_input_history": net_change_input_history,
        "net_change_output_history": net_change_output_history,
    }
