import copy

import torch
import torch.nn as nn

from tabula.finetune.model.aging import (
    AgingScoreRegressor,
    calibrate_age_score,
    compute_fold_change,
    optimize_aging_trajectory,
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


class ToyScoreModel(nn.Module):
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


def test_regressor_keeps_legacy_sequential_state_dict_keys():
    model = AgingScoreRegressor(in_features=4, hidden_features=3)
    assert set(model.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    }
    assert model(torch.ones(2, 4)).shape == (2, 2)


def test_compute_fold_change_preserves_reference_sign_without_clamping_input():
    expression = torch.tensor([[2.0, -3.0, -4.0]])
    reference_mean = torch.tensor([0.0, -1e-12, 2.0])

    result = compute_fold_change(expression, reference_mean, epsilon=1e-2)

    expected = torch.tensor([[200.0, 300.0, -2.0]])
    torch.testing.assert_close(result, expected)


def test_calibrate_age_score_uses_linear_extrapolation():
    result = calibrate_age_score(
        young_score=1.0,
        old_score=3.0,
        young_age=20.0,
        old_age=40.0,
        target_age=10.0,
    )
    assert result == 0.0


def test_optimization_freezes_models_and_matches_reference_path_accumulation():
    old_expression = torch.tensor([[2.0, 1.0], [1.5, 0.5]])
    young_expression = torch.tensor([[0.5, 1.5], [1.0, 2.0]])
    gene_ids = torch.tensor([10, 11])

    reconstruction_model = ToyReconstructionModel()
    score_model = ToyScoreModel()
    reconstruction_before = copy.deepcopy(reconstruction_model.state_dict())
    score_before = copy.deepcopy(score_model.state_dict())

    result = optimize_aging_trajectory(
        old_expression=old_expression,
        young_expression=young_expression,
        gene_ids=gene_ids,
        reconstruction_model=reconstruction_model,
        score_model=score_model,
        young_age=20.0,
        old_age=40.0,
        target_age=10.0,
        identity_weight=0.7,
        learning_rate=0.05,
        max_iterations=4,
        center="median",
        calibration_space="recon",
        score_input_transform="fc",
        reconstruction_batch_size=1,
    )

    for name, value in reconstruction_model.state_dict().items():
        torch.testing.assert_close(value, reconstruction_before[name])
    for name, value in score_model.state_dict().items():
        torch.testing.assert_close(value, score_before[name])
    assert all(
        not parameter.requires_grad
        for parameter in reconstruction_model.parameters()
    )
    assert all(not parameter.requires_grad for parameter in score_model.parameters())

    reference = _run_reference_optimization(
        old_expression=old_expression,
        young_expression=young_expression,
        reconstruction_model=ToyReconstructionModel(),
        score_model=ToyScoreModel(),
        young_age=20.0,
        old_age=40.0,
        target_age=10.0,
        identity_weight=0.7,
        learning_rate=0.05,
        max_iterations=4,
    )

    keys = [
        "optimized_expression",
        "reconstructed_expression",
        "pos_path_X",
        "neg_path_X",
        "pos_path_recon",
        "neg_path_recon",
        "n_up_X",
        "n_down_X",
        "n_up_recon",
        "n_down_recon",
        "delta_X",
        "delta_recon",
    ]
    for key in keys:
        torch.testing.assert_close(result[key], reference[key])

    assert result["loss_history"].shape == (4,)
    assert result["steps"].tolist() == [0, 1, 2, 3, 4]


def _run_reference_optimization(
    old_expression,
    young_expression,
    reconstruction_model,
    score_model,
    young_age,
    old_age,
    target_age,
    identity_weight,
    learning_rate,
    max_iterations,
):
    reconstruction_model.eval().requires_grad_(False)
    score_model.eval().requires_grad_(False)

    with torch.no_grad():
        old_recon = reconstruction_model(
            genes=torch.tensor([10, 11]),
            values=old_expression,
            head="reconstruction",
        )["reconstruction"]
        young_recon = reconstruction_model(
            genes=torch.tensor([10, 11]),
            values=young_expression,
            head="reconstruction",
        )["reconstruction"]
        recon_reference_mean = torch.cat([old_recon, young_recon], dim=0).mean(
            dim=0, keepdim=True
        )
        old_scores = score_model(
            compute_fold_change(old_recon, recon_reference_mean)
        )
        young_scores = score_model(
            compute_fold_change(young_recon, recon_reference_mean)
        )
        age_target = calibrate_age_score(
            young_score=float(young_scores[:, 0].median()),
            old_score=float(old_scores[:, 0].median()),
            young_age=young_age,
            old_age=old_age,
            target_age=target_age,
        )
        identity_target = float(young_scores[:, 1].median())

    expression = nn.Parameter(old_expression.clone())
    optimizer = torch.optim.Adam([expression], lr=learning_rate)
    previous_recon = old_recon.clone()

    pos_x = torch.zeros(2)
    neg_x = torch.zeros(2)
    pos_r = torch.zeros(2)
    neg_r = torch.zeros(2)
    up_x = torch.zeros(2, dtype=torch.int32)
    down_x = torch.zeros(2, dtype=torch.int32)
    up_r = torch.zeros(2, dtype=torch.int32)
    down_r = torch.zeros(2, dtype=torch.int32)
    delta_x = [torch.zeros(2)]
    delta_r = [torch.zeros(2)]

    for _ in range(max_iterations):
        optimizer.zero_grad()
        previous_expression = expression.detach().clone()
        reconstruction = reconstruction_model(
            genes=torch.tensor([10, 11]),
            values=expression,
            head="reconstruction",
        )["reconstruction"]
        scores = score_model(
            compute_fold_change(reconstruction, recon_reference_mean)
        )
        loss = (scores[:, 0] - age_target).pow(2).mean()
        loss += identity_weight * (scores[:, 1] - identity_target).pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_recon = reconstruction_model(
                genes=torch.tensor([10, 11]),
                values=expression,
                head="reconstruction",
            )["reconstruction"]
            step_x = (expression - previous_expression).mean(dim=0)
            step_r = (current_recon - previous_recon).mean(dim=0)
            pos_x += torch.clamp(step_x, min=0.0)
            neg_x += torch.clamp(-step_x, min=0.0)
            pos_r += torch.clamp(step_r, min=0.0)
            neg_r += torch.clamp(-step_r, min=0.0)
            up_x += (step_x > 0).to(torch.int32)
            down_x += (step_x < 0).to(torch.int32)
            up_r += (step_r > 0).to(torch.int32)
            down_r += (step_r < 0).to(torch.int32)
            delta_x.append((expression - old_expression).mean(dim=0))
            delta_r.append((current_recon - old_recon).mean(dim=0))
            previous_recon = current_recon.clone()

    return {
        "optimized_expression": expression.detach(),
        "reconstructed_expression": previous_recon,
        "pos_path_X": pos_x,
        "neg_path_X": neg_x,
        "pos_path_recon": pos_r,
        "neg_path_recon": neg_r,
        "n_up_X": up_x,
        "n_down_X": down_x,
        "n_up_recon": up_r,
        "n_down_recon": down_r,
        "delta_X": torch.stack(delta_x),
        "delta_recon": torch.stack(delta_r),
    }
