from __future__ import annotations

import torch

from unified_eval.fisher_grad import (
    conservative_replacement_indices,
    nonnegative_natural_direction,
    normalize_direction,
    scale_artifact,
    shrunk_fisher,
)


def test_conservative_replacement_keeps_budget_and_uses_tail() -> None:
    scores = torch.tensor([10.0, 1.0, 9.0, 0.0, 8.0, 7.0])
    selected = conservative_replacement_indices(scores, base_k=4, replace_count=2)
    assert set(selected.tolist()) == {0, 2, 4, 5}


def test_shrunk_fisher_preserves_diagonal_and_adds_relative_damping() -> None:
    fisher = torch.tensor([[4.0, 2.0], [2.0, 1.0]])
    matrix, damping = shrunk_fisher(fisher, shrinkage=0.5, damping_ratio=0.1)
    assert damping == 0.1
    assert torch.allclose(matrix, torch.tensor([[4.1, 1.0], [1.0, 1.1]]))


def test_nonnegative_direction_respects_positive_only_constraint() -> None:
    matrix = torch.tensor([[2.0, 1.5], [1.5, 2.0]])
    gradient = torch.tensor([1.0, 0.1])
    direction, diagnostics = nonnegative_natural_direction(matrix, gradient)
    assert diagnostics["success"]
    assert torch.all(direction >= 0)
    assert direction[0] > 0
    assert direction[1] == 0


def test_normalized_direction_hits_quadratic_budget() -> None:
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    direction = torch.tensor([1.0, 2.0])
    delta = normalize_direction(direction, matrix, epsilon=0.25)
    assert torch.allclose(0.5 * delta @ matrix @ delta, torch.tensor(0.25))


def test_scale_artifact_records_multipliers() -> None:
    ranking = [
        {"rank": 1, "source_rank": 2, "layer": 3, "neuron": 4, "mean_g": 0.5}
    ]
    artifact = scale_artifact("shared", ranking, torch.tensor([0.75]), {})
    assert artifact["direction"] == "positive-only"
    assert artifact["scope"] == "last"
    assert artifact["rows"][0]["delta"] == 0.75
    assert artifact["rows"][0]["multiplier"] == 1.75
