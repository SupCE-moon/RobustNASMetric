from __future__ import annotations

import pytest
import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.importance_batched import compute_importance_metric_batched
from robust_metric_lab.metrics import (
    ActivationRecorder,
    _anchor_indices,
    _channel_importance,
    _perturbation_capacity_importance,
)


class TinyImportanceNet(torch.nn.Module):
    def __init__(self, *, zero_output: bool = False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(5, 3, bias=False)
        torch.manual_seed(17)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.2, 0.2)
        if zero_output:
            torch.nn.init.zeros_(self.fc.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu1(self.conv1(inputs))
        hidden = self.relu2(self.conv2(hidden))
        return self.fc(self.pool(hidden).flatten(1))


def _full_batch_definition(model, clean, labels, *, epsilon: float, max_layers: int) -> float:
    importance = _channel_importance(model, clean)
    adversarial = make_adversarial(model, clean, labels, "fgsm", epsilon, pgd_steps=1)
    with torch.no_grad():
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(clean)
        clean_features = recorder.outputs
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(adversarial)
        adversarial_features = recorder.outputs
    indices = _anchor_indices(min(len(clean_features), len(adversarial_features)), max_layers=max_layers)
    values = [
        _perturbation_capacity_importance(
            clean_features[index], adversarial_features[index], importance[index]
        )
        for index in indices
    ]
    return torch.stack(values).mean().item()


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_importance_chunking_matches_full_batch_definition(batch_size: int) -> None:
    model = TinyImportanceNet().eval()
    clean = torch.linspace(-0.9, 0.9, 5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1])
    expected = _full_batch_definition(model, clean, labels, epsilon=8 / 255, max_layers=4)
    actual = compute_importance_metric_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=batch_size, max_layers=4
    )
    assert actual == pytest.approx(expected, rel=2e-5, abs=2e-7)


def test_zero_gradient_importance_is_finite_zero() -> None:
    model = TinyImportanceNet(zero_output=True).eval()
    clean = torch.randn(4, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 0])
    value = compute_importance_metric_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=1, max_layers=4
    )
    assert value == 0.0
    assert torch.isfinite(torch.tensor(value))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_importance_rejects_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        compute_importance_metric_batched(
            TinyImportanceNet(),
            torch.randn(2, 3, 6, 6),
            torch.tensor([0, 1]),
            epsilon=8 / 255,
            batch_size=batch_size,
        )
