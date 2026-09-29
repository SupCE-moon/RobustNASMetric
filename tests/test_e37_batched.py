from __future__ import annotations

import pytest
import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.e37_batched import compute_e37_metrics_batched
from robust_metric_lab.e37_metrics import E37_METRIC_NAMES, compute_e37_per_sample
from robust_metric_lab.metrics import ActivationRecorder, _anchor_indices


class TinyE37Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 6, 3, padding=1, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(6, 3, bias=False)
        torch.manual_seed(3711)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.2, 0.2)

    def forward(self, inputs):
        hidden = self.relu1(self.conv1(inputs))
        hidden = self.relu2(self.conv2(hidden))
        return self.fc(self.pool(hidden).flatten(1))


def _direct_full_batch(model, clean, labels, *, epsilon: float, max_layers: int):
    adversarial = make_adversarial(model, clean, labels, "fgsm", epsilon, pgd_steps=1)
    with torch.no_grad():
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(clean)
        clean_features = recorder.outputs
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(adversarial)
        adversarial_features = recorder.outputs
    indices = _anchor_indices(min(len(clean_features), len(adversarial_features)), max_layers=max_layers)
    return {
        name: float(values.mean().item())
        for name, values in compute_e37_per_sample(clean_features, adversarial_features, indices).items()
    }


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_e37_chunking_matches_direct_full_batch(batch_size: int) -> None:
    model = TinyE37Net().eval()
    clean = torch.linspace(-0.8, 0.8, 5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1])
    expected = _direct_full_batch(model, clean, labels, epsilon=8 / 255, max_layers=4)
    actual = compute_e37_metrics_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=batch_size, max_layers=4
    )
    assert set(actual) == set(E37_METRIC_NAMES)
    for name in E37_METRIC_NAMES:
        assert actual[name] == pytest.approx(expected[name], rel=3e-5, abs=3e-8)


@pytest.mark.parametrize(
    ("clean", "labels", "batch_size", "match"),
    [
        (torch.randn(2, 3, 6, 6), torch.tensor([0, 1]), 0, "batch_size"),
        (torch.empty(0, 3, 6, 6), torch.empty(0, dtype=torch.long), 1, "positive length"),
        (torch.randn(2, 3, 6, 6), torch.tensor([0]), 1, "same positive length"),
    ],
)
def test_e37_batched_rejects_invalid_sample_contract(clean, labels, batch_size, match) -> None:
    with pytest.raises(ValueError, match=match):
        compute_e37_metrics_batched(
            TinyE37Net(), clean, labels, epsilon=8 / 255, batch_size=batch_size
        )


def test_e37_batched_rejects_fewer_than_two_anchors() -> None:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 2),
    )
    with pytest.raises(ValueError, match="two anchors"):
        compute_e37_metrics_batched(
            model,
            torch.randn(2, 3, 4, 4),
            torch.tensor([0, 1]),
            epsilon=8 / 255,
            batch_size=1,
            max_layers=1,
        )
