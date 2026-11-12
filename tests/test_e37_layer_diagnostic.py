from __future__ import annotations

import math

import pytest
import torch

from robust_metric_lab.e37_batched import compute_e37_metrics_batched
from robust_metric_lab.e37_metrics import E37_METRIC_NAMES
from robust_metric_lab.e37_layer_diagnostic import diagnose_e37, leaf_call_trace
from robust_metric_lab.metrics import ActivationRecorder


class SharedConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.shared(x))


def _no_tensors_or_nonfinite(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for item in value.values():
            _no_tensors_or_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_tensors_or_nonfinite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_reused_leaf_retains_call_number_and_recorder_order() -> None:
    model = SharedConvModel().eval()
    x = torch.ones(1, 3, 4, 4)
    features, trace = leaf_call_trace(model, x)
    assert [(row["module_path"], row["invocation"]) for row in trace] == [
        ("shared", 1), ("shared", 2)
    ]
    with torch.no_grad(), ActivationRecorder(model, activation_mode="all_leaf") as recorder:
        model(x)
    assert len(features) == len(recorder.outputs)
    for observed, expected in zip(features, recorder.outputs):
        torch.testing.assert_close(observed, expected)


def test_components_are_numeric_and_nine_scores_match_pinned_e37() -> None:
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(4, 6, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(6, 3),
    ).eval()
    clean = torch.linspace(0, 1, 5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1])
    actual = diagnose_e37(model, clean, labels, epsilon=8 / 255)
    expected = compute_e37_metrics_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=1
    )
    assert actual["sample_count"] == 5
    assert actual["feature_count"] == 5
    assert len(actual["anchor_metadata"]) == 4
    assert len(actual["sample_components"]) == 5
    for sample in actual["sample_components"]:
        assert len(sample["anchors"]) == 4
        assert len(sample["transitions"]) == 3
        assert set(sample["anchors"][0]) == {
            "clean_rms", "adv_rms", "clean_centered_l2", "adv_centered_l2",
            "channel", "spatial", "spectrum_fisher", "spectrum_triangular", "spectrum_ks",
        }
        assert set(sample["transitions"][0]) == {"flow"}
    _no_tensors_or_nonfinite(actual)
    assert set(actual["metric_values"]) == set(E37_METRIC_NAMES)
    for name in E37_METRIC_NAMES:
        assert actual["metric_values"][name] == pytest.approx(expected[name], rel=3e-5, abs=3e-8)


def test_bad_batch_and_too_few_anchors_fail() -> None:
    class OneAnchorModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x).mean(dim=(2, 3))

    model = OneAnchorModel().eval()
    with pytest.raises(ValueError, match="positive length"):
        diagnose_e37(model, torch.empty(0, 3, 4, 4), torch.empty(0, dtype=torch.long), epsilon=8 / 255)
    with pytest.raises(ValueError, match="two anchors"):
        diagnose_e37(model, torch.ones(1, 3, 4, 4), torch.zeros(1, dtype=torch.long), epsilon=8 / 255)
