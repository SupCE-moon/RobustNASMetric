import math

import torch

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.metrics import (
    _e36_cross_layer_stats,
    _e36_within_layer_stats,
    compute_feature_metrics,
)


E36_METRICS = (
    "feature_measure_channel_gram_cosine_delta",
    "feature_measure_channel_gram_centered_rel_delta",
    "feature_measure_spatial_gram_cosine_delta",
    "feature_measure_spectrum_js_delta",
    "feature_measure_spectrum_emd_delta",
    "feature_measure_spectrum_hellinger_delta",
    "feature_measure_flow_corr_angular_delta",
    "feature_measure_flow_corr_symmetric_rel_delta",
)


def test_e36_within_layer_metrics_are_zero_for_identical_features():
    torch.manual_seed(3601)
    clean = torch.randn(3, 7, 4, 4)

    values = _e36_within_layer_stats(clean, clean.clone())

    assert len(values) == 6
    for value in values:
        assert torch.isfinite(value)
        assert value.item() == 0.0


def test_e36_within_layer_metrics_ignore_global_feature_scale():
    torch.manual_seed(3602)
    clean = torch.randn(2, 6, 4, 4)

    values = _e36_within_layer_stats(clean, 3.0 * clean)

    for value in values:
        assert torch.isfinite(value)
        assert value.item() < 1e-6


def test_e36_within_layer_metrics_detect_structural_perturbation():
    clean = torch.zeros(2, 4, 4, 4)
    clean[:, 0, :2, :] = 1.0
    clean[:, 1, 2:, :] = 1.0
    clean[:, 2, :, :2] = 1.0
    clean[:, 3, :, 2:] = 1.0
    adv = clean + 0.35 * clean.roll(shifts=1, dims=1)
    adv = adv + 0.2 * clean.roll(shifts=1, dims=2)

    values = _e36_within_layer_stats(clean, adv)

    assert all(torch.isfinite(value) for value in values)
    assert sum(value.item() > 1e-5 for value in values) >= 4


def test_e36_cross_layer_metrics_are_zero_for_identical_flows():
    torch.manual_seed(3603)
    left = torch.randn(2, 5, 4, 4)
    right = torch.randn(2, 8, 2, 2)

    angular, symmetric_rel = _e36_cross_layer_stats(left, right, left, right)

    assert angular.item() == 0.0
    assert symmetric_rel.item() == 0.0


def test_e36_metrics_are_registered_as_lower_is_better():
    for name in E36_METRICS:
        assert METRIC_BY_NAME[name].lower_is_better


def test_compute_feature_metrics_exposes_all_e36_metrics():
    torch.manual_seed(3604)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 5, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(5, 3),
    )
    clean = torch.randn(2, 3, 4, 4)
    adv = clean + 0.03 * torch.sign(torch.randn_like(clean))
    labels = torch.tensor([0, 2])

    result = compute_feature_metrics(model, clean, adv, labels, max_layers=3)

    for name in E36_METRICS:
        assert math.isfinite(getattr(result, name))
