from __future__ import annotations

import pytest
import torch

from robust_metric_lab.e37_metrics import E37_METRIC_NAMES, compute_e37_per_sample


EXPECTED_NAMES = (
    "feature_measure_flow_corr_worst_delta",
    "feature_measure_flow_corr_rms_delta",
    "feature_measure_flow_corr_tophalf_delta",
    "feature_measure_spectrum_fisher_rao_delta",
    "feature_measure_spectrum_triangular_delta",
    "feature_measure_spectrum_ks_delta",
    "feature_measure_multiview_geometric_delta",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
)


def _features() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3701)
    return [
        torch.randn(4, 3, 8, 8, generator=generator),
        torch.randn(4, 5, 4, 4, generator=generator),
        torch.randn(4, 7, 2, 2, generator=generator),
    ]


def test_e37_metric_names_are_frozen() -> None:
    assert E37_METRIC_NAMES == EXPECTED_NAMES


def test_identical_features_are_finite_and_near_zero() -> None:
    features = _features()
    result = compute_e37_per_sample(features, [value.clone() for value in features], (0, 1, 2))
    assert set(result) == set(EXPECTED_NAMES)
    for values in result.values():
        assert values.shape == (4,)
        assert torch.isfinite(values).all()
        assert values.abs().max().item() < 1e-5


def test_e37_metrics_are_symmetric_in_clean_and_perturbed_inputs() -> None:
    clean = _features()
    adversarial = [value + 0.07 * torch.tanh(value) for value in clean]
    forward = compute_e37_per_sample(clean, adversarial, (0, 1, 2))
    backward = compute_e37_per_sample(adversarial, clean, (0, 1, 2))
    for name in EXPECTED_NAMES:
        assert torch.allclose(forward[name], backward[name], rtol=2e-5, atol=2e-7)


def test_e37_metrics_ignore_positive_layerwise_scale() -> None:
    clean = _features()
    adversarial = [value + 0.05 * torch.sin(value) for value in clean]
    baseline = compute_e37_per_sample(clean, adversarial, (0, 1, 2))
    clean_scales = (0.25, 3.0, 1.5)
    adversarial_scales = (2.0, 0.4, 5.0)
    scaled = compute_e37_per_sample(
        [value * scale for value, scale in zip(clean, clean_scales)],
        [value * scale for value, scale in zip(adversarial, adversarial_scales)],
        (0, 1, 2),
    )
    for name in EXPECTED_NAMES:
        assert torch.allclose(baseline[name], scaled[name], rtol=5e-5, atol=5e-7)


def test_e37_requires_two_valid_anchors_and_matching_batches() -> None:
    features = _features()
    with pytest.raises(ValueError, match="two anchors"):
        compute_e37_per_sample(features, features, (0,))
    with pytest.raises(ValueError, match="batch"):
        compute_e37_per_sample(features, [value[:3] for value in features], (0, 1))
