from dataclasses import asdict

import pytest
import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.metrics import compute_feature_metrics


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


def test_batched_e36_matches_frozen_full_batch_definition():
    try:
        from robust_metric_lab.e36_batched import compute_e36_metrics_batched
    except ImportError:
        pytest.fail("memory-bounded E36 evaluator is not implemented")

    torch.manual_seed(3605)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 5, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(5, 3),
    ).eval()
    clean = torch.rand(4, 3, 4, 4)
    labels = torch.tensor([0, 2, 1, 0])

    adversarial = make_adversarial(model, clean, labels, "fgsm", 8 / 255, pgd_steps=1)
    frozen = asdict(
        compute_feature_metrics(model, clean, adversarial, labels, max_layers=3)
    )
    batched = compute_e36_metrics_batched(
        model,
        clean,
        labels,
        epsilon=8 / 255,
        batch_size=2,
        max_layers=3,
    )

    assert set(batched) == set(E36_METRICS)
    for metric in E36_METRICS:
        assert batched[metric] == pytest.approx(frozen[metric], rel=2e-5, abs=2e-6)


def test_batched_e36_rejects_nonpositive_batch_size():
    try:
        from robust_metric_lab.e36_batched import compute_e36_metrics_batched
    except ImportError:
        pytest.fail("memory-bounded E36 evaluator is not implemented")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, kernel_size=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(2, 2),
    ).eval()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        compute_e36_metrics_batched(
            model,
            torch.rand(1, 3, 2, 2),
            torch.tensor([0]),
            epsilon=8 / 255,
            batch_size=0,
        )
