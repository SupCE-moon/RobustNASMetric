from __future__ import annotations

from collections.abc import Mapping

import torch

from .attacks import make_adversarial
from .metrics import (
    ActivationRecorder,
    _anchor_indices,
    _e36_cross_layer_stats,
    _e36_within_layer_stats,
)


E36_METRIC_NAMES = (
    "feature_measure_channel_gram_cosine_delta",
    "feature_measure_channel_gram_centered_rel_delta",
    "feature_measure_spatial_gram_cosine_delta",
    "feature_measure_spectrum_js_delta",
    "feature_measure_spectrum_emd_delta",
    "feature_measure_spectrum_hellinger_delta",
    "feature_measure_flow_corr_angular_delta",
    "feature_measure_flow_corr_symmetric_rel_delta",
)


def _chunk_e36_metrics(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    max_layers: int,
) -> dict[str, float]:
    adversarial = make_adversarial(
        model, clean, labels, "fgsm", epsilon, pgd_steps=1
    )
    with torch.no_grad():
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(clean)
        clean_features = recorder.outputs
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(adversarial)
        adversarial_features = recorder.outputs

    count = min(len(clean_features), len(adversarial_features))
    indices = _anchor_indices(count, max_layers=max_layers)
    if not indices:
        raise ValueError("E36 evaluation requires at least one 4D activation")

    within = [
        _e36_within_layer_stats(clean_features[index], adversarial_features[index])
        for index in indices
    ]
    within_values = [
        torch.stack([layer[position] for layer in within]).mean().item()
        for position in range(6)
    ]

    pairs = [
        _e36_cross_layer_stats(
            clean_features[left],
            clean_features[right],
            adversarial_features[left],
            adversarial_features[right],
        )
        for left, right in zip(indices, indices[1:])
    ]
    if not pairs:
        raise ValueError("E36 flow metrics require at least two anchor activations")
    cross_values = [
        torch.stack([pair[position] for pair in pairs]).mean().item()
        for position in range(2)
    ]
    return dict(zip(E36_METRIC_NAMES, [*within_values, *cross_values]))


def compute_e36_metrics_batched(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    max_layers: int = 4,
) -> Mapping[str, float]:
    """Compute the frozen E36 formulas in memory-bounded sample chunks.

    Every E36 primitive is a per-sample quantity followed by a mean. Weighting
    chunk means by their sample counts therefore preserves the frozen 64-sample
    definition while avoiding a single 64-sample activation/gradient graph.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if clean.shape[0] != labels.shape[0] or clean.shape[0] == 0:
        raise ValueError("clean inputs and labels must have the same positive length")

    totals = {name: 0.0 for name in E36_METRIC_NAMES}
    total_samples = int(clean.shape[0])
    for start in range(0, total_samples, batch_size):
        stop = min(start + batch_size, total_samples)
        chunk = _chunk_e36_metrics(
            model,
            clean[start:stop],
            labels[start:stop],
            epsilon=epsilon,
            max_layers=max_layers,
        )
        weight = stop - start
        for name in E36_METRIC_NAMES:
            totals[name] += chunk[name] * weight
        if clean.is_cuda:
            torch.cuda.empty_cache()

    return {name: totals[name] / total_samples for name in E36_METRIC_NAMES}
