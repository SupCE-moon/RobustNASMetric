from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from .attacks import make_adversarial
from .e37_metrics import E37_METRIC_NAMES, compute_e37_per_sample
from .metrics import ActivationRecorder, _anchor_indices


def compute_e37_metrics_batched(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    max_layers: int = 4,
) -> Mapping[str, float]:
    """Compute the frozen E37 per-sample formulas with exact sample weighting."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if clean.shape[0] == 0:
        raise ValueError("clean inputs and labels must have a positive length")
    if clean.shape[0] != labels.shape[0]:
        raise ValueError("clean inputs and labels must have the same positive length")
    model.eval()
    totals = {name: 0.0 for name in E37_METRIC_NAMES}
    total_samples = int(clean.shape[0])
    expected_indices: tuple[int, ...] | None = None
    for start in range(0, total_samples, batch_size):
        stop = min(start + batch_size, total_samples)
        clean_chunk = clean[start:stop]
        adversarial = make_adversarial(
            model,
            clean_chunk,
            labels[start:stop],
            "fgsm",
            epsilon,
            pgd_steps=1,
        )
        with torch.no_grad():
            with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
                model(clean_chunk)
            clean_features = recorder.outputs
            with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
                model(adversarial)
            adversarial_features = recorder.outputs
        indices = tuple(
            _anchor_indices(
                min(len(clean_features), len(adversarial_features)),
                max_layers=max_layers,
            )
        )
        if len(indices) < 2:
            raise ValueError("E37 evaluation requires at least two anchors")
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError("E37 activation anchors changed between chunks")
        per_sample = compute_e37_per_sample(
            clean_features,
            adversarial_features,
            indices,
        )
        for name in E37_METRIC_NAMES:
            values = per_sample[name]
            if values.shape != (stop - start,):
                raise ValueError(f"E37 metric {name} returned an invalid sample shape")
            if not torch.isfinite(values).all():
                raise ValueError(f"E37 metric {name} produced nonfinite values")
            totals[name] += float(values.sum().item())
        if clean.is_cuda:
            torch.cuda.empty_cache()
    result = {name: total / total_samples for name, total in totals.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("E37 evaluation produced a nonfinite aggregate")
    return result
