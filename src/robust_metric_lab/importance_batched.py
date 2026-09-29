from __future__ import annotations

import math

import torch

from .attacks import make_adversarial
from .metrics import (
    ActivationRecorder,
    _anchor_indices,
    _channel_importance,
    _perturbation_capacity_importance,
)


def accumulate_global_channel_importance(
    model: torch.nn.Module,
    clean: torch.Tensor,
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Return the exact sample mean of per-channel spatial RMS gradients."""
    totals: dict[int, torch.Tensor] = {}
    total_samples = int(clean.shape[0])
    for start in range(0, total_samples, batch_size):
        stop = min(start + batch_size, total_samples)
        chunk = _channel_importance(model, clean[start:stop])
        weight = stop - start
        if totals and set(chunk) != set(totals):
            raise ValueError("activation layout changed between importance chunks")
        for index, value in chunk.items():
            weighted = value.detach() * weight
            totals[index] = weighted if index not in totals else totals[index] + weighted
        if clean.is_cuda:
            torch.cuda.empty_cache()
    return {index: value / total_samples for index, value in totals.items()}


def accumulate_weighted_capacity_drift(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    importance: dict[int, torch.Tensor],
    *,
    epsilon: float,
    batch_size: int,
    max_layers: int,
) -> float:
    total_samples = int(clean.shape[0])
    weighted_total = 0.0
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
        if not indices:
            raise ValueError("importance evaluation requires at least one 4D activation")
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError("activation anchors changed between importance chunks")
        missing = [index for index in indices if index not in importance]
        if missing:
            raise ValueError(f"missing global channel importance for anchors: {missing}")
        chunk_value = torch.stack(
            [
                _perturbation_capacity_importance(
                    clean_features[index],
                    adversarial_features[index],
                    importance[index],
                )
                for index in indices
            ]
        ).mean()
        value = float(chunk_value.item())
        if not math.isfinite(value):
            raise ValueError("importance metric produced a nonfinite chunk")
        weighted_total += value * (stop - start)
        if clean.is_cuda:
            torch.cuda.empty_cache()
    result = weighted_total / total_samples
    if not math.isfinite(result):
        raise ValueError("importance metric produced a nonfinite result")
    return result


def compute_importance_metric_batched(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    max_layers: int = 4,
) -> float:
    """Compute frozen importance-gated capacity using exact two-pass chunks."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if clean.shape[0] == 0 or clean.shape[0] != labels.shape[0]:
        raise ValueError("clean inputs and labels must have the same positive length")
    model.eval()
    importance = accumulate_global_channel_importance(
        model,
        clean,
        batch_size=batch_size,
    )
    return accumulate_weighted_capacity_drift(
        model,
        clean,
        labels,
        importance,
        epsilon=epsilon,
        batch_size=batch_size,
        max_layers=max_layers,
    )
