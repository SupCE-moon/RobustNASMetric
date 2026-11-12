"""Frozen adversarial-excess feature drift and its deterministic control."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch

from .attacks import clamp_linf


RANDOM_SIGN_SEED = 20261024


def fixed_random_linf(
    clean: torch.Tensor,
    epsilon: float,
    global_indices: Sequence[int],
) -> torch.Tensor:
    """Perturb each sample with a model-independent PCG64 random sign."""
    indices = tuple(global_indices)
    if clean.ndim != 4 or not clean.is_floating_point() or clean.shape[0] == 0:
        raise ValueError("clean must be a nonempty floating NCHW tensor")
    if not torch.isfinite(clean).all():
        raise ValueError("clean contains nonfinite values")
    if len(indices) != clean.shape[0]:
        raise ValueError("one global index is required for each sample")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError("global indices must be nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("global indices must be unique")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")

    shape = tuple(clean.shape[1:])
    signs = []
    for index in indices:
        bits = np.random.Generator(np.random.PCG64(RANDOM_SIGN_SEED + index)).integers(
            0, 2, size=shape, dtype=np.int8
        )
        signs.append(torch.as_tensor(bits * 2 - 1, dtype=clean.dtype, device=clean.device))
    reference = clean.detach()
    return clamp_linf(reference + epsilon * torch.stack(signs), reference, epsilon).detach()


def _unit_centered(feature: torch.Tensor) -> torch.Tensor:
    flat = feature.double().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norm = centered.norm(p=2, dim=1, keepdim=True)
    return torch.where(
        norm > 1e-12,
        centered / norm.clamp_min(1e-12),
        torch.zeros_like(centered),
    )


def _distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 0.5 * (_unit_centered(left) - _unit_centered(right)).norm(p=2, dim=1)


def _sample_median(values: torch.Tensor) -> torch.Tensor:
    ordered = values.sort(dim=1).values
    count = ordered.shape[1]
    if count % 2:
        return ordered[:, count // 2]
    return (ordered[:, count // 2 - 1] + ordered[:, count // 2]) / 2


def adv_excess_per_sample(
    clean_features: Sequence[torch.Tensor],
    adversarial_features: Sequence[torch.Tensor],
    random_features: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample excess, adversarial, and random median drifts."""
    selected = tuple(indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("anchor indices must be nonempty and unique")
    counts = (len(clean_features), len(adversarial_features), len(random_features))
    if len(set(counts)) != 1:
        raise ValueError("the three feature lists must have identical lengths")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= counts[0] for i in selected):
        raise ValueError("anchor index is out of range")

    adv_layers: list[torch.Tensor] = []
    random_layers: list[torch.Tensor] = []
    batch: int | None = None
    for index in selected:
        clean = clean_features[index]
        adversarial = adversarial_features[index]
        random = random_features[index]
        if clean.ndim != 4 or clean.shape != adversarial.shape or clean.shape != random.shape:
            raise ValueError("anchor feature shapes must match and be NCHW")
        if clean.shape[0] == 0 or (batch is not None and clean.shape[0] != batch):
            raise ValueError("anchor batch sizes must be identical and nonempty")
        if not all(
            value.is_floating_point() and torch.isfinite(value).all()
            for value in (clean, adversarial, random)
        ):
            raise ValueError("anchor features must be finite floating tensors")
        batch = clean.shape[0]
        adv_layers.append(_distance(clean, adversarial))
        random_layers.append(_distance(clean, random))

    adversarial_drift = torch.stack(adv_layers, dim=1)
    random_drift = torch.stack(random_layers, dim=1)
    excess = _sample_median(adversarial_drift - random_drift)
    adv_summary = _sample_median(adversarial_drift)
    random_summary = _sample_median(random_drift)
    return excess, adv_summary, random_summary
