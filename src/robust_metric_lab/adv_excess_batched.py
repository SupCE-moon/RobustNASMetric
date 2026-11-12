"""Memory-bounded evaluation of the frozen adversarial-excess metric."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .adv_excess_metric import adv_excess_per_sample, fixed_random_linf
from .attacks import make_adversarial
from .metrics import ActivationRecorder, _anchor_indices


@dataclass(frozen=True)
class AdvExcessResult:
    value: float
    diagnostics: dict[str, float | int]


def _record_features(model: torch.nn.Module, inputs: torch.Tensor) -> list[torch.Tensor]:
    with torch.no_grad():
        with ActivationRecorder(model, activation_mode="all_leaf") as recorder:
            model(inputs)
        return recorder.outputs


def _perturbation_norms(
    changed: torch.Tensor, clean: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = (changed - clean).double().flatten(1)
    return difference.abs().max(dim=1).values, difference.norm(p=2, dim=1)


def compute_adv_excess_batched(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    max_layers: int = 4,
    sample_offset: int = 0,
) -> AdvExcessResult:
    """Average per-sample excess exactly, independent of chunk boundaries."""
    if batch_size <= 0 or max_layers <= 0:
        raise ValueError("batch_size and max_layers must be positive")
    if sample_offset < 0:
        raise ValueError("sample_offset must be nonnegative")
    if clean.ndim != 4 or clean.shape[0] == 0 or clean.shape[0] != labels.shape[0]:
        raise ValueError("clean and labels must have the same positive length")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not torch.isfinite(clean).all():
        raise ValueError("clean inputs must be finite")

    model.eval()
    total_samples = int(clean.shape[0])
    totals = {
        "excess": 0.0,
        "adv_drift": 0.0,
        "random_drift": 0.0,
        "adv_linf": 0.0,
        "adv_l2": 0.0,
        "random_linf": 0.0,
        "random_l2": 0.0,
    }
    expected_indices: tuple[int, ...] | None = None
    expected_feature_count: int | None = None
    for start in range(0, total_samples, batch_size):
        stop = min(start + batch_size, total_samples)
        clean_chunk = clean[start:stop]
        adversarial = make_adversarial(
            model, clean_chunk, labels[start:stop], "fgsm", epsilon, pgd_steps=1
        )
        random_input = fixed_random_linf(
            clean_chunk, epsilon, range(sample_offset + start, sample_offset + stop)
        )
        clean_features = _record_features(model, clean_chunk)
        adversarial_features = _record_features(model, adversarial)
        random_features = _record_features(model, random_input)
        feature_count = len(clean_features)
        if feature_count != len(adversarial_features) or feature_count != len(random_features):
            raise ValueError("feature count changed between clean, adversarial, and random")
        if any(
            not torch.isfinite(feature).all()
            for features in (clean_features, adversarial_features, random_features)
            for feature in features
        ):
            raise ValueError("nonfinite feature activation")
        indices = tuple(_anchor_indices(feature_count, max_layers=max_layers))
        if len(indices) < 2:
            raise ValueError("evaluation requires at least two feature anchors")
        if expected_indices is None:
            expected_indices = indices
            expected_feature_count = feature_count
        elif indices != expected_indices or feature_count != expected_feature_count:
            raise ValueError("feature anchors changed between chunks")

        score, adv_drift, random_drift = adv_excess_per_sample(
            clean_features, adversarial_features, random_features, indices
        )
        adv_linf, adv_l2 = _perturbation_norms(adversarial, clean_chunk)
        random_linf, random_l2 = _perturbation_norms(random_input, clean_chunk)
        values = {
            "excess": score,
            "adv_drift": adv_drift,
            "random_drift": random_drift,
            "adv_linf": adv_linf,
            "adv_l2": adv_l2,
            "random_linf": random_linf,
            "random_l2": random_l2,
        }
        for name, samples in values.items():
            if samples.shape != (stop - start,) or not torch.isfinite(samples).all():
                raise ValueError(f"{name} produced invalid per-sample values")
            totals[name] += float(samples.sum().item())
        if clean.is_cuda:
            torch.cuda.empty_cache()

    means = {name: total / total_samples for name, total in totals.items()}
    if any(not math.isfinite(value) for value in means.values()):
        raise ValueError("evaluation produced nonfinite summary")
    diagnostics: dict[str, float | int] = {"samples": total_samples}
    diagnostics.update({f"{name}_mean": mean for name, mean in means.items() if name != "excess"})
    return AdvExcessResult(value=means["excess"], diagnostics=diagnostics)
