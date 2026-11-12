"""Numeric-only per-layer diagnostics for the frozen E37 DARTS evaluation."""

from __future__ import annotations

import math

import torch

from .attacks import make_adversarial
from .e37_metrics import (
    E37_METRIC_NAMES,
    _centered_channel_drift,
    _flow_drift,
    _spatial_cosine_drift,
    _spectrum_probabilities,
    compute_e37_per_sample,
)
from .metrics import _anchor_indices


def leaf_call_trace(
    model: torch.nn.Module, inputs: torch.Tensor
) -> tuple[list[torch.Tensor], list[dict]]:
    """Record every 4D leaf invocation in the same forward-hook order as E37."""
    features: list[torch.Tensor] = []
    metadata: list[dict] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    counts: dict[str, int] = {}

    def make_hook(path: str):
        def hook(_module, _args, output) -> None:
            if torch.is_tensor(output) and output.ndim == 4:
                counts[path] = counts.get(path, 0) + 1
                features.append(output.detach())
                metadata.append({
                    "module_path": path,
                    "invocation": counts[path],
                    "shape": list(output.shape),
                })
        return hook

    try:
        for path, module in model.named_modules():
            if not tuple(module.children()):
                handles.append(module.register_forward_hook(make_hook(path)))
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return features, metadata


def _scalar(value: torch.Tensor) -> float:
    number = float(value.item())
    if not math.isfinite(number):
        raise ValueError("nonfinite layer component")
    return number


def _anchor_values(clean: torch.Tensor, adversarial: torch.Tensor) -> dict[str, float]:
    p = _spectrum_probabilities(clean)
    q = _spectrum_probabilities(adversarial)
    affinity = (p.sqrt() * q.sqrt()).sum(dim=1)
    clean_flat = clean.flatten(1)
    adv_flat = adversarial.flatten(1)
    return {
        "clean_rms": _scalar(clean_flat.square().mean(dim=1).sqrt()),
        "adv_rms": _scalar(adv_flat.square().mean(dim=1).sqrt()),
        "clean_centered_l2": _scalar((clean_flat - clean_flat.mean(dim=1, keepdim=True)).norm(dim=1)),
        "adv_centered_l2": _scalar((adv_flat - adv_flat.mean(dim=1, keepdim=True)).norm(dim=1)),
        "channel": _scalar(_centered_channel_drift(clean, adversarial)),
        "spatial": _scalar(_spatial_cosine_drift(clean, adversarial)),
        "spectrum_fisher": _scalar(2.0 * torch.acos(affinity.clamp(0.0, 1.0))),
        "spectrum_triangular": _scalar(
            0.5 * ((p - q).square() / (p + q).clamp_min(1e-12)).sum(dim=1)
        ),
        "spectrum_ks": _scalar((p - q).cumsum(dim=1).abs().max(dim=1).values),
    }


def diagnose_e37(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    max_layers: int = 4,
) -> dict:
    """Rebuild nine original E37 scores while collecting only scalar summaries."""
    if clean.shape[0] == 0:
        raise ValueError("clean inputs and labels must have a positive length")
    if clean.shape[0] != labels.shape[0]:
        raise ValueError("clean inputs and labels must have the same positive length")
    model.eval()
    totals = {name: 0.0 for name in E37_METRIC_NAMES}
    sample_components: list[dict] = []
    anchor_metadata: list[dict] | None = None
    feature_count: int | None = None
    for sample_index in range(int(clean.shape[0])):
        x = clean[sample_index:sample_index + 1]
        y = labels[sample_index:sample_index + 1]
        adversarial = make_adversarial(model, x, y, "fgsm", epsilon, pgd_steps=1)
        with torch.no_grad():
            clean_features, clean_trace = leaf_call_trace(model, x)
            adv_features, adv_trace = leaf_call_trace(model, adversarial)
            if clean_trace != adv_trace:
                raise ValueError(f"trace mismatch at sample {sample_index}: clean/adv")
            indices = tuple(_anchor_indices(
                min(len(clean_features), len(adv_features)), max_layers=max_layers
            ))
            if len(indices) < 2:
                raise ValueError("E37 evaluation requires at least two anchors")
            selected_metadata = [dict(clean_trace[index], anchor_index=index) for index in indices]
            if anchor_metadata is None:
                anchor_metadata = selected_metadata
                feature_count = len(clean_features)
            elif selected_metadata != anchor_metadata or len(clean_features) != feature_count:
                raise ValueError(f"trace mismatch at sample {sample_index}: anchors")
            values = compute_e37_per_sample(clean_features, adv_features, indices)
            for name in E37_METRIC_NAMES:
                metric = values[name]
                if metric.shape != (1,) or not torch.isfinite(metric).all():
                    raise ValueError(f"E37 metric {name} has invalid sample value")
                totals[name] += float(metric.sum().item())
            layers = [_anchor_values(clean_features[index], adv_features[index]) for index in indices]
            transitions = [
                {"flow": _scalar(_flow_drift(
                    clean_features[left], clean_features[right],
                    adv_features[left], adv_features[right],
                ))}
                for left, right in zip(indices, indices[1:])
            ]
            sample_components.append({
                "sample_index": sample_index,
                "anchors": layers,
                "transitions": transitions,
            })
        if clean.is_cuda:
            torch.cuda.empty_cache()
    metric_values = {name: total / int(clean.shape[0]) for name, total in totals.items()}
    if any(not math.isfinite(value) for value in metric_values.values()):
        raise ValueError("E37 evaluation produced a nonfinite aggregate")
    return {
        "sample_components": sample_components,
        "anchor_metadata": anchor_metadata,
        "metric_values": metric_values,
        "sample_count": int(clean.shape[0]),
        "feature_count": feature_count,
    }
