"""Preregistered Conv/Linear sign-survival metric (higher is better)."""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from .attacks import make_adversarial


def call_survival(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Reference oracle for one aligned invocation, including zero as nonpositive."""
    if clean.shape != adversarial.shape or clean.numel() == 0:
        raise ValueError("activation shape mismatch or empty")
    if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(adversarial).all()):
        raise ValueError("nonfinite activation")
    return 1.0 - float((clean.gt(0) != adversarial.gt(0)).sum().item()) / clean.numel()


def mean_sample_scores(calls_by_sample: list[list[float]]) -> float:
    if not calls_by_sample or any(not calls for calls in calls_by_sample):
        raise ValueError("no Conv2d/Linear calls")
    result = math.fsum(math.fsum(calls) for calls in calls_by_sample) / len(calls_by_sample)
    if not math.isfinite(result):
        raise ValueError("nonfinite score")
    return result


def pair_score(model: nn.Module, clean: torch.Tensor, adversarial: torch.Tensor) -> dict:
    """Compare a single clean/adversarial pair without retaining float activations."""
    if clean.ndim < 2 or adversarial.ndim < 2 or clean.shape[0] != 1 or adversarial.shape[0] != 1:
        raise ValueError("pair_score requires one sample per forward pass")
    model.eval()
    calls: list[dict] = []
    rates: list[float] = []
    zero_rates: list[float] = []
    call_ids: list[str] = []
    handles = []
    phase = "clean"
    counts: dict[str, int] = {}
    adv_index = 0
    max_masks = 0

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            nonlocal adv_index, max_masks
            if not torch.is_tensor(output) or output.shape[0] != 1 or output.numel() == 0:
                raise ValueError("empty or non-tensor activation")
            if not bool(torch.isfinite(output).all()):
                raise ValueError("nonfinite activation")
            counts[name] = counts.get(name, 0) + 1
            call_id = f"{name}#{counts[name]}"
            shape = tuple(output.shape)
            positive = output.detach().gt(0).to(device="cpu")
            zero = output.detach().eq(0).to(device="cpu")
            if phase == "clean":
                calls.append({"id": call_id, "shape": shape, "positive": positive, "zero": zero})
                max_masks = max(max_masks, len(calls))
                return
            if adv_index >= len(calls) or calls[adv_index]["id"] != call_id:
                raise ValueError("clean/adv call identity mismatch")
            prior = calls[adv_index]
            if prior["shape"] != shape:
                raise ValueError("clean/adv activation shape mismatch")
            flips = int((prior["positive"] != positive).sum().item())
            both_zero = int((prior["zero"] & zero).sum().item())
            elements = positive.numel()
            rates.append(1.0 - flips / elements)
            zero_rates.append(both_zero / elements)
            call_ids.append(call_id)
            prior["positive"] = None
            prior["zero"] = None
            adv_index += 1
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            model(clean)
            if not calls:
                raise ValueError("no Conv2d/Linear calls")
            phase = "adversarial"
            counts.clear()
            model(adversarial)
        if adv_index != len(calls):
            raise ValueError("clean/adv call count mismatch")
        score = math.fsum(rates)
        if not math.isfinite(score):
            raise ValueError("nonfinite score")
        return {
            "score": score,
            "call_count": len(calls),
            "call_ids": call_ids,
            "retention": rates,
            "both_zero_rate": zero_rates,
            "max_retained_masks": max_masks,
        }
    finally:
        for handle in handles:
            handle.remove()


def sign_survival(
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float = 8 / 255,
) -> dict:
    """Mean of per-sample sums; caller freezes 64 samples and attack protocol."""
    if clean.ndim < 2 or clean.shape[0] == 0 or clean.shape[0] != labels.shape[0]:
        raise ValueError("clean inputs and labels must have the same positive length")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid epsilon")
    model.eval()
    started = time.perf_counter()
    scores: list[float] = []
    call_ids: list[str] | None = None
    retention_totals: list[float] = []
    zero_totals: list[float] = []
    max_masks = 0
    for index in range(int(clean.shape[0])):
        image = clean[index:index + 1]
        label = labels[index:index + 1]
        adversarial = make_adversarial(model, image, label, "fgsm", epsilon, pgd_steps=1)
        pair = pair_score(model, image, adversarial)
        if call_ids is None:
            call_ids = pair["call_ids"]
            retention_totals = [0.0] * len(call_ids)
            zero_totals = [0.0] * len(call_ids)
        elif pair["call_ids"] != call_ids:
            raise ValueError("Conv2d/Linear call identities changed between samples")
        scores.append(pair["score"])
        max_masks = max(max_masks, pair["max_retained_masks"])
        for j, value in enumerate(pair["retention"]):
            retention_totals[j] += value
        for j, value in enumerate(pair["both_zero_rate"]):
            zero_totals[j] += value
    assert call_ids is not None
    sample_count = len(scores)
    metric_value = math.fsum(scores) / sample_count
    if not math.isfinite(metric_value):
        raise ValueError("nonfinite aggregate score")
    peak_memory = torch.cuda.max_memory_allocated(clean.device) if clean.is_cuda else 0
    return {
        "metric_value": metric_value,
        "sample_count": sample_count,
        "call_count": len(call_ids),
        "diagnostics": {
            "call_ids": call_ids,
            "mean_retention": [value / sample_count for value in retention_totals],
            "mean_both_zero_rate": [value / sample_count for value in zero_totals],
            "retention_per_call_mean": metric_value / len(call_ids),
            "elapsed_sec": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(peak_memory),
            "max_retained_masks": max_masks,
        },
    }
