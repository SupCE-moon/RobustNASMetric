"""Produce per-image evidence for a future sealed DARTS PGD-20 holdout.

This module has no run entrypoint until the independent architecture ledger and
all final weights are sealed. It never reads Ours or TRNAS scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from robust_metric_lab.attacks import make_adversarial
from scripts.verify_holdout_targets import _CIFAR10_TEST_IDS_SHA256, _DARTS_NORMALIZATION


TARGET_PROTOCOL = {
    "target_attack": "pgd20",
    "target_dataset": "cifar10_test",
    "target_epsilon": 8 / 255,
    "target_step_size": 2 / 255,
    "target_steps": 20,
    "target_restarts": 1,
    "target_random_start": True,
    "target_batch_size": 128,
    "target_random_seed": 400000,
    "target_normalization": _DARTS_NORMALIZATION,
}


def collect_pgd_correctness(model: torch.nn.Module, loader, device: str,
                            epsilon: float, steps: int) -> list[bool]:
    """Mirror `pgd_accuracy`, keeping its individual boolean results."""
    model.eval()
    correct: list[bool] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = make_adversarial(model, x, y, "pgd", epsilon, pgd_steps=steps)
        with torch.no_grad():
            batch = (model(x_adv).argmax(1) == y).tolist()
        correct.extend(bool(value) for value in batch)
    return correct


def build_target_evidence(identifier: str, final_checkpoint_sha256: str,
                          correct: list[bool]) -> tuple[dict, dict]:
    """Build the evidence record and matching verifier item, without writing."""
    if not isinstance(identifier, str) or not identifier.isdecimal():
        raise ValueError("numeric architecture ID required")
    if len(final_checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in final_checkpoint_sha256):
        raise ValueError("final checkpoint SHA-256 required")
    if len(correct) != 10000:
        raise ValueError("target evidence requires 10,000 images")
    if not all(type(value) is bool for value in correct):
        raise ValueError("target evidence requires boolean per-image results")
    accuracy = sum(correct) / 10000
    if not math.isfinite(accuracy):
        raise ValueError("nonfinite target accuracy")
    record = {
        "status": "ok",
        "model_or_arch_id": identifier,
        "checkpoint_sha256": final_checkpoint_sha256,
        "protocol": dict(TARGET_PROTOCOL),
        "sample_ids": list(range(10000)),
        "correct": correct,
    }
    item = {
        "setting": "darts_e100",
        "model_or_arch_id": identifier,
        "checkpoint_sha256": final_checkpoint_sha256,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "target_checkpoint_sha256": final_checkpoint_sha256,
        "target_value": accuracy,
        "target_source_id": f"darts-{identifier}-fulltest-pgd20",
        "target_sample_count": 10000,
        "target_sample_ids_sha256": _CIFAR10_TEST_IDS_SHA256,
        **TARGET_PROTOCOL,
    }
    return record, item


def write_target_evidence(path: Path, record: dict) -> str:
    """Write one immutable result and return the SHA-256 of its exact bytes."""
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()
