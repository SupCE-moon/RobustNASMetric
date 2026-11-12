"""Safely substitute successful recomputations for the exact failed raw keys."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from robust_metric_lab.student_runner import UNIQUE_KEY_COLUMNS


IMMUTABLE_FIELDS = (
    "benchmark", "setting", "model_or_arch_id", "checkpoint_epoch",
    "metric_name", "direction", "target_value", "seed", "sample_offset",
    "num_samples", "epsilon", "checkpoint_path",
)


def _key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in UNIQUE_KEY_COLUMNS)


def _finite_success(row: Mapping[str, str]) -> bool:
    try:
        return row["status"] == "ok" and math.isfinite(float(row["metric_value"]))
    except (KeyError, TypeError, ValueError):
        return False


def merge_failed_rows(
    original: Sequence[Mapping[str, str]],
    repairs: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Replace failed rows only; every original successful row remains byte-equivalent."""
    original_by_key = {}
    for row in original:
        key = _key(row)
        if key in original_by_key:
            raise ValueError(f"duplicate original key: {key}")
        original_by_key[key] = row
    repair_by_key = {}
    for repair in repairs:
        key = _key(repair)
        if key in repair_by_key:
            raise ValueError(f"duplicate repair key: {key}")
        prior = original_by_key.get(key)
        if prior is None or prior["status"] == "ok":
            raise ValueError(f"repair does not target one failed key: {key}")
        if not _finite_success(repair):
            raise ValueError(f"repair is not a finite success: {key}")
        for field in IMMUTABLE_FIELDS:
            if str(prior[field]) != str(repair[field]):
                raise ValueError(f"repair changes {field}: {key}")
        repair_by_key[key] = repair
    return [dict(repair_by_key.get(_key(row), row)) for row in original]
