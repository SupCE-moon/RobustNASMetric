"""Validate a complete, unseen three-space activation-survival holdout package."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path
from pathlib import PurePosixPath


HOLDOUT_COUNTS = {
    "rb40_mature": 11,
    **{f"nb201_e{epoch}": 64 for epoch in (0, 5, 10, 25, 50)},
    **{f"darts_e{epoch}": 20 for epoch in (0, 100)},
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field}: expected lowercase SHA-256")


def validate_holdout_package(package: dict) -> dict:
    """Reject missing, duplicated, contaminated or mismatched holdout items.

    This is a pre-seal structural check. A separate target verifier must prove
    the checkpoint/target files and the attack protocol before sealing.
    """
    if package.get("expected_rows") != 371:
        raise ValueError("expected_rows must be 371")
    if not isinstance(package.get("selection_seed"), int):
        raise ValueError("selection_seed required")
    _hash(package.get("sample_sha256"), "sample SHA")

    rows = package.get("objects")
    if not isinstance(rows, list):
        raise ValueError("objects must be a list")
    raw_keys = [
        (row.get("setting"), row.get("model_or_arch_id"))
        for row in rows if isinstance(row, dict)
    ]
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("duplicate holdout key")
    counts = collections.Counter(row.get("setting") for row in rows if isinstance(row, dict))
    for setting, expected in HOLDOUT_COUNTS.items():
        if counts[setting] != expected:
            raise ValueError(f"{setting}: expected {expected}, got {counts[setting]}")
    if len(rows) != 371 or set(counts) != set(HOLDOUT_COUNTS):
        raise ValueError("holdout key set mismatch")

    seen_keys: set[tuple[str, str]] = set()
    per_setting: dict[str, dict[str, str]] = collections.defaultdict(dict)
    rows_by_setting: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    targets: dict[tuple[str, str], tuple[float, str]] = {}
    for row in rows:
        setting = row["setting"]
        model_id = row.get("model_or_arch_id")
        signature = row.get("signature")
        if not isinstance(model_id, str) or not model_id or not isinstance(signature, str) or not signature:
            raise ValueError("model ID and structural signature required")
        key = (setting, model_id)
        if key in seen_keys:
            raise ValueError(f"duplicate holdout key: {key}")
        seen_keys.add(key)

        space = "rb40" if setting == "rb40_mature" else setting.split("_e", 1)[0]
        expected_epoch = None if space == "rb40" else int(setting.rsplit("e", 1)[1])
        if row.get("checkpoint_epoch") != expected_epoch:
            raise ValueError(f"{setting}: checkpoint epoch mismatch")
        if row.get("prior_overlap") is not False:
            raise ValueError(f"{setting}: prior overlap not disproved")
        path = row.get("checkpoint_path")
        if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError(f"{setting}: checkpoint path must be relative and safe")
        _hash(row.get("checkpoint_sha256"), "checkpoint SHA")
        source = row.get("target_source_id")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{setting}: target source required")
        value = row.get("target_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{setting}: finite target required")
        target_key = (space, model_id)
        target_definition = (float(value), source)
        if target_key in targets and targets[target_key] != target_definition:
            raise ValueError(f"{setting}: target differs across stages")
        targets[target_key] = target_definition
        per_setting[setting][model_id] = signature
        rows_by_setting[setting][model_id] = row

    nb_base = per_setting["nb201_e0"]
    for epoch in (5, 10, 25, 50):
        if per_setting[f"nb201_e{epoch}"] != nb_base:
            raise ValueError("NB201 architectures/signatures differ across stages")
    if per_setting["darts_e100"] != per_setting["darts_e0"]:
        raise ValueError("DARTS architectures/signatures differ across stages")
    for model_id, final_row in rows_by_setting["darts_e100"].items():
        final_hash = final_row["checkpoint_sha256"]
        for stage in ("darts_e0", "darts_e100"):
            if rows_by_setting[stage][model_id].get("target_checkpoint_sha256") != final_hash:
                raise ValueError(f"{stage}: target checkpoint mismatch")
    return package


def seal_holdout_package(package: dict, path: Path) -> dict:
    """Write a canonical, content-hashed manifest once after target verification.

    The target verifier supplies ``targets_verified`` and its report hash.
    This function does not itself prove the files or protocol; callers must
    inspect and hash the verifier report before invoking it.
    """
    validate_holdout_package(package)
    if package.get("targets_verified") is not True:
        raise ValueError("verified targets required before sealing")
    for field in (
        "source_audit_sha256", "target_verification_sha256", "protocol_sha256",
        "analysis_sha256", "code_sha256", "environment_sha256",
    ):
        _hash(package.get(field), field)
    payload = {key: value for key, value in package.items() if key not in ("sealed", "sha256")}
    payload_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sealed = {**payload, "sealed": True, "sha256": hashlib.sha256(payload_bytes).hexdigest()}
    output = json.dumps(sealed, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(output)
        stream.flush()
        os.fsync(stream.fileno())
    return sealed
