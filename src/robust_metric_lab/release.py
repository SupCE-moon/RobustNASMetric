from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from .metric_eligibility import MetricEligibility, MetricEligibilityRecord
from .strict_ranking import (
    RAW_KEY,
    ManifestExpectation,
    build_strict_ranking,
    validate_metric_grid,
)
from .student_analysis import SETTING_KEYS


_FROZEN_REPAIR_FIELDS = (
    "benchmark",
    "setting",
    "model_or_arch_id",
    "checkpoint_epoch",
    "metric_name",
    "direction",
    "target_value",
    "seed",
    "sample_offset",
    "num_samples",
    "epsilon",
    "checkpoint_path",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal(value: object) -> object:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _key(row: pd.Series) -> tuple[object, ...]:
    return tuple(_normal(row[column]) for column in RAW_KEY)


def _write_csv_temp(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp = Path(raw_path)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _write_json_temp(payload: Mapping[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp = Path(raw_path)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def repair_existing_metric_grid(
    source_path: str | Path,
    repair_path: str | Path,
    target_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Replace only explicitly failed raw rows and record an auditable hash manifest."""

    source_path, repair_path = Path(source_path), Path(repair_path)
    target_path, manifest_path = Path(target_path), Path(manifest_path)
    if target_path.exists() or manifest_path.exists():
        raise FileExistsError("repair target and manifest must not already exist")

    source = pd.read_csv(source_path, keep_default_na=False)
    repair = pd.read_csv(repair_path, keep_default_na=False)
    missing = sorted(set(_FROZEN_REPAIR_FIELDS) - set(source.columns))
    if missing:
        raise ValueError(f"source missing frozen fields: {missing}")
    missing = sorted(set(source.columns) - set(repair.columns))
    if missing:
        raise ValueError(f"repair missing source columns: {missing}")
    if source.duplicated(list(RAW_KEY), keep=False).any():
        raise ValueError("source contains duplicate raw key")
    if repair.duplicated(list(RAW_KEY), keep=False).any():
        raise ValueError("repair contains duplicate raw key")
    if repair.empty:
        raise ValueError("repair contains no rows")

    source_positions = {_key(row): index for index, row in source.iterrows()}
    replaced_keys: list[dict[str, object]] = []
    result = source.copy()
    for _, repaired in repair.iterrows():
        key = _key(repaired)
        if key not in source_positions:
            identity_fields = ("benchmark", "setting", "model_or_arch_id", "checkpoint_epoch", "metric_name")
            candidates = source[
                source.apply(
                    lambda row: all(_normal(row[field]) == _normal(repaired[field]) for field in identity_fields),
                    axis=1,
                )
            ]
            if len(candidates) == 1:
                original = candidates.iloc[0]
                for field in _FROZEN_REPAIR_FIELDS:
                    if _normal(original[field]) != _normal(repaired[field]):
                        raise ValueError(f"repair changed frozen field {field}: {key}")
            raise ValueError(f"repair key does not exist in source: {key}")
        position = source_positions[key]
        original = source.loc[position]
        if str(original["status"]) == "ok":
            raise ValueError(f"refusing to replace successful row: {key}")
        if str(repaired["status"]) != "ok":
            raise ValueError(f"repair status must be ok: {key}")
        for field in _FROZEN_REPAIR_FIELDS:
            if _normal(original[field]) != _normal(repaired[field]):
                raise ValueError(f"repair changed frozen field {field}: {key}")
        for field in ("metric_value", "target_value"):
            try:
                value = float(repaired[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"repair has nonfinite {field}: {key}") from exc
            if not math.isfinite(value):
                raise ValueError(f"repair has nonfinite {field}: {key}")
        result.loc[position, source.columns] = repaired[source.columns]
        replaced_keys.append({column: _normal(repaired[column]) for column in RAW_KEY})

    target_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        target_temp = _write_csv_temp(result, target_path)
        target_hash = _sha256(target_temp)
        payload: dict[str, object] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source_path.resolve()),
            "repair": str(repair_path.resolve()),
            "target": str(target_path.resolve()),
            "source_sha256": _sha256(source_path),
            "repair_sha256": _sha256(repair_path),
            "target_sha256": target_hash,
            "source_rows": len(source),
            "repair_rows": len(repair),
            "target_rows": len(result),
            "replaced_count": len(replaced_keys),
            "replaced_keys": replaced_keys,
        }
        manifest_temp = _write_json_temp(payload, manifest_path)
        os.replace(target_temp, target_path)
        target_temp = None
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
        return payload
    finally:
        if target_temp is not None:
            target_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def verify_strict_release(
    frame: pd.DataFrame,
    manifests: Mapping[str, ManifestExpectation],
    eligibility: Sequence[MetricEligibilityRecord],
    ranking: pd.DataFrame,
) -> dict[str, object]:
    """Recompute the strict ranking from raw rows and reject any inconsistent release."""

    if frame.duplicated(list(RAW_KEY), keep=False).any():
        raise ValueError("duplicate raw key")
    cross_space = [record for record in eligibility if record.eligibility is MetricEligibility.CROSS_SPACE]
    for record in cross_space:
        validate_metric_grid(frame, record.metric_name, manifests)

    computed, excluded, _ = build_strict_ranking(frame, manifests, eligibility)
    incomplete = excluded[excluded["eligibility"] == "incomplete"]
    if not incomplete.empty:
        raise ValueError(f"strict release contains incomplete metrics: {incomplete.to_dict('records')}")
    if set(computed["metric_name"]) != set(ranking["metric_name"]):
        raise ValueError("ranking metric set mismatch")
    expected_columns = {"rank", "metric_name", *SETTING_KEYS, "all_setting_mean"}
    missing_columns = sorted(expected_columns - set(ranking.columns))
    if missing_columns:
        raise ValueError(f"ranking missing columns: {missing_columns}")

    supplied = ranking.set_index("metric_name")
    for _, row in computed.iterrows():
        name = row["metric_name"]
        supplied_row = supplied.loc[name]
        if int(supplied_row["rank"]) != int(row["rank"]):
            raise ValueError(f"{name}: rank mismatch")
        for field in (*SETTING_KEYS, "all_setting_mean"):
            if not math.isclose(float(supplied_row[field]), float(row[field]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name}: {field} mismatch")

    return {
        "status": "PASS",
        "raw_rows": int(len(frame)),
        "unique_raw_keys": int(frame[list(RAW_KEY)].drop_duplicates().shape[0]),
        "ranked_metrics": int(len(computed)),
        "excluded_metrics": int(len(excluded)),
    }
