from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .metric_eligibility import MetricEligibility, MetricEligibilityRecord
from .metric_registry import METRIC_BY_NAME
from .student_analysis import SETTING_KEYS, analyze_frame
from .student_benchmark import (
    BENCHMARK_SETTINGS,
    EPSILON,
    NUM_SAMPLES,
    SAMPLE_OFFSET,
    SEED,
    BenchmarkManifestItem,
)


RAW_KEY = (
    "benchmark",
    "setting",
    "model_or_arch_id",
    "checkpoint_epoch",
    "metric_name",
    "seed",
    "sample_offset",
    "num_samples",
)

REQUIRED_RAW_COLUMNS = {
    *RAW_KEY,
    "direction",
    "metric_value",
    "target_value",
    "status",
    "epsilon",
}


@dataclass(frozen=True)
class ManifestExpectation:
    benchmark: str
    checkpoint_epoch: int | None
    targets: Mapping[str, float]


def load_unique_metric_rows(paths: Iterable[str | Path]) -> pd.DataFrame:
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("at least one raw metric CSV is required")
    frames: list[pd.DataFrame] = []
    for path in path_list:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_RAW_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"raw metric CSV {path} is missing columns {missing}; "
                "precomputed correlation tables are not valid ranking input"
            )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    duplicate_mask = result.duplicated(list(RAW_KEY), keep=False)
    if duplicate_mask.any():
        example = result.loc[duplicate_mask, list(RAW_KEY)].iloc[0].to_dict()
        raise ValueError(f"duplicate raw key: {example}")
    return result


def manifest_expectations(
    rb40_items: Sequence[BenchmarkManifestItem],
    nb201_items: Sequence[BenchmarkManifestItem],
    darts_items: Sequence[BenchmarkManifestItem],
) -> dict[str, ManifestExpectation]:
    by_benchmark = {"rb40": rb40_items, "nb201": nb201_items, "darts": darts_items}
    result: dict[str, ManifestExpectation] = {}
    for setting in BENCHMARK_SETTINGS:
        matching = [
            item
            for item in by_benchmark[setting.benchmark]
            if item.checkpoint_epoch == setting.checkpoint_epoch
        ]
        targets = {item.model_or_arch_id: float(item.target_value) for item in matching}
        if len(targets) != len(matching):
            raise ValueError(f"duplicate manifest ID for {setting.key}")
        if len(targets) != setting.expected_rows:
            raise ValueError(
                f"manifest {setting.key}: expected {setting.expected_rows}, found {len(targets)}"
            )
        result[setting.key] = ManifestExpectation(
            benchmark=setting.benchmark,
            checkpoint_epoch=setting.checkpoint_epoch,
            targets=targets,
        )
    return result


def _finite(value: object, field: str, *, metric: str, setting: str, identifier: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metric}/{setting}/{identifier}: nonfinite {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{metric}/{setting}/{identifier}: nonfinite {field}")
    return number


def _epoch_matches(value: object, expected: int | None) -> bool:
    if expected is None:
        return value is None or value == "" or pd.isna(value)
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def validate_metric_grid(
    frame: pd.DataFrame,
    metric_name: str,
    manifests: Mapping[str, ManifestExpectation],
) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_RAW_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"raw rows missing required columns: {missing_columns}")
    if metric_name not in METRIC_BY_NAME:
        raise ValueError(f"unknown metric: {metric_name}")
    expected_settings = {setting.key for setting in BENCHMARK_SETTINGS}
    if set(manifests) != expected_settings:
        raise ValueError("manifest expectations must contain exactly the eight frozen settings")

    selected = frame[frame["metric_name"] == metric_name].copy()
    if selected.empty:
        raise ValueError(f"{metric_name}: no raw rows")
    duplicate_mask = selected.duplicated(list(RAW_KEY), keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{metric_name}: duplicate raw key")

    for setting in BENCHMARK_SETTINGS:
        rows = selected[selected["setting"] == setting.key]
        if len(rows) != setting.expected_rows:
            raise ValueError(
                f"{metric_name}/{setting.key}: expected {setting.expected_rows} rows, found {len(rows)}"
            )
        manifest = manifests[setting.key]
        actual_ids = set(rows["model_or_arch_id"].astype(str))
        expected_ids = set(manifest.targets)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"{metric_name}/{setting.key}: manifest ID mismatch; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        for _, row in rows.iterrows():
            identifier = str(row["model_or_arch_id"])
            prefix = f"{metric_name}/{setting.key}/{identifier}"
            if row["benchmark"] != manifest.benchmark or row["benchmark"] != setting.benchmark:
                raise ValueError(f"{prefix}: benchmark mismatch")
            if not _epoch_matches(row["checkpoint_epoch"], manifest.checkpoint_epoch):
                raise ValueError(f"{prefix}: checkpoint_epoch mismatch")
            if row["status"] != "ok":
                raise ValueError(f"{prefix}: status must be ok")
            if row["direction"] != METRIC_BY_NAME[metric_name].direction:
                raise ValueError(f"{prefix}: direction mismatch")
            for field, expected in (
                ("seed", SEED),
                ("sample_offset", SAMPLE_OFFSET),
                ("num_samples", NUM_SAMPLES),
            ):
                if _finite(row[field], field, metric=metric_name, setting=setting.key, identifier=identifier) != expected:
                    raise ValueError(f"{prefix}: {field} mismatch")
            epsilon = _finite(row["epsilon"], "epsilon", metric=metric_name, setting=setting.key, identifier=identifier)
            if not math.isclose(epsilon, EPSILON, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{prefix}: epsilon mismatch")
            _finite(row["metric_value"], "metric_value", metric=metric_name, setting=setting.key, identifier=identifier)
            target = _finite(row["target_value"], "target_value", metric=metric_name, setting=setting.key, identifier=identifier)
            if target != float(manifest.targets[identifier]):
                raise ValueError(f"{prefix}: target mismatch")

    unknown_settings = set(selected["setting"].astype(str)) - expected_settings
    if unknown_settings:
        raise ValueError(f"{metric_name}: unknown settings {sorted(unknown_settings)}")
    if len(selected) != 560:
        raise ValueError(f"{metric_name}: expected 560 rows, found {len(selected)}")
    return selected


def build_strict_ranking(
    frame: pd.DataFrame,
    manifests: Mapping[str, ManifestExpectation],
    eligibility: Sequence[MetricEligibilityRecord],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    accepted: list[pd.DataFrame] = []
    excluded_rows: list[dict[str, str]] = []
    for record in eligibility:
        if record.eligibility is not MetricEligibility.CROSS_SPACE:
            excluded_rows.append(
                {
                    "metric_name": record.metric_name,
                    "eligibility": record.eligibility.value,
                    "reason": record.reason,
                }
            )
            continue
        try:
            accepted.append(validate_metric_grid(frame, record.metric_name, manifests))
        except ValueError as exc:
            excluded_rows.append(
                {
                    "metric_name": record.metric_name,
                    "eligibility": "incomplete",
                    "reason": str(exc),
                }
            )

    columns = ["rank", "metric_name", *SETTING_KEYS, "all_setting_mean"]
    excluded = pd.DataFrame(
        excluded_rows,
        columns=["metric_name", "eligibility", "reason"],
    )
    if not accepted:
        return (
            pd.DataFrame(columns=columns),
            excluded,
            {"correlations": [], "all_setting_table": [], "input_files": []},
        )

    analysis = analyze_frame(pd.concat(accepted, ignore_index=True))
    table = pd.DataFrame(analysis["all_setting_table"])
    complete = table[table["status"] == "complete"].copy()
    if len(complete) != len(accepted):
        raise ValueError("validated raw grids did not produce complete analysis rows")
    complete = complete.sort_values(
        ["all_setting_mean", "metric_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    complete.insert(0, "rank", range(1, len(complete) + 1))
    return complete[columns], excluded, analysis
