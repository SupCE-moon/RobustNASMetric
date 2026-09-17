"""Uniform, input-only analysis for the frozen student benchmark."""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .metric_registry import METRIC_BY_NAME, METRIC_SPECS
from .student_benchmark import BENCHMARK_SETTINGS, SETTING_BY_KEY, all_setting_mean


REQUIRED_COLUMNS = {
    "setting",
    "metric_name",
    "model_or_arch_id",
    "metric_value",
    "target_value",
    "status",
}
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260808
SETTING_KEYS = tuple(setting.key for setting in BENCHMARK_SETTINGS)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _require_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _direction_adjust(value: float | None, direction: str) -> float | None:
    if value is None:
        return None
    return -value if direction == "lower" else value


def _correlations(metric_values: list[float], targets: list[float], direction: str) -> dict[str, float | None]:
    if len(metric_values) < 2:
        return {"raw_spearman": None, "spearman": None, "pearson": None, "kendall": None}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        raw_spearman = float(stats.spearmanr(metric_values, targets).statistic)
        pearson = float(stats.pearsonr(metric_values, targets).statistic)
        kendall = float(stats.kendalltau(metric_values, targets).statistic)
    values: dict[str, float | None] = {}
    for key, value in (("raw_spearman", raw_spearman), ("pearson", pearson), ("kendall", kendall)):
        values[key] = value if math.isfinite(value) else None
    values["spearman"] = _direction_adjust(values["raw_spearman"], direction)
    return values


def _bootstrap(metric_values: list[float], targets: list[float], direction: str) -> tuple[float | None, float | None, int]:
    if len(metric_values) < 2:
        return None, None, 0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_array = np.asarray(metric_values, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPEATS):
        indexes = rng.integers(0, len(metric_array), size=len(metric_array))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            value = float(stats.spearmanr(metric_array[indexes], target_array[indexes]).statistic)
        if math.isfinite(value):
            values.append(-value if direction == "lower" else value)
    if len(values) < 2:
        return None, None, len(values)
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(low), float(high), len(values)


def summarize_metric(frame: pd.DataFrame, metric_name: str, setting: str) -> dict[str, Any]:
    """Summarize one metric/setting, retaining diagnostics for unusable rows."""
    if metric_name not in METRIC_BY_NAME:
        raise ValueError(f"unknown metric: {metric_name}")
    if setting not in SETTING_BY_KEY:
        raise ValueError(f"unknown benchmark setting: {setting}")
    _require_schema(frame)
    selected = frame[(frame["metric_name"] == metric_name) & (frame["setting"] == setting)]
    total_rows = len(selected)
    failed_rows = 0
    invalid_rows = 0
    valid_values: list[float] = []
    valid_targets: list[float] = []
    valid_ids: list[str] = []
    runtime_values: list[float] = []
    has_runtime = "runtime_sec" in selected.columns

    for _, row in selected.iterrows():
        status = str(row.get("status", ""))
        if status != "ok":
            failed_rows += 1
            continue
        value = _as_float(row.get("metric_value"))
        target = _as_float(row.get("target_value"))
        if value is None or target is None:
            invalid_rows += 1
            continue
        valid_values.append(value)
        valid_targets.append(target)
        valid_ids.append(str(row.get("model_or_arch_id", "")))
        if has_runtime:
            runtime = _as_float(row.get("runtime_sec"))
            if runtime is not None:
                runtime_values.append(runtime)

    duplicate_ids = len(valid_ids) - len(set(valid_ids))
    reasons: list[str] = []
    if total_rows == 0:
        reasons.append("no rows for setting")
    if failed_rows:
        reasons.append(f"{failed_rows} failure row(s)")
    if invalid_rows:
        reasons.append(f"{invalid_rows} invalid/nonfinite row(s)")
    if duplicate_ids:
        reasons.append(f"duplicate model/architecture ID(s): {duplicate_ids}")

    correlations = _correlations(valid_values, valid_targets, METRIC_BY_NAME[metric_name].direction)
    ci_low, ci_high, bootstrap_valid = _bootstrap(
        valid_values, valid_targets, METRIC_BY_NAME[metric_name].direction
    )
    if len(valid_values) < 2:
        reasons.append("insufficient valid samples")
    elif correlations["spearman"] is None or correlations["pearson"] is None or correlations["kendall"] is None:
        reasons.append("constant vector or undefined correlation")
    if ci_low is None or ci_high is None:
        reasons.append("insufficient defined bootstrap repeats")

    status = "ok" if not reasons else "invalid"
    runtime_total = sum(runtime_values) if has_runtime and runtime_values else None
    runtime_mean = (runtime_total / len(runtime_values)) if runtime_total is not None else None
    result: dict[str, Any] = {
        "metric_name": metric_name,
        "setting": setting,
        "expected_rows": SETTING_BY_KEY[setting].expected_rows,
        "total_rows": total_rows,
        "valid_rows": len(valid_values),
        "failed_rows": failed_rows,
        "invalid_rows": invalid_rows,
        "duplicate_ids": duplicate_ids,
        "raw_spearman": correlations["raw_spearman"],
        "direction_adjusted_spearman": correlations["spearman"],
        "spearman": correlations["spearman"],
        "pearson": correlations["pearson"],
        "kendall": correlations["kendall"],
        "kendall_tau": correlations["kendall"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_valid_repeats": bootstrap_valid,
        "runtime_total_sec": runtime_total,
        "runtime_mean_sec": runtime_mean,
        "status": status,
        "reason": "; ".join(reasons),
    }
    return result


def _finite_summary(summary: Mapping[str, Any]) -> bool:
    return all(
        _as_float(summary.get(key)) is not None
        for key in ("spearman", "ci_low", "ci_high")
    )


def analyze_frame(frame: pd.DataFrame, *, input_files: Iterable[str] | None = None) -> dict[str, Any]:
    """Analyze a unified metric-value frame under the frozen eight-setting grid."""
    _require_schema(frame)
    unknown_metrics = sorted({str(value) for value in frame["metric_name"].dropna() if str(value) not in METRIC_BY_NAME})
    if unknown_metrics:
        raise ValueError(f"unknown metric: {', '.join(unknown_metrics)}")
    unknown_settings = sorted({str(value) for value in frame["setting"].dropna() if str(value) not in SETTING_BY_KEY})
    if unknown_settings:
        raise ValueError(f"unknown benchmark setting: {', '.join(unknown_settings)}")

    metric_names = [spec.name for spec in METRIC_SPECS if spec.name in set(frame["metric_name"].astype(str))]
    correlations: list[dict[str, Any]] = []
    by_metric: dict[str, dict[str, dict[str, Any]]] = {}
    for metric_name in metric_names:
        by_metric[metric_name] = {}
        for setting in SETTING_KEYS:
            summary = summarize_metric(frame, metric_name, setting)
            if summary["total_rows"] != 0 and summary["total_rows"] != summary["expected_rows"]:
                summary["status"] = "invalid"
                extra = f"expected {summary['expected_rows']} rows, found {summary['total_rows']}"
                summary["reason"] = f"{summary['reason']}; {extra}" if summary["reason"] else extra
            if summary["status"] == "ok" and not _finite_summary(summary):
                summary["status"] = "invalid"
                summary["reason"] = "nonfinite correlation or confidence interval"
            by_metric[metric_name][setting] = summary
            correlations.append(summary)

    all_setting_table: list[dict[str, Any]] = []
    for metric_name in metric_names:
        summaries = by_metric[metric_name]
        values: dict[str, float] = {}
        missing: list[str] = []
        reasons: list[str] = []
        for setting in SETTING_KEYS:
            summary = summaries[setting]
            utility = summary["spearman"] if summary["status"] == "ok" else None
            if utility is None:
                missing.append(setting)
                if summary["reason"]:
                    reasons.append(f"{setting}: {summary['reason']}")
            else:
                values[setting] = utility
        complete = len(values) == len(SETTING_KEYS)
        mean = all_setting_mean(values) if complete else None
        row: dict[str, Any] = {"metric_name": metric_name}
        row.update({setting: values.get(setting) for setting in SETTING_KEYS})
        row["all_setting_mean"] = mean
        row["status"] = "complete" if complete else "incomplete"
        row["reason"] = "" if complete else "missing/invalid settings: " + ", ".join(missing) + ("; " + " | ".join(reasons) if reasons else "")
        all_setting_table.append(row)
    return {
        "correlations": correlations,
        "all_setting_table": all_setting_table,
        "input_files": list(input_files or []),
    }


CORRELATION_FIELDS = [
    "metric_name", "setting", "expected_rows", "total_rows", "valid_rows", "failed_rows", "invalid_rows",
    "duplicate_ids", "raw_spearman", "direction_adjusted_spearman", "pearson", "kendall", "ci_low", "ci_high",
    "bootstrap_repeats", "bootstrap_valid_repeats", "runtime_total_sec", "runtime_mean_sec", "status", "reason",
]
TABLE_FIELDS = ["metric_name", *SETTING_KEYS, "all_setting_mean", "status", "reason"]


def _write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _report(result: Mapping[str, Any]) -> str:
    correlations = list(result["correlations"])
    complete = [row["metric_name"] for row in result["all_setting_table"] if row["status"] == "complete"]
    incomplete = [row for row in result["all_setting_table"] if row["status"] != "complete"]
    lines = [
        "# Student benchmark analysis",
        "",
        "Input: " + (", ".join(result.get("input_files", [])) or "in-memory frame"),
        "",
        "Protocol: frozen eight settings from `student_benchmark.py`; metric directions from `METRIC_BY_NAME`; bootstrap=2,000 paired resamples, seed=20260808.",
        "",
        f"Complete metrics: {len(complete)} ({', '.join(complete) or 'none'}).",
        f"Incomplete metrics: {len(incomplete)}.",
        "Incomplete results have no `all_setting_mean` and must not be used for ranking.",
        "",
        f"Analyzed per-setting rows: {len(correlations)}.",
    ]
    for row in incomplete:
        lines.append(f"- `{row['metric_name']}`: {row['reason']}")
    return "\n".join(lines) + "\n"


def write_analysis_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write the three analysis artifacts, refusing to overwrite any target."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "correlations.csv", output / "all_setting_table.csv", output / "analysis_report.md"]
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output: " + ", ".join(existing))
    _write_csv(targets[0], CORRELATION_FIELDS, result["correlations"])
    _write_csv(targets[1], TABLE_FIELDS, result["all_setting_table"])
    targets[2].write_text(_report(result), encoding="utf-8")
