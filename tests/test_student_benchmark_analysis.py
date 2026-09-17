from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_metric_lab.student_analysis import (
    analyze_frame,
    summarize_metric,
)
from robust_metric_lab.student_benchmark import BENCHMARK_SETTINGS


def _rows(metric: str = "feature_delta", setting: str = "rb40_mature", n: int = 6):
    return [
        {
            "setting": setting,
            "metric_name": metric,
            "model_or_arch_id": f"m{i}",
            "metric_value": float(i),
            "target_value": float(n - i),
            "status": "ok",
            "runtime_sec": float(i + 1),
        }
        for i in range(n)
    ]


def test_direction_adjusts_lower_but_not_higher_and_unknown_fails():
    lower = summarize_metric(pd.DataFrame(_rows()), "feature_delta", "rb40_mature")
    higher_rows = _rows("trnas_r_score")
    higher = summarize_metric(pd.DataFrame(higher_rows), "trnas_r_score", "rb40_mature")

    assert lower["raw_spearman"] == pytest.approx(-1.0)
    assert lower["spearman"] == pytest.approx(1.0)
    assert higher["raw_spearman"] == pytest.approx(-1.0)
    assert higher["spearman"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="unknown metric"):
        summarize_metric(pd.DataFrame(_rows("not_registered")), "not_registered", "rb40_mature")


def test_other_correlations_runtime_and_invalid_counts():
    frame = pd.DataFrame(_rows())
    frame.loc[0, "metric_value"] = np.nan
    frame.loc[1, "target_value"] = np.inf
    frame.loc[2, "status"] = "failed"
    result = summarize_metric(frame, "feature_delta", "rb40_mature")

    assert result["total_rows"] == 6
    assert result["valid_rows"] == 3
    assert result["failed_rows"] == 1
    assert result["invalid_rows"] == 2
    assert result["runtime_total_sec"] == pytest.approx(4 + 5 + 6)
    assert result["runtime_mean_sec"] == pytest.approx((4 + 5 + 6) / 3)
    assert math.isfinite(result["pearson"])
    assert math.isfinite(result["kendall"])
    assert result["status"] == "invalid"


def test_constant_vectors_are_invalid_without_pseudo_correlation():
    frame = pd.DataFrame(_rows())
    frame["metric_value"] = 1.0
    result = summarize_metric(frame, "feature_delta", "rb40_mature")
    assert result["status"] == "invalid"
    assert result["spearman"] is None
    assert "constant" in result["reason"]


def test_bootstrap_is_deterministic_and_reports_ci():
    frame = pd.DataFrame(_rows(n=12))
    first = summarize_metric(frame, "feature_delta", "rb40_mature")
    second = summarize_metric(frame, "feature_delta", "rb40_mature")
    assert first["bootstrap_repeats"] == 2000
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["ci_low"] <= first["spearman"] <= first["ci_high"]


def test_all_setting_mean_requires_complete_valid_grid():
    rows = []
    for setting in BENCHMARK_SETTINGS:
        rows.extend(_rows(setting=setting.key, n=setting.expected_rows))
    result = analyze_frame(pd.DataFrame(rows))
    assert result["all_setting_table"][0]["all_setting_mean"] == pytest.approx(1.0)

    incomplete = pd.DataFrame([row for row in rows if row["setting"] != "darts_e100"])
    incomplete_result = analyze_frame(incomplete)
    missing = next(r for r in incomplete_result["all_setting_table"] if r["metric_name"] == "feature_delta")
    assert missing["all_setting_mean"] is None
    assert missing["status"] == "incomplete"
    assert "darts_e100" in missing["reason"]


def test_duplicate_id_failure_nonfinite_do_not_produce_mean():
    rows = []
    for setting in BENCHMARK_SETTINGS:
        values = _rows(setting=setting.key, n=setting.expected_rows)
        if setting.key == "rb40_mature":
            values[1]["model_or_arch_id"] = values[0]["model_or_arch_id"]
        rows.extend(values)
    result = analyze_frame(pd.DataFrame(rows))
    row = result["all_setting_table"][0]
    assert row["all_setting_mean"] is None
    assert "duplicate" in row["reason"]


def test_schema_and_output_writer_are_strict(tmp_path: Path):
    from robust_metric_lab.student_analysis import write_analysis_outputs

    frame = pd.DataFrame(_rows())
    with pytest.raises(ValueError, match="missing required columns"):
        analyze_frame(frame.drop(columns=["target_value"]))
    outputs = analyze_frame(frame)
    write_analysis_outputs(outputs, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "correlations.csv",
        "all_setting_table.csv",
        "analysis_report.md",
    }
    with pytest.raises(FileExistsError):
        write_analysis_outputs(outputs, tmp_path)
