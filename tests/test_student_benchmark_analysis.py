from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from robust_metric_lab.student_analysis import (
    BOOTSTRAP_MIN_VALID_REPEATS,
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
    assert lower["raw_pearson"] == pytest.approx(-1.0)
    assert lower["direction_adjusted_pearson"] == pytest.approx(1.0)
    assert lower["raw_kendall"] == pytest.approx(-1.0)
    assert lower["direction_adjusted_kendall"] == pytest.approx(1.0)
    assert higher["raw_spearman"] == pytest.approx(-1.0)
    assert higher["spearman"] == pytest.approx(-1.0)
    assert higher["raw_pearson"] == pytest.approx(-1.0)
    assert higher["direction_adjusted_pearson"] == pytest.approx(-1.0)
    assert higher["raw_kendall"] == pytest.approx(-1.0)
    assert higher["direction_adjusted_kendall"] == pytest.approx(-1.0)
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


@pytest.mark.parametrize("column,value", [("metric_name", np.nan), ("metric_name", "  "), ("setting", np.nan), ("setting", "  ")])
def test_blank_or_nan_metric_and_setting_fail_with_row_diagnostic(column, value):
    frame = pd.DataFrame(_rows())
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=r"row 2"):
        analyze_frame(frame)


def test_runtime_invalid_values_invalidate_setting_and_are_not_averaged():
    frame = pd.DataFrame(_rows())
    frame.loc[0, "runtime_sec"] = np.nan
    frame.loc[1, "runtime_sec"] = np.inf
    frame.loc[2, "runtime_sec"] = -1.0
    result = summarize_metric(frame, "feature_delta", "rb40_mature")
    assert result["runtime_invalid_rows"] == 3
    assert result["runtime_total_sec"] is None
    assert result["runtime_mean_sec"] is None
    assert result["status"] == "invalid"
    assert "runtime" in result["reason"]


def test_bootstrap_minimum_is_explicit_and_constant_input_has_no_ci():
    assert BOOTSTRAP_MIN_VALID_REPEATS == 2
    frame = pd.DataFrame(_rows(n=2))
    frame["metric_value"] = 1.0
    result = summarize_metric(frame, "feature_delta", "rb40_mature")
    assert result["bootstrap_valid_repeats"] < BOOTSTRAP_MIN_VALID_REPEATS
    assert result["ci_low"] is None and result["ci_high"] is None


def _write_cli_csv(path: Path, metrics=("feature_delta",)) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for setting_spec in BENCHMARK_SETTINGS:
            rows.extend(_rows(metric, setting_spec.key, setting_spec.expected_rows))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_cli_multi_input_outputs_are_readable_and_stably_sorted(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    full = _write_cli_csv(first, ("trnas_r_score",))
    second_frame = _write_cli_csv(second, ("feature_delta",))
    output = tmp_path / "out"
    script = Path(__file__).parents[1] / "scripts" / "analyze_student_benchmark.py"
    env = {**__import__("os").environ, "PYTHONPATH": str(script.parents[1] / "src")}
    command = [sys.executable, str(script), "--input", str(first), str(second), "--output-dir", str(output)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    correlations = pd.read_csv(output / "correlations.csv")
    table = pd.read_csv(output / "all_setting_table.csv")
    assert len(correlations) == 16
    assert correlations["metric_name"].drop_duplicates().tolist() == ["feature_delta", "trnas_r_score"]
    assert correlations["setting"].drop_duplicates().tolist() == [s.key for s in BENCHMARK_SETTINGS]
    assert table["metric_name"].tolist() == ["feature_delta", "trnas_r_score"]
    assert "must not be used for ranking" in (output / "analysis_report.md").read_text(encoding="utf-8")
    second_run = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second_run.returncode != 0
    assert "overwrite" in second_run.stderr.lower()


def test_cli_rejects_schema_missing_and_cross_file_duplicate(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    frame = _write_cli_csv(first)
    frame.drop(columns=["status"]).to_csv(second, index=False)
    script = Path(__file__).parents[1] / "scripts" / "analyze_student_benchmark.py"
    env = {**__import__("os").environ, "PYTHONPATH": str(script.parents[1] / "src")}
    missing = subprocess.run([sys.executable, str(script), "--input", str(second), "--output-dir", str(tmp_path / "missing")], env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "missing required columns" in missing.stderr.lower()
    frame.head(1).to_csv(second, index=False)
    duplicate = subprocess.run([sys.executable, str(script), "--input", str(first), str(second), "--output-dir", str(tmp_path / "duplicate")], env=env, capture_output=True, text=True)
    assert duplicate.returncode != 0
    assert "duplicate key" in duplicate.stderr.lower()


@pytest.mark.parametrize("mutation", ["wrong_count", "failure", "nonfinite"])
def test_expected_count_failure_or_nonfinite_keeps_mean_empty(mutation):
    rows = []
    for setting_spec in BENCHMARK_SETTINGS:
        values = _rows(setting=setting_spec.key, n=setting_spec.expected_rows)
        if mutation == "wrong_count" and setting_spec.key == "rb40_mature":
            values = values[:-1]
        elif mutation == "failure" and setting_spec.key == "rb40_mature":
            values[0]["status"] = "failed"
        elif mutation == "nonfinite" and setting_spec.key == "rb40_mature":
            values[0]["metric_value"] = np.nan
        rows.extend(values)
    result = analyze_frame(pd.DataFrame(rows))
    row = result["all_setting_table"][0]
    assert row["all_setting_mean"] is None
    assert row["status"] == "incomplete"


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
