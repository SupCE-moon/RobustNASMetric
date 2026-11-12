"""Tests for ``tools/build_metric_index.py``.

The index is only useful if it is honest about what it does not know. These
tests pin the two honesty rules: a metric with no data is left blank rather
than counted as zero, and a metric declared unusable but ranked is surfaced
instead of silently trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_metric_index as bmi  # noqa: E402


def registry(**extra) -> dict[str, dict]:
    base = {
        "phi": {"direction": "higher", "attribution": "TRNAS", "reference": "[11]"},
        "max": {"direction": "lower", "attribution": "Ours", "reference": "E37"},
    }
    base.update(extra)
    return base


def eligibility(**overrides) -> dict[str, dict]:
    base = {
        "phi": {"eligibility": "cross_space", "supported_settings": list(bmi.SETTINGS), "reason": "same"},
        "max": {"eligibility": "cross_space", "supported_settings": list(bmi.SETTINGS), "reason": "same"},
    }
    base.update(overrides)
    return base


def table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["strict_rank"] = pd.to_numeric(frame["strict_rank"], errors="coerce")
    return frame


def row(metric: str, rank, score: float, status: str = "ranked") -> dict:
    return {
        "metric_name": metric,
        "strict_rank": rank,
        "strict_status": status,
        "all_setting_mean": score,
        "raw_rows": 560,
        "exclusion_reason": None,
    }


def test_family_prefers_the_longest_matching_prefix() -> None:
    # "feature_measure_" must win over the shorter "feature_" prefix.
    assert bmi.family_of("feature_measure_multiview_max_delta") == "特征漂移度量家族"
    assert bmi.family_of("feature_hausdorff_delta") == "特征空间（其他）"
    assert bmi.family_of("trnas_consistency_phi") == "TRNAS 移植（Φ/R/Ψ）"


def test_unknown_prefix_falls_back_to_other() -> None:
    assert bmi.family_of("something_new") == "其他"


def test_coverage_is_none_when_rows_not_supplied() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78), row("max", 2, 0.52)]),
        registry(),
        eligibility(),
        None,
    )
    assert payload["coverage_supplied"] is False
    assert all(item["coverage"] is None for item in payload["rows"])


def test_coverage_counts_only_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {"metric_name": "phi", "setting": "rb40_mature", "metric_value": 1.0},
            {"metric_name": "phi", "setting": "nb201_e0", "metric_value": None},
            {"metric_name": "phi", "setting": "nb201_e5", "metric_value": 2.0},
            {"metric_name": "max", "setting": "rb40_mature", "metric_value": None},
        ]
    ).to_csv(path, index=False)
    coverage = bmi.coverage_from_rows(path)
    assert coverage["phi"] == 2  # the null row is not a setting with data
    # "max" has a row but no finite value, so it is absent from the map entirely;
    # build() reads that absence as 0 settings of usable data.
    assert coverage.get("max", 0) == 0


def test_missing_metric_in_release_table_is_a_problem() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78)]),
        registry(),
        eligibility(),
        None,
    )
    assert any("注册表有而发布表没有" in item for item in payload["problems"])


def test_metric_in_table_but_not_registry_is_a_problem() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78), row("max", 2, 0.52), row("ghost", 3, 0.4)]),
        registry(),
        eligibility(),
        None,
    )
    assert any("发布表有而注册表没有" in item for item in payload["problems"])


def test_missing_eligibility_record_is_a_problem() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78), row("max", 2, 0.52)]),
        registry(),
        {"phi": eligibility()["phi"]},
        None,
    )
    assert any("注册表有而资格表没有" in item for item in payload["problems"])


def test_declared_error_but_ranked_is_a_warning_not_an_error() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78), row("max", 2, 0.52)]),
        registry(),
        eligibility(
            phi={
                "eligibility": "implementation_error",
                "supported_settings": list(bmi.SETTINGS),
                "reason": "underflow",
            }
        ),
        None,
    )
    assert payload["problems"] == []
    assert [item["metric"] for item in payload["warnings"]] == ["phi"]


def test_warnings_are_ordered_by_rank() -> None:
    payload = bmi.build(
        table([row("phi", 4, 0.6), row("max", 1, 0.78)]),
        registry(),
        eligibility(
            phi={"eligibility": "implementation_error", "supported_settings": list(bmi.SETTINGS), "reason": "x"},
            max={"eligibility": "implementation_error", "supported_settings": list(bmi.SETTINGS), "reason": "y"},
        ),
        None,
    )
    assert [item["rank"] for item in payload["warnings"]] == [1, 4]


def test_excluded_metric_keeps_its_reason() -> None:
    frame = table([row("phi", 1, 0.78), row("max", None, 0.1, status="excluded")])
    frame.loc[1, "exclusion_reason"] = "not defined for fixed networks"
    payload = bmi.build(frame, registry(), eligibility(), None)
    excluded = [item for item in payload["rows"] if item["metric"] == "max"][0]
    assert excluded["rank"] is None
    assert excluded["exclusion_reason"] == "not defined for fixed networks"


def test_render_marks_missing_coverage_instead_of_printing_zero() -> None:
    payload = bmi.build(
        table([row("phi", 1, 0.78), row("max", 2, 0.52)]),
        registry(),
        eligibility(),
        None,
    )
    payload["table"] = {"path": "x.csv", "sha256": "0" * 64}
    payload["rows_source"] = None
    payload["schema_version"] = 1
    payload["eligibility"] = "configs/metric_eligibility.json"
    text = bmi.render_markdown(payload)
    assert "**未提供原始行，留空**（不是 0）" in text
    assert "| 1 | `phi`" in text
