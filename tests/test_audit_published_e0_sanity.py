"""Tests for the published-e0 sanity check.

The load-bearing judgement is the spike rule: `metrics.py` clamps relative feature
metrics at 1e3, so a published value above that cannot have come from this code.
These tests pin that rule, the four verdict buckets, and the refusal to divide by a
zero median.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_published_e0_sanity as sanity  # noqa: E402


def _published(tmp_path, rows):
    path = tmp_path / "published.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(arch, metric, value, setting="nb201_e0"):
    return {
        "setting": setting,
        "model_or_arch_id": str(arch),
        "metric_name": metric,
        "metric_value": value,
        "status": "ok",
    }


def test_load_published_rejects_a_table_without_metric_value(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"setting": ["nb201_e0"], "model_or_arch_id": ["0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="metric_value"):
        sanity.load_published(path)


def test_load_published_keeps_numeric_ids_as_text(tmp_path):
    rows = [_row(i, "feature_delta", 0.1 + i / 100) for i in range(12)]
    frame = sanity.load_published(_published(tmp_path, rows))
    assert frame.index[0] == "0"
    assert frame.index.dtype == object


def test_compare_column_counts_values_above_the_ceiling():
    published = pd.Series([0.1, 0.2, 5e8], index=["0", "1", "2"])
    current = pd.Series([0.1, 0.2, 0.3], index=["0", "1", "2"])
    entry = sanity.compare_column(published, current)
    assert entry["spikes"] == 1
    assert entry["published_max_abs"] == pytest.approx(5e8)


def test_compare_column_reports_none_for_a_constant_column():
    published = pd.Series([0.5] * 12, index=[str(i) for i in range(12)])
    current = pd.Series(np.linspace(0, 1, 12), index=[str(i) for i in range(12)])
    entry = sanity.compare_column(published, current)
    assert entry["rho"] is None


def test_compare_column_refuses_to_divide_by_a_zero_median():
    published = pd.Series([1.0, 2.0], index=["0", "1"])
    current = pd.Series([0.0, 0.0], index=["0", "1"])
    entry = sanity.compare_column(published, current)
    assert entry["median_ratio"] is None


def test_classify_puts_a_spike_first_even_when_correlated():
    assert sanity.classify({"rho": 0.95, "spikes": 2}) == "spiked"
    assert sanity.classify({"rho": 0.95, "spikes": 0}) == "clean"
    assert sanity.classify({"rho": 0.4, "spikes": 0}) == "divergent"
    assert sanity.classify({"rho": None, "spikes": 0}) == "uncomparable"


def test_analyse_separates_clean_from_spiked():
    published = pd.DataFrame(
        {
            "good_metric": np.linspace(0, 1, 20),
            "spiky_metric": np.r_[np.linspace(0, 1, 19), [1e8]],
            "divergent_metric": np.linspace(0, 1, 20),
        },
        index=[str(i) for i in range(20)],
    )
    current = pd.DataFrame(
        {
            "good_metric": np.linspace(0, 1, 20),
            "spiky_metric": np.linspace(0, 1, 20),
            "divergent_metric": -np.linspace(0, 1, 20),
        },
        index=[str(i) for i in range(20)],
    )
    report = sanity.analyse(published, current)
    verdicts = {row["metric"]: row["verdict"] for row in report["columns"]}
    assert verdicts["good_metric"] == "clean"
    assert verdicts["spiky_metric"] == "spiked"
    assert verdicts["divergent_metric"] == "divergent"
    assert report["counts"]["clean"] == 1


def test_analyse_skips_columns_missing_on_the_current_side():
    published = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}, index=["0", "1"])
    current = pd.DataFrame({"a": [1.0, 2.0]}, index=["0", "1"])
    report = sanity.analyse(published, current)
    assert [row["metric"] for row in report["columns"]] == ["a"]


def test_render_markdown_states_the_ceiling_and_the_risk():
    report = {
        "setting": "nb201_e0",
        "metrics": 1,
        "spike_ceiling": 1e3,
        "rho_ok": 0.8,
        "counts": {"spiked": 1},
        "columns": [
            {
                "metric": "bad",
                "verdict": "spiked",
                "rho": 0.3,
                "spikes": 4,
                "published_max_abs": 1e8,
                "current_max_abs": 0.3,
                "median_ratio": 2.0,
                "n": 20,
            }
        ],
    }
    text = sanity.render_markdown(report, Path("p.csv"), Path("c.csv"), "sha", "sha2")
    assert "clamp_max(1e3)" in text
    assert "主导 Spearman" in text
    assert "bad" in text
