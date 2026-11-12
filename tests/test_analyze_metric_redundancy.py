"""Tests for ``scripts/analyze_metric_redundancy.py``.

The analysis is descriptive, but three of its steps can silently produce wrong
numbers: orienting metrics by ``direction``, averaging correlation matrices that
contain missing pairs, and telling "no data" apart from "a low correlation".
Each test below pins one of those down with a hand-checkable case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_metric_redundancy as amr  # noqa: E402


def write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_rows(
    *,
    metrics: dict[str, dict[str, list[float]]],
    directions: dict[str, str] | None = None,
) -> list[dict]:
    """Build long-form rows from {metric: {setting: values}}."""
    directions = directions or {}
    out: list[dict] = []
    for metric, per_setting in metrics.items():
        for setting, values in per_setting.items():
            for index, value in enumerate(values):
                out.append(
                    {
                        "setting": setting,
                        "metric_name": metric,
                        "model_or_arch_id": f"m{index}",
                        "status": "ok",
                        "metric_value": value,
                        "direction": directions.get(metric, "higher"),
                    }
                )
    return out


def test_lower_direction_is_flipped(tmp_path: Path) -> None:
    rows = make_rows(
        metrics={"a": {"rb40_mature": [1.0, 2.0, 3.0]}},
        directions={"a": "lower"},
    )
    frame = amr.load_rows(write_rows(tmp_path, rows))
    assert frame["oriented"].tolist() == [-1.0, -2.0, -3.0]


def test_unknown_direction_is_rejected(tmp_path: Path) -> None:
    rows = make_rows(
        metrics={"a": {"rb40_mature": [1.0, 2.0, 3.0]}},
        directions={"a": "sideways"},
    )
    with pytest.raises(SystemExit):
        amr.load_rows(write_rows(tmp_path, rows))


def test_coverage_counts_settings_with_finite_values(tmp_path: Path) -> None:
    rows = make_rows(
        metrics={
            "full": {"rb40_mature": [1.0] * 9, "nb201_e0": [1.0] * 9},
            "sparse": {"rb40_mature": [1.0] * 9},
        }
    )
    frame = amr.load_rows(write_rows(tmp_path, rows))
    coverage = amr.coverage_by_metric(frame)
    assert coverage["full"] == 2
    assert coverage["sparse"] == 1


def test_failed_rows_are_not_counted_as_data(tmp_path: Path) -> None:
    rows = make_rows(metrics={"a": {"rb40_mature": [1.0, 2.0]}})
    rows.append(
        {
            "setting": "nb201_e0",
            "metric_name": "a",
            "model_or_arch_id": "m0",
            "status": "failed",
            "metric_value": "",
            "direction": "higher",
        }
    )
    frame = amr.load_rows(write_rows(tmp_path, rows))
    assert amr.coverage_by_metric(frame)["a"] == 1


def test_identical_metrics_share_a_cluster() -> None:
    mean = np.array(
        [
            [1.0, 1.0, 0.05],
            [1.0, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ]
    )
    groups = amr.cluster(mean, ["a", "b", "c"], height=0.5)
    by_metric = {name: group["cluster"] for group in groups for name in group["members"]}
    assert by_metric["a"] == by_metric["b"]
    assert by_metric["c"] != by_metric["a"]
    assert len(groups) == 2


def test_missing_pairs_average_over_available_settings() -> None:
    """A pair present in one setting keeps that value; it is not diluted by NaN."""
    matrices = [
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, 0.5], [0.5, 1.0]]),
    ]
    mean = amr.mean_matrix(matrices)
    assert np.isfinite(mean).all()
    assert mean[0, 1] == pytest.approx(0.5)


def test_pair_missing_everywhere_averages_to_zero() -> None:
    """No shared setting at all must become 0, not NaN: NaN would break linkage()."""
    nan = np.array([[1.0, np.nan], [np.nan, 1.0]])
    mean = amr.mean_matrix([nan, nan])
    assert np.isfinite(mean).all()
    assert mean[0, 1] == pytest.approx(0.0)


def test_mean_matrix_is_symmetric_with_unit_diagonal() -> None:
    matrices = [np.array([[1.0, 0.3], [0.2, 1.0]])]
    mean = amr.mean_matrix(matrices)
    assert mean[0, 1] == pytest.approx(mean[1, 0])
    assert mean[0, 0] == pytest.approx(1.0)
    assert mean[1, 1] == pytest.approx(1.0)


def test_correlation_recovers_the_planted_order(tmp_path: Path) -> None:
    """Two metrics that are monotone in each other should correlate at 1."""
    n = 12
    increasing = list(range(n))
    rows = make_rows(
        metrics={
            "a": {"rb40_mature": [float(v) for v in increasing]},
            "b": {"rb40_mature": [float(v * 3 + 1) for v in increasing]},
        }
    )
    frame = amr.load_rows(write_rows(tmp_path, rows))
    matrices, metrics, _, _ = amr.correlation_matrices(frame)
    assert metrics == ["a", "b"]
    assert matrices[0][0, 1] == pytest.approx(1.0)
