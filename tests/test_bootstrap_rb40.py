"""Tests for tools/bootstrap_rb40.py.

The tool is an audit of an audit, so the tests aim at the two ways it could
lie: silently dropping a metric (the TRNAS repair case) or reporting an
interval narrower than the data supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bootstrap_rb40 as bb  # noqa: E402

COLUMNS = [
    "run_id",
    "benchmark",
    "setting",
    "model_or_arch_id",
    "checkpoint_epoch",
    "metric_name",
    "direction",
    "metric_value",
    "target_value",
    "status",
    "error",
]


def write_rows(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def make_rows(targets, metrics, setting="rb40_mature", status="ok"):
    """Build one row per (model, metric); ``metrics`` maps name -> (values, direction)."""
    rows = []
    for name, (values, direction) in metrics.items():
        for model, value, target in zip(range(len(targets)), values, targets):
            rows.append(
                {
                    "run_id": "t",
                    "benchmark": "rb40",
                    "setting": setting,
                    "model_or_arch_id": f"m{model:02d}",
                    "checkpoint_epoch": None,
                    "metric_name": name,
                    "direction": direction,
                    "metric_value": value,
                    "target_value": target,
                    "status": status,
                    "error": None,
                }
            )
    return rows


def test_perfect_and_reverse_metrics_get_the_expected_rho(tmp_path):
    targets = np.linspace(0.0, 1.0, 20)
    metrics = {
        "perfect": (targets.copy(), "higher"),
        "reversed": (-targets, "higher"),
        "lower_is_better": (-targets, "lower"),
    }
    rows = write_rows(tmp_path / "rows.csv", make_rows(targets, metrics))
    matrix, target, direction, _ = bb.load_rb40(rows)
    result = bb.bootstrap(bb.orient(matrix, direction), target, resamples=30, seed=1)

    assert result["point_rho"]["perfect"] == pytest.approx(1.0)
    assert result["point_rho"]["reversed"] == pytest.approx(-1.0)
    # 'lower' metrics are negated before ranking, so they behave like 'perfect'.
    assert result["point_rho"]["lower_is_better"] == pytest.approx(1.0)
    assert result["point_rank"]["reversed"] == max(result["point_rank"].values())


def test_noise_interval_covers_zero(tmp_path):
    rng = np.random.default_rng(0)
    targets = rng.normal(size=40)
    metrics = {"noise": (rng.normal(size=40), "higher")}
    rows = write_rows(tmp_path / "rows.csv", make_rows(targets, metrics))
    matrix, target, direction, _ = bb.load_rb40(rows)
    result = bb.bootstrap(bb.orient(matrix, direction), target, resamples=200, seed=7)

    low, high = result["rho_low"]["noise"], result["rho_high"]["noise"]
    assert low < 0 < high, (low, high)
    assert bb.verdict(low, high).startswith("inconclusive")


def test_constant_metric_is_dropped_rather_than_ranked(tmp_path):
    targets = np.linspace(0.0, 1.0, 10)
    metrics = {
        "informative": (targets.copy(), "higher"),
        "constant": (np.ones(10), "higher"),
    }
    rows = write_rows(tmp_path / "rows.csv", make_rows(targets, metrics))
    matrix, target, direction, _ = bb.load_rb40(rows)
    result = bb.bootstrap(bb.orient(matrix, direction), target, resamples=20, seed=3)

    assert "constant" in result["dropped_metrics"]
    assert "constant" not in result["metrics"]


def test_repair_rows_restore_a_failed_metric(tmp_path):
    """The TRNAS underflow case: without the patch the metric would vanish."""
    targets = np.linspace(0.0, 1.0, 12)
    informative = (targets.copy(), "higher")
    patched = (targets * 2.0, "higher")

    rows = make_rows(targets, {"informative": informative, "patched": patched})
    # Mark the 'patched' rows as failed with no value, as the real export does.
    for row in rows:
        if row["metric_name"] == "patched":
            row["status"] = "failed"
            row["metric_value"] = float("nan")
    rows_path = write_rows(tmp_path / "rows.csv", rows)

    patch_rows = [r for r in make_rows(targets, {"patched": patched}) if r["metric_name"] == "patched"]
    patch_path = write_rows(tmp_path / "repair_rows.csv", patch_rows)

    without = bb.load_rb40(rows_path)
    assert "patched" not in without[0].columns

    matrix, target, direction, repaired = bb.load_rb40(rows_path, patch_path)
    assert "patched" in matrix.columns
    assert repaired == ["patched"]
    assert matrix["patched"].notna().all()


def test_same_seed_gives_the_same_interval(tmp_path):
    rng = np.random.default_rng(11)
    targets = rng.normal(size=30)
    metrics = {"m": (rng.normal(size=30), "higher")}
    rows = write_rows(tmp_path / "rows.csv", make_rows(targets, metrics))
    matrix, target, direction, _ = bb.load_rb40(rows)
    oriented = bb.orient(matrix, direction)

    first = bb.bootstrap(oriented, target, resamples=64, seed=99)
    second = bb.bootstrap(oriented, target, resamples=64, seed=99)
    assert first["rho_low"]["m"] == second["rho_low"]["m"]
    assert first["rank_high"]["m"] == second["rank_high"]["m"]


def test_model_without_target_is_refused(tmp_path):
    targets = np.linspace(0.0, 1.0, 6)
    metrics = {"m": (targets.copy(), "higher")}
    rows = make_rows(targets, metrics)
    rows[0]["target_value"] = float("nan")
    # Leave one model with every target NaN so the group has nothing to take.
    for row in rows:
        if row["model_or_arch_id"] == rows[0]["model_or_arch_id"]:
            row["target_value"] = float("nan")
    rows_path = write_rows(tmp_path / "rows.csv", rows)

    with pytest.raises(ValueError, match="no target_value"):
        bb.load_rb40(rows_path)


def test_verdict_names_the_direction_the_interval_supports():
    assert bb.verdict(0.1, 0.6).startswith("positive")
    assert bb.verdict(-0.6, -0.1).startswith("negative")
    assert bb.verdict(-0.2, 0.3).startswith("inconclusive")


def test_only_rb40_rows_are_used(tmp_path):
    targets = np.linspace(0.0, 1.0, 10)
    metrics = {"m": (targets.copy(), "higher")}
    rows = make_rows(targets, metrics, setting="rb40_mature")
    # A different setting must not leak into the RB40 matrix.
    rows += make_rows(targets, {"m": (targets[::-1], "higher")}, setting="nb201_e0")
    rows_path = write_rows(tmp_path / "rows.csv", rows)

    matrix, _, _, _ = bb.load_rb40(rows_path)
    assert matrix.shape[0] == 10
