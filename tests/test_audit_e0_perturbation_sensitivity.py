"""Tests for the perturbation-sensitivity check.

The point of this script is to tell whether the early-signal ranking is a property
of the architecture or of one epsilon.  These tests pin the two things that would
quietly invalidate it: an index misalignment between the metric table and the
target (everything becomes NaN) and a control set that differs between variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_e0_perturbation_sensitivity as sensitivity  # noqa: E402


def _table(n=30, seed=0, effect=0.0, extra=None):
    rng = np.random.RandomState(seed)
    target = rng.rand(n)
    frame = pd.DataFrame(
        {
            "arch_id": [str(i) for i in range(n)],
            "clean": rng.rand(n),
            "params": 1000 + rng.rand(n) * 100,
            "target": target,
            "signal": target * (1 + effect) + rng.rand(n) * (1 - effect),
            "noise": rng.rand(n),
        }
    )
    if extra:
        for key, value in extra.items():
            frame[key] = value
    return frame


def test_per_metric_keeps_signal_and_drops_noise():
    table = _table(effect=0.9)
    scores = sensitivity.per_metric(table, table["target"])
    assert scores["signal"]["controlled"] > 0.5
    assert abs(scores["noise"]["controlled"]) < 0.5


def test_per_metric_aligns_by_position_not_by_index():
    """A target indexed 0..n-1 against a table with the same order must not be NaN."""
    table = _table()
    shifted = table["target"].copy()
    shifted.index = [f"arch{i}" for i in range(len(table))]
    scores = sensitivity.per_metric(table, shifted)
    assert scores["signal"]["controlled"] is not None


def test_partial_rho_accepts_no_controls():
    table = _table()
    value = sensitivity.partial_rho(table["signal"], table["target"], [])
    assert value is not None
    assert value > 0.5


def test_partial_rho_refuses_a_degenerate_control():
    table = _table()
    constant = pd.Series(np.ones(len(table)))
    assert sensitivity.partial_rho(table["signal"], table["target"], [constant]) is None


def test_compare_variants_reports_the_ranking_shift():
    baseline = {
        "a": {"raw": 0.5, "controlled": 0.6},
        "b": {"raw": 0.4, "controlled": 0.2},
        "c": {"raw": 0.1, "controlled": -0.1},
        "d": {"raw": 0.3, "controlled": 0.4},
        "e": {"raw": 0.2, "controlled": 0.1},
    }
    flipped = {
        "a": {"raw": 0.5, "controlled": -0.5},
        "b": {"raw": 0.4, "controlled": 0.4},
        "c": {"raw": 0.1, "controlled": 0.2},
        "d": {"raw": 0.3, "controlled": -0.2},
        "e": {"raw": 0.2, "controlled": 0.6},
    }
    comparison = sensitivity.compare_variants(baseline, flipped)
    assert comparison["metrics"] == 5
    assert comparison["rho_between_variants"] is not None
    assert comparison["rho_between_variants"] < 0.5
    assert comparison["largest_moves"][0]["metric"] == "a"


def test_compare_variants_counts_the_top_five_overlap():
    baseline = {"m%d" % i: {"raw": None, "controlled": 0.5 - i / 100} for i in range(10)}
    same = {"m%d" % i: {"raw": None, "controlled": 0.5 - i / 100} for i in range(10)}
    comparison = sensitivity.compare_variants(baseline, same)
    assert comparison["top5_overlap"] == 5
    assert comparison["top5_baseline"][0] == "m0"


def test_compare_variants_handles_missing_values():
    baseline = {"a": {"raw": None, "controlled": None}}
    variant = {"a": {"raw": None, "controlled": None}}
    comparison = sensitivity.compare_variants(baseline, variant)
    assert comparison["rho_between_variants"] is None
    assert comparison["largest_moves"][0]["delta"] is None


def test_analyse_reports_one_block_per_variant():
    table = _table(effect=0.8)
    report = sensitivity.analyse(
        table,
        {"weak": table.copy(), "pgd": table.copy()},
        table["target"],
    )
    assert report["n_architectures"] == len(table)
    assert set(report["variants"]) == {"weak", "pgd"}
    assert report["variants"]["weak"]["n"] == len(table)


def test_load_architectures_rejects_a_table_without_a_target(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({"arch_id": ["0"], "arch_string": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target"):
        sensitivity.load_architectures(path, None)


def test_load_architectures_limits_after_sorting(tmp_path):
    path = tmp_path / "wide.csv"
    frame = pd.DataFrame(
        {
            "arch_id": ["2", "0", "1"],
            "arch_string": ["x"] * 3,
            "target": [0.3, 0.1, 0.2],
            "clean": [0.5] * 3,
        }
    )
    frame.to_csv(path, index=False)
    loaded = sensitivity.load_architectures(path, 2)
    assert list(loaded["arch_id"]) == ["0", "1"]


def test_merge_safely_returns_the_existing_table_when_nothing_is_new(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"arch_id": ["0", "1"], "x": [1.0, 2.0]}).to_csv(path, index=False)
    merged = sensitivity._merge_safely([], path)
    assert list(merged["arch_id"]) == ["0", "1"]


def test_merge_safely_handles_no_rows_and_no_file(tmp_path):
    merged = sensitivity._merge_safely([], tmp_path / "missing.csv")
    assert list(merged.columns) == ["arch_id"]


def test_render_markdown_states_the_config_and_the_limit():
    report = {
        "n_architectures": 3,
        "baseline": {"a": {"raw": 0.5, "controlled": 0.4}},
        "variants": {
            "weak": {
                "n": 3,
                "config": {"attack": "fgsm", "epsilon": 4 / 255, "steps": 1},
                "scores": {"a": {"raw": 0.4, "controlled": 0.3}},
                "comparison": {
                    "metrics": 1,
                    "rho_between_variants": 0.9,
                    "best_baseline": "a",
                    "best_variant": "a",
                    "largest_moves": [
                        {
                            "metric": "a",
                            "baseline": 0.4,
                            "variant": 0.3,
                            "delta": -0.1,
                        }
                    ],
                },
            }
        },
    }
    text = sensitivity.render_markdown(report, Path("w.csv"), "sha")
    assert "FGSM 4/255" in text
    assert "扰动强度" in text
    assert "a" in text
