"""Tests for the selection-value audit.

The script's whole purpose is to decide whether an indicator is worth acting on,
so the tests cover the decision rule and the direction handling. The direction
handling already produced one real bug: `clean` and `params` are not registered
metrics, so they inherited the "lower is better" fallback and the selection kept
the *least* accurate architectures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_rq2_selection_value as audit  # noqa: E402


def _frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.15, 0.45, n)
    return pd.DataFrame(
        {
            "arch_id": [str(i) for i in range(n)],
            "arch_string": ["|nor_conv_3x3~0|"] * n,
            "target": target,
            "clean": target + rng.normal(0, 0.02, n),
            "params": rng.integers(90_000, 2_700_000, n).astype(float),
            "good": target + rng.normal(0, 0.03, n),
            "feature_delta": -target + rng.normal(0, 0.03, n),
            "noise": rng.normal(0, 1, n),
        }
    )


def test_clean_and_params_are_higher_is_better():
    """Regression: both are absent from METRIC_SPECS and once defaulted to lower."""
    assert audit.higher_is_better("clean") is True
    assert audit.higher_is_better("params") is True


def test_registered_lower_metric_stays_lower():
    assert audit.higher_is_better("feature_delta") is False


def test_curve_takes_the_best_end_for_a_lower_metric():
    frame = _frame()
    rows = audit.curve(frame, "feature_delta", (10,))
    assert rows[0]["delta"] > 0.10  # feature_delta was built to run opposite the target


def test_curve_takes_the_best_end_for_a_higher_metric():
    frame = _frame()
    frame["trnas_consistency_phi"] = frame["target"] + np.random.default_rng(9).normal(0, 0.03, len(frame))
    rows = audit.curve(frame, "trnas_consistency_phi", (10,))
    assert rows[0]["delta"] > 0.10


def test_curve_on_noise_is_near_zero():
    frame = _frame()
    rows = audit.curve(frame, "noise", (10,))
    assert abs(rows[0]["delta"]) < 0.10


def test_random_baseline_centres_on_zero():
    rng = np.random.default_rng(3)
    target = np.random.default_rng(4).uniform(0.15, 0.45, 400)
    null = audit.random_baseline(target, 10, 500, rng)
    assert abs(float(null.mean())) < 0.02


def test_p_against_random_is_bounded_and_monotone():
    null = np.linspace(-0.1, 0.1, 1000)
    assert audit.p_against_random(0.5, null) == pytest.approx(1 / 1001, abs=1e-4)
    assert audit.p_against_random(-0.5, null) == pytest.approx(1.0, abs=1e-4)


def test_verdict_says_signal_but_weaker_than_params():
    rows = [
        {"metric": "early", "k_percent": 10, "delta": 0.028, "p_random": 0.0002},
        {"metric": "params", "k_percent": 10, "delta": 0.051},
        {"metric": "clean", "k_percent": 10, "delta": 0.060},
    ]
    deltas = {("early", 10): {"low": 0.018, "high": 0.036}}
    assert "不及免费基线" in audit.verdict(rows, deltas)


def test_verdict_accepts_an_early_metric_that_beats_params():
    rows = [
        {"metric": "early", "k_percent": 10, "delta": 0.080, "p_random": 0.0002},
        {"metric": "params", "k_percent": 10, "delta": 0.051},
        {"metric": "clean", "k_percent": 10, "delta": 0.060},
    ]
    deltas = {("early", 10): {"low": 0.070, "high": 0.090}}
    assert "可用于筛选" in audit.verdict(rows, deltas)


def test_verdict_rejects_an_interval_that_crosses_zero():
    rows = [
        {"metric": "early", "k_percent": 10, "delta": 0.007, "p_random": 0.21},
        {"metric": "params", "k_percent": 10, "delta": 0.051},
    ]
    deltas = {("early", 10): {"low": -0.007, "high": 0.021}}
    assert "不足以筛选" in audit.verdict(rows, deltas)


def test_verdict_never_names_clean_or_params_as_the_winner():
    rows = [
        {"metric": "params", "k_percent": 10, "delta": 0.051},
        {"metric": "clean", "k_percent": 10, "delta": 0.060},
    ]
    deltas = {("params", 10): {"low": 0.043, "high": 0.058}}
    assert "没有可判定的早期指标" in audit.verdict(rows, deltas)


def test_pick_candidates_keeps_only_stable_metrics_then_baselines():
    frame = _frame()
    scores = pd.DataFrame(
        [
            {"metric": "good", "partial_rho": 0.4, "abs_partial": 0.4},
            {"metric": "noise", "partial_rho": 0.1, "abs_partial": 0.1},
        ]
    )
    seeds = {
        "good": {"value_rho": 0.9, "signal_verdict": "stable_signal"},
        "noise": {"value_rho": 0.9, "signal_verdict": "signal_flips"},
    }
    chosen = audit.pick_candidates(frame, scores, seeds, 3)
    assert "good" in chosen
    assert "noise" not in chosen
    assert "clean" in chosen and "params" in chosen


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    frame = _frame(n=60)
    first = audit.bootstrap(frame, "good", (10,), 60)
    second = audit.bootstrap(frame, "good", (10,), 60)
    assert first[10]["low"] == second[10]["low"]


def test_load_wide_rejects_a_table_without_a_target(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({"arch_id": ["0"], "clean": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        audit.load_wide(path)


def test_load_seed_report_reads_the_two_fields(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps({"metrics": [{"metric": "a", "value": {"mean_rho": 0.9}, "signal_verdict": "stable_signal"}]}),
        encoding="utf-8",
    )
    table = audit.load_seed_report(path)
    assert table["a"]["value_rho"] == pytest.approx(0.9)
    assert table["a"]["signal_verdict"] == "stable_signal"


def test_render_markdown_states_the_limits():
    report = {
        "schema_version": 1,
        "wide": {"path": "w.csv", "sha256": "0" * 64},
        "ks": [10],
        "resamples": 10,
        "verdict": "不足以筛选",
        "candidates": ["good"],
        "rows": [
            {
                "metric": "good",
                "k_percent": 10,
                "n_selected": 10,
                "mean_target": 0.4,
                "delta": 0.01,
                "ci_low": -0.01,
                "ci_high": 0.02,
                "p_random": 0.3,
            }
        ],
    }
    text = audit.render_markdown(report)
    assert "## 怎么读" in text
    assert "## 这份报告不能说明什么" in text
    assert "6466" in text
