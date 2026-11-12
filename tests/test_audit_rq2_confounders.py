"""Tests for the RQ2 second-confounder audit.

The numbers need the real 400-architecture table; what is testable here is the
plumbing: partial correlations with zero, one and two controls, the verdict
buckets (including the one that keeps a metric from being called "explained" when
it was never strong), and the parameter count derived from an architecture string.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_rq2_confounders as audit  # noqa: E402

ARCH_3X3 = "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|"
ARCH_1X1 = "|nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|"


def test_parameter_counts_grows_with_convolution_size():
    counts = audit.parameter_counts(pd.Series({"a": ARCH_1X1, "b": ARCH_3X3}))
    assert counts["a"] < counts["b"]


def test_parameter_counts_skips_unparsable_strings():
    counts = audit.parameter_counts(pd.Series({"a": "not an arch"}))
    assert counts == {}


def test_partial_rho_without_controls_matches_spearman():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = 0.8 * x + rng.normal(size=80)
    from scipy.stats import spearmanr

    expected = spearmanr(x, y)[0]
    assert audit.partial_rho(x, y, []) == pytest.approx(expected, abs=1e-9)


def test_partial_rho_removes_a_shared_driver():
    rng = np.random.default_rng(1)
    control = rng.normal(size=120)
    x = control + 0.6 * rng.normal(size=120)
    y = control + 0.6 * rng.normal(size=120)
    assert audit.partial_rho(x, y, []) > 0.4
    assert audit.partial_rho(x, y, [control]) < 0.2


def test_partial_rho_accepts_two_controls():
    rng = np.random.default_rng(2)
    c1 = rng.normal(size=150)
    c2 = rng.normal(size=150)
    x = c1 + c2 + 0.5 * rng.normal(size=150)
    y = c1 + c2 + 0.5 * rng.normal(size=150)
    assert audit.partial_rho(x, y, [c1, c2]) < 0.25
    assert audit.partial_rho(x, y, []) > 0.5


def test_partial_rho_keeps_an_independent_signal():
    rng = np.random.default_rng(3)
    control = rng.normal(size=150)
    x = rng.normal(size=150)
    y = x + control + 0.4 * rng.normal(size=150)
    assert audit.partial_rho(x, y, [control]) > 0.3


def test_partial_rho_refuses_a_tiny_sample():
    assert audit.partial_rho(np.arange(5.0), np.arange(5.0), []) is None


def test_bootstrap_ci_is_deterministic_for_one_seed():
    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    y = x + rng.normal(size=60)
    first = audit.bootstrap_ci(x, y, [], resamples=200, seed=7)
    second = audit.bootstrap_ci(x, y, [], resamples=200, seed=7)
    assert first[:2] == second[:2]
    assert first[0] < first[1]


def test_classify_separates_weak_from_explained():
    # Never strong to begin with: capacity did not explain anything away.
    assert audit.classify(0.10, 0.02) == "weak_before_capacity"
    # Strong without capacity, gone with it.
    assert audit.classify(0.55, 0.10) == "explained_by_capacity"
    # Strong, and drops by more than SHRINK_LIMIT but stays above the floor.
    assert audit.classify(0.60, 0.42) == "shrinks"
    # Strong and stable.
    assert audit.classify(0.42, 0.40) == "survives"
    # Sign flip is its own bucket.
    assert audit.classify(0.40, -0.35) == "flips"
    assert audit.classify(None, 0.4) == "uncovered"


def test_load_wide_rejects_a_table_without_arch_string(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({"arch_id": [0], "target": [0.2], "clean": [0.8]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="arch_string"):
        audit.load_wide(path)


def test_analyse_end_to_end_on_a_small_table():
    rng = np.random.default_rng(5)
    n = 30
    params = rng.choice([1000, 5000, 20000], size=n)
    clean = params / 30000 + rng.normal(scale=0.05, size=n)
    target = clean + 0.05 * rng.normal(size=n)
    signal = target + rng.normal(scale=0.05, size=n)

    frame = pd.DataFrame(
        {
            "arch_id": [str(i) for i in range(n)],
            "arch_string": [ARCH_1X1 if i % 2 else ARCH_3X3 for i in range(n)],
            "target": target,
            "clean": clean,
            "trnas_consistency_phi": signal,
        }
    )
    report = audit.analyse(frame, resamples=100)
    assert report["n_architectures"] == n
    names = [row["metric"] for row in report["metrics"]]
    assert "trnas_consistency_phi" in names
    row = next(r for r in report["metrics"] if r["metric"] == "trnas_consistency_phi")
    assert row["rho_target"] > 0.7
    assert row["ci_low"] < row["rho_clean_params"] < row["ci_high"]


def test_render_markdown_states_the_buckets():
    report = {
        "n_architectures": 10,
        "parameters_range": [1000, 2000],
        "rho_target_clean": 0.9,
        "rho_target_params": 0.5,
        "rho_clean_params": 0.5,
        "metrics": [
            {
                "metric": "feature_delta",
                "n": 10,
                "direction": "lower",
                "rho_target": 0.8,
                "rho_clean": 0.5,
                "rho_clean_params": 0.4,
                "ci_low": 0.3,
                "ci_high": 0.5,
                "delta_from_clean": -0.1,
                "verdict": "survives",
            }
        ],
    }
    text = audit.render_markdown(report, Path("wide.csv"), "abc")
    assert "参数量" in text
    assert "survives" in text
    assert "不是因果" in text
