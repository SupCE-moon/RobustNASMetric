"""Tests for the weight-seed audit.

The real numbers need 100 architectures and three GPU passes; what is testable is
the plumbing: value stability between seeds, signal stability across seeds, the
three verdict buckets, and the sign flip on ``lower`` metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_e0_weight_seed as audit  # noqa: E402


def _tables(n: int = 40, seed: int = 0):
    """Two seeds of one metric: `stable` is identical, `noisy` is not."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    ids = [str(index) for index in range(n)]
    first = pd.DataFrame(
        {
            "arch_id": ids,
            "stable": base,
            "noisy": rng.normal(size=n),
            "clean": rng.uniform(0.1, 0.5, size=n),
            "params": rng.uniform(1e5, 2e6, size=n),
        }
    )
    second = first.copy()
    second["stable"] = base * 2.0 + 1.0  # monotone: same ranking
    second["noisy"] = rng.normal(size=n)
    return {0: first, 1: second}


def test_value_stability_sees_a_monotone_metric_as_stable():
    values = audit.value_stability(_tables())
    assert values["stable"]["mean_rho"] == pytest.approx(1.0)
    assert values["stable"]["pairs"] == 1


def test_value_stability_flags_an_independent_metric():
    values = audit.value_stability(_tables())
    assert abs(values["noisy"]["mean_rho"]) < 0.8


def test_value_stability_needs_two_seeds():
    one = {0: _tables()[0]}
    assert audit.value_stability(one) == {}


def test_value_stability_skips_short_overlap():
    tables = _tables(n=10)
    assert audit.value_stability(tables) == {}


def test_signal_scores_flips_lower_metrics():
    # `feature_delta` is registered as "lower": smaller is better, so a metric
    # that falls as the target rises must come out positive after the flip.
    frame = _tables()[0].rename(columns={"stable": "feature_delta"})
    target = -frame["feature_delta"]
    scores = audit.signal_scores(frame, target, [])
    assert scores["feature_delta"] == pytest.approx(1.0, abs=1e-6)


def test_signal_scores_returns_none_without_enough_rows():
    frame = _tables(n=10)[0]
    target = pd.Series(frame["stable"])
    scores = audit.signal_scores(frame, target, [])
    assert scores["stable"] is None


def test_signal_stability_detects_a_sign_flip():
    per_seed = {0: {"m": 0.4}, 1: {"m": -0.3}}
    block = audit.signal_stability(per_seed)["m"]
    assert block["sign_flips"] is True
    assert audit._signal_verdict(block) == "signal_flips"


def test_signal_stability_accepts_a_close_pair():
    per_seed = {0: {"m": 0.40}, 1: {"m": 0.45}}
    block = audit.signal_stability(per_seed)["m"]
    assert block["sign_flips"] is False
    assert audit._signal_verdict(block) == "stable_signal"
    assert block["spread"] == pytest.approx(0.05)


def test_signal_stability_needs_two_seeds():
    assert audit.signal_stability({0: {"m": 0.4}}) == {}
    assert audit._signal_verdict({}) == "uncovered"


def test_analyse_counts_both_verdicts():
    tables = _tables()
    source = tables[0].copy()
    source["target"] = source["stable"]
    report = audit.analyse(tables, source, ["clean", "params"])
    assert report["counts"]["metrics"] == 2  # stable and noisy; clean/params are controls
    assert report["counts"]["stable_value"] >= 1
    names = [row["metric"] for row in report["metrics"]]
    assert names[0] == "stable"


def test_load_architectures_rejects_a_table_without_the_required_columns(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({"arch_id": ["0"], "arch_string": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        audit.load_architectures(path, None)


def test_parameter_count_is_none_for_a_bad_string():
    assert audit.parameter_count("not-an-arch") is None


def test_render_markdown_states_the_two_questions():
    tables = _tables()
    source = tables[0].copy()
    source["target"] = source["stable"]
    report = audit.analyse(tables, source, ["clean", "params"])
    text = audit.render_markdown(report, Path("wide.csv"), "abc")
    assert "值稳定" in text
    assert "信号稳定" in text
    assert "不能说明什么" in text
