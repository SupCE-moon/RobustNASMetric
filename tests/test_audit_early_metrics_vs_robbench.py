"""Tests for the early-metrics versus RobBench audit.

The important properties are not the numbers (those need the real data) but the
plumbing: a metric registered as "lower" must be sign-flipped before correlation,
the partial correlation must actually remove the control variable, and the verdict
must not reward a metric that only tracks clean accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_early_metrics_vs_robbench as audit  # noqa: E402


def _rows_frame(values, direction="higher", setting=audit.E0_SETTING):
    """Build a long-format frame with one metric per architecture."""
    records = []
    for arch, value in values.items():
        records.append(
            {
                "setting": setting,
                "model_or_arch_id": str(arch),
                "metric_name": "m",
                "metric_value": float(value),
                "direction": direction,
                "status": "ok",
            }
        )
    return pd.DataFrame(records)


def test_load_rows_flips_lower_direction():
    frame = _rows_frame({"0": 1.0, "1": 2.0}, direction="lower")
    loaded = audit.load_rows_from_frame(frame)
    assert loaded.loc["0", "m"] == pytest.approx(-1.0)
    assert loaded.loc["1", "m"] == pytest.approx(-2.0)


def test_load_rows_keeps_higher_direction():
    frame = _rows_frame({"0": 1.0, "1": 2.0}, direction="higher")
    loaded = audit.load_rows_from_frame(frame)
    assert loaded.loc["0", "m"] == pytest.approx(1.0)


def test_load_rows_filters_to_one_setting():
    frame = pd.concat(
        [_rows_frame({"0": 1.0}), _rows_frame({"0": 5.0}, setting="nb201_e5")],
        ignore_index=True,
    )
    loaded = audit.load_rows_from_frame(frame)
    assert list(loaded.index) == ["0"]
    assert loaded.loc["0", "m"] == pytest.approx(1.0)


def test_partial_spearman_removes_a_shared_driver():
    """If x and y are both driven by control, the partial correlation must collapse."""
    rng = np.random.default_rng(0)
    control = rng.normal(size=60)
    x = control + 0.6 * rng.normal(size=60)
    y = control + 0.6 * rng.normal(size=60)
    raw = audit._rho(x, y)
    partial = audit.partial_spearman(x, y, control)
    assert raw > 0.4
    assert abs(partial) < 0.35


def test_partial_spearman_keeps_independent_signal():
    rng = np.random.default_rng(1)
    control = rng.normal(size=60)
    shared = rng.normal(size=60)
    x = 0.4 * control + shared
    y = 0.4 * control + shared + 0.3 * rng.normal(size=60)
    assert audit.partial_spearman(x, y, control) > 0.45


def test_classify_branches():
    assert audit.classify(0.80, 0.60) == "direct"
    assert audit.classify(0.80, 0.02) == "via_clean"
    assert audit.classify(-0.60, 0.50) == "suppressed"
    assert audit.classify(0.20, 0.10) == "weak"
    assert audit.classify(None, 0.50) == "uncovered"
    assert audit.classify(0.80, None) == "uncovered"


def test_bootstrap_ci_is_deterministic_for_one_seed():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    control = rng.normal(size=40)
    first = audit.bootstrap_ci(x, y, control, resamples=200, seed=7)
    second = audit.bootstrap_ci(x, y, control, resamples=200, seed=7)
    assert first[0] == pytest.approx(second[0])
    assert first[1] == pytest.approx(second[1])


def test_permutation_p_small_for_planted_signal_and_large_for_noise():
    rng = np.random.default_rng(3)
    control = rng.normal(size=50)
    shared = rng.normal(size=50)
    x = shared + 0.2 * rng.normal(size=50)
    y = shared + 0.2 * rng.normal(size=50)
    observed = audit.partial_spearman(x, y, control)
    assert audit.permutation_p(x, y, control, observed, resamples=200) < 0.05

    noise = rng.normal(size=50)
    noise_observed = audit.partial_spearman(noise, y, control)
    assert audit.permutation_p(noise, y, control, noise_observed, resamples=200) > 0.05


def test_permutation_p_never_reports_zero():
    """With B shuffles the smallest possible p is 1/(B+1); 0.0 is false precision."""
    rng = np.random.default_rng(4)
    x = np.arange(30.0)
    y = np.arange(30.0)
    control = np.zeros(30)
    value = audit.permutation_p(x, y, control, 1.0, resamples=50)
    assert value >= 1.0 / 51.0


def test_analyse_flags_a_clean_only_metric_as_via_clean():
    """A metric that equals clean accuracy must not be sold as robustness evidence."""
    rng = np.random.default_rng(5)
    n = 40
    archs = [str(i) for i in range(n)]
    clean = rng.uniform(0.5, 0.95, size=n)
    target = clean + 0.01 * rng.normal(size=n)
    proxy = clean + 0.001 * rng.normal(size=n)  # metric tracks clean almost exactly
    metrics = pd.DataFrame({"proxy": proxy}, index=archs)
    robbench = pd.DataFrame({"target": target, "clean": clean}, index=archs)

    report = audit.analyse(metrics, robbench, resamples=100)
    row = report["metrics"][0]
    assert row["rho_target"] > 0.5
    assert row["rho_clean"] > 0.5
    assert abs(row["partial_clean"]) < audit.DIRECT_RHO
    assert row["verdict"] == "via_clean"


def test_analyse_detects_signal_beyond_clean():
    rng = np.random.default_rng(6)
    n = 40
    archs = [str(i) for i in range(n)]
    clean = rng.uniform(0.5, 0.95, size=n)
    extra = rng.normal(size=n)
    target = clean + 0.25 * extra + 0.01 * rng.normal(size=n)
    metrics = pd.DataFrame({"signal": extra}, index=archs)
    robbench = pd.DataFrame({"target": target, "clean": clean}, index=archs)

    report = audit.analyse(metrics, robbench, resamples=100)
    assert abs(report["metrics"][0]["partial_clean"]) > 0.5


def test_analyse_drops_architectures_without_a_target():
    archs = ["0", "1", "2"]
    metrics = pd.DataFrame({"m": [1.0, 2.0, 3.0]}, index=archs)
    robbench = pd.DataFrame({"target": [0.1, np.nan, 0.3], "clean": [0.5, 0.6, 0.7]}, index=archs)
    report = audit.analyse(metrics, robbench, resamples=50)
    assert report["metrics"][0]["n"] == 2


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "sample.bin"
    path.write_bytes(b"robust")
    expected = hashlib.sha256(b"robust").hexdigest()
    assert audit.sha256_file(path) == expected


def test_render_markdown_states_the_clean_accuracy_trap():
    archs = [str(i) for i in range(30)]
    rng = np.random.default_rng(7)
    clean = rng.uniform(0.5, 0.9, size=30)
    target = clean + 0.01 * rng.normal(size=30)
    metrics = pd.DataFrame({"m": clean}, index=archs)
    robbench = pd.DataFrame({"target": target, "clean": clean}, index=archs)
    report = audit.analyse(metrics, robbench, resamples=50)
    report["inputs"] = {"rows": {"path": "x", "sha256": "0"}, "robbench": {"path": "y", "sha256": "1"}}
    text = audit.render_markdown(report)
    assert "为什么必须控制 clean 准确率" in text
    assert "这份报告不能说明什么" in text
    assert "via_clean" in text
