"""Tests for tools/audit_metric_invariants.py.

The audit's value is that it can fail: a distance that does not vanish at zero
perturbation, or one that shrinks as the perturbation grows, must be reported
rather than averaged away.  These tests build that failure path with synthetic
series first, then pin the real bundle's current behaviour.

The real-network sweep is slow (~7 s) and therefore runs once per module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_metric_invariants as ami  # noqa: E402


SYNTHETIC_EPS = [0.0, 0.001, 0.002, 0.004]


def test_is_distance_recognises_delta_names():
    assert ami.is_distance("feature_delta")
    assert ami.is_distance("feature_measure_spectrum_js_delta")
    assert ami.is_distance("logit_kl") is True


def test_is_distance_excludes_input_properties_and_metadata():
    assert not ami.is_distance("input_grad_l2")
    assert not ami.is_distance("jacobian_spectral_norm")
    assert not ami.is_distance("natural_loss")
    assert not ami.is_distance("num_layers")


def test_spearman_detects_monotone_and_reversed_series():
    assert ami.spearman(SYNTHETIC_EPS, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert ami.spearman(SYNTHETIC_EPS, [0.0, -1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_spearman_returns_none_for_constant_series():
    assert ami.spearman(SYNTHETIC_EPS, [5.0, 5.0, 5.0, 5.0]) is None


def test_spearman_handles_ties_without_dividing_by_zero():
    value = ami.spearman(SYNTHETIC_EPS, [0.0, 0.0, 1.0, 1.0])
    assert value is not None and value > 0


def test_classify_flags_a_distance_that_is_nonzero_at_zero():
    row = ami.classify("feature_delta", [0.3, 0.4, 0.5, 0.6], SYNTHETIC_EPS)
    assert row["status"] == "nonzero_at_zero"


def test_classify_flags_a_decreasing_distance():
    row = ami.classify("perturbation_capacity_signed_delta", [0.0, -1.0, -2.0, -3.0], SYNTHETIC_EPS)
    assert row["status"] == "decreasing"


def test_classify_flags_an_unordered_series():
    row = ami.classify("feature_measure_spectrum_js_delta", [0.0, 0.5, 0.2, 0.4], SYNTHETIC_EPS)
    assert row["status"] == "unordered"


def test_classify_accepts_a_recorded_expectation():
    row = ami.classify("feature_box_dimension_delta", [0.0, 0.09, 0.08, 0.09], SYNTHETIC_EPS)
    assert row["status"] == "ok_expected"
    assert "quantised" in row["detail"]


def test_classify_keeps_the_known_nan_placeholder_separate():
    row = ami.classify("perturbation_capacity_importance_delta", [float("nan")] * 4, SYNTHETIC_EPS)
    assert row["status"] == "known_nan"


def test_classify_reports_metadata_without_judging_finiteness():
    row = ami.classify("sampled_layer_indices", ["0,1", "0,1", "0,1", "0,1"], SYNTHETIC_EPS)
    assert row["status"] == "metadata"
    assert row["finite"] is True


def test_fgsm_sign_is_unit_and_not_all_zero():
    """Guards the bug this module shipped with: x.detach().sign() is all zeros."""
    model, x, y = ami.build_inputs(seed=0, batch=2)
    sign = ami.fgsm_sign(model, x, y)
    assert sign.shape == x.shape
    assert set(sign.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert float(sign.abs().sum()) > 0


@pytest.fixture(scope="module")
def bundle_report():
    # batch=6 matches the CLI default.  The signed-capacity verdict is
    # batch-size sensitive: with 4 images it comes out "unordered" instead of
    # "decreasing", which is itself worth knowing about that metric.
    model, x, y = ami.build_inputs(seed=0, batch=6)
    return ami.analyse(model, x, y)


def test_sweep_starts_at_zero_epsilon_and_increases(bundle_report):
    eps = bundle_report["epsilons"]
    assert eps[0] == 0.0
    assert eps == sorted(eps)
    assert eps[-1] == pytest.approx(ami.SETTING)


def test_real_bundle_has_no_nonzero_at_zero(bundle_report):
    assert bundle_report["counts"].get("nonzero_at_zero", 0) == 0


def test_real_bundle_has_no_unexpected_non_finite(bundle_report):
    assert bundle_report["counts"].get("non_finite", 0) == 0


def test_real_bundle_flags_signed_capacity_delta_as_decreasing(bundle_report):
    """Current fact, not a requirement.

    `perturbation_capacity_signed_delta` is a signed relative change and is
    declared `lower`.  It shrinks as epsilon grows, i.e. its sign convention is
    the opposite of the other 55 deltas.  If someone changes its definition or
    its declared direction, this test must be updated deliberately.
    """
    rows = {r["name"]: r for r in bundle_report["rows"]}
    assert rows["perturbation_capacity_signed_delta"]["status"] == "decreasing"


def test_real_bundle_treats_importance_delta_as_a_known_placeholder(bundle_report):
    rows = {r["name"]: r for r in bundle_report["rows"]}
    assert rows["perturbation_capacity_importance_delta"]["status"] == "known_nan"


def test_real_bundle_passes_feature_delta(bundle_report):
    rows = {r["name"]: r for r in bundle_report["rows"]}
    assert rows["feature_delta"]["status"] == "ok"


def test_render_markdown_lists_flagged_metrics_and_reading_notes(bundle_report):
    text = ami.render_markdown(bundle_report)
    assert "需要人看的" in text
    assert "perturbation_capacity_signed_delta" in text
    assert "怎么读" in text
    assert "不检查" in text
