"""Tests for tools/audit_signed_capacity.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import audit_signed_capacity as asc


def panel_from(signed, absolute, target, setting="nb201_e0"):
    index = pd.MultiIndex.from_tuples(
        [(setting, f"m{i}") for i in range(len(signed))], names=["setting", "model_or_arch_id"]
    )
    return pd.DataFrame(
        {asc.SIGNED: signed, asc.ABSOLUTE: absolute, "target": target}, index=index
    )


def test_identical_metrics_are_read_as_the_same_metric():
    values = np.linspace(0.01, 0.5, 8)
    panel = panel_from(values, values, np.linspace(0.9, 0.2, 8))
    report = asc.analyse(panel)
    assert report["summary"]["median_rho_signed_absolute"] == pytest.approx(1.0)
    assert report["verdict"] == "consistent up to sign"


def test_unrelated_orderings_are_not_the_same_metric():
    absolute = np.linspace(0.05, 0.60, 12)
    signed = absolute.copy()
    rng = np.random.default_rng(0)
    signed = rng.permutation(signed) * 0.5  # keeps |signed| <= |absolute|
    panel = panel_from(signed, absolute, np.linspace(0.8, 0.1, 12))
    report = asc.analyse(panel)
    assert report["summary"]["median_rho_signed_absolute"] < 0.8
    assert report["verdict"] == "not the same metric"


def test_pairing_self_check_flags_a_row_where_signed_exceeds_absolute():
    absolute = np.array([0.10, 0.20, 0.30, 0.40])
    signed = np.array([0.05, 0.15, 0.35, 0.20])  # third row exceeds its absolute value
    panel = panel_from(signed, absolute, np.array([0.7, 0.6, 0.5, 0.4]))
    report = asc.analyse(panel)
    assert report["summary"]["abs_le_abs_violations"] == 1


def test_ratio_is_within_minus_one_and_one():
    absolute = np.array([0.10, 0.20, 0.30])
    signed = np.array([0.10, -0.10, 0.0])
    panel = panel_from(signed, absolute, np.array([0.7, 0.6, 0.5]))
    values = asc.ratio(panel)
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert values.loc[("nb201_e0", "m1")] == pytest.approx(-0.5)


def test_sign_flips_are_counted_and_named():
    n = 8
    absolute = np.linspace(0.1, 0.8, n)
    signed = absolute.copy()  # positively related to target
    target = np.linspace(0.9, 0.2, n)
    table = asc.per_setting(panel_from(signed, absolute, target))
    flipped = asc.per_setting(panel_from(-signed, absolute, target))
    assert table[0]["rho_signed_target"] * table[0]["rho_absolute_target"] > 0
    summary = asc.summarise(table, panel_from(signed, absolute, target))
    assert summary["n_rank_flips"] == 0
    summary_flip = asc.summarise(flipped, panel_from(-signed, absolute, target))
    assert summary_flip["n_rank_flips"] == 1


def test_verdict_three_branches():
    assert asc.verdict({"median_rho_signed_absolute": None, "n_rank_flips": 0}) == "insufficient data"
    assert asc.verdict({"median_rho_signed_absolute": 0.9, "n_rank_flips": 0}) == "consistent up to sign"
    assert asc.verdict({"median_rho_signed_absolute": 0.4, "n_rank_flips": 0}) == "not the same metric"
    # A high median is not enough when several settings flip sign.
    assert asc.verdict({"median_rho_signed_absolute": 0.9, "n_rank_flips": 2}) == "not the same metric"


def test_repair_rows_override_failed_values(tmp_path):
    rows = pd.DataFrame(
        {
            "setting": ["nb201_e0"] * 4,
            "model_or_arch_id": ["m0", "m1", "m0", "m1"],
            "metric_name": [asc.SIGNED, asc.SIGNED, asc.ABSOLUTE, asc.ABSOLUTE],
            "metric_value": [np.nan, np.nan, 0.10, 0.20],
            "target_value": [0.9, 0.8, 0.9, 0.8],
            "status": ["failed", "failed", "ok", "ok"],
        }
    )
    rows_path = tmp_path / "rows.csv"
    rows.to_csv(rows_path, index=False)
    without = asc.load_panel(rows_path)
    assert len(without) == 0  # NaN signed column drops every row

    repair = pd.DataFrame(
        {
            "setting": ["nb201_e0", "nb201_e0"],
            "model_or_arch_id": ["m0", "m1"],
            "metric_name": [asc.SIGNED, asc.SIGNED],
            "metric_value": [0.1, 0.15],
        }
    )
    repair_path = tmp_path / "repair.csv"
    repair.to_csv(repair_path, index=False)
    with_repair = asc.load_panel(rows_path, repair_path)
    assert len(with_repair) == 2
    assert with_repair[asc.SIGNED].tolist() == pytest.approx([0.1, 0.15])


def test_render_markdown_reports_the_pairing_self_check():
    values = np.linspace(0.01, 0.4, 6)
    panel = panel_from(values, values, np.linspace(0.9, 0.3, 6))
    text = asc.render_markdown(asc.analyse(panel))
    assert "自检" in text
    assert "不是同一个指标" in text or "只差一个符号" in text
