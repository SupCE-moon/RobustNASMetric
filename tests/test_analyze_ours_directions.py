"""Tests for scripts/analyze_ours_directions.py.

The point of the script is to rank *our* metrics inside several different
questions, so the tests focus on the three things that could silently lie:
reading a BOM-prefixed CSV, keeping "Ours" separate from external methods, and
reporting a rank that includes the external ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_ours_directions as aod  # noqa: E402


def make_table(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows: (metric_name, source_type, rb40_mature, all_setting_mean)."""
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "source_type": source,
                "rb40_mature": rb40,
                "nb201_e0": rb40,
                "nb201_mean": rb40,
                "darts_e100": rb40,
                "all_setting_mean": total,
            }
            for name, source, rb40, total in rows
        ]
    )


def test_load_table_strips_the_bom(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("metric_name,source_type\nm,Ours\n", encoding="utf-8-sig")

    frame = aod.load_table(path)

    # Without utf-8-sig the first column would be named "\ufeffmetric_name".
    assert list(frame.columns)[0] == "metric_name"


def test_ours_names_excludes_external_methods():
    table = make_table([("a", "Ours", 0.5, 0.4), ("b", "论文/已有方法", 0.9, 0.8), ("c", "其他基线", 0.1, 0.2)])

    assert aod.ours_names(table) == {"a"}


def test_scores_from_table_skips_missing_values():
    table = make_table([("a", "Ours", 0.5, 0.4), ("b", "Ours", None, 0.2)])

    scores = aod.scores_from_table(table, "rb40_mature")

    # "missing" must not be read as zero: b simply has no number here.
    assert scores == {"a": 0.5}


def test_rank_block_orders_by_value_descending():
    block = aod.rank_block({"low": 0.1, "high": 0.7, "mid": 0.4}, {"low", "high", "mid"}, top_k=3)

    assert [row["metric"] for row in block["top"]] == ["high", "mid", "low"]
    assert block["top"][0]["rank_ours"] == 1


def test_rank_block_counts_external_metrics_in_rank_all():
    block = aod.rank_block(
        {"ext1": 0.9, "ext2": 0.8, "ours_best": 0.7, "ours2": 0.2},
        {"ours_best", "ours2"},
        top_k=2,
    )

    # The champion is first among Ours but third on the whole board: that gap is
    # exactly what the report is trying to expose.
    assert block["champion"]["metric"] == "ours_best"
    assert block["champion"]["rank_ours"] == 1
    assert block["champion"]["rank_all"] == 3
    assert block["best_all_rank"] == 3
    assert block["n_all"] == 4
    assert block["n_ours"] == 2


def test_rank_block_drops_nan_scores():
    block = aod.rank_block({"a": 0.5, "b": float("nan")}, {"a", "b"}, top_k=5)

    assert [row["metric"] for row in block["top"]] == ["a"]


def test_scores_from_confounders_uses_the_two_control_column():
    report = {
        "metrics": [
            {"metric": "a", "rho_clean": 0.9, "rho_clean_params": 0.4},
            {"metric": "b", "rho_clean": 0.8, "rho_clean_params": None},
        ]
    }

    # rho_clean is the weaker control; using it would overstate every metric.
    assert aod.scores_from_confounders(report) == {"a": 0.4}


def test_scores_from_seed_uses_the_mean_rho():
    report = {"metrics": [{"metric": "a", "value": {"mean_rho": 0.97}}, {"metric": "b", "value": {}}]}

    assert aod.scores_from_seed(report) == {"a": 0.97}


def test_scores_from_selection_keeps_only_the_requested_k():
    report = {
        "rows": [
            {"k_percent": 10, "metric": "a", "delta": 0.03},
            {"k_percent": 20, "metric": "a", "delta": 0.05},
            {"k_percent": 10, "metric": "b", "delta": None},
        ]
    }

    assert aod.scores_from_selection(report, k_percent=10) == {"a": 0.03}


def test_analyse_covers_every_direction():
    table = make_table([("a", "Ours", 0.5, 0.4), ("b", "论文/已有方法", 0.9, 0.8)])

    report = aod.analyse(table, {"metrics": []}, {"metrics": []}, {"rows": []})

    expected = {name for name, _, _ in aod.DIRECTIONS}
    assert set(report["directions"]) == expected
    assert len(report["gaps"]) == len(aod.DIRECTIONS)


def test_analyse_sorts_multi_direction_by_count_then_name():
    # "wide" has a number in every direction; "narrow" is missing one, so it can
    # only enter four top-ten lists and must be sorted after "wide".
    wide = {"metric_name": "wide", "source_type": "Ours"}
    narrow = {"metric_name": "narrow", "source_type": "Ours"}
    for name, _, _ in aod.DIRECTIONS:
        wide[name] = 0.9
        narrow[name] = 0.5
    narrow["nb201_e0"] = float("nan")
    table = pd.DataFrame([wide, narrow])

    report = aod.analyse(table, {"metrics": []}, {"metrics": []}, {"rows": []})

    first, second = report["multi_direction"][:2]
    # The last three directions are fed by the other reports, which are empty
    # here, so only the table columns can contribute a rank — but "narrow" must
    # still end up one short, behind "wide".
    assert first["metric"] == "wide"
    assert second["metric"] == "narrow"
    assert first["n_directions"] == second["n_directions"] + 1




def test_render_mentions_labels_and_the_honest_sentence():
    table = make_table([("a", "Ours", 0.5, 0.4)])
    report = aod.analyse(table, {"metrics": []}, {"metrics": []}, {"rows": []})

    text = aod.render_markdown(report)

    assert "已训练防御模型" in text
    assert "a" in text
    # The report must never claim "we are the best overall".
    assert "总榜" in text
    assert report["n_ours"] == 1


def test_render_survives_a_direction_without_any_ours_metric():
    table = make_table([("ext", "论文/已有方法", 0.9, 0.8)])

    report = aod.analyse(table, {"metrics": []}, {"metrics": []}, {"rows": []})
    text = aod.render_markdown(report)

    assert report["n_ours"] == 0
    assert isinstance(text, str) and text
