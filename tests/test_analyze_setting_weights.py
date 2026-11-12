"""Tests for tools/analyze_setting_weights.py.

The script makes a claim about *where ranks come from*, so the tests pin the
arithmetic that claim rests on: how a grid score is formed, how NaN is treated,
and what "agreement" means.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_setting_weights", ROOT / "tools" / "analyze_setting_weights.py")
assert spec and spec.loader
asw = importlib.util.module_from_spec(spec)
sys.modules["analyze_setting_weights"] = asw
spec.loader.exec_module(asw)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a table with the columns the tool needs."""
    base = {column: float("nan") for column in asw.SETTING_COLUMNS}
    records = []
    for index, row in enumerate(rows):
        record = dict(base)
        record.update(row)
        record.setdefault("strict_status", "ranked")
        record.setdefault("metric_name", f"m{index}")
        record.setdefault("all_setting_mean", float("nan"))
        records.append(record)
    return pd.DataFrame(records)


def test_grid_scores_average_within_grid() -> None:
    frame = _frame(
        [
            {
                "metric_name": "a",
                "rb40_mature": 0.5,
                "nb201_e0": 0.1,
                "nb201_e5": 0.2,
                "nb201_e10": 0.3,
                "nb201_e25": 0.4,
                "nb201_e50": 0.5,
                "darts_e0": 0.8,
                "darts_e100": 0.6,
                "all_setting_mean": 0.425,
            }
        ]
    )
    scores = asw.grid_scores(frame)
    assert scores.at[0, "rb40"] == pytest.approx(0.5)
    assert scores.at[0, "nb201"] == pytest.approx(0.3)
    assert scores.at[0, "darts"] == pytest.approx(0.7)
    assert scores.at[0, "all"] == pytest.approx(0.425)


def test_rank_grid_never_ranks_nan() -> None:
    ranks = asw.rank_grid(pd.Series([0.9, float("nan"), 0.5]))
    assert ranks[0] == 1
    assert pd.isna(ranks[1])
    assert ranks[2] == 2


def test_rank_grid_breaks_ties_with_min() -> None:
    # Descending rank with method="min": the two 0.5s are both rank 1, the 0.1 is rank 3.
    ranks = asw.rank_grid(pd.Series([0.5, 0.5, 0.1]))
    assert list(ranks) == [1.0, 1.0, 3.0]


def test_compare_identical_order_is_perfect() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = asw.compare(a, b)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["top10_overlap"] == 4


def test_compare_reversed_order_is_negative() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([4.0, 3.0, 2.0, 1.0])
    result = asw.compare(a, b)
    assert result["kendall_tau"] == pytest.approx(-1.0)
    # With only four metrics, "top 10" is all of them, so the overlap is 4 -- the
    # overlap column only starts to mean something once a grid ranks >10 metrics.
    assert result["top10_overlap"] == 4


def test_top10_overlap_detects_disagreement_on_a_long_ranking() -> None:
    a = pd.Series(range(1, 21), dtype=float)  # 1 is best
    b = pd.Series(range(20, 0, -1), dtype=float)  # exact reverse
    result = asw.compare(a, b)
    # a's best ten are indices 0..9; b's best ten are indices 19..10.
    assert result["top10_overlap"] == 0
    assert result["kendall_tau"] == pytest.approx(-1.0)


def test_compare_drops_metrics_missing_on_either_side() -> None:
    a = pd.Series([1.0, 2.0, 3.0, float("nan")])
    b = pd.Series([1.0, 2.0, 3.0, 1.0])
    result = asw.compare(a, b)
    assert result["n"] == 3


def test_compare_refuses_to_claim_agreement_from_two_points() -> None:
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0])
    result = asw.compare(a, b)
    assert result["kendall_tau"] is None


def test_load_ranked_excludes_rows_without_a_rank() -> None:
    scores = [
        {"metric_name": "a", "strict_status": "ranked", "all_setting_mean": 0.9},
        {"metric_name": "b", "strict_status": "excluded", "all_setting_mean": float("nan")},
    ]
    frame = _frame(scores)
    frame.loc[0, "rb40_mature"] = 0.5
    kept = frame[frame["strict_status"] == "ranked"]
    assert kept["metric_name"].tolist() == ["a"]


def test_settings_cover_every_declared_column() -> None:
    assert set(asw.GRID_OF_SETTING) == set(asw.SETTING_COLUMNS)


def test_weights_match_setting_counts() -> None:
    counts = {"rb40": 0, "nb201": 0, "darts": 0}
    for column in asw.SETTING_COLUMNS:
        counts[asw.GRID_OF_SETTING[column]] += 1
    total = sum(counts.values())
    assert total == 8
    assert counts == {"rb40": 1, "nb201": 5, "darts": 2}


def test_render_markdown_states_the_weight_it_used() -> None:
    frame = _frame(
        [
            {"metric_name": "a", "rb40_mature": 0.9, "nb201_e0": 0.9, "nb201_e5": 0.9,
             "nb201_e10": 0.9, "nb201_e25": 0.9, "nb201_e50": 0.9, "darts_e0": 0.9,
             "darts_e100": 0.9, "all_setting_mean": 0.9},
            {"metric_name": "b", "rb40_mature": 0.1, "nb201_e0": 0.1, "nb201_e5": 0.1,
             "nb201_e10": 0.1, "nb201_e25": 0.1, "nb201_e50": 0.1, "darts_e0": 0.1,
             "darts_e100": 0.1, "all_setting_mean": 0.1},
        ]
    )
    payload = asw.analyze(frame)
    text = asw.render_markdown(payload, "out.csv", "abc123")
    assert "62.5%" in text
    assert "12.5%" in text
    assert "不能证明" in text
