"""Tests for the offline strict-ranking recomputation tool.

The tool is the safety net for "the server is down, is this rank real?".
These tests pin the three places where it could silently lie:
direction handling, the eight-setting requirement, and the comparison verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import verify_strict_ranking as vsr  # noqa: E402


SMALL = {s: 4 for s in vsr.SETTING_KEYS}


def _rows(metric: str, direction: str, *, settings: tuple[str, ...] | None = None, n: int = 4):
    """Build rows where the oriented correlation is exactly +1 for every setting."""
    rows = []
    chosen = settings if settings is not None else vsr.SETTING_KEYS
    for setting in chosen:
        for i in range(n):
            target = 0.1 + 0.25 * i
            value = target if direction == "higher" else -target
            rows.append(
                {
                    "run_id": "synthetic",
                    "benchmark": setting.split("_")[0],
                    "setting": setting,
                    "model_or_arch_id": f"m{i}",
                    "checkpoint_epoch": 0,
                    "metric_name": metric,
                    "direction": direction,
                    "metric_value": value,
                    "target_value": target,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def test_lower_direction_is_negated() -> None:
    table, rejected = vsr.recompute(_rows("f_lower", "lower"), SMALL)
    assert not rejected
    assert table.loc[0, "rank"] == 1
    for setting in vsr.SETTING_KEYS:
        assert table.loc[0, setting] == pytest.approx(1.0)
    assert table.loc[0, "all_setting_mean"] == pytest.approx(1.0)


def test_higher_direction_is_kept() -> None:
    table, _ = vsr.recompute(_rows("f_higher", "higher"), SMALL)
    assert table.loc[0, "all_setting_mean"] == pytest.approx(1.0)


def test_metric_missing_a_setting_is_rejected_not_zero_filled() -> None:
    partial = vsr.SETTING_KEYS[:-1]
    table, rejected = vsr.recompute(_rows("f_partial", "lower", settings=partial), SMALL)
    assert table.empty
    assert "f_partial" in rejected
    assert "darts_e100" in rejected["f_partial"]


def test_rank_breaks_ties_by_metric_name() -> None:
    frame = pd.concat([_rows("b_metric", "lower"), _rows("a_metric", "lower")], ignore_index=True)
    table, _ = vsr.recompute(frame, SMALL)
    assert list(table["metric_name"]) == ["a_metric", "b_metric"]
    assert list(table["rank"]) == [1, 2]


def test_repair_rows_replace_failed_originals(tmp_path: Path) -> None:
    frame = _rows("trnas_x", "higher")
    broken = frame.copy()
    broken.loc[0, "status"] = "failed"
    broken.loc[0, "metric_value"] = None
    rows_path = tmp_path / "rows.csv"
    broken.to_csv(rows_path, index=False)

    repair = frame.iloc[[0]].copy()
    repair_path = tmp_path / "repair.csv"
    repair.to_csv(repair_path, index=False)

    without_repair, rejected = vsr.recompute(vsr.load_rows(rows_path, None), SMALL)
    assert "trnas_x" in rejected, "a failed row must make the metric unusable"

    with_repair, rejected2 = vsr.recompute(vsr.load_rows(rows_path, repair_path), SMALL)
    assert not rejected2
    assert with_repair.loc[0, "all_setting_mean"] == pytest.approx(1.0)


def test_compare_flags_a_changed_value() -> None:
    mine, _ = vsr.recompute(_rows("only", "lower"), SMALL)
    official = pd.read_csv(_official_from(mine))
    verdict = vsr.compare(mine, official)
    assert verdict["exact_match"] is True
    assert verdict["rank_order_identical"] is True

    tampered = official.copy()
    tampered.loc[0, "all_setting_mean"] = tampered.loc[0, "all_setting_mean"] - 0.01
    verdict = vsr.compare(mine, tampered)
    assert verdict["exact_match"] is False
    assert verdict["max_abs_error"] == pytest.approx(0.01)


def test_extra_rows_are_appended_before_recomputing(tmp_path: Path) -> None:
    """The main raw-row export predates one metric; its rows arrive in a second file.

    Before the server copy was recovered the recomputation covered 83 of the 84
    ranked metrics, and the report said the missing one could not be checked
    offline. Appending the extra file is what makes the fourth-place-through-
    eighty-fourth comparison complete, so the append must happen *before* the
    repair merge and before the eight-setting check.
    """
    frame = _rows("late_metric", "higher")
    first = frame.iloc[: len(frame) // 2]
    second = frame.iloc[len(frame) // 2 :]
    rows_path = tmp_path / "rows.csv"
    extra_path = tmp_path / "extra.csv"
    first.to_csv(rows_path, index=False)
    second.to_csv(extra_path, index=False)

    _, rejected = vsr.recompute(vsr.load_rows(rows_path, None), SMALL)
    assert "late_metric" in rejected, "half the grid must not be enough"

    table, rejected2 = vsr.recompute(vsr.load_rows(rows_path, None, [extra_path]), SMALL)
    assert not rejected2
    assert table.loc[0, "all_setting_mean"] == pytest.approx(1.0)


def _official_from(table: pd.DataFrame) -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "ranking.csv"
    table.drop(columns=["rank"]).to_csv(path, index=False)
    return path
