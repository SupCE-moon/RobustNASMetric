"""Tests for ``tools/verify_doc_claims.py``.

The tool's whole value is that it fails loudly when prose and table disagree.
These tests make sure it really fails in that case, and that the rounding
tolerance is neither so tight that three-decimal tables always fail nor so loose
that a genuine mismatch slips through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_doc_claims as vdc  # noqa: E402


def table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["strict_rank"] = pd.to_numeric(frame["strict_rank"], errors="coerce")
    return frame


def row(metric: str, rank, mean: float, *, status: str = "ranked", stage: float = 0.5) -> dict:
    return {
        "metric_name": metric,
        "strict_rank": rank,
        "strict_status": status,
        "raw_rows": 560,
        "release_id": "rel",
        "all_setting_mean": mean,
        "nb201_e0": stage,
    }


def test_field_claim_passes_within_rounding() -> None:
    frame = table([row("a", 1, 0.782)])
    claim = {
        "kind": "field",
        "metric": "a",
        "field": "all_setting_mean",
        "expected": 0.7822,
        "ok": None,
    }
    result = vdc.evaluate(frame, claim)
    assert result["ok"] is True
    assert result["error"] == pytest.approx(-0.0002)


def test_field_claim_fails_beyond_tolerance() -> None:
    frame = table([row("a", 1, 0.700)])
    claim = {
        "kind": "field",
        "metric": "a",
        "field": "all_setting_mean",
        "expected": 0.7822,
        "ok": None,
    }
    result = vdc.evaluate(frame, claim)
    assert result["ok"] is False
    assert result["error"] == pytest.approx(-0.0822)


def test_diff_claim_uses_signed_gap() -> None:
    frame = table([row("max", 6, 0.524), row("phi", 1, 0.782)])
    claim = {
        "kind": "diff",
        "left": "max",
        "right": "phi",
        "field": "all_setting_mean",
        "expected": -0.258,
        "ok": None,
    }
    result = vdc.evaluate(frame, claim)
    assert result["ok"] is True
    assert result["actual"] == pytest.approx(-0.258)


def test_diff_claim_rejects_reversed_sign() -> None:
    frame = table([row("max", 6, 0.524), row("phi", 1, 0.782)])
    claim = {
        "kind": "diff",
        "left": "phi",
        "right": "max",
        "field": "all_setting_mean",
        "expected": -0.258,
        "ok": None,
    }
    assert vdc.evaluate(frame, claim)["ok"] is False


def test_lookup_rejects_missing_metric() -> None:
    with pytest.raises(SystemExit):
        vdc.lookup(table([row("a", 1, 0.5)]), "missing", "all_setting_mean")


def test_lookup_rejects_duplicate_metric() -> None:
    with pytest.raises(SystemExit):
        vdc.lookup(table([row("a", 1, 0.5), row("a", 2, 0.6)]), "a", "all_setting_mean")


def test_lookup_rejects_empty_value() -> None:
    frame = table([row("a", 1, 0.5)])
    frame.loc[0, "nb201_e0"] = None
    with pytest.raises(SystemExit):
        vdc.lookup(frame, "a", "nb201_e0")


def test_invariants_accept_a_clean_table() -> None:
    rows = [row(f"m{i}", i, 0.5) for i in range(1, 85)]
    rows += [row("x", None, 0.1, status="excluded"), row("y", None, 0.1, status="excluded")]
    checks = vdc.check_invariants(table(rows))
    assert all(check["ok"] for check in checks), checks


def test_invariants_catch_a_rank_gap() -> None:
    rows = [row(f"m{i}", i, 0.5) for i in range(1, 84)]  # one rank short
    rows += [row("x", None, 0.1, status="excluded"), row("y", None, 0.1, status="excluded")]
    checks = {check["name"]: check for check in vdc.check_invariants(table(rows))}
    assert checks["名次连续 1..84"]["ok"] is False


def test_invariants_catch_wrong_raw_rows() -> None:
    rows = [row(f"m{i}", i, 0.5) for i in range(1, 85)]
    rows += [row("x", None, 0.1, status="excluded"), row("y", None, 0.1, status="excluded")]
    frame = table(rows)
    frame.loc[0, "raw_rows"] = 559
    checks = {check["name"]: check for check in vdc.check_invariants(frame)}
    assert checks["每行原始行数唯一值"]["ok"] is False
