"""Tests for the wide-to-long conversion used by the e0 audits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import wide_e0_to_rows as convert  # noqa: E402


def wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arch_id": [3, 1],
            "arch_string": ["s3", "s1"],
            "target": [0.4, 0.2],
            "feature_delta": [0.5, 0.1],
            "natural_loss": [1.0, np.nan],
            "not_a_metric": [9.0, 9.0],
        }
    )


def test_to_long_emits_one_row_per_metric_and_architecture():
    long = convert.to_long(wide_frame(), {"feature_delta": "lower", "natural_loss": "lower"})
    assert len(long) == 4
    assert set(long["metric_name"]) == {"feature_delta", "natural_loss"}
    assert list(long.columns) == [
        "setting",
        "model_or_arch_id",
        "metric_name",
        "metric_value",
        "status",
        "direction",
        "target_value",
    ]
    assert set(long["setting"]) == {"nb201_e0"}


def test_to_long_drops_columns_that_are_not_registered():
    long = convert.to_long(wide_frame(), {"feature_delta": "lower"})
    assert set(long["metric_name"]) == {"feature_delta"}


def test_to_long_records_missing_values_as_failed_not_dropped():
    long = convert.to_long(wide_frame(), {"natural_loss": "lower"})
    statuses = dict(zip(long["model_or_arch_id"], long["status"]))
    assert statuses == {"1": "failed", "3": "ok"}


def test_to_long_keeps_the_registry_direction():
    long = convert.to_long(wide_frame(), {"feature_delta": "lower", "natural_loss": "higher"})
    by_metric = dict(zip(long["metric_name"], long["direction"]))
    assert by_metric == {"feature_delta": "lower", "natural_loss": "higher"}


def test_to_long_casts_arch_ids_to_strings():
    long = convert.to_long(wide_frame(), {"feature_delta": "lower"})
    assert all(isinstance(v, str) for v in long["model_or_arch_id"])
    assert set(long["model_or_arch_id"]) == {"1", "3"}


def test_to_long_carries_the_target_on_every_row():
    long = convert.to_long(wide_frame(), {"feature_delta": "lower"})
    targets = dict(zip(long["model_or_arch_id"], long["target_value"]))
    assert targets == {"1": pytest.approx(0.2), "3": pytest.approx(0.4)}


def test_to_long_rejects_a_table_without_arch_id():
    frame = wide_frame().drop(columns=["arch_id"])
    with pytest.raises(ValueError, match="arch_id"):
        convert.to_long(frame, {"feature_delta": "lower"})


def test_to_long_rejects_a_table_without_a_target():
    frame = wide_frame().drop(columns=["target"])
    with pytest.raises(ValueError, match="target"):
        convert.to_long(frame, {"feature_delta": "lower"})


def test_to_long_accepts_oracle_target_as_the_target_column():
    # The e27 tables name the target "oracle_target"; the n400 table names it
    # "target".  Both feed the same audit, so the converter accepts either.
    frame = wide_frame().rename(columns={"target": "oracle_target"})
    long = convert.to_long(frame, {"feature_delta": "lower"})
    assert long["target_value"].notna().all()
    assert set(long["target_value"]) == set(frame["oracle_target"])


def test_to_long_rejects_a_table_with_no_registered_metric():
    frame = pd.DataFrame({"arch_id": [1], "target": [0.2], "unknown": [1.0]})
    with pytest.raises(ValueError, match="registered metric"):
        convert.to_long(frame, {"feature_delta": "lower"})


def test_registry_directions_covers_the_published_metrics():
    # The conversion must not invent a direction for a metric it cannot find.
    directions = convert.directions()
    assert directions["feature_delta"] == "lower"
    assert directions["trnas_consistency_phi"] == "higher"
    assert len(directions) >= 86
