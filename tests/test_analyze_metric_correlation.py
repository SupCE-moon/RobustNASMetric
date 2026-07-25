import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_metric_correlation.py"
SPEC = importlib.util.spec_from_file_location("analyze_metric_correlation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_filter_rows_matches_epoch_as_csv_text():
    rows = [
        {"arch_id": "1", "train_epochs": "5"},
        {"arch_id": "1", "train_epochs": "10"},
        {"arch_id": "2", "train_epochs": "5"},
    ]

    filtered = MODULE.filter_rows(rows, "train_epochs", "5")

    assert [row["arch_id"] for row in filtered] == ["1", "2"]


def test_filter_rows_rejects_unpaired_arguments():
    with pytest.raises(ValueError, match="must be provided together"):
        MODULE.filter_rows([], "train_epochs", None)


def test_kendall_tau_b_accounts_for_ties():
    xs = [1.0, 1.0, 2.0]
    ys = [1.0, 2.0, 3.0]

    assert MODULE.kendall_tau_b(xs, ys) == pytest.approx(2.0 / (6.0**0.5))
