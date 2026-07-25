import csv
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_metric_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_metric_shards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arch_id", "train_epochs"])
        writer.writeheader()
        writer.writerows(rows)


def test_merge_rows_and_validate_complete_grid(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    write_rows(left, [{"arch_id": "1", "train_epochs": "5"}, {"arch_id": "1", "train_epochs": "10"}])
    write_rows(right, [{"arch_id": "2", "train_epochs": "5"}, {"arch_id": "2", "train_epochs": "10"}])

    _, rows = MODULE.merge_rows([left, right], ["arch_id", "train_epochs"])
    MODULE.validate_grid(rows, 2, [5, 10])

    assert len(rows) == 4


def test_merge_rows_rejects_duplicate_keys(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    row = {"arch_id": "1", "train_epochs": "5"}
    write_rows(left, [row])
    write_rows(right, [row])

    with pytest.raises(ValueError, match="Duplicate key"):
        MODULE.merge_rows([left, right], ["arch_id", "train_epochs"])
