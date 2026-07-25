import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_e17_capacity_validation.py"
SPEC = importlib.util.spec_from_file_location("analyze_e17_capacity_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, arch_id: int, epochs: list[int]) -> None:
    pd.DataFrame(
        [
            {
                "arch_id": arch_id,
                "train_epochs": epoch,
                "oracle_target": arch_id / 10,
                "perturbation_capacity_delta": 1 / (arch_id + epoch),
                "clean_acc": arch_id / 20,
                "natural_loss": 1 / arch_id,
                "train_epoch_loss": 2 / arch_id,
            }
            for epoch in epochs
        ]
    ).to_csv(path, index=False)


def test_load_complete_rows_drops_partial_duplicate_source(tmp_path):
    full = tmp_path / "full.csv"
    partial = tmp_path / "partial.csv"
    write_rows(full, arch_id=1, epochs=[10, 25, 50])
    write_rows(partial, arch_id=1, epochs=[10])

    result = MODULE.load_complete_rows([full, partial], (10, 25, 50))

    assert result["arch_id"].nunique() == 1
    assert result["train_epochs"].tolist() == [10, 25, 50]
    assert set(result["source_file"]) == {"full.csv"}


def test_partial_rank_correlation_returns_finite_values():
    frame = pd.DataFrame(
        {
            "metric": [5, 4, 3, 2, 1],
            "target": [1, 2, 3, 5, 4],
            "control": [1, 3, 2, 5, 4],
        }
    )

    value, pvalue = MODULE.partial_rank_correlation(
        frame, "metric", "target", ["control"]
    )

    assert -1 <= value <= 1
    assert 0 <= pvalue <= 1
