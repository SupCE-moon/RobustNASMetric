import csv
import importlib.util
import random
import sys
from pathlib import Path

from robust_metric_lab.nasrobbench201 import NasRobRecord


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "search_nasrobbench201.py"
SPEC = importlib.util.spec_from_file_location("search_nasrobbench201", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(index: int, target: float, proxy: float, clean: float | None):
    return MODULE.Candidate(
        arch_id=str(index),
        canonical_id=str(index),
        arch_string=None,
        target=target,
        proxy=proxy,
        clean=clean,
    )


def test_proxy_csv_epoch_filter(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arch_id", "train_epochs", "metric", "clean_acc"])
        writer.writeheader()
        writer.writerow({"arch_id": "1", "train_epochs": 5, "metric": 0.5, "clean_acc": 0.6})
        writer.writerow({"arch_id": "1", "train_epochs": 10, "metric": 0.2, "clean_acc": 0.7})

    values = MODULE.read_proxy_csv(path, "arch_id", "metric", "train_epochs", "5")

    assert values == {"1": 0.5}


def test_build_candidates_uses_early_clean_csv():
    records = [
        NasRobRecord(
            arch_id="1",
            canonical_id="1",
            arch_string=None,
            dataset="cifar10",
            metrics={"autoattack": 0.4, "val_acc": 0.99},
        )
    ]

    rows = MODULE.build_candidates(
        records,
        target_metric="autoattack",
        clean_metric=None,
        proxy_column="metric",
        proxy_csv={"1": 0.2},
        clean_csv={"1": 0.7},
    )

    assert len(rows) == 1
    assert rows[0].clean == 0.7


def test_random_search_emits_unguided_and_clean_controls():
    candidates = [
        candidate(0, target=0.1, proxy=0.4, clean=0.9),
        candidate(1, target=0.5, proxy=0.1, clean=0.7),
        candidate(2, target=0.3, proxy=0.2, clean=0.8),
    ]

    rows = MODULE.random_search(
        candidates,
        budget=3,
        trials=2,
        rng=random.Random(7),
        proxy_direction="lower",
        global_best_target=0.5,
    )

    assert {row.algorithm for row in rows} == {
        "random_uniform",
        "random_proxy",
        "random_clean",
        "random_proxy_clean",
    }
    assert len(rows) == 8


def test_front_recommendation_uses_only_front_values():
    front = [
        candidate(0, target=0.2, proxy=1.0, clean=0.9),
        candidate(1, target=0.3, proxy=0.0, clean=0.5),
    ]

    selected = MODULE.recommend_from_front(front, proxy_direction="lower", use_clean=True)

    assert selected in front
