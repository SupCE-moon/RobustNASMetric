import importlib.util
import sys
from pathlib import Path

import torch
from robust_metric_lab.nasrobbench201 import NasRobRecord


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compute_nb201_metric_subset.py"
SPEC = importlib.util.spec_from_file_location("compute_nb201_metric_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def records(count: int) -> list[NasRobRecord]:
    return [
        NasRobRecord(
            arch_id=str(index),
            canonical_id=str(index),
            arch_string=None,
            dataset="cifar10",
            metrics={"autoattack": index / count},
        )
        for index in range(count)
    ]


def test_uniform_sampling_is_independent_of_training_seed(monkeypatch):
    rows = records(30)
    monkeypatch.setattr(MODULE, "iter_records", lambda *_: iter(rows))

    first = MODULE.choose_architectures(
        {}, {}, "cifar10", "autoattack", sample_size=10, seed=17, sampling="uniform"
    )
    second = MODULE.choose_architectures(
        {}, {}, "cifar10", "autoattack", sample_size=10, seed=17, sampling="uniform"
    )

    assert first == second
    assert len(first) == len(set(first)) == 10


def test_sampling_modes_produce_valid_unique_subsets(monkeypatch):
    rows = records(30)
    monkeypatch.setattr(MODULE, "iter_records", lambda *_: iter(rows))

    for sampling in ("uniform", "target_stratified"):
        selected = MODULE.choose_architectures(
            {}, {}, "cifar10", "autoattack", sample_size=10, seed=23, sampling=sampling
        )
        assert len(selected) == len(set(selected)) == 10
        assert set(selected).issubset({row.arch_id for row in rows})


def test_save_training_checkpoint_contains_reusable_model_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "arch_1_epoch_10_seed_0.pt"

    MODULE.save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        metadata={"arch_id": "1", "epoch": 10},
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"] == {"arch_id": "1", "epoch": 10}
    assert checkpoint["scheduler_state_dict"] is None
    assert set(checkpoint["model_state_dict"]) == {"weight", "bias"}
    assert not path.with_suffix(".pt.tmp").exists()
