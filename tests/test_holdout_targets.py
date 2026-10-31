"""A DARTS holdout target must match the frozen full-test PGD-20 protocol."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.verify_holdout_targets import (
    load_official_nb201_sources,
    verify_nb201_target_item,
    verify_target_file,
    verify_target_item,
)


def darts_target() -> dict:
    return {
        "setting": "darts_e0",
        "model_or_arch_id": "17",
        "checkpoint_sha256": "0" * 64,
        "target_checkpoint_sha256": "1" * 64,
        "final_checkpoint_sha256": "1" * 64,
        "target_value": 0.47,
        "target_source_id": "new_darts_full_test_pgd20",
        "target_source_sha256": "2" * 64,
        "target_attack": "pgd20",
        "target_dataset": "cifar10_test",
        "target_sample_count": 10000,
        "target_sample_ids_sha256": hashlib.sha256(
            json.dumps(list(range(10000)), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_epsilon": 8 / 255,
        "target_step_size": 2 / 255,
        "target_steps": 20,
        "target_restarts": 1,
        "target_random_start": True,
        "target_batch_size": 128,
        "target_random_seed": 400000,
        "target_normalization": "cifar10_125.3_123.0_113.9__63.0_62.1_66.7",
    }


def test_darts_e0_target_is_bound_to_final_checkpoint() -> None:
    verify_target_item(darts_target())


@pytest.mark.parametrize("field, replacement, message", [
    ("target_checkpoint_sha256", "9" * 64, "checkpoint"),
    ("target_sample_count", 9999, "10,000"),
    ("target_sample_ids_sha256", "3" * 64, "sample IDs"),
    ("target_attack", "fgsm", "PGD-20"),
    ("target_epsilon", 4 / 255, "epsilon"),
    ("target_step_size", 1 / 255, "step size"),
    ("target_steps", 10, "steps"),
    ("target_restarts", 2, "restarts"),
    ("target_random_start", False, "random start"),
    ("target_batch_size", 64, "batch size"),
    ("target_random_seed", 400001, "seed"),
    ("target_dataset", "cifar10_train", "dataset"),
    ("target_normalization", "none", "normalization"),
    ("target_source_sha256", "", "source SHA"),
])
def test_wrong_darts_target_protocol_rejected(field: str, replacement: object, message: str) -> None:
    item = copy.deepcopy(darts_target())
    item[field] = replacement
    with pytest.raises(ValueError, match=message):
        verify_target_item(item)


def write_target_evidence(tmp_path, item: dict, *, correct=None, checkpoint=None) -> dict:
    record = {
        "status": "ok", "model_or_arch_id": item["model_or_arch_id"],
        "checkpoint_sha256": checkpoint or item["final_checkpoint_sha256"],
        "sample_ids": list(range(10000)),
        "correct": correct if correct is not None else [True] * 4700 + [False] * 5300,
        "protocol": {key: item[key] for key in (
            "target_attack", "target_dataset", "target_epsilon", "target_step_size",
            "target_steps", "target_restarts", "target_random_start", "target_batch_size",
            "target_random_seed", "target_normalization",
        )},
    }
    path = tmp_path / "darts_target.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    item["target_source_path"] = path.name
    item["target_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return item


def test_darts_target_file_recomputes_full_accuracy(tmp_path) -> None:
    item = write_target_evidence(tmp_path, darts_target())
    assert verify_target_file(item, tmp_path) == 0.47


@pytest.mark.parametrize("tamper, message", [
    ("missing_image", "10,000"),
    ("old_checkpoint", "checkpoint"),
    ("wrong_accuracy", "accuracy"),
    ("changed_file", "source SHA"),
])
def test_darts_target_file_rejects_incomplete_or_stale_evidence(tmp_path, tamper: str, message: str) -> None:
    item = darts_target()
    if tamper == "missing_image":
        item = write_target_evidence(tmp_path, item, correct=[True] * 4700 + [False] * 5299)
    elif tamper == "old_checkpoint":
        item = write_target_evidence(tmp_path, item, checkpoint="9" * 64)
    else:
        item = write_target_evidence(tmp_path, item)
        if tamper == "wrong_accuracy":
            item["target_value"] = 0.48
        else:
            (tmp_path / item["target_source_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_target_file(item, tmp_path)


def test_nb201_target_matches_official_canonical_record_only() -> None:
    benchmark = {"cifar10": {"7": {"autoattack": 0.37}}}
    meta = {"ids": {"7": {"isomorph": 7, "nb201-string": "arch-seven"}}}
    item = {"setting": "nb201_e5", "model_or_arch_id": "7",
            "signature": "arch-seven", "target_value": 0.37,
            "target_source_id": "official_nasrobbench201_cifar10_autoattack",
            "target_source_sha256": "a" * 64}
    assert verify_nb201_target_item(item, benchmark, meta, "a" * 64) == 0.37
    for field, replacement, message in (
        ("target_value", 0.38, "official"),
        ("signature", "wrong", "structure"),
        ("target_source_sha256", "b" * 64, "source SHA"),
    ):
        bad = {**item, field: replacement}
        with pytest.raises(ValueError, match=message):
            verify_nb201_target_item(bad, benchmark, meta, "a" * 64)
    meta["ids"]["7"]["isomorph"] = 3
    with pytest.raises(ValueError, match="canonical"):
        verify_nb201_target_item(item, benchmark, meta, "a" * 64)


def test_official_nb201_source_loader_checks_file_hashes(tmp_path) -> None:
    benchmark_path = tmp_path / "official.json"
    meta_path = tmp_path / "meta.json"
    benchmark_path.write_text(json.dumps({"cifar10": {"7": {"autoattack": 0.37}}}))
    meta_path.write_text(json.dumps({"ids": {"7": {"isomorph": 7,
                                                      "nb201-string": "arch-seven"}}}))
    benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    benchmark, meta = load_official_nb201_sources(benchmark_path, meta_path,
                                                   benchmark_sha, meta_sha)
    assert "7" in benchmark["cifar10"] and "7" in meta["ids"]
    with pytest.raises(ValueError, match="official results SHA"):
        load_official_nb201_sources(benchmark_path, meta_path, "0" * 64, meta_sha)
