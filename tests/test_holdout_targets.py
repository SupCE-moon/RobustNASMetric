"""A DARTS holdout target must match the frozen full-test PGD-20 protocol."""

from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from scripts.verify_holdout_targets import (
    load_official_nb201_sources,
    load_frozen_rb11_manifest,
    load_rb11_recovery_report,
    verify_rb11_target_rows,
    verify_nb201_target_item,
    verify_nb201_target_rows,
    verify_target_file,
    verify_target_item,
    verify_darts_target_rows,
    verify_checkpoint_files,
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


def test_darts_target_file_rejects_duplicate_json_field(tmp_path) -> None:
    item = write_target_evidence(tmp_path, darts_target())
    path = tmp_path / item["target_source_path"]
    raw = path.read_text()
    path.write_text(raw.replace('"status": "ok"', '"status": "ok", "status": "ok"', 1))
    item["target_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        verify_target_file(item, tmp_path)


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


def _nb201_grid():
    benchmark = {"cifar10": {str(i): {"autoattack": 0.2 + i / 1000} for i in range(64)}}
    meta = {"ids": {str(i): {"isomorph": i, "nb201-string": f"arch-{i}"}
                    for i in range(64)}}
    rows = [
        {"setting": f"nb201_e{epoch}", "model_or_arch_id": str(i),
         "signature": f"arch-{i}", "target_value": 0.2 + i / 1000,
         "target_source_id": "official_nasrobbench201_cifar10_autoattack",
         "target_source_sha256": "a" * 64}
        for epoch in (0, 5, 10, 25, 50) for i in range(64)
    ]
    return rows, benchmark, meta


def test_nb201_all_320_targets_required_without_exposing_values() -> None:
    rows, benchmark, meta = _nb201_grid()
    report = verify_nb201_target_rows(rows, benchmark, meta, "a" * 64)
    assert report["status"] == "PASS"
    assert report["verified_rows"] == 320
    assert report["verified_architectures"] == 64
    assert "target_value" not in json.dumps(report)

    with pytest.raises(ValueError, match="320"):
        verify_nb201_target_rows(rows[:-1], benchmark, meta, "a" * 64)
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(ValueError, match="duplicate"):
        verify_nb201_target_rows(duplicate, benchmark, meta, "a" * 64)
    changed = [dict(row) for row in rows]
    changed[-1]["target_value"] = 0.99
    with pytest.raises(ValueError, match="official"):
        verify_nb201_target_rows(changed, benchmark, meta, "a" * 64)
    malformed = [dict(row) for row in rows]
    malformed[-1]["model_or_arch_id"] = ["63"]
    with pytest.raises(ValueError, match="architecture ID"):
        verify_nb201_target_rows(malformed, benchmark, meta, "a" * 64)


def test_rb11_targets_match_frozen_manifest_and_replacement_overlay(tmp_path) -> None:
    path = tmp_path / "rb40_models.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "robust_accuracy"])
        writer.writeheader()
        writer.writerows({"model_name": f"model_{i}", "robust_accuracy": "0.4"}
                         for i in range(10))
        writer.writerow({"model_name": "Rebuffi2021Fixing_70_16_cutmix_ddpm",
                         "robust_accuracy": "0.6"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = load_frozen_rb11_manifest(path, digest)
    rows = [{"setting": "rb40_mature", "model_or_arch_id": name,
             "target_value": record["robust_accuracy"],
             "target_source_id": "frozen_rb11_manifest_20260922",
             "target_source_sha256": digest,
             "checkpoint_sha256": ("b" * 64 if name.startswith("Rebuffi") else "c" * 64)}
            for name, record in manifest.items()]
    recovery = {"frozen_overlay": {"base_manifest_sha256": digest,
                                    "expected_unique_models": 11},
                "fresh_container_load_audit": {"status": "PASS", "loadable_unique_models": 11},
                "model_name": "Rebuffi2021Fixing_70_16_cutmix_ddpm",
                "replacement_weight": {"sha256": "b" * 64}}
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    recovery_sha = hashlib.sha256(recovery_path.read_bytes()).hexdigest()
    assert load_rb11_recovery_report(recovery_path, recovery_sha) == recovery
    with pytest.raises(ValueError, match="recovery SHA"):
        load_rb11_recovery_report(recovery_path, "0" * 64)
    report = verify_rb11_target_rows(rows, manifest, digest, recovery)
    assert report["status"] == "PASS" and report["verified_rows"] == 11
    with pytest.raises(ValueError, match="11"):
        verify_rb11_target_rows(rows[:-1], manifest, digest, recovery)
    wrong = [dict(row) for row in rows]
    wrong[-1]["target_value"] = 0.7
    with pytest.raises(ValueError, match="frozen"):
        verify_rb11_target_rows(wrong, manifest, digest, recovery)
    recovery["replacement_weight"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="replacement"):
        verify_rb11_target_rows(rows, manifest, digest, recovery)


def test_darts_all_40_stage_keys_share_20_fulltest_targets(tmp_path) -> None:
    rows = []
    for index in range(20):
        item = darts_target()
        item["model_or_arch_id"] = str(1000 + index)
        item["checkpoint_sha256"] = f"{index + 1:064x}"
        item["final_checkpoint_sha256"] = f"{index + 101:064x}"
        item["target_checkpoint_sha256"] = item["final_checkpoint_sha256"]
        record = {
            "status": "ok", "model_or_arch_id": item["model_or_arch_id"],
            "checkpoint_sha256": item["final_checkpoint_sha256"],
            "sample_ids": list(range(10000)),
            "correct": [True] * 4700 + [False] * 5300,
            "protocol": {key: item[key] for key in (
                "target_attack", "target_dataset", "target_epsilon", "target_step_size",
                "target_steps", "target_restarts", "target_random_start", "target_batch_size",
                "target_random_seed", "target_normalization",
            )},
        }
        path = tmp_path / f"darts_{index}.json"
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        item["target_source_path"] = path.name
        item["target_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        final = {**item, "setting": "darts_e100",
                 "checkpoint_sha256": item["final_checkpoint_sha256"]}
        rows.extend((item, final))
    report = verify_darts_target_rows(rows, tmp_path)
    assert report["status"] == "PASS" and report["verified_rows"] == 40
    with pytest.raises(ValueError, match="40"):
        verify_darts_target_rows(rows[:-1], tmp_path)
    wrong = [dict(row) for row in rows]
    wrong[1]["target_source_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="stage|source SHA"):
        verify_darts_target_rows(wrong, tmp_path)


def test_checkpoint_files_bound_to_project_and_sha(tmp_path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"verified checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = [{"setting": "darts_e0", "model_or_arch_id": "1000",
             "checkpoint_path": checkpoint.name, "checkpoint_sha256": digest}]
    report = verify_checkpoint_files(rows, tmp_path)
    assert report["status"] == "PASS" and report["verified_files"] == 1
    assert report["verified_rows"] == 1
    with pytest.raises(ValueError, match="SHA"):
        verify_checkpoint_files([{**rows[0], "checkpoint_sha256": "0" * 64}], tmp_path)
    with pytest.raises(ValueError, match="outside|relative"):
        verify_checkpoint_files([{**rows[0], "checkpoint_path": "../weights.pt"}], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        verify_checkpoint_files(rows + rows, tmp_path)
