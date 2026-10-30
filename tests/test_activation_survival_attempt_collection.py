"""Formal attempts are kept per setting and flattened without altering bytes."""

from __future__ import annotations

import json
import hashlib

import pytest

from scripts.activation_survival_manifest import SETTING_COUNTS, _canonical
from scripts.collect_activation_survival_attempts import collect, inspect_attempt_files
from scripts.verify_activation_survival import _strict_json, verify_records


def test_collector_requires_all_setting_directories_and_folder_key_match(tmp_path) -> None:
    attempts = tmp_path / "attempts"
    for setting in SETTING_COUNTS:
        (attempts / setting).mkdir(parents=True)
    row = {"key": ["rb40_mature", "id", None], "status": "failed", "error": "OOM"}
    (attempts / "rb40_mature" / "one.json").write_text(json.dumps(row), encoding="utf-8")
    files = inspect_attempt_files(attempts)
    assert len(files) == 1
    assert files[0][0].name == "one.json"
    (attempts / "rb40_mature" / "one.json").unlink()
    (attempts / "darts_e0" / "one.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="folder"):
        inspect_attempt_files(attempts)
    (attempts / "nb201_e5").rmdir()
    with pytest.raises(ValueError, match="setting directories"):
        inspect_attempt_files(attempts)


def test_collector_rejects_duplicate_filename_across_settings(tmp_path) -> None:
    attempts = tmp_path / "attempts"
    for setting in SETTING_COUNTS:
        (attempts / setting).mkdir(parents=True)
    for setting in ("rb40_mature", "nb201_e0"):
        (attempts / setting / "same.json").write_text(
            json.dumps({"key": [setting, "id", None]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate filename"):
        inspect_attempt_files(attempts)


def test_full_collector_keeps_failed_evidence_and_flat_dir_is_verifier_ready(tmp_path) -> None:
    objects = []
    attempts = tmp_path / "attempts"
    source = {"candidate.py": "a" * 64}
    for setting, count in SETTING_COUNTS.items():
        (attempts / setting).mkdir(parents=True)
        epoch = None if setting == "rb40_mature" else int(setting.rsplit("e", 1)[1])
        for index in range(count):
            objects.append({"setting": setting, "model_or_arch_id": str(index),
                            "checkpoint_epoch": epoch, "target_value": 0.5,
                            "checkpoint_sha256": "b" * 64,
                            "checkpoint_path": str(tmp_path / "weight.pt"),
                            "artifact_sha256": {"weight.pt": "b" * 64}})
    payload = {"expected_rows": 560, "objects": objects, "source_sha256": source}
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    manifest = {"frozen_sha256": digest, "frozen_payload": payload}
    manifest_path = tmp_path / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for number, obj in enumerate(objects):
        record = {"key": [obj["setting"], obj["model_or_arch_id"], obj["checkpoint_epoch"]],
                  "status": "ok", "metric_value": 1.0, "target_value": 0.5,
                  "sample_count": 64, "call_count": 1,
                  "checkpoint_sha256": "b" * 64,
                  "artifact_sha256": {"weight.pt": "b" * 64},
                  "manifest_sha256": digest, "source_sha256": source,
                  "diagnostics": {"elapsed_sec": 1.0}}
        (attempts / obj["setting"] / f"{number:03d}.json").write_text(json.dumps(record), encoding="utf-8")
    failure = {"key": ["rb40_mature", "0", None], "status": "failed",
               "error": "old OOM", "manifest_sha256": digest, "source_sha256": source}
    (attempts / "rb40_mature" / "old-failure.json").write_text(json.dumps(failure), encoding="utf-8")
    flat = tmp_path / "flat"
    result = collect(manifest_path, attempts, flat)
    assert result["unique_success_keys"] == 560
    assert result["failed_attempts_retained"] == 1
    assert len(list(flat.glob("*.json"))) == 561
    assert (flat / "collection_verification.meta").is_file()
    assert verify_records(manifest, [_strict_json(path) for path in flat.glob("*.json")])["failed_attempts"] == 1
