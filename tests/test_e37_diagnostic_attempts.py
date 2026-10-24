from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_e37_layer_diagnostic import (
    run_one,
    validate_repair_keys,
    write_attempt,
)


def test_attempt_file_is_exclusive_and_preserves_failure(tmp_path) -> None:
    path = tmp_path / "attempts" / "a1" / "darts_e0" / "17.json"
    record = {
        "setting": "darts_e0", "model_or_arch_id": "17", "status": "failed",
        "error_type": "RuntimeError", "error": "out of memory",
        "traceback": "RuntimeError: out of memory",
    }
    write_attempt(path, record)
    assert json.loads(path.read_text(encoding="utf-8")) == record
    with pytest.raises(FileExistsError):
        write_attempt(path, record)


def test_repair_is_exact_failed_or_missing_set() -> None:
    expected = {("darts_e0", "17"), ("darts_e100", "17")}
    successful = {("darts_e0", "17")}
    validate_repair_keys({("darts_e100", "17")}, expected, successful)
    with pytest.raises(ValueError, match="successful"):
        validate_repair_keys({("darts_e0", "17")}, expected, successful)
    with pytest.raises(ValueError, match="unknown"):
        validate_repair_keys({("darts_e100", "other")}, expected, successful)
    with pytest.raises(ValueError, match="all missing/failed"):
        validate_repair_keys(set(), expected, successful)


def test_one_success_and_one_loader_failure_keep_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    from scripts.e37_diagnostic_manifest import sha256_file

    obj = {
        "setting": "darts_e0", "model_or_arch_id": "17",
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
    }
    item = SimpleNamespace(model_or_arch_id="17", checkpoint_epoch=0)
    diagnostic = {
        "sample_count": 64, "anchor_metadata": [{"module_path": str(i)} for i in range(4)],
        "sample_components": [], "metric_values": {}, "feature_count": 4,
    }
    success = run_one(
        item, obj, loader=lambda _: object(), samples=(object(), object()),
        diagnose=lambda *_args, **_kwargs: diagnostic,
        manifest_hash="manifest", source_hashes={"e37": "source"}, epsilon=8 / 255,
    )
    assert success["status"] == "ok"
    assert success["diagnostic"]["sample_count"] == 64
    assert success["manifest_sha256"] == "manifest"
    assert success["checkpoint_sha256"] == obj["checkpoint_sha256"]

    def fail(_item):
        raise RuntimeError("out of memory")

    failure = run_one(
        item, obj, loader=fail, samples=(object(), object()),
        diagnose=lambda *_args, **_kwargs: diagnostic,
        manifest_hash="manifest", source_hashes={"e37": "source"}, epsilon=8 / 255,
    )
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert "out of memory" in failure["error"]
    assert "RuntimeError: out of memory" in failure["traceback"]
    assert "diagnostic" not in failure
    assert failure["manifest_sha256"] == "manifest"
