import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_activation_survival_student import (
    evaluate_keys,
    pending_keys,
    read_success_keys,
    write_attempt,
)


MANIFEST_HASH = "a" * 64
SOURCE = {"src/robust_metric_lab/activation_sign_survival.py": "b" * 64}
KEY = ("darts_e0", "arch-1", 0)


def record(status, **extra):
    return {
        "key": list(KEY), "status": status, "manifest_sha256": MANIFEST_HASH,
        "source_sha256": SOURCE, **extra,
    }


def test_retry_preserves_failure_and_uses_distinct_files(tmp_path):
    first = write_attempt(tmp_path, record("failed", error="OOM", traceback="trace"))
    second = write_attempt(tmp_path, record("ok", metric_value=1.0))
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["error"] == "OOM"
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert read_success_keys(tmp_path, MANIFEST_HASH, SOURCE) == {KEY}


def test_duplicate_success_is_rejected(tmp_path):
    write_attempt(tmp_path, record("ok", metric_value=1.0))
    write_attempt(tmp_path, record("ok", metric_value=2.0))
    with pytest.raises(ValueError, match="duplicate"):
        read_success_keys(tmp_path, MANIFEST_HASH, SOURCE)


def test_nonfinite_attempt_never_leaves_partial_file(tmp_path):
    with pytest.raises(ValueError):
        write_attempt(tmp_path, record("ok", metric_value=float("nan")))
    assert list(tmp_path.glob("*.json")) == []


def test_changed_manifest_or_source_refuses_resume(tmp_path):
    write_attempt(tmp_path, record("failed", error="OOM", traceback="trace"))
    with pytest.raises(ValueError, match="manifest"):
        read_success_keys(tmp_path, "c" * 64, SOURCE)
    with pytest.raises(ValueError, match="source"):
        read_success_keys(tmp_path, MANIFEST_HASH, {"changed.py": "d" * 64})


def test_only_missing_success_keys_are_pending(tmp_path):
    write_attempt(tmp_path, record("failed", error="OOM", traceback="trace"))
    expected = {KEY, ("darts_e0", "arch-2", 0)}
    assert pending_keys(expected, tmp_path, MANIFEST_HASH, SOURCE) == expected
    write_attempt(tmp_path, record("ok", metric_value=1.0))
    assert pending_keys(expected, tmp_path, MANIFEST_HASH, SOURCE) == {("darts_e0", "arch-2", 0)}


def test_loader_failure_is_recorded_and_exact_key_retry_keeps_evidence(tmp_path):
    checkpoint = tmp_path / "weight.pt"
    checkpoint.write_bytes(b"weight")
    item = SimpleNamespace(
        model_or_arch_id="arch-1", checkpoint_epoch=0,
        checkpoint_path=checkpoint, target_value=0.5,
    )
    payload = {
        "objects": [{
            "setting": "darts_e0", "model_or_arch_id": "arch-1",
            "checkpoint_epoch": 0, "checkpoint_sha256": hashlib.sha256(b"weight").hexdigest(),
            "target_value": 0.5,
        }],
    }
    manifest = {"frozen_sha256": MANIFEST_HASH, "frozen_payload": payload}
    calls = []

    def bad_loader(_item):
        calls.append("bad")
        raise RuntimeError("load failed")

    samples = lambda: (torch.ones(1, 1), torch.zeros(1, dtype=torch.long))
    first = evaluate_keys([item], bad_loader, samples, manifest, tmp_path / "attempts",
                          setting="darts_e0", source_sha256=SOURCE)
    assert first == {"written": 1, "failed": 1, "skipped": 0}
    before = sorted((tmp_path / "attempts").glob("*.json"))
    assert len(before) == 1
    assert "load failed" in before[0].read_text(encoding="utf-8")

    def good_loader(_item):
        calls.append("good")
        return object()

    def compute(_model, _clean, _labels, *, epsilon):
        return {"metric_value": 1.5, "sample_count": 1, "call_count": 1, "diagnostics": {}}

    second = evaluate_keys([item], good_loader, samples, manifest, tmp_path / "attempts",
                           setting="darts_e0", source_sha256=SOURCE, compute=compute)
    assert second == {"written": 1, "failed": 0, "skipped": 0}
    assert calls == ["bad", "good"]
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2
    assert before[0].exists()
    third = evaluate_keys([item], good_loader, samples, manifest, tmp_path / "attempts",
                          setting="darts_e0", source_sha256=SOURCE, compute=compute)
    assert third == {"written": 0, "failed": 0, "skipped": 1}
