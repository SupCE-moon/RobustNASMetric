"""The independent NB201 weight task must be target blind and append only."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.train_activation_holdout_nb201 import (
    _verified_complete,
    expected_training_plan,
    load_frozen_selection,
    write_or_verify_plan,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selection():
    return {
        "kind": "activation_holdout_nb201_selection",
        "status": "unsealed_weights_and_targets_missing",
        "expected_architectures": 64,
        "expected_keys": 320,
        "checkpoint_epochs": [0, 5, 10, 25, 50],
        "architectures": [
            {"arch_id": str(i), "canonical_id": str(i), "signature": f"arch-{i}"}
            for i in range(64)
        ],
    }


def test_selection_rejects_wrong_hash_and_duplicate_or_target(tmp_path):
    path = tmp_path / "selection.json"
    raw = json.dumps(_selection()).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash"):
        load_frozen_selection(path, "0" * 64)

    parsed = load_frozen_selection(path, _sha(raw))
    assert len(parsed["architectures"]) == 64

    bad = _selection()
    bad["architectures"][1]["arch_id"] = "0"
    bad["architectures"][1]["canonical_id"] = "0"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_selection(path, _sha(path.read_bytes()))

    bad = _selection()
    bad["architectures"][0]["oracle_target"] = 0.7
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="target"):
        load_frozen_selection(path, _sha(path.read_bytes()))


def test_training_plan_pins_old_checkpoint_protocol_without_targets():
    plan = expected_training_plan(_selection(), selection_sha="a" * 64,
                                  trainer_sha="b" * 64, attack_sha="c" * 64,
                                  meta_node_sha="d" * 64)
    assert plan["expected_checkpoint_keys"] == 320
    assert plan["training"]["seed"] == 0
    assert plan["training"]["train_pgd_steps"] == 7
    assert plan["training"]["train_samples"] == 50000
    assert plan["training"]["batch_size"] == 256
    assert plan["training"]["epochs"] == 50
    assert "target" not in json.dumps(plan).lower()


def test_plan_is_exclusive_and_rejects_changed_run(tmp_path):
    plan = expected_training_plan(_selection(), selection_sha="a" * 64,
                                  trainer_sha="b" * 64, attack_sha="c" * 64,
                                  meta_node_sha="d" * 64)
    path = tmp_path / "plan.json"
    assert write_or_verify_plan(path, plan) == "created"
    assert write_or_verify_plan(path, plan) == "verified"
    changed = json.loads(json.dumps(plan))
    changed["training"]["seed"] = 1
    with pytest.raises(ValueError, match="frozen"):
        write_or_verify_plan(path, changed)


def test_complete_receipt_skips_only_five_matching_checkpoints(tmp_path):
    arch_dir = tmp_path / "arch_7"
    assert not _verified_complete(arch_dir, "7", "a" * 64)
    attempt = arch_dir / "attempt_first"
    attempt.mkdir(parents=True)
    checkpoints = {}
    for epoch in (0, 5, 10, 25, 50):
        path = attempt / f"arch_7_epoch_{epoch}_seed_0.pt"
        path.write_bytes(f"checkpoint {epoch}".encode())
        checkpoints[str(epoch)] = _sha(path.read_bytes())
    (attempt / "complete.json").write_text(json.dumps({
        "arch_id": "7", "frozen_training_plan_sha256": "a" * 64,
        "checkpoints": checkpoints,
    }))
    assert _verified_complete(arch_dir, "7", "a" * 64)
    with pytest.raises(ValueError, match="plan"):
        _verified_complete(arch_dir, "7", "b" * 64)
    (attempt / "arch_7_epoch_25_seed_0.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        _verified_complete(arch_dir, "7", "a" * 64)
