"""A DARTS holdout target must match the frozen full-test PGD-20 protocol."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.verify_holdout_targets import verify_target_item


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
