"""Check independent target evidence before a holdout package can be sealed.

The DARTS protocol matches the project's full-test PGD-20 implementation:
``evaluate_darts_checkpoints.py`` calls ``pgd_accuracy`` with the default
``pgd_linf`` step size, random start and one attack run per test image.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CIFAR10_TEST_IDS_SHA256 = hashlib.sha256(
    json.dumps(list(range(10000)), separators=(",", ":")).encode("utf-8")
).hexdigest()
_DARTS_NORMALIZATION = "cifar10_125.3_123.0_113.9__63.0_62.1_66.7"


def verify_target_item(item: dict) -> None:
    """Reject a DARTS label from a different weight, subset or attack setup.

    File existence, file digest and the actual PGD outputs are audited by the
    package-level verifier; metadata passing this function is not that proof.
    """
    if item.get("setting") not in {"darts_e0", "darts_e100"}:
        raise ValueError("DARTS setting required")
    for field in ("checkpoint_sha256", "final_checkpoint_sha256", "target_source_sha256"):
        if not isinstance(item.get(field), str) or not _SHA256.fullmatch(item[field]):
            raise ValueError(f"{field}: source SHA or checkpoint SHA required")
    if item.get("target_checkpoint_sha256") != item["final_checkpoint_sha256"]:
        raise ValueError("target checkpoint mismatch")
    if item["setting"] == "darts_e100" and item["checkpoint_sha256"] != item["final_checkpoint_sha256"]:
        raise ValueError("final checkpoint mismatch")
    value = item.get("target_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite target accuracy in [0,1] required")
    if not str(item.get("target_source_id", "")).strip():
        raise ValueError("target source ID required")
    if item.get("target_attack") != "pgd20":
        raise ValueError("PGD-20 attack required")
    if item.get("target_dataset") != "cifar10_test":
        raise ValueError("CIFAR-10 test dataset required")
    if item.get("target_sample_count") != 10000:
        raise ValueError("full 10,000-image target required")
    if item.get("target_sample_ids_sha256") != _CIFAR10_TEST_IDS_SHA256:
        raise ValueError("CIFAR-10 test sample IDs mismatch")
    if item.get("target_epsilon") != 8 / 255:
        raise ValueError("PGD epsilon mismatch")
    if item.get("target_step_size") != 2 / 255:
        raise ValueError("PGD step size mismatch")
    if item.get("target_steps") != 20:
        raise ValueError("PGD steps mismatch")
    if item.get("target_restarts") != 1:
        raise ValueError("PGD restarts mismatch")
    if item.get("target_random_start") is not True:
        raise ValueError("PGD random start mismatch")
    if item.get("target_batch_size") != 128:
        raise ValueError("PGD batch size mismatch")
    if item.get("target_random_seed") != 400000:
        raise ValueError("PGD seed mismatch")
    if item.get("target_normalization") != _DARTS_NORMALIZATION:
        raise ValueError("DARTS normalization mismatch")
