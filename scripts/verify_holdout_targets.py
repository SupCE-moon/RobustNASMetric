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
from pathlib import Path


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CIFAR10_TEST_IDS_SHA256 = hashlib.sha256(
    json.dumps(list(range(10000)), separators=(",", ":")).encode("utf-8")
).hexdigest()
_DARTS_NORMALIZATION = "cifar10_125.3_123.0_113.9__63.0_62.1_66.7"
_NB201_SOURCE_ID = "official_nasrobbench201_cifar10_autoattack"


def load_official_nb201_sources(benchmark_path: Path, meta_path: Path,
                                benchmark_sha256: str, meta_sha256: str) -> tuple[dict, dict]:
    """Load only the exact official source bytes frozen before holdout scoring."""
    def load_one(path: Path, expected: str, name: str) -> dict:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{name} SHA mismatch")
        def no_duplicate_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{name} duplicate JSON field: {key}")
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=no_duplicate_fields,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(value, dict):
            raise ValueError(f"{name} JSON object required")
        return value
    benchmark = load_one(benchmark_path, benchmark_sha256, "official results")
    meta = load_one(meta_path, meta_sha256, "official meta")
    if not isinstance(benchmark.get("cifar10"), dict) or not isinstance(meta.get("ids"), dict):
        raise ValueError("official NB201 source schema invalid")
    return benchmark, meta


def verify_nb201_target_item(item: dict, benchmark: dict, meta: dict,
                              benchmark_sha256: str) -> float:
    """Recompute a single NB201 target from its official canonical record."""
    if item.get("setting") not in {f"nb201_e{x}" for x in (0, 5, 10, 25, 50)}:
        raise ValueError("NB201 setting invalid")
    identifier = item.get("model_or_arch_id")
    if not isinstance(identifier, str) or not identifier.isdecimal():
        raise ValueError("NB201 architecture ID invalid")
    metadata = meta["ids"].get(identifier)
    record = benchmark["cifar10"].get(identifier)
    if not isinstance(metadata, dict) or not isinstance(record, dict):
        raise ValueError("official NB201 architecture missing")
    if str(metadata.get("isomorph")) != identifier:
        raise ValueError("NB201 architecture is noncanonical")
    if item.get("signature") != metadata.get("nb201-string"):
        raise ValueError("NB201 structure differs from official meta")
    if item.get("target_source_id") != _NB201_SOURCE_ID:
        raise ValueError("NB201 official target source ID mismatch")
    if item.get("target_source_sha256") != benchmark_sha256:
        raise ValueError("NB201 official source SHA mismatch")
    actual = record.get("autoattack")
    expected = item.get("target_value")
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual) or not 0 <= actual <= 1):
        raise ValueError("NB201 official accuracy invalid")
    if (isinstance(expected, bool) or not isinstance(expected, (int, float))
            or not math.isfinite(expected)
            or not math.isclose(float(expected), float(actual), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("NB201 target differs from official accuracy")
    return float(actual)


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


def verify_target_file(item: dict, project_root: Path) -> float:
    """Recompute a DARTS accuracy from immutable per-image attack evidence.

    The root must be the permitted project. This checks file bytes and internal
    consistency; it does not replace a separate code/weight execution audit.
    """
    verify_target_item(item)
    root = Path(project_root).resolve()
    relative = item.get("target_source_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("target source path must be project-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("target source file missing or outside project")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["target_source_sha256"]:
        raise ValueError("target source SHA mismatch")
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("target source JSON invalid") from exc
    if record.get("status") != "ok" or str(record.get("model_or_arch_id")) != item["model_or_arch_id"]:
        raise ValueError("target evidence status or architecture mismatch")
    if record.get("checkpoint_sha256") != item["final_checkpoint_sha256"]:
        raise ValueError("target evidence checkpoint mismatch")
    protocol_fields = (
        "target_attack", "target_dataset", "target_epsilon", "target_step_size",
        "target_steps", "target_restarts", "target_random_start", "target_batch_size",
        "target_random_seed", "target_normalization",
    )
    if record.get("protocol") != {key: item[key] for key in protocol_fields}:
        raise ValueError("target evidence protocol mismatch")
    ids = record.get("sample_ids")
    correct = record.get("correct")
    if not isinstance(ids, list) or not isinstance(correct, list) or len(ids) != 10000 or len(correct) != 10000:
        raise ValueError("target evidence requires 10,000 ordered images")
    if ids != list(range(10000)) or not all(type(value) is bool for value in correct):
        raise ValueError("target evidence sample IDs or correctness invalid")
    accuracy = sum(correct) / 10000
    if not math.isclose(accuracy, float(item["target_value"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("target accuracy differs from per-image evidence")
    return accuracy
