"""Check independent target evidence before a holdout package can be sealed.

The DARTS protocol matches the project's full-test PGD-20 implementation:
``evaluate_darts_checkpoints.py`` calls ``pgd_accuracy`` with the default
``pgd_linf`` step size, random start and one attack run per test image.
"""

from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import re
from pathlib import PurePosixPath
from collections import Counter
from pathlib import Path


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CIFAR10_TEST_IDS_SHA256 = hashlib.sha256(
    json.dumps(list(range(10000)), separators=(",", ":")).encode("utf-8")
).hexdigest()
_DARTS_NORMALIZATION = "cifar10_125.3_123.0_113.9__63.0_62.1_66.7"
_NB201_SOURCE_ID = "official_nasrobbench201_cifar10_autoattack"
_RB11_SOURCE_ID = "frozen_rb11_manifest_20260922"


def verify_checkpoint_files(rows: list[dict], project_root: Path) -> dict:
    """Read each referenced checkpoint under the permitted project and hash its bytes."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("checkpoint project root missing")
    seen_keys = set()
    paths = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("checkpoint row invalid")
        key = (row.get("setting"), row.get("model_or_arch_id"))
        if not all(isinstance(part, str) and part for part in key):
            raise ValueError("checkpoint key invalid")
        if key in seen_keys:
            raise ValueError("duplicate checkpoint key")
        seen_keys.add(key)
        relative = row.get("checkpoint_path")
        if (not isinstance(relative, str) or not relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or "\\" in relative):
            raise ValueError("checkpoint path must be project-relative")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("checkpoint file missing or outside project")
        digest = row.get("checkpoint_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("checkpoint SHA invalid")
        if path in paths and paths[path] != digest:
            raise ValueError("checkpoint path assigned conflicting SHA")
        paths[path] = digest
    for path, expected in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"checkpoint SHA mismatch: {path.relative_to(root)}")
    return {"kind": "holdout_checkpoint_file_verification", "status": "PASS",
            "verified_rows": len(rows), "verified_files": len(paths)}


def load_rb11_recovery_report(path: Path, expected_sha256: str) -> dict:
    """Bind the one-file RB11 repair claim to its recorded JSON bytes."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("RB11 recovery SHA mismatch")
    def no_duplicate_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate RB11 recovery field: {key}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=no_duplicate_fields,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, dict):
        raise ValueError("RB11 recovery report object required")
    return value


def load_frozen_rb11_manifest(path: Path, expected_sha256: str) -> dict[str, dict]:
    """Read the frozen 11-model target list only after its bytes match."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen RB11 manifest SHA mismatch")
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"model_name", "robust_accuracy"} <= set(reader.fieldnames or []):
            raise ValueError("frozen RB11 manifest columns missing")
        rows = list(reader)
    if len(rows) != 11:
        raise ValueError("frozen RB11 manifest requires 11 models")
    names = [row["model_name"] for row in rows]
    if any(not name or name != name.strip() for name in names) or len(set(names)) != 11:
        raise ValueError("frozen RB11 model names invalid or duplicate")
    result = {}
    for row in rows:
        try:
            value = float(row["robust_accuracy"])
        except (TypeError, ValueError) as exc:
            raise ValueError("frozen RB11 target accuracy invalid") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("frozen RB11 target accuracy invalid")
        result[row["model_name"]] = {**row, "robust_accuracy": value}
    return result


def verify_rb11_target_rows(rows: list[dict], manifest: dict[str, dict],
                             manifest_sha256: str, recovery: dict) -> dict:
    """Check all 11 target labels against the frozen overlay's base manifest."""
    if len(rows) != 11 or len(manifest) != 11:
        raise ValueError("RB11 holdout requires exactly 11 target rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("RB11 holdout row invalid")
    names = [row.get("model_or_arch_id") for row in rows]
    if not all(isinstance(name, str) for name in names) or len(set(names)) != 11:
        raise ValueError("RB11 holdout duplicate or invalid model ID")
    if set(names) != set(manifest):
        raise ValueError("RB11 holdout differs from frozen model list")
    overlay = recovery.get("frozen_overlay", {})
    load_audit = recovery.get("fresh_container_load_audit", {})
    if (overlay.get("base_manifest_sha256") != manifest_sha256
            or overlay.get("expected_unique_models") != 11
            or load_audit.get("status") != "PASS"
            or load_audit.get("loadable_unique_models") != 11):
        raise ValueError("RB11 recovery does not match frozen 11-model manifest")
    replacement = recovery.get("model_name")
    replacement_sha = recovery.get("replacement_weight", {}).get("sha256")
    if replacement not in manifest or not isinstance(replacement_sha, str) or not _SHA256.fullmatch(replacement_sha):
        raise ValueError("RB11 replacement mapping invalid")
    for row in rows:
        name = row["model_or_arch_id"]
        if row.get("setting") != "rb40_mature":
            raise ValueError("RB11 holdout setting invalid")
        if (row.get("target_source_id") != _RB11_SOURCE_ID
                or row.get("target_source_sha256") != manifest_sha256):
            raise ValueError("RB11 frozen target source mismatch")
        value = row.get("target_value")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(float(value), manifest[name]["robust_accuracy"], rel_tol=0, abs_tol=1e-12)):
            raise ValueError("RB11 target differs from frozen manifest")
        digest = row.get("checkpoint_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("RB11 checkpoint SHA invalid")
        if name == replacement and digest != replacement_sha:
            raise ValueError("RB11 replacement checkpoint SHA mismatch")
    key_bytes = json.dumps(sorted(names), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"kind": "rb11_frozen_target_verification", "status": "PASS",
            "verified_rows": 11, "verified_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
            "frozen_manifest_sha256": manifest_sha256,
            "replacement_checkpoint_sha256": replacement_sha}


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


def verify_nb201_target_rows(rows: list[dict], benchmark: dict, meta: dict,
                              benchmark_sha256: str) -> dict:
    """Verify the exact 64-architecture × five-stage official target grid."""
    epochs = (0, 5, 10, 25, 50)
    settings = {f"nb201_e{epoch}" for epoch in epochs}
    if len(rows) != 320:
        raise ValueError("NB201 holdout requires exactly 320 target rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("NB201 holdout target row must be an object")
    if not all(isinstance(row.get("model_or_arch_id"), str)
               and row["model_or_arch_id"].isdecimal() for row in rows):
        raise ValueError("NB201 architecture ID invalid")
    keys = [(row.get("setting"), row.get("model_or_arch_id")) for row in rows]
    if len(set(keys)) != 320:
        raise ValueError("duplicate NB201 holdout target key")
    counts = Counter(row.get("setting") for row in rows if isinstance(row, dict))
    if set(counts) != settings or any(counts[setting] != 64 for setting in settings):
        raise ValueError("NB201 holdout requires 64 targets in each of five stages")
    ids_by_setting = {
        setting: {identifier for row_setting, identifier in keys if row_setting == setting}
        for setting in settings
    }
    base_ids = ids_by_setting["nb201_e0"]
    if any(ids != base_ids for ids in ids_by_setting.values()):
        raise ValueError("NB201 architecture IDs differ across stages")
    signatures = {row["model_or_arch_id"]: row.get("signature")
                  for row in rows if row["setting"] == "nb201_e0"}
    if len(set(signatures.values())) != 64:
        raise ValueError("duplicate NB201 holdout structure")
    for row in rows:
        verify_nb201_target_item(row, benchmark, meta, benchmark_sha256)
    key_bytes = json.dumps(sorted(keys), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "kind": "nb201_official_target_verification",
        "status": "PASS",
        "verified_rows": 320,
        "verified_architectures": 64,
        "verified_counts": {setting: 64 for setting in sorted(settings)},
        "verified_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "official_source_sha256": benchmark_sha256,
    }


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
    def no_duplicate_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate target source JSON field: {key}")
            result[key] = value
        return result
    try:
        record = json.loads(
            raw,
            object_pairs_hook=no_duplicate_fields,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"target source JSON invalid: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("target source JSON object required")
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


def verify_darts_target_rows(rows: list[dict], project_root: Path) -> dict:
    """Verify 20 final-weight targets shared across both DARTS stages."""
    if len(rows) != 40:
        raise ValueError("DARTS holdout requires exactly 40 target rows")
    if not all(isinstance(row, dict) and isinstance(row.get("model_or_arch_id"), str)
               for row in rows):
        raise ValueError("DARTS target row or architecture ID invalid")
    keys = [(row.get("setting"), row["model_or_arch_id"]) for row in rows]
    if len(set(keys)) != 40:
        raise ValueError("duplicate DARTS target key")
    counts = Counter(setting for setting, _ in keys)
    if counts != {"darts_e0": 20, "darts_e100": 20}:
        raise ValueError("DARTS holdout requires 20 targets at each stage")
    by_key = {(row["setting"], row["model_or_arch_id"]): row for row in rows}
    first_ids = {identifier for setting, identifier in keys if setting == "darts_e0"}
    final_ids = {identifier for setting, identifier in keys if setting == "darts_e100"}
    if first_ids != final_ids:
        raise ValueError("DARTS architecture IDs differ across stages")
    source_hashes = []
    for identifier in sorted(first_ids):
        early = by_key[("darts_e0", identifier)]
        final = by_key[("darts_e100", identifier)]
        verify_target_item(early)
        for field in ("target_value", "target_source_id", "target_source_path",
                      "target_source_sha256", "target_checkpoint_sha256",
                      "final_checkpoint_sha256"):
            if early.get(field) != final.get(field):
                raise ValueError(f"DARTS stage target {field} mismatch: {identifier}")
        verify_target_file(final, project_root)
        source_hashes.append(final["target_source_sha256"])
    hash_bytes = json.dumps(source_hashes, separators=(",", ":")).encode("utf-8")
    return {"kind": "darts_fulltest_target_verification", "status": "PASS",
            "verified_rows": 40, "verified_architectures": 20,
            "verified_counts": {"darts_e0": 20, "darts_e100": 20},
            "target_files_sha256": hashlib.sha256(hash_bytes).hexdigest()}
