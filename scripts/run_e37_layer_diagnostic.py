"""Run frozen E37 layer diagnostics into immutable, key-specific attempts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.e37_diagnostic_manifest import (  # noqa: E402
    PINNED_SOURCES, expected_darts_keys, sha256_file,
)
from robust_metric_lab.e37_layer_diagnostic import diagnose_e37  # noqa: E402
from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON, load_darts_items,
)
from robust_metric_lab.student_runner import load_cifar10_samples  # noqa: E402


def write_attempt(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_repair_keys(
    requested: set[tuple[str, str]],
    expected: set[tuple[str, str]],
    successful: set[tuple[str, str]],
) -> None:
    if not requested <= expected:
        raise ValueError(f"unknown repair keys: {sorted(requested - expected)}")
    if requested & successful:
        raise ValueError(f"previously successful repair keys: {sorted(requested & successful)}")
    if requested != expected - successful:
        raise ValueError("repair keys must equal all missing/failed keys")


def select_smoke_keys(
    two_ids: str, expected: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    identifiers = two_ids.split(",")
    if len(identifiers) != 2 or len(set(identifiers)) != 2 or any(
        not identifier or identifier != identifier.strip() for identifier in identifiers
    ):
        raise ValueError("smoke requires two distinct canonical architecture IDs")
    keys = {(f"darts_e{stage}", identifier) for identifier in identifiers for stage in (0, 100)}
    if not keys <= expected:
        raise ValueError(f"unknown smoke keys: {sorted(keys - expected)}")
    return keys


def attempts_root_for_mode(output_root: Path, smoke: bool) -> Path:
    return output_root / ("smoke_attempts" if smoke else "attempts")


def _key(record: dict) -> tuple[str, str]:
    return str(record["setting"]), str(record["model_or_arch_id"])


def read_previous_attempts(attempts_root: Path) -> list[dict]:
    records: list[dict] = []
    if not attempts_root.exists():
        return records
    for path in sorted(attempts_root.glob("*/*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") not in {"ok", "failed"}:
            raise ValueError(f"invalid prior attempt status: {path}")
        records.append(record)
    return records


def _read_repair_keys(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"setting", "model_or_arch_id"}:
            raise ValueError("repair CSV must have exactly setting,model_or_arch_id")
        rows = [(str(row["setting"]), str(row["model_or_arch_id"])) for row in reader]
    if len(rows) != len(set(rows)):
        raise ValueError("duplicate repair key")
    return set(rows)


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def load_frozen_manifest(path: Path, bundle_root: Path) -> tuple[dict, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["frozen_payload"]
    frozen_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if frozen_hash != document.get("frozen_sha256"):
        raise ValueError("frozen manifest content/hash mismatch")
    if payload.get("expected_rows") != 200 or len(payload.get("objects", [])) != 200:
        raise ValueError("frozen manifest must contain 200 objects")
    expected_darts_keys([type("Key", (), {
        "model_or_arch_id": row["model_or_arch_id"],
        "checkpoint_epoch": row["checkpoint_epoch"],
    }) for row in payload["objects"]])
    actual_sources = {name: sha256_file(ROOT / name) for name in PINNED_SOURCES}
    if actual_sources != payload["source_sha256"]:
        raise ValueError("pinned E37 source mismatch")
    for label, hash_field in (
        ("original_e37_manifest", "original_e37_manifest_sha256"),
        ("strict_csv", "strict_csv_sha256"),
        ("control_csv", "control_csv_sha256"),
    ):
        if sha256_file(Path(payload["input_paths"][label])) != payload[hash_field]:
            raise ValueError(f"frozen {label} mismatch")
    bundle_manifest = bundle_root / "manifests/darts100_architectures.csv"
    if bundle_manifest.resolve() != Path(payload["input_paths"]["bundle_manifest"]).resolve():
        raise ValueError("DARTS bundle manifest path mismatch")
    if sha256_file(bundle_manifest) != payload["bundle_manifest_sha256"]:
        raise ValueError("DARTS bundle manifest mismatch")
    items = load_darts_items(bundle_manifest, bundle_root)
    expected_darts_keys(items)
    observed = {
        (f"darts_e{item.checkpoint_epoch}", str(item.model_or_arch_id)):
        (str(item.checkpoint_path.resolve()), float(item.target_value))
        for item in items
    }
    frozen = {
        _key(row): (str(Path(row["checkpoint_path"]).resolve()), float(row["target_value"]))
        for row in payload["objects"]
    }
    if observed != frozen:
        raise ValueError("DARTS frozen key/path/target mismatch")
    return payload, frozen_hash


def run_one(
    item: object,
    frozen_object: dict,
    *,
    loader: Callable,
    samples: tuple,
    diagnose: Callable,
    manifest_hash: str,
    source_hashes: dict,
    diagnostic_code_hashes: dict,
    epsilon: float,
    device: str = "cpu",
) -> dict:
    import torch

    setting = f"darts_e{item.checkpoint_epoch}"
    if (setting, str(item.model_or_arch_id)) != _key(frozen_object):
        raise ValueError("item/frozen key mismatch")
    started = time.monotonic()
    record = {
        "setting": setting,
        "model_or_arch_id": str(item.model_or_arch_id),
        "manifest_sha256": manifest_hash,
        "source_sha256": source_hashes,
        "diagnostic_code_sha256": diagnostic_code_hashes,
        "checkpoint_sha256": frozen_object["checkpoint_sha256"],
        "protocol": {"epsilon": epsilon, "sample_count": 64, "batch_size": 1},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if sha256_file(Path(frozen_object["checkpoint_path"])) != frozen_object["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = loader(item)
        diagnostic = diagnose(model, samples[0], samples[1], epsilon=epsilon)
        if diagnostic.get("sample_count") != 64 or len(diagnostic.get("anchor_metadata", [])) != 4:
            raise ValueError("diagnostic did not return 64 samples/four anchors")
        record["status"] = "ok"
        record["diagnostic"] = diagnostic
    except Exception as exc:
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    record["wall_seconds"] = time.monotonic() - started
    if str(device).startswith("cuda") and torch.cuda.is_available():
        record["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--repair-keys", type=Path)
    parser.add_argument("--smoke-ids", help="Exactly two predeclared IDs, e.g. 0,1; isolated from formal attempts")
    return parser


def main(argv: list[str] | None = None) -> int:
    import torch
    from scripts.evaluate_darts100_student import make_model_loader

    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.attempt_id):
        raise ValueError("attempt-id must be a safe single directory name")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ROOT):
        raise ValueError("output-root must stay within this RobustMetric project")
    payload, manifest_hash = load_frozen_manifest(args.frozen_manifest, args.bundle_root.resolve())
    items = load_darts_items(
        args.bundle_root / "manifests/darts100_architectures.csv", args.bundle_root
    )
    objects = {_key(row): row for row in payload["objects"]}
    expected = set(objects)
    smoke = args.smoke_ids is not None
    if smoke and args.repair_keys is not None:
        raise ValueError("smoke and repair modes are mutually exclusive")
    attempts_root = attempts_root_for_mode(output_root, smoke)
    previous = read_previous_attempts(attempts_root)
    successful = {_key(row) for row in previous if row["status"] == "ok"}
    if len(successful) != sum(row["status"] == "ok" for row in previous):
        raise ValueError("duplicate previous successful attempt")
    if smoke:
        if previous:
            raise ValueError("prior smoke attempts exist; use a new diagnostic run directory")
        requested = select_smoke_keys(args.smoke_ids, expected)
    elif args.repair_keys is None:
        if previous:
            raise ValueError("prior attempts exist; use exact repair-keys CSV")
        requested = expected
    else:
        requested = _read_repair_keys(args.repair_keys)
        validate_repair_keys(requested, expected, successful)
    attempt_root = attempts_root / args.attempt_id
    if attempt_root.exists():
        raise FileExistsError(f"attempt ID already exists: {args.attempt_id}")
    attempt_root.mkdir(parents=True, exist_ok=False)
    log_path = attempt_root / "launcher.log"
    loader = make_model_loader(args.device)
    try:
        samples = load_cifar10_samples(args.data_root, args.device)
    except Exception:
        write_attempt(attempt_root / "input_failure.json", {
            "status": "failed", "error_type": "InputLoadFailure", "traceback": traceback.format_exc(),
            "manifest_sha256": manifest_hash,
        })
        raise
    failures = 0
    diagnostic_code_hashes = {
        "collector": sha256_file(ROOT / "src/robust_metric_lab/e37_layer_diagnostic.py"),
        "runner": sha256_file(Path(__file__)),
    }
    with log_path.open("x", encoding="utf-8") as log:
        for item in sorted(items, key=lambda row: (row.checkpoint_epoch, str(row.model_or_arch_id))):
            key = (f"darts_e{item.checkpoint_epoch}", str(item.model_or_arch_id))
            if key not in requested:
                continue
            record = run_one(
                item, objects[key], loader=loader, samples=samples, diagnose=diagnose_e37,
                manifest_hash=manifest_hash, source_hashes=payload["source_sha256"],
                diagnostic_code_hashes=diagnostic_code_hashes,
                epsilon=EPSILON, device=args.device,
            )
            write_attempt(attempt_root / key[0] / f"{key[1]}.json", record)
            log.write(f"{key[0]},{key[1]},{record['status']}\n")
            log.flush()
            if record["status"] != "ok":
                failures += 1
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    print(f"attempt_id={args.attempt_id} selected={len(requested)} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
