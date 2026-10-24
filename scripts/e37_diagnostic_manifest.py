"""Freeze the exact DARTS/E37 diagnostic inputs before GPU execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PINNED_SOURCES = (
    "src/robust_metric_lab/e37_metrics.py",
    "src/robust_metric_lab/e37_batched.py",
    "scripts/evaluate_e37_student.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_darts_keys(items: Iterable[object]) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    stages: dict[int, set[str]] = {0: set(), 100: set()}
    for item in items:
        epoch = int(item.checkpoint_epoch)
        if epoch not in stages:
            raise ValueError(f"unsupported DARTS stage: {epoch}")
        identifier = str(item.model_or_arch_id)
        if not identifier or identifier != identifier.strip():
            raise ValueError("DARTS architecture ID must be canonical")
        key = (identifier, epoch)
        if key in seen:
            raise ValueError(f"duplicate DARTS key: {key}")
        seen.add(key)
        stages[epoch].add(identifier)
    if len(stages[0]) != 100 or len(stages[100]) != 100 or stages[0] != stages[100]:
        raise ValueError("missing or unequal DARTS stage keys")
    return seen


def _canonical(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def write_or_verify_manifest(path: Path, payload: dict) -> str:
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_payload = existing.get("frozen_payload")
        if (
            existing.get("frozen_sha256") != digest
            or not isinstance(old_payload, dict)
            or hashlib.sha256(_canonical(old_payload)).hexdigest() != digest
        ):
            raise ValueError("manifest mismatch: frozen inputs differ")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_sha256": digest,
        "frozen_payload": payload,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return write_or_verify_manifest(path, payload)
    return digest


def build_payload(
    items: Iterable[object],
    *,
    source_paths: Mapping[str, Path],
    pinned_source_sha256: Mapping[str, str],
    original_manifest: Path,
    bundle_manifest: Path,
    strict_csv: Path,
    control_csv: Path,
    environment: Mapping[str, str],
) -> dict:
    item_list = list(items)
    expected_darts_keys(item_list)
    if set(source_paths) != set(pinned_source_sha256):
        raise ValueError("pinned source set mismatch")
    source_hashes: dict[str, str] = {}
    for name, path in source_paths.items():
        actual = sha256_file(Path(path))
        if actual != pinned_source_sha256[name]:
            raise ValueError(f"pinned source mismatch: {name}")
        source_hashes[name] = actual

    checkpoints: dict[str, str] = {}
    objects: list[dict] = []
    for item in sorted(item_list, key=lambda it: (int(it.checkpoint_epoch), str(it.model_or_arch_id))):
        path = Path(item.checkpoint_path).resolve()
        path_key = str(path)
        if path_key not in checkpoints:
            checkpoints[path_key] = sha256_file(path)
        target = float(item.target_value)
        if not math.isfinite(target):
            raise ValueError(f"nonfinite target for {item.model_or_arch_id}")
        objects.append({
            "setting": f"darts_e{int(item.checkpoint_epoch)}",
            "model_or_arch_id": str(item.model_or_arch_id),
            "checkpoint_epoch": int(item.checkpoint_epoch),
            "checkpoint_path": path_key,
            "checkpoint_sha256": checkpoints[path_key],
            "target_value": target,
        })
    return {
        "diagnostic_name": "darts_e37_layer_diagnostic",
        "expected_rows": 200,
        "objects": objects,
        "protocol": {
            "benchmark": "darts",
            "stages": [0, 100],
            "seed": 20260808,
            "sample_offset": 0,
            "num_samples": 64,
            "batch_size": 1,
            "attack": "fgsm",
            "epsilon": 8 / 255,
            "activation_mode": "all_leaf",
            "max_layers": 4,
        },
        "source_sha256": source_hashes,
        "original_e37_manifest_sha256": sha256_file(original_manifest),
        "bundle_manifest_sha256": sha256_file(bundle_manifest),
        "strict_csv_sha256": sha256_file(strict_csv),
        "control_csv_sha256": sha256_file(control_csv),
        "environment": dict(environment),
    }


def pinned_source_hashes(original_manifest: Path) -> dict[str, str]:
    document = json.loads(original_manifest.read_text(encoding="utf-8"))
    candidates = [document]
    if isinstance(document, dict) and isinstance(document.get("frozen_payload"), dict):
        candidates.append(document["frozen_payload"])
    for candidate in candidates:
        hashes = candidate.get("sha256") or candidate.get("source_sha256") or candidate.get("source_hashes")
        if not isinstance(hashes, dict):
            continue
        result = {name: str(hashes[name]) for name in PINNED_SOURCES if name in hashes}
        if len(result) == len(PINNED_SOURCES):
            return result
    raise ValueError("original E37 manifest does not pin all three source hashes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--strict-release", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--control-csv", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from robust_metric_lab.student_benchmark import load_darts_items

    project = args.project.resolve()
    bundle = args.bundle_root.resolve()
    bundle_manifest = bundle / "manifests/darts100_architectures.csv"
    items = load_darts_items(bundle_manifest, bundle)
    pinned = pinned_source_hashes(args.original_manifest)
    payload = build_payload(
        items,
        source_paths={name: project / name for name in PINNED_SOURCES},
        pinned_source_sha256=pinned,
        original_manifest=args.original_manifest,
        bundle_manifest=bundle_manifest,
        strict_csv=args.strict_release / "metric_values_merged.csv",
        control_csv=args.control_csv,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": str(torch.version.cuda),
        },
    )
    digest = write_or_verify_manifest(args.run_dir / "frozen_manifest.json", payload)
    print(f"frozen_manifest_sha256={digest} expected_keys={payload['expected_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
