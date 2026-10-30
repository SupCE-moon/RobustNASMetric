"""Provenance checks for the preregistered activation-survival run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


SETTING_COUNTS = {
    "rb40_mature": 40,
    **{f"nb201_e{epoch}": 64 for epoch in (0, 5, 10, 25, 50)},
    **{f"darts_e{epoch}": 100 for epoch in (0, 100)},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_project_path(path: Path, project_root: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(project_root).resolve()):
        raise ValueError(f"path outside permitted project: {resolved}")
    return resolved


def verify_source_inventory(inventory: dict[str, str], root: Path) -> None:
    """Refuse a missing or altered audited source file."""
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source mismatch: {name}")


def expected_main_keys(
    items_by_setting: Mapping[str, Iterable[object]],
) -> set[tuple[str, str, int | None]]:
    """Demand the complete eight-setting grid with phase-aligned IDs and targets."""
    if set(items_by_setting) != set(SETTING_COUNTS):
        raise ValueError("missing or unknown setting")
    keys: set[tuple[str, str, int | None]] = set()
    identifiers: dict[str, set[str]] = {}
    targets: dict[tuple[str, str], float] = {}
    for setting, expected_count in SETTING_COUNTS.items():
        items = list(items_by_setting[setting])
        if len(items) != expected_count:
            raise ValueError(f"missing setting count: {setting}")
        expected_epoch = None if setting == "rb40_mature" else int(setting.split("_e")[-1])
        space = setting.split("_")[0]
        seen: set[str] = set()
        for item in items:
            identifier = str(item.model_or_arch_id)
            if not identifier or identifier != identifier.strip():
                raise ValueError("noncanonical model/architecture ID")
            if item.checkpoint_epoch != expected_epoch:
                raise ValueError(f"stage mismatch: {setting}/{identifier}")
            if identifier in seen:
                raise ValueError(f"duplicate ID: {setting}/{identifier}")
            seen.add(identifier)
            target = float(item.target_value)
            if not math.isfinite(target):
                raise ValueError(f"nonfinite target: {setting}/{identifier}")
            target_key = (space, identifier)
            if target_key in targets and targets[target_key] != target:
                raise ValueError(f"target differs across stages: {identifier}")
            targets[target_key] = target
            keys.add((setting, identifier, expected_epoch))
        identifiers[setting] = seen
    for space, stages in {
        "nb201": (0, 5, 10, 25, 50), "darts": (0, 100)
    }.items():
        first = identifiers[f"{space}_e{stages[0]}"]
        if any(identifiers[f"{space}_e{epoch}"] != first for epoch in stages[1:]):
            raise ValueError(f"unequal {space} stage IDs")
    if len(keys) != 560:
        raise ValueError("missing exact 560 keys")
    return keys


def _canonical(payload: Mapping) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def write_or_verify_manifest(path: Path, payload: dict) -> str:
    """Write frozen inputs once; refuse mutation of either payload or digest."""
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
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_payload": payload,
        "frozen_sha256": digest,
    }
    serialized = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return write_or_verify_manifest(path, payload)
    return digest


def build_payload(
    items_by_setting: Mapping[str, Iterable[object]],
    *,
    source_paths: Mapping[str, Path],
    bundle_manifests: Mapping[str, Path],
    sample_paths: Mapping[str, Path],
    strict_csv: Path,
    environment: Mapping[str, str],
) -> dict:
    """Hash the complete candidate input bundle before any model evaluation."""
    items_by_setting = {setting: list(items) for setting, items in items_by_setting.items()}
    expected_main_keys(items_by_setting)
    if set(bundle_manifests) != {"rb40", "nb201", "darts"} or not sample_paths or not source_paths:
        raise ValueError("missing source, bundle or sample inputs")
    source_sha256 = {name: sha256_file(Path(path)) for name, path in sorted(source_paths.items())}
    bundle_sha256 = {name: sha256_file(Path(path)) for name, path in sorted(bundle_manifests.items())}
    sample_sha256 = {name: sha256_file(Path(path)) for name, path in sorted(sample_paths.items())}
    strict_sha256 = sha256_file(Path(strict_csv))
    checkpoint_cache: dict[str, str] = {}

    def checkpoint_hash(path: Path) -> str:
        absolute = str(Path(path).resolve())
        if absolute not in checkpoint_cache:
            checkpoint_cache[absolute] = sha256_file(Path(path))
        return checkpoint_cache[absolute]

    objects: list[dict] = []
    for setting in SETTING_COUNTS:
        for item in sorted(items_by_setting[setting], key=lambda value: str(value.model_or_arch_id)):
            checkpoint = Path(item.checkpoint_path).resolve()
            metadata = dict(getattr(item, "metadata", {}))
            artifact_paths = [Path(value).resolve() for value in metadata.get("artifact_paths", [str(checkpoint)])]
            if checkpoint not in artifact_paths:
                raise ValueError("primary checkpoint absent from artifact list")
            artifact_hashes = {str(path): checkpoint_hash(path) for path in artifact_paths}
            objects.append({
                "setting": setting,
                "model_or_arch_id": str(item.model_or_arch_id),
                "checkpoint_epoch": item.checkpoint_epoch,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash(checkpoint),
                "artifact_sha256": artifact_hashes,
                "target_value": float(item.target_value),
                "metadata": metadata,
            })
    return {
        "candidate": "feature_activation_sign_survival_sum",
        "direction": "higher",
        "expected_rows": 560,
        "objects": objects,
        "source_sha256": source_sha256,
        "bundle_manifest_sha256": bundle_sha256,
        "sample_sha256": sample_sha256,
        "strict_csv_sha256": strict_sha256,
        "input_paths": {
            "bundle": {name: str(Path(path).resolve()) for name, path in bundle_manifests.items()},
            "sample": {name: str(Path(path).resolve()) for name, path in sample_paths.items()},
            "strict_csv": str(Path(strict_csv).resolve()),
        },
        "protocol": {
            "seed": 20260808, "sample_offset": 0, "num_samples": 64,
            "batch_size": 1, "attack": "fgsm", "epsilon": 8 / 255,
            "activation_types": ["Conv2d", "Linear"],
            "sign_threshold": 0, "score_reduction": "sum_calls_mean_samples",
        },
        "environment": dict(environment),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the exact 560-key activation candidate")
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--strict-csv", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        bundle = require_project_path(args.bundle_root, ROOT)
        data = require_project_path(args.data_root, ROOT)
        strict = require_project_path(args.strict_csv, ROOT)
        run_dir = require_project_path(args.run_dir, ROOT)
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from robust_metric_lab.student_benchmark import (
            load_darts_items, load_nb201_items, load_rb40_items, resolve_under_root,
        )
        import numpy
        import scipy
        import torch

        manifests = {
            "rb40": resolve_under_root(bundle, "manifests/rb40_models.csv"),
            "nb201": resolve_under_root(bundle, "manifests/nb201_64_architectures.csv"),
            "darts": resolve_under_root(bundle, "manifests/darts100_architectures.csv"),
        }
        rb_items = load_rb40_items(manifests["rb40"], bundle)
        nb_items = load_nb201_items(manifests["nb201"], bundle)
        darts_items = load_darts_items(manifests["darts"], bundle)
        grid = {"rb40_mature": rb_items}
        grid.update({f"nb201_e{epoch}": [item for item in nb_items if item.checkpoint_epoch == epoch]
                     for epoch in (0, 5, 10, 25, 50)})
        grid.update({f"darts_e{epoch}": [item for item in darts_items if item.checkpoint_epoch == epoch]
                     for epoch in (0, 100)})
        expected_main_keys(grid)
        test_batch = require_project_path(data / "cifar-10-batches-py" / "test_batch", ROOT)
        if not test_batch.is_file():
            raise FileNotFoundError(f"frozen CIFAR-10 test batch missing: {test_batch}")
        source_names = [
            "scripts/activation_survival_manifest.py",
            "scripts/evaluate_activation_survival_student.py",
            "scripts/verify_activation_survival.py",
            "scripts/evaluate_rb40_student.py",
            "scripts/evaluate_nb201_student.py",
            "scripts/evaluate_darts100_student.py",
        ]
        source_paths = {
            str(path.relative_to(ROOT)).replace("\\", "/"): path
            for path in sorted((ROOT / "src/robust_metric_lab").glob("*.py"))
        }
        source_paths.update({name: ROOT / name for name in source_names})
        payload = build_payload(
            grid, source_paths=source_paths, bundle_manifests=manifests,
            sample_paths={"cifar10_test_batch": test_batch},
            strict_csv=strict,
            environment={
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": numpy.__version__, "scipy": scipy.__version__,
                "cuda": str(torch.version.cuda),
            },
        )
        payload["run_id"] = run_dir.name
        manifest_path = run_dir / "frozen_manifest.json"
        digest = write_or_verify_manifest(manifest_path, payload)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(manifest_path), "sha256": digest, "expected_rows": 560}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
