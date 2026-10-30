"""Append-only evaluator for the preregistered sign-survival candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid
from typing import Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from robust_metric_lab.activation_sign_survival import sign_survival  # noqa: E402
from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED,
    load_darts_items, load_nb201_items, load_rb40_items, resolve_under_root,
)
from robust_metric_lab.student_runner import load_cifar10_samples  # noqa: E402
from evaluate_darts100_student import make_model_loader as make_darts_loader  # noqa: E402
from evaluate_nb201_student import make_model_loader as make_nb201_loader  # noqa: E402
from evaluate_rb40_student import make_model_loader as make_rb40_loader  # noqa: E402
from scripts.activation_survival_manifest import sha256_file  # noqa: E402


def item_key(setting: str, item: object) -> tuple[str, str, int | None]:
    return setting, str(item.model_or_arch_id), item.checkpoint_epoch


def row_key(row: Mapping) -> tuple[str, str, int | None]:
    setting, identifier, epoch = row["key"]
    return str(setting), str(identifier), None if epoch is None else int(epoch)


def write_attempt(directory: Path, payload: dict) -> Path:
    """Never replace an earlier success or failure, even for the same key."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def read_success_keys(
    directory: Path, manifest_sha256: str, source_sha256: Mapping[str, str],
) -> set[tuple[str, str, int | None]]:
    successes: set[tuple[str, str, int | None]] = set()
    if not directory.exists():
        return successes
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant in {path.name}: {value}")
            ))
        if row.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"manifest mismatch in attempt: {path.name}")
        if row.get("source_sha256") != dict(source_sha256):
            raise ValueError(f"source mismatch in attempt: {path.name}")
        key = row_key(row)
        if row.get("status") == "ok":
            value = float(row["metric_value"])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite success: {key}")
            if key in successes:
                raise ValueError(f"duplicate success: {key}")
            successes.add(key)
        elif row.get("status") != "failed":
            raise ValueError(f"invalid attempt status: {path.name}")
    return successes


def pending_keys(
    expected: set[tuple[str, str, int | None]],
    directory: Path,
    manifest_sha256: str,
    source_sha256: Mapping[str, str],
) -> set[tuple[str, str, int | None]]:
    successes = read_success_keys(directory, manifest_sha256, source_sha256)
    if not successes.issubset(expected):
        raise ValueError("attempt contains unknown successful key")
    return expected - successes


def evaluate_keys(
    items: Iterable[object],
    load_model: Callable,
    samples: Callable,
    manifest: dict,
    attempts_dir: Path,
    *,
    setting: str,
    source_sha256: Mapping[str, str],
    compute: Callable = sign_survival,
) -> dict[str, int]:
    """Evaluate only missing successful keys under one frozen source/manifest."""
    items = list(items)
    manifest_sha256 = str(manifest["frozen_sha256"])
    objects = {
        (row["setting"], str(row["model_or_arch_id"]), row["checkpoint_epoch"]): row
        for row in manifest["frozen_payload"]["objects"]
        if row["setting"] == setting
    }
    expected = set(objects)
    supplied = [item_key(setting, item) for item in items]
    if len(supplied) != len(set(supplied)) or set(supplied) != expected:
        raise ValueError("item keys differ from frozen manifest")
    missing = pending_keys(expected, attempts_dir, manifest_sha256, source_sha256)
    result = {"written": 0, "failed": 0, "skipped": len(expected) - len(missing)}
    sample_value = None
    for item in items:
        key = item_key(setting, item)
        if key not in missing:
            continue
        attempt = {
            "key": list(key),
            "manifest_sha256": manifest_sha256,
            "source_sha256": dict(source_sha256),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_value": float(item.target_value),
            "checkpoint_path": str(item.checkpoint_path),
        }
        try:
            frozen = objects[key]
            if not math.isfinite(float(item.target_value)) or float(item.target_value) != float(frozen["target_value"]):
                raise ValueError("target differs from frozen manifest")
            checkpoint_hash = sha256_file(Path(item.checkpoint_path))
            if checkpoint_hash != frozen["checkpoint_sha256"]:
                raise ValueError("checkpoint differs from frozen manifest")
            attempt["checkpoint_sha256"] = checkpoint_hash
            if sample_value is None:
                sample_value = samples()
            model = load_model(item)
            score = compute(model, *sample_value, epsilon=EPSILON)
            value = float(score["metric_value"])
            if not math.isfinite(value):
                raise ValueError("nonfinite metric value")
            attempt.update({
                "status": "ok", "metric_value": value,
                "sample_count": int(score["sample_count"]),
                "call_count": int(score["call_count"]),
                "diagnostics": score["diagnostics"],
            })
        except Exception as exc:
            attempt.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            result["failed"] += 1
        attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_attempt(attempts_dir, attempt)
        result["written"] += 1
    return result


def _setting_and_items(benchmark: str, epoch: int | None, bundle: Path, device: str):
    if benchmark == "rb40":
        if epoch is not None:
            raise ValueError("RB40 does not accept epoch")
        items = load_rb40_items(resolve_under_root(bundle, "manifests/rb40_models.csv"), bundle)
        return "rb40_mature", items, make_rb40_loader(bundle, device)
    if benchmark == "nb201":
        if epoch not in {0, 5, 10, 25, 50}:
            raise ValueError("unsupported NB201 epoch")
        items = load_nb201_items(resolve_under_root(bundle, "manifests/nb201_64_architectures.csv"), bundle)
        return f"nb201_e{epoch}", [item for item in items if item.checkpoint_epoch == epoch], make_nb201_loader(device)
    if benchmark == "darts":
        if epoch not in {0, 100}:
            raise ValueError("unsupported DARTS epoch")
        items = load_darts_items(resolve_under_root(bundle, "manifests/darts100_architectures.csv"), bundle)
        return f"darts_e{epoch}", [item for item in items if item.checkpoint_epoch == epoch], make_darts_loader(device)
    raise ValueError("unknown benchmark")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("rb40", "nb201", "darts"))
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--attempts-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=SAMPLE_OFFSET)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if (args.batch_size, args.sample_offset, args.num_samples, args.seed) != (1, 0, 64, 20260808):
        parser.error("sample and batch protocol is frozen")
    for path in (args.bundle_root, args.data_root, args.manifest, args.attempts_dir):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            parser.error("all paths must remain inside the permitted project")
    try:
        with args.manifest.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        setting, items, loader = _setting_and_items(args.benchmark, args.epoch, args.bundle_root.resolve(), args.device)
        sources = manifest["frozen_payload"]["source_sha256"]
        for relative, expected_hash in sources.items():
            if sha256_file(ROOT / relative) != expected_hash:
                raise ValueError(f"source mismatch: {relative}")
        result = evaluate_keys(
            items, loader, lambda: load_cifar10_samples(args.data_root, args.device),
            manifest, args.attempts_dir, setting=setting, source_sha256=sources,
        )
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
