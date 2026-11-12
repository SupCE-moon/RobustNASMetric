"""Create or verify the frozen input/source manifest for one new metric run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def expected_counts(mode: str) -> dict[str, int]:
    if mode == "smoke":
        rb40, nb201, darts = 4, 4, 4
    elif mode == "full":
        rb40, nb201, darts = 40, 64, 100
    else:
        raise ValueError("mode must be smoke or full")
    return {
        "rb40_mature": rb40,
        **{f"nb201_e{epoch}": nb201 for epoch in (0, 5, 10, 25, 50)},
        **{f"darts_e{epoch}": darts for epoch in (0, 100)},
    }


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_or_verify_manifest(path: Path, payload: dict) -> str:
    """Never overwrite a frozen manifest; accept only byte-equivalent inputs."""
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_payload = existing.get("frozen_payload")
        old_digest = existing.get("frozen_sha256")
        if old_digest != digest or old_digest != hashlib.sha256(_canonical(old_payload)).hexdigest():
            raise ValueError("manifest mismatch: run directory has different frozen inputs")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_sha256": digest,
        "frozen_payload": payload,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return write_or_verify_manifest(path, payload)
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(project: Path, bundle: Path, mode: str, device: str, batch_size: int) -> dict:
    import numpy as np
    import torch

    from robust_metric_lab.student_benchmark import (
        EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED,
        load_darts_items, load_nb201_items, load_rb40_items,
    )

    counts = expected_counts(mode)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifests = {
        "rb40_mature": bundle / "manifests/rb40_models.csv",
        "nb201": bundle / "manifests/nb201_64_architectures.csv",
        "darts": bundle / "manifests/darts100_architectures.csv",
    }
    rb = load_rb40_items(manifests["rb40_mature"], bundle)
    nb = load_nb201_items(manifests["nb201"], bundle)
    da = load_darts_items(manifests["darts"], bundle)
    all_by_setting = {
        "rb40_mature": rb,
        **{f"nb201_e{epoch}": [item for item in nb if item.checkpoint_epoch == epoch]
           for epoch in (0, 5, 10, 25, 50)},
        **{f"darts_e{epoch}": [item for item in da if item.checkpoint_epoch == epoch]
           for epoch in (0, 100)},
    }
    for setting, wanted in expected_counts("full").items():
        if len(all_by_setting[setting]) != wanted:
            raise ValueError(f"incomplete manifest for {setting}: {len(all_by_setting[setting])}/{wanted}")

    checkpoint_hashes: dict[str, str] = {}
    objects: dict[str, list[dict]] = {}
    for setting, wanted in counts.items():
        chosen = all_by_setting[setting][:wanted]
        entries = []
        for item in chosen:
            checkpoint = item.checkpoint_path.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
            key = str(checkpoint)
            if key not in checkpoint_hashes:
                checkpoint_hashes[key] = _sha256(checkpoint)
            entries.append({
                "model_or_arch_id": item.model_or_arch_id,
                "checkpoint_epoch": item.checkpoint_epoch,
                "target_value": item.target_value,
                "checkpoint_path": key,
                "checkpoint_sha256": checkpoint_hashes[key],
            })
        objects[setting] = entries

    source_rel = (
        "docs/superpowers/specs/2026-10-24-adversarial-excess-feature-drift-design.md",
        "src/robust_metric_lab/adv_excess_metric.py",
        "src/robust_metric_lab/adv_excess_batched.py",
        "src/robust_metric_lab/attacks.py",
        "src/robust_metric_lab/metrics.py",
        "src/robust_metric_lab/metric_registry.py",
        "src/robust_metric_lab/student_benchmark.py",
        "src/robust_metric_lab/student_runner.py",
        "configs/metric_eligibility.json",
        "scripts/evaluate_adv_excess_student.py",
        "scripts/adv_excess_run_manifest.py",
        "scripts/run_adv_excess_full.sh",
    )
    sources = {relative: _sha256(project / relative) for relative in source_rel}
    bundle_hashes = {str(path.relative_to(project)): _sha256(path) for path in manifests.values()}
    return {
        "metric_name": "feature_adv_excess_centered_unit_l2",
        "direction": "lower",
        "mode": mode,
        "expected_rows": sum(counts.values()),
        "expected_counts": counts,
        "formula": "mean_samples(median_anchors(0.5*||u(clean)-u(FGSM)||_2 - 0.5*||u(clean)-u(random)||_2))",
        "protocol": {
            "seed": SEED, "sample_offset": SAMPLE_OFFSET,
            "num_samples": NUM_SAMPLES, "epsilon": EPSILON,
            "attack": "fgsm", "max_layers": 4,
            "random_sign_seed": 20261024, "random_sign_generator": "numpy.PCG64(seed+global_index)",
        },
        "device": device,
        "batch_size": batch_size,
        "objects": objects,
        "source_sha256": sources,
        "bundle_manifest_sha256": bundle_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "pip_freeze": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ).splitlines(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    payload = build_payload(args.project, args.bundle, args.mode, args.device, args.batch_size)
    digest = write_or_verify_manifest(args.run_dir / "run_manifest.json", payload)
    print(f"manifest_sha256={digest} expected_rows={payload['expected_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
