from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.attacks import make_adversarial  # noqa: E402
from robust_metric_lab.student_benchmark import EPSILON, load_rb40_items, resolve_under_root  # noqa: E402
from robust_metric_lab.student_runner import load_cifar10_samples  # noqa: E402
from robust_metric_lab.trnas import _run_trnas_once, compute_trnas_metrics_stable  # noqa: E402

from evaluate_rb40_student import make_model_loader  # noqa: E402


def _attempt(name, operation):
    try:
        result = operation()
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    values = asdict(result)
    finite = all(math.isfinite(float(values[key])) for key in ("activation_psi", "consistency_phi", "r_score"))
    return {"name": name, "status": "ok" if finite else "nonfinite", "finite": finite, "result": values}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose only Kang2021Stable TRNAS underflow")
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic output")

    root = args.bundle_root.resolve()
    items = load_rb40_items(resolve_under_root(root, "manifests/rb40_models.csv"), root)
    matching = [item for item in items if item.model_or_arch_id == "Kang2021Stable"]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one Kang2021Stable manifest item, found {len(matching)}")
    item = matching[0]
    model = make_model_loader(root, args.device)(item)
    clean, labels = load_cifar10_samples(args.data_root, args.device)
    adversarial = make_adversarial(model, clean, labels, "fgsm", EPSILON, pgd_steps=1)

    attempts = [
        _attempt(
            "float32_direct",
            lambda: _run_trnas_once(model, clean, adversarial, labels, weight_step_size=1e-6),
        ),
        _attempt(
            "float64_direct",
            lambda: _run_trnas_once(
                model.double(), clean.double(), adversarial.double(), labels, weight_step_size=1e-6
            ),
        ),
    ]
    # Reload a fresh float32 model because the direct float64 diagnostic mutates
    # only this diagnostic instance's dtype.
    stable_model = make_model_loader(root, args.device)(item)
    attempts.append(
        _attempt(
            "stable_wrapper",
            lambda: compute_trnas_metrics_stable(
                stable_model,
                clean.float(),
                adversarial.float(),
                labels,
                weight_step_size=1e-6,
            ),
        )
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_or_arch_id": item.model_or_arch_id,
        "checkpoint_path": str(item.checkpoint_path),
        "device": args.device,
        "weight_step_size": 1e-6,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if attempts[-1]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
