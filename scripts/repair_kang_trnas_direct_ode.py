"""Isolated, auditable repair of the three failed Kang2021Stable TRNAS rows.

Only the ODE differentiation route changes: torchdiffeq.odeint has the same
forward integrator as odeint_adjoint, but backpropagates through its forward
steps instead of integrating an unstable adjoint ODE.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import pandas as pd


TRNAS_NAMES = ("trnas_activation_psi", "trnas_consistency_phi", "trnas_r_score")


@contextmanager
def direct_ode_context(sodef_module, direct_solver):
    original = sodef_module.odeint
    sodef_module.odeint = direct_solver
    try:
        yield
    finally:
        sodef_module.odeint = original


def select_repair_rows(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw[
        (raw["setting"] == "rb40_mature")
        & (raw["model_or_arch_id"] == "Kang2021Stable")
        & raw["metric_name"].isin(TRNAS_NAMES)
    ].copy()
    if (
        len(selected) != 3
        or set(selected.metric_name) != set(TRNAS_NAMES)
        or not (selected.status == "failed").all()
    ):
        raise ValueError("expected exactly three failed Kang2021Stable TRNAS rows")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite output directory")

    import torch
    import torchdiffeq
    import robustbench.model_zoo.architectures.sodef_layers as sodef
    from evaluate_rb40_student import make_model_loader
    from robust_metric_lab.attacks import make_adversarial
    from robust_metric_lab.student_benchmark import (
        EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED, load_rb40_items, resolve_under_root,
    )
    from robust_metric_lab.student_runner import load_cifar10_samples
    from robust_metric_lab.trnas import compute_trnas_metrics

    raw = pd.read_csv(args.source_raw, keep_default_na=False)
    previous = select_repair_rows(raw)
    if set(previous.seed.astype(int)) != {SEED} or set(previous.num_samples.astype(int)) != {NUM_SAMPLES}:
        raise ValueError("source sample protocol differs from frozen protocol")
    if set(previous.sample_offset.astype(int)) != {SAMPLE_OFFSET}:
        raise ValueError("source sample offset differs from frozen protocol")
    if any(not math.isclose(float(v), EPSILON, rel_tol=0, abs_tol=1e-15) for v in previous.epsilon):
        raise ValueError("source epsilon differs from frozen protocol")

    bundle = args.bundle_root.resolve()
    manifest = resolve_under_root(bundle, "manifests/rb40_models.csv")
    items = load_rb40_items(manifest, bundle)
    matching = [item for item in items if item.model_or_arch_id == "Kang2021Stable"]
    if len(matching) != 1:
        raise ValueError(f"expected one Kang2021Stable manifest item, got {len(matching)}")
    item = matching[0]
    torch.manual_seed(SEED)
    model = make_model_loader(bundle, args.device)(item).eval()
    clean, labels = load_cifar10_samples(args.data_root, args.device)
    if len(clean) != NUM_SAMPLES:
        raise ValueError(f"expected {NUM_SAMPLES} images, got {len(clean)}")
    start = time.perf_counter()
    adversarial = make_adversarial(model, clean, labels, "fgsm", EPSILON, pgd_steps=1)
    with torch.no_grad():
        original_clean = model(clean).detach()
        original_adv = model(adversarial).detach()
    with direct_ode_context(sodef, torchdiffeq.odeint):
        with torch.no_grad():
            direct_clean = model(clean).detach()
            direct_adv = model(adversarial).detach()
        clean_delta = float((original_clean - direct_clean).abs().max().item())
        adv_delta = float((original_adv - direct_adv).abs().max().item())
        if max(clean_delta, adv_delta) > 1e-6:
            raise ValueError(f"ODE forward mismatch: clean={clean_delta}, adversarial={adv_delta}")
        result = compute_trnas_metrics(model, clean, adversarial, labels, weight_step_size=1e-6)
    values = {
        "trnas_activation_psi": result.activation_psi,
        "trnas_consistency_phi": result.consistency_phi,
        "trnas_r_score": result.r_score,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"nonfinite TRNAS result: {values}")
    if not math.isclose(result.r_score, result.activation_psi**0.5 * result.consistency_phi, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("R-Score formula mismatch")

    elapsed = time.perf_counter() - start
    repair = previous.copy()
    for name, value in values.items():
        repair.loc[repair.metric_name == name, "metric_value"] = float(value)
    repair["run_id"] = args.output_dir.name
    repair["status"] = "ok"
    repair["error"] = ""
    repair["runtime_sec"] = elapsed
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repair.to_csv(args.output_dir / "repair_rows.csv", index=False)
    diagnostic = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw": str(args.source_raw.resolve()),
        "checkpoint_path": str(item.checkpoint_path),
        "device": args.device,
        "seed": SEED,
        "sample_offset": SAMPLE_OFFSET,
        "num_samples": NUM_SAMPLES,
        "epsilon": EPSILON,
        "weight_step_size": 1e-6,
        "gradient_route": "torchdiffeq.odeint direct autograd only for Kang TRNAS; unchanged forward solver",
        "original_route": "torchdiffeq.odeint_adjoint",
        "forward_clean_max_abs_delta": clean_delta,
        "forward_adversarial_max_abs_delta": adv_delta,
        "result": asdict(result),
        "runtime_sec": elapsed,
        "replaced_metric_names": list(TRNAS_NAMES),
    }
    (args.output_dir / "diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "values": values, "forward_max_delta": max(clean_delta, adv_delta)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
