from __future__ import annotations

import argparse
import csv
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics


METRIC_FIELDS = (
    "feature_measure_spatial_gram_delta",
    "feature_measure_spatial_gram_rel_delta",
    "feature_measure_robust_spatial_delta",
    "feature_measure_robust_spatial_rel_delta",
    "feature_measure_robust_hybrid_delta",
    "feature_measure_robust_hybrid_rel_delta",
    "feature_measure_flow_rel_delta",
    "fsp_delta",
    "logit_kl",
    "input_grad_l2",
    "local_logit_lipschitz",
    "margin_input_grad_ratio",
    "clean_acc",
    "adv_acc",
)


@dataclass(frozen=True)
class LegacySpec:
    name: str
    model_type: str
    checkpoint: Path


class TensorOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def install_legacy_imports(legacy_root: Path, dataset: str) -> None:
    sys.path.insert(0, str(legacy_root))
    config_mod = types.ModuleType("config")
    config_mod.__path__ = [str(legacy_root / "config 2")]
    config_mod.dataset_name = dataset
    config_mod.numberwork = 0
    sys.modules["config"] = config_mod


def newest_checkpoint(directory: Path, pattern: str = "*.pt") -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def build_manifest(legacy_root: Path, dataset: str) -> list[LegacySpec]:
    specs: list[LegacySpec] = []
    save_root = legacy_root / "demo" / "train_diff_models" / "save_models" / dataset

    if dataset == "mnist":
        for run in ("mnist_nat", "mnist_nat2", "mnist_nat3"):
            ckpt = newest_checkpoint(save_root / run, "cifar_best.pt")
            if ckpt:
                specs.append(LegacySpec(run, "standard", ckpt))
        at_root = legacy_root / "advdata" / "whitebox_trained_models" / "gate_at" / "mnist_AT"
        for run in ("PGD_mnist_AT_1", "PGD_mnist_AT_2", "PGD_mnist_AT_3"):
            ckpt = newest_checkpoint(at_root / run, "mnist_best_adv.pt")
            if ckpt:
                specs.append(LegacySpec(run, "standard", ckpt))
        denoise_root = legacy_root / "advdata" / "whitebox_trained_models" / "nongate" / "PGD_mnist"
        for run in ("PGD_mnist_denoise_1", "PGD_mnist_denoise_2", "PGD_mnist_denoise_3"):
            ckpt = newest_checkpoint(denoise_root / run, "cifar_best_adv.pt")
            if ckpt:
                specs.append(LegacySpec(run, "denoise", ckpt))
    elif dataset == "svhn":
        for run in ("svhn_nat", "svhn_nat1", "svhn_nat2", "svhn_nat3"):
            ckpt = newest_checkpoint(save_root / run, "cifar_best.pt")
            if ckpt:
                specs.append(LegacySpec(run, "standard", ckpt))
        for run in ("svhn_AT1",):
            ckpt = newest_checkpoint(save_root / run, "cifar_best_adv.pt")
            if ckpt:
                specs.append(LegacySpec(run, "standard", ckpt))
        for run in ("fgsm_svhn_denoise_1",):
            ckpt = newest_checkpoint(save_root / run, "cifar_best_adv.pt")
            if ckpt:
                specs.append(LegacySpec(run, "denoise", ckpt))
        denoise_root = legacy_root / "advdata" / "whitebox_trained_models" / "nongate" / "PGD_svhn"
        for run in ("pgd_svhn_denoise_1", "pgd_svhn_denoise_3"):
            ckpt = newest_checkpoint(denoise_root / run, "cifar_best_adv.pt")
            if ckpt:
                specs.append(LegacySpec(run, "denoise", ckpt))
    else:
        raise ValueError(f"Unsupported legacy dataset: {dataset}")

    return specs


def build_model(model_type: str) -> torch.nn.Module:
    if model_type in {"standard", "denoise"}:
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet

        return WideResNet(depth=28, widen_factor=10, denoise=(model_type == "denoise"))
    raise ValueError(f"Unsupported legacy model type: {model_type}")


def load_legacy_model(spec: LegacySpec, device: str) -> torch.nn.Module:
    model = build_model(spec.model_type)
    state = torch.load(spec.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = TensorOutputWrapper(model)
    model.eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mnist", "svhn"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--legacy-root", default=r"D:\old_research_code\adversarial_research")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--attack", default="fgsm", choices=["none", "fgsm", "pgd", "pgd-lite"])
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    legacy_root = Path(args.legacy_root)
    install_legacy_imports(legacy_root, args.dataset)
    specs = build_manifest(legacy_root, args.dataset)
    if not specs:
        raise SystemExit(f"No legacy checkpoints found for {args.dataset}")

    loader = get_dataset_loader(
        dataset_name=args.dataset,
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=0,
        limit=args.sample_limit,
    )

    rows = []
    for spec in tqdm(specs, desc="legacy models"):
        try:
            model = load_legacy_model(spec, args.device)
        except Exception as exc:
            print(f"SKIP {spec.name}: {exc}", file=sys.stderr)
            continue
        totals = {field: 0.0 for field in METRIC_FIELDS}
        num_samples = 0
        num_layers = 0
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            batch_n = x.shape[0]
            x_adv = make_adversarial(model, x, y, args.attack, args.epsilon, pgd_steps=args.pgd_steps)
            result = compute_feature_metrics(model, x, x_adv, y, max_layers=args.max_layers)
            for field in METRIC_FIELDS:
                totals[field] += getattr(result, field) * batch_n
            num_samples += batch_n
            num_layers = result.num_layers
            del x, y, x_adv

        row = {
            "model_name": spec.name,
            "model_type": spec.model_type,
            "dataset": args.dataset,
            "attack": args.attack,
            "batch_size": args.batch_size,
            "num_samples": num_samples,
            "num_layers": num_layers,
            "checkpoint": str(spec.checkpoint),
        }
        row.update({field: totals[field] / num_samples for field in METRIC_FIELDS})
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No legacy models were successfully evaluated.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
