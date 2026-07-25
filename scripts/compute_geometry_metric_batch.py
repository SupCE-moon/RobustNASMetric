from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.config import load_config, resolve_data_root
from robust_metric_lab.datasets import find_dataset_roots, get_dataset_loader
from robust_metric_lab.metrics import GEOMETRY_METRIC_FIELDS, compute_feature_geometry_metrics
from robust_metric_lab.robustbench_io import load_rb_model, validate_model_names


def choose_data_root(config, override: str | None) -> Path:
    explicit = resolve_data_root(config, override)
    if explicit:
        return explicit
    roots = find_dataset_roots(config.dataset, [])
    if not roots:
        raise SystemExit(f"No {config.dataset} root found. Set --data-root or DATA_ROOT.")
    return roots[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-limit", type=int)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attack", choices=("none", "fgsm", "pgd", "pgd-lite"), default="fgsm")
    parser.add_argument("--pgd-steps", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--activation-mode", choices=("all_leaf", "post_relu", "post_activation"), default="all_leaf")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    names = config.models[: args.model_limit] if args.model_limit else config.models
    names, missing = validate_model_names(names, config.dataset, config.threat_model)
    if missing:
        raise SystemExit(f"Unknown RobustBench model names: {missing}")
    loader = get_dataset_loader(
        dataset_name=config.dataset,
        data_root=choose_data_root(config, args.data_root),
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        limit=args.sample_limit,
    )

    rows: list[dict[str, object]] = []
    for name in tqdm(names, desc="models"):
        model = load_rb_model(
            name,
            config.dataset,
            config.threat_model,
            args.device,
            model_dir=args.model_dir or config.model_dir or "models",
        )
        totals = {field: 0.0 for field in GEOMETRY_METRIC_FIELDS}
        total = 0
        audit: dict[str, object] = {}
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            x_adv = make_adversarial(
                model, x, y, args.attack, config.epsilon, pgd_steps=args.pgd_steps
            )
            result = compute_feature_geometry_metrics(
                model,
                x,
                x_adv,
                max_layers=args.max_layers,
                activation_mode=args.activation_mode,
            )
            values = asdict(result)
            for field in GEOMETRY_METRIC_FIELDS:
                totals[field] += float(values[field]) * x.shape[0]
            total += x.shape[0]
            audit = {
                "num_layers": result.num_layers,
                "num_candidate_layers": result.num_candidate_layers,
                "sampled_layer_indices": result.sampled_layer_indices,
            }
        rows.append(
            {
                "model_name": name,
                "attack": args.attack,
                "num_samples": total,
                "max_layers": args.max_layers,
                "activation_mode": args.activation_mode,
                **{field: totals[field] / total for field in GEOMETRY_METRIC_FIELDS},
                **audit,
            }
        )
        write_rows(args.output, rows)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
