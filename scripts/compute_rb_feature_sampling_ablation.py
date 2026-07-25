from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.config import load_config
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.robustbench_io import load_rb_model, validate_model_names


RESULT_METADATA = {"num_layers", "num_candidate_layers", "sampled_layer_indices"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-start", type=int, default=0)
    parser.add_argument("--model-stop", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--attack", default="fgsm", choices=("fgsm", "pgd"))
    parser.add_argument("--pgd-steps", type=int, default=1)
    parser.add_argument("--max-layers", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument(
        "--activation-modes",
        nargs="+",
        choices=("all_leaf", "post_relu", "post_activation"),
        default=("all_leaf", "post_relu"),
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if any(value < 2 for value in args.max_layers):
        raise SystemExit("Every --max-layers value must be at least 2 for pairwise metrics.")
    config = load_config(args.config)
    valid, missing = validate_model_names(config.models, config.dataset, config.threat_model)
    if missing:
        raise SystemExit(f"Unknown RobustBench model names: {missing}")
    model_stop = len(valid) if args.model_stop is None else args.model_stop
    if args.model_start < 0 or model_stop > len(valid) or args.model_start >= model_stop:
        raise SystemExit(f"Invalid model slice [{args.model_start}:{model_stop}].")
    names = valid[args.model_start:model_stop]

    loader = get_dataset_loader(
        dataset_name=config.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        limit=args.sample_limit,
    )
    conditions = [
        (activation_mode, max_layers)
        for activation_mode in args.activation_modes
        for max_layers in sorted(set(args.max_layers))
    ]
    output_rows: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name in tqdm(names, desc="models"):
        model = load_rb_model(
            name,
            config.dataset,
            config.threat_model,
            args.device,
            model_dir=str(args.model_dir or config.model_dir or "models"),
        )
        totals: dict[tuple[str, int], dict[str, float]] = {condition: {} for condition in conditions}
        audit: dict[tuple[str, int], dict[str, object]] = {condition: {} for condition in conditions}
        num_samples = 0
        set_seed(args.seed)

        for x_clean, labels in loader:
            x_clean = x_clean.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            x_adv = make_adversarial(
                model,
                x_clean,
                labels,
                args.attack,
                config.epsilon,
                pgd_steps=args.pgd_steps,
            )
            batch_size = x_clean.size(0)
            for condition in conditions:
                activation_mode, max_layers = condition
                result = compute_feature_metrics(
                    model,
                    x_clean,
                    x_adv,
                    labels,
                    max_layers=max_layers,
                    activation_mode=activation_mode,
                )
                values = asdict(result)
                audit[condition] = {key: values[key] for key in RESULT_METADATA}
                for key, value in values.items():
                    if key in RESULT_METADATA:
                        continue
                    totals[condition][key] = totals[condition].get(key, 0.0) + float(value) * batch_size
            num_samples += batch_size
            del x_clean, labels, x_adv

        if num_samples == 0:
            raise RuntimeError("No samples were loaded.")
        for activation_mode, max_layers in conditions:
            condition = (activation_mode, max_layers)
            row: dict[str, object] = {
                "model_name": name,
                "dataset": config.dataset,
                "threat_model": config.threat_model,
                "attack": args.attack,
                "attack_source": "white_box",
                "epsilon": config.epsilon,
                "pgd_steps": args.pgd_steps,
                "seed": args.seed,
                "num_samples": num_samples,
                "activation_mode": activation_mode,
                "max_layers": max_layers,
                "model_start": args.model_start,
                "model_stop": model_stop,
            }
            row.update({key: value / num_samples for key, value in totals[condition].items()})
            row.update(audit[condition])
            output_rows.append(row)

        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
