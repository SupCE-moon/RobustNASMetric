from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from tqdm import tqdm

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.config import load_config, resolve_data_root
from robust_metric_lab.datasets import find_dataset_roots, get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.nasrobbench201 import racl_fixed_network_lipschitz_stats
from robust_metric_lab.robustbench_io import load_rb_model, validate_model_names
from robust_metric_lab.trnas import compute_trnas_metrics

METRIC_FIELDS = (
    "feature_delta",
    "feature_measure_swd",
    "feature_measure_mmd",
    "feature_measure_cov_delta",
    "feature_hausdorff_delta",
    "feature_quantile_hausdorff_delta",
    "feature_box_dimension_delta",
    "feature_graph_fundamental_rank_delta",
    "feature_measure_moment_delta",
    "feature_measure_local_delta",
    "feature_measure_weighted_delta",
    "feature_measure_struct_delta",
    "feature_measure_struct01_delta",
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "logit_kl",
    "natural_loss",
    "adversarial_loss",
    "input_loss_delta",
    "input_grad_l2",
    "local_logit_lipschitz",
    "margin_input_grad_ratio",
    "feature_measure_flow_delta",
    "feature_measure_flow_rel_delta",
    "feature_measure_spatial_gram_delta",
    "feature_measure_spatial_gram_rel_delta",
    "feature_measure_relu_active_delta",
    "feature_measure_relu_active_spatial_delta",
    "feature_measure_relu_active_spatial_rel_delta",
    "feature_measure_relu_active_hybrid_delta",
    "feature_measure_relu_active_hybrid_rel_delta",
    "feature_measure_robust_delta",
    "feature_measure_robust_rel_delta",
    "feature_measure_robust_spatial_delta",
    "feature_measure_robust_spatial_rel_delta",
    "feature_measure_robust_hybrid_delta",
    "feature_measure_robust_hybrid_rel_delta",
    "fsp_strict_delta",
    "fsp_delta",
    "clean_acc",
    "adv_acc",
)


def choose_data_root(config, override: str | None) -> Path:
    explicit = resolve_data_root(config, override)
    if explicit:
        return explicit
    roots = find_dataset_roots(config.dataset, [])
    if not roots:
        raise SystemExit(f"No {config.dataset} root found. Set --data-root or DATA_ROOT.")
    return roots[0]


def get_shared_adv_loader(cache_path: str, batch_size: int) -> tuple[torch.utils.data.DataLoader, dict]:
    cache = torch.load(cache_path, map_location="cpu")
    required = {"x", "y", "x_adv"}
    missing = sorted(required - set(cache))
    if missing:
        raise SystemExit(f"Adversarial cache is missing fields: {missing}")
    dataset = torch.utils.data.TensorDataset(cache["x"], cache["y"], cache["x_adv"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metadata = {key: value for key, value in cache.items() if key not in required}
    return loader, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="outputs/metrics.csv")
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--attack", default="fgsm", choices=["none", "fgsm", "pgd", "pgd-lite"])
    parser.add_argument("--adv-cache", default=None, help="Shared adversarial examples saved by generate_adv_cache.py")
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument(
        "--include-published-nas-metrics",
        action="store_true",
        help="Also compute the two RACL terms and TRNAS Psi/Phi/R-Score.",
    )
    parser.add_argument("--trnas-weight-step-size", type=float, default=1e-6)
    parser.add_argument(
        "--published-metric-samples",
        type=int,
        default=16,
        help="Joint sample count for TRNAS; kept separate to fit large RobustBench models.",
    )
    parser.add_argument(
        "--activation-mode",
        choices=("all_leaf", "post_relu", "post_activation"),
        default="all_leaf",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    names = config.models[: args.model_limit] if args.model_limit else config.models
    valid, missing = validate_model_names(names, config.dataset, config.threat_model)
    if missing:
        raise SystemExit(f"Unknown RobustBench model names: {missing}")

    batch_size = args.batch_size or config.batch_size
    adv_cache_meta = {}
    if args.adv_cache:
        loader, adv_cache_meta = get_shared_adv_loader(args.adv_cache, batch_size)
    else:
        data_root = choose_data_root(config, args.data_root)
        loader = get_dataset_loader(
            dataset_name=config.dataset,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=config.num_workers,
            limit=args.sample_limit,
        )

    rows = []
    for name in tqdm(valid, desc="models"):
        model = load_rb_model(
            name,
            config.dataset,
            config.threat_model,
            args.device,
            model_dir=args.model_dir or config.model_dir or "models",
        )
        totals = {field: 0.0 for field in METRIC_FIELDS}
        num_samples = 0
        num_layers = 0
        published_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        for batch in loader:
            if args.adv_cache:
                x, y, x_adv = batch
            else:
                x, y = batch
                x_adv = None
            x = x.to(args.device)
            y = y.to(args.device)
            batch_n = x.shape[0]
            if x_adv is None:
                x_adv = make_adversarial(model, x, y, args.attack, config.epsilon, pgd_steps=args.pgd_steps)
            else:
                x_adv = x_adv.to(args.device)

            result = compute_feature_metrics(
                model,
                x,
                x_adv,
                y,
                max_layers=args.max_layers,
                activation_mode=args.activation_mode,
            )
            for field in METRIC_FIELDS:
                totals[field] += getattr(result, field) * batch_n

            num_samples += batch_n
            num_layers = result.num_layers
            if (
                args.include_published_nas_metrics
                and sum(item[0].shape[0] for item in published_batches)
                < args.published_metric_samples
            ):
                published_batches.append((x.detach(), y.detach(), x_adv.detach()))
            del x, y, x_adv

        if num_samples == 0:
            raise SystemExit("No CIFAR-10 samples were loaded.")

        row = {
                "model_name": name,
                "dataset": config.dataset,
                "threat_model": config.threat_model,
                "attack": adv_cache_meta.get("attack", args.attack),
                "attack_source": adv_cache_meta.get("source_model", "white_box"),
                "adv_cache": args.adv_cache or "",
                "batch_size": batch_size,
                "num_samples": num_samples,
                "feature_delta": totals["feature_delta"] / num_samples,
                "feature_measure_swd": totals["feature_measure_swd"] / num_samples,
                "feature_measure_mmd": totals["feature_measure_mmd"] / num_samples,
                "feature_measure_cov_delta": totals["feature_measure_cov_delta"] / num_samples,
                "feature_hausdorff_delta": totals["feature_hausdorff_delta"] / num_samples,
                "feature_quantile_hausdorff_delta": totals["feature_quantile_hausdorff_delta"] / num_samples,
                "feature_box_dimension_delta": totals["feature_box_dimension_delta"] / num_samples,
                "feature_graph_fundamental_rank_delta": totals["feature_graph_fundamental_rank_delta"] / num_samples,
                "feature_measure_moment_delta": totals["feature_measure_moment_delta"] / num_samples,
                "feature_measure_local_delta": totals["feature_measure_local_delta"] / num_samples,
                "feature_measure_weighted_delta": totals["feature_measure_weighted_delta"] / num_samples,
                "feature_measure_struct_delta": totals["feature_measure_struct_delta"] / num_samples,
                "feature_measure_struct01_delta": totals["feature_measure_struct01_delta"] / num_samples,
                "perturbation_capacity_delta": totals["perturbation_capacity_delta"] / num_samples,
                "perturbation_capacity_relu_active_delta": totals["perturbation_capacity_relu_active_delta"] / num_samples,
                "perturbation_capacity_above_mean_delta": totals["perturbation_capacity_above_mean_delta"] / num_samples,
                "logit_kl": totals["logit_kl"] / num_samples,
                "natural_loss": totals["natural_loss"] / num_samples,
                "adversarial_loss": totals["adversarial_loss"] / num_samples,
                "input_loss_delta": totals["input_loss_delta"] / num_samples,
                "input_grad_l2": totals["input_grad_l2"] / num_samples,
                "local_logit_lipschitz": totals["local_logit_lipschitz"] / num_samples,
                "margin_input_grad_ratio": totals["margin_input_grad_ratio"] / num_samples,
                "feature_measure_flow_delta": totals["feature_measure_flow_delta"] / num_samples,
                "feature_measure_flow_rel_delta": totals["feature_measure_flow_rel_delta"] / num_samples,
                "feature_measure_spatial_gram_delta": totals["feature_measure_spatial_gram_delta"] / num_samples,
                "feature_measure_spatial_gram_rel_delta": totals["feature_measure_spatial_gram_rel_delta"] / num_samples,
                "feature_measure_relu_active_delta": totals["feature_measure_relu_active_delta"] / num_samples,
                "feature_measure_relu_active_spatial_delta": totals["feature_measure_relu_active_spatial_delta"] / num_samples,
                "feature_measure_relu_active_spatial_rel_delta": totals["feature_measure_relu_active_spatial_rel_delta"] / num_samples,
                "feature_measure_relu_active_hybrid_delta": totals["feature_measure_relu_active_hybrid_delta"] / num_samples,
                "feature_measure_relu_active_hybrid_rel_delta": totals["feature_measure_relu_active_hybrid_rel_delta"] / num_samples,
                "feature_measure_robust_delta": totals["feature_measure_robust_delta"] / num_samples,
                "feature_measure_robust_rel_delta": totals["feature_measure_robust_rel_delta"] / num_samples,
                "feature_measure_robust_spatial_delta": totals["feature_measure_robust_spatial_delta"] / num_samples,
                "feature_measure_robust_spatial_rel_delta": totals["feature_measure_robust_spatial_rel_delta"] / num_samples,
                "feature_measure_robust_hybrid_delta": totals["feature_measure_robust_hybrid_delta"] / num_samples,
                "feature_measure_robust_hybrid_rel_delta": totals["feature_measure_robust_hybrid_rel_delta"] / num_samples,
                "fsp_strict_delta": totals["fsp_strict_delta"] / num_samples,
                "fsp_delta": totals["fsp_delta"] / num_samples,
                "clean_acc": totals["clean_acc"] / num_samples,
                "adv_acc": totals["adv_acc"] / num_samples,
                "num_layers": num_layers,
                "activation_mode": args.activation_mode,
                "max_layers": args.max_layers,
            }
        if args.include_published_nas_metrics:
            clean_inputs = torch.cat([item[0] for item in published_batches])[
                : args.published_metric_samples
            ]
            labels = torch.cat([item[1] for item in published_batches])[
                : args.published_metric_samples
            ]
            adversarial_inputs = torch.cat([item[2] for item in published_batches])[
                : args.published_metric_samples
            ]
            racl = racl_fixed_network_lipschitz_stats(model)
            trnas = compute_trnas_metrics(
                model,
                clean_inputs,
                adversarial_inputs,
                labels,
                weight_step_size=args.trnas_weight_step_size,
            )
            row.update(
                {
                    "racl_fixed_network_log_upper_proxy": racl.log_upper,
                    "racl_input_gradient_lower": totals["input_grad_l2"] / num_samples,
                    "racl_upper_mode": racl.mode,
                    "racl_upper_num_cells": racl.num_cells,
                    "racl_upper_num_nodes": racl.num_nodes,
                    "racl_zero_node_sums": racl.zero_node_sums,
                    "trnas_activation_psi": trnas.activation_psi,
                    "trnas_consistency_phi": trnas.consistency_phi,
                    "trnas_r_score": trnas.r_score,
                    "trnas_num_relu_units": trnas.num_relu_units,
                    "trnas_num_consistency_layers": trnas.num_consistency_layers,
                    "trnas_weight_step_size": args.trnas_weight_step_size,
                    "published_metric_samples": clean_inputs.shape[0],
                }
            )
        rows.append(row)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
