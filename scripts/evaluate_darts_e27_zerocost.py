"""
E27 zero-cost evaluation on DARTS: static spectral + Jacobian metrics at random init.

Reconstructs each DartsGenotype from its `signature`, instantiates the network
with random weights, computes the 7 new metrics on clean data (no attack), and
correlates them against final_pgd_accuracy (the oracle).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.darts_space import DartsGenotype, DartsNetworkCIFAR
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics

EPSILON = 8 / 255

NEW_METRICS = [
    "perturbation_capacity_signed_delta",
    "feature_spectral_erank_static",
    "feature_spectral_participation_ratio_static",
    "feature_spectral_decay_exponent_static",
    "jacobian_spectral_norm",
    "jacobian_effective_rank",
    "jacobian_spectral_decay",
]


def parse_gene(gene_str: str) -> list[tuple[str, int]]:
    inner = gene_str[gene_str.index("[") + 1 : gene_str.rindex("]")]
    return [
        (op, int(idx))
        for op, idx in (p.split("~") for p in inner.split(";") if p)
    ]


def parse_signature(sig: str) -> DartsGenotype:
    normal_str, reduce_str = sig.split("|")
    normal = parse_gene(normal_str)
    reduce = parse_gene(reduce_str)
    steps = len(normal) // 2
    concat = tuple(range(2, steps + 2))
    return DartsGenotype(
        normal=tuple(normal), normal_concat=concat,
        reduce=tuple(reduce), reduce_concat=concat,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/e23_cross_space/darts20/metrics_20_e0e50e100.csv")
    parser.add_argument("--data-root", default="D:/Project/RobustNAS/DynamicBN/DynamicBN20251203/data")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/e27_darts_e0_zerocost.csv")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv)
    e0 = df[df["train_epochs"] == 0].drop_duplicates(subset=["arch_id"])
    print(f"DARTS e0 architectures: {len(e0)}")

    loader = get_dataset_loader(
        dataset_name="cifar10", data_root=args.data_root,
        batch_size=args.samples, num_workers=0, limit=args.samples,
    )
    x, y = next(iter(loader))
    x = x.to(args.device)
    y = y.to(args.device)

    rows = []
    for _, row in e0.iterrows():
        genotype = parse_signature(row["signature"])
        channels = int(row["channels"])
        layers = int(row["layers"])
        torch.manual_seed(args.weight_seed)
        model = DartsNetworkCIFAR(genotype, channels=channels, layers=layers).to(args.device)
        x_adv = make_adversarial(model, x, y, "fgsm", EPSILON, pgd_steps=1)
        result = compute_feature_metrics(model, x, x_adv, y, max_layers=4)
        out = {
            "arch_id": row["arch_id"],
            "final_pgd_accuracy": row["final_pgd_accuracy"],
        }
        for k, v in result.__dict__.items():
            if isinstance(v, (int, float)):
                out[k] = v
        rows.append(out)
        print(f"  arch {row['arch_id']:2d}  pgd={row['final_pgd_accuracy']:.4f}  "
              f"erank={out['feature_spectral_erank_static']:.3f}  "
              f"jac_norm={out['jacobian_spectral_norm']:.3f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    from scipy import stats
    print("\n=== Zero-cost Spearman (raw, vs final_pgd_accuracy) ===")
    for m in NEW_METRICS:
        valid = out_df[[m, "final_pgd_accuracy"]].dropna()
        if len(valid) >= 3:
            rho, p = stats.spearmanr(valid[m], valid["final_pgd_accuracy"])
            print(f"  {m:45s} raw_rho={rho:+.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
