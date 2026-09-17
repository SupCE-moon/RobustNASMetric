"""E27 NB201 zero-cost evaluation: compute all metrics on random-init NB201 archs.

Validates the reimplemented NB201 network by reproducing feature_delta's
correlation with oracle_target (~0.806 in E23), then reports every new metric.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.nasrobbench201 import parse_nb201_arch
from robust_metric_lab.nb201_space import Nb201TinyNetwork

EPSILON = 8 / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv")
    parser.add_argument("--data-root", default="D:/Project/RobustNAS/DynamicBN/DynamicBN20251203/data")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "fixed"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/e27_nb201_e0_zerocost.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    e0 = df[df["train_epochs"] == 0].drop_duplicates(subset=["arch_id"])
    print(f"NB201 e0 architectures: {len(e0)}")

    loader = get_dataset_loader(
        dataset_name="cifar10", data_root=args.data_root,
        batch_size=args.samples, num_workers=0, limit=args.samples,
    )
    x, y = next(iter(loader))
    x, y = x.to(args.device), y.to(args.device)

    rows = []
    for _, row in e0.iterrows():
        ops = parse_nb201_arch(str(row["arch_string"]))
        if ops is None:
            continue
        torch.manual_seed(args.weight_seed)
        model = Nb201TinyNetwork(ops).to(args.device)
        x_adv = make_adversarial(model, x, y, args.attack, EPSILON, pgd_steps=1)
        result = compute_feature_metrics(model, x, x_adv, y, max_layers=4)
        out = {"arch_id": row["arch_id"], "oracle_target": row["oracle_target"]}
        for k, v in result.__dict__.items():
            if isinstance(v, (int, float)):
                out[k] = v
        rows.append(out)
        del model, x_adv
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Validation: feature_delta should correlate ~0.806 with oracle_target.
    metric_cols = [c for c in out_df.columns if c not in ("arch_id", "oracle_target", "num_layers", "num_candidate_layers")]
    print("\n=== Validation + new-metric correlations (raw Spearman vs oracle_target) ===")
    for m in metric_cols:
        valid = out_df[[m, "oracle_target"]].dropna()
        if len(valid) < 3 or valid[m].std() == 0:
            continue
        rho, p = stats.spearmanr(valid[m], valid["oracle_target"])
        flag = "  <-- VALIDATION" if m == "feature_delta" else ""
        print(f"  {m:50s} rho={rho:+.4f}  p={p:.3f}{flag}")


if __name__ == "__main__":
    main()
