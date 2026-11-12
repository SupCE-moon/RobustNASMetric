"""
E27: Static spectral concentration + Jacobian spectrum + signed capacity.

Computes direction-adjusted Spearman + bootstrap CI against RobustBench robust
accuracy for every metric in a metrics CSV, using metric_registry directions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.metric_registry import METRIC_BY_NAME


NON_METRIC_COLS = {
    "model_name", "dataset", "threat_model", "attack", "attack_source",
    "adv_cache", "batch_size", "num_samples", "num_layers", "activation_mode",
    "max_layers", "clean_acc", "adv_acc", "rb_clean_acc", "rb_robust_acc",
    "rank",
}

NEW_METRICS = [
    "perturbation_capacity_signed_delta",
    "feature_spectral_erank_static",
    "feature_spectral_participation_ratio_static",
    "feature_spectral_decay_exponent_static",
    "jacobian_spectral_norm",
    "jacobian_effective_rank",
    "jacobian_spectral_decay",
]


def direction_for(metric: str) -> str:
    spec = METRIC_BY_NAME.get(metric)
    if spec is not None:
        return spec.direction
    return "lower"


def compute_spearman(df, metric, target, direction):
    valid = df[[metric, target]].dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(valid[metric], valid[target])
    return (-rho if direction == "lower" else rho), p


def bootstrap_ci(df, metric, target, direction, n_bootstrap=2000, seed=20260910):
    rng = np.random.RandomState(seed)
    valid = df[[metric, target]].dropna()
    estimates = []
    n = len(valid)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        samp = valid.iloc[idx]
        rho, _ = compute_spearman(samp, metric, target, direction)
        if not np.isnan(rho):
            estimates.append(rho)
    if not estimates:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(estimates)),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--scores-csv", default="outputs/robustbench_cifar10_linf_scores.csv")
    parser.add_argument("--target", default="rb_robust_acc")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    scores = pd.read_csv(args.scores_csv)
    score_map = {row["model_name"]: row[args.target] for _, row in scores.iterrows()}
    df[args.target] = df["model_name"].map(score_map)
    df = df.dropna(subset=[args.target])
    print(f"Loaded {len(df)} models with target {args.target}")

    metric_cols = [c for c in df.columns if c not in NON_METRIC_COLS]
    rows = []
    for metric in metric_cols:
        direction = direction_for(metric)
        rho, p = compute_spearman(df, metric, args.target, direction)
        mean_b, lo, hi = bootstrap_ci(df, metric, args.target, direction)
        rows.append({
            "metric": metric, "direction": direction, "spearman": rho,
            "p_value": p, "bootstrap_mean": mean_b,
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "is_new": metric in NEW_METRICS,
        })

    results = pd.DataFrame(rows).sort_values("spearman", ascending=False, na_position="last")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("\n=== ALL METRICS RANKED (direction-adjusted Spearman) ===")
    for i, r in results.iterrows():
        tag = " *** NEW" if r["is_new"] else ""
        print(f"{int(results.index.get_loc(i))+1:3d}. {r['metric']:45s} rho={r['spearman']:+.4f}  "
              f"CI=[{r['bootstrap_ci_low']:+.3f},{r['bootstrap_ci_high']:+.3f}]{tag}")

    print("\n=== NEW METRIC HIGHLIGHTS ===")
    for metric in NEW_METRICS:
        if metric in results["metric"].values:
            row = results[results["metric"] == metric].iloc[0]
            rank = int(results.index.get_loc(results[results["metric"] == metric].index[0]) + 1)
            print(f"  {metric:45s} rank={rank:2d}/{len(results)}  rho={row['spearman']:+.4f}  "
                  f"CI=[{row['bootstrap_ci_low']:+.3f},{row['bootstrap_ci_high']:+.3f}]")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
