"""E27 validation: capacity-family metrics across RB20 / NB201-e0 / DARTS-e0.

Direction-adjusted Spearman + bootstrap CI + structural confound (partial
Spearman controlling for NB201 op counts / parameter count).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.nasrobbench201 import parse_nb201_arch

NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

CAPACITY_METRICS = [
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "perturbation_capacity_multiscale_delta",
    "perturbation_capacity_relu_active_multiscale_delta",
    "perturbation_capacity_topk_delta",
    "perturbation_capacity_iqr_delta",
    "perturbation_capacity_iqr_active_delta",
    "perturbation_capacity_importance_delta",
    "perturbation_capacity_gini_delta",
    "perturbation_capacity_channelwise_delta",
    "perturbation_capacity_layerprofile_delta",
    "perturbation_capacity_flow_rel_delta",
]
REFERENCE = ["feature_delta", "feature_measure_mmd", "feature_measure_swd", "feature_measure_flow_rel_delta"]

NB201_OP_COLS = [
    "nb201_op_nor_conv_3x3", "nb201_op_nor_conv_1x1", "nb201_op_avg_pool_3x3",
    "nb201_op_skip_connect", "nb201_op_none",
]


def direction_for(m):
    spec = METRIC_BY_NAME.get(m)
    return spec.direction if spec else "lower"


def spearman(df, m, target):
    valid = df[[m, target]].dropna()
    if len(valid) < 3 or valid[m].std() == 0:
        return np.nan
    rho, _ = stats.spearmanr(valid[m], valid[target])
    d = direction_for(m)
    return -rho if d == "lower" else rho


def bootstrap_ci(df, m, target, n=2000, seed=20260911):
    rng = np.random.RandomState(seed)
    valid = df[[m, target]].dropna()
    est = []
    for _ in range(n):
        idx = rng.choice(len(valid), len(valid), replace=True)
        r = spearman(valid.iloc[idx], m, target)
        if not np.isnan(r):
            est.append(r)
    if not est:
        return np.nan, np.nan
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def partial_spearman(df, m, target, controls):
    cols = [m, target] + controls
    valid = df[cols].dropna()
    if len(valid) < len(controls) + 3:
        return np.nan
    X = valid[controls].values
    ones = np.ones((len(X), 1))
    Xm = np.column_stack([ones, X])
    beta_m, *_ = np.linalg.lstsq(Xm, valid[m].values, rcond=None)
    beta_t, *_ = np.linalg.lstsq(Xm, valid[target].values, rcond=None)
    r_m = valid[m].values - Xm @ beta_m
    r_t = valid[target].values - Xm @ beta_t
    rho, _ = stats.spearmanr(r_m, r_t)
    d = direction_for(m)
    return -rho if d == "lower" else rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb20", default="outputs/e27_rb20_whitebox_fgsm_n64_v3.csv")
    ap.add_argument("--nb201", default="outputs/e27_nb201_e0_zerocost_v2.csv")
    ap.add_argument("--darts", default="outputs/e27_darts_e0_zerocost_v2.csv")
    ap.add_argument("--nb201-orig", default="outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv")
    ap.add_argument("--rb20-scores", default="outputs/robustbench_cifar10_linf_scores.csv")
    args = ap.parse_args()

    # --- RB20 ---
    rb = pd.read_csv(args.rb20)
    scores = pd.read_csv(args.rb20_scores)
    rb["target"] = rb["model_name"].map(dict(zip(scores.model_name, scores.rb_robust_acc)))
    print("=" * 90)
    print("RB20 (mature, white-box FGSM)")
    print("-" * 90)
    for m in CAPACITY_METRICS + REFERENCE:
        rho = spearman(rb, m, "target")
        lo, hi = bootstrap_ci(rb, m, "target")
        star = "  <-- NEW (multiscale)" if m == "perturbation_capacity_multiscale_delta" else ""
        print(f"  {m:50s} rho={rho:+.4f}  CI=[{lo:+.3f},{hi:+.3f}]{star}")

    # --- NB201 ---
    nb = pd.read_csv(args.nb201)
    nb_orig = pd.read_csv(args.nb201_orig).drop_duplicates("arch_id")
    op_rows = []
    for arch_str in nb_orig["arch_string"]:
        ops = parse_nb201_arch(str(arch_str)) or ("", "", "", "", "", "")
        op_rows.append({f"op_{op}": ops.count(op) for op in NB201_OPS})
    op_df = pd.DataFrame(op_rows, index=nb_orig.index)
    nb_orig = pd.concat([nb_orig[["arch_id"]], op_df], axis=1)
    op_cols = [f"op_{op}" for op in NB201_OPS]
    nb = nb.merge(nb_orig, on="arch_id", how="left")
    print("\n" + "=" * 90)
    print("NB201 e0 (zero-cost)")
    print("-" * 90)
    for m in CAPACITY_METRICS + REFERENCE:
        rho = spearman(nb, m, "oracle_target")
        lo, hi = bootstrap_ci(nb, m, "oracle_target")
        part_ops = partial_spearman(nb, m, "oracle_target", op_cols)
        star = "  <-- NEW (multiscale)" if m == "perturbation_capacity_multiscale_delta" else ""
        print(f"  {m:50s} rho={rho:+.4f} CI=[{lo:+.3f},{hi:+.3f}]  partial(ops)={part_ops:+.4f}{star}")

    # --- DARTS ---
    darts = pd.read_csv(args.darts)
    if "final_pgd_accuracy" in darts.columns:
        print("\n" + "=" * 90)
        print("DARTS e0 (zero-cost)")
        print("-" * 90)
        print(f"  target variance: std={darts['final_pgd_accuracy'].std():.4f} (tiny -> weak test)")
        for m in CAPACITY_METRICS + REFERENCE:
            if m in darts.columns:
                rho = spearman(darts, m, "final_pgd_accuracy")
                lo, hi = bootstrap_ci(darts, m, "final_pgd_accuracy")
                print(f"  {m:50s} rho={rho:+.4f}  CI=[{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
