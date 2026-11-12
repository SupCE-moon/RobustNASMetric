"""Quick NB201-only deconfounding analysis for the new capacity metrics."""
import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.nasrobbench201 import parse_nb201_arch

NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

METRICS = [
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "perturbation_capacity_multiscale_delta",
    "perturbation_capacity_relu_active_multiscale_delta",
    "perturbation_capacity_topk_delta",
    "perturbation_capacity_iqr_delta",
    "perturbation_capacity_iqr_active_delta",
    "perturbation_capacity_profile_delta",
    "perturbation_capacity_importance_delta",
    "perturbation_capacity_gini_delta",
    "perturbation_capacity_channelwise_delta",
    "perturbation_capacity_layerprofile_delta",
    "perturbation_capacity_flow_rel_delta",
    "feature_delta",
    "feature_measure_flow_rel_delta",
    "feature_measure_mmd",
    "trnas_r_score",
]


def direction(m):
    spec = METRIC_BY_NAME.get(m)
    return spec.direction if spec else "lower"


def adj_rho(df, m, target):
    v = df[[m, target]].dropna()
    if len(v) < 3 or v[m].std() == 0:
        return np.nan
    rho, _ = stats.spearmanr(v[m], v[target])
    return -rho if direction(m) == "lower" else rho


def partial(df, m, target, controls):
    cols = [m, target] + controls
    v = df[cols].dropna()
    if len(v) < len(controls) + 3:
        return np.nan
    X = v[controls].values
    Xm = np.column_stack([np.ones(len(X)), X])
    bm, *_ = np.linalg.lstsq(Xm, v[m].values, rcond=None)
    bt, *_ = np.linalg.lstsq(Xm, v[target].values, rcond=None)
    rm = v[m].values - Xm @ bm
    rt = v[target].values - Xm @ bt
    rho, _ = stats.spearmanr(rm, rt)
    return -rho if direction(m) == "lower" else rho


def bootstrap_ci(df, m, target, n=2000, seed=20260911):
    rng = np.random.RandomState(seed)
    v = df[[m, target]].dropna()
    est = []
    for _ in range(n):
        idx = rng.choice(len(v), len(v), replace=True)
        r = adj_rho(v.iloc[idx], m, target)
        if not np.isnan(r):
            est.append(r)
    if not est:
        return np.nan, np.nan
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


nb = pd.read_csv("outputs/e28_capacity_family/nb201_e0.csv")
nb_orig = pd.read_csv("outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv").drop_duplicates("arch_id")
op_rows = []
for a in nb_orig["arch_string"]:
    ops = parse_nb201_arch(str(a)) or ("", "", "", "", "", "")
    op_rows.append({f"op_{o}": ops.count(o) for o in NB201_OPS})
op_df = pd.DataFrame(op_rows, index=nb_orig.index)
nb_orig = pd.concat([nb_orig[["arch_id"]], op_df], axis=1)
nb = nb.merge(nb_orig, on="arch_id", how="left")
op_cols = [f"op_{o}" for o in NB201_OPS]

print(f"{'metric':48s} {'raw_adj':>8s} {'partial_ops':>12s} {'CI':>20s}  survival")
for m in METRICS:
    if m not in nb.columns:
        print(f"{m:48s}  MISSING")
        continue
    r = adj_rho(nb, m, "oracle_target")
    p = partial(nb, m, "oracle_target", op_cols)
    lo, hi = bootstrap_ci(nb, m, "oracle_target")
    surv = (p / r * 100) if (r and not np.isnan(r) and not np.isnan(p)) else np.nan
    print(f"{m:48s} {r:+8.4f} {p:+12.4f} [{lo:+.3f},{hi:+.3f}]  {surv:5.0f}%")
