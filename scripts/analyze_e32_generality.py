"""E32 generality analysis: signed-capacity endpoint consistency + deconfounding.

Tests E31's mechanism hypothesis: unsigned capacity takes |·| and drops the
sign/structure of the drift, which is why it flips at NB201 e5. The signed
counterpart `perturbation_capacity_signed_delta = (adv-clean)/|clean|` keeps the
sign. This script checks, on data we ALREADY have, whether signed capacity is
direction-consistent at both the zero-cost endpoint (NB201 e0) and the mature
endpoint (RB20), and whether it survives op-count deconfounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.nasrobbench201 import parse_nb201_arch

NB201_OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

# The hypothesis metric first, unsigned capacity for contrast, then gram family.
METRICS = [
    "perturbation_capacity_signed_delta",
    "perturbation_capacity_delta",
    "fsp_delta",
    "feature_measure_moment_delta",
    "feature_measure_struct01_delta",
    "feature_measure_fsp_rel_multiscale_delta",
]


def direction(m: str) -> str:
    spec = METRIC_BY_NAME.get(m)
    return spec.direction if spec else "lower"


def adj_rho(v: pd.DataFrame, m: str, target: str) -> float:
    d = v[[m, target]].dropna()
    if len(d) < 3 or d[m].std() == 0:
        return np.nan
    rho, _ = stats.spearmanr(d[m], d[target])
    return -rho if direction(m) == "lower" else rho


def bootstrap_ci(v: pd.DataFrame, m: str, target: str, n=2000, seed=20260912):
    rng = np.random.RandomState(seed)
    d = v[[m, target]].dropna()
    est = []
    for _ in range(n):
        idx = rng.choice(len(d), len(d), replace=True)
        r = adj_rho(d.iloc[idx], m, target)
        if not np.isnan(r):
            est.append(r)
    if not est:
        return np.nan, np.nan
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def partial(v: pd.DataFrame, m: str, target: str, controls: list[str]) -> float:
    cols = [m, target] + controls
    d = v[cols].dropna()
    if len(d) < len(controls) + 3:
        return np.nan
    X = d[controls].values
    Xm = np.column_stack([np.ones(len(X)), X])
    bm, *_ = np.linalg.lstsq(Xm, d[m].values, rcond=None)
    bt, *_ = np.linalg.lstsq(Xm, d[target].values, rcond=None)
    rm = d[m].values - Xm @ bm
    rt = d[target].values - Xm @ bt
    rho, _ = stats.spearmanr(rm, rt)
    return -rho if direction(m) == "lower" else rho


def nb201_with_opcounts(csv: str) -> pd.DataFrame:
    """Merge NB201 metric CSV with per-arch op counts (for deconfounding)."""
    nb = pd.read_csv(csv)
    src = pd.read_csv(
        "outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv"
    ).drop_duplicates("arch_id")
    rows = []
    for a in src["arch_string"]:
        ops = parse_nb201_arch(str(a)) or ("", "", "", "", "", "")
        rows.append({f"op_{o}": ops.count(o) for o in NB201_OPS})
    op_df = pd.DataFrame(rows, index=src.index)
    src = pd.concat([src[["arch_id"]], op_df], axis=1)
    return nb.merge(src, on="arch_id", how="left")


def main() -> None:
    op_cols = [f"op_{o}" for o in NB201_OPS]

    # ---- NB201 e0 (zero-cost endpoint) ----
    nb = nb201_with_opcounts("outputs/e28_capacity_family/nb201_e0_v3.csv")
    print("=" * 78)
    print("NB201 e0 (zero-cost endpoint) vs oracle_target, n=%d" % len(nb))
    print("=" * 78)
    print(f"{'metric':44s} {'raw_rho':>9s} {'adj_rho':>9s} {'partial_ops':>12s} {'CI':>20s}")
    for m in METRICS:
        if m not in nb.columns:
            print(f"{m:44s}  MISSING")
            continue
        d = nb[[m, "oracle_target"]].dropna()
        raw, _ = stats.spearmanr(d[m], d["oracle_target"]) if len(d) >= 3 else (np.nan, np.nan)
        a = adj_rho(nb, m, "oracle_target")
        p = partial(nb, m, "oracle_target", op_cols)
        lo, hi = bootstrap_ci(nb, m, "oracle_target")
        print(f"{m:44s} {raw:+9.4f} {a:+9.4f} {p:+12.4f} [{lo:+.3f},{hi:+.3f}]")

    # ---- RB20 (mature endpoint) ----
    rb = pd.read_csv("outputs/e28_capacity_family/rb20_whitebox_v3.csv")
    print()
    print("=" * 78)
    print("RB20 (mature endpoint) vs adv_acc, n=%d" % len(rb))
    print("=" * 78)
    print(f"{'metric':44s} {'raw_rho':>9s} {'adj_rho':>9s} {'CI':>20s}")
    for m in METRICS:
        if m not in rb.columns:
            print(f"{m:44s}  MISSING")
            continue
        d = rb[[m, "adv_acc"]].dropna()
        raw, _ = stats.spearmanr(d[m], d["adv_acc"]) if len(d) >= 3 else (np.nan, np.nan)
        a = adj_rho(rb, m, "adv_acc")
        lo, hi = bootstrap_ci(rb, m, "adv_acc")
        print(f"{m:44s} {raw:+9.4f} {a:+9.4f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
