"""E33 reproducibility: generality mechanism = scale-invariance.

Three analyses on existing CSVs (no training):
  1. Cross-training ranking stability (e0->e5, e0->e50) from the e23 5-epoch CSV.
  2. Cross-protocol consistency (e23 vs reimplemented nb201_e0_v3) per metric.
  3. Capacity-variant characterization: magnitude-like (~plain capacity) vs
     structure-like (~gram).

Run:  python scripts/analyze_e33_mechanism.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.metric_registry import METRIC_BY_NAME

E23_CSV = "outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv"
V3_CSV = "outputs/e28_capacity_family/nb201_e0_v3.csv"
RB20_CSV = "outputs/e28_capacity_family/rb20_whitebox_v3.csv"


def rho_between(df: pd.DataFrame, metric: str, e1: int, e2: int) -> float:
    a = df[df.train_epochs == e1][["arch_id", metric]].rename(columns={metric: metric + "_a"})
    b = df[df.train_epochs == e2][["arch_id", metric]].rename(columns={metric: metric + "_b"})
    m = a.merge(b, on="arch_id").dropna()
    if len(m) < 3 or m[metric + "_a"].std() == 0 or m[metric + "_b"].std() == 0:
        return np.nan
    r, _ = stats.spearmanr(m[metric + "_a"], m[metric + "_b"])
    return r


def main() -> None:
    e23 = pd.read_csv(E23_CSV)
    e23e0 = e23[e23.train_epochs == 0].drop_duplicates("arch_id")
    v3 = pd.read_csv(V3_CSV)
    rb = pd.read_csv(RB20_CSV)

    # ---- 1. cross-training ranking stability ----
    print("=" * 70)
    print("1) Cross-training ranking stability (e23 CSV)")
    print("=" * 70)
    for m in ["perturbation_capacity_delta", "fsp_delta",
              "feature_measure_struct01_delta", "feature_measure_moment_delta",
              "adversarial_loss"]:
        s05 = rho_between(e23, m, 0, 5)
        s050 = rho_between(e23, m, 0, 50)
        print(f"  {m:38s} e0->e5={s05:+.3f}  e0->e50={s050:+.3f}")

    # ---- 2. cross-protocol consistency ----
    print()
    print("=" * 70)
    print("2) Cross-protocol consistency (e23 vs reimplemented v3), same 64 archs")
    print("=" * 70)
    num_e23 = [c for c in e23e0.columns if pd.api.types.is_numeric_dtype(e23e0[c])]
    num_v3 = [c for c in v3.columns if pd.api.types.is_numeric_dtype(v3[c])]
    exclude = {"arch_id", "oracle_target", "oracle_val_acc", "canonical_id",
               "train_epochs", "num_layers", "clean_acc", "adv_acc",
               "train_samples", "metric_samples", "sampled_arch_count",
               "arch_start", "arch_stop"}
    common = [m for m in num_e23 if m in num_v3 and m not in exclude]
    rows = []
    for m in common:
        mm = e23e0[["arch_id", m]].merge(v3[["arch_id", m]], on="arch_id",
                                         suffixes=("_e23", "_v3")).dropna()
        if len(mm) < 3 or mm[m + "_e23"].std() == 0 or mm[m + "_v3"].std() == 0:
            continue
        r, _ = stats.spearmanr(mm[m + "_e23"], mm[m + "_v3"])
        rows.append((r, m))
    rows.sort()
    for r, m in rows:
        tag = "ROBUST" if r > 0.5 else ("FRAGILE" if r < 0 else "intermediate")
        print(f"  {m:44s} {r:+8.3f}  {tag}")

    # ---- 3. capacity-variant characterization ----
    print()
    print("=" * 70)
    print("3) New capacity variants: magnitude-like (~plain capacity) vs structure (~gram)")
    print("=" * 70)
    def s(a, b):
        v = v3[[a, b]].dropna()
        if len(v) < 3 or v[a].std() == 0:
            return np.nan
        return stats.spearmanr(v[a], v[b])[0]
    new_cap = [
        "perturbation_capacity_multiscale_delta",
        "perturbation_capacity_relu_active_multiscale_delta",
        "perturbation_capacity_topk_delta",
        "perturbation_capacity_channelwise_delta",
        "perturbation_capacity_gini_delta",
        "perturbation_capacity_iqr_delta",
        "perturbation_capacity_iqr_active_delta",
        "perturbation_capacity_flow_rel_delta",
        "perturbation_capacity_layerprofile_delta",
        "perturbation_capacity_signed_delta",
    ]
    print(f"  {'variant':46s} {'~capacity':>10s} {'~gram':>10s}")
    for c in new_cap:
        if c not in v3.columns:
            continue
        print(f"  {c:46s} {s(c, 'perturbation_capacity_delta'):+10.3f} {s(c, 'fsp_delta'):+10.3f}")


if __name__ == "__main__":
    main()
