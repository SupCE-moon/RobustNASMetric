"""E34 analysis: rank the e0->e5 ranking stability of every metric from the
local NB201 adversarial-training run, to confirm E33's prediction that
dispersion-magnitude capacity variants flip (negative/weak) while gram stays
stable (positive).

Run:  python scripts/analyze_e34_stability.py --csv outputs/e34_nb201_e5_stability_pgd1.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def rho_between(df: pd.DataFrame, metric: str, e1: int, e2: int) -> float:
    a = df[df.train_epochs == e1][["arch_id", metric]].rename(columns={metric: metric + "_a"})
    b = df[df.train_epochs == e2][["arch_id", metric]].rename(columns={metric: metric + "_b"})
    m = a.merge(b, on="arch_id").dropna()
    if len(m) < 8 or m[metric + "_a"].std() == 0 or m[metric + "_b"].std() == 0:
        return np.nan
    r, _ = stats.spearmanr(m[metric + "_a"], m[metric + "_b"])
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/e34_nb201_e5_stability_pgd1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    epochs = sorted(int(e) for e in df["train_epochs"].unique())
    print(f"arches={df['arch_id'].nunique()}  epochs={epochs}")

    metric_cols = [c for c in df.columns if c not in
                   {"arch_id", "oracle_target", "train_epochs", "num_layers",
                    "num_candidate_layers"}]
    rows = []
    for m in metric_cols:
        s05 = rho_between(df, m, epochs[0], epochs[-1])
        if np.isnan(s05):
            continue
        rows.append((s05, m))
    rows.sort()

    print(f"\n{'metric':46s} {f'e{epochs[0]}->e{epochs[-1]}':>10s}   verdict")
    print("-" * 76)
    for r, m in rows:
        verdict = "STABLE (gram-like)" if r > 0.5 else ("FLIPS (capacity-like)" if r < 0 else "weak/unstable")
        print(f"{m:46s} {r:+10.3f}   {verdict}")


if __name__ == "__main__":
    main()
