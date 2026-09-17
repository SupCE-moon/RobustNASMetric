"""E35 补充分析：E27+ 新指标的三空间方向校正 Spearman，与 gram 族对照。

重新分析 DARTS 上 E27 新指标的表现（此前被「DARTS 不可测」误判跳过），
并与 gram 族参照指标在 NB201 e0 / RB20 / DARTS e0 三空间并列对比。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# direction = "higher" 的指标；其余视为 lower_is_better
HIGHER = {
    "feature_spectral_decay_exponent_static",
    "jacobian_spectral_decay",
}

# E27 在 DARTS 上有数据的新指标（12 个）
NEW_METRICS = [
    "perturbation_capacity_signed_delta",
    "perturbation_capacity_iqr_delta",
    "perturbation_capacity_multiscale_delta",
    "perturbation_capacity_relu_active_multiscale_delta",
    "perturbation_capacity_topk_delta",
    "perturbation_capacity_iqr_active_delta",
    "feature_spectral_erank_static",
    "feature_spectral_participation_ratio_static",
    "feature_spectral_decay_exponent_static",
    "jacobian_spectral_norm",
    "jacobian_effective_rank",
    "jacobian_spectral_decay",
]

# gram 族参照指标（现在的最好指标）
GRAM_REF = [
    "fsp_delta",
    "feature_measure_struct01_delta",
    "feature_measure_moment_delta",
    "feature_measure_flow_rel_delta",
    "feature_delta",
]


def adj_spearman(df: pd.DataFrame, metric: str, target: str) -> float:
    d = df[[metric, target]].dropna()
    if len(d) < 3:
        return np.nan
    rho = spearmanr(d[metric], d[target])[0]
    return rho if metric in HIGHER else -rho


def main() -> None:
    # DARTS：用 E23 的 10,000 样本 full-test PGD-20 作为 target（更可靠，与跨空间总表一致）
    darts = pd.read_csv("outputs/e27_darts_e0_zerocost_v3.csv")
    full = pd.read_csv("outputs/e23_cross_space/darts20/metrics_20_e0e50e100_fulltest_pgd20.csv")
    f100 = full[full["train_epochs"] == 100][["arch_id", "full_pgd20_accuracy"]]
    darts = darts.merge(f100, on="arch_id", how="left")

    nb201 = pd.read_csv("outputs/e28_capacity_family/nb201_e0_v3.csv")
    rb20 = pd.read_csv("outputs/e28_capacity_family/rb20_whitebox_v3.csv")

    frames = [
        ("NB201 e0", nb201, "oracle_target"),
        ("RB20", rb20, "adv_acc"),
        ("DARTS e0", darts, "full_pgd20_accuracy"),
    ]

    all_metrics = GRAM_REF + NEW_METRICS
    rows = []
    for m in all_metrics:
        row = {"metric": m}
        for label, df, target in frames:
            row[label] = adj_spearman(df, m, target)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.round(4)

    # 打印：新指标表
    print("=== 新指标（E27+，DARTS 有数据）三空间方向校正 Spearman ===")
    new = out[out["metric"].isin(NEW_METRICS)]
    print(new.to_string(index=False))

    # 打印：gram 参照
    print("\n=== gram 族参照（同一数据源、同一 target）三空间 ===")
    gram = out[out["metric"].isin(GRAM_REF)]
    print(gram.to_string(index=False))

    # 保存合并表
    out.to_csv("outputs/e35_newmetrics_three_space.csv", index=False)
    print("\nwrote outputs/e35_newmetrics_three_space.csv")


if __name__ == "__main__":
    main()
