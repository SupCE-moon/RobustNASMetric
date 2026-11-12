# 严格排名离线复算（从原始行重建）

生成时间（UTC）：2026-11-12T15:47:04+00:00

- 原始行：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv`（47,600 行，sha256 `b5137e2b3fe89ce5…`）
- 官方排名表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\.worktrees\mainline-consolidation\outputs\strict_adv_excess_release_20261024T110422Z\strict_ranking.csv`
- 复算成功指标：**83**；被排除：2
- 与官方逐值比较：83 项，最大绝对误差 `0.000e+00`（trnas_activation_psi::all_setting_mean）
- 逐值完全一致（≤1e-09）：**True**
- 共享子集内名次顺序一致：**True**

## 官方表里有、这批原始行里没有的指标

- `feature_adv_excess_centered_unit_l2`

这些指标是在这批原始行冻结之后才加入的，其原始行只存在于服务器侧，离线无法复算。

## 被排除的指标

- `racl_arch_lipschitz_log_upper`：missing or degenerate settings: rb40_mature (0/40 ok rows)
- `racl_fixed_network_log_upper_proxy`：missing or degenerate settings: nb201_e0 (0/64 ok rows), nb201_e5 (0/64 ok rows), nb201_e10 (0/64 ok rows), nb201_e25 (0/64 ok rows), nb201_e50 (0/64 ok rows), darts_e0 (0/100 ok rows), darts_e100 (0/100 ok rows)
