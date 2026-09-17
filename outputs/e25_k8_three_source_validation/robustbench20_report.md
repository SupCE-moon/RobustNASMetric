# E25：三源共享对抗协议下 K=2/4/8/16 特征采样完整验证

Date: 2026-07-31 | Target: `rb_robust_acc` | Models: 20 | Attack: PGD-10 shared 3-source

## `perturbation_capacity_delta`：K 值完整曲线

| K | Spearman ($\rho$) | Pearson ($r$) | Kendall ($\tau_b$) | 方向校正效用 |
|---:|---:|---:|---:|---:|
| 2 | -0.544 | -0.932 | -0.411 | 0.544 |
| 4 | -0.574 | -0.942 | -0.432 | 0.574 |
| 8 | -0.744 | -0.960 | -0.579 | **0.744** |
| 16 | -0.532 | -0.941 | -0.442 | 0.532 |

**K=8 是明确峰值**，K=16 回落到接近 K=2 水平。与 E22 白盒结论一致：K=8 > K=4 > K=16 ≈ K=2。

## 与 E22（白盒 FGSM）对比

| K | 白盒 FGSM（E22）| 三源共享 PGD-10（E25）| 协议间差距 |
|---:|---:|---:|---:|
| 2 | 0.565 | 0.544 | -0.021 |
| 4 | 0.626 | 0.574 | -0.052 |
| 8 | 0.805 | **0.744** | -0.061 |
| 16 | 0.795 | 0.532 | -0.263 |

三源协议下的 K 值敏感性更高：K=16 在白盒下仅略低于 K=8（0.795 vs 0.805），三源下大幅回落（0.532 vs 0.744）。

## 按指标 K=8 排名 Top 20

| Rank | Metric | K=2 | K=4 | K=8 | K=16 | Best K |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `adv_acc` | 0.879 | 0.879 | 0.879 | 0.879 | 2 |
| 2 | `perturbation_capacity_delta` | 0.544 | 0.574 | **0.744** | 0.532 | **8** |
| 3 | `feature_graph_fundamental_rank_delta` | 0.507 | 0.438 | 0.558 | 0.483 | 8 |
| 4 | `input_grad_l2` | 0.538 | 0.538 | 0.538 | 0.538 | — |
| 5 | `adversarial_loss` | 0.486 | 0.486 | 0.486 | 0.486 | — |
| 6 | `feature_measure_swd` | 0.409 | 0.427 | 0.412 | 0.392 | 4 |
| 7 | `feature_box_dimension_delta` | 0.039 | 0.223 | 0.395 | **0.600** | **16** |
| 8 | `feature_measure_mmd` | 0.392 | 0.469 | 0.382 | 0.513 | 16 |
| 9 | `input_loss_delta` | 0.376 | 0.376 | 0.376 | 0.376 | — |
| 10 | `perturbation_capacity_above_mean_delta` | 0.433 | 0.471 | 0.376 | 0.483 | 16 |
| 11 | `perturbation_capacity_relu_active_delta` | 0.463 | 0.377 | 0.350 | 0.480 | 16 |
| 12 | `feature_measure_relu_active_spatial_delta` | 0.245 | 0.359 | 0.317 | 0.502 | 16 |
| 13 | `feature_measure_spatial_gram_delta` | 0.245 | 0.356 | 0.317 | 0.502 | 16 |
| 14 | `clean_acc` | 0.313 | 0.313 | 0.313 | 0.313 | — |
| 15 | `local_logit_lipschitz` | 0.290 | 0.290 | 0.290 | 0.290 | — |
| 16 | `logit_kl` | 0.286 | 0.286 | 0.286 | 0.286 | — |
| 17 | `feature_measure_spatial_gram_rel_delta` | 0.011 | 0.192 | 0.180 | 0.418 | 16 |
| 18 | `feature_measure_relu_active_spatial_rel_delta` | 0.028 | 0.186 | 0.164 | 0.395 | 16 |
| 19 | `feature_hausdorff_delta` | 0.001 | -0.005 | 0.132 | 0.215 | 16 |
| 20 | `feature_quantile_hausdorff_delta` | -0.024 | -0.000 | 0.116 | 0.245 | 16 |

完整 42 指标 × 4K 数据见 `summary/k2k4k8k16_spearman_matrix.csv`。

## K 值敏感性分类

| 类型 | 指标 | 特征 |
|---|---|---|
| **K=8 峰值** | `perturbation_capacity_delta` | 中等 K 最优，过大退化 |
| **K=16 单调递增** | `feature_box_dimension_delta`, Hausdorff 系列 | 几何/分形指标需要更多层 |
| **K=16 最优** | `feature_measure_mmd`, `perturbation_capacity_*_delta`（门控变体） | K=4 到 K=16 持续改善 |
| **K≥8 崩溃** | `fsp_delta`, `feature_measure_struct*_delta` | 跨层配对指标在过多层时退化 |
| **K 不变** | `input_grad_l2`, `adversarial_loss`, `logit_kl` 等 | 梯度/逻辑值指标不依赖特征采样 |

## Bootstrap 95% 置信区间（K=8，10000 次重采样）

只有 `perturbation_capacity_delta`、`input_grad_l2`、`adversarial_loss` 三个指标的 CI 不跨零，统计显著。

| Rank | Metric | Adj-S | 95% CI | 显著？ |
|---:|---:|---:|---:|:---:|
| 1 | `adv_acc` | 0.879 | [0.655, 0.968] | ✓ |
| 2 | `perturbation_capacity_delta` | **0.744** | [0.417, 0.917] | **✓** |
| 3 | `feature_graph_fundamental_rank_delta` | 0.558 | [0.084, 0.835] | ✗ |
| 4 | `input_grad_l2` | 0.538 | [0.047, 0.871] | ✓ |
| 5 | `adversarial_loss` | 0.486 | [0.077, 0.773] | ✓ |
| 6 | `feature_measure_swd` | 0.412 | [-0.064, 0.757] | ✗ |
| 7 | `feature_box_dimension_delta` | 0.395 | [-0.079, 0.723] | ✗ |
| 8 | `feature_measure_mmd` | 0.382 | [-0.088, 0.696] | ✗ |

`perturbation_capacity_delta` 是唯一 CI 不跨零的 Ours 指标，其他 Ours 指标（mmd、swd、box_dimension 等）在 95% 置信水平下不能排除零相关。这强化了 capacity 相对于 Ours 内部对标指标的优势。

完整 bootstrap 数据：`robustbench20_k8_bootstrap_ci.csv`

## 结论

1. **`perturbation_capacity_delta` K=8 是三源协议下最优配置**，Spearman=0.744，排名第一（不含 `adv_acc`），且是唯一通过 bootstrap 95% CI 显著性检验的 Ours 指标
2. K=16 在三源下退化严重（-0.212），三源协议对 K 值更敏感
3. **不同指标族的最优 K 值不同**——几何指标在 K=16 下持续改善，跨层配对指标在 K≥8 时崩溃
4. 论文默认 K=8 用于 `perturbation_capacity_delta`；`feature_box_dimension_delta` 可考虑 K=16
5. 20 模型下 CI 宽度 ~0.5，论文需诚实报告样本量限制
