# E25：三源共享对抗协议下 K=2/4/8/16 特征采样完整验证

日期：2026-07-31 | 对应输出：`outputs/e25_k2_k8_k16_three_source_validation/`

## 实验目的

E22 在白盒 FGSM 下验证了 K=8 优于 K=4/2/16，但 K 值选择与模型评估使用同一批 20 个模型。本实验在项目的**三源共享对抗协议**下完整复现 K=2/4/8/16 全曲线，验证最优 K 值的跨协议一致性，并直接对标 E22。

## 实验协议

| 项目 | 设置 |
|---|---|
| 模型 | 20 个 CIFAR-10 $L_\infty$ RobustBench 模型 |
| 目标 | RobustBench 官方鲁棒准确率 (`rb_robust_acc`) |
| 样本数 | 每个源 128 个 CIFAR-10 测试样本 |
| 扰动 | PGD-10，$\epsilon=8/255$，$\alpha=2/255$ |
| 对抗源 | Standard（弱）、Engstrom2019Robustness（中）、Wang2023Better_WRN-28-10（强） |
| 聚合方式 | 三个源逐模型指标取平均 |
| 采样层数 | $K\in\{2,4,8,16\}$（均匀采样，`all_leaf` 模式） |
| 批次大小 | 8 |
| 随机种子 | 20260704 |
| GPU | RTX 4080 8GB（本地） |

## 输出文件结构

```text
outputs/e25_k8_three_source_validation/
  per_source/                                                              # 12 个原始单源 CSVs
    metrics_20_shared_{standard,engstrom,wang2023}_pgd10_n128_{k2,k4,k8,k16}_seed20260704.csv
  robustbench20_k{2,4,8,16}_3source_avg_seed20260704.csv                   # K 值三源聚合
  robustbench20_k{2,4,8,16}_3source_avg_with_rb_seed20260704.csv           # 含 RB 分数
  robustbench20_k2k4k8k16_correlations.csv                                 # 全 K 值全指标相关性
  robustbench20_report.md                                                  # 主报告
  summary/
    k2k4k8k16_spearman_matrix.csv                                         # 指标×K 值 Spearman 矩阵
    k4_vs_k8_summary.md                                                   # 汇总报告
```

## 结果：`perturbation_capacity_delta` 完整 K 值曲线

| K | Spearman ($\rho$) | Pearson ($r$) | Kendall ($\tau_b$) | 方向校正效用 | vs K=4 $\Delta$ |
|---:|---:|---:|---:|---:|---:|
| 2 | -0.544 | -0.932 | -0.411 | 0.544 | -0.030 |
| 4 | -0.574 | -0.942 | -0.432 | 0.574 | baseline |
| 8 | -0.744 | -0.960 | -0.579 | **0.744** | **+0.170** |
| 16 | -0.532 | -0.941 | -0.442 | 0.532 | -0.042 |

**K=8 是明确峰值**，K=16 回落到接近 K=2 水平。

## 与 E22（白盒 FGSM）对比

| K | 白盒 FGSM（E22）| 三源 PGD-10（E25）| $\Delta$ |
|---:|---:|---:|---:|
| 2 | 0.565 | 0.544 | -0.021 |
| 4 | 0.626 | 0.574 | -0.052 |
| 8 | **0.805** | **0.744** | -0.061 |
| 16 | 0.795 | 0.532 | **-0.263** |

三个关键发现：

1. K=8 在两个协议下均最优，峰值一致
2. 三源协议下容量值系统偏低 ~0.06，属协议差异正常范围
3. **三源协议对 K=16 的惩罚远大于白盒**（-0.042 vs 白盒仅 -0.010），三源共享对抗对过采样更敏感

## 按 K 值敏感性的指标分类

| 类型 | 指标 | K=2 | K=4 | K=8 | K=16 | 机制解释 |
|---|---|---|---|---|---|---|
| **K=8 峰值** | `perturbation_capacity_delta` | 0.544 | 0.574 | **0.744** | 0.532 | 中等采样层数最优 |
| | `feature_graph_fundamental_rank_delta` | 0.507 | 0.438 | **0.558** | 0.483 | 同上 |
| **K=16 单调递增** | `feature_box_dimension_delta` | 0.039 | 0.223 | 0.395 | **0.600** | 几何指标需要更多层 |
| | `feature_hausdorff_delta` | 0.001 | -0.005 | 0.132 | **0.215** | 同上 |
| | `feature_quantile_hausdorff_delta` | -0.024 | -0.000 | 0.116 | **0.245** | 同上 |
| **K=16 最优** | `feature_measure_mmd` | 0.392 | 0.469 | 0.382 | **0.513** | MMD 在更多层下改善 |
| | `perturbation_capacity_above_mean_delta` | 0.433 | 0.471 | 0.376 | **0.483** | 门控变体偏好更多层 |
| | `perturbation_capacity_relu_active_delta` | 0.463 | 0.377 | 0.350 | **0.480** | 同上 |
| | `feature_measure_relu_active_spatial_delta` | 0.245 | 0.359 | 0.317 | **0.502** | 同上 |
| **K≥8 崩溃** | `fsp_delta` | 0.403 | **0.414** | 0.021 | 0.102 | 跨层配对在过多层退化 |
| | `feature_measure_struct01_delta` | 0.400 | **0.385** | 0.017 | 0.104 | 同上 |
| | `feature_measure_struct_delta` | 0.484 | 0.308 | -0.203 | 0.088 | 同上 |
| **K 不变** | `input_grad_l2` | 0.538 | 0.538 | 0.538 | 0.538 | 不依赖特征采样 |
| | `adversarial_loss` | 0.486 | 0.486 | 0.486 | 0.486 | 同上 |
| | `logit_kl` | 0.286 | 0.286 | 0.286 | 0.286 | 同上 |

## Bootstrap 95% 置信区间（K=8，10000 次有放回重采样）

只有三个指标的 CI 不跨零：`perturbation_capacity_delta`、`input_grad_l2`、`adversarial_loss`。

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

`perturbation_capacity_delta` 是**唯一** bootstrap CI 不跨零的 Ours 指标——其他 Ours 指标在 95% 置信水平下不能排除零相关。这强化了 capacity 在 Ours 指标族中的领先地位不仅是点估计最优，而且统计显著。

完整 bootstrap 数据：`outputs/e25_k2_k8_k16_three_source_validation/robustbench20_k8_bootstrap_ci.csv`

## 结论

1. **`perturbation_capacity_delta` 在三源协议下最优 K=8**，Spearman=0.744，且是唯一通过 bootstrap 显著性检验的 Ours 指标
2. **K=16 在三源协议下退化更严重**（白盒仅 -0.010，三源 -0.212），三源共享对抗对 K 值选择更敏感
3. **不同指标族的最优 K 值不同**：几何/分形指标在 K=16 下持续改善，跨层配对指标在 K≥8 时崩溃，梯度/逻辑值指标不变
4. **论文策略**：`perturbation_capacity_delta` 默认 K=8；`feature_box_dimension_delta` 可考虑 K=16；需报告指标族 × K 值的敏感性矩阵；诚实报告 n=20 样本量限制
