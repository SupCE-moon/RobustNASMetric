# E25 三源共享对抗 K=2/4/8/16 Spearman 汇总

所有数值为方向校正后 Spearman（越大越好）。目标：`rb_robust_acc`。协议：3-source shared PGD-10。

## `perturbation_capacity_delta` 完整曲线

| K | Adj-Spearman | vs K=4 $\Delta$ |
|---:|---:|---:|
| 2 | 0.544 | -0.030 |
| 4 | 0.574 | baseline |
| 8 | **0.744** | **+0.170** |
| 16 | 0.532 | -0.042 |

峰值在 K=8，之后回落。与 E22 白盒结论一致。

## E22（白盒）vs E25（三源）对比

| K | 白盒 FGSM（E22）| 三源 PGD-10（E25）| $\Delta$ |
|---:|---:|---:|---:|
| 2 | 0.565 | 0.544 | -0.021 |
| 4 | 0.626 | 0.574 | -0.052 |
| 8 | 0.805 | 0.744 | -0.061 |
| 16 | 0.795 | 0.532 | -0.263 |

三源协议对 K 值更敏感，K=16 退化幅度更大。

## 不同指标的最优 K 值

| 最优 K | 代表指标 | Adj-Spearman |
|---|---|---|
| K=8 | `perturbation_capacity_delta` | 0.744 |
| K=16 | `feature_box_dimension_delta` | 0.600 |
| K=16 | `feature_measure_mmd` | 0.513 |
| K=16 | `feature_measure_relu_active_spatial_delta` | 0.502 |
| K=16 | `perturbation_capacity_above_mean_delta` | 0.483 |
| K=16 | `perturbation_capacity_relu_active_delta` | 0.480 |
| K=不变 | `input_grad_l2`, `adversarial_loss`, `logit_kl` 等 | — |

## K≥8 崩溃的指标

| Metric | K=4 | K=8 | K=16 |
|---|---:|---:|---:|
| `fsp_delta` | 0.414 | 0.021 | 0.102 |
| `feature_measure_struct01_delta` | 0.385 | 0.017 | 0.104 |
| `feature_measure_struct_delta` | 0.308 | -0.203 | 0.088 |
| `feature_measure_moment_delta` | 0.280 | -0.074 | 0.255 |

跨层配对指标（FSP、struct 系列）在 K≥8 时显著退化。

## 结论

论文使用 K=8 作为 `perturbation_capacity_delta` 默认值，几何指标可考虑 K=16。
