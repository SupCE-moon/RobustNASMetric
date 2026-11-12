# 注册指标 ↔ 实现入口 对照审计（自动生成）

由 `tools/audit_metric_implementation.py` 生成，不要手改。

- 注册指标：**86**
- 一次性 bundle（`compute_feature_metrics`）里的指标字段：**70**（`MetricResult` 共 75 个字段，其余是元数据）
- 由其他模块专门计算：**15**
- 只有声明、找不到实现痕迹：**1**
- 哪儿都找不到：**0**

## 由其他模块计算的指标

| 指标 | 出现的文件 |
| --- | --- |
| `feature_measure_flow_corr_rms_delta` | `e37_metrics.py` |
| `feature_measure_flow_corr_tophalf_delta` | `e37_metrics.py` |
| `feature_measure_flow_corr_worst_delta` | `e37_metrics.py` |
| `feature_measure_multiview_geometric_delta` | `e37_metrics.py` |
| `feature_measure_multiview_max_delta` | `e37_metrics.py` |
| `feature_measure_multiview_rms_delta` | `e37_metrics.py` |
| `feature_measure_spectrum_fisher_rao_delta` | `e37_metrics.py` |
| `feature_measure_spectrum_ks_delta` | `e37_metrics.py` |
| `feature_measure_spectrum_triangular_delta` | `e37_metrics.py` |
| `racl_arch_lipschitz_log_upper` | `student_runner.py` |
| `racl_fixed_network_log_upper_proxy` | `student_runner.py` |
| `racl_input_gradient_lower` | `student_runner.py` |
| `trnas_activation_psi` | `student_runner.py` |
| `trnas_consistency_phi` | `student_runner.py` |
| `trnas_r_score` | `student_runner.py` |

## ⚠ 只有声明、没有实现痕迹

这些名字只在 `metric_registry.py` 里出现。要么实现从没落地，要么它写出去的键名和注册名不一致（例如用 f-string 拼出来的名字）。

- `feature_adv_excess_centered_unit_l2`

## 怎么读

- 「一次性 bundle」= 跑 `compute_feature_metrics` 就能拿到，主线流水线每步都算它。
- 「其他模块」= 需要额外条件（多视图采样、逐架构 Jacobian 上界、ODE 求解、两遍重要性向量），由专门脚本调用；不代表它算不出来，只代表它不在那个 bundle 里。
- 「只有声明」才是问题：注册了一个没人写出来的名字。
