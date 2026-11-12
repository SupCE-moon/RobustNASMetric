# RQ2 的正面结论是否只来自「稳得住」的指标（2026-11-12）

宽表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_n400_20261112.csv`（400 个架构，sha256 `b618ecf32fede276…`）  
稳定性来源：`results\audits\e0-weight-seed-20261112.json`  
控制变量：`clean`, `params`；值稳定阈值 0.8

## 结论

**前十全部可复现，但前五里有指标只是信号稳定（值需多次采样）**

## 分组统计

| 组 | 指标数 | 最强指标 | 最强 \|rho\| | 中位 \|rho\| | \|rho\|≥0.30 |
| --- | --- | --- | --- | --- | --- |
| 值与信号都稳定 | 49 | `feature_measure_robust_delta` | +0.400 | +0.250 | 12 |
| 只有值稳定 | 4 | `feature_measure_spatial_gram_rel_delta` | +0.132 | +0.127 | 0 |
| 只有信号稳定 | 6 | `feature_measure_local_delta` | +0.397 | +0.278 | 3 |
| 都不稳定 | 10 | `perturbation_capacity_layerprofile_delta` | +0.274 | +0.143 | 0 |
| 未测稳定性 | 4 | `clean_acc` | -0.080 | +0.080 | 0 |

## 前二十（按 |偏相关| 降序）

| 指标 | 偏相关 | 值稳定 | 信号 | 组 |
| --- | --- | --- | --- | --- |
| `feature_measure_robust_delta` | +0.400 | +0.885 | stable_signal | 值与信号都稳定 |
| `feature_measure_weighted_delta` | +0.397 | +0.832 | stable_signal | 值与信号都稳定 |
| `feature_measure_local_delta` | +0.397 | +0.791 | stable_signal | 只有信号稳定 |
| `feature_measure_struct_delta` | +0.374 | +0.763 | stable_signal | 只有信号稳定 |
| `feature_measure_robust_hybrid_delta` | +0.368 | +0.887 | stable_signal | 值与信号都稳定 |
| `input_grad_l2` | +0.367 | +0.974 | stable_signal | 值与信号都稳定 |
| `feature_measure_robust_rel_delta` | +0.362 | +0.910 | stable_signal | 值与信号都稳定 |
| `jacobian_spectral_norm` | +0.361 | +0.973 | stable_signal | 值与信号都稳定 |
| `feature_measure_robust_hybrid_rel_delta` | +0.352 | +0.906 | stable_signal | 值与信号都稳定 |
| `feature_measure_robust_spatial_delta` | +0.338 | +0.888 | stable_signal | 值与信号都稳定 |
| `feature_measure_robust_spatial_rel_delta` | +0.334 | +0.885 | stable_signal | 值与信号都稳定 |
| `feature_spectral_participation_ratio_static` | +0.328 | +0.835 | stable_signal | 值与信号都稳定 |
| `feature_spectral_erank_static` | +0.315 | +0.848 | stable_signal | 值与信号都稳定 |
| `perturbation_capacity_topk_delta` | +0.306 | +0.883 | stable_signal | 值与信号都稳定 |
| `feature_measure_struct01_delta` | +0.303 | +0.705 | stable_signal | 只有信号稳定 |
| `perturbation_capacity_flow_rel_delta` | +0.279 | +0.925 | stable_signal | 值与信号都稳定 |
| `feature_measure_mmd` | +0.279 | +0.894 | stable_signal | 值与信号都稳定 |
| `feature_measure_channel_gram_centered_rel_delta` | +0.279 | +0.926 | stable_signal | 值与信号都稳定 |
| `perturbation_capacity_channelwise_delta` | +0.279 | +0.883 | stable_signal | 值与信号都稳定 |
| `perturbation_capacity_profile_delta` | +0.278 | +0.918 | stable_signal | 值与信号都稳定 |

## 怎么读

- 偏相关是**控制训练后 clean 与参数量之后**的秩偏相关，`lower` 指标已翻号（正数=方向对）。
- 值稳定与信号稳定来自 `e0-weight-seed-20261112`（100 架构 × 3 个权重种子）。
  稳定性是指标的性质，不依赖样本量，所以拿它给 400 架构的结果分组是成立的。
- 分组不是「好坏」评级：`只有信号稳定` 的指标仍可用于预测（多次采样取均值），只是不能当固定特征。

## 这份报告不能说明什么

- 稳定性只在 100 个架构（字典序前 100）上测过，没有覆盖全部 400 个。
- 输出空间一族（`adversarial_loss` 等）在这里被判为不稳定，但榜单里它们排得很前——
  那评的是**已训练模型**，与这里的**未训练网络**不是同一个对象。两者不能互相引用。
- 相关不是因果；分组只说明这个数有多可信，不说明它测的是鲁棒性的成因。

## 复现

```powershell
python scripts\audit_rq2_stable_subset.py `
  --wide <n400 宽表>.csv `
  --seed-report results\audits\e0-weight-seed-20261112.json `
  --json results\audits\rq2-stable-subset-20261112.json `
  --report results\audits\rq2-stable-subset-20261112.md
```
