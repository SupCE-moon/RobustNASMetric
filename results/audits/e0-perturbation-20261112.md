# 早期信号换一个扰动还在不在（2026-11-12）

基准表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_n400_20261112.csv`（sha256 `b618ecf32fede27655386ba7fc81c87c9f76e45bc5429d221dfa25e779a9014c`，400 个架构）
两个变体：`weak` = FGSM 4/255（半径减半）、`pgd` = PGD-3 8/255（同半径、更强攻击）。
所有数字都是**控制 clean 准确率与参数量之后**的偏秩相关。

## 结论

**weak**：最强指标从 `feature_measure_robust_delta`（0.400）变成 `feature_measure_flow_delta`（0.552）；前五名重合 **0/5**；所有指标的偏相关排序相关 0.044。
**pgd**：最强指标从 `feature_measure_robust_delta`（0.400）变成 `feature_measure_flow_delta`（0.551）；前五名重合 **0/5**；所有指标的偏相关排序相关 0.031。

## 怎么读

- `rho_between_variants` 衡量的是「哪些指标在这种扰动下强」这件事有没有变。接近 1 说明排序不变，接近 0 说明换一个扰动就是另一批指标赢。
- **单指标本身有抽样误差**：n=150 时约 ±0.16，n=400 时约 ±0.10（1.96/√(n−3)）。所以两个变体之间差 0.2 以内，不要当成差别。
- 真正该看的是**前五名重合度**：重合高说明「换扰动后还是同一批指标赢」，重合低说明这批指标的强弱是配置的产物。
- 单个指标的偏相关本身有抽样误差（这里只有几百个架构），所以**不要**拿两个变体的差值当效应量，要看的是整体排序是否稳定。
- 这只检验一种稳健性：扰动强度。没有检验别的随机初始化、别的样本子集、别的训练协议。

## 移动最大的十个指标

| 变体 | 指标 | 基准 | 变体 | 差 |
| --- | --- | --- | --- | --- |
| weak | `jacobian_spectral_decay` | +0.196 | -0.557 | -0.753 |
| weak | `jacobian_effective_rank` | +0.222 | -0.508 | -0.729 |
| weak | `feature_measure_local_delta` | +0.397 | -0.296 | -0.693 |
| weak | `feature_measure_weighted_delta` | +0.397 | -0.280 | -0.677 |
| weak | `feature_measure_robust_delta` | +0.400 | -0.271 | -0.671 |
| weak | `feature_measure_robust_hybrid_delta` | +0.368 | -0.271 | -0.639 |
| weak | `feature_spectral_erank_static` | +0.315 | -0.288 | -0.603 |
| weak | `feature_measure_robust_spatial_delta` | +0.338 | -0.262 | -0.601 |
| weak | `feature_measure_robust_spatial_rel_delta` | +0.334 | -0.262 | -0.595 |
| weak | `feature_measure_struct_delta` | +0.374 | -0.197 | -0.572 |
| pgd | `jacobian_spectral_decay` | +0.196 | -0.557 | -0.753 |
| pgd | `jacobian_effective_rank` | +0.222 | -0.508 | -0.729 |
| pgd | `feature_measure_local_delta` | +0.397 | -0.292 | -0.689 |
| pgd | `feature_measure_weighted_delta` | +0.397 | -0.279 | -0.677 |
| pgd | `feature_measure_robust_delta` | +0.400 | -0.275 | -0.675 |
| pgd | `feature_measure_robust_hybrid_delta` | +0.368 | -0.269 | -0.637 |
| pgd | `feature_spectral_erank_static` | +0.315 | -0.288 | -0.603 |
| pgd | `feature_measure_robust_spatial_delta` | +0.338 | -0.261 | -0.600 |
| pgd | `feature_measure_robust_spatial_rel_delta` | +0.334 | -0.260 | -0.594 |
| pgd | `feature_measure_struct_delta` | +0.374 | -0.205 | -0.579 |

## 复现

```powershell
python scripts/audit_e0_perturbation_sensitivity.py `
  --wide <n400 宽表>.csv `
  --data-root <cifar10 目录> `
  --limit 150 `
  --json results\audits\e0-perturbation-20261112.json `
  --report results\audits\e0-perturbation-20261112.md
```
