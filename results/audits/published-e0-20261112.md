# 发布表的 nb201_e0 能不能直接引用（2026-11-12）

发布表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv`（sha256 `b5137e2b3fe89ce5a99a99ea94bafa48d1abb81c45837551e5b1a54fe42ac92d`）
当前代码重算：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_verify_20261112.csv`（sha256 `d9fc7176732dd2724952690dcca89d709508f89ec468fe922811f9192f501ad2`）
共比较 70 个指标；判定阈值：相对特征指标的代码上限 1000、秩相关 0.8

## 结论

- **0 个指标可以引用**：没有超过 1000 的值，且与当前代码秩相关 ≥ 0.8。
- **14 个指标出现过超过 1000 的值**：当前代码在 `metrics.py` 里对相对类指标做了 `clamp_max(1e3)`，所以这些值**不可能**由本仓库的代码产出。它们的分母接近 0，是数值伪影，不是真实度量。
- **55 个指标没有极端值，但与当前代码秩相关 < 0.8**：同一名字、不同的量。
- 1 个指标无法比较（共同架构数 < 10，或某一侧方差为 0）。

## 怎么读

- 「超过 1e3」这条判据来自代码本身，不是我们挑的阈值：`metrics.py` 里 `rel = ((am - cm).abs() / cm.abs().clamp_min(1e-6)).clamp_max(1e3)`，输出上界就是 1e3。
- 伪影行会**主导 Spearman**：秩相关只看大小顺序，一个 1e8 的值会把它所在的架构顶到最前或最后，所以受影响的不是「数值不好看」，而是**排序结论**。
- 这不说明发布表整体是错的：它是那次流水线的真实输出。它说明的是**那一版代码与本仓库不同**，因此在引用发布表 e0 得出的任何幅度之前，必须先看这一列判成什么。

## 需要人看的列（前 15）

| 指标 | 判定 | 秩相关 | 超过 1e3 的行 | 发布最大绝对值 | 当前最大绝对值 | 中位数比 |
| --- | --- | --- | --- | --- | --- | --- |
| `feature_measure_swd` | spiked | +0.316 | 38 | 5.868e+10 | 4.523 | 5.46e+04 |
| `margin_input_grad_ratio` | spiked | -0.091 | 21 | 5.393e+12 | 3.38e+05 | 0.0059 |
| `fsp_strict_delta` | spiked | +0.603 | 14 | 3.474e+21 | 2.396e-05 | 1.09e+06 |
| `natural_loss` | spiked | -0.363 | 13 | 1.024e+06 | 2.315 | 1.01 |
| `adversarial_loss` | spiked | -0.332 | 13 | 1.362e+06 | 2.315 | 1.01 |
| `feature_measure_robust_delta` | spiked | +0.584 | 13 | 4.234e+08 | 0.03369 | 10.5 |
| `feature_measure_robust_hybrid_delta` | spiked | +0.588 | 13 | 2.117e+08 | 0.04178 | 7 |
| `feature_measure_flow_delta` | spiked | +0.637 | 13 | 4.234e+08 | 0.0001139 | 911 |
| `feature_measure_relu_active_hybrid_delta` | spiked | +0.650 | 13 | 2.117e+08 | 0.04641 | 8.26 |
| `jacobian_spectral_norm` | spiked | +0.655 | 10 | 5.033e+05 | 0.0001327 | 1.49e+05 |
| `input_loss_delta` | spiked | -0.011 | 6 | 3.411e+05 | 1.192e-06 | — |
| `local_logit_lipschitz` | spiked | +0.292 | 5 | 2.855e+05 | 1.69e-05 | 8.06e+04 |
| `input_grad_l2` | spiked | +0.623 | 4 | 2.342e+05 | 3.887e-05 | 1.71e+05 |
| `logit_kl` | spiked | +0.067 | 1 | 1.536e+04 | 6.123e-08 | 5.58e+03 |
| `feature_graph_fundamental_rank_delta` | divergent | -0.229 | 0 | 0.6319 | 0.2134 | 7.49 |

## 复现

```powershell
python tools/audit_published_e0_sanity.py `
  --published <发布表.csv> `
  --current <当前代码 e0 表.csv> `
  --json results\audits\published-e0-20261112.json `
  --report results\audits\published-e0-20261112.md
```
