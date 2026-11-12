# e0 指标对权重种子的稳定性（2026-11-12）

架构：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_n400_20261112.csv`（sha256 `b618ecf32fede27655386ba7fc81c87c9f76e45bc5429d221dfa25e779a9014c`，100 个）
权重种子：[0, 1, 2]；每个种子重算 69 个指标，FGSM 8/255 单步、64 张图。

## 结论

- **值稳定**（种子间秩相关 ≥ 0.8）的指标：**53 / 69**。
- **信号稳定**（不翻号且极差 ≤ 0.2）的指标：**55 / 69**。

值稳定和信号稳定是两件事：一个指标可以每次初始化都算出不同的数，但只要这个抖动对所有架构是共同的，它和 target 的相关就可能仍然稳定。

## 值稳定性最好的十个

| 指标 | 种子间平均秩相关 | 最差一对 |
| --- | --- | --- |
| `input_grad_l2` | +0.974 | +0.969 |
| `feature_measure_flow_rel_allpairs_delta` | +0.973 | +0.968 |
| `jacobian_spectral_norm` | +0.973 | +0.969 |
| `feature_delta` | +0.967 | +0.963 |
| `feature_measure_flow_rel_multiscale_delta` | +0.967 | +0.962 |
| `feature_measure_flow_rel_delta` | +0.967 | +0.961 |
| `feature_quantile_hausdorff_delta` | +0.964 | +0.957 |
| `feature_measure_relu_active_hybrid_rel_delta` | +0.963 | +0.958 |
| `feature_hausdorff_delta` | +0.963 | +0.957 |
| `feature_measure_spectrum_hellinger_delta` | +0.960 | +0.959 |

## 值稳定性最差的十个

| 指标 | 种子间平均秩相关 | 最差一对 |
| --- | --- | --- |
| `margin_input_grad_ratio` | +0.706 | +0.654 |
| `feature_measure_struct01_delta` | +0.705 | +0.643 |
| `fsp_delta` | +0.704 | +0.638 |
| `feature_measure_spatial_gram_cosine_delta` | +0.674 | +0.646 |
| `feature_box_dimension_delta` | +0.613 | +0.541 |
| `perturbation_capacity_signed_delta` | +0.422 | +0.248 |
| `logit_kl` | +0.166 | +0.121 |
| `natural_loss` | +0.094 | -0.036 |
| `adversarial_loss` | +0.094 | -0.036 |
| `input_loss_delta` | +0.035 | -0.178 |

## 信号跨种子

| 指标 | seed 0 | seed 1 | seed 2 | 极差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `input_grad_l2` | +0.292 | +0.297 | +0.278 | +0.018 | stable_signal |
| `feature_measure_flow_rel_allpairs_delta` | +0.172 | +0.199 | +0.186 | +0.028 | stable_signal |
| `jacobian_spectral_norm` | +0.294 | +0.315 | +0.276 | +0.038 | stable_signal |
| `feature_delta` | +0.132 | +0.160 | +0.122 | +0.038 | stable_signal |
| `feature_measure_flow_rel_multiscale_delta` | +0.270 | +0.314 | +0.303 | +0.044 | stable_signal |
| `feature_measure_flow_rel_delta` | +0.274 | +0.312 | +0.304 | +0.039 | stable_signal |
| `feature_quantile_hausdorff_delta` | +0.175 | +0.211 | +0.164 | +0.047 | stable_signal |
| `feature_measure_relu_active_hybrid_rel_delta` | +0.165 | +0.211 | +0.123 | +0.089 | stable_signal |
| `feature_hausdorff_delta` | +0.185 | +0.211 | +0.154 | +0.058 | stable_signal |
| `feature_measure_spectrum_hellinger_delta` | +0.153 | +0.194 | +0.161 | +0.042 | stable_signal |
| `feature_measure_relu_active_hybrid_delta` | +0.151 | +0.180 | +0.079 | +0.101 | stable_signal |
| `feature_graph_fundamental_rank_delta` | +0.165 | +0.183 | +0.117 | +0.066 | stable_signal |
| `feature_measure_relu_active_delta` | +0.152 | +0.182 | +0.105 | +0.077 | stable_signal |
| `feature_measure_spectrum_js_delta` | +0.065 | +0.136 | +0.061 | +0.075 | stable_signal |
| `perturbation_capacity_multiscale_delta` | +0.210 | +0.249 | +0.210 | +0.039 | stable_signal |

## 怎么读

- `值稳定` 只看同一架构在不同权重种子下的指标值是否同序；它不涉及 target。
- `信号稳定` 看每个种子单独算出的偏相关是否同号、同量级。
- 只有**值稳定**的指标才适合当作架构的固定特征写进模型；值不稳但信号稳的指标只能作为多次采样后的统计量使用（那需要明说采了几次）。

## 配方一致性

本脚本刻意**不**调用 `.eval()`，与 `scripts/evaluate_nb201_e27_zerocost.py` 的调用序列逐行一致。
验证办法：用本脚本算 seed 0，再把结果与发布宽表按 `arch_id` 对齐，逐列应完全相同（`max|diff| = 0`）。

这一点值得单列，因为早期版本在这里加了 `.eval()`，结果值稳定的指标一度只有 14 / 69。
那个数字是配方漂移造成的假象，不是指标的性质——同一份数据，只是网络多切了一次模式。

## 这份报告不能说明什么

- 只用了三个种子，`min_rho` 是三个里最差的一对，置信度有限。
- 只检验了权重种子，没有检验样本子集、攻击配置（后者见 `e0-perturbation-20261112`）。
- 这 100 个架构是 `arch_id` 字典序的前 100 个，不是随机抽样；它们的信号量级（约 +0.30）低于 400 架构全域（约 +0.40），子集效应本身没有被控制。
- 相关不是因果；信号稳也不代表这个指标测量的是鲁棒性的成因。

## 复现

```powershell
python scripts\audit_e0_weight_seed.py `
  --wide <n400 宽表>.csv `
  --data-root data\cifar10 `
  --limit 100 `
  --json results\audits\e0-weight-seed-20261112.json `
  --report results\audits\e0-weight-seed-20261112.md
```
