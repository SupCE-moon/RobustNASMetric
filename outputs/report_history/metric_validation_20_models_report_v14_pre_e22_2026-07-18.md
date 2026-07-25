# 20-Model Robust Metric Validation

Date: 2026-07-16

## 快速结论：目前哪个 Robust Metric 最好

当前主排序采用“三源 shared-adv 对 RobustBench 官方 robust accuracy 的 Spearman 相关系数”。这是目前最能减少单一攻击源偏差的一组 20-model 结果。

除 `margin_input_grad_ratio` 外，本报告中的 robustness displacement、loss、gradient 和 sensitivity 指标理论上都是 lower-is-better。为了让所有指标都按“越大越好”阅读，定义统一 Spearman 得分：

$$
S_{\mathrm{Spearman}}
=
\begin{cases}
-\rho_s, & \text{lower-is-better metric},\\
\rho_s, & \text{higher-is-better metric}.
\end{cases}
$$

因此，$S_{\mathrm{Spearman}}$ 越大，metric 对真实鲁棒性排序得越好；得分小于 0 表示实际排序方向与理论方向相反。

当前可以直接得出六个结论：

1. **当前总体最佳、同时也是最佳 Ours：`perturbation_capacity_delta (Ours)`。** 三源 Spearman 为 $-0.565$，统一得分为 $0.565$；三种协议平均得分也是最高的 $0.548$。
2. **当前最佳外部 baseline：`input_grad_l2` [4,7]。** 三源 Spearman 为 $-0.538$。
3. **当前最佳二阶特征结构 Ours：`feature_measure_relu_active_spatial_delta (Ours)`。** 三源 Spearman 为 $-0.359$，但只比未加 ReLU 权重的 spatial Gram 高 $0.003$，不能认为存在实质提升。
4. 如果综合 white-box、Standard shared-adv 和三源 shared-adv 三种协议取平均，未加 ReLU 的 `feature_measure_spatial_gram_delta` 得分为 $0.355$，反而略高于 ReLU-active spatial 的 $0.353$。因此两者目前应视为基本持平。
5. **`perturbation_capacity_relu_active_delta (Ours)` 只在 white-box 中提升。** White-box Spearman 从 $-0.626$ 提升到 $-0.699$，但三源结果从 $-0.565$ 降到 $-0.392$，Standard 源甚至为 $0.018$。因此它应作为 ReLU 消融保留，不能替代原始 `perturbation_capacity_delta`。
6. **`perturbation_capacity_above_mean_delta (Ours)` 是更好的 hard-gated capacity，但仍不如不门控。** 三源 Spearman 为 $-0.471$，高于 $a>0$ 版本的 $-0.392$，在 36 个指标中排第 4；但原始全位置 capacity 仍以 $-0.565$ 排第 1。

### 全部 36 个指标简明排序

下表只保留最需要看的三列。排序依据是三源 shared-adv 的 $S_{\mathrm{Spearman}}$，不是 Pearson，也不是 Spearman 绝对值。

| 排名 | Metric | 归属 | 原始 Spearman $\rho_s$ | 统一得分 $S_{\mathrm{Spearman}}$ |
| ---: | --- | --- | ---: | ---: |
| 1 | `perturbation_capacity_delta` | **Ours，当前最佳** | -0.565 | **0.565** |
| 2 | `input_grad_l2` | [4,7]，最佳 baseline | -0.538 | 0.538 |
| 3 | `adversarial_loss` | [10] | -0.486 | 0.486 |
| 4 | `perturbation_capacity_above_mean_delta` | **Ours，最佳 gated capacity** | -0.471 | 0.471 |
| 5 | `feature_measure_mmd` | 分布测度 baseline | -0.469 | 0.469 |
| 6 | `feature_measure_swd` | 分布测度 baseline | -0.427 | 0.427 |
| 7 | `fsp_delta` | [1,2] | -0.414 | 0.414 |
| 8 | `perturbation_capacity_relu_active_delta` | **Ours，ReLU capacity 消融** | -0.392 | 0.392 |
| 9 | `feature_measure_struct01_delta` | Ours | -0.385 | 0.385 |
| 10 | `input_loss_delta` | [8,9] | -0.368 | 0.368 |
| 11 | `feature_measure_relu_active_spatial_delta` | **Ours，最佳二阶特征项** | -0.359 | 0.359 |
| 12 | `feature_measure_spatial_gram_delta` | Ours | -0.356 | 0.356 |
| 13 | `feature_measure_struct_delta` | Ours | -0.308 | 0.308 |
| 14 | `local_logit_lipschitz` | [5,6] | -0.290 | 0.290 |
| 15 | `feature_measure_moment_delta` | Ours | -0.284 | 0.284 |
| 16 | `logit_kl` | [3] | -0.262 | 0.262 |
| 17 | `feature_measure_spatial_gram_rel_delta` | Ours | -0.192 | 0.192 |
| 18 | `feature_measure_relu_active_spatial_rel_delta` | Ours | -0.186 | 0.186 |
| 19 | `feature_measure_cov_delta` | 分布测度 baseline | -0.186 | 0.186 |
| 20 | `feature_measure_flow_rel_delta` | Ours | -0.0887 | 0.0887 |
| 21 | `feature_measure_relu_active_hybrid_rel_delta` | Ours | -0.0662 | 0.0662 |
| 22 | `feature_measure_robust_hybrid_rel_delta` | Ours | -0.0436 | 0.0436 |
| 23 | `feature_measure_robust_spatial_delta` | Ours | -0.0286 | 0.0286 |
| 24 | `feature_measure_robust_spatial_rel_delta` | Ours | -0.0180 | 0.0180 |
| 25 | `feature_measure_relu_active_delta` | Ours | -0.00301 | 0.00301 |
| 26 | `feature_delta` | Ours | 0.00301 | -0.00301 |
| 27 | `feature_measure_robust_rel_delta` | Ours | 0.00602 | -0.00602 |
| 28 | `feature_measure_relu_active_hybrid_delta` | Ours | 0.0271 | -0.0271 |
| 29 | `feature_measure_local_delta` | Ours | 0.0361 | -0.0361 |
| 30 | `natural_loss` | [10]，clean-loss control | 0.0421 | -0.0421 |
| 31 | `feature_measure_robust_hybrid_delta` | Ours | 0.0647 | -0.0647 |
| 32 | `feature_measure_weighted_delta` | Ours | 0.0722 | -0.0722 |
| 33 | `feature_measure_robust_delta` | Ours | 0.113 | -0.113 |
| 34 | `feature_measure_flow_delta` | Ours | 0.152 | -0.152 |
| 35 | `fsp_strict_delta` | [1,2] | 0.177 | -0.177 |
| 36 | `margin_input_grad_ratio` | [7]，理论上 higher-is-better | -0.292 | -0.292 |

### 三个 Shared-Adv Source 分别是什么

三个 source model 只负责生成三份不同的共享对抗样本 cache。随后，完全相同的 20 个 target model 分别在三份 cache 上计算 metric。

| Source 类型 | Source model | 官方 clean accuracy | 官方 robust accuracy | 作用 |
| --- | --- | ---: | ---: | --- |
| 弱源 | `Standard` | 0.9478 | 0.0000 | 自然训练模型，生成最偏向非鲁棒决策边界的共享扰动 |
| 中等鲁棒源 | `Engstrom2019Robustness` | 0.8703 | 0.4925 | 提供中等强度、迁移性较好的共享扰动 |
| 强鲁棒源 | `Wang2023Better_WRN-28-10` | 0.9244 | 0.6731 | 提供来自强鲁棒模型决策边界的共享扰动 |

三个 cache 都使用 CIFAR-10、$L_\infty$ 约束 $\epsilon=8/255$、PGD-10、128 个样本和相同 seed。

“三源聚合 Spearman”不是下面三个单源 Spearman 的算术平均。计算顺序是：

$$
\bar m_k
=
\frac{1}{3}
\left(
m_k^{\mathrm{Standard}}
+m_k^{\mathrm{Engstrom}}
+m_k^{\mathrm{Wang}}
\right),
$$

其中 $m_k$ 是第 $k$ 个 target model 的 metric 值。然后计算：

$$
\rho_s^{\mathrm{3src}}
=
\rho_s
\left(
\{\bar m_k\}_{k=1}^{20},
\{A_{\mathrm{rob},k}\}_{k=1}^{20}
\right).
$$

### 每个 Source 的 Spearman 完整对比

下表仍按三源聚合后的主榜顺序排列。除 `margin_input_grad_ratio` 外，原始 Spearman 越负越好。

| 主榜排名 | Metric | Standard 源 $\rho_s$ | Engstrom 源 $\rho_s$ | Wang2023 源 $\rho_s$ | 三源聚合 $\rho_s$ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `perturbation_capacity_delta` | -0.454 | -0.583 | -0.574 | -0.565 |
| 2 | `input_grad_l2` | -0.538 | -0.538 | -0.538 | -0.538 |
| 3 | `adversarial_loss` | -0.244 | -0.707 | -0.192 | -0.486 |
| 4 | `perturbation_capacity_above_mean_delta` | 0.0271 | -0.535 | -0.442 | -0.471 |
| 5 | `feature_measure_mmd` | -0.424 | -0.424 | -0.423 | -0.469 |
| 6 | `feature_measure_swd` | -0.456 | -0.480 | -0.374 | -0.427 |
| 7 | `fsp_delta` | -0.299 | -0.520 | -0.332 | -0.414 |
| 8 | `perturbation_capacity_relu_active_delta` | 0.0180 | -0.421 | -0.405 | -0.392 |
| 9 | `feature_measure_struct01_delta` | -0.277 | -0.489 | -0.322 | -0.385 |
| 10 | `input_loss_delta` | -0.275 | -0.489 | 0.0692 | -0.368 |
| 11 | `feature_measure_relu_active_spatial_delta` | -0.242 | -0.439 | -0.358 | -0.359 |
| 12 | `feature_measure_spatial_gram_delta` | -0.256 | -0.441 | -0.352 | -0.356 |
| 13 | `feature_measure_struct_delta` | -0.0226 | -0.469 | -0.239 | -0.308 |
| 14 | `local_logit_lipschitz` | -0.346 | -0.325 | -0.223 | -0.290 |
| 15 | `feature_measure_moment_delta` | -0.200 | -0.451 | -0.194 | -0.284 |
| 16 | `logit_kl` | -0.353 | -0.398 | -0.150 | -0.262 |
| 17 | `feature_measure_spatial_gram_rel_delta` | -0.135 | -0.208 | -0.192 | -0.192 |
| 18 | `feature_measure_relu_active_spatial_rel_delta` | -0.131 | -0.205 | -0.174 | -0.186 |
| 19 | `feature_measure_cov_delta` | 0.262 | -0.341 | 0.00451 | -0.186 |
| 20 | `feature_measure_flow_rel_delta` | 0.0451 | -0.250 | -0.0902 | -0.0887 |
| 21 | `feature_measure_relu_active_hybrid_rel_delta` | 0.0737 | -0.317 | -0.0511 | -0.0662 |
| 22 | `feature_measure_robust_hybrid_rel_delta` | 0.0902 | -0.215 | -0.0256 | -0.0436 |
| 23 | `feature_measure_robust_spatial_delta` | 0.0195 | -0.280 | -0.0496 | -0.0286 |
| 24 | `feature_measure_robust_spatial_rel_delta` | 0.0256 | -0.248 | -0.0932 | -0.0180 |
| 25 | `feature_measure_relu_active_delta` | 0.107 | -0.253 | 0.0135 | -0.00301 |
| 26 | `feature_delta` | 0.0632 | -0.122 | 0.0737 | 0.00301 |
| 27 | `feature_measure_robust_rel_delta` | 0.176 | -0.147 | 0.0286 | 0.00602 |
| 28 | `feature_measure_relu_active_hybrid_delta` | 0.0737 | -0.188 | 0.00150 | 0.0271 |
| 29 | `feature_measure_local_delta` | 0.0346 | -0.214 | 0.0361 | 0.0361 |
| 30 | `natural_loss` | 0.0421 | 0.0421 | 0.0421 | 0.0421 |
| 31 | `feature_measure_robust_hybrid_delta` | 0.0632 | -0.0526 | 0.0526 | 0.0647 |
| 32 | `feature_measure_weighted_delta` | 0.114 | -0.218 | 0.0331 | 0.0722 |
| 33 | `feature_measure_robust_delta` | 0.205 | 0.105 | 0.149 | 0.113 |
| 34 | `feature_measure_flow_delta` | 0.137 | 0.224 | 0.197 | 0.152 |
| 35 | `fsp_strict_delta` | 0.102 | 0.212 | 0.232 | 0.177 |
| 36 | `margin_input_grad_ratio` | -0.292 | -0.292 | -0.292 | -0.292 |

从逐源结果看，`perturbation_capacity_delta` 在三个 source 上方向一致，并在中、强鲁棒源下都达到较强相关，是当前最稳定的 Ours。Above-mean capacity 在 Engstrom 和 Wang2023 源上明显强于 $a>0$ 门控，但两个 hard gate 在 Standard 源的 Spearman 都为正，说明门控仍引入明显 source sensitivity。Engstrom 源通常给出最清晰的模型区分度；许多 feature metric 在 Standard 和 Wang2023 源上明显变弱。`input_grad_l2`、`natural_loss` 和 `margin_input_grad_ratio` 在三个源下完全相同，是因为它们只使用 clean input 或 clean gradient，并不依赖共享的 $x_{\mathrm{adv}}$。

### 对论文 Metric 设计的直接判断

- 如果只按当前 20-model 实验的预测排序能力选择，应该优先研究 `perturbation_capacity_delta`。
- 如果论文核心必须落在“对抗扰动造成的特征空间二阶结构变化”，当前应保留 `feature_measure_spatial_gram_delta` 与 ReLU-active spatial 两个候选；两者尚未分出胜负。
- 当前把一阶项和二阶项直接相加的 `feature_measure_robust_*` 系列普遍较弱，部分甚至方向相反，不适合作为论文最终主 metric。
- 排名只能说明相关性和排序能力，不能证明因果机制。36 个指标在同一批 20 个模型上反复比较还存在选择偏差，最终主 metric 需要在未参与设计的 held-out 模型或 NAS 架构集合上确认。

## Setup

- Dataset: CIFAR-10 test subset, reused from local existing copy.
- Model set: 20 RobustBench CIFAR-10 Linf model IDs from `configs/cifar10_linf_20.yaml`.
- Official target: RobustBench CIFAR-10 Linf reported robust accuracy from the RobustBench README model-zoo table.
- White-box run: each model generated its own FGSM adversarial examples on 64 samples.
- Shared-adv run: one shared cache generated by `Standard` with PGD-10 on 128 samples, then reused for all 20 models.
- Spatial Gram implementation: for each selected layer, compute the Gram matrix between normalized spatial feature atoms. Early high-resolution feature maps are adaptively pooled to at most 16x16 spatial positions before the position-position Gram matrix is computed.

## New Files

- `scripts/generate_adv_cache.py`: generate reusable `x, y, x_adv` tensor caches.
- `scripts/fetch_robustbench_scores.py`: fetch RobustBench CIFAR-10 Linf clean/robust accuracy table.
- `scripts/join_robustbench_scores.py`: merge metric CSVs with RobustBench official scores.
- `outputs/adv_cache/cifar10_linf_standard_pgd10_n128_seed20260704.pt`: shared adversarial-example cache.
- `outputs/metrics_20_whitebox_fgsm_n64_spatial_seed20260704.csv`: white-box run with spatial Gram metrics.
- `outputs/metrics_20_shared_standard_pgd10_n128_spatial_seed20260704.csv`: shared-adv run with spatial Gram metrics.
- `outputs/shared_adv_multisource_validation_report.md`: multi-source shared-adv validation using weak, mid-robust, and strong source models.
- `outputs/metrics_20_whitebox_fgsm_n64_relu_with_rb_scores.csv`: recomputed 20-model white-box results including all five ReLU-active metrics and official scores.
- `outputs/metrics_20_shared_multisource_pgd10_n128_rel_relu_avg_with_rb_scores.csv`: recomputed three-source average including all five ReLU-active metrics and official scores.
- `outputs/correlations/e05_multisource_pgd10_n128_rel_relu_avg_vs_rb.csv`: complete correlation output for all 34 metric fields.
- `outputs/relu_active_20model_full.log`: server execution log for the ReLU-active recomputation.
- `outputs/metrics_20_shared_multisource_pgd10_n128_capacity_relu_avg_with_rb_scores.csv`: E14 three-source results including ReLU-active perturbation capacity.
- `outputs/correlations/e14_multisource_capacity_relu_avg_vs_rb.csv`: E14 complete 35-metric correlation output.
- `outputs/perturbation_capacity_relu_active_20model_report.md`: E14 definition, results, and interpretation.
- `outputs/metrics_20_shared_multisource_pgd10_n128_capacity_above_mean_avg_with_rb_scores.csv`: E15 three-source results including above-mean capacity.
- `outputs/correlations/e15_multisource_capacity_above_mean_avg_vs_rb.csv`: E15 complete 36-metric correlation output.
- `outputs/perturbation_capacity_above_mean_20model_report.md`: E15 definition, bootstrap comparison, and interpretation.

## Spearman Ranking: Target = RobustBench Official Robust Accuracy

The main scientific objective is architecture ranking, so this report uses Spearman correlation as the primary criterion. Metric direction is fixed by its mathematical definition, not inferred from the observed correlation. For a metric whose expected direction is lower-is-better, a good result is a negative Spearman correlation with robust accuracy. For a higher-is-better metric, a good result is positive.

For a single readable ranking, define the direction-adjusted ranking score:

$$
S_{\mathrm{rank}}(m)
=
\begin{cases}
-\rho_s(m, A_{\mathrm{rob}}), & m\text{ is lower-is-better},\\
\rho_s(m, A_{\mathrm{rob}}), & m\text{ is higher-is-better}.
\end{cases}
$$

Larger $S_{\mathrm{rank}}$ is better. A negative score means that the observed ordering contradicts the metric's theoretical direction. The main table is sorted by the latest three-source shared-adversarial result because it is less dependent on one attack-source model. Raw Spearman values are retained so the sign remains auditable.

### Complete 20-Model Leaderboard

| Rank | Metric | Attribution | Expected direction | White-box $\rho_s$ | Shared Standard $\rho_s$ | Three-source $\rho_s$ | $S_{\mathrm{rank}}$ |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `perturbation_capacity_delta` | Ours | lower | -0.626 | -0.454 | -0.565 | 0.565 |
| 2 | `input_grad_l2` | [4,7] | lower | -0.429 | -0.538 | -0.538 | 0.538 |
| 3 | `adversarial_loss` | [10] | lower | -0.805 | -0.244 | -0.486 | 0.486 |
| 4 | `perturbation_capacity_above_mean_delta` | Ours | lower | -0.555 | 0.0271 | -0.471 | 0.471 |
| 5 | `feature_measure_mmd` | distribution-measure baseline | lower | -0.314 | -0.424 | -0.469 | 0.469 |
| 6 | `feature_measure_swd` | distribution-measure baseline | lower | -0.492 | -0.456 | -0.427 | 0.427 |
| 7 | `fsp_delta` | [1,2] | lower | -0.493 | -0.299 | -0.414 | 0.414 |
| 8 | `perturbation_capacity_relu_active_delta` | Ours | lower | -0.699 | 0.0180 | -0.392 | 0.392 |
| 9 | `feature_measure_struct01_delta` | Ours | lower | -0.499 | -0.277 | -0.385 | 0.385 |
| 10 | `input_loss_delta` | [8,9] | lower | -0.483 | -0.275 | -0.368 | 0.368 |
| 11 | `feature_measure_relu_active_spatial_delta` | Ours | lower | -0.457 | -0.242 | -0.359 | 0.359 |
| 12 | `feature_measure_spatial_gram_delta` | Ours | lower | -0.454 | -0.256 | -0.356 | 0.356 |
| 13 | `feature_measure_struct_delta` | Ours | lower | -0.456 | -0.0226 | -0.308 | 0.308 |
| 14 | `local_logit_lipschitz` | [5,6] | lower | -0.329 | -0.346 | -0.290 | 0.290 |
| 15 | `feature_measure_moment_delta` | Ours | lower | -0.380 | -0.200 | -0.284 | 0.284 |
| 16 | `logit_kl` | [3] | lower | -0.382 | -0.353 | -0.262 | 0.262 |
| 17 | `feature_measure_spatial_gram_rel_delta` | Ours | lower | -0.235 | -0.135 | -0.192 | 0.192 |
| 18 | `feature_measure_relu_active_spatial_rel_delta` | Ours | lower | -0.209 | -0.131 | -0.186 | 0.186 |
| 19 | `feature_measure_cov_delta` | distribution-measure baseline | lower | -0.537 | 0.262 | -0.186 | 0.186 |
| 20 | `feature_measure_flow_rel_delta` | Ours | lower | -0.205 | 0.0451 | -0.0887 | 0.0887 |
| 21 | `feature_measure_relu_active_hybrid_rel_delta` | Ours | lower | -0.290 | 0.0737 | -0.0662 | 0.0662 |
| 22 | `feature_measure_robust_hybrid_rel_delta` | Ours | lower | -0.283 | 0.0902 | -0.0436 | 0.0436 |
| 23 | `feature_measure_robust_spatial_delta` | Ours | lower | -0.340 | 0.0195 | -0.0286 | 0.0286 |
| 24 | `feature_measure_robust_spatial_rel_delta` | Ours | lower | -0.329 | 0.0256 | -0.0180 | 0.0180 |
| 25 | `feature_measure_relu_active_delta` | Ours | lower | -0.355 | 0.107 | -0.00301 | 0.00301 |
| 26 | `feature_delta` | Ours | lower | -0.135 | 0.0632 | 0.00301 | -0.00301 |
| 27 | `feature_measure_robust_rel_delta` | Ours | lower | -0.221 | 0.176 | 0.00602 | -0.00602 |
| 28 | `feature_measure_relu_active_hybrid_delta` | Ours | lower | -0.286 | 0.0737 | 0.0271 | -0.0271 |
| 29 | `feature_measure_local_delta` | Ours | lower | -0.287 | 0.0346 | 0.0361 | -0.0361 |
| 30 | `natural_loss` | [10], clean-loss control | lower | 0.0226 | 0.0421 | 0.0421 | -0.0421 |
| 31 | `feature_measure_robust_hybrid_delta` | Ours | lower | -0.242 | 0.0632 | 0.0647 | -0.0647 |
| 32 | `feature_measure_weighted_delta` | Ours | lower | -0.253 | 0.114 | 0.0722 | -0.0722 |
| 33 | `feature_measure_robust_delta` | Ours | lower | 0.0150 | 0.205 | 0.113 | -0.113 |
| 34 | `feature_measure_flow_delta` | Ours | lower | 0.174 | 0.137 | 0.152 | -0.152 |
| 35 | `fsp_strict_delta` | [1,2] | lower | 0.170 | 0.102 | 0.177 | -0.177 |
| 36 | `margin_input_grad_ratio` | [7] | higher | -0.235 | -0.292 | -0.292 | -0.292 |

### Top Five Under Each Protocol

| Rank | White-box FGSM, 64 samples | Shared Standard PGD-10, 128 samples | Three-source shared PGD-10, 128 samples |
| ---: | --- | --- | --- |
| 1 | `adversarial_loss` (-0.805) | `input_grad_l2` (-0.538) | `perturbation_capacity_delta` (Ours, -0.565) |
| 2 | `perturbation_capacity_relu_active_delta` (Ours, -0.699) | `feature_measure_swd` (-0.456) | `input_grad_l2` (-0.538) |
| 3 | `perturbation_capacity_delta` (Ours, -0.626) | `perturbation_capacity_delta` (Ours, -0.454) | `adversarial_loss` (-0.486) |
| 4 | `perturbation_capacity_above_mean_delta` (Ours, -0.555) | `feature_measure_mmd` (-0.424) | `perturbation_capacity_above_mean_delta` (Ours, -0.471) |
| 5 | `feature_measure_cov_delta` (-0.537) | `logit_kl` [3] (-0.353) | `feature_measure_mmd` (-0.469) |

### Coverage of All Implemented Metrics

All 36 currently implemented metric fields now have directly comparable white-box, Standard-source shared-adv, and three-source shared-adv results on the same 20 RobustBench models. The six ReLU-active fields and the above-mean gated-capacity field contain 20 finite values in every raw result file; none were missing or NaN.

The strongest ReLU-aware metric is `feature_measure_relu_active_spatial_delta`. It improves over the unmasked spatial Gram by only 0.003 in direction-adjusted Spearman under both white-box and three-source evaluation, while it is 0.014 worse under the Standard-source shared attack. This is best interpreted as a near tie, not a demonstrated superiority claim.

The first-order ReLU-active term and both ReLU-active hybrids remain unstable under shared attacks. Therefore, the present 20-model evidence supports keeping the ReLU mask as a theoretically motivated ablation, but it does not yet support making ReLU activation weighting the paper's sole core innovation.

The ReLU-active perturbation-capacity ablation is stronger than its unmasked parent only under white-box attack. Under the primary three-source shared protocol it ranks eighth with $\rho_s=-0.392$, while the original remains first with $\rho_s=-0.565$. The Standard-source Spearman of $0.018$ shows that hard clean gating is not source-stable.

The above-mean gate improves the three-source result to $\rho_s=-0.471$ and rank 4/36, substantially recovering the loss caused by the absolute zero threshold. It still trails the original capacity and has a positive Standard-source Spearman of $0.027$. Paired model bootstrap intervals for its difference from the original cross zero under every protocol.

Exploratory coefficient-calibrated combinations are not treated as independent leaderboard metrics because their coefficients were selected using these same 20 models. Their results remain recorded in `outputs/residual_value_multisource/residual_value_calibration_report.md` and should be evaluated on a held-out model set before being ranked against fixed metrics.

## Interpretation

The clearest current result is that `perturbation_capacity_delta` (Ours) is the best-ranked metric under the three-source shared-adversarial protocol, with $\rho_s=-0.565$. It is third in the updated white-box experiment and third in the single-source shared experiment, making it the most consistently strong Ours metric in this 20-model batch. Its ReLU-active ablation rises to second under white-box but falls to eighth under three-source shared evaluation; the above-mean gate ranks fourth under both white-box and three-source evaluation.

The strongest baseline changes with the protocol. `adversarial_loss` is strongest in white-box evaluation, while `input_grad_l2` is strongest under the Standard-source shared attack. This protocol sensitivity is why one Spearman value should not be presented as a universal ordering.

Among the explicitly second-order feature metrics, `feature_measure_relu_active_spatial_delta` and `feature_measure_spatial_gram_delta` are the strongest current Ours candidates, ranking eleventh and twelfth with $\rho_s=-0.359$ and $-0.356$. Their difference is negligible at the present sample size. Both consistently outperform the additive `feature_measure_robust_spatial_delta`, indicating that the uncalibrated first-order addition weakens rather than strengthens ranking in the current model set.

`feature_measure_flow_delta`, `fsp_strict_delta`, and several additive ROBUST variants have signs opposite to their theoretical lower-is-better direction. They should not be relabeled as higher-is-better after observing the data; they currently count as failed or unstable variants.

These rankings are descriptive results from only 20 models. The next statistical version of this table should add bootstrap confidence intervals for Spearman and test rank stability across model subsets before making superiority claims.

## RobustBench / Adversarial Example Database Note

RobustBench is best treated here as a model zoo and standardized evaluation benchmark. It provides public models and evaluation tooling, but not a ready-made downloadable database of pre-generated CIFAR-10 adversarial examples for every attack/model pair. For shared-adv experiments, we should therefore create reproducible caches with explicit metadata: source model, attack, epsilon, steps, sample indices, and seed.

## Multi-Source Shared-Adv Follow-Up

The recommended shared-adv design was executed with three source caches:

1. weak source: `Standard`;
2. mid-robust source: `Engstrom2019Robustness`;
3. strong source: `Wang2023Better_WRN-28-10`.

The ReLU-active recomputation shows that activation weighting changes the spatial metric only slightly:

| Metric | Attribution | Pearson | Spearman |
| --- | --- | ---: | ---: |
| `perturbation_capacity_delta` | Ours | -0.942 | -0.565 |
| `perturbation_capacity_above_mean_delta` | Ours, above-mean capacity ablation | -0.916 | -0.471 |
| `perturbation_capacity_relu_active_delta` | Ours, ReLU capacity ablation | -0.922 | -0.392 |
| `feature_measure_relu_active_spatial_delta` | Ours | -0.699 | -0.359 |
| `feature_measure_spatial_gram_delta` | Ours | -0.717 | -0.356 |
| `feature_measure_relu_active_spatial_rel_delta` | Ours | -0.729 | -0.186 |
| `feature_measure_relu_active_hybrid_rel_delta` | Ours | -0.887 | -0.0662 |
| `feature_measure_relu_active_delta` | Ours | -0.904 | -0.00301 |
| `feature_measure_relu_active_hybrid_delta` | Ours | -0.883 | 0.0271 |
| `fsp_delta` | [1,2] | -0.852 | -0.414 |
| `input_grad_l2` | [4,7] | -0.928 | -0.538 |

The implication is that source averaging reduces single-source attack bias, while the standalone spatial second-order signal remains substantially stronger than the additive first-order-plus-second-order Ours metrics. Replacing the zero gate with a channel-mean threshold recovers part of the lost capacity signal, but both hard gates remain weaker and less source-stable than the unmasked capacity. Coefficient/layer normalization and held-out validation remain necessary before selecting the paper's final metric.

## References

[1] Dong et al., `When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks`.

[2] Yim et al., `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning`.

[3] Zhang et al., `Theoretically Principled Trade-off between Robustness and Accuracy`.

[4] Jakubovitz and Giryes, `Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization`.

[5] Dong et al., `Adversarially Robust Neural Architectures`.

[6] Cisse et al., `Parseval Networks: Improving Robustness to Adversarial Examples`.

[7] DSRNA, `Differentiable Search of Robust Neural Architectures`.

[8] AdvRush, `Searching for Adversarially Robust Neural Architectures`.

[9] Wu et al., `Adversarial Weight Perturbation Helps Robust Generalization`.

[10] Ou, Feng, and Sun, `Towards Accurate and Robust Architectures via Neural Architecture Search`.
