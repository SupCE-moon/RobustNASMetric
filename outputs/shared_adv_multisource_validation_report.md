# Multi-Source Shared-Adversarial Validation

Date: 2026-07-07

## Purpose

This experiment tests whether the proposed feature-space robustness metrics are stable under shared adversarial examples generated from multiple source models. The goal is to reduce dependence on a single white-box attack source and to test whether the metric captures target-model representation stability rather than source-specific attack artifacts.

## Sources

| source type | source model | attack | samples |
| --- | --- | --- | ---: |
| weak | `Standard` | PGD-10, Linf 8/255 | 128 |
| mid-robust | `Engstrom2019Robustness` | PGD-10, Linf 8/255 | 128 |
| strong | `Wang2023Better_WRN-28-10` | PGD-10, Linf 8/255 | 128 |

For each source cache, all 20 target RobustBench CIFAR-10 Linf models were evaluated on the same shared adversarial examples. Then per-model metric values were averaged across the three source caches.

## Added Normalized Metrics

The previous spatial Gram metric already divided the Frobenius displacement by the number of spatial positions. This run adds relative normalization against the clean structure:

$$
D_l^{\mathrm{spatial\_rel}}
=
\frac{\|K_l(x_{\mathrm{adv}})-K_l(x)\|_F}{\|K_l(x)\|_F+\epsilon}.
$$

For cross-layer feature-flow:

$$
D_{i,j}^{\mathrm{flow\_rel}}
=
\frac{\|G_{i,j}(x_{\mathrm{adv}})-G_{i,j}(x)\|_F}{\|G_{i,j}(x)\|_F+\epsilon}.
$$

## Multi-Source Average Results

Target: RobustBench official CIFAR-10 Linf robust accuracy.

| metric | Pearson | Spearman | Kendall tau-a | direction |
| --- | ---: | ---: | ---: | --- |
| `feature_measure_flow_delta` (Ours ablation) | 0.2474 | 0.1519 | 0.0947 | higher is better |
| `feature_measure_flow_rel_delta` (Ours ablation) | -0.8797 | -0.0887 | -0.0632 | lower is better |
| `feature_measure_spatial_gram_delta` (Ours ablation) | -0.7170 | -0.3564 | -0.2421 | lower is better |
| `feature_measure_spatial_gram_rel_delta` (Ours ablation) | -0.7528 | -0.1925 | -0.1579 | lower is better |
| `feature_measure_robust_delta` (Ours) | -0.8538 | 0.1128 | 0.1053 | lower is better |
| `feature_measure_robust_rel_delta` (Ours) | -0.8898 | 0.0060 | 0.0000 | lower is better |
| `feature_measure_robust_spatial_delta` (Ours) | -0.8932 | -0.0286 | -0.0105 | lower is better |
| `feature_measure_robust_spatial_rel_delta` (Ours) | -0.8731 | -0.0180 | -0.0211 | lower is better |
| `feature_measure_robust_hybrid_delta` (Ours) | -0.8779 | 0.0647 | 0.0632 | lower is better |
| `feature_measure_robust_hybrid_rel_delta` (Ours) | -0.8853 | -0.0436 | -0.0421 | lower is better |
| `fsp_delta` [1,2] | -0.8523 | -0.4135 | -0.2947 | lower is better |
| `logit_kl` [3] | -0.9266 | -0.2617 | -0.1579 | lower is better |
| `input_grad_l2` [4,7] | -0.9285 | -0.5383 | -0.4316 | lower is better |
| `local_logit_lipschitz` [5,6] | -0.9310 | -0.2902 | -0.2105 | lower is better |

## Per-Source Spatial Results

| source | spatial Gram Spearman | spatial Gram relative Spearman | robust spatial Spearman | local logit Lipschitz Spearman |
| --- | ---: | ---: | ---: | ---: |
| `Standard` | -0.2556 | -0.1353 | 0.0195 | -0.3459 |
| `Engstrom2019Robustness` | -0.4406 | -0.2075 | -0.2797 | -0.3248 |
| `Wang2023Better_WRN-28-10` | -0.3519 | -0.1925 | -0.0496 | -0.2226 |
| three-source average | -0.3564 | -0.1925 | -0.0286 | -0.2902 |

## Interpretation

The multi-source shared-adversarial design did not automatically improve rank correlation. The plain spatial Gram term remains the most useful proposed shared-adv metric in this batch, but averaging weak, mid, and strong sources reduced its Spearman from the best single-source result (-0.4406 from `Engstrom2019Robustness`) to -0.3564.

Relative normalization improved Pearson for some metrics, especially cross-layer flow, but it did not improve Spearman. This suggests that clean-structure relative scaling helps coarse magnitude prediction, but does not solve ranking instability across architectures.

The combined robust spatial metrics are still not strong rankers in the current additive form. The likely reason is scale competition: the first-order weighted atom displacement and second-order spatial Gram displacement have different architecture-dependent ranges, and a fixed coefficient of 1.0 is too crude.

## Current Research Implication

The shared-adv experiment supports keeping `feature_measure_spatial_gram_delta` as the most promising Ours ablation. It does not yet support using `feature_measure_robust_spatial_delta` as the main ranking metric without further normalization or learned/calibrated weighting.

For RobustNAS motivation, the strongest current claim should be:

> Spatial feature-relation displacement provides an architecture-interpretable representation-space signal that is complementary to output-space and gradient-space proxies, but the current additive robust metric still needs calibration before it can outperform strong post-hoc baselines.

## Next Recommended Step

Run a calibration experiment rather than adding more source models:

1. Standardize each layer metric by model-local or layer-local statistics before summation.
2. Search a small scalar weight for first-order vs second-order terms using only a validation subset of models.
3. Test whether calibrated spatial Gram plus `local_logit_lipschitz` improves residual prediction over `local_logit_lipschitz` alone.
4. Repeat on early-training checkpoints or same-training-different-architecture models, where representation-level metrics should have a stronger RobustNAS advantage.

## Residual-Value Follow-Up

The residual-value and coefficient calibration experiment has been executed. See:

- `outputs/residual_value_multisource/residual_value_calibration_report.md`
- `outputs/residual_value_multisource/residual_value_summary.csv`
- `outputs/residual_value_multisource/coefficient_calibration_grid.csv`

Main finding:

| test | result |
| --- | --- |
| Spatial Gram after controlling for `local_logit_lipschitz` | partial Spearman = -0.3706, but delta LOOCV R2 = -0.2687 |
| Spatial Gram after controlling for `input_grad_l2` | partial Spearman = -0.4200, but delta LOOCV R2 = -3.3383 |
| Best positive-weight `spatial + lambda * weighted` | lambda = 0.0, so the first-order weighted term does not help this batch |
| Best positive-weight `local_logit_lipschitz + gamma * spatial` | gamma = 0.1, Spearman improves from -0.2902 to -0.3940 |
| Best positive-weight `input_grad_l2 + gamma * spatial` | gamma = 0.0, no improvement over `input_grad_l2` |

Conclusion: the current data do not support a strong incremental-prediction claim over `input_grad_l2` or `local_logit_lipschitz`. The spatial Gram signal remains useful as an architecture-interpretable representation diagnostic, but its predictive value should be re-tested on same-training-different-architecture or early-training RobustNAS settings.
