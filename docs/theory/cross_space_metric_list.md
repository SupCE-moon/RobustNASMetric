# Cross-Space Robust Metric List

Last updated: 2026-07-21

本清单用于 E23-E24 三空间相关性实验。E24 排行榜包含 **45 个 canonical metric**：E23 的 41 项，以及四个 feature geometry/fractal/topology 指标。

方向在实验前固定。除 `margin_input_grad_ratio` 和 TRNAS 三项外，全部按 lower-is-better 解释。分析脚本不会根据观测相关系数改变方向。

完整数学定义见 `docs/theory/robust_metric_inventory.md`，机器可读注册表见 `src/robust_metric_lab/metric_registry.py`。

## Ours: 29 Metrics

| Canonical field | Direction | Role |
|---|---|---|
| `feature_delta (Ours)` | lower | GAP feature displacement |
| `feature_measure_mmd (Ours)` | lower | feature-distribution MMD |
| `feature_measure_moment_delta (Ours)` | lower | historical moment composite |
| `feature_measure_local_delta (Ours)` | lower | local feature-atom displacement |
| `feature_measure_weighted_delta (Ours)` | lower | energy-weighted local displacement |
| `feature_measure_struct_delta (Ours)` | lower | local/channel-structure composite |
| `feature_measure_struct01_delta (Ours)` | lower | 0.1-weighted structural composite |
| `feature_hausdorff_delta (Ours)` | lower | symmetric support-set Hausdorff displacement |
| `feature_quantile_hausdorff_delta (Ours)` | lower | robust 95%-quantile Hausdorff discrepancy |
| `feature_box_dimension_delta (Ours)` | lower | finite-scale box-dimension drift |
| `feature_graph_fundamental_rank_delta (Ours)` | lower | multiscale graph fundamental-group rank drift |
| `perturbation_capacity_delta (Ours)` | lower | unmasked perturbation-capacity displacement |
| `perturbation_capacity_relu_active_delta (Ours)` | lower | positive-ReLU support ablation |
| `perturbation_capacity_above_mean_delta (Ours)` | lower | above-channel-mean support ablation |
| `feature_measure_flow_delta (Ours)` | lower | cross-layer flow displacement |
| `feature_measure_flow_rel_delta (Ours)` | lower | relative cross-layer flow displacement |
| `feature_measure_spatial_gram_delta (Ours)` | lower | same-layer spatial Gram displacement |
| `feature_measure_spatial_gram_rel_delta (Ours)` | lower | relative spatial Gram displacement |
| `feature_measure_relu_active_delta (Ours)` | lower | ReLU-weighted first-order displacement |
| `feature_measure_relu_active_spatial_delta (Ours)` | lower | ReLU-weighted spatial Gram displacement |
| `feature_measure_relu_active_spatial_rel_delta (Ours)` | lower | relative ReLU spatial Gram displacement |
| `feature_measure_relu_active_hybrid_delta (Ours)` | lower | ReLU first/second-order hybrid |
| `feature_measure_relu_active_hybrid_rel_delta (Ours)` | lower | relative ReLU hybrid |
| `feature_measure_robust_delta (Ours)` | lower | flow plus first-order composite |
| `feature_measure_robust_rel_delta (Ours)` | lower | relative flow composite |
| `feature_measure_robust_spatial_delta (Ours)` | lower | spatial Gram plus first-order composite |
| `feature_measure_robust_spatial_rel_delta (Ours)` | lower | relative spatial composite |
| `feature_measure_robust_hybrid_delta (Ours)` | lower | flow/spatial/local hybrid |
| `feature_measure_robust_hybrid_rel_delta (Ours)` | lower | relative hybrid |

## Existing Baselines: 11 Metrics

| Canonical field | Direction | Attribution |
|---|---|---|
| `feature_measure_swd` | lower | sliced-Wasserstein baseline |
| `feature_measure_cov_delta` | lower | covariance-displacement baseline |
| `fsp_strict_delta` | lower | FSP/RobNet [1,2] |
| `fsp_delta` | lower | historical same-layer Gram field [1,2] |
| `logit_kl` | lower | TRADES divergence [3] |
| `natural_loss` | lower | clean-loss control [10] |
| `adversarial_loss` | lower | adversarial-loss control [10] |
| `input_loss_delta` | lower | input-loss landscape [8,9] |
| `input_grad_l2` | lower | input-gradient/Jacobian sensitivity [4,7] |
| `local_logit_lipschitz` | lower | local logit sensitivity [5,6] |
| `margin_input_grad_ratio` | higher | first-order robust radius [7] |

## RACL: 2 Metrics

### `racl_arch_lipschitz_log_upper [5]`

For a fixed searched cell, selected operation and connection coefficients equal one. The architecture-dependent discrete counterpart of RACL Eq. (10) is stored in log form:

$$
M_{\mathrm{RACL-upper}}
=
\sum_{k,j}
\log\left(
\max\left\{
\sum_{i<j}\lambda_{o_{ij}},
\varepsilon
\right\}
\right).
$$

Lower is better. On NAS-Bench-201 and DARTS, the code uses explicit cell paths. RobustBench models do not contain RACL architecture distributions or DARTS cells, so their source field is `racl_fixed_network_log_upper_proxy`, a Conv/Linear sequential spectral-product proxy. The analyzer maps both fields to one canonical name but preserves `source_field` and `racl_upper_mode` for audit. The generic residual-network value is not claimed to be a strict global Lipschitz bound.

### `racl_input_gradient_lower [5]`

$$
M_{\mathrm{RACL-grad}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left\|
\nabla_{x_b}\ell(f(x_b),y_b)
\right\|_2.
$$

Lower is better. This is numerically identical to `input_grad_l2`; both rows are retained because the user explicitly requested the two RACL terms. The duplicate must be disclosed when counting statistically independent baselines.

## TRNAS: 3 Metrics

| Canonical field | Direction | Definition |
|---|---|---|
| `trnas_activation_psi [11,12]` | higher | number of distinct sample-wise binary ReLU activation patterns |
| `trnas_consistency_phi [11]` | higher | sum of feature and parameter cosine-consistency products after one adversarial weight step |
| `trnas_r_score [11]` | higher | $\sqrt{\Psi}\,\Phi$ |

The TRNAS paper does not release the perturbation-step magnitude. E23 fixes $\eta=10^{-6}$ from the pre-existing E21 sensitivity audit and records it in every CSV. TRNAS uses one joint 16-sample batch in all three spaces; the other clean/adversarial displacement metrics use 64 samples.

## Field Count Audit

| Group | Canonical metrics | Notes |
|---|---:|---|
| Ours | 29 | includes MMD, feature/capacity variants, and four E24 geometry/topology fields |
| Existing baselines and controls | 11 | original 20-model comparison fields |
| RACL | 2 | gradient row duplicates the numerical `input_grad_l2` values |
| TRNAS | 3 | $\Psi$, $\Phi$, and R-Score |
| **Total** | **45** | one RACL upper canonical field has two space-specific source names |

## References

[1] Dong et al., `When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks`.

[2] Yim et al., `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning`.

[3] Zhang et al., `Theoretically Principled Trade-off between Robustness and Accuracy`.

[4] Jakubovitz and Giryes, `Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization`.

[5] Dong et al., `Adversarially Robust Neural Architectures`.

[6] Cisse et al., `Parseval Networks: Improving Robustness to Adversarial Examples`.

[7] DSRNA, `Differentiable Search of Robust Neural Architectures`.

[8] Mok et al., `AdvRush: Searching for Adversarially Robust Neural Architectures`.

[9] Wu et al., `Adversarial Weight Perturbation Helps Robust Generalization`.

[10] Ou, Feng, and Sun, `Towards Accurate and Robust Architectures via Neural Architecture Search`.

[11] Yang et al., `TRNAS: A Training-Free Robust Neural Architecture Search`, ICCV 2025.

[12] Peng et al., `SWAP-NAS: Sample-Wise Activation Patterns for Ultra-fast NAS`, ICLR 2024.
