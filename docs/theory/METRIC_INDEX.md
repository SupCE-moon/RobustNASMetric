# 指标总索引

> 本文件由 `tools/build_metric_index.py` 自动生成，**不要手改**。
> 改指标请改 `src/robust_metric_lab/metric_registry.py` 或 `configs/metric_eligibility.json`，然后重跑脚本。

- 注册表指标数：86；入榜 84；无名次 2
- 覆盖列：已按原始行统计

## 入榜指标（按名次）

| 名次 | 指标 | 家族 | 方向 | 归属 | 八设置均值 | 覆盖 | 资格 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `trnas_consistency_phi` | TRNAS 移植（Φ/R/Ψ） | higher | TRNAS consistency | 0.782 | 8/8 | implementation_error |
| 2 | `trnas_r_score` | TRNAS 移植（Φ/R/Ψ） | higher | TRNAS R-Score | 0.738 | 8/8 | implementation_error |
| 3 | `adversarial_loss` | 输出空间 | lower | Loss control | 0.614 | 8/8 | cross_space |
| 4 | `trnas_activation_psi` | TRNAS 移植（Φ/R/Ψ） | higher | TRNAS activation capability | 0.598 | 8/8 | implementation_error |
| 5 | `natural_loss` | 输出空间 | lower | Loss control | 0.591 | 8/8 | cross_space |
| 6 | `feature_measure_multiview_max_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.524 | 8/8 | cross_space |
| 7 | `feature_measure_multiview_rms_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.495 | 8/8 | cross_space |
| 8 | `fsp_delta` | FSP / Gram 结构 | lower | FSP | 0.479 | 8/8 | cross_space |
| 9 | `feature_measure_flow_corr_rms_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.479 | 8/8 | cross_space |
| 10 | `feature_measure_flow_corr_symmetric_rel_delta` | 特征漂移度量家族 | lower | Ours E36 (flow-correlation symmetric drift) | 0.478 | 8/8 | cross_space |
| 11 | `feature_measure_flow_corr_angular_delta` | 特征漂移度量家族 | lower | Ours E36 (flow-correlation angular drift) | 0.478 | 8/8 | cross_space |
| 12 | `feature_measure_flow_corr_rel_delta` | 特征漂移度量家族 | lower | Ours (cross-layer correlation flow) | 0.477 | 8/8 | cross_space |
| 13 | `feature_measure_flow_corr_tophalf_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.477 | 8/8 | cross_space |
| 14 | `feature_measure_moment_delta` | 特征漂移度量家族 | lower | Ours | 0.472 | 8/8 | cross_space |
| 15 | `feature_measure_channel_gram_cosine_delta` | 特征漂移度量家族 | lower | Ours E36 (channel-gram angular drift) | 0.470 | 8/8 | cross_space |
| 16 | `feature_measure_struct01_delta` | 特征漂移度量家族 | lower | Ours | 0.469 | 8/8 | cross_space |
| 17 | `feature_measure_fsp_rel_multiscale_delta` | 特征漂移度量家族 | lower | Ours (multiscale channel gram) | 0.465 | 8/8 | cross_space |
| 18 | `perturbation_capacity_profile_delta` | 扰动容量 | lower | Ours (capacity profile direction) | 0.464 | 8/8 | cross_space |
| 19 | `feature_measure_flow_corr_worst_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.455 | 8/8 | cross_space |
| 20 | `feature_adv_excess_centered_unit_l2` | 特征空间（其他） | lower | Ours preregistered adversarial-excess feature drift | 0.431 | 0/8 | cross_space |
| 21 | `perturbation_capacity_channelwise_delta` | 扰动容量 | lower | Ours (per-channel relative capacity) | 0.419 | 8/8 | cross_space |
| 22 | `feature_delta` | 特征空间（其他） | lower | Ours | 0.409 | 8/8 | cross_space |
| 23 | `feature_measure_flow_rel_allpairs_delta` | 特征漂移度量家族 | lower | Ours (all-pairs flow) | 0.399 | 8/8 | cross_space |
| 24 | `feature_measure_struct_delta` | 特征漂移度量家族 | lower | Ours | 0.398 | 8/8 | cross_space |
| 25 | `feature_measure_flow_rel_multiscale_delta` | 特征漂移度量家族 | lower | Ours (multiscale flow) | 0.392 | 8/8 | cross_space |
| 26 | `feature_measure_flow_rel_delta` | 特征漂移度量家族 | lower | Ours | 0.380 | 8/8 | cross_space |
| 27 | `margin_input_grad_ratio` | 输入/输出敏感性 | higher | Margin/gradient radius | 0.379 | 8/8 | cross_space |
| 28 | `feature_measure_channel_gram_centered_rel_delta` | 特征漂移度量家族 | lower | Ours E36 (centered channel-gram drift) | 0.358 | 8/8 | cross_space |
| 29 | `feature_measure_multiview_geometric_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.350 | 8/8 | cross_space |
| 30 | `feature_measure_robust_rel_delta` | 特征漂移度量家族 | lower | Ours | 0.340 | 8/8 | cross_space |
| 31 | `feature_measure_cov_delta` | 特征漂移度量家族 | lower | Distribution discrepancy | 0.314 | 8/8 | cross_space |
| 32 | `perturbation_capacity_signed_delta` | 扰动容量 | lower | Ours (signed capacity, ablation) | 0.279 | 8/8 | cross_space |
| 33 | `feature_measure_robust_hybrid_rel_delta` | 特征漂移度量家族 | lower | Ours | 0.264 | 8/8 | cross_space |
| 34 | `feature_measure_flow_delta` | 特征漂移度量家族 | lower | Ours | 0.256 | 8/8 | cross_space |
| 35 | `feature_measure_spectrum_hellinger_delta` | 特征漂移度量家族 | lower | Ours E36 (spectrum Hellinger drift) | 0.246 | 8/8 | cross_space |
| 36 | `feature_measure_relu_active_hybrid_rel_delta` | 特征漂移度量家族 | lower | Ours | 0.244 | 8/8 | cross_space |
| 37 | `perturbation_capacity_flow_rel_delta` | 扰动容量 | lower | Ours (cross-layer dispersion flow) | 0.243 | 8/8 | cross_space |
| 38 | `feature_measure_spectrum_fisher_rao_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.243 | 8/8 | cross_space |
| 39 | `feature_measure_spectrum_ks_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.241 | 8/8 | cross_space |
| 40 | `feature_measure_mmd` | 特征漂移度量家族 | lower | Ours | 0.229 | 8/8 | cross_space |
| 41 | `feature_measure_spectrum_js_delta` | 特征漂移度量家族 | lower | Ours E36 (spectrum JS drift) | 0.217 | 8/8 | cross_space |
| 42 | `feature_measure_spectrum_triangular_delta` | 特征漂移度量家族 | lower | Ours E37 preregistered | 0.215 | 8/8 | cross_space |
| 43 | `feature_box_dimension_delta` | 特征空间（其他） | lower | Ours | 0.213 | 8/8 | cross_space |
| 44 | `perturbation_capacity_layerprofile_delta` | 扰动容量 | lower | Ours (cross-layer capacity profile) | 0.174 | 8/8 | cross_space |
| 45 | `feature_measure_robust_delta` | 特征漂移度量家族 | lower | Ours | 0.143 | 8/8 | cross_space |
| 46 | `fsp_strict_delta` | FSP / Gram 结构 | lower | FSP | 0.123 | 8/8 | cross_space |
| 47 | `feature_measure_robust_hybrid_delta` | 特征漂移度量家族 | lower | Ours | 0.117 | 8/8 | cross_space |
| 48 | `feature_measure_relu_active_hybrid_delta` | 特征漂移度量家族 | lower | Ours | 0.104 | 8/8 | cross_space |
| 49 | `feature_measure_swd` | 特征漂移度量家族 | lower | Distribution discrepancy | 0.094 | 8/8 | cross_space |
| 50 | `feature_measure_relu_active_delta` | 特征漂移度量家族 | lower | Ours | 0.076 | 8/8 | cross_space |
| 51 | `feature_measure_weighted_delta` | 特征漂移度量家族 | lower | Ours | 0.071 | 8/8 | cross_space |
| 52 | `feature_measure_robust_spatial_delta` | 特征漂移度量家族 | lower | Ours | 0.036 | 8/8 | cross_space |
| 53 | `perturbation_capacity_topk_delta` | 扰动容量 | lower | Ours (top-k capacity) | 0.027 | 8/8 | cross_space |
| 54 | `feature_measure_robust_spatial_rel_delta` | 特征漂移度量家族 | lower | Ours | 0.024 | 8/8 | cross_space |
| 55 | `feature_measure_local_delta` | 特征漂移度量家族 | lower | Ours | 0.014 | 8/8 | cross_space |
| 56 | `jacobian_spectral_decay` | 雅可比谱 | higher | Ours (Jacobian spectrum) | -0.005 | 8/8 | cross_space |
| 57 | `feature_measure_spatial_gram_delta` | 特征漂移度量家族 | lower | Ours | -0.011 | 8/8 | cross_space |
| 58 | `feature_measure_spatial_gram_rel_delta` | 特征漂移度量家族 | lower | Ours | -0.027 | 8/8 | cross_space |
| 59 | `perturbation_capacity_gini_delta` | 扰动容量 | lower | Ours (spatial-dispersion Gini capacity) | -0.027 | 8/8 | cross_space |
| 60 | `feature_measure_relu_active_spatial_delta` | 特征漂移度量家族 | lower | Ours | -0.028 | 8/8 | cross_space |
| 61 | `feature_measure_relu_active_spatial_rel_delta` | 特征漂移度量家族 | lower | Ours | -0.048 | 8/8 | cross_space |
| 62 | `perturbation_capacity_multiscale_delta` | 扰动容量 | lower | Ours (multiscale capacity) | -0.065 | 8/8 | cross_space |
| 63 | `perturbation_capacity_delta` | 扰动容量 | lower | Ours | -0.083 | 8/8 | cross_space |
| 64 | `perturbation_capacity_importance_delta` | 扰动容量 | lower | Ours (importance-gated capacity) | -0.092 | 8/8 | cross_space |
| 65 | `input_loss_delta` | 输入/输出敏感性 | lower | Input-loss change | -0.104 | 8/8 | cross_space |
| 66 | `perturbation_capacity_relu_active_multiscale_delta` | 扰动容量 | lower | Ours (multiscale active capacity) | -0.106 | 8/8 | cross_space |
| 67 | `perturbation_capacity_above_mean_delta` | 扰动容量 | lower | Ours | -0.125 | 8/8 | cross_space |
| 68 | `perturbation_capacity_iqr_delta` | 扰动容量 | lower | Ours (IQR capacity) | -0.148 | 8/8 | cross_space |
| 69 | `feature_measure_spectrum_emd_delta` | 特征漂移度量家族 | lower | Ours E36 (spectrum EMD drift) | -0.153 | 8/8 | cross_space |
| 70 | `local_logit_lipschitz` | 输入/输出敏感性 | lower | Local logit sensitivity | -0.163 | 8/8 | cross_space |
| 71 | `feature_spectral_participation_ratio_static` | 特征空间（其他） | lower | Ours (static spectral) | -0.175 | 8/8 | cross_space |
| 72 | `feature_measure_spatial_gram_cosine_delta` | 特征漂移度量家族 | lower | Ours E36 (spatial-gram angular drift) | -0.187 | 8/8 | cross_space |
| 73 | `perturbation_capacity_relu_active_delta` | 扰动容量 | lower | Ours | -0.192 | 8/8 | cross_space |
| 74 | `jacobian_effective_rank` | 雅可比谱 | lower | Ours (Jacobian spectrum) | -0.208 | 8/8 | cross_space |
| 75 | `perturbation_capacity_iqr_active_delta` | 扰动容量 | lower | Ours (active IQR capacity) | -0.235 | 8/8 | cross_space |
| 76 | `feature_spectral_erank_static` | 特征空间（其他） | lower | Ours (static spectral) | -0.253 | 8/8 | cross_space |
| 77 | `logit_kl` | 输入/输出敏感性 | lower | TRADES divergence | -0.261 | 8/8 | cross_space |
| 78 | `input_grad_l2` | 输入/输出敏感性 | lower | Input gradient | -0.272 | 8/8 | cross_space |
| 79 | `racl_input_gradient_lower` | RACL 理论界 | lower | RACL lower term (alias of input_grad_l2) | -0.272 | 8/8 | cross_space |
| 80 | `jacobian_spectral_norm` | 雅可比谱 | lower | Ours (Jacobian spectrum) | -0.286 | 8/8 | cross_space |
| 81 | `feature_graph_fundamental_rank_delta` | 特征空间（其他） | lower | Ours | -0.338 | 8/8 | cross_space |
| 82 | `feature_quantile_hausdorff_delta` | 特征空间（其他） | lower | Ours | -0.352 | 8/8 | cross_space |
| 83 | `feature_hausdorff_delta` | 特征空间（其他） | lower | Ours | -0.367 | 8/8 | cross_space |
| 84 | `feature_spectral_decay_exponent_static` | 特征空间（其他） | higher | Ours (static spectral) | -0.375 | 8/8 | cross_space |

## 无名次的指标

| 指标 | 家族 | 方向 | 状态 | 原因 |
| --- | --- | --- | --- | --- |
| `racl_arch_lipschitz_log_upper` | RACL 理论界 | lower | excluded | Architecture-cell upper bound is not mathematically defined for fixed RB40 networks. |
| `racl_fixed_network_log_upper_proxy` | RACL 理论界 | lower | excluded | Fixed-network proxy is not the same mathematical quantity as the architecture-cell bound. |

## 怎么读这张表

- **方向**：`higher` 表示指标越大越鲁棒，`lower` 表示越小越鲁棒。冻结协议在算秩相关时按这个方向统一符号，所以方向写错会让一个指标的排名整体翻转。
- **资格**：`cross_space` 表示三个空间定义一致、可以跨空间比较；其余取值只在单一空间有意义。
- **覆盖**：该指标在 8 个设置中有几个给出有限值。冻结协议要求 8/8 才入榜，所以覆盖不足的指标不会出现在名次列。
- **家族**是**按名字前缀猜的**，不是注册表里的正式字段。它只用于快速浏览，不要拿它当统计分组的依据。

## ⚠ 需要人看的不一致：声明为不可用，却在榜上有名

`configs/metric_eligibility.json` 把这些指标标成 `implementation_error`，但发布表给了它们名次。两件事可能都对（先声明有错、后来修好了），但没人写下哪一条是当前口径，所以需要人确认。

| 名次 | 指标 | 配置里的理由 |
| --- | --- | --- |
| 4 | `trnas_activation_psi` | Kang2021Stable float32 step underflow is unresolved under the frozen 1e-6 definition. |
| 1 | `trnas_consistency_phi` | Kang2021Stable float32 step underflow is unresolved under the frozen 1e-6 definition. |
| 2 | `trnas_r_score` | Kang2021Stable float32 step underflow is unresolved under the frozen 1e-6 definition. |