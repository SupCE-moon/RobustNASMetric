# 指标冗余度：84 项严格榜里到底有几个独立信号

- 输入：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv`（sha256 `b5137e2b3fe89ce5a99a99ea94bafa48d1abb81c45837551e5b1a54fe42ac92d`）
- 参与统计的设置：rb40_mature, nb201_e0, nb201_e5, nb201_e10, nb201_e25, nb201_e50, darts_e0, darts_e100
- 输入里出现的指标：85 个；实际参与统计：85 个
- 生成脚本：`scripts/analyze_metric_redundancy.py`

## 方法

先按 `direction` 把每个指标统一成「越大越好」，再在每个设置内算指标两两的 Spearman 秩相关，然后对八个设置取平均，最后用平均连接（average linkage）对 `1 - rho` 聚类。相关矩阵按指标对逐对平均，某个指标在某设置里缺值时不会污染其他指标对。

## 覆盖不足的指标（相关只基于部分设置）

冻结协议要求八个设置齐全才入榜，所以下面这些指标在正式榜单里被整项排除；但它们在部分设置上确实有值，因此仍参与了本报告的冗余度统计。**读它们的相关系数时必须记住样本少**——尤其只有 1 个设置的指标，相关系数只由 40 或 100 个模型决定，和其余指标的 560 个模型不可直接比较。

| 指标 | 有数据的设置数 |
| --- | --- |
| `racl_fixed_network_log_upper_proxy` | 1/8 |
| `racl_arch_lipschitz_log_upper` | 7/8 |

## 与其他指标的平均相关（最高的 12 个）

| 指标 | 与其他 84 项的平均 rho |
| --- | --- |
| `feature_measure_robust_hybrid_rel_delta` | +0.447 |
| `feature_measure_relu_active_hybrid_rel_delta` | +0.445 |
| `feature_measure_robust_rel_delta` | +0.440 |
| `feature_measure_multiview_geometric_delta` | +0.428 |
| `feature_measure_struct_delta` | +0.419 |
| `feature_measure_robust_hybrid_delta` | +0.412 |
| `feature_measure_flow_rel_delta` | +0.412 |
| `feature_measure_flow_rel_allpairs_delta` | +0.410 |
| `feature_measure_relu_active_hybrid_delta` | +0.409 |
| `feature_measure_relu_active_delta` | +0.409 |
| `feature_measure_multiview_rms_delta` | +0.404 |
| `feature_measure_robust_delta` | +0.400 |

平均相关越高，说明这个指标越「没有新信息」——它和榜上其他指标测的是同一件事。

## 与其他指标的平均相关（最低的 8 个）

| 指标 | 与其他 84 项的平均 rho |
| --- | --- |
| `jacobian_effective_rank` | -0.190 |
| `jacobian_spectral_decay` | -0.149 |
| `feature_spectral_decay_exponent_static` | -0.060 |
| `feature_spectral_erank_static` | +0.012 |
| `racl_arch_lipschitz_log_upper` ⚠7/8 | +0.014 |
| `feature_spectral_participation_ratio_static` | +0.035 |
| `margin_input_grad_ratio` | +0.041 |
| `racl_input_gradient_lower` | +0.043 |

负值意味着这个指标与榜单主流方向相反（低相关不等于它不好，可能恰恰是独立信息）。

## 切树高度 rho >= 0.60 → 24 个簇

| 簇 | 成员数 | 代表指标（簇内平均相关最高） | 成员 |
| --- | --- | --- | --- |
| 17 | 22 | `feature_measure_robust_hybrid_rel_delta` | `feature_measure_robust_hybrid_rel_delta`, `feature_measure_relu_active_hybrid_rel_delta`, `feature_measure_robust_rel_delta`, `feature_measure_multiview_geometric_delta`, `feature_measure_struct_delta`, `feature_measure_flow_rel_delta`, `feature_measure_flow_rel_allpairs_delta`, `feature_measure_multiview_rms_delta`, `feature_measure_flow_rel_multiscale_delta`, `feature_measure_multiview_max_delta`, `feature_measure_moment_delta`, `feature_measure_struct01_delta`, `feature_measure_flow_corr_symmetric_rel_delta`, `feature_measure_flow_corr_rel_delta`, `feature_measure_flow_corr_tophalf_delta`, `feature_delta`, `fsp_delta`, `feature_measure_flow_corr_angular_delta`, `feature_measure_flow_corr_rms_delta`, `feature_measure_fsp_rel_multiscale_delta`, `feature_measure_channel_gram_cosine_delta`, `feature_measure_flow_corr_worst_delta` |
| 18 | 13 | `feature_measure_robust_hybrid_delta` | `feature_measure_robust_hybrid_delta`, `feature_measure_relu_active_hybrid_delta`, `feature_measure_relu_active_delta`, `feature_measure_robust_delta`, `feature_measure_weighted_delta`, `feature_measure_robust_spatial_delta`, `feature_measure_local_delta`, `feature_measure_robust_spatial_rel_delta`, `feature_measure_spatial_gram_delta`, `feature_measure_spatial_gram_rel_delta`, `feature_measure_relu_active_spatial_delta`, `feature_measure_relu_active_spatial_rel_delta`, `feature_measure_spatial_gram_cosine_delta` |
| 19 | 9 | `feature_measure_cov_delta` | `feature_measure_cov_delta`, `feature_measure_channel_gram_centered_rel_delta`, `perturbation_capacity_profile_delta`, `feature_measure_spectrum_hellinger_delta`, `feature_measure_spectrum_fisher_rao_delta`, `perturbation_capacity_flow_rel_delta`, `feature_measure_spectrum_triangular_delta`, `feature_measure_spectrum_js_delta`, `feature_measure_spectrum_ks_delta` |
| 8 | 8 | `perturbation_capacity_relu_active_multiscale_delta` | `perturbation_capacity_relu_active_multiscale_delta`, `perturbation_capacity_importance_delta`, `perturbation_capacity_multiscale_delta`, `perturbation_capacity_topk_delta`, `perturbation_capacity_delta`, `perturbation_capacity_above_mean_delta`, `perturbation_capacity_relu_active_delta`, `perturbation_capacity_iqr_active_delta` |
| 1 | 6 | `local_logit_lipschitz` | `local_logit_lipschitz`, `jacobian_spectral_norm`, `logit_kl`, `input_loss_delta`, `input_grad_l2`, `racl_input_gradient_lower` |
| 7 | 3 | `feature_quantile_hausdorff_delta` | `feature_quantile_hausdorff_delta`, `feature_hausdorff_delta`, `feature_graph_fundamental_rank_delta` |
| 13 | 3 | `trnas_consistency_phi` | `trnas_consistency_phi`, `trnas_r_score`, `trnas_activation_psi` |
| 4 | 2 | `jacobian_spectral_decay` | `jacobian_spectral_decay`, `jacobian_effective_rank` |
| 5 | 2 | `feature_spectral_participation_ratio_static` | `feature_spectral_participation_ratio_static`, `feature_spectral_erank_static` |
| 14 | 2 | `adversarial_loss` | `adversarial_loss`, `natural_loss` |
| 16 | 2 | `fsp_strict_delta` | `fsp_strict_delta`, `feature_measure_flow_delta` |
| 2 | 1 | `racl_fixed_network_log_upper_proxy` | `racl_fixed_network_log_upper_proxy` |
| 3 | 1 | `feature_measure_swd` | `feature_measure_swd` |
| 6 | 1 | `feature_spectral_decay_exponent_static` | `feature_spectral_decay_exponent_static` |
| 9 | 1 | `perturbation_capacity_iqr_delta` | `perturbation_capacity_iqr_delta` |
| 10 | 1 | `perturbation_capacity_gini_delta` | `perturbation_capacity_gini_delta` |
| 11 | 1 | `feature_box_dimension_delta` | `feature_box_dimension_delta` |
| 12 | 1 | `perturbation_capacity_signed_delta` | `perturbation_capacity_signed_delta` |
| 15 | 1 | `margin_input_grad_ratio` | `margin_input_grad_ratio` |
| 20 | 1 | `feature_measure_spectrum_emd_delta` | `feature_measure_spectrum_emd_delta` |
| 21 | 1 | `perturbation_capacity_channelwise_delta` | `perturbation_capacity_channelwise_delta` |
| 22 | 1 | `feature_measure_mmd` | `feature_measure_mmd` |
| 23 | 1 | `perturbation_capacity_layerprofile_delta` | `perturbation_capacity_layerprofile_delta` |
| 24 | 1 | `racl_arch_lipschitz_log_upper` | `racl_arch_lipschitz_log_upper` |

## 切树高度 rho >= 0.50 → 21 个簇

| 簇 | 成员数 | 代表指标（簇内平均相关最高） | 成员 |
| --- | --- | --- | --- |
| 15 | 35 | `feature_measure_robust_hybrid_rel_delta` | `feature_measure_robust_hybrid_rel_delta`, `feature_measure_relu_active_hybrid_rel_delta`, `feature_measure_robust_rel_delta`, `feature_measure_multiview_geometric_delta`, `feature_measure_struct_delta`, `feature_measure_robust_hybrid_delta`, `feature_measure_flow_rel_delta`, `feature_measure_flow_rel_allpairs_delta`, `feature_measure_relu_active_hybrid_delta`, `feature_measure_relu_active_delta`, `feature_measure_multiview_rms_delta`, `feature_measure_robust_delta`, `feature_measure_weighted_delta`, `feature_measure_robust_spatial_delta`, `feature_measure_flow_rel_multiscale_delta`, `feature_measure_multiview_max_delta`, `feature_measure_moment_delta`, `feature_measure_struct01_delta`, `feature_measure_local_delta`, `feature_measure_flow_corr_symmetric_rel_delta`, `feature_measure_flow_corr_rel_delta`, `feature_measure_flow_corr_tophalf_delta`, `feature_delta`, `fsp_delta`, `feature_measure_robust_spatial_rel_delta`, `feature_measure_flow_corr_angular_delta`, `feature_measure_flow_corr_rms_delta`, `feature_measure_fsp_rel_multiscale_delta`, `feature_measure_channel_gram_cosine_delta`, `feature_measure_flow_corr_worst_delta`, `feature_measure_spatial_gram_delta`, `feature_measure_spatial_gram_rel_delta`, `feature_measure_relu_active_spatial_delta`, `feature_measure_relu_active_spatial_rel_delta`, `feature_measure_spatial_gram_cosine_delta` |
| 7 | 9 | `perturbation_capacity_relu_active_multiscale_delta` | `perturbation_capacity_relu_active_multiscale_delta`, `perturbation_capacity_importance_delta`, `perturbation_capacity_multiscale_delta`, `perturbation_capacity_topk_delta`, `perturbation_capacity_delta`, `perturbation_capacity_above_mean_delta`, `perturbation_capacity_relu_active_delta`, `perturbation_capacity_iqr_delta`, `perturbation_capacity_iqr_active_delta` |
| 16 | 9 | `feature_measure_cov_delta` | `feature_measure_cov_delta`, `feature_measure_channel_gram_centered_rel_delta`, `perturbation_capacity_profile_delta`, `feature_measure_spectrum_hellinger_delta`, `feature_measure_spectrum_fisher_rao_delta`, `perturbation_capacity_flow_rel_delta`, `feature_measure_spectrum_triangular_delta`, `feature_measure_spectrum_js_delta`, `feature_measure_spectrum_ks_delta` |
| 1 | 6 | `local_logit_lipschitz` | `local_logit_lipschitz`, `jacobian_spectral_norm`, `logit_kl`, `input_loss_delta`, `input_grad_l2`, `racl_input_gradient_lower` |
| 5 | 3 | `feature_spectral_participation_ratio_static` | `feature_spectral_participation_ratio_static`, `feature_spectral_erank_static`, `feature_spectral_decay_exponent_static` |
| 6 | 3 | `feature_quantile_hausdorff_delta` | `feature_quantile_hausdorff_delta`, `feature_hausdorff_delta`, `feature_graph_fundamental_rank_delta` |
| 11 | 3 | `trnas_consistency_phi` | `trnas_consistency_phi`, `trnas_r_score`, `trnas_activation_psi` |
| 4 | 2 | `jacobian_spectral_decay` | `jacobian_spectral_decay`, `jacobian_effective_rank` |
| 12 | 2 | `adversarial_loss` | `adversarial_loss`, `natural_loss` |
| 14 | 2 | `fsp_strict_delta` | `fsp_strict_delta`, `feature_measure_flow_delta` |
| 2 | 1 | `racl_fixed_network_log_upper_proxy` | `racl_fixed_network_log_upper_proxy` |
| 3 | 1 | `feature_measure_swd` | `feature_measure_swd` |
| 8 | 1 | `perturbation_capacity_gini_delta` | `perturbation_capacity_gini_delta` |
| 9 | 1 | `feature_box_dimension_delta` | `feature_box_dimension_delta` |
| 10 | 1 | `perturbation_capacity_signed_delta` | `perturbation_capacity_signed_delta` |
| 13 | 1 | `margin_input_grad_ratio` | `margin_input_grad_ratio` |
| 17 | 1 | `feature_measure_spectrum_emd_delta` | `feature_measure_spectrum_emd_delta` |
| 18 | 1 | `perturbation_capacity_channelwise_delta` | `perturbation_capacity_channelwise_delta` |
| 19 | 1 | `feature_measure_mmd` | `feature_measure_mmd` |
| 20 | 1 | `perturbation_capacity_layerprofile_delta` | `perturbation_capacity_layerprofile_delta` |
| 21 | 1 | `racl_arch_lipschitz_log_upper` | `racl_arch_lipschitz_log_upper` |

## 切树高度 rho >= 0.40 → 15 个簇

| 簇 | 成员数 | 代表指标（簇内平均相关最高） | 成员 |
| --- | --- | --- | --- |
| 11 | 35 | `feature_measure_robust_hybrid_rel_delta` | `feature_measure_robust_hybrid_rel_delta`, `feature_measure_relu_active_hybrid_rel_delta`, `feature_measure_robust_rel_delta`, `feature_measure_multiview_geometric_delta`, `feature_measure_struct_delta`, `feature_measure_robust_hybrid_delta`, `feature_measure_flow_rel_delta`, `feature_measure_flow_rel_allpairs_delta`, `feature_measure_relu_active_hybrid_delta`, `feature_measure_relu_active_delta`, `feature_measure_multiview_rms_delta`, `feature_measure_robust_delta`, `feature_measure_weighted_delta`, `feature_measure_robust_spatial_delta`, `feature_measure_flow_rel_multiscale_delta`, `feature_measure_multiview_max_delta`, `feature_measure_moment_delta`, `feature_measure_struct01_delta`, `feature_measure_local_delta`, `feature_measure_flow_corr_symmetric_rel_delta`, `feature_measure_flow_corr_rel_delta`, `feature_measure_flow_corr_tophalf_delta`, `feature_delta`, `fsp_delta`, `feature_measure_robust_spatial_rel_delta`, `feature_measure_flow_corr_angular_delta`, `feature_measure_flow_corr_rms_delta`, `feature_measure_fsp_rel_multiscale_delta`, `feature_measure_channel_gram_cosine_delta`, `feature_measure_flow_corr_worst_delta`, `feature_measure_spatial_gram_delta`, `feature_measure_spatial_gram_rel_delta`, `feature_measure_relu_active_spatial_delta`, `feature_measure_relu_active_spatial_rel_delta`, `feature_measure_spatial_gram_cosine_delta` |
| 4 | 12 | `perturbation_capacity_relu_active_multiscale_delta` | `perturbation_capacity_relu_active_multiscale_delta`, `perturbation_capacity_importance_delta`, `perturbation_capacity_multiscale_delta`, `perturbation_capacity_topk_delta`, `perturbation_capacity_delta`, `perturbation_capacity_above_mean_delta`, `perturbation_capacity_relu_active_delta`, `feature_quantile_hausdorff_delta`, `feature_hausdorff_delta`, `feature_graph_fundamental_rank_delta`, `perturbation_capacity_iqr_delta`, `perturbation_capacity_iqr_active_delta` |
| 12 | 11 | `feature_measure_cov_delta` | `feature_measure_cov_delta`, `feature_measure_channel_gram_centered_rel_delta`, `perturbation_capacity_profile_delta`, `feature_measure_spectrum_hellinger_delta`, `feature_measure_spectrum_fisher_rao_delta`, `perturbation_capacity_flow_rel_delta`, `feature_measure_spectrum_triangular_delta`, `feature_measure_spectrum_js_delta`, `feature_measure_spectrum_ks_delta`, `perturbation_capacity_channelwise_delta`, `feature_measure_spectrum_emd_delta` |
| 1 | 8 | `feature_measure_swd` | `feature_measure_swd`, `racl_fixed_network_log_upper_proxy`, `local_logit_lipschitz`, `jacobian_spectral_norm`, `logit_kl`, `input_loss_delta`, `input_grad_l2`, `racl_input_gradient_lower` |
| 3 | 3 | `feature_spectral_participation_ratio_static` | `feature_spectral_participation_ratio_static`, `feature_spectral_erank_static`, `feature_spectral_decay_exponent_static` |
| 8 | 3 | `trnas_consistency_phi` | `trnas_consistency_phi`, `trnas_r_score`, `trnas_activation_psi` |
| 9 | 3 | `adversarial_loss` | `adversarial_loss`, `natural_loss`, `margin_input_grad_ratio` |
| 2 | 2 | `jacobian_spectral_decay` | `jacobian_spectral_decay`, `jacobian_effective_rank` |
| 10 | 2 | `fsp_strict_delta` | `fsp_strict_delta`, `feature_measure_flow_delta` |
| 5 | 1 | `perturbation_capacity_gini_delta` | `perturbation_capacity_gini_delta` |
| 6 | 1 | `feature_box_dimension_delta` | `feature_box_dimension_delta` |
| 7 | 1 | `perturbation_capacity_signed_delta` | `perturbation_capacity_signed_delta` |
| 13 | 1 | `feature_measure_mmd` | `feature_measure_mmd` |
| 14 | 1 | `perturbation_capacity_layerprofile_delta` | `perturbation_capacity_layerprofile_delta` |
| 15 | 1 | `racl_arch_lipschitz_log_upper` | `racl_arch_lipschitz_log_upper` |

## 结论边界

- 这是**描述性**分析：它说明哪些指标彼此携带重复信息，**不能**说明哪个指标「正确」或「更好」。
- 高相关不等于因果。两个指标高度相关，可能是因为它们都由同一个更底层的性质（比如特征漂移的幅度）驱动。
- 簇的数量依赖切树高度，报告里给了 0.40 / 0.50 / 0.60 三档，不做单一断言。
- 相关是在「模型之间的排序」上算的，因此只对「用指标给模型排序」这个用途有效。
