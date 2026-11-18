# 自研指标的分方向对比（2026-11-18）

总榜 84 项里自研 69 项、已有方法 15 项。总榜是八设置平均，会把“只在某一类条件下强”的指标埋掉；下面按 8 个方向分别排座次（每列只在小范围内比较）。

## 1 每个方向的冠军

| 方向 | 自研冠军 | 分数 | 自研内名次 | 全榜名次 | 全榜数量 |
|---|---|---|---|---|---|
| 已训练防御模型（RB40，40 个 RobustBench 模型） | `feature_adv_excess_centered_unit_l2` | +0.766 | 1 | 1 | 84 |
| 早期架构·单次前向（NB201 e0） | `feature_measure_relu_active_hybrid_delta` | +0.756 | 1 | 3 | 84 |
| 早期架构·五阶段平均（NB201） | `feature_measure_flow_corr_symmetric_rel_delta` | +0.709 | 1 | 5 | 84 |
| DARTS 搜索空间（e100） | `jacobian_spectral_decay` | +0.625 | 1 | 7 | 84 |
| 八设置总榜 | `feature_measure_multiview_max_delta` | +0.524 | 1 | 6 | 84 |
| 400 架构真实预测力（控 clean + 参数量） | `feature_measure_robust_delta` | +0.400 | 1 | 1 | 69 |
| 跨初始化稳定性（值稳定 rho） | `feature_measure_flow_rel_allpairs_delta` | +0.973 | 1 | 2 | 69 |
| 决策价值（挑前 10% 的提升） | `feature_measure_robust_hybrid_delta` | +0.028 | 1 | 3 | 5 |

## 已训练防御模型（RB40，40 个 RobustBench 模型）

> 对已经训练好的防御模型排序。共 84 个指标有分数，其中自研 69 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_adv_excess_centered_unit_l2` | +0.766 | 1 |
| 2 | `perturbation_capacity_topk_delta` | +0.727 | 3 |
| 3 | `perturbation_capacity_channelwise_delta` | +0.665 | 4 |
| 4 | `perturbation_capacity_iqr_delta` | +0.662 | 5 |
| 5 | `perturbation_capacity_delta` | +0.616 | 6 |
| 6 | `perturbation_capacity_flow_rel_delta` | +0.604 | 8 |
| 7 | `perturbation_capacity_multiscale_delta` | +0.587 | 9 |
| 8 | `perturbation_capacity_profile_delta` | +0.584 | 10 |
| 9 | `feature_measure_channel_gram_centered_rel_delta` | +0.579 | 11 |
| 10 | `feature_measure_spectrum_fisher_rao_delta` | +0.561 | 13 |

## 早期架构·单次前向（NB201 e0）

> 只跑一次随机初始化的前向。共 84 个指标有分数，其中自研 69 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_relu_active_hybrid_delta` | +0.756 | 3 |
| 2 | `feature_measure_fsp_rel_multiscale_delta` | +0.743 | 5 |
| 3 | `feature_measure_channel_gram_cosine_delta` | +0.741 | 6 |
| 4 | `feature_measure_struct01_delta` | +0.729 | 8 |
| 5 | `feature_measure_flow_rel_allpairs_delta` | +0.727 | 10 |
| 6 | `jacobian_spectral_norm` | +0.727 | 11 |
| 7 | `feature_measure_flow_corr_angular_delta` | +0.721 | 12 |
| 8 | `feature_measure_moment_delta` | +0.721 | 13 |
| 9 | `feature_measure_flow_delta` | +0.719 | 14 |
| 10 | `feature_measure_robust_hybrid_delta` | +0.719 | 15 |

## 早期架构·五阶段平均（NB201）

> 跨训练阶段的平均表现。共 84 个指标有分数，其中自研 69 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_flow_corr_symmetric_rel_delta` | +0.709 | 5 |
| 2 | `feature_measure_flow_corr_rel_delta` | +0.709 | 6 |
| 3 | `feature_measure_multiview_max_delta` | +0.701 | 7 |
| 4 | `feature_measure_flow_corr_rms_delta` | +0.697 | 8 |
| 5 | `feature_measure_flow_corr_angular_delta` | +0.697 | 9 |
| 6 | `feature_measure_flow_corr_tophalf_delta` | +0.693 | 10 |
| 7 | `feature_measure_moment_delta` | +0.693 | 11 |
| 8 | `feature_measure_multiview_rms_delta` | +0.685 | 12 |
| 9 | `feature_measure_fsp_rel_multiscale_delta` | +0.684 | 13 |
| 10 | `feature_measure_struct01_delta` | +0.662 | 15 |

## DARTS 搜索空间（e100）

> 换一个搜索空间是否还成立。共 84 个指标有分数，其中自研 69 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `jacobian_spectral_decay` | +0.625 | 7 |
| 2 | `jacobian_effective_rank` | +0.448 | 8 |
| 3 | `perturbation_capacity_signed_delta` | +0.404 | 9 |
| 4 | `feature_measure_flow_corr_worst_delta` | +0.115 | 12 |
| 5 | `feature_measure_flow_corr_angular_delta` | +0.103 | 13 |
| 6 | `feature_measure_flow_corr_rms_delta` | +0.090 | 14 |
| 7 | `feature_measure_mmd` | +0.080 | 15 |
| 8 | `feature_measure_flow_delta` | +0.078 | 16 |
| 9 | `feature_measure_flow_corr_tophalf_delta` | +0.067 | 17 |
| 10 | `perturbation_capacity_layerprofile_delta` | +0.066 | 18 |

## 八设置总榜

> 官方总榜口径。共 84 个指标有分数，其中自研 69 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_multiview_max_delta` | +0.524 | 6 |
| 2 | `feature_measure_multiview_rms_delta` | +0.495 | 7 |
| 3 | `feature_measure_flow_corr_rms_delta` | +0.479 | 9 |
| 4 | `feature_measure_flow_corr_symmetric_rel_delta` | +0.478 | 10 |
| 5 | `feature_measure_flow_corr_angular_delta` | +0.478 | 11 |
| 6 | `feature_measure_flow_corr_rel_delta` | +0.477 | 12 |
| 7 | `feature_measure_flow_corr_tophalf_delta` | +0.477 | 13 |
| 8 | `feature_measure_moment_delta` | +0.472 | 14 |
| 9 | `feature_measure_channel_gram_cosine_delta` | +0.470 | 15 |
| 10 | `feature_measure_struct01_delta` | +0.469 | 16 |

## 400 架构真实预测力（控 clean + 参数量）

> 最接近“能不能预测”的那个数。共 69 个指标有分数，其中自研 58 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_robust_delta` | +0.400 | 1 |
| 2 | `feature_measure_weighted_delta` | +0.397 | 2 |
| 3 | `feature_measure_local_delta` | +0.397 | 3 |
| 4 | `feature_measure_struct_delta` | +0.374 | 4 |
| 5 | `feature_measure_robust_hybrid_delta` | +0.368 | 5 |
| 6 | `feature_measure_robust_rel_delta` | +0.362 | 7 |
| 7 | `jacobian_spectral_norm` | +0.361 | 8 |
| 8 | `feature_measure_robust_hybrid_rel_delta` | +0.352 | 9 |
| 9 | `feature_measure_robust_spatial_delta` | +0.339 | 10 |
| 10 | `feature_measure_robust_spatial_rel_delta` | +0.334 | 11 |

## 跨初始化稳定性（值稳定 rho）

> 换个随机种子序还一样吗。共 69 个指标有分数，其中自研 58 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_flow_rel_allpairs_delta` | +0.973 | 2 |
| 2 | `jacobian_spectral_norm` | +0.973 | 3 |
| 3 | `feature_delta` | +0.967 | 4 |
| 4 | `feature_measure_flow_rel_multiscale_delta` | +0.967 | 5 |
| 5 | `feature_measure_flow_rel_delta` | +0.967 | 6 |
| 6 | `feature_quantile_hausdorff_delta` | +0.964 | 7 |
| 7 | `feature_measure_relu_active_hybrid_rel_delta` | +0.963 | 8 |
| 8 | `feature_hausdorff_delta` | +0.963 | 9 |
| 9 | `feature_measure_spectrum_hellinger_delta` | +0.960 | 10 |
| 10 | `feature_measure_relu_active_hybrid_delta` | +0.960 | 11 |

## 决策价值（挑前 10% 的提升）

> 真拿它当筛选器能赚多少。共 5 个指标有分数，其中自研 3 个。

| 自研内名次 | 指标 | 分数 | 全榜名次 |
|---|---|---|---|
| 1 | `feature_measure_robust_hybrid_delta` | +0.028 | 3 |
| 2 | `feature_measure_weighted_delta` | +0.010 | 4 |
| 3 | `feature_measure_robust_delta` | +0.007 | 5 |

## 多面手：在几个方向上都进过自研前 10

| 指标 | 方向数 | 命中的方向 |
|---|---|---|
| `feature_measure_flow_corr_angular_delta` | 4 | nb201_e0, nb201_mean, darts_e100, all_setting_mean |
| `feature_measure_flow_corr_rms_delta` | 3 | nb201_mean, darts_e100, all_setting_mean |
| `feature_measure_flow_corr_tophalf_delta` | 3 | nb201_mean, darts_e100, all_setting_mean |
| `feature_measure_moment_delta` | 3 | nb201_e0, nb201_mean, all_setting_mean |
| `feature_measure_robust_hybrid_delta` | 3 | nb201_e0, rq2_partial, selection_delta |
| `feature_measure_struct01_delta` | 3 | nb201_e0, nb201_mean, all_setting_mean |
| `jacobian_spectral_norm` | 3 | nb201_e0, rq2_partial, seed_stability |
| `feature_measure_channel_gram_cosine_delta` | 2 | nb201_e0, all_setting_mean |
| `feature_measure_flow_corr_rel_delta` | 2 | nb201_mean, all_setting_mean |
| `feature_measure_flow_corr_symmetric_rel_delta` | 2 | nb201_mean, all_setting_mean |
| `feature_measure_flow_delta` | 2 | nb201_e0, darts_e100 |
| `feature_measure_flow_rel_allpairs_delta` | 2 | nb201_e0, seed_stability |
| `feature_measure_fsp_rel_multiscale_delta` | 2 | nb201_e0, nb201_mean |
| `feature_measure_multiview_max_delta` | 2 | nb201_mean, all_setting_mean |
| `feature_measure_multiview_rms_delta` | 2 | nb201_mean, all_setting_mean |

## 怎么读

- 每张表都是“降序、越大越好”，方向符号已经统一过（`lower` 类指标在数据侧翻成正号）。
- “全榜名次”是 84 项里的位置，可以看出自研在这个方向有没有输给外部方法。
- 只覆盖当前 `MetricResult` 能算出的 69 个指标的三个方向（D6/D7/D8），其余标 `—`，不是 0。
