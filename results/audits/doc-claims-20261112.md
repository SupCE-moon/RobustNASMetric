# 文档声称 vs 发布表：数字对账

- 发布表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\.worktrees\mainline-consolidation\outputs\strict_adv_excess_release_20261024T110422Z\RobustMetric_严格全部指标总表_86项_三位小数.csv`（sha256 `f69717bdbf02f7aba03985bae8a813ab3bd0190936604f9f3662f51357ef5276`）
- 声称条数：33，通过 33，失败 0
- 容差：0.0006（表保留三位小数，文档常写四位）

## 逐条核对

| 文档 | 位置 | 声称 | 表里的值 | 差值 | 结果 |
| --- | --- | --- | --- | --- | --- |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `trnas_consistency_phi` 的 `all_setting_mean` = 0.7822 | 0.782 | -0.000200 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `trnas_consistency_phi` 的 `strict_rank` = 1 | 1.0 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `trnas_r_score` 的 `all_setting_mean` = 0.7378 | 0.738 | +0.000200 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `trnas_r_score` 的 `strict_rank` = 2 | 2.0 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `adversarial_loss` 的 `all_setting_mean` = 0.614 | 0.614 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `trnas_activation_psi` 的 `all_setting_mean` = 0.5981 | 0.598 | -0.000100 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `natural_loss` 的 `all_setting_mean` = 0.591 | 0.591 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `feature_measure_multiview_max_delta` 的 `all_setting_mean` = 0.5241 | 0.524 | -0.000100 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `feature_measure_multiview_max_delta` 的 `strict_rank` = 6 | 6.0 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `feature_measure_multiview_rms_delta` 的 `all_setting_mean` = 0.495 | 0.495 | +0.000000 | 通过 |
| `docs/PROJECT_STATUS.md` | §3 / §4 榜单 | `fsp_delta` 的 `all_setting_mean` = 0.479 | 0.479 | +0.000000 | 通过 |
| `docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md` | 八设置均值差 | `feature_measure_multiview_max_delta` − `trnas_consistency_phi` = -0.258 | -0.258 | +0.000000 | 通过 |
| `docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md` | 八设置均值差 | `feature_measure_multiview_max_delta` − `trnas_r_score` = -0.214 | -0.214 | +0.000000 | 通过 |
| `docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md` | 八设置均值差 | `feature_measure_multiview_max_delta` − `trnas_activation_psi` = -0.074 | -0.074 | +0.000000 | 通过 |
| `docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md` | 八设置均值差 | `feature_measure_multiview_rms_delta` − `trnas_consistency_phi` = -0.287 | -0.287 | -0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 Φ | `trnas_consistency_phi` 的 `nb201_e0` = 0.849 | 0.849 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 Φ | `trnas_consistency_phi` 的 `nb201_e5` = 0.825 | 0.825 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 Φ | `trnas_consistency_phi` 的 `nb201_e10` = 0.797 | 0.797 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 Φ | `trnas_consistency_phi` 的 `nb201_e25` = 0.753 | 0.753 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 Φ | `trnas_consistency_phi` 的 `nb201_e50` = 0.771 | 0.771 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 max | `feature_measure_multiview_max_delta` 的 `nb201_e0` = 0.686 | 0.686 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 max | `feature_measure_multiview_max_delta` 的 `nb201_e5` = 0.735 | 0.735 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 max | `feature_measure_multiview_max_delta` 的 `nb201_e10` = 0.75 | 0.75 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 max | `feature_measure_multiview_max_delta` 的 `nb201_e25` = 0.677 | 0.677 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 max | `feature_measure_multiview_max_delta` 的 `nb201_e50` = 0.657 | 0.657 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 RMS | `feature_measure_multiview_rms_delta` 的 `nb201_e0` = 0.663 | 0.663 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 RMS | `feature_measure_multiview_rms_delta` 的 `nb201_e5` = 0.716 | 0.716 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 RMS | `feature_measure_multiview_rms_delta` 的 `nb201_e10` = 0.761 | 0.761 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 RMS | `feature_measure_multiview_rms_delta` 的 `nb201_e25` = 0.653 | 0.653 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | NB201 五阶段 RMS | `feature_measure_multiview_rms_delta` 的 `nb201_e50` = 0.63 | 0.63 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | DARTS 塌陷 | `feature_measure_multiview_max_delta` 的 `darts_e100` = 0.006 | 0.006 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | DARTS 塌陷 | `feature_measure_multiview_rms_delta` 的 `darts_e100` = -0.036 | -0.036 | +0.000000 | 通过 |
| `docs/experiments/自研指标定位与TRNAS对照进度_20261030.md` | DARTS 对照 | `trnas_consistency_phi` 的 `darts_e100` = 0.847 | 0.847 | +0.000000 | 通过 |

## 结构性检查

| 检查项 | 期望 | 实际 | 结果 |
| --- | --- | --- | --- |
| 总行数 | 86 | 86 | 通过 |
| 入榜行数 | 84 | 84 | 通过 |
| 排除行数 | 2 | 2 | 通过 |
| 每行原始行数唯一值 | [560] | [560] | 通过 |
| 名次连续 1..84 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84] | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84] | 通过 |
| release_id 唯一 | 1 | 1 | 通过 |

## 这条检查能证明什么、不能证明什么

- 能证明：文档里写的这些数字，与当前发布表**逐个对得上**；没有出现「引用旧版发布」或抄写错误。
- 不能证明：这些数字**本身是对的**。它只保证文档与表一致；表是否正确，要看 `results/audits/strict-ranking-recompute-20261112.md` 的独立复算。
- 声称清单是**手工维护**的，只覆盖被反复引用、最容易出错的那批数字；它不代表全部文档都已核对。
