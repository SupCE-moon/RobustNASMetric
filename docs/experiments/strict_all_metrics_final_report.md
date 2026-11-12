# 严格全指标实验最终报告

## 结论

本次总榜只使用通过完整性门槛的指标。最终共有 80 个跨空间指标进入排名，`All-setting mean` 为八个预先固定设置的方向统一 Spearman 相关系数均值。原始合并表有 47,600 行，47,600 个唯一键；发布校验结果为 `PASS`。

总榜前五为：

| 排名 | 指标 | All-setting mean |
|---:|---|---:|
| 1 | `adversarial_loss` | 0.6143 |
| 2 | `natural_loss` | 0.5913 |
| 3 | `feature_measure_multiview_max_delta`（Ours E37） | 0.5241 |
| 4 | `feature_measure_multiview_rms_delta`（Ours E37） | 0.4950 |
| 5 | `fsp_delta` | 0.4794 |

因此，新增指标中整体表现最好的两个是 `feature_measure_multiview_max_delta` 和 `feature_measure_multiview_rms_delta`。它们在八设置总榜中分别为第 3、4 名；这是严格主实验结果，不是只在 NB201 或 DARTS 子集上的结果。

## 固定实验协议与覆盖

每个合格的跨空间指标均在下列 560 个对象/检查点上计算，且每个原始行使用固定的 `sample_offset=0`、`num_samples=64`、`seed=20260808`、`epsilon=8/255`：

| 空间 | 设置 | 每设置对象数 | 合计 |
|---|---:|---:|---:|
| RobustBench RB40 | mature | 40 | 40 |
| NAS-Bench-201 | e0/e5/e10/e25/e50 | 64 | 320 |
| DARTS | e0/e100 | 100 | 200 |
| **总计** | **8 个设置** |  | **560** |

RB40 的标签为 RobustBench AutoAttack；NB201 的标签为官方最终 AutoAttack；DARTS 的标签为完整 CIFAR-10 测试集 PGD-20。相关性均按指标预先登记的方向统一后计算。任何一项若不能给出 560 条唯一、有限且 `status=ok` 的原始行，就不能进入 `All-setting mean`。

## 原始数据合并与验收

最终发布目录为服务器容器内：

`/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/strict_final_release_20261008T000000Z/`

它合并了：既有指标的严格 V4 修复数据、E36 的 8 个指标共 4,480 行，以及 E37 的 9 个预注册指标共 5,040 行。发布校验为：

```json
{"status":"PASS","raw_rows":47600,"unique_raw_keys":47600,"ranked_metrics":80,"excluded_metrics":5}
```

这意味着没有以“RB40 只跑一部分模型”或“缺少 DARTS 设置”的方式计算总榜。E36 和 E37 均以完整 560 行网格参与本次严格排名。

## 冻结留出验证

E37 的 9 个指标额外使用冻结的 11 个 RB40 权重验证，未用这些模型选择或调整指标。结果为 99 条唯一原始行：其中 10 个有效模型 × 9 指标 = 90 条成功行；`Rebuffi2021Fixing_70_16_cutmix_ddpm` 为已确认损坏权重，9 条对应行保留为显式失败，不以零或插补值替代。

在 20,000 次模型 bootstrap 下，`feature_measure_multiview_max_delta` 的留出方向统一 rho 为 0.5394，95% 区间为 [-0.0566, 0.8765]；`feature_measure_multiview_rms_delta` 为 -0.3939，区间 [-0.9500, 0.3585]。10 个有效模型的样本量较小、区间较宽，因此这些结果用于检验外推稳定性，不能替代主实验总榜，也不能作为进一步按标签调参的依据。

## 结构混杂诊断

诊断使用冻结的、与鲁棒标签无关的清单字段：RB40 使用 `log(parameter_count)` 和架构族；NB201 使用五类操作计数；DARTS 使用 `log(parameter_count)` 和七类操作计数。共输出 680 条“指标×设置”诊断记录；672 条可计算偏 Spearman。8 条不可计算记录恰好来自两个已排除的 RACL 指标，它们没有可用的原始成功行。

该诊断是稳健性检查，而非新的排名标准：总榜仍只由原始、方向统一的 Spearman 及完整性规则决定。控制变量、变换和来源记录在发布目录的 `controls/manifest_control_provenance.json`。

## 排除项与已知限制

下列 5 个项目没有进入排名，理由已经固定并保留在 `ranking/excluded_metrics.csv`：

| 指标 | 原因 |
|---|---|
| `racl_arch_lipschitz_log_upper` | 架构单元上界对固定 RB40 网络无数学定义，属于空间特异量。 |
| `racl_fixed_network_log_upper_proxy` | 固定网络代理量不是同一数学对象，属于空间特异量。 |
| `trnas_activation_psi`、`trnas_consistency_phi`、`trnas_r_score` | 在冻结的 `1e-6` 定义下，Kang/TRNAS 的步长计算发生未解决的数值下溢；CPU 隔离复现实验仍然失败。 |

这些不是“用部分成功结果继续排名”的情况。它们的原始失败和诊断证据被保留；在得到可复现、定义不变的修复前，不应进入跨空间比较。

## 可复核文件

- `metric_values_merged.csv`：47,600 条原始行。
- `ranking/strict_ranking.csv`：80 个合格指标的严格总榜。
- `ranking/excluded_metrics.csv`：5 个排除项与理由。
- `ranking/confound_diagnostics.csv`：680 条结构混杂诊断。
- `release_verification.json`：机器可读的 PASS 验收。
- `heldout_rb12_20260922/e37_heldout_analysis_v1/`：E37 冻结留出验证、相关性表和 bootstrap 区间。
