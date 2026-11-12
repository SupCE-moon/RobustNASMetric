# 三空间指标补充验证（2026-09-22）

## 数据与范围

只使用 `wangjiahao` 容器中 `/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/existing_metrics_full_v3_cpu_repair/` 的既有结果。本次没有重新训练模型、修改原始指标值或改动老师提供的清单。原始 `metric_values.csv` 有 38,080 条数据记录、38,080 个唯一协议键；其中 62 个指标具备八设置完整排名资格。其余 6 个指标仍按原报告的不适用/失败规则排除。

源文件 SHA-256：`metric_values.csv` 为 `703a3e82c40784c28d7e8d051ce8886180cd2e2855fd29db60edfd81a10e4f03`；原始 `analysis/all_setting_table.csv` 为 `ad31af694bb2cf92292a8d8133c8b7f6800190e86d5fa918a4e3934da7059a21`。原始状态是 `COMPLETE_WITH_DECLARED_EXCLUSIONS`，不是无例外的全成功。

## 三种比较口径

- 原始八设置均值：RB40 1 项、NB201 5 个训练阶段、DARTS 2 个训练阶段分别等权求均值。NB201 因此占总权重的 5/8。
- 三空间等权：先分别计算 RB40、NB201 五阶段均值、DARTS 两阶段均值，再将这三项各按 1/3 求均值。这是补充敏感性分析，不替换原始八设置总表。
- 排除 `Standard`：只重新计算 RB40 的方向校正 Spearman（39 个模型）；NB201、DARTS 及指标定义保持不变。`Standard` 是自然训练模型，目标鲁棒准确率为 0，适合检查单个离群模型对相关性的影响。排名始终只在原本完整的 62 个指标内比较。

| 指标 | 原八设置均值/名次 | 三空间等权/名次 | 去 Standard 后 RB40 ρ | 去 Standard 后三空间名次 |
|---|---:|---:|---:|---:|
| `adversarial_loss` | 0.614290 / 1 | 0.624001 / 1 | 0.722470 | 1 |
| `natural_loss` | 0.591311 / 2 | 0.455011 / 2 | 0.318016 | 2 |
| `feature_measure_flow_corr_rel_delta`（E29） | 0.477195 / 4 | 0.256041 / 13 | -0.245142 | 14 |
| `fsp_delta` | 0.479411 / 3 | 0.313765 / 5 | 0.010324 | 6 |
| `feature_measure_struct01_delta` | 0.468764 / 6 | 0.307759 / 6 | 0.010526 | 7 |
| `perturbation_capacity_profile_delta` | 0.463704 / 8 | 0.393931 / 3 | 0.550810 | 3 |
| `perturbation_capacity_channelwise_delta` | 0.419443 / 9 | 0.388006 / 4 | 0.638866 | 4 |

这些名次是描述性排名，并非独立测试结果。`adversarial_loss` 和 `natural_loss` 是有标签的损失基线，不能与“无需标签/训练”的新指标直接当作同等成本的 NAS 指标比较。三空间等权下两个容量指标靠前，但它们在 DARTS e100 的原始相关分别为 -0.174、-0.157，并不是八设置全正。62 个完整指标中，八设置相关均为正的只有上述两个损失基线。

## 架构与规模混杂检查

方法沿用项目既有 `scripts/analyze_darts_structural_confound.py` 的 partial Spearman 口径：对指标、目标和控制变量分别取秩；用最小二乘剔除控制变量；对两组残差计算 Spearman，再按指标方向校正。RB40 控制参数量；NB201 控制四种操作计数（第五种因操作总数固定而省略）；DARTS 控制参数量和六种操作计数（第七种省略）。三个设计矩阵分别有 40/64/100 行，含截距秩分别为 2/5/8，均为满秩。该计算是关联敏感性分析，不构成因果证明；NB201 五个阶段仍是同一批 64 架构。

| 设置 | E29 原始 ρ | E29 控制后 ρ | 容量 profile 原始 ρ | 容量 profile 控制后 ρ |
|---|---:|---:|---:|---:|
| RB40 | -0.154 | -0.133 | 0.584 | 0.582 |
| NB201 e0 | 0.713 | 0.086 | 0.672 | 0.130 |
| NB201 e5 | 0.738 | 0.364 | 0.622 | 0.370 |
| NB201 e10 | 0.756 | 0.376 | 0.827 | 0.519 |
| NB201 e25 | 0.679 | 0.326 | 0.527 | 0.191 |
| NB201 e50 | 0.659 | 0.253 | 0.568 | 0.189 |
| DARTS e0 | 0.421 | -0.123 | 0.084 | -0.179 |
| DARTS e100 | 0.005 | -0.009 | -0.174 | 0.041 |

E29 旧文档 `e29_flow_rel_variants_ceiling.md` 的 NB201 e0 原始/控制后数值是 0.645/0.480，来自旧实验协议；它与这次统一三空间协议的 0.713/0.086 不可直接合并或视为复现。本次分析严格使用 v3 的同一批指标、目标和老师清单操作计数。

## 独立留出验证的现状与冻结方案

本次正式清单的 RB40 40 个、NB201 64 个、DARTS 100 个 ID 均已在 v3 中出现，没有可从清单直接划出的未见样本。安装在容器里的 RobustBench CIFAR-10 Linf Model Zoo 注册了 76 个名称，其中 36 个名称不在 RB40 清单；名称差异不保证权重独立，也不保证同口径的官方 AutoAttack 标签已核实，所以目前不能报告留出相关性。RobustBench 的[官方项目](https://github.com/RobustBench/robustbench)提供模型加载与基准评测入口。

后续先固定两条不调参的候选：E29 `feature_measure_flow_corr_rel_delta`（原八设置目标），以及 `perturbation_capacity_profile_delta`（三空间等权敏感性目标）；以 `adversarial_loss`、`natural_loss` 和 `fsp_delta` 为对照。不得根据新样本结果反复更换公式、方向或挑选子集，再把同一批样本称为留出集。

RB 留出候选的**预选规则**已在计算任何新模型指标前固定：取当前容器 RobustBench 1.1.1 注册的 CIFAR-10 Linf 名称，减去老师 RB40 清单的精确名称，对余下 36 个名称按 `SHA-256(UTF-8 名称)` 的十六进制值升序排序，取前 12 个。顺序为：`Jia2022LAS-AT_34_10`、`Wu2020Adversarial_extra`、`Cui2023Decoupled_WRN-34-10`、`Gowal2021Improving_70_16_ddpm_100m`、`Rebuffi2021Fixing_70_16_cutmix_extra`、`Debenedetti2022Light_XCiT-M12`、`Wang2020Improving`、`Wu2020Adversarial`、`Rebuffi2021Fixing_70_16_cutmix_ddpm`、`Cui2023Decoupled_WRN-28-10`、`Rebuffi2021Fixing_28_10_cutmix_ddpm`、`Gowal2021Improving_R18_ddpm_100m`。这只是防止按标签/指标成绩选模的候选冻结，不是通过权重和官方标签验收的最终样本。2026-09-22 查看官方 README 时，12 个名称在 CIFAR-10 Linf 表格中各出现一次；其中至少一个名称在其他威胁模型表格也出现，因此取标签时必须限定到 CIFAR-10 Linf 8/255 区块。服务器读取官方 GitHub 原文件曾成功，随后一次读取超时；网络可用性也未通过持续性验收。

下一批 RB 模型必须先核验与 RB40 无相同权重/别名、权重可加载、官方 CIFAR-10 Linf 8/255 AutoAttack 目标可信；固定名单和文件哈希后，在 `wangjiahao` 容器内用同一输入样本/扰动/种子只计算冻结的指标。任何失败需逐项列出，不能填 0。NB201/DARTS 若要真正跨空间独立验证，还需要新的架构 ID、对应检查点与同协议最终鲁棒标签；现有老师清单不足以完成这一步。

**结论：** 原表 E29 的“前五”成立，但对空间权重、RB40 `Standard` 和操作组成控制均不稳健。当前不应宣称已获得稳定通用的跨空间新指标，也不应在同一数据上继续调配方追求名次。独立留出实验尚未完成。

**留出模型下载试验：** 在 `wangjiahao` 容器的 `outputs/student_metrics/wangjiahao/heldout_rb12_20260922/models/` 中尝试加载预选候选 `Wang2020Improving`。RobustBench 开始下载权重，但 120 秒限时后退出码为 124；复查目标目录没有文件。因此该模型尚未通过可用性验收，本次没有计算任何新模型指标或留出相关性。后续不能将下载失败记为指标 0。
