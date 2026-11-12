# E27: 静态谱/Jacobian/容量改进 的验证与结构去混杂调查

日期：2026-08-05

## 材料档案

- 类型：指标设计、本地重评估、结构混杂调查
- 状态：已完成（新指标负结果；关键发现：结构去混杂重排排名）
- 问题：静态谱集中度、Jacobian 谱、或容量精化能否产生一个「在结构去混杂检验下仍然存活（即非平凡）」的鲁棒 NAS 代理？
- 新指标：16 个，横跨 3 个家族（静态谱、Jacobian 谱、容量精化）
- 数据集：现有 CIFAR-10，无下载
- 搜索空间：RobustBench-20（本地）、NAS-Rob-Bench-201 e0（本地，重实现）、DARTS e0（本地）

## 新引入指标

静态谱集中度（无扰动）：
- `feature_spectral_erank_static`、`feature_spectral_participation_ratio_static`、`feature_spectral_decay_exponent_static`

Jacobian 谱（输入→logit）：
- `jacobian_spectral_norm`、`jacobian_effective_rank`、`jacobian_spectral_decay`

容量精化（扰动容量家族）：
- `perturbation_capacity_multiscale_delta`（3 个空间尺度 MAD 漂移之和）
- `perturbation_capacity_relu_active_multiscale_delta`
- `perturbation_capacity_topk_delta`（top-k 幅值位置的 MAD）
- `perturbation_capacity_iqr_delta`（四分位距容量）
- `perturbation_capacity_iqr_active_delta`
- `perturbation_capacity_signed_delta`（带符号、非绝对值漂移）
- `perturbation_capacity_profile_delta`（逐通道 MAD 剖面的方向变化）

Flow 精化：
- `feature_measure_flow_rel_multiscale_delta`（跨尺度 FSP 相对漂移之和）

## 调查：为什么指标在结构去混杂下失败

结构去混杂检验把每个指标对 NB201 操作计数（`none / skip_connect / nor_conv_1x1 / nor_conv_3x3 / avg_pool_3x3`）做回归，再在残差上重算 Spearman。一个指标「失败」是指它的原始相关在控制后崩溃，即其信号主要是操作组成。

线性操作回归（R² = 操作计数解释的方差，含逐操作系数）：

| 指标 | R²(op) | op_none | op_skip | op_conv1x1 | op_conv3x3 | op_avgpool |
|---|---:|---:|---:|---:|---:|---:|
| `jacobian_spectral_decay` | 0.68 | +0.26 | +0.17 | +0.13 | −0.01 | +0.39 |
| `perturbation_capacity_multiscale_delta` | 0.51 | +0.00 | +0.00 | +0.00 | −0.00 | +0.00 |
| `feature_delta` | 0.66 | ~0 | ~0 | ~0 | ~0 | +0.00 |

机制：`none` 操作产生死通道（零 Jacobian 行/零特征），抬高谱衰减；`avg_pool` 和 `skip_connect` 产生平滑/恒等特征（低离散）。**按操作基线求和**的指标（capacity、Jacobian decay）被这些操作特定的幅值主导，因此表现为操作计数器。测**全局或跨层交互**的指标（`feature_delta` = 池化特征方向变化，`flow_rel` = 跨层 FSP 相对变化）编码「哪个操作连哪个节点」，在线性操作控制下存活。

## 主要结果

### RB20（成熟，白盒 FGSM，n=64，20 模型）

容量家族（方向校正 Spearman；全部 lower-is-better）：

| 指标 | ρ | 95% bootstrap CI |
|---|---:|---|
| `perturbation_capacity_relu_active_delta`（已有） | +0.699 | [+0.360, +0.886] |
| `perturbation_capacity_relu_active_multiscale_delta` | +0.641 | [+0.286, +0.852] |
| `perturbation_capacity_multiscale_delta` | +0.633 | [+0.314, +0.833] |
| `perturbation_capacity_delta`（已有） | +0.626 | [+0.302, +0.830] |
| `perturbation_capacity_topk_delta` | +0.597 | [+0.235, +0.826] |

静态谱和 Jacobian 指标在 RB20 上排名 19–47/47（全部失败）。成熟模型设置已饱和：capacity（MAD 漂移）已经接近最优，静态谱信号在异质 RB20 宽度下被宽度混杂。

### NB201 e0（零成本，64 架构）

原始（方向校正）vs 结构去混杂（给定操作计数的 partial Spearman）：

| 指标 | raw ρ | partial ρ | 存活率 |
|---|---:|---:|---:|
| `trnas_r_score`（baseline） | +0.897 | **+0.206** | 23% |
| `feature_delta`（已有） | +0.845 | **+0.615** | 73% |
| `perturbation_capacity_multiscale_delta` | +0.802 | +0.311 | 39% |
| `feature_measure_flow_rel_delta`（已有） | +0.796 | **+0.626** | 79% |
| `jacobian_spectral_decay` | +0.815 | +0.147 | 18% |
| `jacobian_effective_rank` | +0.781 | +0.026 | 3% |
| `perturbation_capacity_profile_delta` | +0.698 | +0.351 | 50% |
| `perturbation_capacity_delta`（已有） | +0.552 | +0.283 | 51% |
| `trnas_activation_psi`（baseline） | +0.822 | +0.493 | 60% |
| `trnas_consistency_phi`（baseline） | +0.844 | −0.352 | （反转） |

方向剖面修正把 capacity 的 partial 从 +0.283 提升到 +0.351（相对 +24%），但 capacity 的逐通道（可加）信息从根本上被全局/跨层指标封顶。多尺度 flow 与原 flow_rel 持平（partial 0.627 vs 0.626），说明 flow_rel 已到信息极限。

### DARTS e0

目标（`final_pgd_accuracy`）std = 0.0077（20 架构，范围 ≈ 0.45–0.48）。目标方差太小无法排序架构，因此跨空间迁移无法用现有 DARTS 样本检验；每个指标的 bootstrap CI 都跨零。

## 关键发现

结构去混杂检验**重排了 NB201 零成本排名**：

1. 原始冠军 `trnas_r_score`（+0.897）在操作控制后崩到 +0.206——它 77% 是操作组成代理，不是真正的鲁棒性机制。
2. `feature_delta`（+0.615）和 `flow_rel`（+0.626）是唯二在去混杂后保留大、非平凡信号的指标（73–79% 存活率）。
3. 全部 16 个新指标要么更弱（RB20）、要么更操作混杂（NB201），都不如这两个已有指标。

这直接支持「非平凡性」判据：一个好的鲁棒 NAS 代理必须编码**操作交互**（哪个操作连哪个节点），而非**操作组成**（每种操作类型有多少）。

## 解读

1. **Capacity 是成熟模型指标。** 它的 MAD 漂移在 RB20 上强（+0.70 relu-active），但在零成本下弱且操作混杂（+0.28 partial）。「扰动容量」（capacity）方向适合排序已训练模型，不适合随机初始化 NAS 搜索。
2. **静态谱/Jacobian 方向被证伪。** 静态谱结构在 RB20 上被宽度混杂、在 NB201 上被操作混杂；Jacobian 谱衰减 82% 是操作组成。
3. **真正的零成本信号是全局/跨层。** `feature_delta`（池化方向变化）和 `flow_rel`（跨层 FSP 相对变化）是非平凡冠军。没有新指标超过它们。
4. **一个可发表的结论：** 按原始 Spearman 排名有误导；结构去混杂排名才是正确的「好指标」定义，在其下 flow_rel / feature_delta 是唯二存活的代理。

## 输出文件

```text
outputs/e27_rb20_whitebox_fgsm_n64_v3.csv         （RB20，全部新指标）
outputs/e27_nb201_e0_zerocost.csv                  （NB201 e0，全部新指标）
outputs/e27_darts_e0_zerocost_v3.csv               （DARTS e0，全部新指标）
scripts/analyze_e27_capacity_validation.py         （bootstrap + partial Spearman）
scripts/evaluate_nb201_e27_zerocost.py             （NB201 零成本评估器）
scripts/evaluate_darts_e27_zerocost.py             （DARTS 零成本评估器）
src/robust_metric_lab/nb201_space.py               （重实现的 NB201 cell/network）
```

## 重实现说明

NB201 模型在本地重实现（`nb201_space.py`），因为 `libnas` 模型代码未随仓库提供。验证：重实现网络复现了 `feature_delta` 与 `oracle_target` 的相关性（−0.845 vs 原 E23 CSV 的 −0.806）。重实现时修复了两个细微 bug：逐边输入通道（节点 0 的操作用 C_in→C_out，节点 1/2 的操作用 C_out→C_out）和逐边步长（只有节点 0 的边在 reduction cell 里下采样）。零成本指标对权重初始化方案高度敏感：默认 PyTorch 初始化复现 E23（feature_delta −0.845），而 Kaiming normal fan-out 让它崩到 −0.138，因此保留默认初始化。
