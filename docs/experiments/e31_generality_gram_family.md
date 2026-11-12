# E31: 通用性分析——gram 族是真正的通用指标，capacity 不是

日期：2026-08-24

## 材料档案

- 类型：跨设置通用性诊断 + gram 族强化
- 状态：已完成（关键发现：通用指标是 gram 族，capacity 在 e5 符号翻转）
- 问题：哪个指标在所有设置（成熟 + 零成本 + 各训练阶段）方向一致且不差？
- 新指标：`feature_measure_fsp_rel_multiscale_delta`（多尺度通道 gram）
- 数据集：现有 CIFAR-10，无下载
- 搜索空间：RobustBench-20（成熟）、NAS-Rob-Bench-201（e0/e5/e10/e25/e50）

## 通用性判据

「好指标必须有通用性，不能这个环境下行那个环境下不行」。为此，对每个指标检查它在 **RB20（成熟异构）** 和 **NB201 五个训练阶段（e0/e5/e10/e25/e50）** 的方向校正 Spearman 是否**方向一致**且**处处为正**。

## 核心结果

### NB201 各训练阶段方向校正 Spearman

| 指标 | e0 | e5 | e10 | e25 | e50 | min |
|---|---:|---:|---:|---:|---:|---:|
| **`fsp_delta`（通道 gram）** | +0.77 | +0.73 | +0.69 | +0.65 | +0.67 | **+0.65** |
| **`feature_measure_moment_delta`** | +0.79 | +0.73 | +0.71 | +0.67 | +0.65 | **+0.65** |
| **`feature_measure_struct01_delta`** | +0.77 | +0.72 | +0.68 | +0.63 | +0.66 | **+0.63** |
| `feature_measure_flow_rel_delta` | +0.79 | +0.63 | +0.68 | +0.53 | +0.50 | +0.50 |
| `feature_measure_cov_delta` | +0.76 | +0.42 | +0.62 | +0.33 | +0.32 | +0.32 |
| `perturbation_capacity_delta` | +0.70 | **−0.40** | +0.19 | +0.24 | +0.21 | **−0.40** |

### RB20（成熟）方向校正 Spearman

| 指标 | ρ |
|---|---:|
| `feature_measure_struct01_delta` | +0.499 |
| `fsp_delta` | +0.493 |
| `feature_measure_moment_delta` | +0.380 |
| `feature_measure_flow_rel_delta` | +0.205 |

## 关键发现

1. **通用指标是 gram 族**：`fsp_delta`（通道 gram 相对变化）、`moment`（方向+gram）、`struct01`（局部+gram）在 NB201 全部五个训练阶段方向一致、min ≥ 0.63，在 RB20 上也稳定在 0.38–0.50。它们处处不为负，从不在某个环境下失效。

2. **capacity 不是通用指标**：`perturbation_capacity_delta` 在 e0 为 +0.70、e5 翻转为 −0.40。这正是 E17/E27 观察到的「capacity 训练后反向」现象——它测的「训练质量」在随机初始化阶段方向相反。capacity 是「成熟模型专用」，不是通用。

3. **flow_rel 是「纯零成本」但非通用**：去混杂 partial 最高（0.626），但 RB20 只有 0.205。

4. **通用性与纯度是两个正交维度**：gram 族通用（处处正）但零成本去混杂 partial 中等（0.46–0.54）；flow_rel 纯（partial 0.626）但不通用；capacity 成熟强（0.699）但不通用（e5 翻转）。

## 结论

- **真正的通用指标是 gram 族**：`fsp_delta`、`feature_measure_moment_delta`、`feature_measure_struct01_delta`，全部是 `(Ours)` 的 feature_measure 家族（扰动响应 delta）。
- 它们满足「处处不差」：NB201 全阶段 min ≥ 0.63，RB20 0.38–0.50。
- 多尺度通道 gram（`feature_measure_fsp_rel_multiscale_delta`）在 NB201 e0 有提升（raw 0.599→0.690）但 RB20 反而下降（0.493→0.370），故不作为通用指标保留。
- 「扰动容量」（capacity）方向应**从 MAD 离散度（capacity）转向特征结构（gram）**：gram 族保留符号/结构（这正是去混杂后存活的信号），因此通用；capacity 取绝对值丢结构，因此只在成熟模型有效且 e5 翻转。

## 输出文件

```text
outputs/e28_capacity_family/nb201_e0_v3.csv      （含多尺度 gram）
outputs/e28_capacity_family/rb20_whitebox_v3.csv  （RB20 全指标，含多尺度 gram）
src/robust_metric_lab/metrics.py                  （+多尺度通道 gram）
```
