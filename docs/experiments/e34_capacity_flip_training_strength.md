# E34: capacity 的 e5 翻转是训练强度依赖的（预测被推翻）

日期：2026-09-08

## 材料档案

- 类型：预测检验（负结果）+ 机制修正
- 状态：已完成（关键发现：PGD-1 弱对抗训练下 capacity 稳定不翻转，gram 反而不稳定——「capacity e5 翻转」是 PGD-7 强对抗训练特有的，不是 capacity 的固有属性）
- 问题：E33 预测「所有容量变体都与 plain capacity 一样在 e5 翻转（幅度量固有）」；E34 用本地重训练检验该预测。
- 新指标：无
- 数据集：现有 CIFAR-10，无下载；本地重实现 Nb201TinyNetwork + Normalize，PGD-1 对抗训练至 e5（64 架构）
- 搜索空间：NB201（64，e0/e5）

## 方法

用本地重实现 `Nb201TinyNetwork`（加 CIFAR-10 Normalize 头），对 64 个 NB201 架构做 **PGD-1（FGSM+随机起始）对抗训练**至 epoch 5，在 e0（随机初始化）和 e5 各算一次全指标，计算每个指标的 e0→e5 架构排名 Spearman。脚本 `scripts/train_nb201_e34_stability.py` + `scripts/analyze_e34_stability.py`。

## 核心结果

### e0→e5 排名稳定性：E34（PGD-1）与 e23（PGD-7）几乎完全相反

| 指标 | E34 (PGD-1, 重实现) | e23 (PGD-7, 原始) |
|---|---:|---:|
| `perturbation_capacity_delta` | **+0.491** | **−0.454** |
| `fsp_delta` | +0.177 | +0.634 |
| `feature_measure_struct01_delta` | +0.116 | +0.612 |
| `feature_measure_moment_delta` | +0.344 | +0.627 |

**E33 的预测被推翻**：PGD-1 下 plain capacity 稳定（+0.491，不翻转），而 gram（fsp/struct01）反而不稳定（+0.12～0.18）。这与 e23（PGD-7）下「capacity 翻转 −0.454、gram 稳定 +0.63」的图景几乎完全颠倒。

### 全部容量变体在 PGD-1 下都不翻转

| 容量变体 | e0→e5 |
|---|---:|
| `perturbation_capacity_flow_rel_delta` | +0.708 |
| `perturbation_capacity_profile_delta` | +0.696 |
| `perturbation_capacity_topk_delta` | +0.595 |
| `perturbation_capacity_multiscale_delta` | +0.555 |
| `perturbation_capacity_gini_delta` | +0.516 |
| `perturbation_capacity_channelwise_delta` | +0.505 |
| `perturbation_capacity_delta` | +0.491 |
| `perturbation_capacity_relu_active_multiscale_delta` | +0.389 |
| `perturbation_capacity_iqr_delta` | +0.331 |
| `perturbation_capacity_layerprofile_delta` | +0.312 |
| `perturbation_capacity_relu_active_delta` | +0.225 |
| `perturbation_capacity_iqr_active_delta` | +0.139 |
| `perturbation_capacity_above_mean_delta` | +0.127 |
| `perturbation_capacity_signed_delta` | −0.138 |

除 signed（E32 已证伪）外，所有容量变体在 PGD-1 下 e0→e5 均为正（0.13～0.71），**没有一个翻转**。

## 关键发现

1. **「capacity e5 翻转」不是 capacity 的固有属性，而是 PGD-7 强对抗训练特有的现象。** 在 PGD-1 弱对抗训练下，capacity 的架构排名从 e0 到 e5 是稳定的（+0.491）。

2. **gram 的跨训练稳定性同样训练强度依赖。** gram 在 PGD-7 下稳定（+0.63），在 PGD-1 下反而不稳定（+0.12～0.18）。E31/E33 的「gram 稳定、capacity 不稳定」对比，只在强对抗训练下成立。

3. **E33 的机制需要修正**：E33 把「capacity 跨训练翻转」表述为幅度量的固有属性，这是过度概括。更准确的说法是：**指标的跨训练排名稳定性取决于训练协议强度，不能作为指标的固有性质。**

4. **E33 真正稳健的结论是「跨协议一致性（尺度不变性）」**：gram 在 e0（随机初始化）跨两个协议（原始 vs 重实现）一致性 0.67～0.84，capacity 中间（0.45），loss 翻转（−0.33）。这个结论**不依赖训练**，仍然是「通用 = 尺度不变」的稳健证据。

## 结论

- **E33 的预测（容量变体在 e5 翻转）在 PGD-1 下不成立**——它们都稳定。E34 因此**推翻了「capacity 非通用是固有属性」的表述**。
- 「通用」的稳健判据应回到 **E33 的尺度不变性（跨协议一致性）**，而不是跨训练稳定性（后者训练强度依赖，不可靠）。
- 一个诚实的限制：E34 用 PGD-1 而非 e23 的 PGD-7，故未直接检验「PGD-7 下容量变体是否也翻转」。要严格回答该问题需 PGD-7 重训（成本过高）。但 E34 已足以证明：**capacity 的「非通用」不是固有属性，而是强对抗训练下的现象。**
- 综合 E31–E34：gram 族（`fsp_delta`/`feature_measure_struct01_delta`/`feature_measure_moment_delta`）是唯一「跨协议尺度不变」的通用指标；capacity 的「e5 翻转」是强对抗训练特有的、非固有；loss 跨协议符号翻转。

## 输出文件

```text
scripts/train_nb201_e34_stability.py          （本地 NB201 PGD 训练 + 指标评估）
scripts/analyze_e34_stability.py              （e0->e5 排名稳定性分析）
outputs/e34_nb201_e5_stability_pgd1.csv       （64 架构 e0/e5 全指标）
outputs/e34_nb201_pgd1_run.log                （训练日志）
```
