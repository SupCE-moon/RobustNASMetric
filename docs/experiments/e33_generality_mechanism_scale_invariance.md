# E33: 通用性的机制——尺度不变性（相对结构）决定通用

日期：2026-09-03

## 材料档案

- 类型：机制诊断 + 尺度不变性原理
- 状态：已完成（关键发现：通用指标 = 尺度不变的相对结构量；capacity（幅度）跨训练不稳定；loss（绝对幅度）跨协议翻转；gram 是唯一同时满足两者的通用指标）
- 问题：为什么 gram 族通用、capacity 族 e5 翻转、loss 又是什么行为？三者背后有没有统一的机制？
- 新指标：无（本轮为机制诊断）
- 数据集：现有 CIFAR-10，无下载；纯分析（e23 CSV 全 5 阶段 + nb201_e0_v3.csv + rb20_whitebox_v3.csv）
- 搜索空间：NB201（64，e0/e5/e10/e25/e50）、RB20（成熟 20）、DARTS（20，仅 oracle 退化记录）

## 动机

E31/E32 建立了「gram 通用、capacity 非通用（e5 翻转）」的经验事实，但 E32 同时证伪了「取绝对值丢符号」的解释。E33 追问真正的机制：把「跨训练阶段排名稳定性」和「跨协议（网络实现/归一化）一致性」两个正交的稳健性维度都测一遍，看三者（gram / capacity / loss）分别落在哪里，能否统一成一个原理。

## 方法

1. **跨训练排名稳定性**：对每个指标，计算它在 e23 CSV 中 e0↔e5、e0↔e50 的架构排名 Spearman。
2. **跨协议一致性**：对每个指标，计算同一 64 个架构在「原始 nasrobbench201 + 归一化」（e23）与「重实现 Nb201TinyNetwork + 无归一化」（nb201_e0_v3）两个协议下的指标值 Spearman。
3. **容量变体归类**：把 E28 新增的 10 个容量变体按「与 plain capacity 的相似度（幅度）/ 与 gram 的相似度（结构）」归类。

## 核心结果

### 1. 跨训练排名稳定性：capacity 不稳定，gram 稳定

| 指标 | e0↔e5 | e0↔e50 |
|---|---:|---:|
| `perturbation_capacity_delta` | **−0.454** | +0.218 |
| `fsp_delta` | +0.634 | +0.670 |
| `feature_measure_struct01_delta` | +0.612 | +0.644 |
| `feature_measure_moment_delta` | +0.627 | +0.688 |

capacity 在 e0→e5 的架构排名几乎反转（−0.454），且 e0 与 e50 仅弱相关（+0.218）；gram 全程稳定（e0↔e5 ≥ 0.61，e0↔e50 ≥ 0.64）。**capacity 测的是随训练动力学剧烈重排的量，gram 测的是由架构接线决定的稳定量。**

补充——各阶段方向校正 Spearman 的 95% bootstrap CI（n=64，2000 次重抽样），证明「处处为正/翻转」是统计显著的：

| 指标 | e0 | e5 | e10 | e25 | e50 |
|---|---|---|---|---|---|
| `fsp_delta` | +0.77 [+0.62,+0.87] | +0.73 [+0.57,+0.83] | +0.69 [+0.52,+0.80] | +0.65 [+0.47,+0.75] | +0.67 [+0.49,+0.78] |
| `feature_measure_struct01_delta` | +0.77 [+0.60,+0.87] | +0.72 [+0.56,+0.83] | +0.68 [+0.51,+0.79] | +0.63 [+0.46,+0.74] | +0.66 [+0.47,+0.76] |
| `feature_measure_moment_delta` | +0.79 [+0.64,+0.88] | +0.73 [+0.56,+0.84] | +0.71 [+0.52,+0.82] | +0.67 [+0.49,+0.78] | +0.65 [+0.44,+0.78] |
| `perturbation_capacity_delta` | +0.70 [+0.56,+0.79] | **−0.40 [−0.58,−0.18]** | +0.19 [−0.04,+0.42] | +0.24 [−0.02,+0.49] | +0.21 [−0.08,+0.48] |
| `perturbation_capacity_relu_active_delta` | +0.40 [+0.18,+0.58] | **−0.46 [−0.62,−0.25]** | −0.28 [−0.48,−0.05] | **−0.41 [−0.63,−0.15]** | **−0.48 [−0.68,−0.23]** |

gram 族全阶段 CI 下界 ≥ +0.44（处处显著为正）；capacity 在 e0 显著正、e5 显著负（CI 不跨零），relu_active 从 e5 起持续显著负。**「gram 通用、capacity 非通用」不是点估计的偶然，而是统计显著的。**

### 2. 跨协议一致性：loss 翻转，gram 稳健，capacity 中间

同一 64 架构在两个协议下的指标值 Spearman（>0.5 记 ROBUST，<0 记 FRAGILE）：

| 指标 | e23↔v3 一致性 | 归类 |
|---|---:|---|
| `adversarial_loss` | **−0.328** | FRAGILE（符号翻转） |
| `natural_loss` | **−0.312** | FRAGILE（符号翻转） |
| `logit_kl` | −0.102 | FRAGILE |
| `perturbation_capacity_delta` | +0.446 | intermediate |
| `perturbation_capacity_relu_active_delta` | +0.113 | intermediate |
| `feature_measure_mmd` | +0.573 | ROBUST |
| `fsp_delta` | +0.699 | ROBUST |
| `feature_measure_struct01_delta` | +0.671 | ROBUST |
| `feature_measure_moment_delta` | +0.707 | ROBUST |
| `feature_measure_flow_rel_delta` | +0.804 | ROBUST |
| `feature_delta` | +0.836 | ROBUST（最稳） |

**loss 的架构排名在两个协议下符号翻转**——它测的是绝对输出幅度，依赖归一化/网络实现，随机初始化时的「loss 信号」不是稳定的架构性质。gram 族（相对结构）一致性 0.67–0.84，几乎不受协议影响。

### 3. 容量变体全部是幅度量

E28 新增容量变体与 plain capacity 的 Spearman：

| 变体 | ~plain capacity | ~gram |
|---|---:|---:|
| `flow_rel` | +0.933 | +0.718 |
| `topk` | +0.926 | +0.722 |
| `channelwise` | +0.908 | +0.798 |
| `gini` | +0.908 | +0.720 |
| `multiscale` | +0.847 | +0.743 |
| `iqr` | +0.824 | +0.525 |
| `relu_active_multiscale` | +0.814 | +0.703 |
| `layerprofile` | **−0.031** | +0.273 |

除 `layerprofile`（结构量，但端点弱：NB201 e0 0.04 / RB20 0.28）外，其余容量变体与 plain capacity 高度相关（0.77–0.93）——它们都是**幅度量**，因此都被预测为 e5 翻转。**capacity 方向里不存在逃过「幅度不稳定」诅咒的变体。**

## 关键发现（统一机制）

> **E34 修正（重要）**：本节的「跨训练稳定性」结论是 **PGD-7 强对抗训练下**的观察。E34 用 PGD-1 弱对抗训练重跑后，capacity 反而稳定（+0.491）、gram 反而不稳定（+0.12～0.18），与 e23 几乎完全颠倒。因此「capacity 跨训练翻转」「gram 跨训练稳定」**都不是指标的固有属性，而是训练强度依赖的**。本报告唯一稳健、不依赖训练的结论是 **「跨协议一致性（尺度不变性）」**（下面第 1 点）。见 [e34_capacity_flip_training_strength.md](e34_capacity_flip_training_strength.md)。

1. **通用性 = 尺度不变性**。相对结构量（gram、flow_rel、feature_delta、local direction）对「归一化/网络实现」不敏感（跨协议一致性 0.7–0.84）——因为它们只测形状/方向，不测绝对值。这是「通用」的**稳健判据**（独立于训练协议）。

2. **幅度量不通用**。分两级：
   - **dispersion 幅度（capacity）**：跨协议中间（0.45）；在 PGD-7 强对抗训练下跨训练翻转（e5 → −0.454），但这是训练强度依赖的（E34 已修正）。它测的是特征离散的绝对幅度。
   - **输出幅度（loss/logit）**：跨协议翻转（−0.33），符号都不稳。它测的是绝对输出，依赖归一化。

3. **E32 的机制修正补全**：容量漂移的信息在「幅度」里（E32 证伪 signed），而「幅度」正是它不通用的根源。两者不矛盾——幅度在成熟模型上强（有信号），但在训练中不稳定、跨协议不可靠。

4. **adversarial_loss 不是通用指标**：它在 e23 协议下看似「稳定 + 强」（min(stab,RB20)=0.58），但跨协议符号翻转（−0.33）。这是 E33 最重要的纠错——不能把单协议下的 loss 通用性当结论。

5. **设计原则**：通用指标必须测**相对结构/方向**，不测绝对幅度。这是未来设计新指标的唯一可靠判据。

## 结论

- **gram 族（`fsp_delta`、`feature_measure_moment_delta`、`feature_measure_struct01_delta`）是「跨协议尺度不变」的通用指标**——这是 E33 的核心、稳健结论（机制：尺度不变的相对结构量）。
- **capacity 的「非通用」需限定表述**：它跨协议中间（0.45），且只在 PGD-7 强对抗训练下 e5 翻转（E34 已证明这是训练强度依赖，非固有属性）。
- **loss 方向（adversarial_loss）跨协议翻转**，不能作为通用指标（虽然它在成熟模型上最强）。
- 通用性与成熟强度的权衡：尺度不变性（通用）牺牲了绝对幅度（成熟模型区分度）。

## 输出文件

```text
scripts/analyze_e33_mechanism.py      （三项分析可复现：排名稳定性 / 跨协议一致性 / 容量变体归类）
```

> 预测检验结果（E34）：multiscale/topk/channelwise/gini/iqr 容量在 e5 的翻转预测，在 PGD-1 重训下**被推翻**——它们 e0→e5 均为正（0.13～0.71）。见 [e34_capacity_flip_training_strength.md](e34_capacity_flip_training_strength.md)。
