# 当前 Robust Metric 总表（可读公式版）

日期：2026-07-03

## 实验设置

- 数据集：CIFAR10 test split
- 模型：10 个 RobustBench CIFAR10 / Linf 模型
- 对抗样本：white-box，每个模型分别生成自己的 `x_adv`
- 攻击：PGD-lite，3 steps
- 样本数：512
- 结果指标：与 `adv_acc` 的 Spearman 相关系数
- `All`：全部 10 个模型
- `Robust-only`：去掉自然训练的 `Standard` 模型

## 符号说明

| 符号 | 含义 |
| --- | --- |
| `x` | clean input |
| `x_adv` | adversarial input |
| `y` | true label |
| `f(x)` | model logits |
| `p(x)` | `Softmax(f(x))` |
| `F_l(x)` | 第 `l` 个中间层特征图 |
| `GAP(.)` | global average pooling |
| `Norm(.)` | L2 normalization |
| `CE(.)` | cross entropy loss |
| `eps` | 防止除零的小常数 |

## 总表

| 类型 | Metric 名称 | 论文具体名称 | 数学原理 | 具体计算公式（不依赖 LaTeX 渲染） | All | Robust-only | 当前结论 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 我们提出 | `feature_delta` | 本研究提出；受 `Feature Denoising for Improving Adversarial Robustness` 启发 | 鲁棒模型在对抗扰动下，中间特征表示应更稳定。 | 对每层 `l`：先算 `z_l(x) = Norm(GAP(F_l(x)))`，再算 `d_l = L2Distance(z_l(x_adv), z_l(x))`，最后 `feature_delta = Average(d_l over layers and samples)`。 | -0.6970 | -0.5833 | 有效，但目前弱于 `fsp_delta` 和 `input_loss_delta`。 |
| 我们提出 | `perturbation_capacity_delta` | 本研究提出；受 activation perturbation / CAS 类分析启发 | 鲁棒模型不应显著放大由扰动引起的激活容量变化。 | 对每层 `l`：`C_l(x) = Sum over channels c of Mean over h,w of Abs(F_l,c,h,w(x) - Mean_hw(F_l,c,*,*(x)))`；然后 `metric = Average(Abs(C_l(x_adv)-C_l(x)) / (Abs(C_l(x))+eps))`。 | -0.3939 | -0.1667 | 当前较弱，暂不适合作为独立主指标。 |
| 输出稳定性 | `logit_kl` | `Theoretically Principled Trade-off between Robustness and Accuracy` / TRADES | clean 与 adversarial 的输出概率分布应尽量一致。 | `p = Softmax(f(x))`，`q = Softmax(f(x_adv))`；`logit_kl = Sum over class k of p_k * Log(p_k / q_k)`。 | -0.7455 | -0.6500 | 简单且较强，是可靠输出空间 baseline。 |
| ARNAS 目标 proxy | `natural_loss` | `Towards Accurate and Robust Architectures via Neural Architecture Search` | ARNAS 同时关注自然精度目标。 | `natural_loss = CE(f(x), y) = -Log(Softmax(f(x))_y)`。 | 0.3939 | 0.1667 | 不是鲁棒性指标本身，只是 ARNAS 多目标中的自然精度项。 |
| ARNAS 目标 proxy | `adversarial_loss` | `Towards Accurate and Robust Architectures via Neural Architecture Search` | ARNAS 同时关注对抗鲁棒目标。 | `adversarial_loss = CE(f(x_adv), y) = -Log(Softmax(f(x_adv))_y)`。 | -0.5273 | -0.3500 | 有方向性，但弱于 loss 增量。 |
| 输入损失曲面 | `input_loss_delta` | `AdvRush: Searching for Adversarially Robust Neural Architectures`；也与 `Adversarial Weight Perturbation Helps Robust Generalization` 相关 | 输入 loss landscape 越平滑，对抗扰动造成的 loss 增量越小。 | `input_loss_delta = CE(f(x_adv), y) - CE(f(x), y)`。也就是 `adversarial_loss - natural_loss`。 | -0.8182 | -0.7500 | 很强的 scalar baseline，比单独 adversarial loss 更有信息量。 |
| Jacobian / 输入梯度 | `input_grad_l2` | `Robust Learning with Jacobian Regularization`；DSRNA 的 Jacobian norm bound 思路 | 输入梯度越小，模型对输入局部扰动越不敏感。 | 对每个样本 `i`：`g_i = Gradient of CE(f(x_i), y_i) with respect to x_i`；`input_grad_l2 = Average(L2Norm(g_i))`。 | -0.7818 | -0.7000 | 当前表现强，是重要 baseline。 |
| Lipschitz proxy | `local_logit_lipschitz` | `Adversarially Robust Neural Architectures`；Parseval / Lipschitz robustness 系列 | 局部 Lipschitz 越小，输入扰动导致的 logit 变化越小。 | 对每个样本：`ratio_i = L2Norm(f(x_adv_i)-f(x_i)) / (L2Norm(x_adv_i-x_i)+eps)`；`local_logit_lipschitz = Average(ratio_i)`。 | -0.6727 | -0.5500 | 只是 fixed-model proxy，不等同于该论文的 architecture-level Lipschitz 正则。 |
| Certified-bound proxy | `margin_input_grad_ratio` | `DSRNA: Differentiable Search of Robust Neural Architectures` | margin 越大、输入梯度越小，局部 certified radius 越可能更大。 | `margin_i = f_y(x_i) - Max over k != y of f_k(x_i)`；`ratio_i = Max(margin_i, 0) / (L2Norm(Gradient_x CE(f(x_i),y_i))+eps)`；`metric = Average(ratio_i)`。 | -0.7818 | -0.7000 | 理论上应 higher-is-better，但当前 white-box 实验中受 clean confidence 影响，实测方向异常，需谨慎。 |
| FSP / feature-flow | `fsp_delta` | `When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks`；FSP 最早来自 `A Gift from Knowledge Distillation` | 鲁棒架构应保持 feature-flow / Gram 结构在扰动下稳定。 | 对每层 `l`：将 `F_l(x)` 展平为 channel by spatial 矩阵 `A_l(x)`；计算 `G_l(x) = Norm(A_l(x)) * Norm(A_l(x))^T / spatial_size`；然后 `fsp_delta = Average(FrobeniusNorm(G_l(x_adv)-G_l(x)) / (FrobeniusNorm(G_l(x))+eps))`。 | -0.8909 | -0.8500 | 当前最强 baseline。注意：现在是同层 Gram 稳定性版本，不是 RobNet 原文的完整跨层 FSP loss。 |
| 方法级必比 | Architecture-level Lipschitz regularization | `Adversarially Robust Neural Architectures` | 通过 architecture parameters 降低 Lipschitz 相关扰动放大，从架构层面提高鲁棒性。 | 原文方法不是简单后验 metric。可抽象为：`Search architecture A to minimize validation loss + lambda * LipschitzBound(A)`。当前实验只用 `local_logit_lipschitz` 作为 fixed-model proxy。 | -0.6727 proxy | -0.5500 proxy | 后续 NAS/search 阶段必须做方法级对比。当前未完整复现。 |
| 方法级必比 | Accurate / robust multi-objective NAS | `Towards Accurate and Robust Architectures via Neural Architecture Search` | 同时优化自然性能和对抗鲁棒性能，而不是只用单一 scalar robustness score。 | 原文是搜索方法。当前 proxy 为：`natural_loss = CE(f(x),y)`，`adversarial_loss = CE(f(x_adv),y)`，以及 `input_loss_delta = adversarial_loss - natural_loss`。 | 见上 | 见上 | 后续 NAS/search 阶段必须对比 ARNAS 方法。当前只做 fixed-model metric proxy。 |

## 当前 Robust-only 排名

按 `Robust-only Spearman` 的绝对值排序：

1. `fsp_delta`: -0.8500
2. `input_loss_delta`: -0.7500
3. `input_grad_l2`: -0.7000
4. `margin_input_grad_ratio`: -0.7000
5. `logit_kl`: -0.6500
6. `feature_delta`: -0.5833
7. `local_logit_lipschitz`: -0.5500
8. `adversarial_loss`: -0.3500
9. `perturbation_capacity_delta`: -0.1667
10. `natural_loss`: 0.1667

## 目前最重要的判断

`fsp_delta` 是当前最强 baseline。`input_loss_delta` 是最重要的 loss-landscape baseline。我们自己的 `feature_delta` 有信号，但还需要进一步增强，例如加入 layer weighting、feature-flow 结构，或与 loss-landscape 信息组合。
