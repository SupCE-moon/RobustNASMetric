# 早期指标能否预测训练后的鲁棒准确率（第一版证据）

- 早期指标：`nb201_e0`（训练第 0 个 epoch，即随机初始化后的前向）
- 目标：NAS-RobBench-201 的 `autoattack`（三 seed 均值，训练后的数据）
- 控制变量：`val_acc`（同一份数据的训练后 clean 准确率）
- 架构数：**64**；重采样 2000 次，种子 20261112

## 为什么必须控制 clean 准确率

在这份数据里，训练后 clean 准确率与 AutoAttack 准确率的秩相关是 **+0.982**。也就是说，任何能预测 clean 准确率的早期指标，都会看起来像在预测鲁棒性。下面的「偏相关」列把 clean 准确率的影响去掉之后再看。

## 判定分布

| 判定 | 数量 | 含义 |
| --- | --- | --- |
| `direct` | 8 | 控制 clean 准确率后仍有相关 |
| `via_clean` | 4 | 与 clean 准确率高度共线，控制后基本消失 |
| `suppressed` | 2 | 控制后符号翻转（抑制效应） |
| `weak` | 55 | 两边都弱 |
| `uncovered` | 1 | 可用样本不足 |

## 偏相关最强的 12 个指标

| 指标 | n | ρ(target) | ρ(clean) | 偏相关 | 95% 区间 | 置换 p | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `feature_measure_cov_delta` | 64 | +0.795 | +0.766 | +0.357 | [+0.014, +0.566] | 0.0060 | direct |
| `feature_measure_spectrum_emd_delta` | 64 | +0.785 | +0.754 | +0.353 | [+0.066, +0.541] | 0.0055 | direct |
| `feature_measure_spectrum_hellinger_delta` | 64 | +0.790 | +0.762 | +0.346 | [+0.041, +0.552] | 0.0060 | direct |
| `perturbation_capacity_profile_delta` | 64 | +0.698 | +0.662 | +0.342 | [-0.008, +0.548] | 0.0080 | direct |
| `perturbation_capacity_flow_rel_delta` | 64 | +0.688 | +0.652 | +0.333 | [-0.040, +0.546] | 0.0115 | direct |
| `perturbation_capacity_topk_delta` | 64 | +0.702 | +0.667 | +0.333 | [-0.040, +0.555] | 0.0105 | direct |
| `feature_measure_spectrum_js_delta` | 64 | +0.819 | +0.796 | +0.323 | [+0.013, +0.548] | 0.0095 | direct |
| `feature_measure_channel_gram_centered_rel_delta` | 64 | +0.651 | +0.617 | +0.304 | [-0.018, +0.512] | 0.0170 | direct |
| `perturbation_capacity_multiscale_delta` | 64 | +0.802 | +0.783 | +0.285 | [-0.068, +0.524] | 0.0250 | weak |
| `perturbation_capacity_delta` | 64 | +0.552 | +0.516 | +0.281 | [-0.097, +0.519] | 0.0255 | weak |
| `feature_measure_robust_hybrid_delta` | 64 | +0.504 | +0.468 | +0.268 | [+0.000, +0.457] | 0.0270 | weak |
| `perturbation_capacity_relu_active_multiscale_delta` | 64 | +0.792 | +0.774 | +0.266 | [-0.077, +0.500] | 0.0405 | weak |

## 怎么读

- `ρ(target)` 是原始相关：早期指标越好、训练后越鲁棒吗。
- `ρ(clean)` 是同一指标与训练后 clean 准确率的相关。它高，说明这个指标主要在测「这个架构好不好训」，而不是「这个架构抗不抗攻击」。
- `偏相关` 把 clean 准确率去掉之后剩下的相关。**这一列才是 RQ2 的证据**。
- `via_clean` 不是「指标没用」，而是「它预测鲁棒性的方式，是预测 clean 准确率」。要谈抗攻击能力，得另找证据。

## 这份报告不能说明什么

- **跨协议**：目标是官方固定训练协议下的结果，早期指标是本仓库对**随机初始化**网络做的一次前向；两者不是一个训练流程的产物，所以不能说「训练早期就能预测」，只能说「架构层面的早期指标与训练后结果相关」。
- **单次初始化**：e0 指标只有一个随机种子，初始化噪声没有估计。
- **n = 64**：Spearman 的 95% 区间约 ±0.25；置换 p 只检验「不相关」这一零假设。
- **不是因果**：控制 clean 准确率只能排除一条混杂路径，参数量、可训练性等仍可能同时影响两者。
- **目标不是 AutoAttack 的最强设置**：RobBench 记录的是他们的评估配置，与发布榜里的 AutoAttack 数字不可直接比较。
