# 静态谱浓度与 Jacobian 谱容量：零成本鲁棒代理的新指标族

Last updated: 2026-09-10

## 1. 动机与问题边界

RobustNAS 需要一个在随机初始化权重下、无需训练即可排序架构鲁棒潜力的代理指标。本项目现有 45 个指标中，零成本（epoch 0）场景下几乎全部是**扰动响应 delta**：喂入 clean 与 adversarial 两张图，测量中间特征分布 / 容量 / 几何结构"变化了多少"。

在随机初始化权重下，这个设计的根本缺陷是：白盒攻击作用于随机网络时，其"对抗方向"本质上是**任意方向**——随机网络的梯度没有对抗语义。因此所有 delta 型指标在 init 处测量的是"特征沿任意随机方向的局部敏感度"，信号弱、且随搜索空间漂移。这解释了已观察到的现象：MMD（NB201 e0 = 0.794）在 DARTS 上迁移失败；E25 方向B 的谱 delta 在 RB20 上排名 32/39/42。

本文提出三个**未在项目中实现过**的新指标族，它们不再测量扰动响应 delta，而是测量随机初始化架构的**静态谱结构**与**解析 Jacobian 谱**：

- **Family A：静态谱浓度**——随机初始化 clean 特征协方差的谱形状（有效秩 / participation ratio / 谱衰减指数），无任何扰动。
- **Family B：Jacobian 谱容量**——输入到 logit 的 Jacobian 奇异值谱（谱范数 / 有效秩 / 衰减指数），即解析的局部 Lipschitz。
- **Family C：方向性容量漂移**——保留符号的容量相对变化（消融）。

## 2. 精确定义

### 2.1 记号

第 $l$ 层特征图为 $F_l(x)\in\mathbb R^{C_l\times S_l}$（$S_l=H_lW_l$）。中心化投影算子

$$
P_l = I_{S_l} - \frac{1}{S_l}\mathbf 1_{S_l}\mathbf 1_{S_l}^\top
$$

（见 `perturbation_capacity_motivation_theory.md` §3.2，已证其为到零均值子空间的正交投影）。中心化特征记为

$$
A_l(x) = F_l(x)P_l \in \mathbb R^{C_l\times S_l}.
$$

其特征协方差矩阵

$$
\Sigma_l(x) = \frac{1}{S_l} A_l A_l^\top \in \mathbb R^{C_l\times C_l}.
$$

设 $\Sigma_l$ 的特征值（升序）为 $\lambda_1\le\cdots\le\lambda_{C_l}$，则 $A_l$ 的奇异值为 $\sigma_i=\sqrt{\lambda_i}$。这组 $\{\sigma_i\}$ 就是**谱容量谱**：它完整描述表征离散度在多少方向、以何种集中度分布。

### 2.2 Family A：静态谱浓度

只对 clean 特征计算，不做任何扰动。三个泛函：

$$
\text{eRank}_l = \exp\!\left(-\sum_{i} p_i\log p_i\right),
\qquad
p_i = \frac{\lambda_i}{\sum_j \lambda_j},
$$

$$
\text{PR}_l = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}
= \frac{\operatorname{tr}(\Sigma_l)^2}{\operatorname{tr}(\Sigma_l^2)},
$$

谱衰减指数 $\alpha_l$：对排序后的 $\lambda_{(1)}\ge\cdots\ge\lambda_{(C)}$ 拟合

$$
\log\lambda_{(i)} = -\alpha_l\log i + \text{const},
$$

即 $\lambda_{(i)}\propto i^{-\alpha_l}$。模型级指标为逐层、逐样本平均。字段名：`feature_spectral_erank_static`、`feature_spectral_participation_ratio_static`、`feature_spectral_decay_exponent_static`。

### 2.3 Family B：Jacobian 谱容量

设 $f:\mathbb R^d\to\mathbb R^K$ 为完整网络（输入到 logit），在随机初始化权重、clean 输入 $x$ 处的 Jacobian

$$
J(x)=\frac{\partial f(x)}{\partial x}\in\mathbb R^{K\times d}.
$$

奇异值 $s_1\ge\cdots\ge s_{\min(K,d)}$。三个泛函：

$$
\|J(x)\|_2 = s_1 \quad\text{(局部 Lipschitz 常数)},
$$

$$
\text{eRank}(J) = \exp\!\left(-\sum_i q_i\log q_i\right),\qquad q_i=\frac{s_i}{\sum_j s_j},
$$

以及 $J$ 奇异值谱的衰减指数 $\alpha_J$（同 §2.2 的拟合）。字段名：`jacobian_spectral_norm`、`jacobian_effective_rank`、`jacobian_spectral_decay`。

实现注：不显式构造 $J$（$K\times d$，CIFAR 下 $d=3072$）。对每个类别 $k$ 求输入梯度 $g_k=\partial z_k/\partial x$，构成 Gram 矩阵 $G=[\langle g_k,g_l\rangle]_{k,l}=JJ^\top\in\mathbb R^{K\times K}$，其特征值即 $\{s_i^2\}$。$K=10$，代价为每样本 $K$ 次反向。

### 2.4 Family C：方向性容量漂移（消融）

沿用现有 MAD 容量 $\mathcal C_l(x)=\frac{1}{S_l}\|F_lP_l\|_{1,1}$，但**不取绝对值**，保留符号并拆分：

$$
D_l^{\text{sgn}} = \frac{\mathcal C_l(x_{\text{adv}})-\mathcal C_l(x)}{\mathcal C_l(x)+\varepsilon},
$$

再拆为容量坍缩（$D<0$）与容量爆炸（$D>0$）两个分量分别聚合。字段名：`perturbation_capacity_signed_delta`。现有指标 `perturbation_capacity_delta` 是 $|D_l^{\text{sgn}}|$ 的平均，本消融检验"符号里是否藏着被绝对值抹掉的鲁棒信号"。

## 3. 数学论证

### 3.1 谱容量是 MAD 容量的一阶→二阶推广

现有容量泛函（`perturbation_capacity_motivation_theory.md` §3.2.4）已证

$$
\mathcal C_l(x)=\frac{1}{S_l}\|A_l\|_{1,1},
$$

即元素级 $L_1$ 范数——**一阶**离散度。它在两个方向上丢失信息：

1. **重排列不变**：MAD 只依赖 $|a_{c,s}-m_c|$ 的集合，对空间位置的任意置换不变。两个架构可有相同 MAD、但特征协方差结构完全不同。
2. **通道解耦**：$\sum_c\kappa_c$ 逐通道相加，通道间相关性完全不可见。

谱容量用 $A_l$ 的奇异值谱 $\{\sigma_i\}$ 替换一阶标量，二者由熟知不等式联系：

$$
\|A_l\|_F^2 = \sum_i \sigma_i^2 = \operatorname{tr}(\Sigma_l) = \sum_c \operatorname{Var}_s(a_{c,\cdot}),
$$

即总能量是逐通道**方差**之和——MAD 用一阶绝对矩、总能量用二阶矩。而

$$
\frac{1}{\sqrt{S_l}}\|A_l\|_{1,1} \le \|A_l\|_F
$$

（元素级 $L_1$ 与 Frobenius 范数的不等式）给出 MAD 容量被谱总能量上界约束。谱泛函同时保留方向（特征向量）与相关性（协方差非对角元），修复上述两个丢失。

### 3.2 静态谱浓度为何预测鲁棒潜力

两条独立文献线支持"快谱衰减 ⟹ 鲁棒表示"：

- **$1/n$ 表示**：神经科学发现对噪声鲁棒的生物表征，其协方差谱按幂律 $\lambda_i\propto i^{-1}$ 衰减；NeurIPS 2020 的 "On 1/n neural representation and robustness" 在人工网络中将中间层激活谱正则化到该衰减，提升了对抗鲁棒性。
- **良性过拟合的害处**：线性模型下，**慢衰减 / 重尾谱**（有效秩高、大量小特征值）会让标准风险趋零的同时让对抗风险**爆炸**；快衰减把能量集中在少数信号方向，对抗者可利用的方向更少。

因此，一个在**随机初始化**时就具有快谱衰减（低 eRank / 低 PR / 大 $\alpha$）的架构，其表示几何天然更接近鲁棒表征，更可能在对抗训练后达到高鲁棒精度。这是关于计算图几何的**静态断言**，不依赖具体训练轨迹或攻击方向，故有望跨搜索空间迁移。

### 3.3 Jacobian 谱给出鲁棒性的因果上界

由一阶展开，对任意扰动 $\delta$：

$$
\|f(x+\delta)-f(x)\|_2 = \|J(x)\delta + o(\|\delta\|_2)\|_2 \le \|J(x)\|_2\,\|\delta\|_2 + o(\|\delta\|_2).
$$

$\|J(x)\|_2=s_1$ 是局部 Lipschitz 常数，**直接上界**对抗扰动在 logit 空间的放大。这是因果约束而非相关：鲁棒架构必须在所有方向上有小的 Lipschitz。与项目现有粗糙标量 `local_logit_lipschitz = \|\logit_\text{step}\|/\|\text{input}_\text{step}\|`（只沿单一样本扰动方向）不同，$s_1$ 是**最坏方向**的精确谱范数；与 RACL 的 $\prod_\ell\|W_\ell\|_2$（逐层权谱范数乘积的粗略上界，忽略残差/跳跃连接）不同，$J$ 是**端到端经验** Jacobian。完整谱 $\{s_i\}$ 还刻画敏感性的各向异性（eRank、衰减），这是 `jacov`（只取 Jacobian 协方差的 trace，平均掉所有方向）与 Feng et al. ICLR 2025（取 input-Hessian 的 $\lambda_{\max}$）都没有提供的。

## 4. 与已失败指标（E25 方向B）的区分

| | E25 方向B（已失败） | 本文 |
|---|---|---|
| 测量对象 | 谱泛函的**扰动 delta**：$\left\|\Phi(\text{adv})-\Phi(\text{clean})\right\|/\Phi(\text{clean})$ | 谱泛函的**静态值**（Family A）；输入→logit Jacobian 谱（Family B） |
| 是否依赖攻击 | 是（白盒攻击随机权重 = 任意方向） | 否（Family A 无扰动；Family B 只算一阶导数） |
| init 处信号 | 沿任意方向的局部敏感度 = 噪声 | 计算图几何的确定性属性 |
| 字段 | `feature_spectral_efficiency_delta` 等 | `feature_spectral_erank_static` 等 |

Family A 与方向B 的区别**不是泛函不同，而是"静态 vs delta"**：方向B 问"谱在扰动下变多少"，本文问"谱本身是什么形状"。后者在 init 处才是良定义的稳定信号。

## 5. 方向假设（预注册，实验前固定）

| 字段 | 假设方向 | 理由 |
|---|---|---|
| `feature_spectral_erank_static` | lower | 谱集中（低有效秩）⟹ 鲁棒潜力 |
| `feature_spectral_participation_ratio_static` | lower | 同上 |
| `feature_spectral_decay_exponent_static` | higher | 快衰减（大 $\alpha$）⟹ 鲁棒 |
| `jacobian_spectral_norm` | lower | 小 Lipschitz ⟹ 鲁棒 |
| `jacobian_effective_rank` | lower | 敏感性方向少 ⟹ 鲁棒 |
| `jacobian_spectral_decay` | higher | 敏感性谱快衰减 ⟹ 鲁棒 |
| `perturbation_capacity_signed_delta` | 待定 | 消融，方向由数据决定 |

## 6. 验证方案

1. **三空间全本地**：RobustBench-20（成熟白盒）、NB201 64 架构 e0、DARTS 20 架构 e0（零成本），复用 E23/E24 的鲁棒精度标签。
2. **主统计**：方向调整后的 Spearman + 2000 次 bootstrap 95% CI（不跨零）。
3. **结构去混杂**：控制参数量 / 操作组成后的 partial Spearman，检验非平凡性。
4. **全对比**：与 45 个现有指标 + `jacov`、Feng 代理变体同榜排名。
5. **阶段鲁棒性**（若可行）：NB201 上 e0/e5/e10/e25/e50 方向一致性。
6. **计算成本**：记录每指标耗时，确认仍属零成本量级。

## 7. 参考文献

1. Cissé et al. *Parseval Networks*. ICLR 2017. (Lipschitz 上界)
2. Tsuzuku et al. *Lipschitz-Margin Training*. ICLR 2018.
3. Roy & Vetterli. *The effective rank: A measure of effective dimensionality*. EUSIPCO 2007.
4. Nassar et al. *On 1/n neural representation and robustness*. NeurIPS 2020. <https://proceedings.nips.cc/paper_files/paper/2020/file/44bf89b63173d40fb39f9842e308b3f9-Review.html>
5. Tsigler & Bartlett. *Benign overfitting in linear regression*. 2020. (慢谱衰减害处)
6. Feng et al. *Zero-cost Proxy for Adversarial Robustness Evaluation*. ICLR 2025. <https://openreview.net/pdf?id=zHf7hOfeer>
7. Lukasik et al. *An Evaluation of Zero-Cost Proxies — from Neural Architecture Performance Prediction to Model Robustness*. IJCV 2024. <https://ar5iv.labs.arxiv.org/html/2307.09365>
8. Dong et al. *RACL: Robust Architecture Curriculum Learning*. (架构 Lipschitz 基线)
9. Mellor et al. *Neural Architecture Search without Training* (NASWOT). ICML 2021.
