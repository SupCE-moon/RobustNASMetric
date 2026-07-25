# Feature Geometry, Fractal, and Graph-Topology Robust Metrics

Last updated: 2026-07-21

## 1. Motivation

顾险峰讲义第 4-5 页提出的问题是：两个曲面在欧氏空间中可以长得很像，但球面上的闭环都能收缩到一点，环面却存在不可收缩环，因此可以借助基本群而不是逐点坐标区分它们。第 54 页进一步给出图基本群：连通无向图 $G=(V,E)$ 的基本群是自由群，其秩为 $|E|-|V|+1$。

我们把这一思路迁移到卷积 feature map，但必须先说明边界：feature map 不是已知的闭曲面或三角网格，有限激活点云一般也不能唯一恢复连续流形的基本群。因此，本研究不直接声称计算了 feature manifold 的 genus 或 $\pi_1$，而是设计四个有限样本、可审计的几何/分形/图拓扑 discrepancy。

已有研究也支持考察激活空间的几何和拓扑：Gebhart 等使用持久同调分析 adversarial input 引起的激活结构变化；Wen 等观察到鲁棒训练倾向于使多层 feature-space 变化更平滑；近期工作还把 intrinsic dimension 与 adversarial vulnerability 联系起来。它们提供研究动机，但不等同于本文四个具体指标。

## 2. Feature Point Cloud

令第 $l$ 层 feature map 为

$$
A_l(x)\in\mathbb R^{C_l\times H_l\times W_l},
\qquad S_l=H_lW_l.
$$

空间位置 $s$ 上的通道向量记为

$$
a_{l,s}(x)\in\mathbb R^{C_l},
\qquad s=1,\ldots,S_l.
$$

只有 $S_l\ge 16$ 的 feature map 才进入几何指标；$1\times1$ 全局池化输出不是可估计维数或环结构的点云。所有合格层中均匀采样最多 $K=4$ 层。

为了比较不同宽度和激活尺度的模型，使用固定、与标签和 robust accuracy 无关的投影矩阵 $R_l\in\mathbb R^{C_l\times d}$。分别中心化 clean/adv 点云：

$$
\bar a_l(x)=\frac1{S_l}\sum_s a_{l,s}(x),
\qquad
\bar a_l(x_{\mathrm{adv}})=\frac1{S_l}\sum_s a_{l,s}(x_{\mathrm{adv}}).
$$

共同 RMS 尺度定义为

$$
\sigma_l^2
=
\frac{1}{2S_l}\sum_s
\left(
\left\|(a_{l,s}(x)-\bar a_l(x))R_l\right\|_2^2
+
\left\|(a_{l,s}(x_{\mathrm{adv}})-\bar a_l(x_{\mathrm{adv}}))R_l\right\|_2^2
\right).
$$

最终点云为

$$
Z_l(x)
=
\left\{
z_{l,s}(x)=
\frac{(a_{l,s}(x)-\bar a_l(x))R_l}{\max(\sigma_l,10^{-6})}
\right\}_{s=1}^{S_l},
$$

adv 点云 $Z_l(x_{\mathrm{adv}})$ 同理。分别中心化使指标关注 feature geometry 的形变，而不是整体通道平移；共同尺度使 clean/adv 保持同一单位。

## 3. Hausdorff Support Displacement

对于有限点集 $X,Y\subset\mathbb R^d$，对称 Hausdorff 距离为

$$
d_H(X,Y)
=
\max\left\{
\max_{u\in X}\min_{v\in Y}\|u-v\|_2,
\max_{v\in Y}\min_{u\in X}\|u-v\|_2
\right\}.
$$

定义

$$
D_l^{\mathrm{Haus}}(x,x_{\mathrm{adv}})
=d_H\!\left(Z_l(x),Z_l(x_{\mathrm{adv}})\right).
$$

`feature_hausdorff_delta (Ours)` 是样本和采样层上的均值，预注册方向为 lower-is-better。它忽略空间位置的一一对应，测量两个 feature support 的最坏集合偏移。对于 $L$-Lipschitz 映射 $T$，有 $d_H(T(X),T(Y))\le Ld_H(X,Y)$；这说明 support 稳定性可以约束后续集合映射的变化。但卷积层会混合邻域，不能把这条集合不等式直接当成整个网络的 robustness certificate。

## 4. Quantile Hausdorff Discrepancy

标准 Hausdorff 由单个最远离群点决定。令

$$
r_X(u)=\min_{v\in Y}\|u-v\|_2,
\qquad
r_Y(v)=\min_{u\in X}\|u-v\|_2.
$$

定义 95% 分位版本

$$
D_l^{\mathrm{QH95}}
=
\max\left\{
Q_{0.95}\bigl(\{r_X(u):u\in X\}\bigr),
Q_{0.95}\bigl(\{r_Y(v):v\in Y\}\bigr)
\right\}.
$$

`feature_quantile_hausdorff_delta (Ours)` 同样 lower-is-better。它是抗离群 discrepancy，但通常不满足严格三角不等式，因此文中不把它称为数学意义上的 metric。

## 5. Finite-Scale Box-Dimension Drift

真实 box-counting/Minkowski 维数定义为

$$
\dim_B(X)
=
\lim_{\varepsilon\to0}
\frac{\log N_X(\varepsilon)}{\log(1/\varepsilon)},
$$

其中 $N_X(\varepsilon)$ 是覆盖 $X$ 所需的 $\varepsilon$-box 数量。有限 feature point cloud 不存在可观测的 $\varepsilon\to0$ 极限，因此代码计算的是固定尺度估计量，而不是真实 Hausdorff 维数。

具体地，把点云固定投影到 3 维，在共同网格中使用

$$
\varepsilon_j=r_jL_l,
\qquad
(r_1,r_2,r_3,r_4)=(0.125,0.2,0.32,0.5),
$$

其中 $L_l$ 是 clean 投影点云的最大坐标跨度。对四组 $(\log(1/\varepsilon_j),\log N(\varepsilon_j))$ 做最小二乘斜率，得到 $\widehat d_{B,l}$。定义

$$
D_l^{\mathrm{Box}}
=
\left|
\widehat d_{B,l}(x_{\mathrm{adv}})-\widehat d_{B,l}(x)
\right|.
$$

`feature_box_dimension_delta (Ours)` lower-is-better。它测量有限尺度复杂度漂移，但依赖投影、网格和尺度区间，不能写成精确分形维数定理。

## 6. Multiscale Graph Fundamental-Rank Drift

从最多 32 个投影点构造多尺度邻域图。用 clean 点云两两距离的分位数

$$
q\in\{0.1,0.2,0.3\}
$$

产生共同阈值 $r_{l,q}$，clean/adv 两点距离不超过该阈值时连接边。为了让图基本群秩可统一比较，每张图并入自己的最小生成树，因此图保持连通。

对连通图 $G_{l,q}$，根据讲义第 54 页，

$$
\pi_1(G_{l,q})
\cong
F_{\beta_{1,l,q}},
\qquad
\beta_{1,l,q}=|E_{l,q}|-|V_{l,q}|+1,
$$

其中 $F_k$ 是 $k$ 个生成元的自由群。定义

$$
D_l^{\pi_1}
=
\frac13\sum_q
\frac{
\left|\beta_{1,l,q}(x_{\mathrm{adv}})-\beta_{1,l,q}(x)\right|
}{
\max\{\beta_{1,l,q}(x),1\}
}.
$$

`feature_graph_fundamental_rank_delta (Ours)` lower-is-better。它严格计算的是邻域图的自由群秩变化，不是未知连续 feature manifold 的基本群，也不是 Vietoris-Rips 复形的 persistent $H_1$。

## 7. Research Interpretation

四个指标对应不同信息层级：Hausdorff 测 support 最坏偏移，QH95 测主体 support 偏移，box drift 测尺度复杂度，graph fundamental rank 测离散环生成元数量变化。它们与 MMD/SWD 的主要区别是：MMD/SWD 比较经验分布，本文指标比较 support geometry、finite-scale complexity 或邻域图拓扑。

E24 的结果不支持把四项中的任何一项直接升级为论文主 metric。图基本群秩在 NAS-RobBench-201 epoch-0 有较强信号，但跨到 DARTS 后明显减弱；Hausdorff 项在训练后还出现方向反转。当前最合理的定位是：图拓扑项作为 training-free 消融，box drift 作为较稳定但较弱的复杂度消融。

## 8. Sources

- David Gu, `Fundamental Group and Covering Space`, pages 4-5 and 54, 2026-07-19.
- Gebhart, Schrater, and Hylton, [Characterizing the Shape of Activation Space in Deep Neural Networks](https://arxiv.org/abs/1901.09496).
- Gebhart and Schrater, [Adversary Detection in Neural Networks via Persistent Homology](https://arxiv.org/abs/1711.10056).
- Wen, Li, and Jia, [Towards Understanding the Regularization of Adversarial Robustness on Neural Networks](https://proceedings.mlr.press/v119/wen20c.html).
- Haldar, Xing, and Song, [Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability](https://proceedings.mlr.press/v238/haldar24a.html).
- Altinisik et al., [Explaining the role of Intrinsic Dimensionality in Adversarial Training](https://proceedings.mlr.press/v267/altinisik25a.html).
