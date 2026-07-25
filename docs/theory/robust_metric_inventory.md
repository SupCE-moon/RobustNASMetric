# Robust Metric Inventory

Last updated: 2026-07-25

> 本文件是公式与代码实现的逐字段审计附件。项目的理论主入口、45-metric 分类和当前研究判断统一维护在 [理论管理.md](理论管理.md)；实验结果统一维护在 [实验管理.md](../experiments/实验管理.md)。不要在本文件单独更新实验结论。

本文档用于人工核对当前代码中每个 robust metric 的数学定义。对应实现文件为：

```text
E:\projects\robust_metric_lab\src\robust_metric_lab\metrics.py
```

注意：本文档描述的是“当前代码实际计算了什么”，不是论文最终推荐的全部指标。论文主指标应从这些实现字段中筛选。

## 1. Common Notation

给定分类模型：

$$
f_\theta:\mathcal{X}\to\mathbb{R}^{K}.
$$

一个 batch 记为：

$$
\{(x_b,y_b)\}_{b=1}^{B},
\qquad
\{x_{b,\mathrm{adv}}\}_{b=1}^{B}.
$$

代码通过 forward hook 记录所有 4D feature map。第 $l$ 个被选中的 feature map 写为：

$$
F_l(x_b)\in\mathbb{R}^{C_l\times H_l\times W_l}.
$$

令：

$$
S_l=H_lW_l.
$$

把空间位置 $s$ 上的通道向量称为一个 feature atom：

$$
a_{b,l,s}(x)\in\mathbb{R}^{C_l},
\qquad
s=1,\dots,S_l.
$$

这里 $a_{b,l,s}(x)$ 不是函数，而是 feature map 在样本 $b$、层 $l$、空间位置 $s$ 处取出的一个 $C_l$ 维向量。

其 L2-normalized 版本为：

$$
u_{b,l,s}(x)
=
\frac{a_{b,l,s}(x)}
{\|a_{b,l,s}(x)\|_2+\varepsilon}.
$$

代码最多选择 `max_layers=4` 个 hook feature map，记为：

$$
\mathcal{L}=\{l_1,l_2,\dots,l_M\},
\qquad M\le 4.
$$

相邻 layer pair 记为：

$$
\mathcal{P}
=
\{(l_i,l_{i+1})\}_{i=1}^{M-1}.
$$

除非特别说明，最终 CSV 中的 metric 是先对 batch 求均值，再对 selected layers 或 selected layer pairs 求均值。

## 2. Output-Space Metrics

### `clean_acc`

$$
\mathrm{clean\_acc}
=
\frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}
\left[
\arg\max_k f_\theta(x_b)_k=y_b
\right].
$$

### `adv_acc`

$$
\mathrm{adv\_acc}
=
\frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}
\left[
\arg\max_k f_\theta(x_{b,\mathrm{adv}})_k=y_b
\right].
$$

### `natural_loss`

$$
\mathrm{natural\_loss}
=
\mathrm{CE}(f_\theta(x),y).
$$

### `adversarial_loss`

$$
\mathrm{adversarial\_loss}
=
\mathrm{CE}(f_\theta(x_{\mathrm{adv}}),y).
$$

### `input_loss_delta`

$$
\mathrm{input\_loss\_delta}
=
\mathrm{CE}(f_\theta(x_{\mathrm{adv}}),y)
-
\mathrm{CE}(f_\theta(x),y).
$$

### `logit_kl`

令：

$$
p_b=\mathrm{softmax}(f_\theta(x_b)),
\qquad
q_b=\mathrm{softmax}(f_\theta(x_{b,\mathrm{adv}})).
$$

代码计算：

$$
\mathrm{logit\_kl}
=
\frac{1}{B}\sum_{b=1}^{B}
\sum_{k=1}^{K}
p_{b,k}\log\frac{p_{b,k}}{q_{b,k}}.
$$

## 3. Input-Space Sensitivity Metrics

### `input_grad_l2`

先计算 clean loss 对输入的梯度：

$$
g_b
=
\nabla_{x_b}\mathrm{CE}(f_\theta(x_b),y_b).
$$

代码中由于 PyTorch cross entropy 默认对 batch 取均值，因此实现里乘了 $B$ 来恢复 per-sample scale。最终：

$$
\mathrm{input\_grad\_l2}
=
\frac{1}{B}\sum_{b=1}^{B}\|g_b\|_2.
$$

### `local_logit_lipschitz`

这是有限差分形式的局部 logit sensitivity，不是全局 Lipschitz 常数：

$$
\mathrm{local\_logit\_lipschitz}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\|f_\theta(x_{b,\mathrm{adv}})-f_\theta(x_b)\|_2
}{
\|x_{b,\mathrm{adv}}-x_b\|_2+\varepsilon
}.
$$

### `margin_input_grad_ratio`

clean logit margin 为：

$$
m_b
=
f_\theta(x_b)_{y_b}
-
\max_{k\ne y_b} f_\theta(x_b)_k.
$$

metric 定义为：

$$
\mathrm{margin\_input\_grad\_ratio}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{\max(m_b,0)}
{\|g_b\|_2+\varepsilon}.
$$

## 4. First-Order Feature Metrics

### `feature_delta`

这是早期 GAP 版本。先对 feature map 做 global average pooling：

$$
z_{b,l}(x)
=
\frac{
\mathrm{GAP}(F_l(x_b))
}{
\|\mathrm{GAP}(F_l(x_b))\|_2+\varepsilon
}.
$$

层 $l$ 上的 displacement：

$$
D_l^{\mathrm{GAP}}
=
\frac{1}{B}\sum_{b=1}^{B}
\|z_{b,l}(x_{\mathrm{adv}})-z_{b,l}(x)\|_2.
$$

最终：

$$
\mathrm{feature\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{GAP}}.
$$

说明：该指标会压缩空间结构，不建议作为论文主指标。

### `feature_measure_local_delta`

逐空间位置比较 normalized feature atom：

$$
D_l^{\mathrm{local}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{1}{S_l}\sum_{s=1}^{S_l}
\|u_{b,l,s}(x_{\mathrm{adv}})-u_{b,l,s}(x)\|_2.
$$

最终：

$$
\mathrm{feature\_measure\_local\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{local}}.
$$

### `feature_measure_weighted_delta`

代码用 clean feature atom 的 L2 energy 作为空间权重：

$$
w_{b,l,s}^{\mathrm{energy}}(x)
=
\frac{\|a_{b,l,s}(x)\|_2}
{\sum_{t=1}^{S_l}\|a_{b,l,t}(x)\|_2+\varepsilon}.
$$

层 $l$ 上：

$$
D_l^{\mathrm{wlocal}}
=
\frac{1}{B}\sum_{b=1}^{B}
\sum_{s=1}^{S_l}
w_{b,l,s}^{\mathrm{energy}}(x)
\|u_{b,l,s}(x_{\mathrm{adv}})-u_{b,l,s}(x)\|_2.
$$

最终：

$$
\mathrm{feature\_measure\_weighted\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{wlocal}}.
$$

## 5. Distribution-Style Feature Metrics

这些指标把 layer $l$ 的 spatial atoms 看作经验分布的样本。代码会先用 clean feature 的均值和标准差对 clean/adv atoms 做标准化。

### `feature_measure_swd`

令 $r=1,\dots,R$ 为固定随机投影方向，代码中 $R=16$。对每个投影方向，分别将 clean/adv atoms 投影并排序：

$$
\alpha_{b,l,r}^{\uparrow}(x),
\qquad
\alpha_{b,l,r}^{\uparrow}(x_{\mathrm{adv}}).
$$

代码计算近似 sliced Wasserstein distance：

$$
D_l^{\mathrm{SWD}}
=
\frac{1}{B}\sum_{b=1}^{B}
\left(
\frac{1}{RS_l}
\sum_{r=1}^{R}\sum_{s=1}^{S_l}
\left[
\alpha_{b,l,r,s}^{\uparrow}(x_{\mathrm{adv}})
-
\alpha_{b,l,r,s}^{\uparrow}(x)
\right]^2
\right)^{1/2}.
$$

### `feature_measure_mmd`

代码用 random Fourier features 近似 kernel mean embedding。令 $\phi(\cdot)$ 为 RFF 映射，则：

$$
D_l^{\mathrm{MMD}}
=
\frac{1}{B}\sum_{b=1}^{B}
\left\|
\frac{1}{S_l}\sum_{s=1}^{S_l}\phi(a_{b,l,s}(x_{\mathrm{adv}}))
-
\frac{1}{S_l}\sum_{s=1}^{S_l}\phi(a_{b,l,s}(x))
\right\|_2.
$$

### `feature_measure_cov_delta`

令 clean/adv 的 centered atom covariance 为：

$$
\Sigma_{b,l}(x)
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
(a_{b,l,s}(x)-\bar a_{b,l}(x))
(a_{b,l,s}(x)-\bar a_{b,l}(x))^\top.
$$

代码计算相对 covariance displacement：

$$
D_l^{\mathrm{cov}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\|\Sigma_{b,l}(x_{\mathrm{adv}})-\Sigma_{b,l}(x)\|_F
}{
\|\Sigma_{b,l}(x)\|_F+\varepsilon
}.
$$

## 6. Same-Layer Gram Metrics

### `fsp_delta`

这个字段是历史遗留命名：它不是 RobNet 原始 FSP，而是 same-layer channel Gram 的相对变化。

令：

$$
\widehat F_l(x_b)
=
\mathrm{Normalize}_{\mathrm{spatial}}(F_l(x_b)),
$$

即对每个 channel 的空间向量做 L2-normalization。same-layer channel Gram 为：

$$
G_{b,l}^{\mathrm{chan}}(x)
=
\frac{1}{S_l}
\widehat F_l(x_b)\widehat F_l(x_b)^\top
\in\mathbb{R}^{C_l\times C_l}.
$$

代码计算：

$$
D_l^{\mathrm{chan,rel}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\|G_{b,l}^{\mathrm{chan}}(x_{\mathrm{adv}})
-
G_{b,l}^{\mathrm{chan}}(x)\|_F
}{
\|G_{b,l}^{\mathrm{chan}}(x)\|_F+\varepsilon
}.
$$

最终：

$$
\mathrm{fsp\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{chan,rel}}.
$$

### `feature_measure_spatial_gram_delta`

这个指标比较同一层内部不同空间位置之间的相似性结构。先构造 normalized atoms：

$$
u_{b,l,s}(x)
=
\frac{a_{b,l,s}(x)}
{\|a_{b,l,s}(x)\|_2+\varepsilon}.
$$

空间 Gram 矩阵为：

$$
G_{b,l}^{\mathrm{spatial}}(x)_{s,t}
=
u_{b,l,s}(x)^\top u_{b,l,t}(x).
$$

代码中的非相对版本为：

$$
D_l^{\mathrm{spatial}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\|G_{b,l}^{\mathrm{spatial}}(x_{\mathrm{adv}})
-
G_{b,l}^{\mathrm{spatial}}(x)\|_F
}{
S_l
}.
$$

最终：

$$
\mathrm{feature\_measure\_spatial\_gram\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{spatial}}.
$$

### `feature_measure_spatial_gram_rel_delta`

相对版本为：

$$
D_l^{\mathrm{spatial,rel}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\|G_{b,l}^{\mathrm{spatial}}(x_{\mathrm{adv}})
-
G_{b,l}^{\mathrm{spatial}}(x)\|_F
}{
\|G_{b,l}^{\mathrm{spatial}}(x)\|_F+\varepsilon
}.
$$

最终：

$$
\mathrm{feature\_measure\_spatial\_gram\_rel\_delta}
=
\frac{1}{M}\sum_{l\in\mathcal{L}}D_l^{\mathrm{spatial,rel}}.
$$

## 7. Cross-Layer Flow / Strict FSP Metrics

对相邻 selected layer pair $(i,j)\in\mathcal{P}$，代码先把两个 feature map 的空间尺寸通过 adaptive average pooling 对齐。

RobNet/FSP-style cross-layer matrix 为：

$$
G_{b,i,j}^{\mathrm{flow}}(x)
=
\frac{1}{S_{i,j}}
F_i(x_b)F_j(x_b)^\top
\in
\mathbb{R}^{C_i\times C_j}.
$$

这里 $F_i(x_b)$ 和 $F_j(x_b)$ 已经被 flatten 成 channel-by-spatial 的矩阵，$S_{i,j}$ 是对齐后的空间位置数。

### `fsp_strict_delta`

这是最接近 RobNet 原始 FSP loss 的当前实现。它没有除以 clean FSP norm，而是计算 squared Frobenius displacement：

$$
D_{i,j}^{\mathrm{strictFSP}}
=
\frac{1}{B}\sum_{b=1}^{B}
\left\|
G_{b,i,j}^{\mathrm{flow}}(x_{\mathrm{adv}})
-
G_{b,i,j}^{\mathrm{flow}}(x)
\right\|_F^2.
$$

最终：

$$
\mathrm{fsp\_strict\_delta}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
D_{i,j}^{\mathrm{strictFSP}}.
$$

### `feature_measure_flow_delta`

这是 cross-layer flow displacement 的 RMS 版本：

$$
D_{i,j}^{\mathrm{flow}}
=
\frac{1}{B}\sum_{b=1}^{B}
\sqrt{
\frac{1}{C_iC_j}
\left\|
G_{b,i,j}^{\mathrm{flow}}(x_{\mathrm{adv}})
-
G_{b,i,j}^{\mathrm{flow}}(x)
\right\|_F^2
}.
$$

最终：

$$
\mathrm{feature\_measure\_flow\_delta}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
D_{i,j}^{\mathrm{flow}}.
$$

### `feature_measure_flow_rel_delta`

相对版本为：

$$
D_{i,j}^{\mathrm{flow,rel}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\left\|
G_{b,i,j}^{\mathrm{flow}}(x_{\mathrm{adv}})
-
G_{b,i,j}^{\mathrm{flow}}(x)
\right\|_F
}{
\left\|G_{b,i,j}^{\mathrm{flow}}(x)\right\|_F+\varepsilon
}.
$$

## 8. ReLU-Active Metrics

ReLU-active 指标的动机是：对 ReLU 网络而言，只有正激活位置真正向后续层传递有效信息。代码没有直接用二值 $0/1$ 空间权重，而是对每个空间位置统计正激活 channel 的比例：

$$
\rho_{b,l,s}(x)
=
\frac{1}{C_l}
\sum_{c=1}^{C_l}
\mathbf{1}\left[a_{b,l,s,c}(x)>0\right].
$$

归一化为权重：

$$
w_{b,l,s}^{\mathrm{ReLU}}(x)
=
\frac{\rho_{b,l,s}(x)}
{\sum_{t=1}^{S_l}\rho_{b,l,t}(x)+\varepsilon}.
$$

### `feature_measure_relu_active_delta`

$$
D_l^{\mathrm{ReLU-local}}
=
\frac{1}{B}\sum_{b=1}^{B}
\sum_{s=1}^{S_l}
w_{b,l,s}^{\mathrm{ReLU}}(x)
\|u_{b,l,s}(x_{\mathrm{adv}})-u_{b,l,s}(x)\|_2.
$$

### `feature_measure_relu_active_spatial_delta`

令：

$$
\widetilde u_{b,l,s}(x)
=
\sqrt{w_{b,l,s}^{\mathrm{ReLU}}(x)}\,u_{b,l,s}(x).
$$

ReLU-weighted spatial Gram 为：

$$
G_{b,l}^{\mathrm{ReLU-spatial}}(x)_{s,t}
=
\widetilde u_{b,l,s}(x)^\top
\widetilde u_{b,l,t}(x).
$$

代码计算：

$$
D_l^{\mathrm{ReLU-spatial}}
=
\frac{1}{B}\sum_{b=1}^{B}
\left\|
G_{b,l}^{\mathrm{ReLU-spatial}}(x_{\mathrm{adv}})
-
G_{b,l}^{\mathrm{ReLU-spatial}}(x)
\right\|_F.
$$

### `feature_measure_relu_active_spatial_rel_delta`

$$
D_l^{\mathrm{ReLU-spatial,rel}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\left\|
G_{b,l}^{\mathrm{ReLU-spatial}}(x_{\mathrm{adv}})
-
G_{b,l}^{\mathrm{ReLU-spatial}}(x)
\right\|_F
}{
\left\|G_{b,l}^{\mathrm{ReLU-spatial}}(x)\right\|_F+\varepsilon
}.
$$

## 9. Combined Ours Metrics

### `feature_measure_robust_delta`

该指标把 cross-layer flow 变化与 energy-weighted first-order local 变化相加。对 pair $(i,j)$：

$$
D_{i,j}^{(1)}
=
\frac{1}{2}
\left(
D_i^{\mathrm{wlocal}}
+
D_j^{\mathrm{wlocal}}
\right).
$$

$$
D_{i,j}^{\mathrm{ROBUST}}
=
D_{i,j}^{\mathrm{flow}}
+
D_{i,j}^{(1)}.
$$

最终：

$$
\mathrm{feature\_measure\_robust\_delta}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
D_{i,j}^{\mathrm{ROBUST}}.
$$

### `feature_measure_robust_rel_delta`

$$
D_{i,j}^{\mathrm{ROBUST,rel}}
=
D_{i,j}^{\mathrm{flow,rel}}
+
D_{i,j}^{(1)}.
$$

### `feature_measure_robust_spatial_delta`

同层 spatial Gram 二阶项加上 weighted local 一阶项：

$$
D_l^{\mathrm{ROBUST-spatial}}
=
D_l^{\mathrm{spatial}}
+
D_l^{\mathrm{wlocal}}.
$$

### `feature_measure_robust_spatial_rel_delta`

$$
D_l^{\mathrm{ROBUST-spatial,rel}}
=
D_l^{\mathrm{spatial,rel}}
+
D_l^{\mathrm{wlocal}}.
$$

### `feature_measure_robust_hybrid_delta`

代码中的 hybrid 是全局平均后的加权组合：

$$
\mathrm{feature\_measure\_robust\_hybrid\_delta}
=
\frac{1}{2}\mathrm{feature\_measure\_flow\_delta}
+
\frac{1}{2}\mathrm{feature\_measure\_spatial\_gram\_delta}
+
\mathrm{feature\_measure\_weighted\_delta}.
$$

### `feature_measure_robust_hybrid_rel_delta`

$$
\mathrm{feature\_measure\_robust\_hybrid\_rel\_delta}
=
\frac{1}{2}\mathrm{feature\_measure\_flow\_rel\_delta}
+
\frac{1}{2}\mathrm{feature\_measure\_spatial\_gram\_rel\_delta}
+
\mathrm{feature\_measure\_weighted\_delta}.
$$

### `feature_measure_relu_active_hybrid_delta`

$$
\mathrm{feature\_measure\_relu\_active\_hybrid\_delta}
=
\frac{1}{2}\mathrm{feature\_measure\_flow\_delta}
+
\frac{1}{2}\mathrm{feature\_measure\_relu\_active\_spatial\_delta}
+
\mathrm{feature\_measure\_relu\_active\_delta}.
$$

### `feature_measure_relu_active_hybrid_rel_delta`

$$
\mathrm{feature\_measure\_relu\_active\_hybrid\_rel\_delta}
=
\frac{1}{2}\mathrm{feature\_measure\_flow\_rel\_delta}
+
\frac{1}{2}\mathrm{feature\_measure\_relu\_active\_spatial\_rel\_delta}
+
\mathrm{feature\_measure\_relu\_active\_delta}.
$$

## 10. Historical Composite Metrics

### `feature_measure_moment_delta`

$$
D_l^{\mathrm{moment}}
=
\sqrt{
\left(D_l^{\mathrm{GAP}}\right)^2
+
\left(D_l^{\mathrm{chan,rel}}\right)^2
}.
$$

### `feature_measure_struct_delta`

$$
D_l^{\mathrm{struct}}
=
\sqrt{
\left(D_l^{\mathrm{wlocal}}\right)^2
+
\left(D_l^{\mathrm{chan,rel}}\right)^2
}.
$$

### `feature_measure_struct01_delta`

$$
D_l^{\mathrm{struct01}}
=
\sqrt{
0.1\left(D_l^{\mathrm{wlocal}}\right)^2
+
\left(D_l^{\mathrm{chan,rel}}\right)^2
}.
$$

### `perturbation_capacity_delta`

令 channel activation capacity 为：

$$
C_{b,l}(x)
=
\sum_{c=1}^{C_l}
\frac{1}{S_l}
\sum_{s=1}^{S_l}
\left|
a_{b,l,s,c}(x)
-
\frac{1}{S_l}\sum_{t=1}^{S_l}a_{b,l,t,c}(x)
\right|.
$$

代码计算：

$$
D_l^{\mathrm{capacity}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
|C_{b,l}(x_{\mathrm{adv}})-C_{b,l}(x)|
}{
|C_{b,l}(x)|+\varepsilon
}.
$$

### `perturbation_capacity_relu_active_delta`

这是 `perturbation_capacity_delta` 的 clean-gated ReLU-active 消融。首先定义逐元素二值门控：

$$
M_{b,l,s,c}(x)
=
\mathbf{1}\!\left[a_{b,l,s,c}(x)>0\right],
\qquad
N_{b,l,c}^{+}(x)
=
\sum_{s=1}^{S_l}M_{b,l,s,c}(x).
$$

对 $z\in\{x,x_{\mathrm{adv}}\}$，在同一个 clean-active support 上定义：

$$
\bar a_{b,l,c}^{+}(z\mid x)
=
\frac{\sum_s M_{b,l,s,c}(x)a_{b,l,s,c}(z)}
{\max(N_{b,l,c}^{+}(x),1)},
$$

$$
C_{b,l}^{+}(z\mid x)
=
\sum_{c=1}^{C_l}
\frac{
\sum_s M_{b,l,s,c}(x)
\left|a_{b,l,s,c}(z)-\bar a_{b,l,c}^{+}(z\mid x)\right|
}{\max(N_{b,l,c}^{+}(x),1)}.
$$

最终：

$$
D_l^{\mathrm{capacity\text{-}ReLU}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\left|C_{b,l}^{+}(x_{\mathrm{adv}}\mid x)-C_{b,l}^{+}(x\mid x)\right|
}{
\left|C_{b,l}^{+}(x\mid x)\right|+\varepsilon
}.
$$

这里固定使用 clean mask，使 clean/adv 在相同 support 上配对比较。它不会显式计算由 $x$ 到 $x_{\mathrm{adv}}$ 的 ReLU 开/关状态转换；独立 adversarial mask 属于另一个待验证指标。

### `perturbation_capacity_above_mean_delta`

该指标把绝对零阈值替换为每个 clean channel 的空间均值：

$$
\mu_{b,l,c}(x)
=
\frac{1}{S_l}\sum_s a_{b,l,s,c}(x),
\qquad
M_{b,l,s,c}^{>\mu}(x)
=
\mathbf{1}\!\left[a_{b,l,s,c}(x)>\mu_{b,l,c}(x)\right].
$$

令 $N_{b,l,c}^{>\mu}(x)=\sum_s M_{b,l,s,c}^{>\mu}(x)$。对 $z\in\{x,x_{\mathrm{adv}}\}$：

$$
\bar a_{b,l,c}^{>\mu}(z\mid x)
=
\frac{\sum_s M_{b,l,s,c}^{>\mu}(x)a_{b,l,s,c}(z)}
{\max(N_{b,l,c}^{>\mu}(x),1)},
$$

$$
C_{b,l}^{>\mu}(z\mid x)
=
\sum_c
\frac{
\sum_s M_{b,l,s,c}^{>\mu}(x)
\left|a_{b,l,s,c}(z)-\bar a_{b,l,c}^{>\mu}(z\mid x)\right|
}{\max(N_{b,l,c}^{>\mu}(x),1)}.
$$

最终计算 clean/adv gated capacity 的相对变化并在样本、层上取平均。该门控对逐 channel 加性平移不变，比 $a>0$ 更不依赖层偏置。

## 11. Geometry, Fractal, and Graph-Topology Metrics

The following four E24 fields are defined rigorously in `fractal_topology_feature_metric_design.md`. They use feature maps with at least 16 spatial atoms, at most four sampled layers, deterministic projection, separate centering, and a shared clean/adv RMS scale.

| Field | Definition | Direction | Important limitation |
|---|---|---|---|
| `feature_hausdorff_delta (Ours)` | symmetric Hausdorff distance between normalized clean/adv feature-support point clouds | lower | sensitive to the largest support outlier |
| `feature_quantile_hausdorff_delta (Ours)` | maximum directed 95%-quantile nearest-neighbor distance | lower | robust discrepancy, not necessarily a formal metric |
| `feature_box_dimension_delta (Ours)` | absolute difference of four-scale grid-counting slopes in a fixed 3D projection | lower | finite-scale estimator, not exact Hausdorff or box dimension |
| `feature_graph_fundamental_rank_delta (Ours)` | relative drift of $|E|-|V|+1$ over three MST-connected proximity graphs | lower | graph fundamental-group proxy, not the unknown manifold's $\pi_1$ |

For a connected proximity graph $G_{l,q}$, the fourth field uses

$$
\operatorname{rank}\pi_1(G_{l,q})
=
|E_{l,q}|-|V_{l,q}|+1.
$$

These fields are experimental metric candidates rather than robustness certificates. E24 finds a strong NAS-RobBench-201 epoch-0 result for graph rank, but weak DARTS transfer and sign reversal after training.

## 12. Naming Caveats

| 字段 | 需要注意 |
|---|---|
| `fsp_strict_delta` | 当前最接近 RobNet 原始 FSP loss，没有 clean-norm 分母。 |
| `fsp_delta` | 历史命名，实际是 same-layer channel Gram relative delta，不是 RobNet 原始 FSP。 |
| `feature_delta` | 使用 GAP，会弱化空间结构，只适合作为早期 baseline。 |
| `feature_measure_relu_active_*` | 使用正激活 channel 比例作为空间权重，不是简单地对整个 atom 做二值 $0/1$。 |
| `perturbation_capacity_relu_active_delta` | 使用逐 channel、逐位置的 clean 二值门控，并按每个 channel 的有效位置数归一化。 |
| `perturbation_capacity_above_mean_delta` | 使用 clean channel 空间均值作为相对阈值；clean/adv 共用门控，并在选中 support 内重新中心化。 |
| `local_logit_lipschitz` | 是 local finite-difference proxy，不是全局 Lipschitz bound。 |

### Published Robust-NAS Fields Used in Cross-Space Experiments

The following fields are emitted beside `MetricResult` by the cross-space runners. NAS-Bench-201 and DARTS use explicit searched-cell paths. RobustBench uses the separate source field `racl_fixed_network_log_upper_proxy`, which is canonicalized to `racl_arch_lipschitz_log_upper` only during correlation analysis.

#### `racl_arch_lipschitz_log_upper`

For every discrete NAS-Bench-201 `InferCell`, the selected operation and connection coefficients are one. The architecture-dependent part of Dong et al. [5], Eq. (10), is stored in log form:

$$
M_{\mathrm{RACL-upper}}
=
\sum_{k,j}
\log\left(
\max\left\{
\sum_{i<j}\lambda_{o_{ij}},\varepsilon
\right\}
\right).
$$

Identity, zero, and average-pooling constants are $1$, $0$, and $S^{-1/2}$. A convolution constant is estimated by 20-step power iteration on its reshaped weight matrix. Lower is better. This is a fixed-discrete-architecture counterpart, not the complete RACL confidence constraint.

#### `racl_input_gradient_lower`

This explicitly names the lower-bound surrogate used in Dong et al. [5], Eq. (16):

$$
M_{\mathrm{RACL-grad}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left\|\nabla_{x_b}\ell(f(x_b),y_b)\right\|_2.
$$

It is numerically identical to `input_grad_l2`; the alias is emitted only in NAS-RobBench experiments so the paper mapping can be audited directly.

#### `racl_zero_node_sums`

This is an audit field, not a robustness metric. It counts intermediate nodes whose selected incoming operation constants sum to zero. Such disconnected nodes can make a Lipschitz upper bound look artificially small while producing a useless constant or low-accuracy network.

### TRNAS fields

The following fields reproduce the two components and Eq. (6) composite from `TRNAS: A Training-Free Robust Neural Architecture Search` [11]. They are implemented in `src/robust_metric_lab/trnas.py` and emitted in all three E23 spaces.

#### `trnas_activation_psi [11,12]`

Concatenate all ReLU outputs for a batch of $B$ samples into:

$$
A_\theta(x)
\in
\mathbb{R}^{B\times V},
$$

where $V$ is the number of recorded ReLU activation units, including spatial positions. Binarize the activations:

$$
P_{b,v}
=
\mathbf 1[A_{b,v}>0].
$$

The sample-wise activation pattern of unit $v$ is the column vector $P_{:,v}\in\{0,1\}^{B}$. Following the official SWAP implementation [12], the unregularized activation capability is the number of distinct column patterns:

$$
\Psi_{\mathcal N,\theta}
=
\left|
\operatorname{unique}
\left\{P_{:,v}:v=1,\ldots,V\right\}
\right|.
$$

Higher is better in TRNAS. This quantity is strongly affected by the number of ReLU units, so `trnas_num_relu_units` must be retained as a structural control.

#### `trnas_consistency_phi [11]`

First generate $x_{\mathrm{adv}}$ with FGSM. The paper states that a disturbed network is obtained after a gradient update but does not publish code or the update magnitude. The auditable implementation uses one loss-ascent step:

$$
\theta'
=
\theta
+
\eta\nabla_\theta
\operatorname{CE}(f_\theta(x),y).
$$

The complete 512-architecture run uses $\eta=10^{-6}$ because it is the largest tested small step with a 100% finite-value rate. For every Conv2d or Linear layer $i$, flatten its feature output and direct parameter tensors into $v_i,v_i',\theta_i,\theta_i'$. Then compute:

$$
V_i
=
1+
\frac{v_i^\top v_i'}{\|v_i\|_2\|v_i'\|_2},
\qquad
\Theta_i
=
1+
\frac{\theta_i^\top\theta_i'}{\|\theta_i\|_2\|\theta_i'\|_2}.
$$

The TRNAS consistency term is:

$$
\Phi_{\mathcal N,\theta}
=
\sum_{i=1}^{I}V_i\Theta_i.
$$

Higher is better. Because the paper uses a sum, $\Phi$ scales with the number $I$ of learnable layers. `trnas_num_consistency_layers` is therefore a required control rather than a robustness metric.

#### `trnas_r_score [11]`

The composite proxy follows TRNAS Eq. (6):

$$
\operatorname{RScore}(\mathcal N)
=
\Psi_{\mathcal N,\theta}^{1/2}
\Phi_{\mathcal N,\theta}.
$$

Higher is better. E21 shows that the raw score ranks NAS-RobBench-201 robust accuracy strongly, but much of this signal is structural size information. The diagnostic fields `trnas_activation_density`, `trnas_consistency_mean`, and `trnas_r_score_normalized` are analysis-only quantities, not metrics claimed by the TRNAS paper.

## 13. Code Mapping

| 指标组 | 代码位置 |
|---|---|
| feature hook 与 layer 选择 | `ActivationRecorder`, `_anchor_indices` |
| GAP feature delta | `_pooled_normalized` |
| local / weighted local | `_spatial_atoms`, `_feature_measure_local_delta`, `_feature_measure_weighted_delta` |
| SWD / MMD / covariance | `_feature_measure_swd`, `_feature_measure_mmd`, `_feature_measure_cov_delta` |
| Hausdorff / box / graph topology | `_feature_geometry_stats`, `_box_counting_dimension`, `_graph_fundamental_rank_delta`, `compute_feature_geometry_metrics` |
| spatial Gram | `_feature_measure_spatial_gram_stats` |
| ReLU-active | `_relu_active_spatial_weights`, `_feature_measure_relu_active_stats` |
| perturbation capacity / gated capacity | `_perturbation_capacity`, `_perturbation_capacity_on_support`, `_perturbation_capacity_relu_active`, `_perturbation_capacity_above_mean` |
| strict FSP / flow | `_fsp_matrix`, `_fsp_strict_stats` |
| output/input metrics | `compute_feature_metrics` bottom section |
| NAS-only RACL upper bound | `racl_architecture_lipschitz_stats` in `src/robust_metric_lab/nasrobbench201.py` |
| NAS-only RACL gradient alias | `append_metric_row` in `scripts/compute_nb201_metric_subset.py` |
| TRNAS $\Psi$, $\Phi$, and R-Score | `compute_trnas_metrics` in `src/robust_metric_lab/trnas.py` |
| NAS-only TRNAS runner | `scripts/compute_nb201_trnas_metrics.py` |

## 14. Reference Numbering

[1] Dong et al., `When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks`.

[2] Yim et al., `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning`.

[3] Zhang et al., `Theoretically Principled Trade-off between Robustness and Accuracy`.

[4] Jakubovitz and Giryes, `Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization`.

[5] Dong et al., `Adversarially Robust Neural Architectures`.

[6] Cisse et al., `Parseval Networks: Improving Robustness to Adversarial Examples`.

[7] DSRNA, `Differentiable Search of Robust Neural Architectures`.

[8] AdvRush, `Searching for Adversarially Robust Neural Architectures`.

[9] Wu et al., `Adversarial Weight Perturbation Helps Robust Generalization`.

[10] Ou, Feng, and Sun, `Towards Accurate and Robust Architectures via Neural Architecture Search`.

[11] Yang et al., `TRNAS: A Training-Free Robust Neural Architecture Search`, ICCV 2025.

[12] Peng et al., `SWAP-NAS: Sample-Wise Activation Patterns for Ultra-fast NAS`, ICLR 2024.

[13] Ha, Kim, and Hwang, `Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations`, NeurIPS 2023.
