# Perturbation Capacity Delta：研究动机、理论推导与论文叙事

Last updated: 2026-07-16

## 1. 结论边界

`perturbation_capacity_delta` 当前是在 **CIFAR-10、20 个 RobustBench 模型、三源 shared-adv 协议**下表现最好的已测试指标，其 Spearman 系数为 $-0.565$。这说明：指标越小，模型的官方 robust accuracy 总体越高。

这个结果支持的准确表述是：

> Relative perturbation-capacity drift is currently the strongest tested cross-architecture ranking proxy under the three-source shared-adversarial protocol.

它不等价于“该指标在任何数据集、攻击和模型集合上都必然最好”，也不构成鲁棒性证书。特别地，在 white-box 协议下，`adversarial_loss` 的 Spearman 为 $-0.805$，优于 `perturbation_capacity_delta` 的 $-0.626$。因此，我们真正要解释的是：**为什么 capacity drift 更适合在共享扰动下进行跨架构排序，而不是为什么它在所有条件下都胜过输出指标。**

## 2. RobustNAS 中真正需要什么指标

RobustNAS 不是只判断一个模型是否被某个样本攻击成功，而是要在大量异构架构之间建立尽可能稳定的鲁棒性次序。一个合适的 proxy 至少应满足：

1. 对中间表征受到的扰动敏感；
2. 不过度依赖 logit 的尺度、温度和置信度校准；
3. 能跨网络宽度、激活尺度和分类头进行比较；
4. 计算成本低，能够在多个层上聚合；
5. 在不同攻击源下保持相对稳定，而不是只拟合某一个 source model 的决策边界。

已有研究观察到，对抗扰动会在网络中形成并传播 feature noise，直接抑制中间特征噪声能够提高鲁棒性。RobNets 也表明，中间层特征结构可以作为鲁棒架构的有效指示量。因此，本文不只观察攻击传播到分类头后的最终后果，而是测量攻击对网络内部表征能力造成的相对改变。

## 3. 从经验测度到特征容量

### 3.1 通道经验测度

设第 $l$ 层特征图为

$$
F_l(x)\in\mathbb{R}^{C_l\times S_l},
$$

其中 $C_l$ 是通道数，$S_l=H_lW_l$ 是空间位置数。第 $c$ 个通道在位置 $s$ 的标量激活记为 $a_{l,c,s}(x)$。

将一个通道的全部空间激活看成标量轴上的经验概率测度：

$$
\mu_{l,c}^{x}
=
\frac{1}{S_l}\sum_{s=1}^{S_l}\delta_{a_{l,c,s}(x)}.
$$

这里 $\delta_t$ 是集中在数值 $t$ 上的 Dirac 测度。这个写法只是把有限集合
$\{a_{l,c,1},\ldots,a_{l,c,S_l}\}$ 表示成一个经验分布，并不是把激活 $a$ 当作函数。

该经验测度的均值为

$$
m_{l,c}(x)
=
\int t\,d\mu_{l,c}^{x}(t)
=
\frac{1}{S_l}\sum_{s=1}^{S_l}a_{l,c,s}(x).
$$

### 3.2 一阶中心绝对矩

定义通道容量为经验分布的一阶中心绝对矩，也就是 mean absolute deviation：

$$
\kappa(\mu_{l,c}^{x})
=
\int |t-m_{l,c}(x)|\,d\mu_{l,c}^{x}(t)
=
\frac{1}{S_l}\sum_{s=1}^{S_l}
|a_{l,c,s}(x)-m_{l,c}(x)|.
$$

它测量该通道围绕自身空间均值所呈现的**有效空间对比度或离散容量**。如果一个通道在所有位置输出同一个常数，其容量为零；如果不同空间位置形成明显响应差异，其容量增大。

这里的“capacity”是本文对该统计泛函的操作性命名，表示特征图承载空间差异响应的幅度；它不是 Shannon information、信道容量、VC dimension，也不能单独衡量特征的语义信息量。

第 $l$ 层总容量定义为

$$
\mathcal{C}_l(x)
=
\sum_{c=1}^{C_l}\kappa(\mu_{l,c}^{x}).
$$

下面不直接跳到矩阵公式，而是逐步把这个标量求和写成矩阵形式。

#### 3.2.1 符号与维度

为简化符号，暂时省略层下标 $l$ 和输入 $x$，记

$$
F=[a_{c,s}]_{\substack{1\le c\le C\\1\le s\le S}}
\in\mathbb{R}^{C\times S}.
$$

其中，$F$ 有 $C$ 行、$S$ 列；第 $c$ 行是第 $c$ 个通道；第 $s$ 列是展平后的第 $s$ 个空间位置；$a_{c,s}$ 是矩阵 $F$ 的第 $(c,s)$ 个元素。另外定义：

- $\mathbf 1_S=(1,\ldots,1)^\top\in\mathbb R^S$：长度为 $S$ 的全一列向量；
- $I_S\in\mathbb R^{S\times S}$：$S$ 阶单位矩阵。

所有通道的空间均值可以一次写成列向量

$$
m(F)
=
\frac{1}{S}F\mathbf 1_S
\in\mathbb R^C.
$$

检查它的第 $c$ 个元素：

$$
[m(F)]_c
=
\frac{1}{S}
\sum_{s=1}^{S}F_{c,s}(\mathbf 1_S)_s
=
\frac{1}{S}
\sum_{s=1}^{S}a_{c,s}
=m_c.
$$

为了从 $F$ 的每一行减去对应的通道均值，把 $m(F)$ 复制到 $S$ 个空间位置：

$$
m(F)\mathbf 1_S^\top
\in\mathbb R^{C\times S}.
$$

该矩阵第 $(c,s)$ 个元素恒为 $m_c$。因此中心化特征矩阵是

$$
\overline F
=
F-m(F)\mathbf 1_S^\top.
$$

#### 3.2.2 中心化矩阵的推导

把 $m(F)=S^{-1}F\mathbf 1_S$ 代入上式：

$$
\begin{aligned}
\overline F
&=F-\left(\frac{1}{S}F\mathbf 1_S\right)\mathbf 1_S^\top \\
&=F-\frac{1}{S}F\mathbf 1_S\mathbf 1_S^\top \\
&=F\left(I_S-\frac{1}{S}\mathbf 1_S\mathbf 1_S^\top\right).
\end{aligned}
$$

于是定义空间中心化矩阵

$$
P_S
=
I_S-\frac{1}{S}\mathbf 1_S\mathbf 1_S^\top
\in\mathbb R^{S\times S}.
$$

恢复层下标后：

$$
P_l
=
I_{S_l}-\frac{1}{S_l}
\mathbf 1_{S_l}\mathbf 1_{S_l}^\top.
$$

$P_l$ 不是网络参数，而是一个确定的线性算子。下面验证它的性质。为缩短公式，令 $S=S_l$、$\mathbf 1=\mathbf 1_{S_l}$。

首先，

$$
P_l^\top
=
\left(I_S-\frac{1}{S}\mathbf 1\mathbf 1^\top\right)^\top
=
I_S-\frac{1}{S}\mathbf 1\mathbf 1^\top
=P_l.
$$

其次，使用 $\mathbf 1^\top\mathbf 1=S$：

$$
\begin{aligned}
P_l^2
&=
\left(I_S-\frac{1}{S}\mathbf 1\mathbf 1^\top\right)
\left(I_S-\frac{1}{S}\mathbf 1\mathbf 1^\top\right) \\
&=
I_S-\frac{2}{S}\mathbf 1\mathbf 1^\top
+
\frac{1}{S^2}
\mathbf 1(\mathbf 1^\top\mathbf 1)\mathbf 1^\top \\
&=
I_S-\frac{2}{S}\mathbf 1\mathbf 1^\top
+
\frac{1}{S}\mathbf 1\mathbf 1^\top \\
&=
P_l.
\end{aligned}
$$

最后，

$$
P_l\mathbf 1
=
\mathbf 1-\frac{1}{S}\mathbf 1(\mathbf 1^\top\mathbf 1)
=
\mathbf 1-\mathbf 1
=0.
$$

所以

$$
P_l^\top=P_l,
\qquad
P_l^2=P_l,
\qquad
P_l\mathbf 1_{S_l}=0.
$$

对任意 $v\in\mathbb R^{S_l}$，都有

$$
\mathbf 1_{S_l}^\top(P_lv)
=(P_l\mathbf 1_{S_l})^\top v
=0.
$$

而当 $\mathbf 1_{S_l}^\top v=0$ 时，

$$
P_lv
=
v-\frac{1}{S_l}\mathbf 1_{S_l}
(\mathbf 1_{S_l}^\top v)
=v.
$$

这严格说明 $P_l$ 是到零均值子空间

$$
\mathcal V_0
:=
\{v\in\mathbb R^{S_l}:
\mathbf 1_{S_l}^\top v=0\}
$$

上的正交投影：它消除常数空间分量，同时保留所有已经为零均值的空间分量。

#### 3.2.3 验证 $F_lP_l$ 的每一个元素

矩阵乘积 $F_lP_l$ 仍属于 $\mathbb R^{C_l\times S_l}$。它的第 $(c,s)$ 个元素为

$$
\begin{aligned}
[F_lP_l]_{c,s}
&=
\sum_{t=1}^{S_l}
a_{l,c,t}
\left[
\mathbb 1\{t=s\}-\frac{1}{S_l}
\right] \\
&=
a_{l,c,s}
-
\frac{1}{S_l}
\sum_{t=1}^{S_l}a_{l,c,t} \\
&=
a_{l,c,s}-m_{l,c}.
\end{aligned}
$$

这里 $\mathbb 1\{t=s\}$ 是指示函数：当 $t=s$ 时取 $1$，否则取 $0$。不要把它与全一向量 $\mathbf 1_{S_l}$ 混淆。

#### 3.2.4 从元素求和得到矩阵范数

对任意矩阵 $A=[A_{c,s}]\in\mathbb R^{C\times S}$，本文定义元素级 $L_1$ 范数

$$
\|A\|_{1,1}
:=
\sum_{c=1}^{C}
\sum_{s=1}^{S}|A_{c,s}|.
$$

这里的 $\|A\|_{1,1}$ 是所有元素绝对值之和，不是矩阵的 induced $1$-norm。于是

$$
\begin{aligned}
\frac{1}{S_l}\|F_lP_l\|_{1,1}
&=
\frac{1}{S_l}
\sum_{c=1}^{C_l}
\sum_{s=1}^{S_l}
|[F_lP_l]_{c,s}| \\
&=
\sum_{c=1}^{C_l}
\frac{1}{S_l}
\sum_{s=1}^{S_l}
|a_{l,c,s}-m_{l,c}| \\
&=
\sum_{c=1}^{C_l}
\kappa(\mu_{l,c}^{x}) \\
&=
\mathcal C_l(x).
\end{aligned}
$$

因此

$$
\boxed{
\mathcal C_l(x)
=
\frac{1}{S_l}\|F_l(x)P_l\|_{1,1}
}
$$

不是新的定义，而是前面逐通道 mean absolute deviation 求和的完全等价写法。

#### 3.2.5 一个单通道例子

取 $C=1,S=3$，特征行为

$$
F=\begin{bmatrix}1&2&6\end{bmatrix}.
$$

空间均值是 $m=(1+2+6)/3=3$，所以中心化结果是

$$
FP
=
\begin{bmatrix}-2&-1&3\end{bmatrix}.
$$

容量为

$$
\mathcal C(F)
=
\frac{|-2|+|-1|+|3|}{3}
=2.
$$

这说明容量统计的是各空间位置离本通道均值的平均绝对距离。它没有使用 GAP 均值 $3$ 作为最终指标，而是使用均值完成中心化，因此不会让均值两侧的空间偏移相互抵消。

#### 3.2.6 这一步证明了什么，没有证明什么

上面的推导严格证明了两种写法完全等价：

$$
\sum_{c=1}^{C_l}
\frac{1}{S_l}
\sum_{s=1}^{S_l}
|a_{l,c,s}-m_{l,c}|
=
\frac{1}{S_l}
\|F_lP_l\|_{1,1}.
$$

它也说明了各设计成分的作用：

- 右乘 $P_l$：去除每个通道的空间常数分量，只保留空间对比；
- 取绝对值：防止高于均值和低于均值的偏差在求和时相互抵消；
- 除以 $S_l$：把空间求和变成每个位置的平均偏差，削弱不同空间分辨率带来的线性倍数；
- 对 $c$ 求和：得到整层所有通道的总离散量。

但是，这个代数等价式本身没有证明 MAD 是唯一或最优的鲁棒性统计量。选择 MAD 的方法学理由是：它保留一阶空间离散强度、计算复杂度为 $O(C_lS_l)$、比平方矩更少受极端激活支配，并具有后文给出的平移、尺度和 Wasserstein 稳定性质。它是否比 variance、Gram、MMD 或 output loss 更适合架构排序，最终仍需要统一协议下的经验验证。

### 3.3 对抗容量漂移

对干净样本 $x$ 及其对抗样本 $x_{\mathrm{adv}}$，定义层级相对容量漂移：

$$
D_l^{\mathrm{cap}}(x,x_{\mathrm{adv}})
=
\frac{
|\mathcal{C}_l(x_{\mathrm{adv}})-\mathcal{C}_l(x)|
}{
\mathcal{C}_l(x)+\varepsilon
}.
$$

最终模型指标是对样本和所选层求平均：

$$
D^{\mathrm{cap}}(f)
=
\frac{1}{|\mathcal{L}|}
\sum_{l\in\mathcal{L}}
\mathbb{E}_{x}
\left[D_l^{\mathrm{cap}}(x,x_{\mathrm{adv}})\right].
$$

指标越小，表示对抗扰动引起的中间表征容量变化越小，预期架构越鲁棒。

## 4. 可以严格证明的性质

### 4.1 平移不变性

令 $b=(b_1,\ldots,b_{C_l})^\top\in\mathbb R^{C_l}$。矩阵

$$
B=b\mathbf 1_{S_l}^\top
\in\mathbb R^{C_l\times S_l}
$$

表示给第 $c$ 个通道的所有空间位置都加上同一个常数 $b_c$。由于

$$
\mathbf 1_{S_l}^\top P_l
=
\mathbf 1_{S_l}^\top
-
\frac{1}{S_l}
\mathbf 1_{S_l}^\top
\mathbf 1_{S_l}
\mathbf 1_{S_l}^\top
=0,
$$

其中 $\mathbf 1_{S_l}^\top\mathbf 1_{S_l}=S_l$，所以

$$
\begin{aligned}
(F_l+B)P_l
&=F_lP_l+BP_l \\
&=F_lP_l+b(\mathbf 1_{S_l}^\top P_l) \\
&=F_lP_l.
\end{aligned}
$$

两边取元素级 $L_1$ 范数并除以 $S_l$，得到

$$
\mathcal C_l(F_l+b\mathbf 1_{S_l}^\top)
=
\mathcal C_l(F_l).
$$

这就是通道平移不变性。它说明通道整体基线改变不会改变该通道内部的空间离散程度。这里并没有证明任意 BatchNorm 变化都不影响指标；只有“对整个空间位置加同一个通道常数”这一类变化被严格消除。

### 4.2 正齐次性与相对尺度归一化

对任意标量 $\alpha\in\mathbb R$，由矩阵乘法的线性性和 $L_1$ 范数的绝对齐次性可得

$$
\begin{aligned}
\mathcal C_l(\alpha F_l)
&=
\frac{1}{S_l}\|(\alpha F_l)P_l\|_{1,1} \\
&=
\frac{1}{S_l}\|\alpha(F_lP_l)\|_{1,1} \\
&=
\frac{|\alpha|}{S_l}\|F_lP_l\|_{1,1} \\
&=
|\alpha|\mathcal C_l(F_l).
\end{aligned}
$$

现在令 $F=F_l(x)$、$F'=F_l(x_{\mathrm{adv}})$，并把 clean 和 adversarial feature 同时缩放为 $\alpha F$ 和 $\alpha F'$。实际代码中带 $\varepsilon$ 的相对漂移变为

$$
\begin{aligned}
D_{l,\varepsilon}^{\mathrm{cap}}(\alpha F,\alpha F')
&=
\frac{|\mathcal C_l(\alpha F')-\mathcal C_l(\alpha F)|}
{\mathcal C_l(\alpha F)+\varepsilon} \\
&=
\frac{|\alpha|\,|\mathcal C_l(F')-\mathcal C_l(F)|}
{|\alpha|\mathcal C_l(F)+\varepsilon} \\
&=
\frac{|\mathcal C_l(F')-\mathcal C_l(F)|}
{\mathcal C_l(F)+\varepsilon/|\alpha|}.
\end{aligned}
$$

因此：

- 当 $\varepsilon=0$ 且 $\alpha\ne0$ 时，指标严格尺度不变；
- 当 $\varepsilon>0$ 时，它不是严格尺度不变，只在 $\varepsilon\ll|\alpha|\mathcal C_l(F)$ 时近似不变；
- $\alpha=0$ 会把全部特征压成零，不能用于尺度不变性讨论。

相对分母也能削弱网络宽度带来的共同缩放，但这不是严格的“宽度完全不变性”：如果新增通道与原通道的扰动行为不同，指标仍会改变。

### 4.3 它是中心化特征位移的稳定见证量

仍令 $F=F_l(x)$、$F'=F_l(x_{\mathrm{adv}})$，并定义

$$
A=F'P_l,
\qquad
B=FP_l.
$$

任意范数都满足反三角不等式

$$
\big|\|A\|-\|B\|\big|
\le
\|A-B\|.
$$

它可由普通三角不等式证明：$\|A\|\le\|A-B\|+\|B\|$ 给出 $\|A\|-\|B\|\le\|A-B\|$；交换 $A,B$ 再得到 $\|B\|-\|A\|\le\|A-B\|$。两式合并即得反三角不等式。

将元素级 $L_1$ 范数和上面的 $A,B$ 代入：

$$
\begin{aligned}
|\mathcal C_l(F')-\mathcal C_l(F)|
&=
\frac{1}{S_l}
\big|
\|F'P_l\|_{1,1}-\|FP_l\|_{1,1}
\big| \\
&\le
\frac{1}{S_l}
\|F'P_l-FP_l\|_{1,1} \\
&=
\frac{1}{S_l}
\|(F'-F)P_l\|_{1,1}.
\end{aligned}
$$

右侧是 clean 与 adversarial feature 之差经过空间中心化后的总绝对位移。这个不等式等价于

$$
\|(F'-F)P_l\|_{1,1}
\ge
S_l|\mathcal C_l(F')-\mathcal C_l(F)|.
$$

所以较大的容量漂移能够推出较大的中心化特征位移下界。反向推理不成立：即使容量完全相同，特征也可能在保持总绝对离差不变的情况下重新排列。

### 4.4 与最优传输的关系

这一节先定义所有符号。设 $\mu$ 和 $\nu$ 是实数轴 $\mathbb R$ 上具有有限一阶矩的两个概率测度，即

$$
\int_{\mathbb R}|u|\,d\mu(u)<\infty,
\qquad
\int_{\mathbb R}|v|\,d\nu(v)<\infty.
$$

它们的均值分别是

$$
m_\mu=\int_{\mathbb R}u\,d\mu(u),
\qquad
m_\nu=\int_{\mathbb R}v\,d\nu(v).
$$

定义

$$
\operatorname{MAD}(\mu)
=
\int_{\mathbb R}|u-m_\mu|\,d\mu(u),
$$

并对 $\nu$ 作相同定义。

一个耦合 $\pi\in\Pi(\mu,\nu)$ 是定义在乘积空间 $\mathbb R\times\mathbb R$ 上的联合概率测度，其第一个边缘分布是 $\mu$，第二个边缘分布是 $\nu$。直观地，$u$ 来自 clean feature 分布 $\mu$，$v$ 来自 adversarial feature 分布 $\nu$，而 $\pi$ 描述如何把二者配对。

一阶 Wasserstein 距离定义为

$$
W_1(\mu,\nu)
:=
\inf_{\pi\in\Pi(\mu,\nu)}
\int_{\mathbb R\times\mathbb R}
|u-v|\,d\pi(u,v).
$$

由于 $\pi$ 的边缘分布分别为 $\mu$ 和 $\nu$，可以把两个 MAD 都写成对同一个联合测度 $\pi$ 的积分：

$$
\operatorname{MAD}(\mu)
=
\int |u-m_\mu|\,d\pi(u,v),
$$

$$
\operatorname{MAD}(\nu)
=
\int |v-m_\nu|\,d\pi(u,v).
$$

于是，对任意 $\pi\in\Pi(\mu,\nu)$，

$$
\begin{aligned}
|\operatorname{MAD}(\mu)-\operatorname{MAD}(\nu)|
&=
\left|
\int
\big(|u-m_\mu|-|v-m_\nu|\big)
\,d\pi(u,v)
\right| \\
&\le
\int
\left|
|u-m_\mu|-|v-m_\nu|
\right|
\,d\pi(u,v) \\
&\le
\int
\big(|u-v|+|m_\mu-m_\nu|\big)
\,d\pi(u,v) \\
&=
\int |u-v|\,d\pi(u,v)
+|m_\mu-m_\nu|.
\end{aligned}
$$

第二个不等式使用了实数绝对值的反三角不等式：

$$
\left|
|u-m_\mu|-|v-m_\nu|
\right|
\le
|(u-m_\mu)-(v-m_\nu)|
\le
|u-v|+|m_\mu-m_\nu|.
$$

均值差也能由同一个耦合控制：

$$
\begin{aligned}
|m_\mu-m_\nu|
&=
\left|
\int (u-v)\,d\pi(u,v)
\right| \\
&\le
\int |u-v|\,d\pi(u,v).
\end{aligned}
$$

将它代回前式，得到

$$
|\operatorname{MAD}(\mu)-\operatorname{MAD}(\nu)|
\le
2\int |u-v|\,d\pi(u,v).
$$

因为该式对每一个可行耦合 $\pi$ 都成立，所以右侧对所有 $\pi\in\Pi(\mu,\nu)$ 取下确界：

$$
|\operatorname{MAD}(\mu)-\operatorname{MAD}(\nu)|
\le 2W_1(\mu,\nu).
$$

最后把这个结论应用于第 $l$ 层的每个通道。由普通三角不等式，

$$
\begin{aligned}
|\mathcal C_l(x_{\mathrm{adv}})-\mathcal C_l(x)|
&=
\left|
\sum_{c=1}^{C_l}
\left[
\operatorname{MAD}(\mu_{l,c}^{x_{\mathrm{adv}}})
-
\operatorname{MAD}(\mu_{l,c}^{x})
\right]
\right| \\
&\le
\sum_{c=1}^{C_l}
\left|
\operatorname{MAD}(\mu_{l,c}^{x_{\mathrm{adv}}})
-
\operatorname{MAD}(\mu_{l,c}^{x})
\right| \\
&\le
2\sum_{c=1}^{C_l}
W_1(
\mu_{l,c}^{x},
\mu_{l,c}^{x_{\mathrm{adv}}}
).
\end{aligned}
$$

移项得到

$$
\sum_{c=1}^{C_l}
W_1(\mu_{l,c}^{x},\mu_{l,c}^{x_{\mathrm{adv}}})
\ge
\frac{1}{2}
|\mathcal C_l(x_{\mathrm{adv}})-\mathcal C_l(x)|.
$$

因此，若容量差很大，则逐通道 Wasserstein 距离之和不可能很小。反过来仍不成立：两个分布可以发生较大传输，但具有相同的 MAD。容量差是 feature transport 的一维统计见证量，不是完整的 optimal-transport distance。

## 5. 局部 Jacobian 解释

### 5.1 特征映射及其微分

把输入图像展平为 $x\in\mathbb R^d$。第 $l$ 层的特征映射记为

$$
\phi_l:\mathbb R^d
\longrightarrow
\mathbb R^{C_l\times S_l},
\qquad
F_l(x)=\phi_l(x).
$$

令输入扰动为

$$
\delta=x_{\mathrm{adv}}-x\in\mathbb R^d.
$$

如果 $\phi_l$ 在 $x$ 处可微，它的 Fréchet 微分

$$
D\phi_l(x):
\mathbb R^d
\longrightarrow
\mathbb R^{C_l\times S_l}
$$

是一个线性映射。它把输入方向 $\delta$ 映射为一阶特征变化

$$
H_l
:=
D\phi_l(x)[\delta]
\in\mathbb R^{C_l\times S_l}.
$$

于是严格的一阶展开为

$$
F_l(x+\delta)
=
F_l(x)+H_l+r_l(\delta),
$$

其中余项 $r_l(\delta)\in\mathbb R^{C_l\times S_l}$ 满足

$$
\lim_{\|\delta\|_2\to0}
\frac{\|r_l(\delta)\|_F}{\|\delta\|_2}
=0.
$$

如果使用普通 Jacobian 矩阵表示，定义

$$
J_l(x)
=
\frac{\partial\,\operatorname{vec}(F_l(x))}
{\partial x}
\in\mathbb R^{(C_lS_l)\times d},
$$

则

$$
\operatorname{vec}(H_l)=J_l(x)\delta.
$$

必须先把 $J_l(x)\delta$ reshape 回 $C_l\times S_l$ 才能右乘 $P_l$。后面统一使用已经具有正确矩阵维度的 $H_l$，避免记号混乱。

含 ReLU 的网络是分段线性的，在 ReLU 转折点之外几乎处处可微；恰好位于转折点时，应使用单侧方向导数或次梯度，而不能直接声称普通导数存在。

### 5.2 容量泛函的方向导数

暂时固定一层并记

$$
F=F_l(x),
\qquad
A=FP_l,
\qquad
B=H_lP_l.
$$

沿特征方向 $H_l$ 移动标量步长 $t$ 后，容量为

$$
\mathcal C_l(F+tH_l)
=
\frac{1}{S_l}
\sum_{c=1}^{C_l}
\sum_{s=1}^{S_l}
|A_{c,s}+tB_{c,s}|.
$$

如果所有 $A_{c,s}\ne0$，那么每一项在 $t=0$ 处可微，并且

$$
\left.
\frac{d}{dt}|A_{c,s}+tB_{c,s}|
\right|_{t=0}
=
\operatorname{sign}(A_{c,s})B_{c,s}.
$$

因此

$$
D\mathcal C_l(F)[H_l]
=
\frac{1}{S_l}
\left\langle
\operatorname{sign}(FP_l),
H_lP_l
\right\rangle_F.
$$

这里

$$
\langle U,V\rangle_F
:=
\sum_{c=1}^{C_l}
\sum_{s=1}^{S_l}
U_{c,s}V_{c,s}
$$

是 Frobenius 内积；$\operatorname{sign}(A)$ 对矩阵逐元素取符号。

如果某些 $A_{c,s}=0$，普通双侧导数可能不存在。此时从 $t\downarrow0$ 的单侧方向导数是

$$
\mathcal C_l'(F;H_l)
=
\frac{1}{S_l}
\left[
\sum_{A_{c,s}\ne0}
\operatorname{sign}(A_{c,s})B_{c,s}
+
\sum_{A_{c,s}=0}|B_{c,s}|
\right].
$$

这就是绝对值函数在零点需要单独处理的原因。

### 5.3 把输入扰动代入容量变化

在 $\phi_l$ 可微且 $F_l(x)P_l$ 没有零元素的局部区域中，把 $H_l=D\phi_l(x)[\delta]$ 代入容量的一阶展开：

$$
\mathcal C_l(F_l(x+\delta))
-
\mathcal C_l(F_l(x))
=
\frac{1}{S_l}
\left\langle
\operatorname{sign}(F_l(x)P_l),
D\phi_l(x)[\delta]P_l
\right\rangle_F
+o(\|\delta\|_2).
$$

忽略高阶余项，并代入相对容量漂移的定义：

$$
D_l^{\mathrm{cap}}
\approx
\frac{
\left|
\left\langle
\operatorname{sign}(F_l(x)P_l),
D\phi_l(x)[\delta]P_l
\right\rangle_F
\right|
}{
\|F_l(x)P_l\|_{1,1}+S_l\varepsilon
}.
$$

分母之所以从 $\mathcal C_l(F)+\varepsilon$ 变成 $\|FP_l\|_{1,1}+S_l\varepsilon$，是因为分子和分母同时乘了 $S_l$。

因此该局部表达可以解释为：输入方向 $\delta$ 经特征微分 $D\phi_l(x)$ 传播后，先由 $P_l$ 去掉空间常数分量，再投影到 $L_1$ 容量的局部法向量 $\operatorname{sign}(F_lP_l)$ 上，最后除以 clean feature 的容量尺度。

它只保留会改变 $\|F_lP_l\|_{1,1}$ 的一阶分量。若某个特征变化很大但恰好沿着该 $L_1$ 等值面的切向方向移动，一阶 capacity drift 可以为零。因此它既不是完整 Jacobian norm，也不是完整 feature distance。

## 6. 它为什么与最终分类错误有关

### 6.1 分类 margin 的严格定义

模型输出 $K$ 个 logits：

$$
z(x)=f(x)\in\mathbb R^K.
$$

在第 $l$ 层把模型分解为

$$
f=h_l\circ\phi_l,
$$

其中

$$
\phi_l:\mathbb R^d\to\mathbb R^{C_l\times S_l}
$$

产生第 $l$ 层特征，而

$$
h_l:\mathbb R^{C_l\times S_l}\to\mathbb R^K
$$

表示从第 $l$ 层到 logits 的剩余网络。因此

$$
z(x)=h_l(F_l(x)).
$$

设真实类别为 $y$，任意错误类别为 $k\ne y$。定义成对分类 margin

$$
\gamma_{y,k}(F)
:=
[h_l(F)]_y-[h_l(F)]_k.
$$

模型在输入 $x$ 上正确分类，当且仅当

$$
\gamma_{y,k}(F_l(x))>0
\quad
\text{对所有 }k\ne y\text{ 都成立}.
$$

通常所说的最小分类 margin 是

$$
\gamma_y(x)
=
\min_{k\ne y}\gamma_{y,k}(F_l(x))
=
z_y(x)-\max_{k\ne y}z_k(x).
$$

由于 $\max$ 在两个类别 logit 相等时可能不可微，下面固定一个竞争类别 $k$，分析可微的成对 margin $\gamma_{y,k}$。这避免把不可微问题隐藏在记号中。

### 6.2 Margin 变化与 capacity 变化是两个不同投影

记实际特征变化为

$$
\Delta F_l
:=
F_l(x_{\mathrm{adv}})-F_l(x)
\in\mathbb R^{C_l\times S_l}.
$$

若 $\gamma_{y,k}$ 在 $F_l(x)$ 处可微，则

$$
\begin{aligned}
\Delta\gamma_{y,k}
&:=
\gamma_{y,k}(F_l+\Delta F_l)
-
\gamma_{y,k}(F_l) \\
&=
\left\langle
\nabla_F\gamma_{y,k}(F_l),
\Delta F_l
\right\rangle_F
+
o(\|\Delta F_l\|_F).
\end{aligned}
$$

这里 $\nabla_F\gamma_{y,k}(F_l)\in\mathbb R^{C_l\times S_l}$ 是成对 margin 对特征矩阵的梯度。

在同一个可微区域内，capacity 变化为

$$
\Delta\mathcal C_l
=
\frac{1}{S_l}
\left\langle
\operatorname{sign}(F_lP_l),
\Delta F_lP_l
\right\rangle_F
+
o(\|\Delta F_l\|_F).
$$

两式虽然共享同一个 $\Delta F_l$，但投影方向不同：

- margin 使用方向 $\nabla_F\gamma_{y,k}(F_l)$；
- capacity 使用方向 $\operatorname{sign}(F_lP_l)P_l/S_l$。

因此，不存在一个无条件定理说“capacity 增大一定导致 margin 减小”。两个线性泛函在一般情况下可以互不相关。

### 6.3 一个可检验的结构分解

令

$$
A_l=F_lP_l
$$

表示 clean feature 的中心化部分。把对抗特征变化的中心化部分分解为

$$
\Delta F_lP_l
=
\eta_lA_l+R_l.
$$

其中：

- $\eta_l\in\mathbb R$ 是沿当前中心化特征结构 $A_l$ 的相对伸缩系数；
- $R_l\in\mathbb R^{C_l\times S_l}$ 是不能由单一伸缩解释的残差；
- 该分解本身并不唯一，除非再规定如何选取 $\eta_l$。一种明确选择是 Frobenius 最小二乘投影

$$
\eta_l
:=
\frac{\langle A_l,\Delta F_lP_l\rangle_F}
{\|A_l\|_F^2},
\qquad A_l\ne0,
$$

并令

$$
R_l
:=
\Delta F_lP_l-\eta_lA_l.
$$

此时由投影构造可知 $\langle A_l,R_l\rangle_F=0$。但要注意，这个 Frobenius 正交条件并不自动推出

$$
\langle\operatorname{sign}(A_l),R_l\rangle_F=0.
$$

把分解代入 capacity 的一阶展开。为保留近似误差，记 Taylor 余项为 $r_l^{\mathcal C}$，并要求

$$
r_l^{\mathcal C}
=
o(\|\Delta F_l\|_F).
$$

于是

$$
\begin{aligned}
\Delta\mathcal C_l
&=
\frac{1}{S_l}
\langle\operatorname{sign}(A_l),\eta_lA_l+R_l\rangle_F
+
r_l^{\mathcal C} \\
&=
\eta_l\mathcal C_l(F_l)
+
e_l^{\mathcal C}
+
r_l^{\mathcal C},
\end{aligned}
$$

其中定义 capacity 法向残差

$$
e_l^{\mathcal C}
:=
\frac{1}{S_l}
\langle\operatorname{sign}(A_l),R_l\rangle_F.
$$

实际相对指标为

$$
D_{l,\varepsilon}^{\mathrm{cap}}
=
\frac{
|\eta_l\mathcal C_l(F_l)+e_l^{\mathcal C}+r_l^{\mathcal C}|
}{
\mathcal C_l(F_l)+\varepsilon
}.
$$

利用反三角不等式，可以给出与 $|\eta_l|$ 的误差上界：

$$
\left|
D_{l,\varepsilon}^{\mathrm{cap}}
-
|\eta_l|
\right|
\le
\frac{
|e_l^{\mathcal C}|
+
|r_l^{\mathcal C}|
+
|\eta_l|\varepsilon
}{
\mathcal C_l(F_l)+\varepsilon
}.
$$

所以只有当右侧误差相对较小时，才能写

$$
D_{l,\varepsilon}^{\mathrm{cap}}
\approx
|\eta_l|.
$$

这明确列出了近似所依赖的三部分误差：非伸缩残差、Taylor 高阶余项和数值稳定项 $\varepsilon$。

### 6.4 从结构伸缩到 margin erosion 还需要三个条件

由于 $P_l$ 是投影矩阵，任意特征变化都可写为

$$
\Delta F_l
=
\Delta F_lP_l
+
\Delta F_l(I_{S_l}-P_l).
$$

第二项是每个通道在空间上的常数变化，记为

$$
Q_l
:=
\Delta F_l(I_{S_l}-P_l).
$$

所以

$$
\Delta F_l
=
\eta_lA_l+R_l+Q_l.
$$

将它代入 margin 的一阶展开，并记 margin 的 Taylor 余项为

$$
r_{l,y,k}^{\gamma}
=
o(\|\Delta F_l\|_F).
$$

则

$$
\Delta\gamma_{y,k}
=
\eta_l q_{l,y,k}
+
e_{l,y,k}
+
r_{l,y,k}^{\gamma},
$$

其中

$$
q_{l,y,k}
:=
\left\langle
\nabla_F\gamma_{y,k}(F_l),
A_l
\right\rangle_F
$$

表示后续网络对当前中心化特征结构的可观测程度，而

$$
e_{l,y,k}
:=
\left\langle
\nabla_F\gamma_{y,k}(F_l),
R_l+Q_l
\right\rangle_F
$$

汇总残差结构和空间常数分量对 margin 的影响。

要让较大的 capacity drift 预示 margin 被侵蚀，至少需要：

1. **下游可观测性：** $|q_{l,y,k}|$ 不能接近零，否则分类头基本不读取 $A_l$ 方向；
2. **对抗方向对齐：** $\eta_lq_{l,y,k}<0$，否则该伸缩分量会增大而不是减小 margin；
3. **残差受控：**

$$
\left|
e_{l,y,k}+r_{l,y,k}^{\gamma}
\right|
<
|\eta_lq_{l,y,k}|,
$$

否则残差可能抵消主项。

若条件 2 成立，则 $\eta_lq_{l,y,k}=-|\eta_lq_{l,y,k}|$。再结合条件 3：

$$
\Delta\gamma_{y,k}
=
-|\eta_lq_{l,y,k}|
+
e_{l,y,k}
+
r_{l,y,k}^{\gamma}
<0.
$$

而且 margin 的下降幅度满足严格下界

$$
-\Delta\gamma_{y,k}
\ge
|\eta_l|\,|q_{l,y,k}|
-
\left|
e_{l,y,k}+r_{l,y,k}^{\gamma}
\right|.
$$

因此，只有在三个条件成立，并且上一节 $D_{l,\varepsilon}^{\mathrm{cap}}\approx|\eta_l|$ 的误差较小时，较大的 capacity drift 才能成为 margin erosion 的代理。

因此，理论链条应准确写成：

$$
\text{capacity drift 大}
\Longrightarrow
\text{中心化 feature transport 不小},
$$

这是前面已经证明的严格结论；而

$$
\text{capacity drift 大}
\Longrightarrow
\text{margin 下降}
$$

只在下游可观测性、对抗方向对齐和残差受控三个条件下成立。如果特征变化落在后续分类头的零空间中，capacity 可以变化而输出不变。该指标是具有机制解释的 ranking proxy，不是鲁棒性的充分必要条件。

## 7. 为什么更接近输出的指标反而可能较差

“接近输出”提高了攻击结果的直接性，却不自动带来跨架构排序的可比性。输出指标通常混合了真实鲁棒性和多种 nuisance factors。

### 7.1 Logit 尺度反例

令新模型输出

$$
z'(x)=\alpha z(x),\qquad \alpha>0.
$$

它对所有输入的 argmax 决策完全不变，因此理论 robust accuracy 不变；但

$$
\frac{\|z'(x_{\mathrm{adv}})-z'(x)\|}{\|x_{\mathrm{adv}}-x\|}
=
\alpha
\frac{\|z(x_{\mathrm{adv}})-z(x)\|}{\|x_{\mathrm{adv}}-x\|}.
$$

所以 `local_logit_lipschitz` 可以在分类器完全相同的情况下任意缩放。交叉熵

$$
\mathcal{L}_{\mathrm{CE}}(\alpha z,y)
=
\log\left(1+\sum_{k\ne y}e^{\alpha(z_k-z_y)}\right)
$$

也会随温度改变。因此 `adversarial_loss` 同时测量攻击效果、margin 和置信度校准，并不只测量 robust accuracy。

### 7.2 公共 logit 平移与无关方向

令

$$
z'(x)=z(x)+c(x)\mathbf{1}.
$$

softmax、预测类别和交叉熵都不变，但若 $c(x_{\mathrm{adv}})\ne c(x)$，logit 差的欧氏范数会改变。`local_logit_lipschitz` 还会统计与所有类别决策边界正交的 logit 移动，而攻击是否成功只取决于 pairwise margin 是否跨过零点。

### 7.3 输出瓶颈掩盖内部传播机制

分类头把高维中间特征压缩成少量 logits。多个通道或层的变化可能在输出处抵消；反过来，一个很小但恰好沿边界法向的变化也可能使输出翻转。因此，终端指标适合描述“这次攻击最终造成了什么”，却未必稳定描述“这个架构在传播扰动时有多敏感”。

多层 capacity drift 测量的是攻击在表征流中的累积响应，并通过跨样本、跨层平均降低单一分类头和单一边界方向带来的方差。

### 7.4 White-box 与 shared-adv 的结果并不矛盾

当前 20 模型结果为：

| 指标 | White-box Spearman | Standard-source Spearman | 三源 Spearman |
|---|---:|---:|---:|
| `perturbation_capacity_delta` (Ours) | -0.626 | -0.454 | **-0.565** |
| `adversarial_loss` | **-0.805** | -0.244 | -0.486 |
| `local_logit_lipschitz` | -0.329 | -0.346 | -0.290 |

`adversarial_loss` 在 white-box 中最强是合理的：攻击针对每个目标模型优化，loss 直接反映该模型的最坏方向。但 shared-adv 样本并不一定沿每个目标模型的最坏决策方向，终端 loss 的指示性会明显衰减。capacity drift 仍能测量同一外部扰动在不同架构内部被放大或抑制的程度，因此更适合当前“固定刺激、比较架构响应”的实验设计。

旧实验中曾出现 `local_logit_lipschitz` 约为 $-0.937$ 的单协议结果，但它不能与当前统一协议的数值混为同一结论；论文应只比较统一样本、统一攻击和统一聚合方式下的结果。

## 8. 为什么完整 capacity 优于 ReLU / above-mean 硬门控

完整 capacity 使用

$$
|a_{l,c,s}-m_{l,c}|
$$

同时保留高于均值和低于均值的空间偏差。低于均值不等于“对网络无效”：

1. 在 post-ReLU 特征中，低于通道均值的正激活仍然是有效响应；
2. 在 pre-ReLU、BatchNorm 或 residual 路径中，负值仍可能影响后续计算；
3. 对抗攻击可能通过改变激活支持集或抑制原有高响应区域起作用；
4. 硬阈值会丢失一部分分布信息，并在阈值穿越处引入额外不连续性和方差。

当前三源结果也与这一解释一致：

| 版本 | 三源 Spearman |
|---|---:|
| 完整 `perturbation_capacity_delta` (Ours) | **-0.565** |
| $a>\operatorname{mean}(a)$ | -0.471 |
| $a>0$ | -0.392 |

因此，hard gate 更适合作为消融：它说明优势不是简单来自“只统计 ReLU 激活”，而来自对**完整双侧空间离散结构**的稳定刻画。

## 9. 论文 Motivation 的推荐叙事

### 9.1 问题

现有 output-level proxy 直接测量攻击后果，但容易受到 logit scale、模型校准、分类头和攻击源的影响；这会妨碍 RobustNAS 中的跨架构公平排序。

### 9.2 观察

对抗扰动首先改变多层中间表征，输出错误是这些改变经过网络传播和分类头投影后的最终事件。鲁棒架构应当在相同外部扰动下保持其内部有效表征容量相对稳定。

### 9.3 方法

将每个通道的空间激活表示为经验测度，以一阶中心绝对矩定义通道容量，再测量 clean 与 adversarial feature 之间的相对容量漂移，并在多个层上聚合。

### 9.4 理论支撑

该指标具有通道平移不变性、近似尺度不变性；它是中心化 $L_1$ feature displacement 和 Wasserstein feature transport 的稳定见证量；其局部形式是攻击方向 feature Jacobian 在当前表征结构方向上的归一化投影。在明确的 downstream observability 假设下，它与分类 margin erosion 建立联系。

### 9.5 经验主张

在当前 20 模型三源 shared-adv 实验中，它比 loss、gradient、logit Lipschitz 和 hard-gated variants 提供更好的鲁棒性排序，并表现出更好的 source stability。

## 10. 论文中可以和不可以写的结论

可以写：

> We introduce relative perturbation-capacity drift, a centered first-absolute-moment functional that measures how adversarial perturbations alter the spatial representational capacity of intermediate features.

> The metric is invariant to channel-wise spatial offsets, approximately invariant to common feature rescaling, and provides a computationally inexpensive witness of centered feature transport.

> Under a downstream observability assumption, capacity drift captures the component of feature-Jacobian amplification that can contribute to classification-margin erosion.

不应写：

- `perturbation_capacity_delta` 在任何场景下都是最好的指标；
- capacity drift 与 robust accuracy 数学等价；
- 小 capacity drift 可以证明模型鲁棒；
- 当前 20 模型相关性已经足以证明对未见架构、数据集和攻击的普适性。

## 11. 下一步理论与实验验证

1. 在 CIFAR-100、SVHN、MNIST、ImageNet 上检验跨数据集排序稳定性；
2. 在未参与指标选择的架构集合上做 held-out validation，避免选择偏差；
3. 审计并归档 `local_logit_lipschitz` 的旧单协议结果，论文主表只保留当前统一协议数值；
4. 报告三源分别的 Spearman、均值、标准差和跨源 rank variance；
5. 控制 clean accuracy、parameter count、FLOPs、feature width 和 logit temperature 后做 partial/residual correlation；
6. 检验 capacity drift 对 RobustNAS 搜索排序的增量价值，而不只检验静态模型相关性。

## 12. 参考文献

1. Guo et al. *When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks*. CVPR 2020. <https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_When_NAS_Meets_Robustness_In_Search_of_Robust_Architectures_Against_CVPR_2020_paper.html>
2. Xie et al. *Feature Denoising for Improving Adversarial Robustness*. CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Feature_Denoising_for_Improving_Adversarial_Robustness_CVPR_2019_paper.pdf>
3. Zhang et al. *Theoretically Principled Trade-off between Robustness and Accuracy*. ICML 2019. <https://proceedings.mlr.press/v97/zhang19p.html>
4. Croce et al. *RobustBench: A Standardized Adversarial Robustness Benchmark*. NeurIPS 2021 Datasets and Benchmarks. <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a3c65c2974270fd093ee8a9bf8ae7d0b-Abstract-round2.html>
5. Ou et al. *Towards Accurate and Robust Architectures via Neural Architecture Search*. CVPR 2024. <https://openaccess.thecvf.com/content/CVPR2024/html/Ou_Towards_Accurate_and_Robust_Architectures_via_Neural_Architecture_Search_CVPR_2024_paper.html>
