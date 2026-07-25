# Feature Measure ROBUST Metric Design

Date: 2026-07-04

MathJax rendered version: `docs/feature_measure_metric_design_math.html`

This note now focuses only on two quantities:

1. strict FSP from *When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks*;
2. our measure-based ROBUST metric.

Older metrics such as SWD, MMD, covariance delta, and logit losses are kept in code for later comparison, but they are not the focus here.

## 1. Strict FSP From RobNets

The RobNets paper defines the FSP matrix for the `l`-th cell using the cell input feature map and output feature map:

$$
G_l(x;\theta)
=
\frac{1}{h w}
\sum_{s=1}^{h}
\sum_{t=1}^{w}
F^{\mathrm{in}}_{l,s,t}(x;\theta)
\left(F^{\mathrm{out}}_{l,s,t}(x;\theta)\right)^{\top}.
$$

Equivalently, if each spatial position gives a paired atom

$$
\left(a^{\mathrm{in}}_{l,s}(x), a^{\mathrm{out}}_{l,s}(x)\right)
\in \mathbb{R}^{C_{\mathrm{in}}}\times \mathbb{R}^{C_{\mathrm{out}}},
$$

then

$$
G_l(x;\theta)
=
\int a^{\mathrm{in}} \left(a^{\mathrm{out}}\right)^{\top}
\, d\rho_l^x(a^{\mathrm{in}}, a^{\mathrm{out}})
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
a^{\mathrm{in}}_{l,s}(x)
\left(a^{\mathrm{out}}_{l,s}(x)\right)^{\top}.
$$

The paper's FSP loss is:

$$
L^{\mathrm{FSP}}_l
=
\frac{1}{N}
\sum_{x}
\left\|G_l(x;\theta)-G_l(x';\theta)\right\|_F^2.
$$

Important correction: there is no denominator such as

$$
\left\|G_l(x;\theta)\right\|_F+\varepsilon
$$

in the original FSP loss. The only averaging factor in the FSP matrix itself is the spatial average `1 / (h w)`, and the loss averages over samples.

The public RobNets repository contains training/evaluation code for the discovered architectures, but the released files do not expose a reusable FSP-search implementation. Therefore, our strict FSP implementation follows the paper equations directly.

For generic RobustBench models, there is no explicit "cell input/cell output" API, so the current implementation uses adjacent selected 4D activation tensors as a model-agnostic approximation of cell input/output feature flow:

$$
\widehat{G}_{i,j}(x)
=
\frac{1}{S}
\sum_{s=1}^{S}
a_{i,s}(x)a_{j,s}(x)^{\top},
\quad i < j.
$$

The experimental field is:

$$
\mathrm{fsp\_strict\_delta}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\frac{1}{N}\sum_x
\left\|
\widehat{G}_{i,j}(x_{\mathrm{adv}})
-
\widehat{G}_{i,j}(x)
\right\|_F^2.
$$

## 2. What "First-Order" and "Second-Order" Mean

For layer `l`, the feature map is:

$$
F_l(x)\in\mathbb{R}^{C_l\times H_l\times W_l}.
$$

Each spatial position `s` gives one local feature atom:

$$
a_{l,s}(x)=F_l(x)[:,h,w]\in\mathbb{R}^{C_l}.
$$

The layer can be viewed as an empirical feature measure:

$$
\mu_l^x
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
\delta_{a_{l,s}(x)}.
$$

Here `\delta_{a}` is the Dirac measure at feature vector `a`: all probability mass is concentrated at that one point. So `\mu_l^x` means "the distribution formed by all spatial feature vectors in this layer".

The adversarial image induces another measure:

$$
\mu_l^{x_{\mathrm{adv}}}
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
\delta_{a_{l,s}(x_{\mathrm{adv}})}.
$$

The metric-design question is:

$$
\text{How much does } \mu_l^x
\text{ move to }
\mu_l^{x_{\mathrm{adv}}}\text{?}
$$

### First-Order

The first-order moment of a measure is its mean:

$$
m_l(x)
=
\int a\,d\mu_l^x(a)
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}a_{l,s}(x).
$$

But using only `m_l(x)` is exactly the GAP problem: it collapses the whole spatial measure into one average vector.

So our first-order term does not use GAP. It measures local displacement of each spatial atom:

$$
u_{l,s}(x)
=
\frac{a_{l,s}(x)}
{\|a_{l,s}(x)\|_2+\varepsilon}.
$$

Clean feature energy defines spatial weights:

$$
w_{l,s}(x)
=
\frac{\|a_{l,s}(x)\|_2}
{\sum_t\|a_{l,t}(x)\|_2+\varepsilon}.
$$

Then:

$$
D^{(1)}_l(x,x_{\mathrm{adv}})
=
\sum_{s=1}^{S_l}
w_{l,s}(x)
\left\|
u_{l,s}(x_{\mathrm{adv}})
-
u_{l,s}(x)
\right\|_2.
$$

This is called first-order because it tracks movement of individual feature atoms under the feature measure, not pairwise relations between atoms/channels/layers.

### Second-Order

A second-order moment measures feature interactions. For one layer, this would look like:

$$
M_l(x)
=
\int aa^{\top}\,d\mu_l^x(a)
=
\frac{1}{S_l}
\sum_s a_{l,s}(x)a_{l,s}(x)^{\top}.
$$

FSP is a cross-layer second-order moment. It measures how input-side features and output-side features co-activate at the same spatial positions:

$$
G_l(x)
=
\int a^{\mathrm{in}}
\left(a^{\mathrm{out}}\right)^{\top}
d\rho_l^x
$$

where `\rho_l^x` is the empirical joint measure of paired input/output feature atoms. This is why FSP is strong: it does not merely ask "did a feature channel change?", but "did the feature-flow relationship through the cell change?"

## 3. Our ROBUST Metric

Our metric keeps FSP's strongest idea, but adds a direct feature-measure displacement term. To keep the original FSP baseline strict, `fsp_strict_delta` remains the paper-style squared Frobenius loss. For the combined ROBUST metric, however, the second-order term is put on a per-FSP-entry RMS scale so it can be added to the first-order atom displacement without being dominated by channel count:

$$
\widetilde{D}^{(2)}_{i,j}(x,x_{\mathrm{adv}})
=
\frac{1}{\sqrt{C_i C_j}}
\left\|
\widehat{G}_{i,j}(x_{\mathrm{adv}})
-
\widehat{G}_{i,j}(x)
\right\|_F.
$$

For each adjacent selected feature-flow pair `(i,j)`, define:

$$
D^{(1)}_{i,j}
=
\frac{1}{2}
\left(
D^{(1)}_i
+
D^{(1)}_j
\right).
$$

Then the current ROBUST metric is:

$$
D^{\mathrm{ROBUST}}_{i,j}
=
\widetilde{D}^{(2)}_{i,j}
+
\lambda D^{(1)}_{i,j},
\quad \lambda=1.
$$

The final score averages over selected feature-flow pairs:

$$
\mathrm{feature\_measure\_robust\_delta}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
D^{\mathrm{ROBUST}}_{i,j}.
$$

Lower is expected to be better: a robust model should make adversarial perturbations induce smaller feature-measure displacement and smaller feature-flow distortion.

## 4. Difference Between Strict FSP and Our ROBUST Metric

| Item | Strict FSP | Our ROBUST metric |
|---|---|---|
| Main object | Cross-layer FSP matrix | Feature measure plus cross-layer FSP matrix |
| Mathematical order | Second-order only | First-order local displacement plus second-order feature flow |
| Formula core | `||G(x_adv)-G(x)||_F^2` | `RMS(D2) + lambda * D1` |
| GAP usage | No | No |
| Denominator by `||G(x)||` | No | No; only dimension RMS scaling by `sqrt(C_i C_j)` |
| What it may miss | Direct local feature-atom movement | Less likely to miss local atom movement because `D1` is explicit |

The old project field `fsp_delta` is not the original RobNets FSP. It is a normalized same-layer Gram variant retained only for historical comparison. For strict comparison with RobNets-style FSP, use `fsp_strict_delta`.
