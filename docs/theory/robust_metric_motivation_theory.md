# ROBUST Metric: Motivation and Mathematical Derivation

Date: 2026-07-04

## 1. Research Motivation

Adversarial robustness studies the stability of a classifier under small but worst-case input perturbations. For a classifier $f_\theta:\mathcal{X}\to\mathbb{R}^K$, the robust prediction condition around a clean sample $x$ is:

$$
\arg\max_k f_{\theta,k}(x+\delta)
=
\arg\max_k f_{\theta,k}(x),
\quad
\|\delta\|_p\le \varepsilon .
$$

Classical adversarial evaluation attacks the final loss or final logits. This is necessary for evaluation, but it is not ideal as a robustness *metric for architecture analysis*:

1. The final logit/loss is a late-stage signal. It tells us whether the perturbation has succeeded, but not where or how the representation has become unstable.
2. A single global pooled feature vector can miss spatially local but semantically important feature changes.
3. A purely second-order FSP-style metric captures feature-flow structure, but can miss direct displacement of local feature atoms.
4. For architecture-level robustness, we want a metric that measures whether adversarial perturbation is absorbed or amplified inside the representation hierarchy.

Our central hypothesis is:

$$
\textbf{A robust architecture should induce small adversarial displacement
in its intermediate feature-space measures and feature-flow moments.}
$$

This hypothesis is consistent with three lines of prior work. Goodfellow et al. showed that small input perturbations can be amplified through high-dimensional linear responses. Madry et al. formulated adversarial robustness as robust optimization over worst-case perturbations. Zhang et al. showed that robustness can be understood through stability between clean and adversarial predictions. RobNets further observed that FSP matrix loss in deeper cells correlates with robustness gaps. Our metric takes the representation-stability view seriously and measures it before the final decision layer.

## 2. Feature Maps as Empirical Measures

Let the $l$-th selected feature map be:

$$
F_l(x)\in\mathbb{R}^{C_l\times H_l\times W_l}.
$$

Each spatial location $s=(h,w)$ defines a local feature atom:

$$
a_{l,s}(x)=F_l(x)[:,h,w]\in\mathbb{R}^{C_l}.
$$

Let $S_l=H_lW_l$. The whole layer can be represented as an empirical measure over feature space:

$$
\mu_l^x
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
\delta_{a_{l,s}(x)} .
$$

Here $\delta_a$ is the Dirac measure at feature vector $a$. Intuitively, it means "put one unit of probability mass exactly at point $a$ and nowhere else." Therefore, $\delta_{a_{l,s}(x)}$ means that the feature vector at spatial position $s$ is treated as one sample point in feature space. The empirical measure $\mu_l^x$ is just the collection of all these feature points, each with weight $1/S_l$.

Under adversarial perturbation $x_{\mathrm{adv}}=x+\delta$, each spatial feature atom may move from $a_{l,s}(x)$ to $a_{l,s}(x_{\mathrm{adv}})$. The layer measure therefore changes to:

$$
\mu_l^{x_{\mathrm{adv}}}
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
\delta_{a_{l,s}(x_{\mathrm{adv}})} .
$$

Therefore, the robustness metric problem can be stated as a measure-discrepancy problem:

$$
\text{How large is the displacement }
\mu_l^x\to\mu_l^{x_{\mathrm{adv}}}
\text{ induced by adversarial perturbation?}
$$

This view is stronger than comparing pooled features, because it keeps the spatial population of local feature atoms.

## 3. First-Order Term: Local Feature-Measure Displacement

The first-order moment of $\mu_l^x$ is:

$$
m_l(x)
=
\int a\,d\mu_l^x(a)
=
\frac{1}{S_l}
\sum_{s=1}^{S_l}
a_{l,s}(x).
$$

Using only $m_l(x)$ is equivalent to global average pooling. It discards spatial distribution and can hide compensating changes: one feature atom can move strongly in one direction while another moves in the opposite direction, leaving the mean almost unchanged.

To avoid this weakness, define normalized local atoms:

$$
u_{l,s}(x)
=
\frac{a_{l,s}(x)}
{\|a_{l,s}(x)\|_2+\varepsilon}.
$$

Use clean feature energy as spatial importance:

$$
w_{l,s}(x)
=
\frac{\|a_{l,s}(x)\|_2}
{\sum_{t=1}^{S_l}\|a_{l,t}(x)\|_2+\varepsilon}.
$$

This gives weighted empirical measures:

$$
\nu_l^x
=
\sum_s w_{l,s}(x)\delta_{u_{l,s}(x)},
\quad
\nu_l^{x_{\mathrm{adv}}}
=
\sum_s w_{l,s}(x)\delta_{u_{l,s}(x_{\mathrm{adv}})}.
$$

We then define:

$$
D_l^{(1)}(x,x_{\mathrm{adv}})
=
\sum_s
w_{l,s}(x)
\left\|
u_{l,s}(x_{\mathrm{adv}})
-
u_{l,s}(x)
\right\|_2 .
$$

### Measure-Theoretic Meaning

Let $\Pi(\nu_l^x,\nu_l^{x_{\mathrm{adv}}})$ be all couplings between the two empirical measures. A coupling $\pi(u,v)$ is a joint distribution over clean feature atom $u$ and adversarial feature atom $v$, whose first marginal is $\nu_l^x$ and second marginal is $\nu_l^{x_{\mathrm{adv}}}$. In other words, $\nu_l^x$ and $\nu_l^{x_{\mathrm{adv}}}$ do not disappear; they define the feasible set of $\pi$.

The 1-Wasserstein distance is:

$$
W_1(\nu_l^x,\nu_l^{x_{\mathrm{adv}}})
=
\inf_{\pi\in\Pi(\nu_l^x,\nu_l^{x_{\mathrm{adv}}})}
\int_{\mathbb{R}^{C_l}\times\mathbb{R}^{C_l}}
\|u-v\|_2\,d\pi(u,v).
$$

The canonical spatial coupling pairs each clean atom with its adversarial counterpart at the same spatial location:

$$
\pi_{\mathrm{spatial}}
=
\sum_s w_{l,s}(x)
\delta_{(u_{l,s}(x),u_{l,s}(x_{\mathrm{adv}}))}.
$$

Since $W_1$ is the infimum over all couplings:

$$
W_1(\nu_l^x,\nu_l^{x_{\mathrm{adv}}})
\le
\sum_s
w_{l,s}(x)
\left\|
u_{l,s}(x_{\mathrm{adv}})
-
u_{l,s}(x)
\right\|_2
=
D_l^{(1)} .
$$

Thus $D_l^{(1)}$ is a computable spatial-coupling upper bound of feature-measure transport. It is not merely a heuristic distance; it has a precise optimal-transport interpretation.

## 4. Second-Order Term: Feature-Flow Moment Displacement

First-order displacement captures movement of individual local feature atoms, but it does not fully capture how feature channels and layers co-activate. Robustness is not only about whether features move, but also about whether the feature transformation flow remains stable.

For two selected layers $i<j$, define the empirical joint measure over paired spatial atoms:

$$
\rho_{i,j}^x
=
\frac{1}{S}
\sum_{s=1}^{S}
\delta_{(a_{i,s}(x),a_{j,s}(x))}.
$$

The cross-layer second moment is:

$$
G_{i,j}(x)
=
\int ab^\top\,d\rho_{i,j}^x(a,b)
=
\frac{1}{S}
\sum_{s=1}^{S}
a_{i,s}(x)a_{j,s}(x)^\top .
$$

This is the same mathematical object as FSP: a cross-layer feature-flow Gramian. The strict RobNets-style FSP loss is:

$$
L_{i,j}^{\mathrm{FSP}}
=
\frac{1}{N}
\sum_x
\left\|
G_{i,j}(x_{\mathrm{adv}})
-
G_{i,j}(x)
\right\|_F^2 .
$$

### Functional-Analytic Meaning

Consider the bilinear test function class:

$$
\mathcal{B}
=
\left\{
\varphi_M(a,b)=a^\top M b:
\|M\|_F\le 1
\right\}.
$$

For any $M$:

$$
\mathbb{E}_{\rho_{i,j}^{x_{\mathrm{adv}}}}\varphi_M
-
\mathbb{E}_{\rho_{i,j}^{x}}\varphi_M
=
\langle M,
G_{i,j}(x_{\mathrm{adv}})-G_{i,j}(x)
\rangle_F .
$$

By Frobenius duality:

$$
\sup_{\|M\|_F\le 1}
\left|
\langle M,
G_{i,j}(x_{\mathrm{adv}})-G_{i,j}(x)
\rangle_F
\right|
=
\left\|
G_{i,j}(x_{\mathrm{adv}})-G_{i,j}(x)
\right\|_F .
$$

Therefore, the FSP displacement is an integral probability metric over a bilinear feature-flow function class. It measures the maximal adversarial change detectable by normalized bilinear probes of cross-layer co-activation.

For the combined metric, we use a dimension-balanced RMS version:

$$
\widetilde{D}^{(2)}_{i,j}(x,x_{\mathrm{adv}})
=
\frac{1}{\sqrt{C_iC_j}}
\left\|
G_{i,j}(x_{\mathrm{adv}})
-
G_{i,j}(x)
\right\|_F .
$$

This is not a normalization by the clean FSP matrix. It only converts a matrix-level Frobenius norm into per-entry RMS scale, so that the second-order term can be combined with the first-order displacement without being dominated by channel dimensionality.

## 5. Final ROBUST Metric

For a feature-flow pair $(i,j)$, define the paired first-order displacement:

$$
D^{(1)}_{i,j}
=
\frac{1}{2}
\left(D_i^{(1)}+D_j^{(1)}\right).
$$

The proposed ROBUST metric for this pair is:

$$
D^{\mathrm{ROBUST}}_{i,j}
=
\widetilde{D}^{(2)}_{i,j}
+
\lambda D^{(1)}_{i,j}.
$$

The final score averages over selected feature-flow pairs:

$$
\mathrm{ROBUST}(x,x_{\mathrm{adv}};\theta)
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
D^{\mathrm{ROBUST}}_{i,j}.
$$

For a model-level score over a dataset $\mathcal{D}$:

$$
\mathrm{ROBUST}(\theta)
=
\frac{1}{|\mathcal{D}|}
\sum_{(x,y)\in\mathcal{D}}
\mathrm{ROBUST}(x,x_{\mathrm{adv}};\theta).
$$

Lower values indicate that adversarial perturbations induce smaller representation-measure displacement and smaller feature-flow distortion.

## 6. Why This Metric Is Theoretically Meaningful

Let the classifier be decomposed as:

$$
f_\theta(x)
=
r_\theta(\Phi_\theta(x)),
$$

where $\Phi_\theta(x)$ collects the selected feature-measure descriptors:

$$
\Phi_\theta(x)
=
\left(
\{\nu_l^x\}_{l\in\mathcal{L}},
\{G_{i,j}(x)\}_{(i,j)\in\mathcal{P}}
\right).
$$

Assume that, locally around the data manifold, the remaining classifier map $r_\theta$ is Lipschitz with respect to the mixed descriptor discrepancy:

$$
\|f_\theta(x_{\mathrm{adv}})-f_\theta(x)\|_2
\le
L_\theta
\cdot
\mathrm{ROBUST}(x,x_{\mathrm{adv}};\theta)
+
R_\theta(x,x_{\mathrm{adv}}),
$$

where $R_\theta$ is the residual from feature information not captured by the selected first- and second-order descriptors.

Let the clean margin be:

$$
\gamma_\theta(x)
=
f_{\theta,y}(x)
-
\max_{k\ne y}f_{\theta,k}(x).
$$

If:

$$
\|f_\theta(x_{\mathrm{adv}})-f_\theta(x)\|_\infty
<
\frac{1}{2}\gamma_\theta(x),
$$

then the predicted class is unchanged. Combining the two inequalities gives the sufficient condition:

$$
L_\theta
\cdot
\mathrm{ROBUST}(x,x_{\mathrm{adv}};\theta)
+
R_\theta(x,x_{\mathrm{adv}})
<
\frac{1}{2}\gamma_\theta(x)
\quad
\Longrightarrow
\quad
\arg\max_k f_{\theta,k}(x_{\mathrm{adv}})
=
\arg\max_k f_{\theta,k}(x).
$$

This establishes the theoretical role of the metric:

$$
\text{smaller representation-measure displacement}
\Rightarrow
\text{smaller logit displacement}
\Rightarrow
\text{higher chance of preserving the clean prediction margin}.
$$

The statement is deliberately a sufficient-condition argument, not an equivalence claim. The metric does not prove robustness by itself; it provides a principled and computable proxy for the internal stability that robust models should exhibit.

## 7. Relationship to FSP

Strict FSP uses only:

$$
\left\|
G_{i,j}(x_{\mathrm{adv}})
-
G_{i,j}(x)
\right\|_F^2 .
$$

Our metric includes this feature-flow idea but adds a first-order transport term:

$$
\underbrace{\widetilde{D}^{(2)}_{i,j}}_{\text{feature-flow stability}}
+
\underbrace{\lambda D^{(1)}_{i,j}}_{\text{local feature-measure stability}}.
$$

This matters because two models can have similar FSP displacement while having different local atom displacement. FSP sees cross-layer co-activation changes, but it does not directly penalize spatially localized shifts of normalized feature atoms. The ROBUST metric therefore keeps FSP's strength while covering a failure mode of second-order-only metrics.

## 8. Research Significance

The proposed metric is meaningful for RobustNAS and robust architecture analysis for four reasons:

1. It is representation-level: it measures where adversarial perturbation propagates inside the network, not only whether the final prediction changes.
2. It is GAP-free: it avoids collapsing feature maps into a single global average vector.
3. It is measure-theoretic: the first-order term is tied to Wasserstein transport under spatial coupling, and the second-order term is an IPM over bilinear feature-flow probes.
4. It is architecture-sensitive: feature-flow stability depends on how layers transform and preserve representations, making it suitable for comparing robust architectures.

## 9. Current Testable Hypotheses

The theory suggests the following empirical hypotheses:

1. Models with lower ROBUST metric should have higher adversarial accuracy under strong white-box attacks.
2. In robust-only model comparisons, ROBUST should rank architectures better than strict FSP because it adds local feature-measure displacement.
3. Deep-layer feature-flow terms should be important, but first-order local displacement should improve robustness ranking when FSP alone becomes insensitive.
4. The metric should remain useful under shared adversarial examples, but white-box adversarial examples are the first controlled setting.

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083). arXiv:1706.06083.
- Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573). ICML 2019.
- Guo, M., Yang, Y., Xu, R., Liu, Z., & Lin, D. [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695). CVPR 2020.

