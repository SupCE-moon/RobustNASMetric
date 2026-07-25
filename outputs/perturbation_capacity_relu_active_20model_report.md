# ReLU-Active Perturbation Capacity: 20-Model Validation

Date: 2026-07-16

## Hypothesis

The original `perturbation_capacity_delta (Ours)` measures the relative clean-to-adversarial change of centered feature activation capacity. This experiment tests whether restricting that capacity to clean ReLU-active entries improves model-robustness ranking.

For sample $b$, selected layer $l$, channel $c$, and spatial position $s$, let

$$
M_{b,l,s,c}(x)
=
\mathbf{1}\!\left[a_{b,l,s,c}(x)>0\right],
\qquad
N_{b,l,c}^{+}(x)
=
\sum_{s=1}^{S_l}M_{b,l,s,c}(x).
$$

The clean mask is fixed for both members of the clean/adversarial pair. For $z\in\{x,x_{\mathrm{adv}}\}$, define the clean-gated channel mean

$$
\bar a_{b,l,c}^{+}(z\mid x)
=
\frac{
\sum_{s=1}^{S_l}M_{b,l,s,c}(x)a_{b,l,s,c}(z)
}{
\max\!\left(N_{b,l,c}^{+}(x),1\right)
},
$$

and ReLU-active capacity

$$
C_{b,l}^{+}(z\mid x)
=
\sum_{c=1}^{C_l}
\frac{
\sum_{s=1}^{S_l}M_{b,l,s,c}(x)
\left|a_{b,l,s,c}(z)-\bar a_{b,l,c}^{+}(z\mid x)\right|
}{
\max\!\left(N_{b,l,c}^{+}(x),1\right)
}.
$$

The proposed ablation is

$$
D_l^{\mathrm{capacity\text{-}ReLU}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\left|C_{b,l}^{+}(x_{\mathrm{adv}}\mid x)-C_{b,l}^{+}(x\mid x)\right|
}{
\left|C_{b,l}^{+}(x\mid x)\right|+\varepsilon
},
$$

followed by the mean over selected layers. The CSV field is `perturbation_capacity_relu_active_delta`.

Using the same clean mask for both terms creates a paired comparison on a fixed support. Independently recomputing the adversarial mask would additionally measure ReLU state transitions and is therefore a different ablation, not the metric tested here.

## Protocol

- Dataset and models: the same 20 RobustBench CIFAR-10 $L_\infty$ models as E03-E05 and E13.
- Target: RobustBench official robust accuracy.
- White-box: per-target-model FGSM, 64 samples.
- Shared-adv: Standard, Engstrom2019Robustness, and Wang2023Better_WRN-28-10 PGD-10 caches, 128 samples each.
- Three-source result: average each target model's metric value over the three caches, then compute correlation.
- Expected direction: lower is better.
- Execution: Chengdu 516 RTX 4090 docker, `/root/robust_metric_lab`.
- Existing CIFAR-10 data, 20 model checkpoints, and three caches were reused; no dataset or model was downloaded.

## Results

| Protocol | Metric | Pearson | Spearman | Kendall tau-a | Spearman rank among 35 |
| --- | --- | ---: | ---: | ---: | ---: |
| White-box FGSM | `perturbation_capacity_delta (Ours)` | -0.881 | -0.626 | -0.432 | 3 |
| White-box FGSM | `perturbation_capacity_relu_active_delta (Ours)` | **-0.956** | **-0.699** | **-0.526** | **2** |
| Standard shared-adv | `perturbation_capacity_delta (Ours)` | -0.934 | **-0.454** | **-0.337** | 3 |
| Standard shared-adv | `perturbation_capacity_relu_active_delta (Ours)` | -0.913 | 0.018 | 0.042 | 17 |
| Engstrom shared-adv | `perturbation_capacity_delta (Ours)` | -0.785 | **-0.583** | **-0.442** | 2 |
| Engstrom shared-adv | `perturbation_capacity_relu_active_delta (Ours)` | **-0.849** | -0.421 | -0.316 | 13 |
| Wang2023 shared-adv | `perturbation_capacity_delta (Ours)` | -0.628 | **-0.574** | **-0.421** | 1 |
| Wang2023 shared-adv | `perturbation_capacity_relu_active_delta (Ours)` | **-0.788** | -0.405 | -0.326 | 4 |
| Three-source aggregate | `perturbation_capacity_delta (Ours)` | **-0.942** | **-0.565** | **-0.421** | **1** |
| Three-source aggregate | `perturbation_capacity_relu_active_delta (Ours)` | -0.922 | -0.392 | -0.316 | 7 |

For the primary three-source protocol, ReLU gating changes Spearman from $-0.565$ to $-0.392$, a reduction of $0.173$ in direction-adjusted ranking score. Across white-box, Standard, and three-source protocols, the mean direction-adjusted score changes from $0.548$ for the original metric to $0.358$ for the ReLU-active variant.

## Interpretation

The ReLU-active perturbation-capacity idea succeeds under white-box evaluation but does not improve the more architecture-oriented shared-adversarial evaluation. The Standard-source result is especially unstable: Pearson remains strongly negative while Spearman becomes slightly positive, so the relation is approximately monotone-failing despite a strong linear trend driven by scale.

The likely mechanism is that hard gating removes information from negative pre-activation entries and from clean-inactive entries that cross the ReLU boundary under attack. Those state transitions may themselves contain robustness information. A clean-derived binary support can also become overly dependent on how a shared source perturbation aligns with each target architecture's active set.

Therefore:

1. retain `perturbation_capacity_delta` as the current primary Ours metric;
2. retain `perturbation_capacity_relu_active_delta` as a useful ablation, not a replacement;
3. do not claim that ReLU gating universally improves perturbation capacity;
4. test a separate state-transition-aware variant before abandoning the ReLU motivation entirely.

This is an exploratory comparison on the same 20 models used for metric development. It supports a design decision but is not held-out confirmation.

## Reproducibility Files

- Raw and score-joined CSVs: `outputs/metrics_20_*_capacity_relu_*.csv`.
- Correlations: `outputs/correlations/e14_*_capacity_relu_*.csv`.
- Server log: `outputs/perturbation_capacity_relu_active_20model_full.log`.
- Implementation: `src/robust_metric_lab/metrics.py`.
- Focused tests: `tests/test_relu_active_capacity.py`.
