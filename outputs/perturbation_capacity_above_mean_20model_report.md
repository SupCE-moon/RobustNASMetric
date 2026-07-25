# Above-Mean Perturbation Capacity: 20-Model Validation

Date: 2026-07-16

## Motivation

The zero-threshold ReLU gate depends on each layer's absolute activation offset. This experiment replaces it with a per-sample, per-layer, per-channel relative threshold. A feature entry participates only when it exceeds its clean channel's spatial mean.

For sample $b$, layer $l$, channel $c$, and spatial position $s$, define

$$
\mu_{b,l,c}(x)
=
\frac{1}{S_l}\sum_{s=1}^{S_l}a_{b,l,s,c}(x),
$$

$$
M_{b,l,s,c}^{>\mu}(x)
=
\mathbf{1}\!\left[a_{b,l,s,c}(x)>\mu_{b,l,c}(x)\right],
\qquad
N_{b,l,c}^{>\mu}(x)
=
\sum_{s=1}^{S_l}M_{b,l,s,c}^{>\mu}(x).
$$

The clean-derived support is fixed for the clean/adversarial pair. For $z\in\{x,x_{\mathrm{adv}}\}$, its gated mean and capacity are

$$
\bar a_{b,l,c}^{>\mu}(z\mid x)
=
\frac{
\sum_s M_{b,l,s,c}^{>\mu}(x)a_{b,l,s,c}(z)
}{
\max\!\left(N_{b,l,c}^{>\mu}(x),1\right)
},
$$

$$
C_{b,l}^{>\mu}(z\mid x)
=
\sum_{c=1}^{C_l}
\frac{
\sum_s M_{b,l,s,c}^{>\mu}(x)
\left|a_{b,l,s,c}(z)-\bar a_{b,l,c}^{>\mu}(z\mid x)\right|
}{
\max\!\left(N_{b,l,c}^{>\mu}(x),1\right)
}.
$$

The metric is

$$
D_l^{\mathrm{capacity\text{-}above\text{-}mean}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{
\left|C_{b,l}^{>\mu}(x_{\mathrm{adv}}\mid x)-C_{b,l}^{>\mu}(x\mid x)\right|
}{
\left|C_{b,l}^{>\mu}(x\mid x)\right|+\varepsilon
},
$$

followed by the mean over selected layers. The CSV field is `perturbation_capacity_above_mean_delta`.

Unlike the $a>0$ gate, this support is invariant to a per-channel additive shift. Because the capacity is relative, it is also invariant to a shared positive rescaling of clean and adversarial features.

## Protocol

- Same 20 RobustBench CIFAR-10 $L_\infty$ models and official robust-accuracy target as E14.
- White-box FGSM on 64 samples.
- Standard, Engstrom2019Robustness, and Wang2023Better_WRN-28-10 shared PGD-10 caches on 128 samples.
- Three-source aggregation averages each target model's metric values before correlation.
- Expected direction: lower is better.
- Run location: Chengdu 516 RTX 4090 docker, `/root/robust_metric_lab`.
- Existing data, model checkpoints, and adversarial caches were reused.

## Correlation Results

| Protocol | Variant | Pearson | Spearman | Kendall tau-a | Rank among 36 |
| --- | --- | ---: | ---: | ---: | ---: |
| White-box | Original capacity | -0.881 | -0.626 | -0.432 | 3 |
| White-box | ReLU $>0$ | **-0.956** | **-0.699** | **-0.526** | **2** |
| White-box | Above mean | -0.948 | -0.555 | -0.442 | 4 |
| Standard shared | Original capacity | **-0.934** | **-0.454** | **-0.337** | **3** |
| Standard shared | ReLU $>0$ | -0.913 | 0.018 | 0.042 | 17 |
| Standard shared | Above mean | -0.920 | 0.027 | 0.021 | 20 |
| Engstrom shared | Original capacity | -0.785 | **-0.583** | **-0.442** | **2** |
| Engstrom shared | ReLU $>0$ | -0.849 | -0.421 | -0.316 | 14 |
| Engstrom shared | Above mean | **-0.881** | -0.535 | **-0.442** | 4 |
| Wang2023 shared | Original capacity | -0.628 | **-0.574** | **-0.421** | **1** |
| Wang2023 shared | ReLU $>0$ | -0.788 | -0.405 | -0.326 | 5 |
| Wang2023 shared | Above mean | **-0.882** | -0.442 | -0.316 | 3 |
| Three-source | Original capacity | **-0.942** | **-0.565** | **-0.421** | **1** |
| Three-source | ReLU $>0$ | -0.922 | -0.392 | -0.316 | 8 |
| Three-source | Above mean | -0.916 | -0.471 | -0.368 | 4 |

The above-mean gate improves three-source Spearman by $0.079$ over the zero-threshold gate in direction-adjusted score, but remains $0.095$ below the original unmasked capacity. Its mean direction-adjusted score over white-box, Standard, and three-source protocols is $0.333$, versus $0.548$ for the original and $0.358$ for the zero-threshold variant. The positive Standard-source Spearman is a major source-stability failure.

## Paired Model Bootstrap

The following paired bootstrap resamples the same 20 model identities 10,000 times. The reported difference is $\rho_s(\text{above mean})-\rho_s(\text{original})$; negative would favor the new lower-is-better metric.

| Protocol | Observed difference | 95% bootstrap interval |
| --- | ---: | ---: |
| White-box | 0.071 | [-0.257, 0.415] |
| Standard shared | 0.481 | [-0.050, 1.000] |
| Engstrom shared | 0.048 | [-0.308, 0.366] |
| Wang2023 shared | 0.132 | [-0.175, 0.452] |
| Three-source | 0.095 | [-0.300, 0.454] |

Every interval crosses zero. With only 20 models, the ranking difference between the original and above-mean capacity is not statistically resolved.

## Conclusion

The relative threshold is a better capacity gate than the absolute zero threshold under Engstrom, Wang2023, and three-source shared attacks. This supports the hypothesis that layer offsets and activation scales harm a fixed $a>0$ gate.

However, discarding below-mean entries still removes useful robustness information. The original all-position capacity remains stronger under every tested protocol and is substantially more source-stable. Therefore:

1. keep `perturbation_capacity_delta` as the primary Ours metric;
2. keep `perturbation_capacity_above_mean_delta` as the strongest current gated-capacity ablation;
3. do not replace the original metric or claim a significant improvement;
4. use the result to motivate a future soft, continuous weighting around the channel mean rather than another hard mask.

## Reproducibility Files

- Raw and joined CSVs: `outputs/metrics_20_*_capacity_above_mean_*.csv`.
- Correlations: `outputs/correlations/e15_*_capacity_above_mean_*.csv`.
- Bootstrap: `outputs/correlations/e15_capacity_variant_bootstrap.csv`.
- Server log: `outputs/perturbation_capacity_above_mean_20model_full.log`.
- Implementation: `src/robust_metric_lab/metrics.py`.
- Unit tests: `tests/test_relu_active_capacity.py`.
