# RobustBench White-Box Attack Parameter Ablation

Date: 2026-07-18

## Question

Test whether the large gap between the RobustBench-20 capacity result and the NAS-RobBench-201 E17 result is primarily caused by using a multi-step PGD metric attack instead of the more local and directly interpretable FGSM attack.

## Matched Protocol

- Models: the same 20 CIFAR-10 RobustBench checkpoints.
- Target: RobustBench official robust accuracy.
- Samples: the same seed-fixed 64-image CIFAR-10 subset.
- Threat model: white-box $L_\infty$, $\epsilon=8/255$.
- Metric: unchanged `perturbation_capacity_delta (Ours)` implementation.
- Changed factor: FGSM versus PGD-3 versus PGD-10.

The existing three-source score of $-0.565$ is not an FGSM result. It averages metric values from three shared PGD-10 caches. The directly comparable existing white-box FGSM score is $-0.626$.

## Results

| White-box attack | Pearson | Spearman | Kendall $\tau_b$ | Capacity mean | Capacity CV |
|---|---:|---:|---:|---:|---:|
| FGSM | -0.881 | **-0.626** | -0.432 | 0.0408 | 35.7% |
| PGD-3 | **-0.934** | -0.621 | **-0.474** | 0.0282 | 51.6% |
| PGD-10 | -0.929 | -0.573 | -0.432 | 0.0437 | 51.9% |

PGD-3 is essentially tied with FGSM in rank correlation. PGD-10 weakens Spearman by 0.053 relative to FGSM but remains substantially stronger than the stopped 91-architecture E17 epoch-50 value of $-0.184$.

For reference, the epoch-50 NAS-Bench-201 PGD-3 capacity values have a coefficient of variation of only 13.1%, compared with 35.7%--51.9% across the heterogeneous RobustBench models. The NAS architecture family therefore provides a much narrower capacity signal even under the same PGD-3 attack family.

## Interpretation

The hypothesis that iterative PGD alone destroys the capacity metric is not supported. FGSM is mathematically more local: its endpoint is determined by the loss gradient at the clean input, while PGD follows a path through several activation regions. This makes FGSM easier to connect to a local Jacobian argument. Empirically, however, capacity ranking remains strong under PGD-3 and reasonably strong under PGD-10 on the same RobustBench models.

The larger RobustBench-to-NAS gap is more plausibly explained by:

1. heterogeneous RobustBench architectures and defense recipes versus a narrow C=16, N=5 NAS-Bench-201 family;
2. compressed capacity variation across the NAS candidates;
3. identical adversarial-training protocol across NAS candidates;
4. NAS-RobBench-201 official clean validation accuracy and AutoAttack accuracy having Spearman $0.982$ across all 6466 architectures;
5. model-family and layer-comparability effects that are not removed by the present normalization.

The proper future NAS attack ablation is still FGSM/PGD-3/PGD-10 on the exact same saved NAS checkpoints and test images. E17 cannot provide that matched comparison because its model weights were not saved.

## Reproducibility Files

```text
outputs/metrics_20_whitebox_fgsm_n64_relu_with_rb_scores.csv
outputs/metrics_20_whitebox_pgd3_n64_attack_ablation_with_rb_scores.csv
outputs/metrics_20_whitebox_pgd10_n64_attack_ablation_with_rb_scores.csv
outputs/correlations/e18_whitebox_pgd3_n64_attack_ablation_vs_rb.csv
outputs/correlations/e18_whitebox_pgd10_n64_attack_ablation_vs_rb.csv
```

