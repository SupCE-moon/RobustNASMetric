# NAS-RobBench-201 5/10-Epoch Fidelity Validation

## Executive Conclusion

This experiment does **not** validate the current `perturbation_capacity_*` family as a lower-is-better early-fidelity RobustNAS proxy.

On 512 NAS-Bench-201 architectures, all three capacity variants correlate with official NAS-RobBench-201 AutoAttack accuracy in the wrong direction. The original `perturbation_capacity_delta (Ours)` has Spearman `+0.506` at epoch 5 and `+0.350` at epoch 10. Because its theoretical direction was fixed as lower-is-better before observing NAS-RobBench, its direction-aligned Spearman values are `-0.506` and `-0.350`.

The best directionally valid metrics are `feature_measure_mmd` with Spearman `-0.565/-0.635`, followed by `adversarial_loss [10]` with `-0.331/-0.524`. Under direct random-proxy selection, 10-epoch `adversarial_loss` selects architectures with mean official AutoAttack accuracy `0.428`, close to the best value `0.444` in the 512-architecture subset.

This is a negative result for the present early-fidelity capacity formulation, not a contradiction of the 20 fully trained RobustBench model result. It shows that a metric useful for comparing trained defense models is not automatically a useful proxy on undertrained NAS candidates.

## Protocol

| Item | Value |
|---|---|
| Benchmark | NAS-RobBench-201, CIFAR-10 |
| Candidate subset | 512 unique architectures |
| Oracle target | Official `autoattack` robust accuracy, used only after proxy evaluation |
| Sampling | Target-stratified sampling followed by seed-fixed shuffle |
| Fidelity | Metrics from the same training trajectory at epochs 5 and 10 |
| Training data | 4096 existing CIFAR-10 training samples |
| Metric data | 256 existing CIFAR-10 test samples |
| Training | PGD-3 adversarial training, SGD, OneCycleLR, maximum LR 0.1 |
| Metric attack | White-box PGD-3, epsilon 8/255 |
| Search budget | 150 of 512 candidates per trial |
| Search repeats | 100 trials |
| Search methods | Random controls and NSGA-II, population 50, 3 generations |

The 512 selected architectures were shuffled before slicing. Their official AutoAttack means are `0.369`, `0.362`, and `0.368` across the three hardware shards, so server assignment is not tied to an ordered robustness interval.

## Correlation Ranking

All listed metrics are defined as lower-is-better. `Aligned Spearman` equals `-Spearman`; a positive value is directionally correct, while a negative value is a failed direction.

| Rank | Metric | Reference | Epoch-5 Spearman | Epoch-5 aligned | Epoch-10 Spearman | Epoch-10 aligned |
|---:|---|---|---:|---:|---:|---:|
| 1 | `feature_measure_mmd` | distribution-measure baseline | -0.565 | 0.565 | **-0.635** | **0.635** |
| 2 | `adversarial_loss` | [10] | -0.331 | 0.331 | -0.524 | 0.524 |
| 3 | `fsp_strict_delta` | [1,2] | -0.032 | 0.032 | -0.103 | 0.103 |
| 4 | `local_logit_lipschitz` | [5,6] | 0.276 | -0.276 | 0.031 | -0.031 |
| 5 | `perturbation_capacity_delta (Ours)` | Ours | 0.506 | -0.506 | 0.350 | -0.350 |
| 6 | `perturbation_capacity_relu_active_delta (Ours)` | Ours | 0.593 | -0.593 | 0.463 | -0.463 |
| 7 | `perturbation_capacity_above_mean_delta (Ours)` | Ours | 0.616 | -0.616 | 0.488 | -0.488 |
| 8 | `input_grad_l2` | [4,7] | 0.722 | -0.722 | 0.720 | -0.720 |

The ReLU and above-mean gates make capacity ranking worse, not better. The original ungated capacity is the least-bad Ours variant, and its wrong-direction correlation weakens from epoch 5 to epoch 10, but it has not crossed into the theoretically valid direction by epoch 10.

## Direct Random-Proxy Search

Each trial randomly queries 150 architectures and selects the candidate with the best proxy value in the pre-specified lower-is-better direction.

| Metric | Epoch-5 mean AutoAttack | Epoch-5 regret | Epoch-10 mean AutoAttack | Epoch-10 regret |
|---|---:|---:|---:|---:|
| `adversarial_loss` [10] | **0.419** | **0.025** | **0.428** | **0.015** |
| `feature_measure_mmd` | 0.407 | 0.036 | 0.396 | 0.048 |
| `local_logit_lipschitz` [5,6] | 0.391 | 0.053 | 0.308 | 0.136 |
| `fsp_strict_delta` [1,2] | 0.291 | 0.152 | 0.277 | 0.167 |
| `perturbation_capacity_relu_active_delta (Ours)` | 0.196 | 0.247 | 0.211 | 0.233 |
| `perturbation_capacity_delta (Ours)` | 0.188 | 0.256 | 0.206 | 0.238 |
| `perturbation_capacity_above_mean_delta (Ours)` | 0.191 | 0.252 | 0.191 | 0.253 |
| `input_grad_l2` [4,7] | 0.187 | 0.257 | 0.188 | 0.256 |

The capacity result is not merely a weak correlation. Selecting architectures with the smallest capacity drift actively chooses low-robustness candidates in this early-training regime.

## Multi-Objective Search

Both `random_proxy_clean` and `nsga2_proxy_clean` optimize the proxy and same-run early `clean_acc`, then apply the same Pareto-front recommendation. Their difference therefore reflects candidate generation, not a different final selector.

| Metric | Random Pareto E5 | NSGA-II E5 | Random Pareto E10 | NSGA-II E10 |
|---|---:|---:|---:|---:|
| `fsp_strict_delta` [1,2] | 0.411 | **0.414** | 0.389 | 0.397 |
| `feature_measure_mmd` | 0.407 | 0.409 | **0.411** | 0.410 |
| `adversarial_loss` [10] | **0.404** | 0.400 | 0.425 | **0.425** |
| `local_logit_lipschitz` [5,6] | 0.350 | 0.356 | 0.353 | 0.363 |
| `perturbation_capacity_delta (Ours)` | **0.333** | 0.326 | **0.403** | 0.402 |
| `perturbation_capacity_relu_active_delta (Ours)` | 0.293 | 0.296 | **0.409** | 0.399 |
| `perturbation_capacity_above_mean_delta (Ours)` | 0.287 | 0.296 | **0.403** | 0.389 |
| `input_grad_l2` [4,7] | 0.274 | 0.280 | 0.266 | 0.277 |

Matched controls:

| Control | Epoch-5 mean AutoAttack | Epoch-10 mean AutoAttack |
|---|---:|---:|
| `random_uniform` | 0.361 | 0.361 |
| `random_clean` | 0.379 | **0.422** |

At epoch 5, FSP and MMD plus clean accuracy improve substantially over clean-only selection. At epoch 10, early clean accuracy alone is already very strong; `adversarial_loss` adds only about `0.003`. NSGA-II does not consistently beat the matched random Pareto baseline, so this run does not establish a general evolutionary-search advantage.

## Why Capacity Reverses Direction

The most plausible mechanism is early-training collapse and representation maturity confounding:

1. A weak or slowly learning architecture can remain close to an uninformative representation. Both clean and adversarial feature maps then have little spatial structure, so their absolute capacity difference is small.
2. A stronger architecture learns nontrivial spatial features earlier. Those features have more energy and structure available to move under PGD, producing a larger absolute capacity difference even when the final architecture is more robust after full adversarial training.
3. The current metric therefore mixes perturbation instability with feature scale, activation richness, and optimization progress. In fully trained RobustBench defenses, this confounding is smaller; among 5/10-epoch NAS candidates it dominates.

This also explains why flipping the sign after seeing NAS-RobBench would be scientifically wrong. A higher-is-better capacity score would be an empirical trainability or representation-maturity proxy, not the lower-is-better perturbation-instability quantity derived in the theory.

## Cross-GPU Audit

Architecture `13675` was recomputed on both server types and repeated on the 516 GPU. Relative differences are of comparable scale for most metrics, indicating that ordinary early-training nondeterminism is at least as important as the hardware change.

| Metric | Cross-server E5 | Same-516 repeat E5 | Cross-server E10 | Same-516 repeat E10 |
|---|---:|---:|---:|---:|
| `perturbation_capacity_delta (Ours)` | 2.42% | 0.44% | 3.48% | 2.20% |
| `perturbation_capacity_relu_active_delta (Ours)` | 2.97% | 1.64% | 2.53% | 1.63% |
| `perturbation_capacity_above_mean_delta (Ours)` | 6.14% | 5.09% | 0.64% | 2.99% |
| `adversarial_loss` [10] | 1.72% | 1.75% | 0.56% | 0.65% |
| `feature_measure_mmd` | 0.66% | 0.10% | 0.39% | 0.14% |
| `fsp_strict_delta` [1,2] | 20.6% | 11.8% | 4.52% | 4.89% |

The capacity sign reversal is much larger than these numerical differences and occurs in every shuffled shard. It cannot reasonably be attributed to one server receiving only weak or strong architectures. For a paper-quality final table, however, multi-seed reruns on one hardware type are still required.

## Research Decision

The current evidence supports four decisions:

1. Do not use the present absolute `perturbation_capacity_delta` as the primary 5/10-epoch RobustNAS search objective.
2. Keep the 20-model RobustBench result as evidence about trained-model ranking, with an explicit scope statement.
3. Develop a scale- and maturity-controlled capacity metric, such as layer-wise relative drift normalized by clean capacity, activation count, and feature energy, with coefficients fixed without consulting NAS-RobBench oracle targets.
4. Validate the revised metric first by residual analysis after controlling for early clean accuracy, training loss, and feature energy. Only then repeat random search and NSGA-II.

## Reproducibility Files

Raw server outputs and logs:

```text
outputs/nasrobbench201_server/fidelity512_e5e10/nb201_fidelity512_e5e10_seed20260707_shard000_171.csv
outputs/nasrobbench201_server/fidelity512_e5e10/nb201_fidelity512_e5e10_seed20260707_shard171_342.csv
outputs/nasrobbench201_server/fidelity512_e5e10/nb201_fidelity512_e5e10_seed20260707_shard342_512.csv
outputs/nasrobbench201_server/fidelity512_e5e10/*.log
```

Validated derived outputs:

```text
outputs/nasrobbench201_server/fidelity512_e5e10/nb201_fidelity512_e5e10_seed20260707.csv
outputs/nasrobbench201_server/fidelity512_e5e10/correlations.csv
outputs/nasrobbench201_server/fidelity512_e5e10/search_trials.csv
outputs/nasrobbench201_server/fidelity512_e5e10/search_summary.csv
```

The merged CSV contains exactly 1024 rows: 512 unique architectures at epochs 5 and 10, with no duplicate `(arch_id, train_epochs)` key.

## References

[1] Dong et al., `When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks`.

[2] Yim et al., `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning`.

[4] Jakubovitz and Giryes, `Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization`.

[5] Dong et al., `Adversarially Robust Neural Architectures`.

[6] Cisse et al., `Parseval Networks: Improving Robustness to Adversarial Examples`.

[7] DSRNA, `Differentiable Search of Robust Neural Architectures`.

[10] Ou, Feng, and Sun, `Towards Accurate and Robust Architectures via Neural Architecture Search`.
