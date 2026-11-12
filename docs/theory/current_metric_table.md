# Current Robust Metric Table

Date: 2026-07-03

Readable Chinese formula version: `docs/theory/current_metric_table_readable_zh.md`

Browser-rendered MathJax formula version: `docs/current_metric_table_math.html`

Experiment setting for the reported results:

- Dataset: CIFAR10 test split
- Models: 10 RobustBench CIFAR10/Linf models
- Protocol: white-box adversarial examples generated separately for each model
- Attack: PGD-lite, 3 steps
- Samples: 512
- Target: adversarial accuracy `adv_acc`
- Reported number: Spearman correlation with `adv_acc`

`All` uses all 10 models. `Robust-only` excludes the naturally trained `Standard` model.

| Type | Metric name | Paper / source | Mathematical principle | Formula used here | All | Robust-only | Current interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Ours | `feature_delta` | Proposed in this project; inspired by feature-level robustness and `Feature Denoising for Improving Adversarial Robustness` | Robust models should keep intermediate feature representations stable under adversarial perturbation. | `mean_l ||norm(pool(F_l(x_adv))) - norm(pool(F_l(x)))||_2` | -0.6970 | -0.5833 | Useful; weaker than FSP and loss-delta baselines in this run. |
| Ours | `perturbation_capacity_delta` | Proposed in this project; inspired by activation perturbation accumulation / CAS-style analysis | Robust models should not amplify centered activation capacity under small input perturbations. | `mean_l |C_l(x_adv)-C_l(x)|/(|C_l(x)|+eps)` | -0.3939 | -0.1667 | Weak standalone metric so far. |
| Output stability | `logit_kl` | Related to TRADES, `Theoretically Principled Trade-off between Robustness and Accuracy` | Clean and adversarial predictive distributions should be consistent. | `KL(softmax(f(x)) || softmax(f(x_adv)))` | -0.7455 | -0.6500 | Strong and simple output-space baseline. |
| ARNAS objective proxy | `natural_loss` | `Towards Accurate and Robust Architectures via Neural Architecture Search` | Accurate architecture search should preserve natural-data performance. | `CE(f(x), y)` | 0.3939 | 0.1667 | Not a robustness metric by itself; included because ARNAS jointly optimizes natural and adversarial objectives. |
| ARNAS objective proxy | `adversarial_loss` | `Towards Accurate and Robust Architectures via Neural Architecture Search` | Robust architecture search should reduce adversarial loss. | `CE(f(x_adv), y)` | -0.5273 | -0.3500 | Directionally useful but weaker than the loss-increase metric. |
| Input loss landscape | `input_loss_delta` | `AdvRush: Searching for Adversarially Robust Neural Architectures`; also related to AWP input-loss-landscape discussion | Robust models should have a flatter input loss landscape; adversarial perturbation should not sharply increase loss. | `CE(f(x_adv), y) - CE(f(x), y)` | -0.8182 | -0.7500 | Strong scalar baseline; more informative than raw adversarial loss here. |
| Jacobian / gradient | `input_grad_l2` | Jacobian regularization literature; DSRNA-style Jacobian norm bound | Smaller input gradient implies lower local sensitivity. | `mean_i ||d CE(f(x_i),y_i)/d x_i||_2` | -0.7818 | -0.7000 | Strong baseline in current experiment. |
| Lipschitz proxy | `local_logit_lipschitz` | `Adversarially Robust Neural Architectures`; Parseval/Lipschitz robustness literature | Smaller local Lipschitzness means less output amplification under input perturbation. | `||f(x_adv)-f(x)||_2 / (||x_adv-x||_2+eps)` | -0.6727 | -0.5500 | Fixed-model proxy only; not the exact architecture-level Lipschitz regularizer in the paper. |
| Certified-bound proxy | `margin_input_grad_ratio` | DSRNA, `Differentiable Search of Robust Neural Architectures` | Larger margin relative to local sensitivity suggests larger local certified radius. | `max(f_y(x)-max_{k!=y}f_k(x),0)/(||d CE/dx||_2+eps)` | -0.7818 | -0.7000 | Empirically negative here because the proxy is affected by clean confidence and attack setting; needs refinement before use as a higher-is-better certificate proxy. |
| FSP / feature-flow | `fsp_delta` | RobNet / `When NAS Meets Robustness`; FSP originally from `A Gift from Knowledge Distillation` | Robust architectures should preserve feature-flow / Gram structure under perturbation. | `mean_l ||G_l(x_adv)-G_l(x)||_F/(||G_l(x)||_F+eps)` | -0.8909 | -0.8500 | Best current baseline. Note: current implementation is FSP-inspired same-layer Gram stability, not exact cross-layer RobNet FSP loss. |
| Method-level mandatory baseline | Architecture-level Lipschitz regularization | `Adversarially Robust Neural Architectures` | Architecture parameters can be optimized to reduce Lipschitz-based perturbation amplification. | Not implemented exactly; current proxy is `local_logit_lipschitz`. | -0.6727 proxy | -0.5500 proxy | Must compare in the search-stage experiment; current RobustBench fixed-model experiment only gives a proxy. |
| Method-level mandatory baseline | Accurate/robust multi-objective NAS | `Towards Accurate and Robust Architectures via Neural Architecture Search` | Search should jointly optimize natural performance and adversarial robustness. | Current proxies: `natural_loss`, `adversarial_loss`, `input_loss_delta`; exact ARNAS search method not implemented. | see above | see above | Must compare as a method-level baseline when NAS/search begins. |

## Key Ranking In This Run

Robust-only Spearman ranking:

1. `fsp_delta`: -0.8500
2. `input_loss_delta`: -0.7500
3. `input_grad_l2`: -0.7000
4. `margin_input_grad_ratio`: -0.7000
5. `logit_kl`: -0.6500
6. `feature_delta`: -0.5833
7. `local_logit_lipschitz`: -0.5500
8. `adversarial_loss`: -0.3500
9. `perturbation_capacity_delta`: -0.1667
10. `natural_loss`: 0.1667

The current best empirical baseline is `fsp_delta`. The most important loss-landscape baseline is `input_loss_delta`. The two required high-quality papers should be treated as method-level baselines, not merely scalar metrics.
