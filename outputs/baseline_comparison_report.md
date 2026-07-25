# Baseline Metric Comparison

Date: 2026-07-03

## Source

The baseline metrics are selected from section 2.5.1 of `F:\OneDrive\work\研究\Idea\当前读书笔记与灵感.docx`, especially the notes around:

- Lipschitz constant
- input loss landscape
- certified lower bound / Jacobian norm bound
- FSP matrix as a robustness indicator

## Experiment

- Protocol: white-box metric evaluation
- Dataset: CIFAR10 test set
- Models: 10 RobustBench CIFAR10/Linf models
- Attack: PGD-lite, 3 steps
- Samples: 512
- Server: Chengdu 516 4090 docker

## Correlation With Adversarial Accuracy

All 10 models:

| Metric | Family | Spearman | Kendall tau-a |
| --- | --- | ---: | ---: |
| `fsp_delta` | FSP / feature-flow stability | -0.9019 | -0.8095 |
| `input_loss_delta` | input loss landscape | -0.7178 | -0.6190 |
| `input_grad_l2` | Jacobian/input-gradient norm | -0.6810 | -0.5238 |
| `feature_delta` | proposed feature stability | -0.6197 | -0.4286 |
| `logit_kl` | output stability | -0.6074 | -0.5238 |
| `margin_input_grad_ratio` | margin-gradient proxy | -0.5951 | -0.4762 |
| `local_logit_lipschitz` | local Lipschitz proxy | -0.5583 | -0.4286 |
| `perturbation_capacity_delta` | proposed activation capacity shift | -0.2884 | -0.2381 |

Robust models only, excluding naturally trained `Standard`:

| Metric | Family | Spearman | Kendall tau-a |
| --- | --- | ---: | ---: |
| `fsp_delta` | FSP / feature-flow stability | -0.8645 | -0.7576 |
| `input_loss_delta` | input loss landscape | -0.6103 | -0.5152 |
| `feature_delta` | proposed feature stability | -0.4746 | -0.2727 |
| `logit_kl` | output stability | -0.4577 | -0.3939 |
| `margin_input_grad_ratio` | margin-gradient proxy | -0.4407 | -0.3333 |
| `local_logit_lipschitz` | local Lipschitz proxy | -0.3899 | -0.2727 |
| `input_grad_l2` | Jacobian/input-gradient norm | -0.5594 | -0.3939 |
| `perturbation_capacity_delta` | proposed activation capacity shift | -0.0170 | -0.0303 |

## Takeaways

`fsp_delta` is the strongest baseline in this smoke test. It remains highly correlated with adversarial accuracy even after excluding the naturally trained `Standard` model, so it is not only detecting the natural-vs-robust gap.

`input_loss_delta` is also a strong and cheap baseline. It directly reflects input loss landscape smoothness and is a good comparison target for the proposed feature-space metric.

`feature_delta` remains directionally useful, but in this first comparison it is weaker than `fsp_delta` and `input_loss_delta`. This suggests the next design iteration should either improve layer weighting or combine feature perturbation with feature-flow information.

`perturbation_capacity_delta` is weak in this setting and should not be treated as a standalone metric yet.

Shared-adversarial metric evaluation is intentionally deferred.
