# Mandatory Paper Comparisons

Date: 2026-07-03

These papers are mandatory comparison targets for the RobustNAS metric project.

## 1. Adversarially Robust Neural Architectures

Citation key: Dong et al., `Adversarially Robust Neural Architectures`.

Why it matters:

- It is one of the cleanest works connecting neural architecture and adversarial robustness.
- Its core idea is not merely to evaluate robustness after training, but to make architecture parameters influence a Lipschitz-based robustness objective.
- The paper argues that perturbation effects can grow through architecture depth/cells, so architecture design can directly affect robustness.

Relevant principle:

```text
robust architecture -> smaller architecture-induced Lipschitzness -> less perturbation amplification
```

Comparison target:

- Exact comparison in search stage: reproduce or approximate its architecture-level Lipschitz regularization.
- E19 fixed-architecture terms: `racl_arch_lipschitz_log_upper` for the architecture-dependent part of Eq. (10), and `racl_input_gradient_lower` for the Eq. (16) input-gradient term.
- E19 NAS-RobBench-201 result: on 512 random-initialized architectures, the two terms have AutoAttack Spearman `-0.163` and `-0.603`; after 5/10 low-fidelity epochs, the gradient term flips to `+0.744/+0.812` and is invalid as an isolated lower-is-better ranker.
- Limitation: fixed discrete architectures do not contain RACL's learned log-normal distributions over $\alpha$ and $\beta$. The full $\mu$, $\sigma^2$, confidence constraint, and search algorithm remain method-level baselines and have not been reproduced by E19.

## 2. Towards Accurate and Robust Architectures via Neural Architecture Search

Citation key: Ou, Feng, Sun, `Towards Accurate and Robust Architectures via Neural Architecture Search`, CVPR 2024.

Why it matters:

- It is recent and directly aligned with accurate + robust architecture search.
- It treats accuracy and robustness as coupled objectives rather than using a single scalar robustness regularizer.
- It is a high-quality baseline for any future RobustNAS experiment.

Relevant principle:

```text
robust architecture search should jointly optimize natural performance and adversarial performance
```

Comparison target:

- Search-stage baseline: ARNAS-style accurate/robust search space and multi-objective optimization.
- Metric-stage proxy: report `natural_loss`, `adversarial_loss`, and optionally gradient-conflict or MGDA-style descent compatibility between them.
- Limitation: this is a method-level baseline, not just a scalar metric.

## Experimental Implication

The metric paper should separate baselines into two layers:

1. Scalar metric baselines:
   - `input_loss_delta`
   - `input_grad_l2`
   - `local_logit_lipschitz`
   - `fsp_delta`
   - `logit_kl`

2. Method-level RobustNAS baselines:
   - `Adversarially Robust Neural Architectures`
   - `Towards Accurate and Robust Architectures via Neural Architecture Search`
   - RobNet / FSP-based robust architecture search
   - DSRNA / Jacobian-bound robust architecture search

E19 now evaluates fixed-architecture counterparts of two RACL terms on NAS-RobBench-201. It does not reproduce the full RACL supernet optimization, so the method-level comparison in layer 2 remains future work.
