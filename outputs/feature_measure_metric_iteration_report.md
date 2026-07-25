# Feature Measure Metric Iteration Report

Date: 2026-07-03

## Motivation

The original `feature_delta` used global average pooling:

```text
feature_delta = ||Norm(GAP(F_l(x_adv))) - Norm(GAP(F_l(x)))||_2
```

This can weaken the signal because `GAP` collapses the spatial feature distribution into a single mean vector. The user pointed out this weakness, and the experiment confirms that the stronger direction is to preserve feature-space structure.

## New Metric Candidates

### 1. feature_measure_swd

Treat each layer feature map as an empirical measure over spatial feature atoms and compare clean/adversarial measures by sliced Wasserstein distance.

Result:

```text
Robust-only Spearman = 0.1423
```

This failed. Generic optimal-transport distance over projected feature atoms did not capture the robustness-relevant structure.

### 2. feature_measure_mmd

Compare empirical feature measures by RBF-kernel mean embedding / MMD.

Result:

```text
Robust-only Spearman = -0.4770
```

This is better than SWD but still not stronger than the old `feature_delta`.

### 3. feature_measure_cov_delta

Compare covariance operators of the feature-space empirical measures.

Result:

```text
Robust-only Spearman = 0.4268
```

This also failed. Plain covariance of spatial feature atoms is not the same signal as FSP.

### 4. feature_measure_moment_delta

Current main candidate. It combines:

1. first-order feature response shift;
2. second-order channel-spatial pattern shift.

Layer formula:

```text
d_1,l = ||Norm(GAP(F_l(x_adv))) - Norm(GAP(F_l(x)))||_2

d_2,l = ||G_l(x_adv) - G_l(x)||_F / (||G_l(x)||_F + eps)

feature_measure_moment_delta
  = average_l sqrt(d_1,l^2 + d_2,l^2)
```

where `G_l` is a normalized channel-spatial Gram operator.

Result:

```text
All Spearman         = -0.9483
Robust-only Spearman = -0.9289
```

This is the best current proposed metric.

## Comparison

| Metric | All Spearman | Robust-only Spearman |
| --- | ---: | ---: |
| `feature_delta` | -0.6201 | -0.4770 |
| `feature_measure_swd` | -0.1702 | 0.1423 |
| `feature_measure_mmd` | -0.1702 | -0.4770 |
| `feature_measure_cov_delta` | 0.0365 | 0.4268 |
| `feature_measure_moment_delta` | -0.9483 | -0.9289 |
| `fsp_delta` | -0.9483 | -0.9289 |

## Interpretation

The key lesson is that avoiding `GAP` alone is not enough. A robust metric needs to preserve the feature-space structure that adversarial perturbations actually disrupt.

The successful metric is the one that combines:

```text
first-order feature response displacement
+ second-order channel-spatial pattern displacement
```

This is consistent with the measure-theoretic view:

```text
adversarial perturbation changes the feature-space measure;
robust models resist both first-order and second-order measure shifts;
these shifts propagate through the network and eventually cause output changes.
```

## Caveat

`feature_measure_moment_delta` currently has the same rank correlation as `fsp_delta`, which means the second-order term is probably dominating. The next refinement should make the first-order term more meaningful without returning to pure `GAP`.

Candidate next step:

```text
replace GAP first-order term with spatially weighted feature measure mean
or class-sensitive feature-measure displacement.
```
