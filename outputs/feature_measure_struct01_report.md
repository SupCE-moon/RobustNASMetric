# Feature Measure Struct01 Report

Date: 2026-07-03

## Goal

Optimize the proposed `feature_delta` metric by removing the weak global-average-pooling first-order term and learning from the strength of `fsp_delta`.

## New Metrics

### feature_measure_local_delta

GAP-free local feature atom displacement:

```text
a_s(x) = F_l(x)[:, h, w]
u_s(x) = a_s(x) / (||a_s(x)||_2 + eps)

feature_measure_local_delta_l
  = average_s ||u_s(x_adv) - u_s(x)||_2
```

### feature_measure_weighted_delta

Energy-weighted local displacement:

```text
w_s(x) = ||a_s(x)||_2 / sum_t ||a_t(x)||_2

feature_measure_weighted_delta_l
  = sum_s w_s(x) * ||u_s(x_adv) - u_s(x)||_2
```

### feature_measure_struct01_delta

Current preferred GAP-free structural metric:

```text
d1_l = feature_measure_weighted_delta_l
d2_l = ||G_l(x_adv) - G_l(x)||_F / (||G_l(x)||_F + eps)

feature_measure_struct01_delta_l
  = sqrt(0.1 * d1_l^2 + d2_l^2)
```

Lower is better.

## Result

Experiment:

- 10 RobustBench CIFAR10/Linf models
- 512 CIFAR10 test samples
- White-box PGD-lite, 3 steps
- Chengdu 516 4090 docker

| Metric | All Spearman | Robust-only Spearman |
| --- | ---: | ---: |
| old `feature_delta` | -0.7052 | -0.5941 |
| `feature_measure_local_delta` | -0.7173 | -0.6109 |
| `feature_measure_weighted_delta` | -0.7173 | -0.6109 |
| `feature_measure_struct_delta` | -0.8571 | -0.8034 |
| `feature_measure_struct01_delta` | -0.8875 | -0.8452 |
| `fsp_delta` | -0.8875 | -0.8452 |

## Interpretation

The GAP-free local feature atom displacement improves over old `feature_delta`, but it is not strong enough alone. The second-order channel-spatial structure term remains the dominant robustness signal.

The best current proposed version is:

```text
feature_measure_struct01_delta
```

It preserves the measure-theoretic interpretation:

```text
adversarial perturbation induces feature-measure displacement;
the metric captures local atom displacement plus second-order structure displacement.
```

The small coefficient `0.1` is important. Equal-weight combination makes the local term too influential and weakens ranking.

## Caveat

These smoke experiments use PGD-lite with random start. The next reproducibility step should add a fixed random seed before comparing close variants.
