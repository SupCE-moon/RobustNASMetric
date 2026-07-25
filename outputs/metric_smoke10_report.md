# CIFAR10 Robust Metric Smoke10 Report

Date: 2026-07-02

## Setup

- Server: Chengdu 516 4090 docker
- Dataset: existing CIFAR10 copy at `/home/ubuntu/hy/data/cifar10`
- Benchmark source: RobustBench CIFAR10/Linf models
- Models: 10
- Attack: PGD-lite, 3 steps
- Samples: 512 CIFAR10 test images
- Batch size: 128

## Main Result

All three candidate metrics have the expected direction: lower values indicate stronger empirical robustness.

| Metric | Pearson vs adv_acc | Spearman vs adv_acc | Kendall tau-a | Direction |
| --- | ---: | ---: | ---: | --- |
| feature_delta | -0.9952 | -0.7052 | -0.5455 | lower is better |
| perturbation_capacity_delta | -0.9742 | -0.4012 | -0.4091 | lower is better |
| logit_kl | -0.9894 | -0.6748 | -0.5455 | lower is better |

## Robust Models Only

After excluding the naturally trained `Standard` model, the outlier effect is reduced. The direction remains negative.

| Metric | Pearson vs adv_acc | Spearman vs adv_acc | Kendall tau-a | Direction |
| --- | ---: | ---: | ---: | --- |
| feature_delta | -0.7564 | -0.5941 | -0.4286 | lower is better |
| perturbation_capacity_delta | -0.3864 | -0.1757 | -0.2571 | lower is better |
| logit_kl | -0.5688 | -0.5523 | -0.4286 | lower is better |

## Interpretation

`feature_delta` is currently the strongest single candidate among the three because it separates natural training from robust training and still preserves moderate ranking signal inside robust models. `logit_kl` is also promising but may be closer to an output-space stability proxy than an architecture-sensitive representation metric. `perturbation_capacity_delta` is directionally correct but weaker in this smoke test.

The next validation step should increase sample count and attack strength before changing the metric definition.
