# NAS-RobBench-201 Quick Validation

## Setup

- Server: Chengdu 516 RTX 5090 docker.
- Benchmark: NAS-RobBench-201, CIFAR-10, 6466 non-isomorphic NAS-Bench-201 architectures.
- Oracle target: `autoattack`.
- Model construction: official `nasrobbench201/libnas` NAS-Bench-201 tiny network.
- Metric computation: local `robust_metric_lab.metrics.compute_feature_metrics`.
- Training used for this quick proxy test: 1 epoch on 2048 CIFAR-10 training samples, FGSM adversarial training.
- Metric batches: 256 CIFAR-10 test samples, PGD-3 adversarial examples.
- Important limitation: this is an early-weight / short-training proxy test, not the full 50-epoch adversarial training protocol from NAS-RobBench-201.

## Generated Files

- `nb201_ours_metric_quick32.csv`
- `nb201_ours_metric_quick128.csv`
- `nb201_ours_metric_quick512.csv`
- `nasrobbench201_quick128_search_summary.csv`
- `nasrobbench201_quick512_search_summary.csv`

## Quick512 Correlation Against Oracle AutoAttack

| Metric | Pearson | Spearman | Kendall | Empirical Direction |
|---|---:|---:|---:|---|
| feature_measure_robust_spatial_delta (Ours) | 0.1053 | 0.0235 | 0.0212 | higher |
| feature_measure_spatial_gram_delta (Ours) | 0.2284 | 0.1582 | 0.1054 | higher |
| feature_measure_weighted_delta (Ours) | 0.0446 | -0.0188 | -0.0059 | lower |
| feature_measure_robust_delta (Ours) | 0.0308 | 0.2156 | 0.1503 | higher |
| feature_measure_robust_hybrid_delta (Ours) | 0.0309 | 0.2181 | 0.1517 | higher |
| fsp_delta | -0.1597 | -0.2757 | -0.1789 | lower |
| local_logit_lipschitz | 0.1093 | 0.0624 | 0.0445 | higher |
| input_grad_l2 | 0.1234 | 0.3737 | 0.2569 | higher |

## Quick512 Search Summary

Budget is 64 architecture queries within the 512 evaluated proxy subset, 100 repeated trials.

| Metric | Direction | Algorithm | Mean Selected AutoAttack | Mean Regret | Best Selected AutoAttack |
|---|---|---|---:|---:|---:|
| fsp_delta | lower | nsga2_proxy_clean | 0.421680 | 0.022120 | 0.4438 |
| feature_measure_robust_delta (Ours) | higher | nsga2_proxy_clean | 0.420450 | 0.023350 | 0.4438 |
| feature_measure_robust_hybrid_delta (Ours) | higher | nsga2_proxy_clean | 0.420450 | 0.023350 | 0.4438 |
| input_grad_l2 | higher | nsga2_proxy_clean | 0.417848 | 0.025952 | 0.4438 |
| local_logit_lipschitz | higher | nsga2_proxy_clean | 0.405677 | 0.038123 | 0.4371 |
| feature_measure_robust_spatial_delta (Ours) | higher | nsga2_proxy_clean | 0.398230 | 0.045570 | 0.4250 |
| feature_measure_spatial_gram_delta (Ours) | higher | nsga2_proxy_clean | 0.394942 | 0.048858 | 0.4301 |
| fsp_delta | lower | random_proxy | 0.400322 | 0.043478 | 0.4324 |
| feature_measure_robust_hybrid_delta (Ours) | higher | random_proxy | 0.394490 | 0.049310 | 0.4371 |
| feature_measure_robust_delta (Ours) | higher | random_proxy | 0.394406 | 0.049394 | 0.4371 |
| feature_measure_robust_spatial_delta (Ours) | higher | random_proxy | 0.391386 | 0.052414 | 0.4250 |
| feature_measure_spatial_gram_delta (Ours) | higher | random_proxy | 0.388201 | 0.055599 | 0.4301 |
| input_grad_l2 | higher | random_proxy | 0.387948 | 0.055852 | 0.4371 |
| local_logit_lipschitz | higher | random_proxy | 0.383739 | 0.060061 | 0.4301 |

## Current Interpretation

The proposed feature-flow style metrics have a measurable but not dominant signal in this quick early-training setting. `feature_measure_robust_delta` and `feature_measure_robust_hybrid_delta` are close to FSP under NSGA-II, but `input_grad_l2` has stronger rank correlation and FSP has the best quick512 NSGA-II regret.

This does not invalidate the proposed metric because the proxy was computed from extremely undertrained models. It does mean the paper should not claim the current spatial-only form is already superior on NAS-RobBench-201. The next stronger validation should increase training fidelity and evaluate residual value over FSP/input-gradient baselines.

## Recommended Next Run

Run 512 or 1024 architectures with:

- 5 epochs instead of 1.
- 8192 to 20000 training samples.
- PGD-3 or PGD-5 adversarial training.
- Metric samples at 1024.
- Compare residual value of `feature_measure_robust_hybrid_delta (Ours)` after controlling for `fsp_delta` and `input_grad_l2`.
