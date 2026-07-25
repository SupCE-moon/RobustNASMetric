# NAS-RobBench-201 Search Validation

## Benchmark Choice

NAS-RobBench-201 is the closest RobustNAS analogue of NAS-Bench-201 for fast tabular validation. It keeps the NAS-Bench-201 cell search space and provides adversarial-training results for the 6466 non-isomorphic architectures.

A related alternative is Jung et al., *Neural Architecture Design and Robustness: A Dataset*, which also provides tabular robustness evaluations for NAS-Bench-201-style architectures across CIFAR-10, CIFAR-100, and ImageNet16-120.

## Validation Question

We do not use the benchmark metric as the proposed metric itself. Instead:

1. Use NAS-RobBench-201 robust accuracy as the oracle target.
2. Compute our feature-measure robustness metric for each architecture, or import a CSV already computed elsewhere.
3. Run low-cost search using only the proxy metric, optionally with clean accuracy as a second objective.
4. Check whether the selected architectures have high oracle robust accuracy.

If the proxy is useful, random search and NSGA-II guided by our metric should obtain lower regret than unguided sampling.

## Script

The runner is:

```powershell
python E:\projects\robust_metric_lab\scripts\search_nasrobbench201.py `
  --benchmark-json E:\datasets\NAS-RobBench-201\NAS-RobBench-201_all.json `
  --meta-json E:\datasets\NAS-RobBench-201\meta.json `
  --dataset cifar10 `
  --target-metric val_pgd_3.0_acc `
  --clean-metric val_acc `
  --proxy-csv E:\projects\robust_metric_lab\outputs\nasrobbench201_ours_metric.csv `
  --proxy-id-column arch_id `
  --proxy-column feature_measure_robust_spatial_delta `
  --proxy-direction lower `
  --budget 200 `
  --trials 20 `
  --population 64 `
  --generations 25 `
  --output E:\projects\robust_metric_lab\outputs\nasrobbench201_search_results.csv
```

Metric names in a downloaded benchmark file can be inspected with:

```powershell
python E:\projects\robust_metric_lab\scripts\search_nasrobbench201.py `
  --benchmark-json E:\datasets\NAS-RobBench-201\NAS-RobBench-201_all.json `
  --meta-json E:\datasets\NAS-RobBench-201\meta.json `
  --dataset cifar10 `
  --list-metrics
```

## Output Interpretation

The output CSV contains one row per trial and algorithm.

- `selected_target`: oracle robust accuracy of the architecture selected by the proxy search.
- `sampled_best_target`: best oracle robust accuracy among all architectures touched by that trial.
- `global_best_target`: best oracle robust accuracy among all usable benchmark architectures.
- `regret_to_global`: `global_best_target - selected_target`; lower is better.

For the paper, the key table should report mean selected robust accuracy and mean regret over repeated trials, ideally across CIFAR-10, CIFAR-100, and ImageNet16-120.
