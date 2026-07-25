# Strict FSP vs Measure ROBUST Metric Report

Date: 2026-07-04
Dataset: CIFAR-10, Linf RobustBench models
Attack: white-box PGD-3, epsilon from config, sample_limit=512, model_limit=7, seed=20260704

## Correlation: all 7 models
input: outputs/metrics_strict_fsp_vs_robust_balanced_seed20260704_m7.csv
models: 7
target: adv_acc

metric,pearson,spearman,kendall_tau_a,empirical_direction
fsp_strict_delta,-0.9844,-0.4643,-0.4286,lower_is_better
feature_measure_robust_delta,-0.9920,-0.8571,-0.7143,lower_is_better
feature_measure_weighted_delta,-0.9932,-0.7857,-0.7143,lower_is_better

## Correlation: robust-only, Standard excluded
input: outputs/metrics_strict_fsp_vs_robust_balanced_seed20260704_m7.csv
models: 6
excluded: Standard
target: adv_acc

metric,pearson,spearman,kendall_tau_a,empirical_direction
fsp_strict_delta,0.0628,-0.1429,-0.2000,higher_is_better
feature_measure_robust_delta,-0.6374,-0.7714,-0.6000,lower_is_better
feature_measure_weighted_delta,-0.6747,-0.6571,-0.6000,lower_is_better

## Per-model rows
| model | adv_acc | fsp_strict_delta | feature_measure_robust_delta | feature_measure_weighted_delta |
|---|---:|---:|---:|---:|
| Standard | 0.0273 | 14.411932 | 0.627892 | 0.602073 |
| Engstrom2019Robustness | 0.7246 | 1.293300 | 0.101909 | 0.106013 |
| Rice2020Overfitting | 0.7168 | 0.083197 | 0.117977 | 0.119793 |
| Wong2020Fast | 0.6426 | 0.185250 | 0.119630 | 0.127452 |
| Carmon2019Unlabeled | 0.7734 | 0.015542 | 0.097834 | 0.093259 |
| Hendrycks2019Using | 0.7285 | 0.406769 | 0.097306 | 0.102866 |
| Zhang2019Theoretically | 0.6934 | 0.129586 | 0.098419 | 0.093587 |
