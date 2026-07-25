# E17 Full-Training Capacity Validation Report

Status: **stopped**, with 91/96 planned architectures complete.

## Capacity Result

| Epoch | Spearman | p-value | Bootstrap 95% CI |
|---:|---:|---:|---:|
| 10 | -0.146 | 0.166 | [-0.344, 0.056] |
| 25 | 0.0538 | 0.612 | [-0.18, 0.289] |
| 50 | -0.184 | 0.0805 | [-0.403, 0.0727] |

## Selected Metric Comparison

| Epoch | Metric | Spearman | p-value |
|---:|---|---:|---:|
| 10 | `adversarial_loss` | -0.942 | 5.44e-44 |
| 10 | `feature_measure_mmd` | -0.497 | 5.37e-07 |
| 10 | `fsp_strict_delta` | -0.439 | 1.36e-05 |
| 10 | `perturbation_capacity_delta` | -0.146 | 0.166 |
| 10 | `perturbation_capacity_above_mean_delta` | 0.0963 | 0.364 |
| 10 | `perturbation_capacity_relu_active_delta` | 0.241 | 0.0211 |
| 10 | `clean_acc` | 0.894 | 7.73e-33 |
| 25 | `adversarial_loss` | -0.98 | 3.98e-64 |
| 25 | `feature_measure_mmd` | -0.0581 | 0.584 |
| 25 | `perturbation_capacity_above_mean_delta` | -0.042 | 0.693 |
| 25 | `perturbation_capacity_delta` | 0.0538 | 0.612 |
| 25 | `fsp_strict_delta` | 0.0583 | 0.583 |
| 25 | `perturbation_capacity_relu_active_delta` | 0.203 | 0.0536 |
| 25 | `clean_acc` | 0.957 | 1.83e-49 |
| 50 | `adversarial_loss` | -0.987 | 1.55e-72 |
| 50 | `perturbation_capacity_above_mean_delta` | -0.228 | 0.0295 |
| 50 | `perturbation_capacity_delta` | -0.184 | 0.0805 |
| 50 | `feature_measure_mmd` | 0.0279 | 0.793 |
| 50 | `perturbation_capacity_relu_active_delta` | 0.101 | 0.34 |
| 50 | `fsp_strict_delta` | 0.579 | 1.81e-09 |
| 50 | `clean_acc` | 0.974 | 2.01e-59 |

## Residual Diagnostic

| Epoch | Control set | Partial Spearman | p-value |
|---:|---|---:|---:|
| 10 | `same_run_clean_acc` | 0.203 | 0.0531 |
| 10 | `official_val_acc_diagnostic` | 0.176 | 0.0945 |
| 10 | `training_maturity` | 0.217 | 0.0384 |
| 25 | `same_run_clean_acc` | 0.137 | 0.195 |
| 25 | `official_val_acc_diagnostic` | 0.213 | 0.043 |
| 25 | `training_maturity` | 0.139 | 0.189 |
| 50 | `same_run_clean_acc` | 0.176 | 0.0942 |
| 50 | `official_val_acc_diagnostic` | 0.0483 | 0.649 |
| 50 | `training_maturity` | 0.154 | 0.145 |

## Checkpoint Change

| From | To | Spearman change | Bootstrap 95% CI |
|---:|---:|---:|---:|
| 10 | 25 | 0.2 | [-0.0229, 0.418] |
| 10 | 50 | -0.0377 | [-0.27, 0.211] |

Raw negative correlation is directionally consistent with a lower-is-better capacity metric. The residual rows test whether that association remains after removing clean-performance or training-maturity rank information. The official-validation control is diagnostic only and must not be used as a search-time input.
