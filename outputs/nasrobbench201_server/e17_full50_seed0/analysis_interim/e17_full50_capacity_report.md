# E17 Full-Training Capacity Validation Report

Status: **interim**, with 83/96 complete architectures.

## Capacity Result

| Epoch | Spearman | p-value | Bootstrap 95% CI |
|---:|---:|---:|---:|
| 10 | -0.193 | 0.0808 | [-0.394, 0.00127] |
| 25 | 0.0141 | 0.899 | [-0.225, 0.25] |
| 50 | -0.234 | 0.033 | [-0.453, 0.0211] |

## Selected Metric Comparison

| Epoch | Metric | Spearman | p-value |
|---:|---|---:|---:|
| 10 | `adversarial_loss` | -0.948 | 4.7e-42 |
| 10 | `feature_measure_mmd` | -0.537 | 1.63e-07 |
| 10 | `fsp_strict_delta` | -0.468 | 8.03e-06 |
| 10 | `perturbation_capacity_delta` | -0.193 | 0.0808 |
| 10 | `perturbation_capacity_above_mean_delta` | 0.0561 | 0.614 |
| 10 | `perturbation_capacity_relu_active_delta` | 0.22 | 0.0461 |
| 10 | `clean_acc` | 0.901 | 3.4e-31 |
| 25 | `adversarial_loss` | -0.982 | 1.12e-60 |
| 25 | `perturbation_capacity_above_mean_delta` | -0.0858 | 0.441 |
| 25 | `feature_measure_mmd` | -0.0824 | 0.459 |
| 25 | `perturbation_capacity_delta` | 0.0141 | 0.899 |
| 25 | `fsp_strict_delta` | 0.0426 | 0.702 |
| 25 | `perturbation_capacity_relu_active_delta` | 0.161 | 0.145 |
| 25 | `clean_acc` | 0.961 | 3.41e-47 |
| 50 | `adversarial_loss` | -0.987 | 1.15e-66 |
| 50 | `perturbation_capacity_above_mean_delta` | -0.262 | 0.0169 |
| 50 | `perturbation_capacity_delta` | -0.234 | 0.033 |
| 50 | `feature_measure_mmd` | -0.00754 | 0.946 |
| 50 | `perturbation_capacity_relu_active_delta` | 0.066 | 0.553 |
| 50 | `fsp_strict_delta` | 0.627 | 2.29e-10 |
| 50 | `clean_acc` | 0.977 | 3.35e-56 |

## Residual Diagnostic

| Epoch | Control set | Partial Spearman | p-value |
|---:|---|---:|---:|
| 10 | `same_run_clean_acc` | 0.162 | 0.143 |
| 10 | `official_val_acc_diagnostic` | 0.158 | 0.154 |
| 10 | `training_maturity` | 0.181 | 0.101 |
| 25 | `same_run_clean_acc` | 0.0756 | 0.497 |
| 25 | `official_val_acc_diagnostic` | 0.199 | 0.0715 |
| 25 | `training_maturity` | 0.132 | 0.234 |
| 50 | `same_run_clean_acc` | 0.131 | 0.239 |
| 50 | `official_val_acc_diagnostic` | -0.00325 | 0.977 |
| 50 | `training_maturity` | 0.139 | 0.212 |

## Checkpoint Change

| From | To | Spearman change | Bootstrap 95% CI |
|---:|---:|---:|---:|
| 10 | 25 | 0.207 | [-0.0166, 0.44] |
| 10 | 50 | -0.0416 | [-0.275, 0.233] |

Raw negative correlation is directionally consistent with a lower-is-better capacity metric. The residual rows test whether that association remains after removing clean-performance or training-maturity rank information. The official-validation control is diagnostic only and must not be used as a search-time input.
