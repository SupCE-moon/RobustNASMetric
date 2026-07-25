# E23 Cross-Space Spearman Summary

All values are direction-adjusted Spearman scores: larger is better. Metric directions were fixed before analysis.

## Zero-Cost Ranking

Mean of NAS-RobBench-201 epoch 0 and DARTS epoch 0.

| Rank | Metric | NB201_e0 | DARTS_e0 | zero_cost_mean |
|---:|---|---:|---:|---:|
| 1 | `trnas_r_score` | 0.897 | 0.75 | 0.824 |
| 2 | `trnas_consistency_phi` | 0.843 | 0.776 | 0.81 |
| 3 | `trnas_activation_psi` | 0.822 | 0.499 | 0.661 |
| 4 | `feature_measure_swd` | 0.751 | 0.501 | 0.626 |
| 5 | `local_logit_lipschitz` | 0.753 | 0.483 | 0.618 |
| 6 | `racl_input_gradient_lower` | 0.709 | 0.456 | 0.582 |
| 7 | `input_grad_l2` | 0.709 | 0.456 | 0.582 |
| 8 | `feature_measure_flow_rel_delta` | 0.793 | 0.329 | 0.561 |
| 9 | `feature_delta` | 0.806 | 0.286 | 0.546 |
| 10 | `feature_measure_moment_delta` | 0.788 | 0.293 | 0.54 |
| 11 | `fsp_delta` | 0.773 | 0.277 | 0.525 |
| 12 | `fsp_strict_delta` | 0.731 | 0.281 | 0.506 |
| 13 | `feature_measure_struct01_delta` | 0.765 | 0.233 | 0.499 |
| 14 | `feature_measure_flow_delta` | 0.751 | 0.215 | 0.483 |
| 15 | `logit_kl` | 0.734 | 0.195 | 0.465 |

## Mature-Model Ranking

Mean of RobustBench-20, NAS-RobBench-201 epoch 50, and DARTS epoch 100.

| Rank | Metric | RobustBench20 | NB201_e50 | DARTS_e100 | mature_mean |
|---:|---|---:|---:|---:|---:|
| 1 | `adversarial_loss` | 0.805 | 0.98 | 0.693 | 0.826 |
| 2 | `racl_arch_lipschitz_log_upper` | 0.633 | 0.454 | 0.69 | 0.593 |
| 3 | `natural_loss` | -0.0226 | 0.983 | 0.738 | 0.566 |
| 4 | `trnas_r_score` | -0.0394 | 0.912 | 0.753 | 0.542 |
| 5 | `trnas_activation_psi` | -0.137 | 0.9 | 0.71 | 0.491 |
| 6 | `trnas_consistency_phi` | -0.143 | 0.785 | 0.764 | 0.469 |
| 7 | `perturbation_capacity_delta` | 0.626 | 0.208 | 0.382 | 0.405 |
| 8 | `fsp_delta` | 0.493 | 0.669 | 0.0436 | 0.402 |
| 9 | `feature_measure_struct01_delta` | 0.499 | 0.656 | 0.0346 | 0.396 |
| 10 | `margin_input_grad_ratio` | -0.235 | 0.954 | 0.421 | 0.38 |
| 11 | `feature_measure_cov_delta` | 0.537 | 0.315 | 0.23 | 0.361 |
| 12 | `feature_measure_struct_delta` | 0.456 | 0.566 | 0.0451 | 0.356 |
| 13 | `feature_measure_moment_delta` | 0.38 | 0.648 | 0.018 | 0.349 |
| 14 | `feature_measure_flow_rel_delta` | 0.205 | 0.498 | 0.0857 | 0.263 |
| 15 | `perturbation_capacity_above_mean_delta` | 0.555 | -0.169 | 0.365 | 0.25 |

## All Nine Settings

| Rank | Metric | RobustBench20 | NB201_e0 | NB201_e5 | NB201_e10 | NB201_e25 | NB201_e50 | DARTS_e0 | DARTS_e50 | DARTS_e100 | all_setting_mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `trnas_r_score` | -0.0394 | 0.897 | 0.894 | 0.907 | 0.91 | 0.912 | 0.75 | 0.746 | 0.753 | 0.748 |
| 2 | `adversarial_loss` | 0.805 | 0.588 | 0.655 | 0.907 | 0.956 | 0.98 | -0.0195 | 0.621 | 0.693 | 0.687 |
| 3 | `trnas_consistency_phi` | -0.143 | 0.843 | 0.794 | 0.782 | 0.768 | 0.785 | 0.776 | 0.771 | 0.764 | 0.682 |
| 4 | `trnas_activation_psi` | -0.137 | 0.822 | 0.831 | 0.897 | 0.906 | 0.9 | 0.499 | 0.701 | 0.71 | 0.681 |
| 5 | `natural_loss` | -0.0226 | 0.572 | 0.612 | 0.87 | 0.953 | 0.983 | -0.012 | 0.695 | 0.738 | 0.599 |
| 6 | `margin_input_grad_ratio` | -0.235 | 0.446 | 0.455 | 0.618 | 0.719 | 0.954 | 0.304 | 0.789 | 0.421 | 0.497 |
| 7 | `fsp_delta` | 0.493 | 0.773 | 0.731 | 0.69 | 0.648 | 0.669 | 0.277 | -0.0571 | 0.0436 | 0.474 |
| 8 | `feature_measure_struct01_delta` | 0.499 | 0.765 | 0.724 | 0.684 | 0.632 | 0.656 | 0.233 | -0.0647 | 0.0346 | 0.462 |
| 9 | `feature_measure_moment_delta` | 0.38 | 0.788 | 0.73 | 0.707 | 0.673 | 0.648 | 0.293 | -0.0812 | 0.018 | 0.462 |
| 10 | `feature_measure_flow_rel_delta` | 0.205 | 0.793 | 0.631 | 0.677 | 0.527 | 0.498 | 0.329 | 0.0722 | 0.0857 | 0.424 |
| 11 | `feature_measure_struct_delta` | 0.456 | 0.736 | 0.686 | 0.645 | 0.517 | 0.566 | 0.135 | 0.0226 | 0.0451 | 0.423 |
| 12 | `feature_measure_cov_delta` | 0.537 | 0.763 | 0.416 | 0.622 | 0.332 | 0.315 | 0.0571 | 0.268 | 0.23 | 0.393 |
| 13 | `feature_delta` | 0.135 | 0.806 | 0.667 | 0.697 | 0.601 | 0.514 | 0.286 | -0.176 | -0.0887 | 0.382 |
| 14 | `feature_measure_robust_rel_delta` | 0.221 | 0.75 | 0.567 | 0.622 | 0.375 | 0.443 | 0.165 | 0.0812 | 0.0842 | 0.368 |
| 15 | `racl_arch_lipschitz_log_upper` | 0.633 | 0.254 | 0.29 | 0.383 | 0.264 | 0.454 | -0.558 | 0.69 | 0.69 | 0.345 |
| 16 | `feature_measure_relu_active_hybrid_rel_delta` | 0.29 | 0.757 | 0.403 | 0.47 | 0.181 | 0.364 | 0.143 | -0.0481 | 0.0571 | 0.291 |
| 17 | `feature_measure_robust_hybrid_rel_delta` | 0.283 | 0.746 | 0.441 | 0.516 | 0.192 | 0.35 | 0.104 | -0.0586 | 0.0135 | 0.287 |
| 18 | `perturbation_capacity_delta` | 0.626 | 0.703 | -0.397 | 0.187 | 0.237 | 0.208 | 0.0511 | 0.262 | 0.382 | 0.251 |
| 19 | `feature_measure_mmd` | 0.314 | 0.794 | 0.646 | 0.502 | 0.12 | 0.0159 | 0.0947 | -0.132 | -0.335 | 0.224 |
| 20 | `feature_measure_flow_delta` | -0.174 | 0.751 | 0.37 | 0.417 | 0.275 | 0.214 | 0.215 | -0.164 | -0.156 | 0.194 |

A high average does not prove mechanism independence. Consult the DARTS structural-confound table and bootstrap intervals before making a paper claim.