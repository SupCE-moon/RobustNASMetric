# Cross-Space Robust Metric Correlation

Target: `rb_robust_acc`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## RobustBench-20: final

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.805 | 0.805 | [0.505, 0.935] |
| 2 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.699 | 0.699 | [0.336, 0.889] |
| 3 | `racl_arch_lipschitz_log_upper` | RACL generic fixed-network proxy [5] | lower | 20 | -0.633 | 0.633 | [0.282, 0.844] |
| 4 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.626 | 0.626 | [0.275, 0.829] |
| 5 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.555 | 0.555 | [0.0654, 0.873] |
| 6 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.537 | 0.537 | [0.109, 0.802] |
| 7 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.499 | 0.499 | [0.0502, 0.796] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.493 | 0.493 | [0.0692, 0.789] |
| 9 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.492 | 0.492 | [0.0418, 0.776] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.483 | 0.483 | [-0.00606, 0.829] |
| 11 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.457 | 0.457 | [-0.0601, 0.79] |
| 12 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.456 | 0.456 | [-0.0138, 0.77] |
| 13 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | -0.454 | 0.454 | [-0.0689, 0.798] |
| 14 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.429 | 0.429 | [-0.102, 0.773] |
| 15 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.429 | 0.429 | [-0.136, 0.785] |
| 16 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 20 | -0.403 | 0.403 | [-0.102, 0.77] |
| 17 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 20 | -0.4 | 0.4 | [-0.0229, 0.703] |
| 18 | `logit_kl` | TRADES divergence [3] | lower | 20 | -0.382 | 0.382 | [-0.15, 0.755] |
| 19 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.38 | 0.38 | [-0.13, 0.753] |
| 20 | `feature_box_dimension_delta` | Ours Ours | lower | 20 | -0.365 | 0.365 | [-0.178, 0.766] |
| 21 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.355 | 0.355 | [-0.13, 0.708] |
| 22 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | -0.34 | 0.34 | [-0.156, 0.693] |
| 23 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | -0.329 | 0.329 | [-0.161, 0.687] |
| 24 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | -0.329 | 0.329 | [-0.221, 0.749] |
| 25 | `feature_hausdorff_delta` | Ours Ours | lower | 20 | -0.328 | 0.328 | [-0.13, 0.665] |
| 26 | `feature_measure_mmd` | Ours Ours | lower | 20 | -0.314 | 0.314 | [-0.146, 0.673] |
| 27 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.29 | 0.29 | [-0.211, 0.669] |
| 28 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.287 | 0.287 | [-0.18, 0.679] |
| 29 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.286 | 0.286 | [-0.242, 0.693] |
| 30 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.283 | 0.283 | [-0.21, 0.675] |
| 31 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.253 | 0.253 | [-0.211, 0.653] |
| 32 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | -0.242 | 0.242 | [-0.302, 0.661] |
| 33 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | -0.235 | 0.235 | [-0.267, 0.664] |
| 34 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.221 | 0.221 | [-0.261, 0.624] |
| 35 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | -0.209 | 0.209 | [-0.287, 0.64] |
| 36 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.205 | 0.205 | [-0.315, 0.613] |
| 37 | `feature_delta` | Ours Ours | lower | 20 | -0.135 | 0.135 | [-0.358, 0.556] |
| 38 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.015 | -0.015 | [-0.546, 0.505] |
| 39 | `natural_loss` | Loss control [10] | lower | 20 | 0.0226 | -0.0226 | [-0.493, 0.489] |
| 40 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | -0.0394 | -0.0394 | [-0.465, 0.4] |
| 41 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | -0.137 | -0.137 | [-0.587, 0.37] |
| 42 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | -0.143 | -0.143 | [-0.652, 0.319] |
| 43 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | 0.17 | -0.17 | [-0.646, 0.398] |
| 44 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.174 | -0.174 | [-0.648, 0.371] |
| 45 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | -0.235 | -0.235 | [-0.67, 0.312] |
