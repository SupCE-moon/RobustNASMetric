# Cross-Space Robust Metric Correlation

Target: `oracle_target`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## NAS-RobBench-201: epoch_0

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.897 | 0.897 | [0.813, 0.938] |
| 2 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.843 | 0.843 | [0.749, 0.9] |
| 3 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.822 | 0.822 | [0.7, 0.893] |
| 4 | `feature_delta` | Ours Ours | lower | 64 | -0.806 | 0.806 | [0.667, 0.891] |
| 5 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.794 | 0.794 | [0.638, 0.894] |
| 6 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.793 | 0.793 | [0.656, 0.877] |
| 7 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.788 | 0.788 | [0.639, 0.883] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.773 | 0.773 | [0.632, 0.866] |
| 9 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.766 | 0.766 | [0.631, 0.856] |
| 10 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.765 | 0.765 | [0.609, 0.865] |
| 11 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.763 | 0.763 | [0.64, 0.836] |
| 12 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.761 | 0.761 | [0.582, 0.866] |
| 13 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.757 | 0.757 | [0.582, 0.857] |
| 14 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | -0.753 | 0.753 | [0.58, 0.866] |
| 15 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.751 | 0.751 | [0.578, 0.869] |
| 16 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.751 | 0.751 | [0.619, 0.835] |
| 17 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | -0.751 | 0.751 | [0.576, 0.864] |
| 18 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 64 | -0.75 | 0.75 | [0.652, 0.812] |
| 19 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.75 | 0.75 | [0.586, 0.855] |
| 20 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.746 | 0.746 | [0.58, 0.844] |
| 21 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.745 | 0.745 | [0.584, 0.854] |
| 22 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.744 | 0.744 | [0.587, 0.843] |
| 23 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.737 | 0.737 | [0.582, 0.843] |
| 24 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.736 | 0.736 | [0.57, 0.845] |
| 25 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | -0.734 | 0.734 | [0.56, 0.855] |
| 26 | `logit_kl` | TRADES divergence [3] | lower | 64 | -0.734 | 0.734 | [0.549, 0.851] |
| 27 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | -0.733 | 0.733 | [0.582, 0.834] |
| 28 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.731 | 0.731 | [0.547, 0.859] |
| 29 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.729 | 0.729 | [0.551, 0.849] |
| 30 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.725 | 0.725 | [0.571, 0.832] |
| 31 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | -0.718 | 0.718 | [0.565, 0.819] |
| 32 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 64 | -0.715 | 0.715 | [0.598, 0.79] |
| 33 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.714 | 0.714 | [0.542, 0.83] |
| 34 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.71 | 0.71 | [0.544, 0.822] |
| 35 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | -0.709 | 0.709 | [0.524, 0.831] |
| 36 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | -0.709 | 0.709 | [0.519, 0.831] |
| 37 | `feature_hausdorff_delta` | Ours Ours | lower | 64 | -0.704 | 0.704 | [0.579, 0.781] |
| 38 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.703 | 0.703 | [0.58, 0.795] |
| 39 | `feature_box_dimension_delta` | Ours Ours | lower | 64 | -0.642 | 0.642 | [0.494, 0.744] |
| 40 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.588 | 0.588 | [0.362, 0.757] |
| 41 | `natural_loss` | Loss control [10] | lower | 64 | -0.572 | 0.572 | [0.34, 0.751] |
| 42 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.446 | 0.446 | [0.207, 0.627] |
| 43 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | -0.419 | 0.419 | [0.214, 0.587] |
| 44 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | -0.398 | 0.398 | [0.179, 0.576] |
| 45 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.254 | 0.254 | [0.0282, 0.445] |

## NAS-RobBench-201: epoch_5

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.894 | 0.894 | [0.813, 0.933] |
| 2 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.831 | 0.831 | [0.72, 0.901] |
| 3 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.794 | 0.794 | [0.646, 0.876] |
| 4 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.731 | 0.731 | [0.574, 0.834] |
| 5 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.73 | 0.73 | [0.55, 0.842] |
| 6 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.724 | 0.724 | [0.556, 0.828] |
| 7 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.686 | 0.686 | [0.513, 0.804] |
| 8 | `feature_delta` | Ours Ours | lower | 64 | -0.667 | 0.667 | [0.471, 0.818] |
| 9 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.655 | 0.655 | [0.448, 0.808] |
| 10 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.646 | 0.646 | [0.433, 0.803] |
| 11 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.631 | 0.631 | [0.424, 0.788] |
| 12 | `natural_loss` | Loss control [10] | lower | 64 | -0.612 | 0.612 | [0.389, 0.786] |
| 13 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.575 | 0.575 | [0.354, 0.743] |
| 14 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.567 | 0.567 | [0.332, 0.735] |
| 15 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.455 | 0.455 | [0.202, 0.665] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.441 | 0.441 | [0.204, 0.642] |
| 17 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.437 | 0.437 | [0.202, 0.624] |
| 18 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.43 | 0.43 | [0.189, 0.628] |
| 19 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.416 | 0.416 | [0.169, 0.628] |
| 20 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.403 | 0.403 | [0.156, 0.614] |
| 21 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.37 | 0.37 | [0.112, 0.582] |
| 22 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.328 | 0.328 | [0.0748, 0.548] |
| 23 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.317 | 0.317 | [0.0485, 0.546] |
| 24 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.316 | 0.316 | [0.0636, 0.536] |
| 25 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.29 | 0.29 | [0.0712, 0.472] |
| 26 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.233 | 0.233 | [-0.0284, 0.478] |
| 27 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | -0.227 | 0.227 | [-0.00363, 0.424] |
| 28 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.164 | 0.164 | [-0.0913, 0.418] |
| 29 | `feature_box_dimension_delta` | Ours Ours | lower | 64 | -0.156 | 0.156 | [-0.0927, 0.377] |
| 30 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.0449 | 0.0449 | [-0.203, 0.295] |
| 31 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.249 | -0.249 | [-0.474, -0.00657] |
| 32 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.283 | -0.283 | [-0.502, -0.0411] |
| 33 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | 0.397 | -0.397 | [-0.584, -0.177] |
| 34 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | 0.397 | -0.397 | [-0.594, -0.174] |
| 35 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | 0.414 | -0.414 | [-0.599, -0.195] |
| 36 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.427 | -0.427 | [-0.604, -0.215] |
| 37 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.428 | -0.428 | [-0.615, -0.206] |
| 38 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 64 | 0.457 | -0.457 | [-0.633, -0.258] |
| 39 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.458 | -0.458 | [-0.629, -0.268] |
| 40 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.458 | -0.458 | [-0.614, -0.259] |
| 41 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.469 | -0.469 | [-0.665, -0.219] |
| 42 | `feature_hausdorff_delta` | Ours Ours | lower | 64 | 0.498 | -0.498 | [-0.65, -0.311] |
| 43 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 64 | 0.716 | -0.716 | [-0.825, -0.54] |
| 44 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.83 | -0.83 | [-0.917, -0.685] |
| 45 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.83 | -0.83 | [-0.92, -0.699] |

## NAS-RobBench-201: epoch_10

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.907 | 0.907 | [0.834, 0.942] |
| 2 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.907 | 0.907 | [0.824, 0.95] |
| 3 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.897 | 0.897 | [0.831, 0.935] |
| 4 | `natural_loss` | Loss control [10] | lower | 64 | -0.87 | 0.87 | [0.76, 0.932] |
| 5 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.782 | 0.782 | [0.639, 0.864] |
| 6 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.707 | 0.707 | [0.523, 0.824] |
| 7 | `feature_delta` | Ours Ours | lower | 64 | -0.697 | 0.697 | [0.525, 0.819] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.69 | 0.69 | [0.524, 0.802] |
| 9 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.684 | 0.684 | [0.511, 0.796] |
| 10 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.677 | 0.677 | [0.497, 0.803] |
| 11 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.645 | 0.645 | [0.464, 0.774] |
| 12 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.622 | 0.622 | [0.423, 0.767] |
| 13 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.622 | 0.622 | [0.403, 0.766] |
| 14 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.618 | 0.618 | [0.397, 0.769] |
| 15 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.55 | 0.55 | [0.315, 0.709] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.516 | 0.516 | [0.288, 0.695] |
| 17 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.502 | 0.502 | [0.256, 0.688] |
| 18 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.47 | 0.47 | [0.235, 0.665] |
| 19 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.465 | 0.465 | [0.212, 0.671] |
| 20 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.417 | 0.417 | [0.175, 0.622] |
| 21 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.389 | 0.389 | [0.135, 0.595] |
| 22 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.383 | 0.383 | [0.165, 0.563] |
| 23 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.339 | 0.339 | [0.0717, 0.556] |
| 24 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.286 | 0.286 | [0.0246, 0.528] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.264 | 0.264 | [-0.00736, 0.5] |
| 26 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.244 | 0.244 | [-0.0185, 0.484] |
| 27 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.189 | 0.189 | [-0.0823, 0.448] |
| 28 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.187 | 0.187 | [-0.0456, 0.403] |
| 29 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.0694 | 0.0694 | [-0.191, 0.33] |
| 30 | `feature_box_dimension_delta` | Ours Ours | lower | 64 | -0.00815 | 0.00815 | [-0.221, 0.232] |
| 31 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.137 | -0.137 | [-0.367, 0.103] |
| 32 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.216 | -0.216 | [-0.442, 0.0342] |
| 33 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.278 | -0.278 | [-0.495, -0.0544] |
| 34 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | 0.304 | -0.304 | [-0.536, -0.0584] |
| 35 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | 0.318 | -0.318 | [-0.538, -0.0536] |
| 36 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.337 | -0.337 | [-0.562, -0.0745] |
| 37 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.373 | -0.373 | [-0.594, -0.12] |
| 38 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 64 | 0.38 | -0.38 | [-0.579, -0.137] |
| 39 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.559 | -0.559 | [-0.714, -0.354] |
| 40 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.574 | -0.574 | [-0.728, -0.38] |
| 41 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 64 | 0.593 | -0.593 | [-0.746, -0.394] |
| 42 | `feature_hausdorff_delta` | Ours Ours | lower | 64 | 0.616 | -0.616 | [-0.765, -0.424] |
| 43 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.671 | -0.671 | [-0.805, -0.491] |
| 44 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.79 | -0.79 | [-0.886, -0.63] |
| 45 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.79 | -0.79 | [-0.887, -0.639] |

## NAS-RobBench-201: epoch_25

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.956 | 0.956 | [0.914, 0.973] |
| 2 | `natural_loss` | Loss control [10] | lower | 64 | -0.953 | 0.953 | [0.899, 0.974] |
| 3 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.91 | 0.91 | [0.833, 0.947] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.906 | 0.906 | [0.844, 0.941] |
| 5 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.768 | 0.768 | [0.613, 0.862] |
| 6 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.719 | 0.719 | [0.526, 0.853] |
| 7 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.673 | 0.673 | [0.495, 0.792] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.648 | 0.648 | [0.479, 0.76] |
| 9 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.632 | 0.632 | [0.45, 0.749] |
| 10 | `feature_delta` | Ours Ours | lower | 64 | -0.601 | 0.601 | [0.376, 0.759] |
| 11 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.527 | 0.527 | [0.287, 0.71] |
| 12 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.517 | 0.517 | [0.332, 0.663] |
| 13 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.375 | 0.375 | [0.121, 0.577] |
| 14 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.332 | 0.332 | [0.074, 0.551] |
| 15 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.275 | 0.275 | [-0.017, 0.527] |
| 16 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.264 | 0.264 | [0.0308, 0.466] |
| 17 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.242 | 0.242 | [-0.0393, 0.475] |
| 18 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.237 | 0.237 | [-0.0297, 0.476] |
| 19 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.192 | 0.192 | [-0.0723, 0.431] |
| 20 | `feature_box_dimension_delta` | Ours Ours | lower | 64 | -0.185 | 0.185 | [-0.0554, 0.4] |
| 21 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.181 | 0.181 | [-0.0714, 0.412] |
| 22 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.12 | 0.12 | [-0.165, 0.392] |
| 23 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.104 | 0.104 | [-0.147, 0.351] |
| 24 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.00421 | 0.00421 | [-0.235, 0.266] |
| 25 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | 0.0928 | -0.0928 | [-0.34, 0.156] |
| 26 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | 0.112 | -0.112 | [-0.356, 0.129] |
| 27 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.117 | -0.117 | [-0.4, 0.188] |
| 28 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | 0.119 | -0.119 | [-0.354, 0.141] |
| 29 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.123 | -0.123 | [-0.361, 0.135] |
| 30 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | 0.124 | -0.124 | [-0.367, 0.139] |
| 31 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | 0.149 | -0.149 | [-0.395, 0.0979] |
| 32 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.151 | -0.151 | [-0.372, 0.113] |
| 33 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | 0.167 | -0.167 | [-0.405, 0.0993] |
| 34 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | 0.173 | -0.173 | [-0.401, 0.106] |
| 35 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.188 | -0.188 | [-0.409, 0.0625] |
| 36 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.258 | -0.258 | [-0.481, -0.000695] |
| 37 | `feature_measure_local_delta` | Ours Ours | lower | 64 | 0.316 | -0.316 | [-0.534, -0.0876] |
| 38 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.409 | -0.409 | [-0.631, -0.171] |
| 39 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.659 | -0.659 | [-0.79, -0.472] |
| 40 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 64 | 0.66 | -0.66 | [-0.786, -0.477] |
| 41 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.725 | -0.725 | [-0.836, -0.544] |
| 42 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.725 | -0.725 | [-0.838, -0.568] |
| 43 | `feature_hausdorff_delta` | Ours Ours | lower | 64 | 0.779 | -0.779 | [-0.856, -0.664] |
| 44 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 64 | 0.786 | -0.786 | [-0.863, -0.66] |
| 45 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.827 | -0.827 | [-0.9, -0.715] |

## NAS-RobBench-201: epoch_50

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `natural_loss` | Loss control [10] | lower | 64 | -0.983 | 0.983 | [0.963, 0.99] |
| 2 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.98 | 0.98 | [0.956, 0.988] |
| 3 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.954 | 0.954 | [0.912, 0.973] |
| 4 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.912 | 0.912 | [0.834, 0.95] |
| 5 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.9 | 0.9 | [0.83, 0.941] |
| 6 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.785 | 0.785 | [0.647, 0.868] |
| 7 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.669 | 0.669 | [0.489, 0.781] |
| 8 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.656 | 0.656 | [0.469, 0.771] |
| 9 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.648 | 0.648 | [0.443, 0.793] |
| 10 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.566 | 0.566 | [0.378, 0.697] |
| 11 | `feature_delta` | Ours Ours | lower | 64 | -0.514 | 0.514 | [0.255, 0.719] |
| 12 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.498 | 0.498 | [0.236, 0.705] |
| 13 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.454 | 0.454 | [0.245, 0.623] |
| 14 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.443 | 0.443 | [0.158, 0.645] |
| 15 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.364 | 0.364 | [0.13, 0.548] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.35 | 0.35 | [0.106, 0.534] |
| 17 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.315 | 0.315 | [0.00991, 0.562] |
| 18 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.214 | 0.214 | [-0.057, 0.455] |
| 19 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.208 | 0.208 | [-0.0636, 0.467] |
| 20 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.208 | 0.208 | [-0.0402, 0.437] |
| 21 | `feature_box_dimension_delta` | Ours Ours | lower | 64 | -0.146 | 0.146 | [-0.109, 0.37] |
| 22 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.101 | 0.101 | [-0.17, 0.367] |
| 23 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.0849 | 0.0849 | [-0.152, 0.302] |
| 24 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.0432 | 0.0432 | [-0.176, 0.26] |
| 25 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.0159 | 0.0159 | [-0.282, 0.283] |
| 26 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | -0.0153 | 0.0153 | [-0.208, 0.231] |
| 27 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | -0.000504 | 0.000504 | [-0.227, 0.232] |
| 28 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | 0.00339 | -0.00339 | [-0.227, 0.215] |
| 29 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | 0.0305 | -0.0305 | [-0.275, 0.197] |
| 30 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | 0.0479 | -0.0479 | [-0.271, 0.179] |
| 31 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | 0.0748 | -0.0748 | [-0.308, 0.149] |
| 32 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | 0.0762 | -0.0762 | [-0.324, 0.186] |
| 33 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.169 | -0.169 | [-0.446, 0.102] |
| 34 | `feature_measure_local_delta` | Ours Ours | lower | 64 | 0.174 | -0.174 | [-0.388, 0.0458] |
| 35 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.475 | -0.475 | [-0.686, -0.241] |
| 36 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | 0.53 | -0.53 | [-0.707, -0.301] |
| 37 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.544 | -0.544 | [-0.728, -0.309] |
| 38 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.63 | -0.63 | [-0.766, -0.438] |
| 39 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 64 | 0.636 | -0.636 | [-0.775, -0.434] |
| 40 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.842 | -0.842 | [-0.915, -0.729] |
| 41 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.844 | -0.844 | [-0.915, -0.712] |
| 42 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.844 | -0.844 | [-0.915, -0.716] |
| 43 | `feature_hausdorff_delta` | Ours Ours | lower | 64 | 0.87 | -0.87 | [-0.917, -0.786] |
| 44 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 64 | 0.883 | -0.883 | [-0.93, -0.799] |
| 45 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.933 | -0.933 | [-0.965, -0.864] |
