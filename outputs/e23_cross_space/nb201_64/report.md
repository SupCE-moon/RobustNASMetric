# Cross-Space Robust Metric Correlation

Target: `oracle_target`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## NAS-RobBench-201: epoch_0

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.897 | 0.897 | [0.814, 0.938] |
| 2 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.843 | 0.843 | [0.754, 0.898] |
| 3 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.822 | 0.822 | [0.705, 0.893] |
| 4 | `feature_delta` | Ours Ours | lower | 64 | -0.806 | 0.806 | [0.666, 0.886] |
| 5 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.794 | 0.794 | [0.633, 0.889] |
| 6 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.793 | 0.793 | [0.656, 0.881] |
| 7 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.788 | 0.788 | [0.645, 0.876] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.773 | 0.773 | [0.626, 0.867] |
| 9 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.766 | 0.766 | [0.632, 0.853] |
| 10 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.765 | 0.765 | [0.615, 0.864] |
| 11 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.763 | 0.763 | [0.645, 0.838] |
| 12 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.761 | 0.761 | [0.593, 0.867] |
| 13 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.757 | 0.757 | [0.587, 0.859] |
| 14 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | -0.753 | 0.753 | [0.569, 0.867] |
| 15 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.751 | 0.751 | [0.572, 0.868] |
| 16 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.751 | 0.751 | [0.621, 0.836] |
| 17 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | -0.751 | 0.751 | [0.583, 0.868] |
| 18 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.75 | 0.75 | [0.582, 0.853] |
| 19 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.746 | 0.746 | [0.591, 0.845] |
| 20 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.745 | 0.745 | [0.58, 0.857] |
| 21 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.744 | 0.744 | [0.583, 0.845] |
| 22 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.737 | 0.737 | [0.569, 0.843] |
| 23 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.736 | 0.736 | [0.559, 0.847] |
| 24 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | -0.734 | 0.734 | [0.558, 0.857] |
| 25 | `logit_kl` | TRADES divergence [3] | lower | 64 | -0.734 | 0.734 | [0.556, 0.854] |
| 26 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | -0.733 | 0.733 | [0.594, 0.834] |
| 27 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.731 | 0.731 | [0.547, 0.851] |
| 28 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.729 | 0.729 | [0.533, 0.848] |
| 29 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.725 | 0.725 | [0.568, 0.833] |
| 30 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | -0.718 | 0.718 | [0.568, 0.825] |
| 31 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.714 | 0.714 | [0.55, 0.822] |
| 32 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.71 | 0.71 | [0.528, 0.832] |
| 33 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | -0.709 | 0.709 | [0.525, 0.833] |
| 34 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | -0.709 | 0.709 | [0.533, 0.836] |
| 35 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.703 | 0.703 | [0.57, 0.787] |
| 36 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.588 | 0.588 | [0.361, 0.769] |
| 37 | `natural_loss` | Loss control [10] | lower | 64 | -0.572 | 0.572 | [0.33, 0.74] |
| 38 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.446 | 0.446 | [0.23, 0.626] |
| 39 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | -0.419 | 0.419 | [0.215, 0.583] |
| 40 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | -0.398 | 0.398 | [0.185, 0.577] |
| 41 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.254 | 0.254 | [0.0327, 0.439] |

## NAS-RobBench-201: epoch_5

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.894 | 0.894 | [0.815, 0.935] |
| 2 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.831 | 0.831 | [0.713, 0.903] |
| 3 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.794 | 0.794 | [0.647, 0.879] |
| 4 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.731 | 0.731 | [0.568, 0.833] |
| 5 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.73 | 0.73 | [0.555, 0.843] |
| 6 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.724 | 0.724 | [0.562, 0.828] |
| 7 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.686 | 0.686 | [0.515, 0.798] |
| 8 | `feature_delta` | Ours Ours | lower | 64 | -0.667 | 0.667 | [0.473, 0.81] |
| 9 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.655 | 0.655 | [0.438, 0.817] |
| 10 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.646 | 0.646 | [0.436, 0.801] |
| 11 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.631 | 0.631 | [0.415, 0.789] |
| 12 | `natural_loss` | Loss control [10] | lower | 64 | -0.612 | 0.612 | [0.395, 0.782] |
| 13 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.575 | 0.575 | [0.363, 0.744] |
| 14 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.567 | 0.567 | [0.342, 0.741] |
| 15 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.455 | 0.455 | [0.206, 0.659] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.441 | 0.441 | [0.203, 0.642] |
| 17 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.437 | 0.437 | [0.188, 0.635] |
| 18 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.43 | 0.43 | [0.193, 0.62] |
| 19 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.416 | 0.416 | [0.179, 0.618] |
| 20 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.403 | 0.403 | [0.15, 0.621] |
| 21 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.37 | 0.37 | [0.119, 0.581] |
| 22 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.328 | 0.328 | [0.079, 0.543] |
| 23 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.317 | 0.317 | [0.0582, 0.542] |
| 24 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.316 | 0.316 | [0.0571, 0.549] |
| 25 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.29 | 0.29 | [0.0732, 0.474] |
| 26 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.233 | 0.233 | [-0.0307, 0.467] |
| 27 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | -0.227 | 0.227 | [-0.00201, 0.422] |
| 28 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.164 | 0.164 | [-0.112, 0.407] |
| 29 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.0449 | 0.0449 | [-0.203, 0.315] |
| 30 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.249 | -0.249 | [-0.48, 0.0124] |
| 31 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.283 | -0.283 | [-0.495, -0.0583] |
| 32 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | 0.397 | -0.397 | [-0.577, -0.171] |
| 33 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | 0.397 | -0.397 | [-0.592, -0.177] |
| 34 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | 0.414 | -0.414 | [-0.596, -0.205] |
| 35 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.427 | -0.427 | [-0.606, -0.215] |
| 36 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.428 | -0.428 | [-0.614, -0.206] |
| 37 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.458 | -0.458 | [-0.634, -0.262] |
| 38 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.458 | -0.458 | [-0.615, -0.27] |
| 39 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.469 | -0.469 | [-0.665, -0.208] |
| 40 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.83 | -0.83 | [-0.919, -0.684] |
| 41 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.83 | -0.83 | [-0.919, -0.693] |

## NAS-RobBench-201: epoch_10

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.907 | 0.907 | [0.834, 0.943] |
| 2 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.907 | 0.907 | [0.822, 0.953] |
| 3 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.897 | 0.897 | [0.829, 0.934] |
| 4 | `natural_loss` | Loss control [10] | lower | 64 | -0.87 | 0.87 | [0.765, 0.931] |
| 5 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.782 | 0.782 | [0.635, 0.865] |
| 6 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.707 | 0.707 | [0.539, 0.82] |
| 7 | `feature_delta` | Ours Ours | lower | 64 | -0.697 | 0.697 | [0.518, 0.817] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.69 | 0.69 | [0.517, 0.795] |
| 9 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.684 | 0.684 | [0.512, 0.793] |
| 10 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.677 | 0.677 | [0.487, 0.804] |
| 11 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.645 | 0.645 | [0.459, 0.766] |
| 12 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.622 | 0.622 | [0.424, 0.759] |
| 13 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.622 | 0.622 | [0.416, 0.763] |
| 14 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.618 | 0.618 | [0.387, 0.771] |
| 15 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.55 | 0.55 | [0.334, 0.708] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.516 | 0.516 | [0.278, 0.692] |
| 17 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.502 | 0.502 | [0.262, 0.686] |
| 18 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.47 | 0.47 | [0.229, 0.66] |
| 19 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | -0.465 | 0.465 | [0.232, 0.664] |
| 20 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.417 | 0.417 | [0.17, 0.618] |
| 21 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | -0.389 | 0.389 | [0.136, 0.598] |
| 22 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.383 | 0.383 | [0.162, 0.554] |
| 23 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | -0.339 | 0.339 | [0.0717, 0.562] |
| 24 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.286 | 0.286 | [0.0282, 0.531] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.264 | 0.264 | [0.00939, 0.498] |
| 26 | `feature_measure_local_delta` | Ours Ours | lower | 64 | -0.244 | 0.244 | [-0.0123, 0.481] |
| 27 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | -0.189 | 0.189 | [-0.0763, 0.432] |
| 28 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.187 | 0.187 | [-0.0605, 0.397] |
| 29 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | -0.0694 | 0.0694 | [-0.204, 0.34] |
| 30 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.137 | -0.137 | [-0.374, 0.105] |
| 31 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.216 | -0.216 | [-0.448, 0.042] |
| 32 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.278 | -0.278 | [-0.497, -0.0359] |
| 33 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | 0.304 | -0.304 | [-0.533, -0.0585] |
| 34 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | 0.318 | -0.318 | [-0.535, -0.0687] |
| 35 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.337 | -0.337 | [-0.566, -0.0909] |
| 36 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.373 | -0.373 | [-0.582, -0.117] |
| 37 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.559 | -0.559 | [-0.709, -0.36] |
| 38 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.574 | -0.574 | [-0.721, -0.385] |
| 39 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.671 | -0.671 | [-0.803, -0.492] |
| 40 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.79 | -0.79 | [-0.887, -0.644] |
| 41 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.79 | -0.79 | [-0.889, -0.641] |

## NAS-RobBench-201: epoch_25

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.956 | 0.956 | [0.914, 0.973] |
| 2 | `natural_loss` | Loss control [10] | lower | 64 | -0.953 | 0.953 | [0.901, 0.973] |
| 3 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.91 | 0.91 | [0.835, 0.946] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.906 | 0.906 | [0.839, 0.942] |
| 5 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.768 | 0.768 | [0.611, 0.863] |
| 6 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.719 | 0.719 | [0.534, 0.847] |
| 7 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.673 | 0.673 | [0.51, 0.788] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.648 | 0.648 | [0.482, 0.759] |
| 9 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.632 | 0.632 | [0.462, 0.75] |
| 10 | `feature_delta` | Ours Ours | lower | 64 | -0.601 | 0.601 | [0.377, 0.754] |
| 11 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.527 | 0.527 | [0.266, 0.71] |
| 12 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.517 | 0.517 | [0.314, 0.67] |
| 13 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.375 | 0.375 | [0.115, 0.577] |
| 14 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.332 | 0.332 | [0.0645, 0.549] |
| 15 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.275 | 0.275 | [-0.0173, 0.538] |
| 16 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.264 | 0.264 | [0.0451, 0.449] |
| 17 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | -0.242 | 0.242 | [-0.036, 0.478] |
| 18 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.237 | 0.237 | [-0.0382, 0.486] |
| 19 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.192 | 0.192 | [-0.0916, 0.432] |
| 20 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.181 | 0.181 | [-0.0855, 0.413] |
| 21 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.12 | 0.12 | [-0.176, 0.386] |
| 22 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.104 | 0.104 | [-0.141, 0.351] |
| 23 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.00421 | 0.00421 | [-0.257, 0.27] |
| 24 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | 0.0928 | -0.0928 | [-0.339, 0.17] |
| 25 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | 0.112 | -0.112 | [-0.362, 0.145] |
| 26 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.117 | -0.117 | [-0.394, 0.151] |
| 27 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | 0.119 | -0.119 | [-0.354, 0.13] |
| 28 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | 0.123 | -0.123 | [-0.35, 0.141] |
| 29 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | 0.124 | -0.124 | [-0.363, 0.13] |
| 30 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | 0.149 | -0.149 | [-0.402, 0.0976] |
| 31 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | 0.151 | -0.151 | [-0.384, 0.106] |
| 32 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | 0.167 | -0.167 | [-0.395, 0.0884] |
| 33 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | 0.173 | -0.173 | [-0.417, 0.0677] |
| 34 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.188 | -0.188 | [-0.413, 0.0641] |
| 35 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.258 | -0.258 | [-0.492, -0.0124] |
| 36 | `feature_measure_local_delta` | Ours Ours | lower | 64 | 0.316 | -0.316 | [-0.525, -0.0757] |
| 37 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.409 | -0.409 | [-0.63, -0.148] |
| 38 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.659 | -0.659 | [-0.788, -0.48] |
| 39 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.725 | -0.725 | [-0.837, -0.567] |
| 40 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.725 | -0.725 | [-0.833, -0.565] |
| 41 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.827 | -0.827 | [-0.898, -0.713] |

## NAS-RobBench-201: epoch_50

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `natural_loss` | Loss control [10] | lower | 64 | -0.983 | 0.983 | [0.961, 0.99] |
| 2 | `adversarial_loss` | Loss control [10] | lower | 64 | -0.98 | 0.98 | [0.957, 0.989] |
| 3 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 64 | 0.954 | 0.954 | [0.912, 0.972] |
| 4 | `trnas_r_score` | TRNAS R-Score [11] | higher | 64 | 0.912 | 0.912 | [0.836, 0.949] |
| 5 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 64 | 0.9 | 0.9 | [0.826, 0.942] |
| 6 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 64 | 0.785 | 0.785 | [0.648, 0.87] |
| 7 | `fsp_delta` | FSP [1,2] | lower | 64 | -0.669 | 0.669 | [0.488, 0.783] |
| 8 | `feature_measure_struct01_delta` | Ours Ours | lower | 64 | -0.656 | 0.656 | [0.476, 0.768] |
| 9 | `feature_measure_moment_delta` | Ours Ours | lower | 64 | -0.648 | 0.648 | [0.451, 0.783] |
| 10 | `feature_measure_struct_delta` | Ours Ours | lower | 64 | -0.566 | 0.566 | [0.361, 0.699] |
| 11 | `feature_delta` | Ours Ours | lower | 64 | -0.514 | 0.514 | [0.255, 0.706] |
| 12 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 64 | -0.498 | 0.498 | [0.219, 0.697] |
| 13 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 64 | -0.454 | 0.454 | [0.248, 0.622] |
| 14 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 64 | -0.443 | 0.443 | [0.165, 0.65] |
| 15 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.364 | 0.364 | [0.112, 0.555] |
| 16 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 64 | -0.35 | 0.35 | [0.103, 0.533] |
| 17 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 64 | -0.315 | 0.315 | [0.0128, 0.56] |
| 18 | `feature_measure_flow_delta` | Ours Ours | lower | 64 | -0.214 | 0.214 | [-0.0473, 0.463] |
| 19 | `perturbation_capacity_delta` | Ours Ours | lower | 64 | -0.208 | 0.208 | [-0.0803, 0.473] |
| 20 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 64 | -0.208 | 0.208 | [-0.0415, 0.445] |
| 21 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 64 | -0.101 | 0.101 | [-0.177, 0.365] |
| 22 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 64 | -0.0849 | 0.0849 | [-0.135, 0.302] |
| 23 | `feature_measure_relu_active_delta` | Ours Ours | lower | 64 | -0.0432 | 0.0432 | [-0.185, 0.247] |
| 24 | `feature_measure_mmd` | Ours Ours | lower | 64 | -0.0159 | 0.0159 | [-0.269, 0.299] |
| 25 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 64 | -0.0153 | 0.0153 | [-0.213, 0.244] |
| 26 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 64 | -0.000504 | 0.000504 | [-0.215, 0.237] |
| 27 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 64 | 0.00339 | -0.00339 | [-0.224, 0.208] |
| 28 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 64 | 0.0305 | -0.0305 | [-0.27, 0.191] |
| 29 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 64 | 0.0479 | -0.0479 | [-0.264, 0.175] |
| 30 | `feature_measure_weighted_delta` | Ours Ours | lower | 64 | 0.0748 | -0.0748 | [-0.313, 0.152] |
| 31 | `feature_measure_robust_delta` | Ours Ours | lower | 64 | 0.0762 | -0.0762 | [-0.319, 0.179] |
| 32 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 64 | 0.169 | -0.169 | [-0.444, 0.112] |
| 33 | `feature_measure_local_delta` | Ours Ours | lower | 64 | 0.174 | -0.174 | [-0.371, 0.0443] |
| 34 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 64 | 0.475 | -0.475 | [-0.678, -0.226] |
| 35 | `fsp_strict_delta` | FSP [1,2] | lower | 64 | 0.53 | -0.53 | [-0.705, -0.279] |
| 36 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 64 | 0.544 | -0.544 | [-0.734, -0.314] |
| 37 | `input_loss_delta` | Input-loss change [8,9] | lower | 64 | 0.63 | -0.63 | [-0.767, -0.426] |
| 38 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 64 | 0.842 | -0.842 | [-0.915, -0.724] |
| 39 | `input_grad_l2` | Input gradient [4,7] | lower | 64 | 0.844 | -0.844 | [-0.917, -0.723] |
| 40 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 64 | 0.844 | -0.844 | [-0.918, -0.723] |
| 41 | `logit_kl` | TRADES divergence [3] | lower | 64 | 0.933 | -0.933 | [-0.965, -0.861] |
