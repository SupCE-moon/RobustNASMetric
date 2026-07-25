# Cross-Space Robust Metric Correlation

Target: `full_pgd20_accuracy`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## DARTS-RACL-20: epoch_0

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.776 | 0.776 | [0.506, 0.903] |
| 2 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.75 | 0.75 | [0.466, 0.909] |
| 3 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.501 | 0.501 | [0.0166, 0.794] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.499 | 0.499 | [0.0292, 0.824] |
| 5 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | -0.483 | 0.483 | [-0.00625, 0.804] |
| 6 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.456 | 0.456 | [-0.054, 0.762] |
| 7 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.456 | 0.456 | [-0.0312, 0.77] |
| 8 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.329 | 0.329 | [-0.209, 0.744] |
| 9 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.304 | 0.304 | [-0.181, 0.704] |
| 10 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.293 | 0.293 | [-0.234, 0.717] |
| 11 | `feature_delta` | Ours Ours | lower | 20 | -0.286 | 0.286 | [-0.253, 0.735] |
| 12 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.281 | 0.281 | [-0.265, 0.715] |
| 13 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.277 | 0.277 | [-0.283, 0.715] |
| 14 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.233 | 0.233 | [-0.325, 0.667] |
| 15 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | -0.215 | 0.215 | [-0.324, 0.667] |
| 16 | `logit_kl` | TRADES divergence [3] | lower | 20 | -0.195 | 0.195 | [-0.384, 0.676] |
| 17 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.185 | 0.185 | [-0.295, 0.666] |
| 18 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | -0.185 | 0.185 | [-0.314, 0.636] |
| 19 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | -0.182 | 0.182 | [-0.297, 0.621] |
| 20 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.168 | 0.168 | [-0.34, 0.627] |
| 21 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.165 | 0.165 | [-0.388, 0.619] |
| 22 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | -0.158 | 0.158 | [-0.335, 0.598] |
| 23 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | -0.155 | 0.155 | [-0.369, 0.607] |
| 24 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.143 | 0.143 | [-0.371, 0.61] |
| 25 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.135 | 0.135 | [-0.442, 0.626] |
| 26 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.131 | 0.131 | [-0.373, 0.605] |
| 27 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.131 | 0.131 | [-0.401, 0.6] |
| 28 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.104 | 0.104 | [-0.415, 0.569] |
| 29 | `feature_measure_mmd` | Ours Ours | lower | 20 | -0.0947 | 0.0947 | [-0.373, 0.554] |
| 30 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | -0.0902 | 0.0902 | [-0.432, 0.559] |
| 31 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | -0.0722 | 0.0722 | [-0.455, 0.544] |
| 32 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.0571 | 0.0571 | [-0.485, 0.54] |
| 33 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.0571 | 0.0571 | [-0.405, 0.501] |
| 34 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.0511 | 0.0511 | [-0.396, 0.483] |
| 35 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.0466 | 0.0466 | [-0.482, 0.545] |
| 36 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | -0.0361 | 0.0361 | [-0.492, 0.527] |
| 37 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.0015 | 0.0015 | [-0.47, 0.479] |
| 38 | `natural_loss` | Loss control [10] | lower | 20 | 0.012 | -0.012 | [-0.442, 0.442] |
| 39 | `adversarial_loss` | Loss control [10] | lower | 20 | 0.0195 | -0.0195 | [-0.452, 0.429] |
| 40 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | 0.14 | -0.14 | [-0.597, 0.336] |
| 41 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | 0.558 | -0.558 | [-0.742, -0.188] |

## DARTS-RACL-20: epoch_50

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.789 | 0.789 | [0.499, 0.953] |
| 2 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.771 | 0.771 | [0.51, 0.906] |
| 3 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.746 | 0.746 | [0.476, 0.889] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.701 | 0.701 | [0.376, 0.878] |
| 5 | `natural_loss` | Loss control [10] | lower | 20 | -0.695 | 0.695 | [0.383, 0.866] |
| 6 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.69 | 0.69 | [0.302, 0.879] |
| 7 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.621 | 0.621 | [0.191, 0.888] |
| 8 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.388 | 0.388 | [-0.102, 0.768] |
| 9 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.268 | 0.268 | [-0.22, 0.658] |
| 10 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.262 | 0.262 | [-0.26, 0.674] |
| 11 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.0962 | 0.0962 | [-0.444, 0.61] |
| 12 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.0812 | 0.0812 | [-0.417, 0.577] |
| 13 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.0767 | 0.0767 | [-0.391, 0.53] |
| 14 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.0722 | 0.0722 | [-0.438, 0.538] |
| 15 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.0226 | 0.0226 | [-0.451, 0.505] |
| 16 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0226 | 0.0226 | [-0.45, 0.462] |
| 17 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.018 | 0.018 | [-0.444, 0.475] |
| 18 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.015 | 0.015 | [-0.477, 0.504] |
| 19 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.012 | 0.012 | [-0.493, 0.511] |
| 20 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.012 | 0.012 | [-0.502, 0.513] |
| 21 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.012 | 0.012 | [-0.453, 0.485] |
| 22 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.00451 | 0.00451 | [-0.433, 0.404] |
| 23 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.0271 | -0.0271 | [-0.513, 0.459] |
| 24 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | 0.0436 | -0.0436 | [-0.51, 0.435] |
| 25 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.0481 | -0.0481 | [-0.498, 0.414] |
| 26 | `fsp_delta` | FSP [1,2] | lower | 20 | 0.0571 | -0.0571 | [-0.496, 0.422] |
| 27 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.0586 | -0.0586 | [-0.505, 0.424] |
| 28 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.0602 | -0.0602 | [-0.458, 0.392] |
| 29 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.0617 | -0.0617 | [-0.492, 0.42] |
| 30 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | 0.0647 | -0.0647 | [-0.496, 0.427] |
| 31 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.0737 | -0.0737 | [-0.497, 0.382] |
| 32 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | 0.0767 | -0.0767 | [-0.521, 0.381] |
| 33 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | 0.0812 | -0.0812 | [-0.522, 0.396] |
| 34 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0902 | -0.0902 | [-0.514, 0.375] |
| 35 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.0962 | -0.0962 | [-0.526, 0.383] |
| 36 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0962 | -0.0962 | [-0.54, 0.373] |
| 37 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.123 | -0.123 | [-0.542, 0.357] |
| 38 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.132 | -0.132 | [-0.607, 0.403] |
| 39 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.164 | -0.164 | [-0.583, 0.253] |
| 40 | `feature_delta` | Ours Ours | lower | 20 | 0.176 | -0.176 | [-0.598, 0.293] |
| 41 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.388 | -0.388 | [-0.64, -0.0106] |

## DARTS-RACL-20: epoch_100

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.764 | 0.764 | [0.503, 0.902] |
| 2 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.753 | 0.753 | [0.481, 0.897] |
| 3 | `natural_loss` | Loss control [10] | lower | 20 | -0.738 | 0.738 | [0.388, 0.906] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.71 | 0.71 | [0.398, 0.891] |
| 5 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.693 | 0.693 | [0.305, 0.899] |
| 6 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.69 | 0.69 | [0.302, 0.879] |
| 7 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.421 | 0.421 | [-0.0246, 0.751] |
| 8 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.382 | 0.382 | [-0.0492, 0.703] |
| 9 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.365 | 0.365 | [-0.0918, 0.733] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.262 | 0.262 | [-0.185, 0.625] |
| 11 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.23 | 0.23 | [-0.275, 0.644] |
| 12 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.107 | 0.107 | [-0.391, 0.615] |
| 13 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.107 | 0.107 | [-0.395, 0.618] |
| 14 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.0857 | 0.0857 | [-0.416, 0.549] |
| 15 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.0842 | 0.0842 | [-0.398, 0.556] |
| 16 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.0571 | 0.0571 | [-0.425, 0.514] |
| 17 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.0451 | 0.0451 | [-0.413, 0.517] |
| 18 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.0436 | 0.0436 | [-0.422, 0.488] |
| 19 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.0406 | 0.0406 | [-0.448, 0.502] |
| 20 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.0346 | 0.0346 | [-0.41, 0.494] |
| 21 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.0211 | 0.0211 | [-0.459, 0.492] |
| 22 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0195 | 0.0195 | [-0.455, 0.468] |
| 23 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.018 | 0.018 | [-0.434, 0.474] |
| 24 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.0135 | 0.0135 | [-0.451, 0.495] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.00451 | 0.00451 | [-0.442, 0.462] |
| 26 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.0015 | -0.0015 | [-0.467, 0.464] |
| 27 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.00451 | -0.00451 | [-0.481, 0.475] |
| 28 | `feature_measure_local_delta` | Ours Ours | lower | 20 | 0.012 | -0.012 | [-0.468, 0.484] |
| 29 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | 0.0451 | -0.0451 | [-0.485, 0.41] |
| 30 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | 0.0451 | -0.0451 | [-0.476, 0.447] |
| 31 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0647 | -0.0647 | [-0.502, 0.389] |
| 32 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.0677 | -0.0677 | [-0.545, 0.388] |
| 33 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.0692 | -0.0692 | [-0.497, 0.425] |
| 34 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0707 | -0.0707 | [-0.517, 0.387] |
| 35 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.0767 | -0.0767 | [-0.52, 0.391] |
| 36 | `feature_delta` | Ours Ours | lower | 20 | 0.0887 | -0.0887 | [-0.52, 0.378] |
| 37 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.12 | -0.12 | [-0.534, 0.355] |
| 38 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | 0.123 | -0.123 | [-0.579, 0.394] |
| 39 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.156 | -0.156 | [-0.536, 0.253] |
| 40 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.335 | -0.335 | [-0.718, 0.153] |
| 41 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.37 | -0.37 | [-0.705, 0.0824] |
