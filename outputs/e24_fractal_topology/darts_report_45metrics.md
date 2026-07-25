# Cross-Space Robust Metric Correlation

Target: `full_pgd20_accuracy`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## DARTS-RACL-20: epoch_0

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.776 | 0.776 | [0.5, 0.898] |
| 2 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.75 | 0.75 | [0.462, 0.911] |
| 3 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.501 | 0.501 | [0.00457, 0.804] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.499 | 0.499 | [0.0168, 0.822] |
| 5 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | -0.483 | 0.483 | [0.0175, 0.8] |
| 6 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.456 | 0.456 | [-0.0523, 0.771] |
| 7 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.456 | 0.456 | [-0.0637, 0.77] |
| 8 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.329 | 0.329 | [-0.217, 0.741] |
| 9 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.304 | 0.304 | [-0.171, 0.707] |
| 10 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 20 | -0.304 | 0.304 | [-0.177, 0.702] |
| 11 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.293 | 0.293 | [-0.235, 0.711] |
| 12 | `feature_delta` | Ours Ours | lower | 20 | -0.286 | 0.286 | [-0.263, 0.746] |
| 13 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.281 | 0.281 | [-0.243, 0.712] |
| 14 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.277 | 0.277 | [-0.258, 0.703] |
| 15 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.233 | 0.233 | [-0.311, 0.67] |
| 16 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | -0.215 | 0.215 | [-0.3, 0.659] |
| 17 | `feature_hausdorff_delta` | Ours Ours | lower | 20 | -0.197 | 0.197 | [-0.274, 0.626] |
| 18 | `logit_kl` | TRADES divergence [3] | lower | 20 | -0.195 | 0.195 | [-0.368, 0.666] |
| 19 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.185 | 0.185 | [-0.305, 0.665] |
| 20 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | -0.185 | 0.185 | [-0.319, 0.632] |
| 21 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | -0.182 | 0.182 | [-0.318, 0.625] |
| 22 | `feature_box_dimension_delta` | Ours Ours | lower | 20 | -0.179 | 0.179 | [-0.372, 0.704] |
| 23 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.168 | 0.168 | [-0.318, 0.624] |
| 24 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.165 | 0.165 | [-0.383, 0.629] |
| 25 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | -0.158 | 0.158 | [-0.341, 0.603] |
| 26 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | -0.155 | 0.155 | [-0.379, 0.601] |
| 27 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.143 | 0.143 | [-0.357, 0.595] |
| 28 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.135 | 0.135 | [-0.452, 0.616] |
| 29 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.131 | 0.131 | [-0.406, 0.606] |
| 30 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.131 | 0.131 | [-0.384, 0.612] |
| 31 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 20 | -0.131 | 0.131 | [-0.342, 0.581] |
| 32 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.104 | 0.104 | [-0.442, 0.575] |
| 33 | `feature_measure_mmd` | Ours Ours | lower | 20 | -0.0947 | 0.0947 | [-0.379, 0.557] |
| 34 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | -0.0902 | 0.0902 | [-0.444, 0.566] |
| 35 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | -0.0722 | 0.0722 | [-0.454, 0.549] |
| 36 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.0571 | 0.0571 | [-0.401, 0.501] |
| 37 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.0571 | 0.0571 | [-0.488, 0.551] |
| 38 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.0511 | 0.0511 | [-0.412, 0.495] |
| 39 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.0466 | 0.0466 | [-0.493, 0.539] |
| 40 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | -0.0361 | 0.0361 | [-0.511, 0.521] |
| 41 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.0015 | 0.0015 | [-0.466, 0.481] |
| 42 | `natural_loss` | Loss control [10] | lower | 20 | 0.012 | -0.012 | [-0.445, 0.439] |
| 43 | `adversarial_loss` | Loss control [10] | lower | 20 | 0.0195 | -0.0195 | [-0.464, 0.433] |
| 44 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | 0.14 | -0.14 | [-0.59, 0.34] |
| 45 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | 0.558 | -0.558 | [-0.74, -0.214] |

## DARTS-RACL-20: epoch_50

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.789 | 0.789 | [0.512, 0.95] |
| 2 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.771 | 0.771 | [0.498, 0.905] |
| 3 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.746 | 0.746 | [0.471, 0.889] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.701 | 0.701 | [0.373, 0.877] |
| 5 | `natural_loss` | Loss control [10] | lower | 20 | -0.695 | 0.695 | [0.369, 0.87] |
| 6 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.69 | 0.69 | [0.328, 0.879] |
| 7 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.621 | 0.621 | [0.193, 0.889] |
| 8 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.388 | 0.388 | [-0.104, 0.763] |
| 9 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.268 | 0.268 | [-0.22, 0.659] |
| 10 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.262 | 0.262 | [-0.262, 0.683] |
| 11 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.0962 | 0.0962 | [-0.451, 0.599] |
| 12 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.0812 | 0.0812 | [-0.424, 0.567] |
| 13 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.0767 | 0.0767 | [-0.387, 0.528] |
| 14 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.0722 | 0.0722 | [-0.435, 0.557] |
| 15 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.0226 | 0.0226 | [-0.453, 0.527] |
| 16 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0226 | 0.0226 | [-0.439, 0.454] |
| 17 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.018 | 0.018 | [-0.434, 0.473] |
| 18 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.015 | 0.015 | [-0.454, 0.491] |
| 19 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.012 | 0.012 | [-0.461, 0.49] |
| 20 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.012 | 0.012 | [-0.498, 0.514] |
| 21 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.012 | 0.012 | [-0.489, 0.508] |
| 22 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 20 | -0.0105 | 0.0105 | [-0.418, 0.421] |
| 23 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.00451 | 0.00451 | [-0.434, 0.401] |
| 24 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.0271 | -0.0271 | [-0.512, 0.482] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | 0.0436 | -0.0436 | [-0.48, 0.431] |
| 26 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.0481 | -0.0481 | [-0.484, 0.421] |
| 27 | `fsp_delta` | FSP [1,2] | lower | 20 | 0.0571 | -0.0571 | [-0.503, 0.411] |
| 28 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.0586 | -0.0586 | [-0.502, 0.426] |
| 29 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.0602 | -0.0602 | [-0.457, 0.412] |
| 30 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.0617 | -0.0617 | [-0.502, 0.412] |
| 31 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | 0.0647 | -0.0647 | [-0.51, 0.397] |
| 32 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.0737 | -0.0737 | [-0.507, 0.406] |
| 33 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | 0.0767 | -0.0767 | [-0.52, 0.378] |
| 34 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | 0.0812 | -0.0812 | [-0.515, 0.403] |
| 35 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0902 | -0.0902 | [-0.519, 0.378] |
| 36 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.0962 | -0.0962 | [-0.532, 0.369] |
| 37 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0962 | -0.0962 | [-0.535, 0.38] |
| 38 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.123 | -0.123 | [-0.55, 0.373] |
| 39 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.132 | -0.132 | [-0.606, 0.392] |
| 40 | `feature_hausdorff_delta` | Ours Ours | lower | 20 | 0.144 | -0.144 | [-0.614, 0.374] |
| 41 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.164 | -0.164 | [-0.566, 0.248] |
| 42 | `feature_delta` | Ours Ours | lower | 20 | 0.176 | -0.176 | [-0.606, 0.304] |
| 43 | `feature_box_dimension_delta` | Ours Ours | lower | 20 | 0.183 | -0.183 | [-0.566, 0.264] |
| 44 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 20 | 0.208 | -0.208 | [-0.622, 0.281] |
| 45 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.388 | -0.388 | [-0.649, 0.0184] |

## DARTS-RACL-20: epoch_100

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.764 | 0.764 | [0.493, 0.897] |
| 2 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.753 | 0.753 | [0.476, 0.895] |
| 3 | `natural_loss` | Loss control [10] | lower | 20 | -0.738 | 0.738 | [0.4, 0.907] |
| 4 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.71 | 0.71 | [0.379, 0.895] |
| 5 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.693 | 0.693 | [0.301, 0.894] |
| 6 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.69 | 0.69 | [0.328, 0.879] |
| 7 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.421 | 0.421 | [-0.0349, 0.752] |
| 8 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.382 | 0.382 | [-0.0477, 0.7] |
| 9 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.365 | 0.365 | [-0.0975, 0.722] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.262 | 0.262 | [-0.179, 0.626] |
| 11 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.23 | 0.23 | [-0.275, 0.655] |
| 12 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.107 | 0.107 | [-0.392, 0.622] |
| 13 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.107 | 0.107 | [-0.389, 0.605] |
| 14 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.0857 | 0.0857 | [-0.417, 0.56] |
| 15 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.0842 | 0.0842 | [-0.408, 0.55] |
| 16 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.0571 | 0.0571 | [-0.415, 0.497] |
| 17 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.0451 | 0.0451 | [-0.418, 0.516] |
| 18 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.0436 | 0.0436 | [-0.41, 0.464] |
| 19 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.0406 | 0.0406 | [-0.433, 0.49] |
| 20 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.0346 | 0.0346 | [-0.411, 0.466] |
| 21 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.0211 | 0.0211 | [-0.46, 0.502] |
| 22 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0195 | 0.0195 | [-0.456, 0.457] |
| 23 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.018 | 0.018 | [-0.414, 0.475] |
| 24 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.0135 | 0.0135 | [-0.452, 0.494] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.00451 | 0.00451 | [-0.436, 0.458] |
| 26 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.0015 | -0.0015 | [-0.471, 0.472] |
| 27 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.00451 | -0.00451 | [-0.49, 0.5] |
| 28 | `feature_measure_local_delta` | Ours Ours | lower | 20 | 0.012 | -0.012 | [-0.464, 0.474] |
| 29 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | 0.0451 | -0.0451 | [-0.478, 0.396] |
| 30 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | 0.0451 | -0.0451 | [-0.481, 0.444] |
| 31 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0647 | -0.0647 | [-0.502, 0.393] |
| 32 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.0677 | -0.0677 | [-0.535, 0.401] |
| 33 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.0692 | -0.0692 | [-0.505, 0.415] |
| 34 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0707 | -0.0707 | [-0.512, 0.393] |
| 35 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.0767 | -0.0767 | [-0.511, 0.379] |
| 36 | `feature_delta` | Ours Ours | lower | 20 | 0.0887 | -0.0887 | [-0.517, 0.401] |
| 37 | `feature_box_dimension_delta` | Ours Ours | lower | 20 | 0.114 | -0.114 | [-0.521, 0.353] |
| 38 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.12 | -0.12 | [-0.541, 0.379] |
| 39 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | 0.123 | -0.123 | [-0.579, 0.385] |
| 40 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.156 | -0.156 | [-0.53, 0.254] |
| 41 | `feature_hausdorff_delta` | Ours Ours | lower | 20 | 0.188 | -0.188 | [-0.608, 0.297] |
| 42 | `feature_graph_fundamental_rank_delta` | Ours Ours | lower | 20 | 0.239 | -0.239 | [-0.599, 0.3] |
| 43 | `feature_quantile_hausdorff_delta` | Ours Ours | lower | 20 | 0.248 | -0.248 | [-0.684, 0.244] |
| 44 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.335 | -0.335 | [-0.728, 0.133] |
| 45 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.37 | -0.37 | [-0.701, 0.0608] |
