# Cross-Space Robust Metric Correlation

Target: `final_pgd_accuracy`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## RACL-DARTS-search-macro: epoch_0

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.628 | 0.628 | [0.203, 0.851] |
| 2 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.578 | 0.578 | [0.108, 0.893] |
| 3 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.547 | 0.547 | [0.076, 0.874] |
| 4 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | -0.288 | 0.288 | [-0.221, 0.721] |
| 5 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.239 | 0.239 | [-0.237, 0.69] |
| 6 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.234 | 0.234 | [-0.295, 0.647] |
| 7 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.175 | 0.175 | [-0.317, 0.637] |
| 8 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.175 | 0.175 | [-0.309, 0.635] |
| 9 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | -0.174 | 0.174 | [-0.351, 0.624] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.164 | 0.164 | [-0.288, 0.594] |
| 11 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.154 | 0.154 | [-0.377, 0.648] |
| 12 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.148 | 0.148 | [-0.286, 0.578] |
| 13 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.147 | 0.147 | [-0.388, 0.629] |
| 14 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.145 | 0.145 | [-0.392, 0.62] |
| 15 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.131 | 0.131 | [-0.342, 0.542] |
| 16 | `natural_loss` | Loss control [10] | lower | 20 | -0.122 | 0.122 | [-0.326, 0.522] |
| 17 | `feature_delta` | Ours Ours | lower | 20 | -0.0813 | 0.0813 | [-0.472, 0.564] |
| 18 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.0768 | 0.0768 | [-0.465, 0.55] |
| 19 | `logit_kl` | TRADES divergence [3] | lower | 20 | -0.0452 | 0.0452 | [-0.499, 0.54] |
| 20 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.0113 | 0.0113 | [-0.528, 0.501] |
| 21 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.00753 | -0.00753 | [-0.509, 0.484] |
| 22 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.012 | -0.012 | [-0.504, 0.479] |
| 23 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.0233 | -0.0233 | [-0.518, 0.473] |
| 24 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.0248 | -0.0248 | [-0.519, 0.456] |
| 25 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | 0.0301 | -0.0301 | [-0.584, 0.493] |
| 26 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | 0.0354 | -0.0354 | [-0.52, 0.463] |
| 27 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | 0.0422 | -0.0422 | [-0.554, 0.475] |
| 28 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | 0.0422 | -0.0422 | [-0.564, 0.454] |
| 29 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.0437 | -0.0437 | [-0.569, 0.452] |
| 30 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.0467 | -0.0467 | [-0.563, 0.446] |
| 31 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.0482 | -0.0482 | [-0.549, 0.437] |
| 32 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.0572 | -0.0572 | [-0.569, 0.44] |
| 33 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | 0.0693 | -0.0693 | [-0.602, 0.434] |
| 34 | `feature_measure_local_delta` | Ours Ours | lower | 20 | 0.0821 | -0.0821 | [-0.588, 0.424] |
| 35 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.0851 | -0.0851 | [-0.621, 0.43] |
| 36 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | 0.148 | -0.148 | [-0.606, 0.338] |
| 37 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | 0.159 | -0.159 | [-0.621, 0.421] |
| 38 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.175 | -0.175 | [-0.581, 0.291] |
| 39 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | 0.207 | -0.207 | [-0.623, 0.287] |
| 40 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | 0.299 | -0.299 | [-0.706, 0.222] |
| 41 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | 0.357 | -0.357 | [-0.719, 0.143] |

## RACL-DARTS-search-macro: epoch_50

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.587 | 0.587 | [0.142, 0.889] |
| 2 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.58 | 0.58 | [0.137, 0.878] |
| 3 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.546 | 0.546 | [0.0874, 0.854] |
| 4 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.508 | 0.508 | [-0.0122, 0.839] |
| 5 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.459 | 0.459 | [-0.0154, 0.804] |
| 6 | `natural_loss` | Loss control [10] | lower | 20 | -0.444 | 0.444 | [-0.0556, 0.759] |
| 7 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.414 | 0.414 | [-0.0464, 0.743] |
| 8 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.163 | 0.163 | [-0.332, 0.597] |
| 9 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | -0.135 | 0.135 | [-0.322, 0.513] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.117 | 0.117 | [-0.397, 0.594] |
| 11 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0949 | 0.0949 | [-0.4, 0.509] |
| 12 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.0663 | 0.0663 | [-0.449, 0.535] |
| 13 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | 0.00377 | -0.00377 | [-0.486, 0.473] |
| 14 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | 0.052 | -0.052 | [-0.537, 0.442] |
| 15 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | 0.073 | -0.073 | [-0.517, 0.373] |
| 16 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | 0.073 | -0.073 | [-0.539, 0.381] |
| 17 | `fsp_delta` | FSP [1,2] | lower | 20 | 0.0783 | -0.0783 | [-0.558, 0.422] |
| 18 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | 0.0896 | -0.0896 | [-0.571, 0.421] |
| 19 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.0941 | -0.0941 | [-0.567, 0.426] |
| 20 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | 0.0979 | -0.0979 | [-0.589, 0.413] |
| 21 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.106 | -0.106 | [-0.571, 0.438] |
| 22 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | 0.143 | -0.143 | [-0.619, 0.407] |
| 23 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | 0.161 | -0.161 | [-0.609, 0.34] |
| 24 | `feature_delta` | Ours Ours | lower | 20 | 0.179 | -0.179 | [-0.601, 0.345] |
| 25 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | 0.184 | -0.184 | [-0.627, 0.304] |
| 26 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.202 | -0.202 | [-0.62, 0.297] |
| 27 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | 0.244 | -0.244 | [-0.639, 0.277] |
| 28 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.256 | -0.256 | [-0.661, 0.247] |
| 29 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.259 | -0.259 | [-0.667, 0.248] |
| 30 | `feature_measure_local_delta` | Ours Ours | lower | 20 | 0.269 | -0.269 | [-0.681, 0.256] |
| 31 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | 0.274 | -0.274 | [-0.686, 0.263] |
| 32 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | 0.276 | -0.276 | [-0.689, 0.216] |
| 33 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | 0.279 | -0.279 | [-0.676, 0.225] |
| 34 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.309 | -0.309 | [-0.745, 0.211] |
| 35 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.319 | -0.319 | [-0.709, 0.166] |
| 36 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.345 | -0.345 | [-0.711, 0.124] |
| 37 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.346 | -0.346 | [-0.735, 0.156] |
| 38 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | 0.351 | -0.351 | [-0.764, 0.145] |
| 39 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.353 | -0.353 | [-0.729, 0.137] |
| 40 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.354 | -0.354 | [-0.769, 0.12] |
| 41 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.367 | -0.367 | [-0.741, 0.149] |

## RACL-DARTS-search-macro: epoch_100

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | 0.596 | 0.596 | [0.177, 0.868] |
| 2 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.593 | 0.593 | [0.191, 0.867] |
| 3 | `natural_loss` | Loss control [10] | lower | 20 | -0.591 | 0.591 | [0.123, 0.883] |
| 4 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | 0.574 | 0.574 | [0.115, 0.884] |
| 5 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | 0.559 | 0.559 | [0.104, 0.858] |
| 6 | `racl_arch_lipschitz_log_upper` | RACL fixed-architecture counterpart [5] | lower | 20 | -0.508 | 0.508 | [-0.0122, 0.839] |
| 7 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | 0.392 | 0.392 | [-0.0459, 0.738] |
| 8 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.277 | 0.277 | [-0.191, 0.65] |
| 9 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.224 | 0.224 | [-0.246, 0.615] |
| 10 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.134 | 0.134 | [-0.382, 0.597] |
| 11 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.114 | 0.114 | [-0.349, 0.545] |
| 12 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | -0.0761 | 0.0761 | [-0.371, 0.472] |
| 13 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | -0.0369 | 0.0369 | [-0.439, 0.457] |
| 14 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | 0.0429 | -0.0429 | [-0.524, 0.474] |
| 15 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | 0.0489 | -0.0489 | [-0.549, 0.437] |
| 16 | `fsp_delta` | FSP [1,2] | lower | 20 | 0.061 | -0.061 | [-0.55, 0.46] |
| 17 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | 0.115 | -0.115 | [-0.596, 0.398] |
| 18 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | 0.121 | -0.121 | [-0.572, 0.347] |
| 19 | `feature_delta` | Ours Ours | lower | 20 | 0.125 | -0.125 | [-0.596, 0.402] |
| 20 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | 0.133 | -0.133 | [-0.565, 0.355] |
| 21 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | 0.133 | -0.133 | [-0.563, 0.361] |
| 22 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | 0.178 | -0.178 | [-0.603, 0.284] |
| 23 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | 0.201 | -0.201 | [-0.643, 0.327] |
| 24 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | 0.22 | -0.22 | [-0.634, 0.285] |
| 25 | `feature_measure_mmd` | Ours Ours | lower | 20 | 0.221 | -0.221 | [-0.641, 0.288] |
| 26 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.232 | -0.232 | [-0.641, 0.248] |
| 27 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | 0.237 | -0.237 | [-0.643, 0.261] |
| 28 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | 0.25 | -0.25 | [-0.699, 0.246] |
| 29 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | 0.267 | -0.267 | [-0.713, 0.204] |
| 30 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | 0.306 | -0.306 | [-0.74, 0.2] |
| 31 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | 0.321 | -0.321 | [-0.706, 0.146] |
| 32 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | 0.325 | -0.325 | [-0.703, 0.126] |
| 33 | `feature_measure_local_delta` | Ours Ours | lower | 20 | 0.328 | -0.328 | [-0.72, 0.195] |
| 34 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.34 | -0.34 | [-0.763, 0.18] |
| 35 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | 0.355 | -0.355 | [-0.736, 0.147] |
| 36 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | 0.36 | -0.36 | [-0.759, 0.142] |
| 37 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | 0.364 | -0.364 | [-0.744, 0.0928] |
| 38 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | 0.387 | -0.387 | [-0.747, 0.127] |
| 39 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | 0.389 | -0.389 | [-0.744, 0.104] |
| 40 | `logit_kl` | TRADES divergence [3] | lower | 20 | 0.395 | -0.395 | [-0.685, 0.0591] |
| 41 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | 0.472 | -0.472 | [-0.789, 0.0317] |
