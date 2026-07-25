# Cross-Space Robust Metric Correlation

Target: `rb_robust_acc`. Directions are fixed in `metric_registry.py` before analysis.
All displayed numbers use three significant digits.

## RobustBench-20: final

| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `adversarial_loss` | Loss control [10] | lower | 20 | -0.805 | 0.805 | [0.511, 0.933] |
| 2 | `perturbation_capacity_relu_active_delta` | Ours Ours | lower | 20 | -0.699 | 0.699 | [0.35, 0.89] |
| 3 | `racl_arch_lipschitz_log_upper` | RACL generic fixed-network proxy [5] | lower | 20 | -0.633 | 0.633 | [0.29, 0.843] |
| 4 | `perturbation_capacity_delta` | Ours Ours | lower | 20 | -0.626 | 0.626 | [0.283, 0.828] |
| 5 | `perturbation_capacity_above_mean_delta` | Ours Ours | lower | 20 | -0.555 | 0.555 | [0.0816, 0.879] |
| 6 | `feature_measure_cov_delta` | Distribution discrepancy baseline | lower | 20 | -0.537 | 0.537 | [0.121, 0.799] |
| 7 | `feature_measure_struct01_delta` | Ours Ours | lower | 20 | -0.499 | 0.499 | [0.0529, 0.799] |
| 8 | `fsp_delta` | FSP [1,2] | lower | 20 | -0.493 | 0.493 | [0.082, 0.787] |
| 9 | `feature_measure_swd` | Distribution discrepancy baseline | lower | 20 | -0.492 | 0.492 | [0.056, 0.768] |
| 10 | `input_loss_delta` | Input-loss change [8,9] | lower | 20 | -0.483 | 0.483 | [-0.0343, 0.824] |
| 11 | `feature_measure_relu_active_spatial_delta` | Ours Ours | lower | 20 | -0.457 | 0.457 | [-0.0422, 0.794] |
| 12 | `feature_measure_struct_delta` | Ours Ours | lower | 20 | -0.456 | 0.456 | [-0.0175, 0.767] |
| 13 | `feature_measure_spatial_gram_delta` | Ours Ours | lower | 20 | -0.454 | 0.454 | [-0.0374, 0.792] |
| 14 | `input_grad_l2` | Input gradient [4,7] | lower | 20 | -0.429 | 0.429 | [-0.101, 0.766] |
| 15 | `racl_input_gradient_lower` | RACL lower term (alias of input_grad_l2) [5] | lower | 20 | -0.429 | 0.429 | [-0.108, 0.795] |
| 16 | `logit_kl` | TRADES divergence [3] | lower | 20 | -0.382 | 0.382 | [-0.152, 0.767] |
| 17 | `feature_measure_moment_delta` | Ours Ours | lower | 20 | -0.38 | 0.38 | [-0.106, 0.753] |
| 18 | `feature_measure_relu_active_delta` | Ours Ours | lower | 20 | -0.355 | 0.355 | [-0.141, 0.706] |
| 19 | `feature_measure_robust_spatial_delta` | Ours Ours | lower | 20 | -0.34 | 0.34 | [-0.138, 0.691] |
| 20 | `feature_measure_robust_spatial_rel_delta` | Ours Ours | lower | 20 | -0.329 | 0.329 | [-0.174, 0.683] |
| 21 | `local_logit_lipschitz` | Local logit sensitivity [5,6] | lower | 20 | -0.329 | 0.329 | [-0.214, 0.731] |
| 22 | `feature_measure_mmd` | Ours Ours | lower | 20 | -0.314 | 0.314 | [-0.157, 0.648] |
| 23 | `feature_measure_relu_active_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.29 | 0.29 | [-0.213, 0.66] |
| 24 | `feature_measure_local_delta` | Ours Ours | lower | 20 | -0.287 | 0.287 | [-0.164, 0.668] |
| 25 | `feature_measure_relu_active_hybrid_delta` | Ours Ours | lower | 20 | -0.286 | 0.286 | [-0.252, 0.691] |
| 26 | `feature_measure_robust_hybrid_rel_delta` | Ours Ours | lower | 20 | -0.283 | 0.283 | [-0.253, 0.666] |
| 27 | `feature_measure_weighted_delta` | Ours Ours | lower | 20 | -0.253 | 0.253 | [-0.205, 0.651] |
| 28 | `feature_measure_robust_hybrid_delta` | Ours Ours | lower | 20 | -0.242 | 0.242 | [-0.277, 0.675] |
| 29 | `feature_measure_spatial_gram_rel_delta` | Ours Ours | lower | 20 | -0.235 | 0.235 | [-0.26, 0.663] |
| 30 | `feature_measure_robust_rel_delta` | Ours Ours | lower | 20 | -0.221 | 0.221 | [-0.26, 0.637] |
| 31 | `feature_measure_relu_active_spatial_rel_delta` | Ours Ours | lower | 20 | -0.209 | 0.209 | [-0.314, 0.647] |
| 32 | `feature_measure_flow_rel_delta` | Ours Ours | lower | 20 | -0.205 | 0.205 | [-0.313, 0.607] |
| 33 | `feature_delta` | Ours Ours | lower | 20 | -0.135 | 0.135 | [-0.338, 0.553] |
| 34 | `feature_measure_robust_delta` | Ours Ours | lower | 20 | 0.015 | -0.015 | [-0.544, 0.492] |
| 35 | `natural_loss` | Loss control [10] | lower | 20 | 0.0226 | -0.0226 | [-0.48, 0.504] |
| 36 | `trnas_r_score` | TRNAS R-Score [11] | higher | 20 | -0.0394 | -0.0394 | [-0.449, 0.399] |
| 37 | `trnas_activation_psi` | TRNAS activation capability [11,12] | higher | 20 | -0.137 | -0.137 | [-0.583, 0.391] |
| 38 | `trnas_consistency_phi` | TRNAS consistency [11] | higher | 20 | -0.143 | -0.143 | [-0.643, 0.32] |
| 39 | `fsp_strict_delta` | FSP [1,2] | lower | 20 | 0.17 | -0.17 | [-0.641, 0.383] |
| 40 | `feature_measure_flow_delta` | Ours Ours | lower | 20 | 0.174 | -0.174 | [-0.657, 0.356] |
| 41 | `margin_input_grad_ratio` | Margin/gradient radius [7] | higher | 20 | -0.235 | -0.235 | [-0.671, 0.305] |
