# Residual-Value and Coefficient Calibration Analysis

Date: 2026-07-07

Input: `outputs\metrics_20_shared_multisource_pgd10_n128_rel_avg_with_rb_scores.csv`

Target: RobustBench official CIFAR-10 Linf robust accuracy.

## Residual-Value Test

This table asks whether an Ours metric improves a strong baseline model of robust accuracy.
`delta_r2` is in-sample incremental explanatory power; `delta_loocv_r2` is leave-one-out predictive change and is more conservative for 20 models.

| baseline | candidate | delta R2 | delta adjusted R2 | delta LOOCV R2 | partial Pearson | partial Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local_logit_lipschitz | `feature_measure_spatial_gram_delta` | 0.00427 | -0.0035 | -0.269 | -0.179 | -0.371 |
| local_logit_lipschitz | `feature_measure_spatial_gram_rel_delta` | 0.000486 | -0.00773 | -0.173 | -0.0604 | -0.271 |
| local_logit_lipschitz | `feature_measure_robust_spatial_delta` | 0.000318 | -0.00792 | -0.0111 | 0.0488 | -0.0749 |
| local_logit_lipschitz | `feature_measure_robust_spatial_rel_delta` | 0.000295 | -0.00795 | -0.00647 | 0.047 | -0.0581 |
| input_grad_l2 | `feature_measure_spatial_gram_delta` | 0.00498 | -0.003 | -3.34 | -0.19 | -0.42 |
| input_grad_l2 | `feature_measure_spatial_gram_rel_delta` | 0.0005 | -0.00801 | -1.55 | -0.0602 | -0.332 |
| input_grad_l2 | `feature_measure_robust_spatial_delta` | 0.000159 | -0.00839 | -0.671 | 0.0339 | -0.184 |
| input_grad_l2 | `feature_measure_robust_spatial_rel_delta` | 0.000237 | -0.0083 | -0.396 | 0.0414 | -0.123 |

## Best Exploratory Coefficient Grid Results

The grid uses z-scored metric components. It is a sensitivity analysis, not a final learned metric.

| combo | best coefficient | Pearson | Spearman | direction |
| --- | ---: | ---: | ---: | --- |
| `hybrid_plus_lambda_weighted` | -3.0 | 0.934 | 0.275 | higher_is_better |
| `hybrid_rel_plus_lambda_weighted` | -0.3 | -0.779 | -0.194 | lower_is_better |
| `input_grad_l2_plus_gamma_feature_measure_spatial_gram_delta` | 0.0 | -0.928 | -0.538 | lower_is_better |
| `input_grad_l2_plus_gamma_feature_measure_spatial_gram_rel_delta` | 0.0 | -0.928 | -0.538 | lower_is_better |
| `local_logit_lipschitz_plus_gamma_feature_measure_spatial_gram_delta` | 0.1 | -0.933 | -0.394 | lower_is_better |
| `local_logit_lipschitz_plus_gamma_feature_measure_spatial_gram_rel_delta` | 0.0 | -0.931 | -0.29 | lower_is_better |
| `spatial_plus_lambda_weighted` | -3.0 | 0.863 | -0.427 | higher_is_better |
| `spatial_rel_plus_lambda_weighted` | -3.0 | 0.894 | -0.331 | higher_is_better |

## Interpretation

A useful positive result would be: adding the spatial feature-relation metric increases adjusted R2 and leave-one-out R2 over `local_logit_lipschitz` or `input_grad_l2`, and the partial correlations are meaningfully nonzero.

If in-sample R2 improves but leave-one-out R2 drops, the metric may contain overlapping or overfit information under the current 20-model setting. In that case, the research value shifts toward architecture interpretability and should be tested on same-training-different-architecture or early-training settings.
