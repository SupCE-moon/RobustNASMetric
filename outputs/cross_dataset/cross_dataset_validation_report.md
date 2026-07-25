# Cross-Dataset Robust Metric Validation

Date: 2026-07-07

## Scope

This report extends the CIFAR10 validation to CIFAR100, ImageNet, MNIST, and SVHN.

- CIFAR100 and ImageNet use RobustBench models and RobustBench official robust accuracy as the ranking target.
- MNIST and SVHN are not supported by the RobustBench model zoo used here, so they use local legacy checkpoints and the same-run white-box FGSM adversarial accuracy as an internal validation target.
- Common datasets were reused from existing server copies. No dataset was downloaded again.

## Data And Execution

| Dataset | Model source | Data source | Run location | Target |
|---|---|---|---|---|
| CIFAR100 | RobustBench, 10 models | copied from 516 4090 `/root/data/cifar-100-python` | local RTX 4060 Ti | RobustBench official robust acc |
| ImageNet | RobustBench, 5 models | A100 `/home/ubuntu/hy/data/imagenet/val` | Chengdu A100 docker | RobustBench official robust acc |
| MNIST | legacy WRN/denoise checkpoints, 6 loaded | copied from 516 4090 `/root/data/MNIST` | local RTX 4060 Ti | same-run FGSM adv_acc |
| SVHN | legacy WRN/denoise checkpoints, 8 loaded | copied from A100 `/home/ubuntu/hy/data/svhn` | local RTX 4060 Ti | same-run FGSM adv_acc |

## Output Files

| Dataset | Metrics CSV |
|---|---|
| CIFAR100 | `outputs/cross_dataset/metrics_cifar100_linf10_fgsm_n128_with_rb_scores.csv` |
| ImageNet | `outputs/cross_dataset/metrics_imagenet_linf5_fgsm_n64_with_rb_scores.csv` |
| MNIST | `outputs/cross_dataset/metrics_mnist_legacy_fgsm_n128_seed20260704.csv` |
| SVHN | `outputs/cross_dataset/metrics_svhn_legacy_fgsm_n128_seed20260704.csv` |

## Compact Result Summary

Values below are absolute Spearman correlations. Direction is omitted here but available in the command outputs.

| Dataset | Models | Target | Best Ours | FSP | input_grad_l2 | local_logit_lipschitz |
|---|---:|---|---:|---:|---:|---:|
| CIFAR100 | 10 | RobustBench robust acc | 0.3697 | 0.0667 | 0.5152 | 0.7212 |
| ImageNet | 5 | RobustBench robust acc | 0.8000 | 0.7000 | 0.5000 | 0.4000 |
| MNIST | 6 | FGSM adv_acc | 0.9276 | 0.7537 | 0.4928 | 0.8407 |
| SVHN | 8 | FGSM adv_acc | 0.5988 | 0.2635 | 0.5868 | 0.3473 |

Best Ours means the strongest metric among:

- `feature_measure_spatial_gram_delta`
- `feature_measure_spatial_gram_rel_delta`
- `feature_measure_robust_spatial_delta`
- `feature_measure_robust_spatial_rel_delta`
- `feature_measure_robust_hybrid_delta`
- `feature_measure_robust_hybrid_rel_delta`
- `feature_measure_flow_rel_delta`

## Dataset-Level Notes

### CIFAR100

The best Ours result is `feature_measure_robust_spatial_rel_delta` with Spearman 0.3697 against RobustBench official robust accuracy. FSP is weak on this subset, while `local_logit_lipschitz` remains strongest.

Interpretation: CIFAR100 is a warning sign that the current Ours metric is meaningful but not yet a universal ranker. It likely still needs dataset/layer/channel normalization and a stronger shared-adv protocol before claiming architecture-level superiority.

### ImageNet

The best Ours result is `feature_measure_spatial_gram_delta` with Spearman 0.8000 against RobustBench official robust accuracy. FSP also performs well at 0.7000.

Interpretation: this is encouraging but very preliminary because only 5 ImageNet models were used. The result supports scaling the metric to high-resolution features, but the sample size is too small for strong claims.

### MNIST

The best Ours result is `feature_measure_robust_hybrid_rel_delta` with Spearman 0.9276 against same-run FGSM adversarial accuracy. FSP is 0.7537 and `local_logit_lipschitz` is 0.8407.

Interpretation: the feature-measure family strongly separates the natural WRN checkpoints from the denoise robust checkpoints. However, this is an internal legacy-checkpoint validation, not an official benchmark comparison.

### SVHN

The best Ours result is `feature_measure_robust_hybrid_delta` with Spearman 0.5988 against same-run FGSM adversarial accuracy. This is slightly stronger than `input_grad_l2` at 0.5868 and stronger than FSP at 0.2635.

Interpretation: SVHN is a useful positive signal because the model pool includes natural, AT, and denoise checkpoints. The metric is competitive with gradient baselines and clearly better than FSP in this first pass.

## Current Limitations

- CIFAR100 and ImageNet are RobustBench-aligned, but MNIST/SVHN are legacy-checkpoint validations.
- MNIST has only 6 successfully loaded checkpoints; 3 AT checkpoints were skipped because their state dict uses a different `stage_list/block_list/mix_bn` architecture.
- ImageNet uses only 5 models and 64 validation images, so it is an exploratory smoke validation.
- The empirical direction of some feature metrics changes by dataset; this suggests we should standardize metric sign/normalization before paper-level reporting.

## Next Recommended Validation

1. Increase CIFAR100 to 20 RobustBench models and 512+ samples.
2. Increase ImageNet to 10-15 RobustBench models if enough weights are available locally or can be transferred.
3. Add a second legacy adapter for the skipped MNIST `stage_list/mix_bn` checkpoints.
4. Re-run MNIST/SVHN with PGD-lite in addition to FGSM.
5. Report both raw Spearman and residual value after controlling for `local_logit_lipschitz` and `input_grad_l2`.
