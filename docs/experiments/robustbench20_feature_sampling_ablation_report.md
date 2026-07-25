# E22: RobustBench-20 Feature Sampling Ablation

Date: 2026-07-18

## Objective

This experiment tests two implementation choices shared by all feature-map robustness metrics:

1. whether uniformly sampling more feature maps than the original $K=4$ improves ranking;
2. whether metrics should use only features immediately after an activation function.

The experiment is exploratory hyperparameter analysis. It does not replace the three-source shared-adversarial main leaderboard because the best configuration is selected on the same 20 models used for evaluation.

## Protocol

| Item | Setting |
|---|---|
| Models | 20 CIFAR-10 $L_\infty$ RobustBench models |
| Target | RobustBench official robust accuracy |
| Samples | Same 64 CIFAR-10 test samples for every model and condition |
| Perturbation | Model-specific white-box FGSM, $\epsilon=8/255$ |
| Layer counts | $K\in\{2,4,8,16\}$ |
| Seed | 20260704 |
| Server | Chengdu 516 RTX 4090 Docker, GPU 0 |
| Data | Existing `/data/cifar-10-batches-py` |
| Weights | Existing `/root/robust_metric_lab/models/cifar10/Linf` |
| Training/download | None |

The same adversarial examples are reused across all 12 sampling conditions for a given model. The run evaluates all 38 currently implemented fields together: 29 feature-map metrics and 9 output/gradient controls.

### Sampling modes

- `all_leaf`: record every leaf module that emits a four-dimensional feature map, then uniformly select $K$ depth anchors. This is the original implementation.
- `post_relu`: record only actual ReLU outputs, including `nn.ReLU`, functional ReLU, and `torch.relu`.
- `post_activation`: record each architecture's native activation outputs. It includes ReLU as well as the Swish/SiLU activations used by five RobustBench models.

Five models do not contain ReLU: the two Gowal models, Rebuffi R18, Rade R18, and Wang2023. Therefore, strict `post_relu` results are reported only on the common 15-model ReLU cohort. Full 20-model comparisons use `all_leaf` and `post_activation`.

## Main Results

For lower-is-better metrics, ranking utility is $S=-\rho_s$. Larger $S$ is better. All displayed values use at most three decimal places.

### Reconciliation with the existing main leaderboard

The E22 `all_leaf, K=4` MMD values are exactly identical, model by model, to the previous white-box file `metrics_20_whitebox_fgsm_n64_capacity_above_mean_with_rb_scores.csv`; the maximum absolute difference over 20 models is zero. The apparently different correlations come from different attack protocols:

| Protocol | Samples and attack | Raw Spearman $\rho_s$ | Ranking utility $S=-\rho_s$ |
|---|---|---:|---:|
| E22 / previous white-box | 64, model-specific FGSM | -0.314 | 0.314 |
| Standard shared-adv | 128, Standard-source PGD-10 | -0.424 | 0.424 |
| Main three-source shared-adv | 128, mean of Standard/Engstrom/Wang PGD-10 metrics | -0.469 | 0.469 |

Thus, `-0.469` in the main report is not the old value of the same white-box experiment. It is the primary three-source shared-adversarial result. E22 uses white-box FGSM because its purpose is a fast controlled layer/hook ablation.

### MMD parameter sweep on all 20 models

The following cells report ranking utility $S=-\rho_s$, not raw Spearman.

| Sampling | $K=2$ | $K=4$ | $K=8$ | $K=16$ |
|---|---:|---:|---:|---:|
| `all_leaf` | 0.176 | 0.314 | **0.323** | 0.259 |
| `post_activation` | 0.171 | 0.269 | 0.278 | 0.278 |

The best MMD setting is `all_leaf, K=8`, but its gain over the original `all_leaf, K=4` is only $0.009$. The paired bootstrap interval for this gain is $[-0.145,0.168]$, so this experiment does not support changing MMD's default from $K=4$.

### Most important layer-count effects

| Metric | $K=2$ | $K=4$ | $K=8$ | $K=16$ | Best |
|---|---:|---:|---:|---:|---|
| `perturbation_capacity_delta (Ours)` | 0.532 | 0.626 | **0.805** | 0.672 | `all_leaf, K=8` |
| `feature_measure_spatial_gram_delta (Ours)` | 0.341 | 0.454 | 0.353 | **0.629** | `all_leaf, K=16` |
| `feature_measure_relu_active_spatial_delta (Ours)` | 0.350 | 0.457 | 0.353 | **0.576** | `all_leaf, K=16` |
| `feature_measure_mmd (Ours)` | 0.176 | 0.314 | **0.323** | 0.259 | `all_leaf, K=8` |
| `feature_measure_swd` | 0.442 | **0.492** | 0.460 | 0.439 | `all_leaf, K=4` |
| `fsp_delta [1,2]` | 0.460 | **0.493** | 0.257 | 0.365 | `all_leaf, K=4` |

The capacity improvement from $K=4$ to $K=8$ is $0.179$ with paired interval $[0.018,0.418]$. This is the only selected key layer-count improvement whose interval excludes zero. Spatial Gram improves at $K=16$, but its paired interval versus $K=4$ is $[-0.012,0.381]$.

### Native post-activation effects on all 20 models

`post_activation` is not a universal improvement:

| Metric and $K$ | `all_leaf` | `post_activation` | Change |
|---|---:|---:|---:|
| SWD, $K=16$ | 0.439 | **0.795** | +0.356 |
| capacity, $K=8$ | **0.805** | 0.720 | -0.084 |
| MMD, $K=8$ | **0.323** | 0.278 | -0.045 |
| spatial Gram, $K=16$ | **0.629** | 0.439 | -0.189 |
| FSP, $K=8$ | 0.257 | **0.343** | +0.086 |

SWD with `post_activation, K=16` reaches $\rho_s=-0.795$ and has bootstrap interval $[-0.941,-0.585]$. Its utility is only $0.009$ below capacity at `all_leaf, K=8`; the paired difference interval $[-0.154,0.213]$ does not distinguish them.

### Strict ReLU cohort

Strict `post_relu` covers 15 models, not 20. On this common cohort, the largest result is `feature_measure_relu_active_spatial_delta` at $K=16$, with $\rho_s=-0.882$. This value must not be compared directly with 20-model correlations because removing the five Swish/SiLU models changes both sample size and model distribution. It is evidence for a ReLU-specific subgroup hypothesis, not a new 20-model main result.

## Best Configuration for Every Feature Metric

| Metric | Best sampling | $K$ | Spearman | Utility |
|---|---|---:|---:|---:|
| `perturbation_capacity_delta (Ours)` | `all_leaf` | 8 | -0.805 | 0.805 |
| `feature_measure_swd` | `post_activation` | 16 | -0.795 | 0.795 |
| `perturbation_capacity_above_mean_delta (Ours)` | `post_activation` | 16 | -0.774 | 0.774 |
| `perturbation_capacity_relu_active_delta (Ours)` | `all_leaf` | 8 | -0.699 | 0.699 |
| `feature_measure_spatial_gram_delta (Ours)` | `all_leaf` | 16 | -0.629 | 0.629 |
| `feature_measure_cov_delta` | `post_activation` | 8 | -0.609 | 0.609 |
| `feature_measure_relu_active_spatial_delta (Ours)` | `all_leaf` | 16 | -0.576 | 0.576 |
| `feature_measure_struct01_delta (Ours)` | `post_activation` | 2 | -0.520 | 0.520 |
| `fsp_delta [1,2]` | `post_activation` | 2 | -0.507 | 0.507 |
| `feature_measure_struct_delta (Ours)` | `post_activation` | 2 | -0.505 | 0.505 |
| `feature_measure_local_delta (Ours)` | `post_activation` | 2 | -0.457 | 0.457 |
| `feature_measure_relu_active_delta (Ours)` | `all_leaf` | 2 | -0.450 | 0.450 |
| `feature_measure_relu_active_hybrid_delta (Ours)` | `all_leaf` | 2 | -0.442 | 0.442 |
| `feature_measure_moment_delta (Ours)` | `post_activation` | 2 | -0.433 | 0.433 |
| `feature_measure_robust_spatial_delta (Ours)` | `all_leaf` | 2 | -0.409 | 0.409 |
| `feature_measure_weighted_delta (Ours)` | `post_activation` | 2 | -0.385 | 0.385 |
| `feature_measure_robust_hybrid_delta (Ours)` | `all_leaf` | 2 | -0.382 | 0.382 |
| `feature_measure_robust_rel_delta (Ours)` | `post_activation` | 2 | -0.364 | 0.364 |
| `feature_measure_relu_active_hybrid_rel_delta (Ours)` | `all_leaf` | 2 | -0.361 | 0.361 |
| `feature_measure_robust_hybrid_rel_delta (Ours)` | `all_leaf` | 2 | -0.346 | 0.346 |
| `feature_measure_robust_spatial_rel_delta (Ours)` | `post_activation` | 2 | -0.341 | 0.341 |
| `feature_measure_robust_delta (Ours)` | `all_leaf` | 2 | -0.340 | 0.340 |
| `feature_measure_mmd (Ours)` | `all_leaf` | 8 | -0.323 | 0.323 |
| `feature_measure_flow_rel_delta (Ours)` | `all_leaf` | 2 | -0.292 | 0.292 |
| `feature_measure_spatial_gram_rel_delta (Ours)` | `all_leaf` | 4 | -0.235 | 0.235 |
| `feature_measure_relu_active_spatial_rel_delta (Ours)` | `all_leaf` | 16 | -0.223 | 0.223 |
| `fsp_strict_delta [1,2]` | `all_leaf` | 2 | -0.162 | 0.162 |
| `feature_delta (Ours)` | `all_leaf` | 4 | -0.135 | 0.135 |
| `feature_measure_flow_delta (Ours)` | `all_leaf` | 2 | 0.020 | -0.020 |

The CSV files retain all conditions, controls, and bootstrap intervals. This table is an exploratory maximum over eight full-model configurations and therefore overestimates held-out performance.

## Conclusions

1. Increasing MMD from four to eight sampled layers does not produce a reliable gain. Keep $K=4$ as the conservative MMD default until held-out confirmation.
2. Capacity benefits materially from $K=8$ under white-box FGSM. This is the strongest parameter result and should be tested under the three-source shared-adversarial protocol before becoming the paper default.
3. Spatial Gram prefers more layers ($K=16$), but the paired uncertainty remains wide.
4. Strict ReLU-only sampling cannot define a metric on all 20 RobustBench models because five use Swish/SiLU. Native post-activation sampling is the fair 20-model version.
5. Native post-activation sampling strongly helps SWD, but weakens MMD and spatial Gram. Activation-only sampling should remain metric-specific rather than a global rule.
6. Capacity $K=8$, post-activation SWD $K=16$, and adversarial loss are statistically unresolved on only 20 models: their paired utility differences all include zero.

## Reproduction

Server metric command:

```bash
cd /root/robust_metric_lab
export PYTHONPATH=$PWD/src:$PWD/.deps

CUDA_VISIBLE_DEVICES=0 python3 scripts/compute_rb_feature_sampling_ablation.py \
  --config configs/cifar10_linf_20.yaml \
  --data-root /data \
  --model-dir models \
  --model-start 0 \
  --model-stop 20 \
  --sample-limit 64 \
  --batch-size 16 \
  --attack fgsm \
  --pgd-steps 1 \
  --max-layers 2 4 8 16 \
  --activation-modes all_leaf post_relu post_activation \
  --seed 20260704 \
  --output outputs/e22_feature_sampling/metrics_20_whitebox_fgsm_n64_k2k4k8k16_three_hooks.csv
```

Local analysis command:

```powershell
cd E:\projects\robust_metric_lab

python scripts\analyze_feature_sampling_ablation.py `
  --input outputs\e22_feature_sampling\metrics_20_whitebox_fgsm_n64_k2k4k8k16_three_hooks.csv `
  --targets outputs\robustbench_cifar10_linf_scores.csv `
  --entity-column model_name `
  --target-column rb_robust_acc `
  --bootstrap-repeats 2000 `
  --seed 20260718 `
  --output-dir outputs\e22_feature_sampling\analysis
```

## Files

```text
outputs\e22_feature_sampling\metrics_20_whitebox_fgsm_n64_k2k4k8k16_three_hooks.csv
outputs\e22_feature_sampling\run_three_hooks.log
outputs\e22_feature_sampling\analysis\correlations_all.csv
outputs\e22_feature_sampling\analysis\best_configuration_per_metric.csv
outputs\e22_feature_sampling\analysis\all_leaf_layer_count_comparison.csv
outputs\e22_feature_sampling\analysis\matched_hook_comparison.csv
outputs\e22_feature_sampling\analysis\best_vs_baseline_paired_bootstrap.csv
outputs\e22_feature_sampling\analysis\selected_metric_comparisons.csv
```
