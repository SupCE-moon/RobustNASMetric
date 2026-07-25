# NAS-RobBench-201 TRNAS vs MMD Validation

Date: 2026-07-18

## Material Passport

- Experiment ID: E21
- Status: completed for one architecture sample and one initialization seed
- Verification status: ANALYZED
- Dataset: CIFAR-10 / NAS-RobBench-201
- Target: official AutoAttack robust accuracy
- Server: Chengdu 516 RTX 4090 Docker, GPU 0
- Main question: whether `feature_measure_mmd (Ours)` ranks final robustness better than the two TRNAS components and composite R-Score

## 1. Metrics

The comparison includes:

| Metric | Source | Direction |
|---|---|---|
| `feature_measure_mmd (Ours)` | This project | lower is better |
| `trnas_activation_psi [11,12]` | TRNAS activation capability / SWAP cardinality | higher is better |
| `trnas_consistency_phi [11]` | TRNAS feature-parameter consistency | higher is better |
| `trnas_r_score [11]` | $\sqrt{\Psi}\Phi$ | higher is better |

TRNAS has no linked official implementation on its ICCV page or supplement. The implementation follows its Eqs. (2)-(6), uses the official SWAP cardinality for $\Psi$, and uses one loss-ascent parameter step for the paper's unspecified gradient update. The main complete run uses $\eta=10^{-6}$; larger steps are retained as sensitivity analyses.

## 2. Matched Protocol

- 512 uniformly sampled canonical NAS-Bench-201 architectures.
- Architecture seed: `20260718`.
- Weight initialization/training seed: `0`.
- Epoch: 0; no candidate training.
- Reused checkpoints: E19's 512 saved epoch-0 `.pt` files.
- Metric data: the same 64 existing CIFAR-10 test samples used by E19.
- Attack: white-box FGSM, $\epsilon=8/255$.
- MMD values: reused from E19; no recomputation or changed sample batch.
- TRNAS values: computed from the same checkpoints and batch.
- No dataset or model was downloaded.

Server command:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 scripts/compute_nb201_trnas_metrics.py \
  --input-metrics outputs/e19_zero_cost_uniform512/metrics.csv \
  --checkpoint-dir outputs/e19_zero_cost_uniform512/checkpoints \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 --data-root /root/data \
  --metric-samples 64 --batch-size 64 \
  --epsilon 0.03137254901960784 \
  --weight-step-size 0.000001 \
  --seed 0 \
  --output outputs/e21_trnas_mmd/metrics_uniform512_step1e6.csv \
  > outputs/e21_trnas_mmd/metrics_uniform512_step1e6.log 2>&1
```

The run produced 512/512 rows with finite $\Psi$, $\Phi$, and R-Score values. Measured wall time was approximately 59 seconds on one RTX 4090.

## 3. Primary Results

All correlations use official AutoAttack accuracy. Ranking utility changes the sign of lower-is-better MMD so that larger is always better.

| Metric | Spearman | Ranking utility | 95% bootstrap CI | Top-10% target mean |
|---|---:|---:|---:|---:|
| `trnas_r_score [11]` | +0.807 | 0.807 | [0.767, 0.837] | 0.418 |
| `trnas_consistency_phi [11]` | +0.789 | 0.789 | [0.748, 0.821] | 0.417 |
| `feature_measure_mmd (Ours)` | -0.705 | 0.705 | [0.642, 0.758] | 0.388 |
| `trnas_activation_psi [11,12]` | +0.567 | 0.567 | [0.500, 0.628] | 0.405 |

Paired bootstrap differences use `MMD utility - comparison utility`:

| Comparison | Difference | 95% CI | Result |
|---|---:|---:|---|
| MMD minus $\Psi$ | +0.137 | [0.0488, 0.222] | MMD is stronger |
| MMD minus $\Phi$ | -0.0843 | [-0.147, -0.0240] | $\Phi$ is stronger |
| MMD minus R-Score | -0.102 | [-0.167, -0.0394] | R-Score is stronger |

Therefore, under raw Spearman ranking, the current experiment does **not** show that MMD is better than TRNAS R-Score or $\Phi$.

## 4. Structural Confound Audit

The raw TRNAS result is heavily affected by architecture size. `trnas_num_consistency_layers` alone has Spearman `+0.809`, slightly higher than R-Score's `+0.807`, and its top-10% target mean is `0.416`.

| Metric | Target Spearman | Spearman with layer count | Partial target Spearman given layer count |
|---|---:|---:|---:|
| `feature_measure_mmd (Ours)` | -0.705 | -0.688 | -0.347 |
| $\Psi$ | +0.567 | +0.559 | +0.236 |
| $\Phi$ | +0.789 | +0.970 | +0.0280 |
| R-Score | +0.807 | +0.939 | +0.232 |

After dividing by activation units/layers:

| Diagnostic | Spearman |
|---|---:|
| $\Phi/I$ | +0.238 |
| $\sqrt{\Psi/V}(\Phi/I)$ | -0.261 |
| $\Psi/V$ | -0.406 |

The normalized quantities are diagnostics, not TRNAS's reported metrics. They show that the raw summed $\Phi$ and R-Score obtain much of their ranking power by rewarding architectures with more learnable layers and activation units.

MMD has weaker raw ranking, but retains more robustness-specific residual signal:

| Metric | Partial given operation counts | Partial given official clean accuracy |
|---|---:|---:|
| `feature_measure_mmd (Ours)` | -0.324 | -0.543 |
| $\Psi$ | +0.144 | -0.260 |
| $\Phi$ | +0.165 | +0.159 |
| R-Score | +0.201 | +0.0839 |

## 5. Weight-Step Sensitivity

The TRNAS paper does not specify the parameter-update magnitude. Larger steps improve the correlation on surviving architectures but create non-finite values, making the comparison selection-biased.

| $\eta$ | Finite $\Phi$/R-Score | $\Phi$ Spearman | R-Score Spearman |
|---:|---:|---:|---:|
| 0 | 512/512 | 0.777 | 0.804 |
| $10^{-6}$ | 512/512 | 0.789 | 0.807 |
| $10^{-4}$ | 498/512 | 0.815 | 0.830 |
| $10^{-3}$ | 494/512 | 0.824 | 0.840 |
| $10^{-2}$ | 442/512 | 0.756 | 0.785 |

The complete $10^{-6}$ run is the main auditable comparison. The larger-step results must not be presented as stronger evidence because their effective architecture sets differ.

## 6. Decision

E21 supports three separate conclusions:

1. `feature_measure_mmd (Ours)` is better than TRNAS activation capability $\Psi$ alone.
2. Raw TRNAS $\Phi$ and R-Score rank official robustness better than MMD on NAS-RobBench-201.
3. Their advantage is not yet evidence of a better robustness mechanism: a layer-count baseline matches or slightly exceeds R-Score, while $\Phi$ loses almost all residual correlation after controlling layer count.

The paper-safe interpretation is that TRNAS is currently the stronger **raw architecture ranker**, whereas MMD is the stronger **feature-perturbation residual metric** after controlling final clean accuracy and retains more signal after controlling simple architecture composition or layer count. A fair next-stage comparison should use complexity-matched architectures or residualize all metrics against parameter count, FLOPs, layer count, and operation counts before search evaluation.

## 7. Local Files and Commands

Primary raw data and logs:

```text
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_step1e6.csv
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_step1e6.log
```

Sensitivity data:

```text
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_step0.csv
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_step1e4.csv
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_fgsm_n64_seed0.csv
outputs/nasrobbench201_server/e21_trnas_mmd/metrics_uniform512_step1e2.csv
```

Analysis command:

```powershell
cd E:\projects\robust_metric_lab
$env:PYTHONPATH="$PWD\src"

python scripts\analyze_trnas_mmd_nb201.py `
  --input outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step1e6.csv `
  --sensitivity-inputs `
    outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step0.csv `
    outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step1e6.csv `
    outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step1e4.csv `
    outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_fgsm_n64_seed0.csv `
    outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step1e2.csv `
  --bootstrap-repeats 5000 --seed 20260718 `
  --output-dir outputs\nasrobbench201_server\e21_trnas_mmd\analysis

python scripts\audit_trnas_structure.py `
  --input outputs\nasrobbench201_server\e21_trnas_mmd\metrics_uniform512_step1e6.csv `
  --output outputs\nasrobbench201_server\e21_trnas_mmd\analysis\structural_residuals.csv
```

Derived outputs:

```text
outputs/nasrobbench201_server/e21_trnas_mmd/analysis/correlations.csv
outputs/nasrobbench201_server/e21_trnas_mmd/analysis/paired_bootstrap_differences.csv
outputs/nasrobbench201_server/e21_trnas_mmd/analysis/weight_step_sensitivity.csv
outputs/nasrobbench201_server/e21_trnas_mmd/analysis/structural_residuals.csv
```
