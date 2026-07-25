# E23: Cross-Space Robust Metric Validation

Date: 2026-07-19

## Material Passport

- Type: code experiment and reproducibility record
- Status: completed
- Question: which of the 41 pre-registered robust metrics consistently ranks robustness in RobustBench models, NAS-Bench-201 cells, and RACL/DARTS cells?
- Primary statistic: direction-adjusted Spearman correlation
- Training seed: 0
- Architecture seed: 20260720
- Dataset: existing CIFAR-10 copies; no dataset was downloaded

## Common Metric Protocol

- White-box metric attack: FGSM, $\epsilon=8/255$.
- Feature/output metrics: fixed 64-image CIFAR-10 test subset.
- TRNAS: first 16 images from the same fixed subset, one joint batch, weight-ascent step $\eta=10^{-6}$.
- Feature sampling: `all_leaf`, $K=4$ anchor layers.
- Directions: fixed by `src/robust_metric_lab/metric_registry.py` before reading targets.
- Display precision: three significant digits; raw CSV retains machine precision.
- Bootstrap: architecture/model resampling, 2,000 samples, 95% percentile interval.

## Space A: RobustBench 20 Models

Status: completed.

- Server: Chengdu 516 RTX 4090 Docker, GPU 0.
- Target: RobustBench official CIFAR-10 $L_\infty$ robust accuracy.
- Model weights: existing `/root/robust_metric_lab/models`.
- Dataset: existing `/data/cifar-10-batches-py`.
- Raw server output: `/root/robust_metric_lab/outputs/e23_cross_space/robustbench20_whitebox_fgsm64_all_metrics.csv`.
- Local raw output: `outputs/e23_cross_space/robustbench20_whitebox_fgsm64_all_metrics.csv`.
- Joined scores: `outputs/e23_cross_space/robustbench20_whitebox_fgsm64_all_metrics_with_scores.csv`.
- Correlations: `outputs/e23_cross_space/robustbench20_correlations.csv`.
- Report: `outputs/e23_cross_space/robustbench20_report.md`.

Server command:

```bash
cd /root/robust_metric_lab
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.deps:src python3 scripts/compute_metric_batch.py \
  --config configs/cifar10_linf_20.yaml \
  --data-root /data --model-dir models \
  --output outputs/e23_cross_space/robustbench20_whitebox_fgsm64_all_metrics.csv \
  --model-limit 20 --sample-limit 64 --batch-size 16 \
  --attack fgsm --max-layers 4 \
  --include-published-nas-metrics --published-metric-samples 16 \
  --trnas-weight-step-size 1e-6 --seed 20260720
```

Local analysis:

```powershell
cd E:\projects\robust_metric_lab
$env:PYTHONPATH="$PWD\src"

python scripts\join_robustbench_scores.py `
  --metrics outputs\e23_cross_space\robustbench20_whitebox_fgsm64_all_metrics.csv `
  --scores outputs\robustbench_cifar10_linf_scores.csv `
  --output outputs\e23_cross_space\robustbench20_whitebox_fgsm64_all_metrics_with_scores.csv

python scripts\analyze_cross_space_metrics.py `
  --input outputs\e23_cross_space\robustbench20_whitebox_fgsm64_all_metrics_with_scores.csv `
  --space RobustBench-20 --target rb_robust_acc `
  --bootstrap-samples 2000 --seed 20260720 `
  --output-csv outputs\e23_cross_space\robustbench20_correlations.csv `
  --output-report outputs\e23_cross_space\robustbench20_report.md
```

## Space B: 64 NAS-RobBench-201 Architectures

Status: completed and analyzed.

- Target: official NAS-RobBench-201 `autoattack` accuracy.
- Architecture sampling: 64 uniformly sampled canonical architectures, seed 20260720; no target stratification.
- Model: official NAS-Bench-201 tiny network, $C=16$, $N=5$.
- Training: full 50,000-image CIFAR-10, 50 epochs, PGD-7, batch 256, SGD momentum 0.9, weight decay $5\times10^{-4}$, OneCycleLR max LR 0.1.
- Checkpoints: epoch 0, 5, 10, 25, and 50 for every architecture.
- Shard 0: Chengdu 516 RTX 4090, sampled positions `[0,32)`.
- Shard 1: Zhongceyuan RTX 4090D GPU 0, sampled positions `[32,64)`; GPU 1 was occupied and not used.

Common command, with `<START>`, `<STOP>`, `<DATA_ROOT>`, and `<OUT>` replaced per shard:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 scripts/compute_nb201_metric_subset.py \
  --benchmark-json data/nasrobbench201/NAS-RobBench-201_all.json \
  --meta-json data/nasrobbench201/meta.json \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 --data-root <DATA_ROOT> \
  --target-metric autoattack --sample-size 64 --sampling uniform \
  --architecture-seed 20260720 --arch-start <START> --arch-stop <STOP> \
  --epochs 50 --eval-epochs 0 5 10 25 50 \
  --checkpoint-epochs 0 5 10 25 50 \
  --train-samples 50000 --metric-samples 64 --batch-size 256 \
  --train-attack pgd --train-pgd-steps 7 \
  --metric-attack fgsm --metric-pgd-steps 1 \
  --max-layers 4 --lr 0.1 --lr-schedule onecycle --seed 0 \
  --include-trnas --trnas-samples 16 --trnas-weight-step-size 1e-6 \
  --checkpoint-dir <OUT>/checkpoints --output <OUT>/metrics.csv
```

Completed merge and analysis command:

```powershell
python scripts\merge_metric_shards.py `
  --inputs `
    outputs\e23_cross_space\nb201_64\shard_000_032.csv `
    outputs\e23_cross_space\nb201_64\shard_032_064.csv `
  --expected-architectures 64 --expected-epochs 0 5 10 25 50 `
  --output outputs\e23_cross_space\nb201_64\metrics_64_e0e5e10e25e50.csv

python scripts\analyze_cross_space_metrics.py `
  --input outputs\e23_cross_space\nb201_64\metrics_64_e0e5e10e25e50.csv `
  --space NAS-RobBench-201 --target oracle_target `
  --epoch-column train_epochs --epochs 0 5 10 25 50 `
  --bootstrap-samples 2000 --seed 20260720 `
  --output-csv outputs\e23_cross_space\nb201_64\correlations.csv `
  --output-report outputs\e23_cross_space\nb201_64\report.md
```

Completed grid audit:

- 64 unique architectures.
- 320 metric rows: 64 architectures times epochs 0/5/10/25/50.
- 320 server checkpoints: five per architecture.
- No missing architecture/epoch cells after merge.

Selected raw Spearman results against official AutoAttack accuracy:

| Metric | Epoch 0 | Epoch 5 | Epoch 10 | Epoch 25 | Epoch 50 |
|---|---:|---:|---:|---:|---:|
| `trnas_r_score [11]` | 0.897 | 0.894 | 0.907 | 0.910 | 0.912 |
| `trnas_activation_psi [11,12]` | 0.822 | 0.831 | 0.897 | 0.906 | 0.900 |
| `trnas_consistency_phi [11]` | 0.843 | 0.794 | 0.782 | 0.768 | 0.785 |
| `feature_measure_mmd (Ours)` | -0.794 | -0.646 | -0.502 | -0.120 | -0.0159 |
| `perturbation_capacity_delta (Ours)` | -0.703 | +0.397 | -0.187 | -0.237 | -0.208 |
| `racl_arch_lipschitz_log_upper [5]` | -0.254 | -0.290 | -0.383 | -0.264 | -0.454 |
| `racl_input_gradient_lower [5]` | -0.709 | +0.830 | +0.790 | +0.725 | +0.844 |
| `adversarial_loss [10]` | -0.588 | -0.655 | -0.907 | -0.956 | -0.980 |

Interpretation at this stage:

- MMD is a strong zero/low-fidelity ranker but its signal decays toward zero by epoch 50.
- Capacity is strong at initialization but is not stable across low-fidelity checkpoints in this fixed-training search space.
- TRNAS remains very strong at all five checkpoints. Structural confound controls remain necessary before claiming an independent robustness mechanism.
- The RACL gradient alias reverses after training and fails its fixed lower-is-better direction. The fixed-cell upper counterpart is moderate at epoch 50.
- Final natural/adversarial loss nearly perfectly ranks the official target, but this is a mature-training performance readout rather than a useful cheap architecture proxy.

## Space C: 20 RACL/DARTS Architectures

Status: completed and analyzed; paper-final macro timing pilot completed on A100 GPU 3.

- Paper training horizon: $N=100$ epochs; requested checkpoints are 0, 50, and 100.
- Search space: DARTS/PC-DARTS cells with four intermediate nodes and two retained incoming edges per node.
- Operations: separable conv 3/5, dilated conv 3/5, max pool, average pool, skip; `none` is excluded during final top-two edge discretization.
- Fixed manifest: `data/darts_e23_architectures_seed20260720.json`.
- Training objective: the RACL evaluation protocol's TRADES-style objective, PGD-10, $\epsilon=8/255$, step $2/255$, $\beta=6$, SGD LR 0.1, momentum 0.9, weight decay $2\times10^{-4}$, gradient clip 5.
- Independent target: final checkpoint PGD-20 accuracy on the full 10,000-image CIFAR-10 test set; full AutoAttack remains a later confirmation because it is substantially more expensive.

The paper forms a 20-cell, initial-channel-36 network with about 3.3M parameters for final evaluation. Training 20 such networks for 100 epochs with PGD-10 was not assumed affordable without measurement. A one-architecture timing gate compared:

1. paper final macro network: $C=36$, 20 cells;
2. explicitly exploratory search-stage macro network: $C=16$, 8 cells, still trained to epoch 100.

The second option tests cross-search-space ranking but is not a strict reproduction of the paper's final architecture evaluation.

Timing-gate decision:

- Search-stage macro pilot, $C=16$, 8 cells: 53.2 seconds for one full PGD-10 epoch plus metrics at epochs 0/1 and a 1,000-sample PGD-20 evaluation.
- Paper-final macro pilot, $C=36$, 20 cells: 493 seconds for one epoch plus metrics at epochs 0/1 and a 200-sample PGD-20 evaluation on A100 GPU 3.
- Naive paper-final extrapolation: about 13.7 GPU-hours per 100-epoch architecture, or about 274 GPU-hours for 20 architectures, before full AutoAttack.
- The random paper-final pilot diverged to `NaN` during epoch 1. All epoch-1 differentiable metrics except activation cardinality became undefined. No automatic retry was performed.

Formal search-stage allocation:

| Server | GPU | Architectures | Output |
|---|---:|---:|---|
| Chengdu 516 RTX 4090 | 0 | 0-9 | `/root/robust_metric_lab_nas/outputs/e23_cross_space/darts20/shard_000_010` |
| Zhongceyuan RTX 4090D | 0 | 10-19 | `/root/robust_metric_lab_nas/outputs/e23_cross_space/darts20/shard_010_020` |
| Chengdu A100 container 17034 | 3 | one paper-final timing pilot only | `/root/robust_metric_lab_e23/outputs/darts_paper_macro_timing_c36_l20` |

Manifest generation:

```powershell
cd E:\projects\robust_metric_lab
$env:PYTHONPATH="$PWD\src"
python scripts\generate_darts_manifest.py `
  --num-architectures 20 --seed 20260720 `
  --output data\darts_e23_architectures_seed20260720.json
```

Formal training command, with `<START>/<STOP>`, `<DATA_ROOT>`, and `<SHARD>` replaced by the allocation above:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 scripts/run_darts_metric_trajectory.py \
  --manifest data/darts_e23_architectures_seed20260720.json \
  --data-root <DATA_ROOT> --arch-start <START> --arch-stop <STOP> \
  --channels 16 --layers 8 --epochs 100 --eval-epochs 0 50 100 \
  --train-samples 50000 --batch-size 128 \
  --metric-samples 64 --metric-batch-size 64 --max-layers 4 \
  --train-attack-steps 10 --trades-beta 6 \
  --robust-eval-samples 1000 --robust-eval-batch-size 128 \
  --robust-eval-pgd-steps 20 --trnas-samples 16 \
  --trnas-weight-step-size 1e-6 --weight-seed 0 \
  --checkpoint-dir <SHARD>/checkpoints --output <SHARD>/metrics.csv
```

Formal result audit:

- 20 unique genotypes from the pre-registered manifest.
- 60 metric rows: 20 architectures times epochs 0/50/100.
- 60 server checkpoints: three per architecture.
- No non-finite metric values in the formal search-stage run.
- Full-test target: PGD-20 on all 10,000 CIFAR-10 test images, range 0.4607-0.4853.
- Final full-test clean/robust Spearman: 0.695.

Selected raw Spearman results against full-test PGD-20 accuracy:

| Metric | Epoch 0 | Epoch 50 | Epoch 100 |
|---|---:|---:|---:|
| `trnas_consistency_phi [11]` | 0.776 | 0.771 | 0.764 |
| `trnas_r_score [11]` | 0.750 | 0.746 | 0.753 |
| `trnas_activation_psi [11,12]` | 0.499 | 0.701 | 0.710 |
| `feature_measure_mmd (Ours)` | -0.0947 | +0.132 | +0.335 |
| `perturbation_capacity_delta (Ours)` | -0.0511 | -0.262 | -0.382 |
| `racl_arch_lipschitz_log_upper [5]` | +0.558 | -0.690 | -0.690 |
| `racl_input_gradient_lower [5]` | -0.456 | -0.0120 | -0.107 |
| `adversarial_loss [10]` | +0.0195 | -0.621 | -0.693 |

Full-test evaluation command, with the server-specific checkpoint/data root and architecture slice:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 scripts/evaluate_darts_checkpoints.py \
  --manifest data/darts_e23_architectures_seed20260720.json \
  --checkpoint-dir <SHARD>/checkpoints --data-root <DATA_ROOT> \
  --arch-start <START> --arch-stop <STOP> \
  --channels 16 --layers 8 --checkpoint-epoch 100 \
  --samples 10000 --batch-size 128 --pgd-steps 20 \
  --output <SHARD>/full_test_pgd20.csv
```

Local merge, score join, correlation, confound, and cross-space summary commands:

```powershell
cd E:\projects\robust_metric_lab
$env:PYTHONPATH="$PWD\src"

python scripts\merge_metric_shards.py `
  --inputs outputs\e23_cross_space\darts20\shard_000_010.csv outputs\e23_cross_space\darts20\shard_010_020.csv `
  --key-columns arch_id train_epochs --expected-architectures 20 --expected-epochs 0 50 100 `
  --output outputs\e23_cross_space\darts20\metrics_20_e0e50e100.csv

python scripts\merge_metric_shards.py `
  --inputs outputs\e23_cross_space\darts20\full_test_pgd20_000_010.csv outputs\e23_cross_space\darts20\full_test_pgd20_010_020.csv `
  --key-columns arch_id --expected-architectures 20 `
  --output outputs\e23_cross_space\darts20\full_test_pgd20_20arch.csv

python scripts\join_darts_final_scores.py `
  --metrics outputs\e23_cross_space\darts20\metrics_20_e0e50e100.csv `
  --scores outputs\e23_cross_space\darts20\full_test_pgd20_20arch.csv `
  --output outputs\e23_cross_space\darts20\metrics_20_e0e50e100_fulltest_pgd20.csv

python scripts\analyze_cross_space_metrics.py `
  --input outputs\e23_cross_space\darts20\metrics_20_e0e50e100_fulltest_pgd20.csv `
  --space DARTS-RACL-20 --target full_pgd20_accuracy `
  --epoch-column train_epochs --epochs 0 50 100 `
  --bootstrap-samples 5000 --seed 20260720 `
  --output-csv outputs\e23_cross_space\darts20\correlations_fulltest.csv `
  --output-report outputs\e23_cross_space\darts20\report_fulltest.md

python scripts\analyze_darts_structural_confound.py `
  --input outputs\e23_cross_space\darts20\metrics_20_e0e50e100_fulltest_pgd20.csv `
  --target full_pgd20_accuracy --epoch-column train_epochs --epochs 0 50 100 `
  --output outputs\e23_cross_space\darts20\structural_confound.csv

python scripts\summarize_cross_space_results.py `
  --robustbench outputs\e23_cross_space\robustbench20_correlations.csv `
  --nb201 outputs\e23_cross_space\nb201_64\correlations.csv `
  --darts outputs\e23_cross_space\darts20\correlations_fulltest.csv `
  --output-dir outputs\e23_cross_space\summary
```

`train_epochs` is the metric checkpoint (0/50/100). The joined `checkpoint_epoch=100` field records which final checkpoint generated the common 10,000-image target and must not be used as the metric-trajectory epoch column.

Final local files:

```text
outputs/e23_cross_space/darts20/metrics_20_e0e50e100_fulltest_pgd20.csv
outputs/e23_cross_space/darts20/full_test_pgd20_20arch.csv
outputs/e23_cross_space/darts20/correlations_fulltest.csv
outputs/e23_cross_space/darts20/report_fulltest.md
outputs/e23_cross_space/darts20/structural_confound.csv
```

Structural-confound result:

- All formal DARTS networks have exactly eight cells, so layer count is fixed.
- Target robustness correlates 0.672 with parameter count.
- TRNAS $\Phi$/R-Score correlate 0.950/0.961 with parameter count at epoch 0. After controlling parameter count they remain 0.567/0.471; after additionally controlling operation counts they remain 0.675/0.409.
- TRNAS $\Psi$ falls from 0.499 to -0.024 after size/operation controls at epoch 0, so its zero-cost signal is mainly structural prior.
- RACL upper correlates -0.922 with parameter count at mature checkpoints. Its epoch-100 raw Spearman -0.690 becomes +0.211 after size/operation controls, so the apparent ranking is not independent of model size.
- MMD epoch-0 partial Spearman is -0.238 after structural controls and remains weak. Capacity epoch-100 partial Spearman is -0.355.

Validated local smoke command:

```powershell
python scripts\run_darts_metric_trajectory.py `
  --manifest data\darts_e23_architectures_seed20260720.json `
  --data-root F:\Datasets\cifar10 --arch-start 0 --arch-stop 1 `
  --channels 4 --layers 2 --epochs 1 --eval-epochs 0 1 `
  --train-samples 128 --batch-size 32 `
  --metric-samples 8 --metric-batch-size 8 --trnas-samples 8 `
  --robust-eval-samples 16 --robust-eval-batch-size 16 `
  --robust-eval-pgd-steps 1 --train-attack-steps 1 --max-layers 2 `
  --checkpoint-dir outputs\command_validation\darts_smoke_ckpt `
  --output outputs\command_validation\darts_smoke.csv --device cpu
```

## Current Resource Audit

Across launch and follow-up stages:

- Chengdu 516 5090 Docker port 7001: connection refused.
- Chengdu 516 4090 GPU 0: NAS-RobBench shard 0, DARTS shard 0-9, then full-test PGD-20 evaluation.
- Zhongceyuan 4090D GPU 0: NAS-RobBench shard 1, DARTS shard 10-19, then full-test PGD-20 evaluation.
- Zhongceyuan GPU 1: pre-existing process, not used.
- Chengdu A100 GPU 3 later became idle. Container 17034 used it for exactly one paper-final macro timing/stability pilot; the other GPUs were not used.

This follows the A100 one-process-per-container and one-distinct-GPU rule and avoids pre-existing workloads.

## Cross-Space Decision

Complete matrices and ranked summaries:

```text
outputs/e23_cross_space/summary/cross_space_correlations_long.csv
outputs/e23_cross_space/summary/cross_space_spearman_matrix.csv
outputs/e23_cross_space/summary/cross_space_summary.md
```

Main conclusions:

1. TRNAS $\Phi$ and R-Score are the strongest current zero-cost robust-NAS baselines across NAS-RobBench-201 and DARTS. Their size dependence is substantial but does not fully explain the DARTS signal.
2. MMD is a strong NAS-RobBench-201 zero/low-fidelity proxy but does not transfer to random DARTS genotypes. It is not supported as a search-space-general training-free robustness metric.
3. Perturbation capacity is strong for heterogeneous RobustBench models and NAS-RobBench initialization but is nearly uninformative at DARTS initialization. At mature DARTS weights it has a moderate negative point estimate with an interval crossing zero.
4. The RACL fixed-cell upper counterpart is not a universal standalone ranker. Its DARTS mature correlation is explained by model size, while the RobustBench generic chain field is only an approximation for branched networks.
5. Mature natural/adversarial loss ranks mature robustness well, but it is a performance readout after expensive training rather than a useful cheap NAS proxy.
6. `feature_measure_struct01_delta (Ours)` and `feature_measure_moment_delta (Ours)` have the highest all-setting means among Ours, but this is a post-hoc aggregate observation and requires a new held-out experiment before metric selection.
