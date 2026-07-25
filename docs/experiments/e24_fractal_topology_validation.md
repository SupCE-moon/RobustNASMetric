# E24: Fractal and Graph-Topology Metric Validation

Date: 2026-07-21

## Material Passport

- Type: metric design, checkpoint re-evaluation, and statistical validation
- Status: completed
- Question: can Hausdorff support displacement, finite-scale box-dimension drift, or graph fundamental-rank drift improve robust-architecture ranking over the previous 41 metrics?
- New metrics: 4, all marked `(Ours)` and pre-registered lower-is-better
- Total comparison fields: 45 canonical metrics
- Dataset: existing CIFAR-10 only; no dataset download
- Training: none; all NAS checkpoints were reused

## Final Metric Protocol

- White-box FGSM, $\epsilon=8/255$.
- Same fixed 64-image subset and seeds as E23.
- `all_leaf` activation collection.
- Geometry-eligible feature maps require at least 16 spatial atoms.
- At most $K=4$ eligible layers are sampled uniformly.
- Hausdorff projection dimension: 16; maximum spatial atoms: 64.
- Box projection dimension: 3; scale ratios: 0.125/0.2/0.32/0.5.
- Graph topology: at most 32 atoms; clean distance quantiles 0.1/0.2/0.3; MST completion.
- Correlation direction was fixed before reading robust-accuracy targets.

The first smoke implementation allowed $1\times1$ maps and produced meaningless normalized Hausdorff values. It was rejected before the formal run. The final implementation excludes feature maps with fewer than 16 spatial atoms and passed an explicit degenerate-map test.

## Server and Checkpoint Audit

| Resource | Status on 2026-07-21 | Use |
|---|---|---|
| Chengdu 516 RTX 5090 docker/host | both ports actively refused connections | not used |
| Chengdu 516 RTX 4090 | connected, GPU 0 idle | RobustBench, NB201 shard 0, DARTS shard 0 |
| Zhongceyuan RTX 4090D | connected, GPU 1 idle | NB201 shard 1, DARTS shard 1 |
| Chengdu A100 containers | all three connected; physical GPU 3 idle, GPUs 0-2 busy | checked but not used because checkpoints were on the 4090 servers |

Saved checkpoints were complete:

- NB201 shard 0: 160 files; shard 1: 160 files.
- DARTS shard 0: 30 files; shard 1: 30 files.
- No model retraining was required.

## Server Commands

RobustBench on Chengdu 516 RTX 4090:

```bash
cd /root/robust_metric_lab
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.deps:src python3 scripts/compute_geometry_metric_batch.py \
  --config configs/cifar10_linf_20.yaml --data-root /data --model-dir models \
  --output outputs/e24_geometry/rb20_geometry.csv \
  --model-limit 20 --sample-limit 64 --batch-size 16 \
  --attack fgsm --max-layers 4 --seed 20260720
```

NB201 command, run once per existing shard:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src python3 scripts/evaluate_nb201_geometry_checkpoints.py \
  --metrics-csv <E23_SHARD_METRICS> --nasrobbench-repo /root/nasrobbench201 \
  --data-root <DATA_ROOT> --metric-samples 64 --batch-size 64 \
  --max-layers 4 --seed 0 --output <E24_SHARD_OUTPUT>
```

DARTS command, run once per existing shard:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src python3 scripts/evaluate_darts_geometry_checkpoints.py \
  --metrics-csv <E23_SHARD_METRICS> \
  --manifest data/darts_e23_architectures_seed20260720.json \
  --data-root <DATA_ROOT> --metric-samples 64 --batch-size 64 \
  --max-layers 4 --seed 0 --output <E24_SHARD_OUTPUT>
```

## Grid and Reproducibility Audit

- RobustBench: 20 rows, no duplicate `model_name`.
- NAS-RobBench-201: 320 rows, exact 64 architectures by epochs 0/5/10/25/50.
- DARTS/RACL: 60 rows, exact 20 architectures by epochs 0/50/100.
- All four new metric columns are finite in every row.
- Old 41 metrics have exactly unchanged Pearson, Spearman, Kendall, and direction-adjusted Spearman values after extension; maximum Spearman difference is zero.
- Unit tests: 37 passed.

## Main Spearman Results

Raw Spearman is shown below. Because all four metrics are lower-is-better, negative values agree with the theoretical direction.

| Setting | Hausdorff | QH95 | Box dimension | Graph $\pi_1$ rank |
|---|---:|---:|---:|---:|
| RobustBench-20 | -0.328 | -0.400 | -0.365 | **-0.403** |
| NB201 epoch 0 | -0.704 | -0.715 | -0.642 | **-0.750** |
| NB201 epoch 5 | +0.498 | +0.457 | -0.156 | +0.716 |
| NB201 epoch 10 | +0.616 | +0.593 | -0.008 | +0.380 |
| NB201 epoch 25 | +0.779 | +0.786 | -0.185 | +0.660 |
| NB201 epoch 50 | +0.870 | +0.883 | -0.146 | +0.636 |
| DARTS epoch 0 | -0.197 | -0.131 | -0.179 | **-0.304** |
| DARTS epoch 50 | +0.144 | +0.208 | +0.183 | -0.0105 |
| DARTS epoch 100 | +0.188 | +0.248 | +0.114 | +0.239 |

Bootstrap intervals:

- NB201 epoch-0 graph rank direction-adjusted Spearman: 0.750, 95% interval $[0.652,0.812]$.
- DARTS epoch-0 graph rank: 0.304, interval $[-0.177,0.702]$.
- RobustBench graph rank: 0.403, interval $[-0.102,0.770]$.
- Every DARTS interval for the four new metrics crosses zero.

## Rank Among 45 Metrics

| Aggregate | Best new metric | Score | Rank |
|---|---|---:|---:|
| Zero-cost mean: NB201 e0 + DARTS e0 | Graph fundamental rank | 0.527 | 11/45 |
| Mature mean: RB + NB201 e50 + DARTS e100 | Box dimension | 0.133 | 23/45 |
| All nine settings | Box dimension | 0.154 | 26/45 |

For comparison, zero-cost TRNAS R-Score remains 0.824, TRNAS $\Phi$ 0.810, SWD 0.626, and MMD 0.444. The new graph metric does not become the strongest zero-cost baseline.

## Structural Confounds

NB201 epoch-0 partial Spearman:

| Metric | Raw | Given operation counts | Given official clean | Given operations + clean |
|---|---:|---:|---:|---:|
| Hausdorff | -0.704 | -0.345 | -0.315 | -0.304 |
| QH95 | -0.715 | -0.370 | -0.321 | -0.312 |
| Box dimension | -0.642 | -0.370 | -0.239 | -0.354 |
| Graph fundamental rank | -0.750 | -0.402 | -0.309 | -0.359 |

The graph signal is not completely explained by operation counts or clean accuracy, but its magnitude is roughly halved.

DARTS epoch-0 graph rank is -0.304 raw and -0.296 after parameter/operation controls. This indicates little size confounding, but the sample is only 20 and the bootstrap interval is wide.

## Interpretation

1. The graph fundamental-rank metric is the most promising of the four for random initialization. It detects an NB201 zero-cost signal that partly survives structural and clean-accuracy controls.
2. It does not transfer strongly to DARTS, so it is not yet a search-space-general robust metric.
3. Hausdorff and graph metrics reverse direction after NB201 training. A plausible interpretation is that mature robust architectures preserve or develop richer internal geometric response under FGSM rather than minimizing normalized support movement. Therefore the original lower-is-better hypothesis fails after training.
4. Box-dimension drift is less spectacular but more stable. Its correlations are weak and do not justify a main-method claim.
5. The experiment supports including graph topology as a theoretically motivated zero-cost ablation. It does not support replacing MMD/SWD/TRNAS or the mature-model capacity metric.

## Output Files

```text
outputs/e24_fractal_topology/robustbench20_45metrics.csv
outputs/e24_fractal_topology/nb201_64_45metrics.csv
outputs/e24_fractal_topology/darts20_45metrics.csv
outputs/e24_fractal_topology/robustbench20_correlations_45metrics.csv
outputs/e24_fractal_topology/nb201_correlations_45metrics.csv
outputs/e24_fractal_topology/darts_correlations_45metrics.csv
outputs/e24_fractal_topology/nb201_structural_confound_45metrics.csv
outputs/e24_fractal_topology/darts_structural_confound_45metrics.csv
outputs/e24_fractal_topology/summary/cross_space_spearman_matrix.csv
outputs/e24_fractal_topology/summary/cross_space_summary.md
```
