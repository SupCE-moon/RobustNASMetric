# NAS-RobBench-201 RACL Baselines and MMD Low-Fidelity Validation

Date: 2026-07-18

## Material Passport

- Experiment ID: E19
- Status: completed for one initialization seed
- Dataset: CIFAR-10 / NAS-RobBench-201
- Target: official AutoAttack robust accuracy
- Main servers: Chengdu 516 RTX 4090 Docker and Zhongceyuan RTX 4090-D Docker GPU 0
- Verification status: 512-row zero-cost table and 96-architecture x 3-epoch matched grid verified

## 1. What Is Actually Taken From RACL

Dong et al., [Adversarially Robust Neural Architectures](https://arxiv.org/abs/2009.00902), does not define two ordinary post-hoc metrics. Its robust formulation contains two evaluable Lipschitz-related terms:

1. The architecture-dependent upper bound in Eq. (10).
2. The input-loss gradient used as a lower-bound surrogate in Eq. (16).

The paper's upper bound is

$$
\lambda_F
\le
C
\prod_{k=1}^{N}
\prod_j
\sum_{i<j}
\beta^{(i,j)}
\sum_{o\in\mathcal O}
\alpha_o^{(i,j)}\lambda_o.
$$

For a fixed discrete NAS-Bench-201 architecture, the selected operation and connection coefficients are one. We therefore evaluate the architecture-dependent log term

$$
M_{\mathrm{RACL-upper}}
=
\sum_{k,j}
\log\left(
\max\left\{
\sum_{i<j}\lambda_{o_{ij}},\varepsilon
\right\}
\right).
$$

The implementation follows the operation constants stated by the paper: identity is 1, zero is 0, average pooling is $S^{-1/2}$, and convolution uses a 20-step power-iteration estimate of the spectral norm of its reshaped weight matrix. The architecture-independent macro factor $C$ is omitted.

The second fixed-model quantity is

$$
M_{\mathrm{RACL-grad}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left\|
\nabla_{x_b}\ell(f(x_b),y_b)
\right\|_2.
$$

It is exported as `racl_input_gradient_lower` and is numerically the existing `input_grad_l2` implementation.

Important scope limit: the RACL mean $\mu$ and variance $\sigma^2$ are induced by learned log-normal distributions over supernet architecture parameters $\alpha$ and $\beta$. A fixed NAS-RobBench architecture has no such learned distributions. E19 therefore does not claim to reproduce the full confidence constraint or the RACL search algorithm.

## 2. MMD Under Test

For each selected layer, clean and adversarial spatial feature atoms are standardized using clean statistics. With fixed random Fourier features $\phi$, the implemented quantity is

$$
M_{\mathrm{MMD}}
=
\frac{1}{|\mathcal L|}
\sum_{l\in\mathcal L}
\frac{1}{B}
\sum_{b=1}^{B}
\left\|
\frac{1}{S_l}\sum_s\phi(\widetilde a^{\mathrm{adv}}_{l,b,s})
-
\frac{1}{S_l}\sum_s\phi(\widetilde a_{l,b,s})
\right\|_2.
$$

Lower is pre-specified as better. At epoch 0 this is a training-free, data-dependent, gradient-dependent zero-cost-style proxy; it is not literally computation-free.

## 3. Zero-Cost Protocol

- Architecture sampling: 512 uniformly sampled canonical architectures.
- Architecture seed: `20260718`.
- Weight initialization seed: `0`.
- Weights: random initialization, no training.
- Metric data: 64 fixed CIFAR-10 test samples from the existing server copy.
- Attack: white-box FGSM, $\epsilon=8/255$.
- Layers: at most four activation anchors.
- Hardware: Chengdu 516 RTX 4090 Docker.
- Runtime: about 258 seconds for 512 architectures, approximately 0.504 seconds per architecture, including checkpoint writes.
- Checkpoints: 512 epoch-0 checkpoints, about 972 MB.

Server command:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compute_nb201_metric_subset.py \
  --benchmark-json data/nasrobbench201/NAS-RobBench-201_all.json \
  --meta-json data/nasrobbench201/meta.json \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 --data-root /root/data \
  --target-metric autoattack --sample-size 512 --sampling uniform \
  --architecture-seed 20260718 --epochs 0 --eval-epochs 0 \
  --train-samples 4096 --metric-samples 64 --batch-size 64 \
  --train-attack none --metric-attack fgsm --metric-pgd-steps 1 \
  --max-layers 4 --seed 0 \
  --checkpoint-dir outputs/e19_zero_cost_uniform512/checkpoints \
  --output outputs/e19_zero_cost_uniform512/metrics.csv
```

## 4. Zero-Cost Results

All values are Spearman correlations with official AutoAttack accuracy. Since all three metrics are lower-is-better, a negative value is the expected direction.

| Metric | All 512 | 95% bootstrap CI | Non-degenerate 493 | Top-10% target mean | Population mean |
|---|---:|---:|---:|---:|---:|
| `feature_measure_mmd` | -0.705 | [-0.758, -0.645] | -0.720 | 0.388 | 0.365 |
| `racl_input_gradient_lower` | -0.603 | [-0.658, -0.537] | -0.618 | 0.385 | 0.365 |
| `racl_arch_lipschitz_log_upper` | -0.163 | [-0.242, -0.0789] | -0.206 | 0.365 | 0.365 |

MMD's ranking utility exceeds the RACL gradient term by 0.102 with paired-bootstrap 95% interval `[0.0777, 0.129]`. It exceeds the upper-bound term by 0.542 with interval `[0.457, 0.624]`.

MMD is consistent across the official NAS-RobBench attacks:

| Target | Spearman |
|---|---:|
| Official validation accuracy | -0.639 |
| Average robustness | -0.673 |
| FGSM, $\epsilon=3/255$ | -0.656 |
| PGD, $\epsilon=3/255$ | -0.654 |
| FGSM, $\epsilon=8/255$ | -0.697 |
| PGD, $\epsilon=8/255$ | -0.695 |
| AutoAttack | -0.705 |

## 5. Matched Epoch 0/5/10 Trajectory

The first 96 architectures from the same 512-architecture list were trained with 4,096 CIFAR-10 training samples per epoch, PGD-3 adversarial training, SGD, and OneCycleLR. Metrics used the same 64 samples and FGSM protocol at epochs 0, 5, and 10. One training seed was used.

Chengdu 516 slice:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compute_nb201_metric_subset.py \
  --benchmark-json data/nasrobbench201/NAS-RobBench-201_all.json \
  --meta-json data/nasrobbench201/meta.json \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 --data-root /root/data \
  --target-metric autoattack --sample-size 512 --sampling uniform \
  --architecture-seed 20260718 --arch-start 0 --arch-stop 48 \
  --epochs 10 --eval-epochs 0 5 10 \
  --train-samples 4096 --metric-samples 64 --batch-size 128 \
  --train-attack pgd --train-pgd-steps 3 \
  --metric-attack fgsm --metric-pgd-steps 1 \
  --max-layers 4 --lr 0.05 --lr-schedule onecycle --seed 0 \
  --checkpoint-dir outputs/e19_trajectory_uniform96/shard_000_048/checkpoints \
  --output outputs/e19_trajectory_uniform96/shard_000_048/metrics.csv
```

Zhongceyuan slice; all arguments are identical except the existing dataset root, slice, and output paths:

```bash
cd /root/robust_metric_lab_nas
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compute_nb201_metric_subset.py \
  --benchmark-json data/nasrobbench201/NAS-RobBench-201_all.json \
  --meta-json data/nasrobbench201/meta.json \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 --data-root /workspace/research/Datasets/cifar10 \
  --target-metric autoattack --sample-size 512 --sampling uniform \
  --architecture-seed 20260718 --arch-start 48 --arch-stop 96 \
  --epochs 10 --eval-epochs 0 5 10 \
  --train-samples 4096 --metric-samples 64 --batch-size 128 \
  --train-attack pgd --train-pgd-steps 3 \
  --metric-attack fgsm --metric-pgd-steps 1 \
  --max-layers 4 --lr 0.05 --lr-schedule onecycle --seed 0 \
  --checkpoint-dir outputs/e19_trajectory_uniform96/shard_048_096/checkpoints \
  --output outputs/e19_trajectory_uniform96/shard_048_096/metrics.csv
```

Local merge command:

```powershell
python scripts\merge_metric_shards.py `
  --inputs `
    outputs\nasrobbench201_server\e19_trajectory_uniform96\shard000_048.csv `
    outputs\nasrobbench201_server\e19_trajectory_uniform96\shard048_096.csv `
  --expected-architectures 96 `
  --expected-epochs 0 5 10 `
  --output outputs\nasrobbench201_server\e19_trajectory_uniform96\metrics_uniform96_e0e5e10.csv
```

| Metric | Epoch 0 | Epoch 5 | Epoch 10 |
|---|---:|---:|---:|
| `feature_measure_mmd` | -0.620 | -0.532 | -0.588 |
| `racl_arch_lipschitz_log_upper` | -0.225 | -0.196 | -0.208 |
| `racl_input_gradient_lower` | -0.486 | +0.744 | +0.812 |

MMD top-10% mean AutoAttack accuracy is 0.384, 0.402, and 0.394 at epochs 0, 5, and 10; the population mean is 0.365. Epoch-5 MMD therefore gives a 3.78 percentage-point mean improvement in this subset.

MMD rank stability is 0.233 between epochs 0 and 5, 0.274 between epochs 0 and 10, and 0.654 between epochs 5 and 10. The predictive signal persists, but initialization and trained representations do not rank every architecture identically.

The RACL gradient term changes to the wrong direction after training. This is not a contradiction of the paper: RACL uses the gradient term jointly with classification loss and a confidence-constrained architecture distribution, not as an isolated architecture-ranking metric. Undertrained low-capacity or nearly constant networks can have small gradients while also having poor accuracy.

## 6. Confound and Residual Audits

MMD is sensitive to the operation composition of NAS-Bench-201. More `nor_conv_3x3` operations tend to reduce MMD and increase robustness; more `avg_pool_3x3` operations tend to increase MMD and reduce robustness.

After controlling the counts of four independent operation types, MMD's partial Spearman is:

| Experiment | Epoch | Raw | Given operation counts |
|---|---:|---:|---:|
| Uniform 512 | 0 | -0.705 | -0.324 |
| Matched 96 | 0 | -0.620 | -0.150 |
| Matched 96 | 5 | -0.532 | -0.335 |
| Matched 96 | 10 | -0.588 | -0.446 |

Thus, operation priors explain a substantial part, but not all, of the MMD signal. Any zero-cost paper must include operation-count baselines and residual analyses.

Controlling official final validation accuracy changes matched epoch-5/10 MMD partial Spearman to `+0.443/+0.386`. NAS-RobBench clean and robust accuracies are extremely coupled, so MMD currently predicts final robust accuracy well but has not shown that it explains the small robustness residual left after final clean accuracy is known.

Epoch-0 values recomputed on the same Chengdu 516 GPU are exactly reproducible. The Zhongceyuan 4090-D half has MMD rank agreement 0.973 with the original Chengdu run, with one numerical outlier. Future seed comparisons should use one hardware/software environment before cross-device replication.

## 7. Decision

`feature_measure_mmd` is supported as a promising low-fidelity and training-free ranking proxy for final NAS-RobBench robust accuracy. It is not yet supported as a mature-model robustness metric or as a clean-accuracy-independent robustness mechanism.

The strongest paper-safe claim at this stage is:

> At random initialization and very early adversarial training, feature-distribution MMD provides a fast predictor of final robust architecture ranking, remains useful after controlling simple operation counts, and outperforms fixed-architecture RACL Lipschitz terms under the tested protocol.

The claim still requires multiple initialization seeds, BN-calibration and attack/sample-size ablations, a hand-designed operation-count baseline, and validation on another search space or dataset.

## 8. Files and Analysis Command

Local raw and derived files:

```text
outputs/nasrobbench201_server/e19_zero_cost/metrics_uniform512_fgsm_n64_seed0.csv
outputs/nasrobbench201_server/e19_zero_cost/analysis_final/
outputs/nasrobbench201_server/e19_trajectory_uniform96/metrics_uniform96_e0e5e10.csv
outputs/nasrobbench201_server/e19_trajectory_uniform96/analysis_final/
```

Server checkpoints:

```text
/root/robust_metric_lab_nas/outputs/e19_zero_cost_uniform512/checkpoints
/root/robust_metric_lab_nas/outputs/e19_trajectory_uniform96/shard_000_048/checkpoints
/root/robust_metric_lab_nas/outputs/e19_trajectory_uniform96/shard_048_096/checkpoints
```

Local analysis command:

```powershell
cd E:\projects\robust_metric_lab
$env:PYTHONPATH="$PWD\src"

python scripts\analyze_racl_mmd_fidelity.py `
  --zero-cost outputs\nasrobbench201_server\e19_zero_cost\metrics_uniform512_fgsm_n64_seed0.csv `
  --early-fidelity outputs\nasrobbench201_server\e19_trajectory_uniform96\metrics_uniform96_e0e5e10.csv `
  --early-name E19-matched-trajectory `
  --mature-fidelity outputs\nasrobbench201_server\e17_full50_seed0\analysis_final_stopped\complete_metrics.csv `
  --bootstrap-repeats 2000 `
  --seed 20260718 `
  --output-dir outputs\nasrobbench201_server\e19_trajectory_uniform96\analysis_final
```

Execution anomaly: an earlier stale launcher resumed after the Chengdu shard ended and attempted an extra Zhongceyuan process with the wrong `/root/data` path. That extra process failed; it was not the valid Zhongceyuan run, which used `/workspace/research/Datasets/cifar10` and produced all 144 expected rows. A four-architecture recovery run was conservatively executed before this cause was identified, but was not merged because its keys duplicate complete rows. The final merged file uses two non-overlapping 48-architecture shards and passes the exact 288-row grid check.
