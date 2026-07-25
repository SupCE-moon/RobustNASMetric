# E17 NAS-RobBench-201 Full-Training Capacity Validation

Status: stopped by research decision on 2026-07-18 after 91/96 architectures completed. The retained report is in `outputs/nasrobbench201_server/e17_full50_seed0/analysis_final_stopped/e17_full50_capacity_report.md`. No E17 model checkpoints were saved by the original runner, so attack parameters cannot be recomputed on these exact weights.

## 1. Question

E16 found that `perturbation_capacity_delta (Ours)` has the wrong-sign Spearman correlation with official NAS-RobBench-201 AutoAttack accuracy:

| Checkpoint | Raw Spearman | Expected direction |
|---:|---:|---|
| epoch 5 | +0.506 | negative |
| epoch 10 | +0.350 | negative |

The main question is whether this is caused by evaluating severely undertrained representations, or whether the metric does not transfer from fully trained RobustBench defenses to the NAS-Bench-201 architecture family.

## 2. Why E16 Is Not The Official Training Protocol

NAS-RobBench-201 trains each NAS-Bench-201 network for 50 epochs. The released training command and `main.py` use the C=16, N=5 tiny network, full CIFAR-10 training set, batch size 256, SGD with momentum 0.9 and weight decay $5\times10^{-4}$, OneCycleLR with maximum learning rate 0.1, and PGD-7 adversarial training at $\epsilon=8/255$ and step size $2/255$.

E16 instead used 4096 training images, batch size 128, PGD-3, and stopped at epochs 5 and 10.

| Quantity | E16 epoch 10 | Official-style epoch 50 | Ratio |
|---|---:|---:|---:|
| Training-image exposures | 40,960 | 2,500,000 | 1.64% |
| Approximate optimizer updates | 320 | 9,800 | 3.27% |
| Training PGD steps | 3 | 7 | 42.9% |

Therefore, E16 primarily tests an early representation-maturity proxy. A slowly learning or nearly collapsed network can have low clean and adversarial feature capacity and consequently a small absolute capacity displacement. That small value can look robust even though the fully trained architecture is weak. A faster-learning architecture can show richer feature structure and larger displacement early, producing the observed positive correlation.

The decrease from $+0.506$ at epoch 5 to $+0.350$ at epoch 10 is consistent with this explanation, but does not prove it. A matched full-training experiment is required.

Official sources:

- NAS-RobBench-201 repository: <https://github.com/TT2408/nasrobbench201>
- Official training script: <https://github.com/TT2408/nasrobbench201/blob/main/main.py>
- ICLR 2024 paper: <https://arxiv.org/abs/2403.13134>

## 3. Timing Pilot

On 2026-07-17, three architectures were trained independently with the official-style 50-epoch core protocol. Each run included one 256-image PGD-3 metric evaluation at epoch 50.

| Server/GPU | Wall time | Minutes |
|---|---:|---:|
| Chengdu 516 RTX 4090 | 703 s | 11.7 |
| Zhongceyuan RTX 4090D GPU 0 | 712 s | 11.9 |
| Zhongceyuan RTX 4090D GPU 1 | 745 s | 12.4 |

Local pilot files:

```text
outputs/nasrobbench201_server/full50_pilot/full50_timing_516_slice0.csv
outputs/nasrobbench201_server/full50_pilot/full50_timing_516_slice0.log
outputs/nasrobbench201_server/full50_pilot/full50_timing_zcy0_slice1.csv
outputs/nasrobbench201_server/full50_pilot/full50_timing_zcy0_slice1.log
outputs/nasrobbench201_server/full50_pilot/full50_timing_zcy1_slice2.csv
outputs/nasrobbench201_server/full50_pilot/full50_timing_zcy1_slice2.log
```

These three architectures are only a runtime pilot and are not used to estimate correlation.

## 4. Recommended Formal Experiment

Use 96 uniformly sampled, unique architectures with one fixed training seed. Keep the architecture sample fixed with `--architecture-seed 20260717` and use training seed 0. Uniform sampling avoids choosing the validation subset using the AutoAttack oracle.

Primary protocol:

| Item | Fixed value |
|---|---|
| Architectures | 96 unique architectures |
| Training seed | 0 |
| Training | 50 epochs, full 50,000-image CIFAR-10 set |
| Network | NAS-Bench-201 tiny network, C=16, N=5 |
| Optimizer | SGD, momentum 0.9, weight decay $5\times10^{-4}$ |
| Schedule | OneCycleLR, maximum LR 0.1 |
| Adversarial training | PGD-7, $\epsilon=8/255$, step size $2/255$ |
| Metric checkpoints | epochs 10, 25, and 50 |
| Metric attack | white-box PGD-3, unchanged from E16 |
| Metric sample | fixed 256 CIFAR-10 test images |
| Oracle target | released final AutoAttack accuracy; never used during training or sampling |

The metric protocol remains identical to E16 so that the main changed variable is representation maturity and official-style training fidelity.

## 5. Sample-Size And One-Day Budget

For a two-sided rank-correlation test at $\alpha=0.05$ and 80% power, detecting a true correlation around $|\rho_s|=0.4$ requires approximately 47 independent architectures; detecting $|\rho_s|=0.3$ requires approximately 85. Thus:

- 20 architectures are suitable for screening but too uncertain for the main NAS conclusion.
- 64 architectures are the minimum practical full-training validation.
- 96 architectures are recommended because they can detect a moderate correlation around 0.3 while fitting the available hardware budget.
- 128 architectures with one seed is feasible, but 96 already provides the intended power for this first directional validation at lower cost.

The measured one-checkpoint base time for 96 architectures and one seed is approximately

$$
\frac{96}{3}\times 720\text{ s}=6.4\text{ h}.
$$

Allowing for two additional metric checkpoints, server variation, merging, and a failed-job reserve gives an expected total of roughly 9--11 hours.

Execution order:

1. Finish all 96 architectures with seed 0.
2. Merge and validate the complete 96-by-3 checkpoint grid.
3. Analyze the epoch-10/25/50 trajectory and make the first research decision.
4. Only if the result supports continuing with this metric, use a second seed later as a paper-level stability experiment.

## 6. Server Allocation

Use equal slices and existing CIFAR-10 copies:

| Slice | Server/GPU | Data root |
|---:|---|---|
| `[0,32)` | Chengdu 516 RTX 4090 | `/root/data` |
| `[32,64)` | Zhongceyuan RTX 4090D GPU 0 | `/workspace/research/Datasets/cifar10` |
| `[64,96)` | Zhongceyuan RTX 4090D GPU 1 | `/workspace/research/Datasets/cifar10` |

Common command arguments:

```bash
python3 scripts/compute_nb201_metric_subset.py \
  --benchmark-json data/nasrobbench201/NAS-RobBench-201_all.json \
  --meta-json data/nasrobbench201/meta.json \
  --nasrobbench-repo /root/nasrobbench201 \
  --dataset cifar10 \
  --target-metric autoattack \
  --sample-size 96 \
  --sampling uniform \
  --architecture-seed 20260717 \
  --eval-epochs 10 25 50 \
  --train-samples 50000 \
  --metric-samples 256 \
  --batch-size 256 \
  --train-attack pgd \
  --train-pgd-steps 7 \
  --metric-attack pgd \
  --metric-pgd-steps 3 \
  --max-layers 4 \
  --lr 0.1 \
  --lr-schedule onecycle \
  --seed 0 \
  --data-root <server-data-root> \
  --arch-start <slice-start> \
  --arch-stop <slice-stop> \
  --output outputs/e17_full50/<slice>.csv
```

## 7. Primary Analysis And Decision Rule

For checkpoint $t\in\{10,25,50\}$, compute

$$
\rho_t
=
\operatorname{Spearman}\!\left(
\left\{D^{\mathrm{cap}}_{a,t}\right\}_{a=1}^{96},
\left\{R^{\mathrm{AA}}_a\right\}_{a=1}^{96}
\right).
$$

Because capacity displacement is pre-specified as lower-is-better and robust accuracy is higher-is-better, the expected sign is $\rho_t<0$.

Report Spearman and architecture-bootstrap 95% confidence intervals at each checkpoint. Also report partial/residual associations after controlling for same-run clean accuracy, natural loss, and `train_epoch_loss`.

Pre-specified interpretation:

- If epoch-50 correlation becomes negative and is materially stronger than epoch 10, the undertraining/maturity explanation is supported.
- If epoch-50 remains near zero, the metric may rank mature RobustBench defenses but not the narrow NAS-Bench-201 cell family.
- If epoch-50 remains positive, absolute capacity displacement is likely dominated by feature scale or architecture-dependent representation richness; changing its sign after observing the result is not allowed.
- If only the residual or normalized version is negative, the paper must present scale control as part of the metric rather than as an optional post-hoc correction.
