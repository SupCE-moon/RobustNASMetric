# E20: MMD Cross-Search-Space Validation in the RACL/DARTS Space

Date: 2026-07-18

## Material Passport

- Type: pre-registered experiment plan
- Status: designed, not yet executed
- Primary question: does epoch-0 `feature_measure_mmd` predict adversarial robustness outside NAS-Bench-201?
- Search space: discrete DARTS/PC-DARTS normal and reduction cells used by RACL
- Dataset: CIFAR-10, reusing existing server copies
- Initial seed policy: one architecture seed and one weight seed
- Maximum initial budget: 12 GPU-hours, with a one-architecture timing gate before formal launch

## 1. Why This Is a Real Cross-Space Test

NAS-Bench-201 uses one cell input, three computed nodes, six fixed edges, five operation choices, and the last node as cell output. RACL uses the DARTS/PC-DARTS cell family: two input nodes, four intermediate nodes, learned edge selection, normal and reduction cells, and an output formed from intermediate nodes.

The RACL operation set is:

```text
sep_conv_3x3
sep_conv_5x5
dil_conv_3x3
dil_conv_5x5
max_pool_3x3
avg_pool_3x3
skip_connect
zero
```

The change introduces depthwise separable convolution, dilated convolution, two-input cells, top-two incoming-edge selection, concatenation, and separately searched reduction cells. Success here cannot be explained only by memorizing NAS-Bench-201's five-operation prior.

## 2. Architecture Sampling

Generate 64 discrete DARTS genotypes before training or reading robustness labels.

For each normal and reduction cell independently:

1. Use two input nodes and four intermediate nodes.
2. For every intermediate node, choose two distinct predecessors uniformly.
3. On every retained edge, choose uniformly from the seven non-zero RACL operations.
4. Concatenate all four intermediate nodes as cell output.
5. Reject exact duplicate normal/reduction genotype pairs.

Use architecture seed `20260720`. Store every genotype in a manifest before metric computation. Do not stratify or resample using MMD, clean accuracy, or robust accuracy.

## 3. Network and Epoch-0 Metric

Use the RACL search-stage macro network for the primary proxy:

- Initial channels: 16.
- Eight cells: six normal and two reduction cells.
- Six nodes per cell when counting the two inputs and four intermediate nodes.
- Random initialization seed: 0.
- Fixed 64 CIFAR-10 test images.
- White-box FGSM, $\epsilon=8/255$.
- Same standardized spatial-atom random Fourier MMD implementation as E19.
- Same GPU/software environment for all 64 architectures.

Record:

```text
feature_measure_mmd
racl_arch_lipschitz_log_upper
racl_input_gradient_lower
parameter_count
FLOPs
latency
normal/reduction operation counts
skip ratio
pool ratio
mean path depth
```

The primary proxy is epoch-0 MMD. Epoch-5/10 MMD is secondary and must not replace a failed epoch-0 result.

## 4. Timing Gate

Before the formal run, evaluate three architectures for one full adversarial-training epoch and one PGD-20 evaluation pass.

Let $t_{\mathrm{epoch}}$ be median seconds per architecture per epoch and $t_{\mathrm{eval}}$ the evaluation time. Estimate

$$
T_{\mathrm{GPU}}
=
64\left(20t_{\mathrm{epoch}}+t_{\mathrm{eval}}\right).
$$

Decision:

- If $T_{\mathrm{GPU}}\le 12$ GPU-hours, use 64 architectures and 20 epochs.
- If $12<T_{\mathrm{GPU}}\le 24$ GPU-hours, use 48 architectures and 15 epochs.
- If $T_{\mathrm{GPU}}>24$ GPU-hours, use 32 architectures and 10 epochs as a falsification pilot; do not call it confirmatory.

Use Chengdu 516 RTX 4090 and Zhongceyuan RTX 4090-D GPU 0. Do not use Zhongceyuan GPU 1 while occupied. The A100 is unnecessary unless the DARTS implementation exceeds 24 GB.

## 5. Independent Robustness Labels

Each candidate must be trained independently; do not use inherited one-shot supernet weights as ground truth.

Primary fast protocol:

- Full CIFAR-10 training set.
- PGD-3 adversarial training, $\epsilon=8/255$, step size $2/255$.
- SGD with momentum 0.9 and weight decay $5\times10^{-4}$.
- OneCycle or cosine schedule fixed across architectures.
- 20 epochs after the timing gate.
- Save checkpoints at epochs 0, 5, 10, and 20.

Evaluation:

- Clean accuracy on all 10,000 test images.
- PGD-20 accuracy on a fixed 2,000-image subset as the primary fast target.
- AutoAttack on a fixed 1,000-image subset for all architectures if the timing gate permits.
- Full-test AutoAttack for 12 architectures selected by pre-registered MMD quantiles: three from each quartile, not only the best architectures.

PGD-20 is the rapid cross-space label. AutoAttack is the confirmation label and must be used before a paper-level claim.

## 6. Baselines and Confound Controls

Minimum baselines:

1. Random ranking.
2. Parameter count and FLOPs.
3. Raw operation counts and a five-fold cross-validated operation-count predictor.
4. RACL fixed-architecture Lipschitz upper term.
5. RACL input-gradient term.
6. Same-run epoch-0 clean accuracy.

Paper-level extension:

7. TRNAS R-Score.
8. CRoZe or another published training-free robust proxy.
9. NASWOT and SynFlow as non-robust zero-cost controls.

The operation-count predictor must be cross-validated. Reporting an in-sample fitted correlation would overstate the baseline.

## 7. Statistical Analysis

Primary endpoint:

$$
\rho_0
=
\rho_{\mathrm{Spearman}}
\left(
M_{\mathrm{MMD}}^{(0)},
\mathrm{PGD20Acc}^{(20)}
\right).
$$

Report:

- Spearman with architecture-bootstrap 95% interval.
- Kendall $\tau$ as a secondary ranking statistic.
- Partial Spearman controlling parameter count, FLOPs, and operation counts.
- Partial Spearman controlling final clean accuracy as a diagnostic only.
- Top-quartile MMD robust-accuracy mean versus population mean.
- Pairwise ranking accuracy for architecture pairs separated by at least two robust-accuracy percentage points.
- MMD rank stability across initialization and epochs 5/10.

Do not select direction from the data. MMD is pre-specified as lower-is-better.

## 8. Decision Rules

Strong cross-space support requires all of:

1. Raw Spearman $\rho_0\le-0.40$.
2. Bootstrap 95% interval excludes zero.
3. Operation/size-controlled partial Spearman $\le-0.20$.
4. Lowest-MMD quartile exceeds population mean PGD-20 accuracy by at least two percentage points.
5. The sign agrees for the AutoAttack subset.

Interpretation:

- Raw and partial criteria pass: MMD transfers beyond the NAS-Bench-201 operation prior.
- Raw passes but partial fails: MMD mainly encodes DARTS operation or size priors.
- PGD-20 passes but AutoAttack fails: MMD is attack-specific under this protocol.
- Results vary sharply by initialization seed: one-seed evidence is exploratory; add seeds 1 and 2 before search experiments.
- Raw correlation fails: stop scaling the DARTS experiment and revise the metric before integrating it into a NAS algorithm.

## 9. Why Not Reproduce Full RACL First

RACL search optimizes a weight-sharing supernet and a stochastic architecture distribution. Reproducing it tests the joint effects of supernet optimization, the Lipschitz constraint, confidence learning, and discretization. That is not the cleanest first test of MMD transfer.

E20 instead asks the narrower question:

> Given independently sampled DARTS architectures, does random-initialization MMD rank their independently obtained robust accuracies?

Only after this question is answered should MMD be inserted into a one-shot DARTS/RACL search objective.

## 10. Required Outputs

```text
data/darts_e20_architectures_seed20260720.json
outputs/darts_e20/epoch0_metrics.csv
outputs/darts_e20/training_metrics.csv
outputs/darts_e20/robust_evaluation.csv
outputs/darts_e20/correlation_analysis.csv
outputs/darts_e20/checkpoints/
outputs/darts_e20/e20_cross_space_report.md
```

Every CSV row must include genotype, architecture seed, weight seed, server/GPU, data root, attack parameters, checkpoint path, and implementation commit or source hash.
