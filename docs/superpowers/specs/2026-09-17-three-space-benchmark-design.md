# Three-Space Robust Metric Benchmark Design

## Objective

Create a reproducible RB40/NB201-64/DARTS-100 evaluation workspace in the
user-owned `wangjiahao` container, complete the missing E25+ metric coverage,
and conditionally develop new metrics without tuning on the final aggregate
ranking.

The research target is to place a defensible metric in the top five by the
pre-registered all-setting mean. This is a target, not a guaranteed outcome;
all candidates and negative results remain reportable.

## Authorized systems and boundaries

- Source container: `huangyue`, accessed only for the teacher-provided project
  and `/root/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1`.
- Destination container: `wangjiahao`.
- Destination host bind mount: `/home/usertest` maps to `/workspace` in
  `wangjiahao`.
- Existing `/workspace/DynamicBN20251203` is preserved unchanged.
- No other containers, user directories, GPUs, or datasets are accessed.
- Credentials are never stored in files, commands, logs, or project outputs.

## Destination layout

```text
/workspace/robust_metric_lab_nas/
  configs/
  docs/
  scripts/
  src/
  tests/
  outputs/
    student_metrics/wangjiahao/
  shared_benchmarks/
    robust_metric_v1/
```

The current local project is copied without `.venv`, `.idea`, Python caches,
or local model/data caches. Historical CSVs and experiment reports are kept
because they are needed to audit E25+ coverage and compare old versus new
protocols.

## Benchmark snapshot and integrity exception

The source bundle reported `TRANSFER_STATUS.json.state=IN_PROGRESS` on
2026-09-17. The teacher explicitly authorized use despite that stale status.
The project must not silently rewrite the status or claim that the original
bundle passed.

Before evaluation, record:

- source container and source path;
- copy timestamp;
- original `TRANSFER_STATUS.json` content and SHA-256;
- manifest SHA-256 values;
- copied file counts and byte totals;
- verifier stdout/stderr and exit code.

Run `verify_student_benchmark_bundle.py` on the copied snapshot. If it passes,
use the complete three-space protocol. If it fails, continue only for a space
whose required manifests/checkpoints pass explicit count, checksum, and model
load tests. A failed or incomplete space is reported as unavailable and is not
silently dropped from an aggregate mean.

## Frozen evaluation protocol

The new benchmark has eight settings:

1. RB40 mature
2. NB201 e0
3. NB201 e5
4. NB201 e10
5. NB201 e25
6. NB201 e50
7. DARTS e0
8. DARTS e100

Targets are frozen manifest labels: RB40 AutoAttack robust accuracy, NB201
official final AutoAttack accuracy, and DARTS full-test PGD-20 accuracy.

Use CIFAR-10 test samples with `sample_offset=0`, `num_samples=64`,
`seed=20260808`, and `epsilon=8/255`. Every architecture within a setting uses
the same ordered samples. E0 metrics that need perturbed inputs use a frozen
Standard-model shared-adversarial cache or a pre-registered label-independent
perturbation, never self-generated adversarial examples from a random model.

Metric direction is registered from its mathematical definition before target
labels are joined. Report raw Spearman, direction-adjusted Spearman utility,
Pearson, Kendall tau, valid count, failures, runtime, and bootstrap 95% CI.
Also report parameter/family controls for RB40 and operation-count partial
Spearman for NB201.

The all-setting mean is the simple mean of the eight direction-adjusted
Spearman utilities only when all eight values are valid. Missing settings do
not receive zero and are not omitted to inflate the mean.

## Execution stages

1. Copy and inventory the project and benchmark snapshot.
2. Recreate a Linux Python environment from declared dependencies.
3. Run the bundle verifier and independent file/checkpoint audits.
4. Smoke-test four entries per space, including model loading, normalization,
   hooks, metric output, and CSV schema.
5. Build the E25+ metric-by-setting coverage matrix.
6. Complete missing evaluation cells using the frozen protocol.
7. Produce the eight-setting baseline table and statistical diagnostics.
8. Only if no existing candidate reaches the target, design a small,
   mechanism-motivated candidate set using development splits.
9. Freeze candidate formula, direction, and hyperparameters before one final
   evaluation on held-out entries and the full benchmark.

## New-metric anti-overfitting rule

New metrics may be selected using deterministic development partitions within
each space. Final held-out ranks and the official all-setting table are read
only after formulas, directions, layer selection, and aggregation are frozen.
No result-dependent sign flips, selective setting removal, or unreported
candidate deletion is allowed.

## Outputs

Each run writes a new timestamped directory under:

```text
outputs/student_metrics/wangjiahao/<metric>/<timestamp>/
```

Required artifacts are `metric_values.csv`, `correlations.csv`, `run.json`,
the exact command, logs, failure rows, environment versions, source snapshot
metadata, and a Markdown report. Historical results are never overwritten.

## Completion criteria

- Destination layout is reproducible and old project data is untouched.
- Benchmark snapshot provenance and verifier result are preserved.
- Every claimed setting has complete expected rows and explicit failure audit.
- E25+ coverage gaps are either filled or documented with a reproducible
  blocker.
- The final table uses the frozen eight-setting definition.
- Any top-five claim includes uncertainty, confound diagnostics, and the full
  candidate ledger.
