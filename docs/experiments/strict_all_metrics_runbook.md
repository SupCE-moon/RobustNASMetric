# Strict all-metric and E37 runbook

This runbook freezes the execution contract used for the strict cross-space release. It is an operational record, not a substitute for raw evidence.

## Scope and safety

- Run only inside container `wangjiahao` under `/workspace/robust_metric_lab_nas`.
- Benchmark bundle: `/workspace/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1`.
- CIFAR-10 root: `/workspace/DynamicBN20251203/data`.
- Never mutate a prior experiment directory. New runs use a new `RUN_ID`; repair scripts write a new target plus a SHA-256 manifest.
- Only raw rows with `status=ok`, finite values, unique keys, exact manifest IDs and the frozen protocol may enter the total ranking.

## Frozen main protocol

The eight settings contain exactly 560 objects per metric: RB40 mature (40), NB201 epochs 0/5/10/25/50 (64 each), and DARTS epochs 0/100 (100 each). The sample protocol is seed 20260808, offset 0, 64 images and epsilon 8/255. Targets come only from the teacher bundle manifests.

## Commands

From the project root, with `PYTHONPATH=src:.`:

```bash
bash scripts/run_strict_all_metrics.sh smoke
bash scripts/run_e37_full.sh smoke
RUN_ID=strict_importance_v1 bash scripts/run_strict_all_metrics.sh full
RUN_ID=e37_full_metric_design_v1 bash scripts/run_e37_full.sh full
```

The smoke phase evaluates four predeclared manifest-prefix objects in every setting. Each script creates an immutable `run_manifest.json` before evaluation, records per-object runtime in the raw CSV, prints PyTorch peak CUDA allocation per setting, and verifies row count, keys, metric names, finiteness and status before writing `state=COMPLETE`.

`scripts/wait_and_run_strict_smoke.sh` is the unattended launcher. It requires at least three consecutive one-minute samples with total compute-process GPU use at or below 1,000 MiB. Every setting repeats the same gate. After both smoke grids pass their built-in assertions, the launcher automatically runs the complete 560-row importance grid and 5,040-row E37 grid.

## Release checks

Build the ranking from raw CSV files with `scripts/build_strict_metric_ranking.py`, then run `scripts/verify_strict_release.py` against the three teacher manifests and the frozen eligibility registry. A metric is absent from the total table unless all 560 raw rows pass. `scripts/repair_existing_metric_grid.py` may replace only failed rows with matching frozen fields and always emits source/repair/target hashes and exact replaced keys.
