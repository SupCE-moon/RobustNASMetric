# RobustMetric Strict All-Metrics and New-Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every cross-space-comparable metric under the frozen eight-setting protocol, formally exclude space-specific metrics, add nine pre-registered interpretable E37 metrics, and publish a strict ranking that cannot admit partial grids.

**Architecture:** Restore the deployed benchmark framework from `wangjiahao` into local version control, then place eligibility and raw-grid validation in front of every ranking operation. Repair the remaining importance/TRNAS cases without changing frozen semantics, implement E37 as per-sample primitives with exactly equivalent chunked aggregation, and run all server experiments into new immutable directories.

**Tech Stack:** Python 3.10, PyTorch 2.5, pandas, NumPy, SciPy, pytest, Bash, Docker, RobustBench 1.1.1, CSV/JSON manifests.

**Spec:** `docs/superpowers/specs/2026-09-30-strict-all-metrics-and-new-metrics-design.md`

## Global Constraints

- Operate only in container `wangjiahao` and `/workspace/robust_metric_lab_nas` on the server.
- Settings are exactly RB40 mature; NB201 e0/e5/e10/e25/e50; DARTS e0/e100, with object counts 40; 64 each; 100 each.
- Protocol is exactly `sample_offset=0`, `num_samples=64`, `seed=20260808`, `epsilon=8/255`.
- Targets are RB40 manifest AutoAttack, NB201 official final AutoAttack, and DARTS full-test PGD-20 accuracy.
- A ranked metric requires 560 unique, finite, `status=ok` raw rows and all eight valid setting correlations.
- Never overwrite or delete original results, failures, logs, or backups.
- Mainline new metrics contain no label-fitted weights. Formulas, directions, layers, normalization, and aggregation freeze before formal labels or held-out results are read.
- Use test-first development and commit only task-related files; preserve unrelated dirty-worktree changes.

## Review Focus

1. A 29/40 RB40 grid must be rejected, not silently ranked (Task 3).
2. RACL space-specific proxies must never leak into cross-space means (Task 2).
3. Chunk sizes 1, 2, and full batch must preserve importance/E37 definitions (Tasks 4 and 7).
4. Kang TRNAS must be finite with unchanged `1e-6` semantics or remain explicitly excluded (Task 5).
5. Duplicate keys, wrong IDs, changed targets, or changed protocol fields must abort release (Tasks 3 and 9).

---

### Task 1: Restore and Lock the Deployed Benchmark Framework

**Files:**
- Create: `src/robust_metric_lab/student_benchmark.py`
- Create: `src/robust_metric_lab/student_runner.py`
- Create: `src/robust_metric_lab/student_analysis.py`
- Create: `scripts/analyze_student_benchmark.py`
- Create: `scripts/audit_student_benchmark.py`
- Create: `scripts/evaluate_rb40_student.py`
- Create: `scripts/evaluate_nb201_student.py`
- Create: `scripts/evaluate_darts100_student.py`
- Create: `tests/test_student_benchmark_protocol.py`
- Create: `tests/test_student_benchmark_manifests.py`
- Create: `tests/test_student_runner.py`
- Create: `tests/test_student_benchmark_analysis.py`
- Modify: `src/robust_metric_lab/metrics.py`
- Modify: `src/robust_metric_lab/metric_registry.py`

**Interfaces:**
- Consumes: deployed passing source in `/workspace/robust_metric_lab_nas`.
- Produces: `BENCHMARK_SETTINGS`, `MetricValueRow`, `validate_complete_setting_grid()`, `evaluate_items()`, `analyze_frame()`, and `write_analysis_outputs()` locally.

- [ ] Record SHA-256 of every listed deployed file using `docker exec wangjiahao sha256sum`; save as `docs/experiments/strict_benchmark_source_sha256.txt`.
- [ ] Copy only missing files from the `/home/usertest/robust_metric_lab_nas` host mapping with `scp -P 17010`. Do not overwrite local `metrics.py` or `metric_registry.py`; merge only missing E36 blocks.
- [ ] Run `pytest tests/test_student_benchmark_protocol.py tests/test_student_benchmark_manifests.py tests/test_student_runner.py tests/test_student_benchmark_analysis.py tests/test_e36_batched.py -v`.
- [ ] Verify copied-file hashes match the server record; use the test suite to verify the two merged files.
- [ ] Commit with `git commit -m "chore: restore strict benchmark framework"`.

Expected: all restored tests pass, and `all_setting_mean()` rejects a mapping with fewer than eight settings.

### Task 2: Add an Explicit Metric Eligibility Registry

**Files:**
- Create: `configs/metric_eligibility.json`
- Create: `src/robust_metric_lab/metric_eligibility.py`
- Create: `scripts/audit_metric_eligibility.py`
- Create: `tests/test_metric_eligibility.py`

**Interfaces:**
- Produces `MetricEligibilityRecord`, `load_metric_eligibility()`, `validate_registry()`, and `eligible_cross_space_names()`.

- [ ] Write failing tests proving both RACL proxies are absent from the cross-space list and every `METRIC_SPECS` name appears exactly once.

```python
class MetricEligibility(str, Enum):
    CROSS_SPACE = "cross_space"
    SPACE_SPECIFIC = "space_specific"
    IMPLEMENTATION_ERROR = "implementation_error"

@dataclass(frozen=True)
class MetricEligibilityRecord:
    metric_name: str
    eligibility: MetricEligibility
    supported_settings: tuple[str, ...]
    reason: str
```

- [ ] Run `pytest tests/test_metric_eligibility.py -v`; expect missing-module failure.
- [ ] Implement JSON loading and reject duplicates, empty reasons, unknown settings, missing registered metrics, and unregistered extras.
- [ ] Initially mark both RACL proxies `space_specific`; importance and three TRNAS metrics `implementation_error`; all other registered metrics `cross_space`.
- [ ] Implement `audit_metric_eligibility.py` to write `metric_name,eligibility,supported_settings,reason` and exit nonzero on registry mismatch.
- [ ] Run tests and commit with `git commit -m "feat: classify metric benchmark eligibility"`.

### Task 3: Build a Raw-Grid-Only Strict Ranking Gate

**Files:**
- Create: `src/robust_metric_lab/strict_ranking.py`
- Create: `scripts/build_strict_metric_ranking.py`
- Create: `tests/test_strict_ranking.py`

**Interfaces:**
- Produces `load_unique_metric_rows()`, `validate_metric_grid()`, `build_strict_ranking()`, `strict_ranking.csv`, and `excluded_metrics.csv`.

- [ ] Write failing tests for 29/40 rejection, duplicate-key abort, nonfinite rejection, protocol mismatch, manifest-ID mismatch, and target mismatch.

```python
RAW_KEY = (
    "benchmark", "setting", "model_or_arch_id", "checkpoint_epoch",
    "metric_name", "seed", "sample_offset", "num_samples",
)
EXPECTED_COUNTS = {
    "rb40_mature": 40, "nb201_e0": 64, "nb201_e5": 64,
    "nb201_e10": 64, "nb201_e25": 64, "nb201_e50": 64,
    "darts_e0": 100, "darts_e100": 100,
}
```

- [ ] Run `pytest tests/test_strict_ranking.py -v`; expect missing-module failure.
- [ ] Validate exact manifest sets, 560 rows, all status `ok`, finite values, frozen protocol, and exact targets before calling `analyze_frame()`.
- [ ] Make the CLI consume raw CSVs only; precomputed correlations are never accepted as ranking input.
- [ ] Write `strict_ranking.csv` and a complete `excluded_metrics.csv` with exact reasons.
- [ ] Run strict-ranking and student-analysis tests; commit with `git commit -m "feat: require complete raw grids for ranking"`.

### Task 4: Re-enable Importance with Exact Two-Pass Chunking

**Files:**
- Create: `src/robust_metric_lab/importance_batched.py`
- Create: `scripts/evaluate_importance_student.py`
- Create: `tests/test_importance_batched.py`
- Modify: `src/robust_metric_lab/metrics.py`
- Modify: `configs/metric_eligibility.json`

**Interfaces:**
- Produces `compute_importance_metric_batched(model, clean, labels, epsilon, batch_size, max_layers=4) -> float`.

- [ ] Write failing batch-size 1/2/full equivalence tests against the full-batch definition and a zero-gradient test requiring finite output.
- [ ] Run `pytest tests/test_importance_batched.py -v`; expect missing-module failure.
- [ ] Implement pass 1 as the exact mean of per-sample, per-channel spatial RMS gradients for each anchor layer.
- [ ] Implement pass 2 by regenerating the same per-sample FGSM inputs, applying the global importance vectors to clean/adversarial MAD capacities, and averaging all 64 relative drifts.

```python
def compute_importance_metric_batched(
    model, clean, labels, *, epsilon, batch_size, max_layers=4
) -> float:
    weights = accumulate_global_channel_importance(
        model, clean, batch_size=batch_size, max_layers=max_layers
    )
    return accumulate_weighted_capacity_drift(
        model, clean, labels, weights,
        epsilon=epsilon, batch_size=batch_size, max_layers=max_layers,
    )
```

- [ ] Implement a `--benchmark rb40|nb201|darts` evaluator that requires exactly `perturbation_capacity_importance_delta` and writes through `evaluate_items()`.
- [ ] Change importance eligibility to `cross_space` only after equivalence tests pass.
- [ ] Run importance, geometry, and runner tests; commit with `git commit -m "fix: compute importance metric with bounded memory"`.

### Task 5: Resolve or Formally Exclude Kang TRNAS Underflow

**Files:**
- Modify: `src/robust_metric_lab/trnas.py`
- Modify: `tests/test_trnas_metrics.py`
- Modify: `configs/metric_eligibility.json`
- Create: `scripts/diagnose_kang_trnas.py`

**Interfaces:**
- Produces `compute_trnas_metrics_stable(...) -> TrnasMetricResult` with unchanged `weight_step_size=1e-6`.

- [ ] Write a failing test where only the exact `AssertionError("underflow in dt 0.0")` triggers a float64 retry; verify result equality with direct float64 computation.
- [ ] Implement `diagnose_kang_trnas.py` to load only `Kang2021Stable`, record complete float32/float64 tracebacks and finite fields, and never write metric rows.
- [ ] Extract the existing formula into `_run_trnas_once()`; retry a deep-copied float64 model only for the exact underflow assertion. Propagate every other exception and keep `1e-6` unchanged.
- [ ] If real Kang float64 succeeds, mark the three TRNAS metrics `cross_space`; if it fails, preserve the diagnostic, keep them `implementation_error`, and never compute 39/40 correlations.
- [ ] Run TRNAS and runner tests; commit with `git commit -m "fix: make TRNAS underflow handling auditable"`.

### Task 6: Implement Nine Pre-Registered E37 Metrics

**Files:**
- Create: `src/robust_metric_lab/e37_metrics.py`
- Create: `tests/test_e37_metrics.py`
- Modify: `src/robust_metric_lab/metric_registry.py`
- Modify: `configs/metric_eligibility.json`

**Interfaces:**
- Produces `E37_METRIC_NAMES` and `compute_e37_per_sample(clean_features, adversarial_features, indices) -> dict[str, Tensor]`, each tensor `[batch]`.

- [ ] Freeze these nine lower-is-better names in tests and registry:

```python
E37_METRIC_NAMES = (
    "feature_measure_flow_corr_worst_delta",
    "feature_measure_flow_corr_rms_delta",
    "feature_measure_flow_corr_tophalf_delta",
    "feature_measure_spectrum_fisher_rao_delta",
    "feature_measure_spectrum_triangular_delta",
    "feature_measure_spectrum_ks_delta",
    "feature_measure_multiview_geometric_delta",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
)
```

- [ ] Write tests requiring finite near-zero identical-input results, clean/perturbed symmetry, and invariance to positive layerwise scaling.
- [ ] Run `pytest tests/test_e37_metrics.py -v`; expect missing-module failure.
- [ ] For adjacent-pair symmetric flow drifts `d`, implement `max(d)`, `sqrt(mean(d²))`, and mean of the largest `ceil(pair_count/2)` values.
- [ ] For trace-normalized spectra `p,q`, implement:

```python
fisher_rao = 2 * torch.acos((p.sqrt() * q.sqrt()).sum(1).clamp(0.0, 1.0))
triangular = 0.5 * ((p - q).square() / (p + q).clamp_min(1e-12)).sum(1)
ks = (p - q).cumsum(1).abs().max(1).values
```

- [ ] Let `c,s,f` be centered-channel, spatial-cosine, and flow-symmetric drifts; set `u(x)=x/(1+x)` and implement cube-root product, maximum, and RMS of `u(c),u(s),u(f)` without fitted coefficients.
- [ ] Register all nine as `Ours E37 preregistered`, direction `lower`, eligibility `cross_space`.
- [ ] Run E37 and E36 regression tests; commit with `git commit -m "feat: add preregistered E37 metric family"`.

### Task 7: Add Exact Memory-Bounded E37 Evaluation

**Files:**
- Create: `src/robust_metric_lab/e37_batched.py`
- Create: `scripts/evaluate_e37_student.py`
- Create: `tests/test_e37_batched.py`

**Interfaces:**
- Produces `compute_e37_metrics_batched(...) -> Mapping[str, float]` and a three-space evaluator.

- [ ] Write batch-size 1/2/full equivalence tests plus rejection tests for zero batch size, empty samples, label mismatch, fewer than two anchors, and nonfinite output.
- [ ] Run the tests; expect missing-module failure.
- [ ] For each chunk, generate per-sample FGSM, record clean/adversarial activations, call `compute_e37_per_sample()`, sum per-sample values, and divide by the fixed total count. Never average unweighted chunk means.
- [ ] Implement `evaluate_e37_student.py --benchmark --setting --batch-size`; require the exact nine-name set and reuse `evaluate_items()` atomic resume.
- [ ] Run E37-batched and runner tests; commit with `git commit -m "feat: evaluate E37 with exact chunking"`.

### Task 8: Add Frozen Confounding Diagnostics

**Files:**
- Create: `src/robust_metric_lab/student_confound.py`
- Create: `scripts/analyze_student_confound.py`
- Create: `tests/test_student_confound.py`

**Interfaces:**
- Produces `partial_spearman(metric, target, controls) -> float` and `confound_diagnostics.csv`.

- [ ] Write a synthetic test where raw Spearman exceeds 0.9 due only to `log_params` but residual partial Spearman is below 0.1; add categorical-family and singular-control tests.
- [ ] Run tests; expect missing-module failure.
- [ ] Average-rank metric and target, regress each on an intercept plus controls via `numpy.linalg.lstsq`, and correlate residuals. Return an explicit invalid reason for fewer than four objects or constant residuals.
- [ ] Freeze controls: RB40 `log(parameter_count)` plus model-family one-hot; NB201 five operation counts; DARTS `log(parameter_count)` plus manifest operation counts.
- [ ] Output one row per metric/setting with raw and partial Spearman, count, controls, status, and reason.
- [ ] Run tests; commit with `git commit -m "feat: add benchmark confound diagnostics"`.

### Task 9: Add Safe Repair and Independent Release Verification

**Files:**
- Create: `scripts/repair_existing_metric_grid.py`
- Create: `tests/test_repair_existing_metric_grid.py`
- Create: `scripts/verify_strict_release.py`
- Create: `tests/test_verify_strict_release.py`

**Interfaces:**
- Consumes immutable source/repair CSVs, manifests, eligibility, and analysis.
- Produces a new repaired CSV, `repair_manifest.json`, and `release_verification.json`.

- [ ] Test that only failed exact keys can be replaced; successful keys, changed targets/protocol, duplicate repairs, and partial writes are rejected.
- [ ] Test release rejection after separately deleting an RB40 row, duplicating an NB201 row, inserting infinity, altering a target, changing a seed, and changing an all-setting mean.
- [ ] Implement temp-file + `fsync` + atomic rename. Record source/repair/target hashes, exact replaced keys, UTC, and command.
- [ ] Independently reload manifests and verify sets, protocol, labels, statuses, finite values, keys, eligibility, correlations, means, and ranks.
- [ ] Run repair/release/strict-ranking tests; commit with `git commit -m "feat: add auditable repair and release verification"`.

### Task 10: Deploy, Test, and Freeze Run Manifests

**Files:**
- Create: `scripts/run_strict_all_metrics.sh`
- Create: `scripts/run_e37_full.sh`
- Create: `docs/experiments/strict_all_metrics_runbook.md`

- [ ] Run full local `pytest -q`; record count and duration.
- [ ] Back up server source inside `wangjiahao`, record hashes, and upload only reviewed files.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest -q` inside `wangjiahao`; require identical success.
- [ ] Generate immutable manifests containing formulas, directions, source hashes, eligibility hash, bundle hashes, environment, protocol, object sets, and expected rows. E37 expects `9*560=5,040` main rows.
- [ ] Run four predeclared objects per space; require finite outputs, exact labels, unique keys, and recorded peak memory/runtime.
- [ ] Commit scripts/runbook with `git commit -m "ops: add strict benchmark runbooks"`.

### Task 11: Complete Existing Eligible Metrics

**Server input:** `existing_metrics_full_v3_cpu_repair/metric_values.csv`

**Server output:** new timestamped `existing_metrics_strict_v4_<UTC>/`

- [ ] Snapshot v3 SHA-256, 38,080 rows, unique keys, status distribution, and all 1,123 failed keys.
- [ ] Run importance across all eight settings with chunk size 1; require exactly 560 finite successful rows.
- [ ] Run the Kang decision. On unchanged-definition success, write exactly three repairs; on failure, preserve diagnostics and keep the three TRNAS metrics excluded.
- [ ] Build v4 atomically without altering v3. Keep both RACL proxies in per-space appendices.
- [ ] Run strict ranking, confound analysis, and independent verification. Every ranked historical metric must be 560/560.

### Task 12: Run E37 Full and Held-Out Validation

**Server output:** new timestamped `e37_full_<UTC>/`

- [ ] Run all eight settings without displaying intermediate rankings; progress may show only row/failure counts.
- [ ] Before analysis require exactly 5,040 unique, finite, successful rows: RB40 40/40, each NB201 64/64, each DARTS 100/100.
- [ ] Generate 2,000-bootstrap setting analysis and confound diagnostics with frozen directions.
- [ ] Run held-out weights exactly once after formula freeze; corrupt weights remain explicit failures, never zeros.
- [ ] Run independent release verification and preserve PASS JSON plus complete test output.

### Task 13: Build and Deliver the Final Strict Ranking

**Files:**
- Create: `docs/experiments/strict_all_metrics_final_report.md`
- Create: `docs/experiments/strict_all_metrics_problem_log.md`
- Create on desktop: `RobustMetric_全部指标严格总榜.csv`
- Create on desktop: `RobustMetric_实验问题与排除说明.md`

- [ ] Merge verified v4, E36, and E37 raw rows, never precomputed ranking tables. Abort overlapping metric names unless every overlapping raw field agrees.
- [ ] Recompute all correlations, 2,000-bootstrap intervals, eight-setting means, confound diagnostics, and ranks.
- [ ] Require independent PASS for sets, protocol, labels, uniqueness, finite values, eligibility, means, ranks, hashes, and tests.
- [ ] Report teacher-era, student-era, paper-derived, E36, and E37 metrics separately; list all excluded metrics, failures, and negative results.
- [ ] Keep provisional files for audit but mark them superseded; only filenames containing `严格总榜` are deliverable finals.
- [ ] Run `pytest -q`, `git diff --check`, and release verification; commit source/tests/scripts/docs but never weights or large generated CSVs.

## Stop Conditions

Stop formal execution and preserve the run whenever bundle IDs/targets differ, local and server tests disagree, chunk equivalence fails, a repair touches a successful key, a partial metric enters ranking, or an E37 formula changes after formal labels are viewed. Record the exact failure and command, fix under a new tested code hash, and resume only exact missing keys.
