# Adversarial Excess Feature Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one frozen cross-space metric, evaluate all 560 required model-setting keys on the server, and admit it to a new strict release only if every key succeeds. This plan is for user review; no implementation or GPU run has begun.

**Architecture:** A pure per-sample metric consumes clean, FGSM, and deterministic-random features. A memory-bounded evaluator and existing student runner produce immutable raw evidence; a separate release builder merges it with the verified 47,600-row TRNAS strict release and applies the same eligibility gate. Server deployment is limited to the `wangjiahao` container and `/workspace/robust_metric_lab_nas`.

**Tech Stack:** Python 3, PyTorch, NumPy PCG64, pytest, pandas, existing RobustMetric student runner/strict ranking, Bash, CUDA only for model evaluation.

**Spec:** `docs/superpowers/specs/2026-10-24-adversarial-excess-feature-drift-design.md`

## Global Constraints

- Exactly one new metric: `feature_adv_excess_centered_unit_l2`, direction `lower`, eligibility `cross_space`. No fitted weight, threshold, or sign.
- Main protocol: `sample_offset=0`, `num_samples=64`, `seed=20260808`, FGSM, `epsilon=8/255`, E37 `all_leaf` anchors with `max_layers=4`.
- Random control: for global sample index `i=0,…,63` use `numpy.random.Generator(numpy.random.PCG64(20261024+i))` and `integers(0,2,size=(C,H,W),dtype=int8)`, mapping bits to ±1 before the existing `clamp_linf`.
- Full grid: RB40 40; NB201 64 at each of e0/e5/e10/e25/e50; DARTS 100 at e0/e100; total 560. No smaller formal ranking after OOM or missing checkpoint.
- Older v3/E36/E37 releases and the 2026-10-24 TRNAS release remain read-only. Do not touch other containers/projects or interrupt other users' processes.
- A formal score needs 560 unique, finite `status=ok` raw keys and eight valid correlation cells. The 11-model RB40 holdout needs 11/11 materials before any external-validation claim.
- Presentation CSV/XLSX rounds score columns to three decimal places; ranking and statistical tests use full precision.
- Current local checkout does not contain server `e37_metrics.py` and `e37_batched.py`. Never deploy an old local registry or evaluator over the current server version without exact source/hash review.

## File and Interface Map

- Create `src/robust_metric_lab/adv_excess_metric.py`: deterministic random control and pure per-sample formula. It must not load models, targets, or manifests.
- Create `src/robust_metric_lab/adv_excess_batched.py`: FGSM, three feature recordings, exact sample-weighted aggregation, and per-model diagnostics.
- Create `scripts/evaluate_adv_excess_student.py`: eight-setting CLI using `evaluate_items`; one raw row per item, plus separate diagnostic evidence.
- Modify only the deployed `src/robust_metric_lab/metric_registry.py` and `configs/metric_eligibility.json` to append the new name. For a new release, extend a copied version of the repaired TRNAS release's eligibility JSON so its three TRNAS repairs remain eligible.
- Create `scripts/run_adv_excess_full.sh`: immutable run manifest, GPU gate, smoke/full modes, exact row checks.
- Create `scripts/merge_adv_excess_repair.py`: failed-key-only replacement from a separate repair sidecar, never replacement of successful rows.
- Create `scripts/build_adv_excess_strict_release.py`: merge verified old raw with complete new raw and call existing strict ranking/verifier.
- Create `scripts/analyze_adv_excess_release.py`: paired-bootstrap comparisons, stage stability, structure controls, cost summary, and teacher-facing three-decimal table.
- Create focused `tests/test_adv_excess_metric.py`, `tests/test_adv_excess_batched.py`, `tests/test_adv_excess_runner.py`, `tests/test_adv_excess_release.py`, and `tests/test_adv_excess_analysis.py`.

## Review Focus

1. Images near 0/1: FGSM and random controls can have different post-clipping norms; record both and test that the input constraint holds without silently rescaling either.
2. Chunk sizes 1 and 64: a global sample index must generate the same random sign and exact 64-sample mean despite differing chunk boundaries.
3. Constant/zero activations and 1×1 maps: output must stay finite without silently dropping anchors or changing the frozen formula.
4. A failed row followed by a repair: keep original failure evidence; reject attempts to replace an already successful key or change its target/protocol fields.
5. Server configuration drift: base eligibility still marks TRNAS as an implementation error, whereas the repaired release's eligibility admits it; the new release must preserve the latter without editing previous releases.

---

### Task 1: Verify the server baseline and isolate the intended changes

**Files:**
- Read: `/workspace/robust_metric_lab_nas/src/robust_metric_lab/e37_metrics.py`, `e37_batched.py`, `student_runner.py`, `metric_registry.py`, `configs/metric_eligibility.json`.
- Read: `/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/strict_trnas_direct_ode_release_20261024T000000Z/`.
- Create: `docs/experiments/对抗超额特征漂移_实施基线记录_20261024.md`.

**Interfaces:**
- Consumes: approved spec and previous strict release.
- Produces: byte hashes and an exact server-file inventory used by Tasks 2–7; no application code.

- [ ] **Step 1: Recheck the untouched strict release**

Run inside `wangjiahao`:
```bash
cd /workspace/robust_metric_lab_nas
sha256sum outputs/student_metrics/wangjiahao/strict_trnas_direct_ode_release_20261024T000000Z/ranking/strict_ranking.csv
cat outputs/student_metrics/wangjiahao/strict_trnas_direct_ode_release_20261024T000000Z/release_verification.json
```
Expected: ranking SHA-256 `e0d6a8a293fce61a98b202f40112b1545143f4b332714e5f5a8a17588a6640e1`; PASS, 47,600 unique raw keys, 83 ranked, 2 excluded.

- [ ] **Step 2: Check server source and the baseline test command**

```bash
cd /workspace/robust_metric_lab_nas
sha256sum src/robust_metric_lab/e37_metrics.py src/robust_metric_lab/e37_batched.py src/robust_metric_lab/student_runner.py src/robust_metric_lab/metric_registry.py configs/metric_eligibility.json
.venv/bin/python -m pytest -q tests/test_e37_metrics.py tests/test_e37_batched.py
```
Expected: each hash recorded verbatim and 12 E37 tests pass; investigate a mismatch before any deployment.

- [ ] **Step 3: Write the baseline record**

Use `apply_patch` for the local Markdown file. Include the actual five hashes, release verification JSON, local dirty-file list, server-only modules, and deployment boundary. Expected: every source file that might change later has a recorded before-hash.

### Task 2: Implement the deterministic control and pure metric

**Files:**
- Create: `src/robust_metric_lab/adv_excess_metric.py`.
- Test: `tests/test_adv_excess_metric.py`.

**Interfaces:**
- Consumes: `clamp_linf(x, clean, epsilon)` from `robust_metric_lab.attacks`.
- Produces: `fixed_random_linf(clean: Tensor, epsilon: float, global_indices: Sequence[int]) -> Tensor` and `adv_excess_per_sample(clean_features, adversarial_features, random_features, indices) -> tuple[Tensor, Tensor, Tensor]`; outputs are main excess, adversarial drift, random drift, each shape `[batch]`.

- [ ] **Step 1: Write failing pure-function tests**

```python
def test_fixed_random_is_chunk_invariant():
    clean = torch.full((4, 3, 2, 2), 0.5)
    full = fixed_random_linf(clean, 8 / 255, [0, 1, 2, 3])
    parts = torch.cat([fixed_random_linf(clean[:2], 8 / 255, [0, 1]),
                       fixed_random_linf(clean[2:], 8 / 255, [2, 3])])
    assert torch.equal(full, parts)

def test_zero_identical_and_positive_rescale():
    features = [torch.tensor([[[[0.0, 1.0]]], [[[1.0, 0.0]]]])]
    attacked = [torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])]
    random_features = [torch.tensor([[[[0.5, 1.0]]], [[[1.0, 0.5]]]])]
    score, adv, random = adv_excess_per_sample(features, features, features, [0])
    assert torch.equal(score, torch.zeros_like(score))
    reference_score = adv_excess_per_sample(features, attacked, random_features, [0])[0]
    changed = adv_excess_per_sample([f * 7 + 3 for f in features],
                                    [f * 7 + 3 for f in attacked],
                                    [f * 7 + 3 for f in random_features], [0])[0]
    assert torch.allclose(changed, reference_score, atol=1e-6)
```
Add independent tests for 1×1 zero maps, odd/even layer medians, nonfinite inputs, duplicate/out-of-range anchors, and post-clipping `L∞≤epsilon`.

- [ ] **Step 2: Run the tests red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_metric.py`.
Expected: import/function-missing failures from the new module, not fixture or environment errors.

- [ ] **Step 3: Implement only the frozen formula**

```python
def _unit_centered(feature):
    flat = feature.double().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norm = centered.norm(p=2, dim=1, keepdim=True)
    return torch.where(norm > 1e-12, centered / norm.clamp_min(1e-12), torch.zeros_like(centered))

def _distance(left, right):
    return 0.5 * (_unit_centered(left) - _unit_centered(right)).norm(p=2, dim=1)
```
Generate each NumPy PCG64 sign from its global index, then use `clamp_linf`. For an even number of anchors, sort per-sample excess values and average the two central values; do not use `torch.median`'s lower-middle convention.

- [ ] **Step 4: Run green and inspect this isolated unit**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_metric.py tests/test_e37_metrics.py`.
Expected: all tests pass. Inspect only the new module/test diff; do not stage or modify existing user changes.

### Task 3: Add exact memory-bounded evaluation

**Files:**
- Create: `src/robust_metric_lab/adv_excess_batched.py`.
- Test: `tests/test_adv_excess_batched.py`.

**Interfaces:**
- Consumes: Task 2 functions, `ActivationRecorder`, `_anchor_indices`, `make_adversarial`.
- Produces: `compute_adv_excess_batched(model, clean, labels, *, epsilon, batch_size, max_layers=4, sample_offset=0) -> AdvExcessResult`, where the frozen dataclass contains `value: float` and finite diagnostic summaries for adversarial/random drift and actual perturbation norms.

- [ ] **Step 1: Write failing batch-equivalence and fault tests**

```python
def test_batch_one_and_full_agree_for_the_same_64_samples():
    one = compute_adv_excess_batched(model, clean, labels, epsilon=8/255, batch_size=1)
    full = compute_adv_excess_batched(model, clean, labels, epsilon=8/255, batch_size=64)
    assert abs(one.value - full.value) < 1e-6
    assert one.diagnostics["samples"] == full.diagnostics["samples"] == 64
```
Also test empty/mismatched inputs, nonpositive batch size, anchor drift across chunks, and one corrupted nonfinite feature. Use a deterministic toy model with hooked leaf layers; no robustness labels in metric code.

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_batched.py`.
Expected: missing evaluator, then no remaining collection errors.

- [ ] **Step 3: Implement per-sample sums, not batch-mean averages**

For each chunk `[start:stop]`, call the deployed E37 FGSM routine on `clean[start:stop]` and `labels[start:stop]`; call `fixed_random_linf` using global indices `sample_offset + start` through `sample_offset + stop - 1`; record clean, FGSM, and random activations with the same `all_leaf` hooks and one frozen anchor-index vector. Pass these three feature lists and the anchor indices to `adv_excess_per_sample`. Add `sample_scores.sum()` to a running sum and divide only once by `total_samples` after the last chunk. Keep the actual attack and feature recording identical to deployed E37 except for the additional random forward. Calculate norm summaries from the actual post-clipping tensors. A feature or anchor mismatch must fail the model row, never silently truncate lists.

- [ ] **Step 4: Run green and inspect**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_metric.py tests/test_adv_excess_batched.py tests/test_e37_batched.py`.
Expected: all pass. Inspect only Task 3 files.

### Task 4: Register one metric and add the three-space runner

**Files:**
- Create: `scripts/evaluate_adv_excess_student.py`.
- Modify: `src/robust_metric_lab/metric_registry.py` (append one `MetricSpec`).
- Modify: `configs/metric_eligibility.json` (append one `cross_space` record; do not change historical TRNAS entries).
- Test: `tests/test_adv_excess_runner.py`.

**Interfaces:**
- Consumes: Task 3 `AdvExcessResult` and existing `evaluate_items`.
- Produces: CLI arguments matching `evaluate_e37_student.py` plus `--batch-size` and one raw row per model-setting key; diagnostics saved by exact key into the new run directory.

- [ ] **Step 1: Write failing registration and CLI tests**

```python
def test_runner_rejects_changed_sample_protocol():
    with pytest.raises(SystemExit):
        main(["--benchmark", "nb201", "--epoch", "0", "--sample-offset", "1",
              "--bundle-root", str(bundle), "--data-root", str(data),
              "--output-dir", str(output), "--run-id", "adv_excess_smoke"])

def test_registry_direction_and_eligibility():
    assert METRIC_BY_NAME["feature_adv_excess_centered_unit_l2"].direction == "lower"
    assert eligibility["feature_adv_excess_centered_unit_l2"] == "cross_space"
```
Define `bundle`, `data`, and `output` as `tmp_path`-based pytest fixtures in this test module; load `eligibility` from the deployed JSON. Add a one-item fake-runner test proving the raw row carries the frozen setting, target, seed, sample count, direction, and a finite metric value. Test that an existing key is skipped without recomputing it.

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_runner.py`.
Expected: missing metric/runner failures.

- [ ] **Step 3: Implement the minimal runner**

```python
def compute(metric, model, sample_value, item):
    if metric != "feature_adv_excess_centered_unit_l2":
        raise ValueError("unexpected metric")
    clean, labels = sample_value
    result = compute_adv_excess_batched(model, clean, labels, epsilon=EPSILON,
                                        batch_size=args.batch_size,
                                        sample_offset=args.sample_offset)
    write_diagnostics_atomically(item, result.diagnostics)
    return result.value
```
Use the existing manifest loaders and `evaluate_items`. Diagnostics go to one file per exact item key using a create-only/atomic operation; the raw row remains the only ranking input. Follow deployed server source, not the older local snapshot.

- [ ] **Step 4: Run green and inspect**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_runner.py tests/test_adv_excess_metric.py tests/test_adv_excess_batched.py tests/test_e37_metrics.py tests/test_e37_batched.py`.
Expected: all pass. Inspect only Task 4 files; confirm old E37 outputs are unchanged.

### Task 5: Freeze the run manifest and GPU-gated smoke/full launcher

**Files:**
- Create: `scripts/run_adv_excess_full.sh`.
- Test: `tests/test_adv_excess_runner.py` (manifest/run-mode cases).

**Interfaces:**
- Consumes: Task 4 CLI; frozen bundle manifests and server source hashes.
- Produces: new `adv_excess_smoke_<UTC>` or `adv_excess_full_<UTC>` directory with create-only `run_manifest.json`, `STATUS.env`, logs, one-metric raw CSV, and per-item diagnostics.

- [ ] **Step 1: Write failing launcher tests**

Test that smoke mode expects exactly 32 rows (4 objects × 8 settings), full mode exactly 560, and that a pre-existing run directory with a different manifest hash aborts before GPU use. Test `bash -n scripts/run_adv_excess_full.sh`.

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_runner.py` and `bash -n scripts/run_adv_excess_full.sh`.
Expected: the new manifest/mode tests fail before the launcher exists.

- [ ] **Step 3: Implement a create-only manifest and wait gate**

```bash
set -euo pipefail
project=/workspace/robust_metric_lab_nas
mode="$1"
case "$mode" in smoke) expected_rows=32 ;; full) expected_rows=560 ;; *) exit 2 ;; esac
# Write manifest once with Python open(path, "x"), containing formula/spec/source,
# bundle, environment, object-list and sample-protocol hashes before evaluating.
# Reuse the existing E37 GPU gate: <=1000 MiB observed on three checks 60 s apart.
# Run rb40, five nb201 epochs, two darts epochs with batch_size=1.
```
If GPU is occupied, wait without killing or pausing any process. If the user's permitted container lacks enough free memory for an RB40 model even at batch 1, record OOM and investigate an exact CPU/streamed implementation; never reduce 40 to fewer models.

- [ ] **Step 4: Run green and smoke**

Run server `.venv/bin/python -m pytest -q tests/`, `bash -n scripts/run_adv_excess_full.sh`, then `bash scripts/run_adv_excess_full.sh smoke` under the GPU gate.
Expected: test suite green, 32 unique finite `ok` smoke rows and a manifest/hash audit. Smoke uses targets only to preserve raw protocol, not to choose sign, threshold, or formula. Stop and preserve evidence on any failure.

### Task 6: Make failure repair and strict-release construction safe

**Files:**
- Create: `scripts/merge_adv_excess_repair.py`, `scripts/build_adv_excess_strict_release.py`.
- Test: `tests/test_adv_excess_release.py`.

**Interfaces:**
- Consumes: old repaired TRNAS raw (47,600), new candidate raw (560), old repaired eligibility overlay, Task 5 frozen manifests.
- Produces: a create-only release with 48,160 unique raw rows and, only if all new keys pass, 84 ranked metrics and the original 2 excluded RACL metrics.

- [ ] **Step 1: Write failing repair/eligibility tests**

```python
def test_repair_replaces_only_identical_failed_key():
    merged = merge_failed_rows(original_failed_and_ok, one_repair)
    assert merged[failed_key]["status"] == "ok"
    assert merged[successful_key] == original_failed_and_ok[successful_key]

def test_partial_new_metric_cannot_enter_ranking():
    with pytest.raises(IncompleteGridError):
        build_release(old_raw, new_raw_with_559_ok_rows, old_eligibility)
```
Define `original_failed_and_ok`, `one_repair`, `failed_key`, `successful_key`, `old_raw`, `new_raw_with_559_ok_rows`, and `old_eligibility` as fixed in-memory fixtures in this test module; construct their 560-key grid from the frozen manifest rather than a row-count-only stub. Also reject changed target, checkpoint, setting, sample protocol, duplicate repair, unknown metric, nonfinite success, and a modified historical TRNAS eligibility row.

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_release.py`.
Expected: missing safe-repair/release functions.

- [ ] **Step 3: Implement exact repair and raw-only release**

```python
assert len(old_raw) == 47600 and old_raw[UNIQUE_KEY_COLUMNS].drop_duplicates().shape[0] == 47600
assert len(new_raw) == 560 and set(new_raw.metric_name) == {"feature_adv_excess_centered_unit_l2"}
assert new_raw.status.eq("ok").all() and np.isfinite(new_raw.metric_value).all()
assert set(old_raw.metric_name).isdisjoint(new_raw.metric_name)
merged = pd.concat([old_raw, new_raw], ignore_index=True)
assert len(merged) == 48160
```
Create the new eligibility file from the repaired TRNAS release overlay plus exactly one new record; compare every existing record byte-for-byte after canonical JSON parsing. Call existing `build_strict_metric_ranking.py` and `verify_strict_release.py` with the frozen three manifests. Do not copy a precomputed ranking.

- [ ] **Step 4: Run green and inspect**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_release.py tests/test_adv_excess_runner.py`.
Expected: all pass, and the synthetic 559/560 case remains excluded. Inspect only Task 6 files.

### Task 7: Run 560 server keys and publish only a verified result

**Files:**
- Create only within `outputs/student_metrics/wangjiahao/adv_excess_full_<UTC>/` and `outputs/student_metrics/wangjiahao/strict_adv_excess_release_<UTC>/`.
- Do not modify old strict release directories.

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces: complete raw evidence, verification JSON, ranking/exclusion CSVs, logs, and hashes; or an explicitly incomplete failed run with no formal new rank.

- [ ] **Step 1: Freeze and audit inputs before GPU work**

Run full server tests with `.venv/bin/python -m pytest -q tests/`; record source, spec, eligibility, manifest, checkpoint list, and bundle SHA-256 in a create-only manifest. Expected: no altered historical source hash and no test failures; if pre-existing failures occur, record exact names and resolve before claiming release readiness.

- [ ] **Step 2: Wait, then execute the full run**

Run: `bash scripts/run_adv_excess_full.sh full` with a unique run ID. Expected: 560 raw rows, one metric name, exact setting counts `40/64/64/64/64/64/100/100`, no duplicates, all finite `ok`. Poll the status/log at sensible intervals; never interfere with another process.

- [ ] **Step 3: Preserve and resolve any failed keys**

If a row fails, preserve the first raw CSV and logs unchanged. Diagnose the exact failure. For an unchanged formula/protocol, compute only the failed key into a separate repair file and run Task 6's verified merge; successful rows are immutable. A formula/source change requires a new versioned run and manifest. Expected: either all 560 verified or no new formal ranking.

- [ ] **Step 4: Construct the strict release from raw rows**

Run Task 6 release builder into a new path, then the existing strict verifier. Expected: PASS, 48,160 unique raw keys, 84 ranked, 2 RACL excluded; zero change to old release SHA-256. Do not write “complete” in `STATUS.env` before the verifier returns PASS.

### Task 8: Analyze stage stability, independent evidence, and delivery

**Files:**
- Create: `scripts/analyze_adv_excess_release.py`.
- Test: `tests/test_adv_excess_analysis.py`.
- Create: `docs/experiments/对抗超额特征漂移_严格跨空间结果_20261024.md`.
- Create: new three-decimal teacher-facing CSV/XLSX inside the new output directory.

**Interfaces:**
- Consumes: Task 7 verified release and its full-precision raw rows.
- Produces: eight-setting and three-space-balanced results, paired CIs, e0/e50 and e0/e100 stage stability, structure-control diagnostics, cost audit, and a three-decimal presentation table without altering ranking precision.

- [ ] **Step 1: Write failing statistical tests**

```python
def test_bootstrap_resamples_the_same_nb201_architectures_across_five_epochs():
    toy_five_epoch_rows = make_five_epoch_fixture(architectures=("a", "b", "c"))
    result = paired_bootstrap(toy_five_epoch_rows, draws=100, seed=20261024)
    assert result["shared_epoch_indices"] is True

def test_teacher_table_rounds_only_display_values():
    full_precision_ranking = make_ranking_fixture(mean=0.123456, rank=1)
    table = make_teacher_table(full_precision_ranking)
    assert table.loc[0, "all_setting_mean"] == "0.123"
    assert table.loc[0, "strict_rank"] == 1
```
Define both fixture factories in the same test module with explicit in-memory rows: three architecture IDs repeated at all five NB201 epochs, and one score row with full-precision mean `0.123456`. Add a test that stage target IDs/order align, and one that missing 11th holdout checkpoint suppresses any external-validity conclusion.

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest -q tests/test_adv_excess_analysis.py`.
Expected: missing analysis functions.

- [ ] **Step 3: Implement predeclared comparisons**

Use the same paired architecture bootstrap logic as `scripts/analyze_trnas_ours_comparison.py` (joint across epochs within each space), seed `20261024`, with full-precision inputs. Compare the new metric against TRNAS Φ/R-Score and E37 max/RMS. Report both prespecified gates from the spec, all negative results, cost/diagnostics, and descriptive structure controls. Do not fit or reorient the score.

- [ ] **Step 4: Verify, render, and hand off**

Run focused tests and full server `.venv/bin/python -m pytest -q tests/`. Verify every score column in the CSV has exactly three digits after the decimal, while full-precision strict ranking and bootstrap inputs remain unchanged. Render/inspect the XLSX using the spreadsheet artifact workflow. Expected: evidence-backed Chinese report, problem record, direct teacher CSV, and original RACL/TRNAS provenance retained.

- [ ] **Step 5: Gate the external holdout separately**

Check all 11 frozen RB40 holdout IDs against their checkpoint files, labels, and sample hashes. If and only if 11/11 exist, run the same frozen formula on those 11 models into a separate holdout raw CSV and verify 11 unique finite `ok` rows before computing any holdout correlation. If fewer than 11 exist, save the missing-ID/material audit and label external validation pending; never publish a 10/11 final result or fold holdout rows into the 560-key development ranking.

## Completion Contract

No new rank is claimed until the strict verifier returns PASS and every one of the 560 new rows is unique, finite, and `ok`. If GPU, model material, numerical validity, or independent holdout is incomplete, preserve the evidence and report that exact limit. The eight-setting development result is never described as independent confirmation of superiority over TRNAS.
