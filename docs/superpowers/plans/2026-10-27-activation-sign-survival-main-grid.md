# Activation Sign Survival: 560-Key Main-Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the single preregistered `feature_activation_sign_survival_sum` candidate and publish a new, independently verified 560-key strict main-grid result without changing any historical release.

**Architecture:** Reconcile the canonical server runner before coding; implement a bounded-memory Conv/Linear call tracer and pure score; freeze inputs and code; write immutable per-key attempts; verify the exact grid; only then join a new strict release and compare against TRNAS Φ. The independent confirmation has a separate plan and cannot be inferred from this development grid.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, SciPy, pytest, CSV/JSON, SHA-256; GPU work solely in `wangjiahao` at `/workspace/robust_metric_lab_nas`.

**Spec:** `docs/superpowers/specs/2026-10-27-activation-boundary-survival-design.md`, approved revision `be2d3ea` on 2026-10-27. This plan does not itself authorize implementation or experiment execution.

## Global Constraints

- One candidate, immutable formula: each actual Conv2d/Linear output invocation contributes its clean/FGSM positive-sign retention; zero is nonpositive; sum over calls, then mean over 64 CIFAR-10 test samples. Higher is better. No layer-count normalization in the ranked score.
- Exact main grid: `rb40_mature` 40; `nb201_e0/e5/e10/e25/e50` 64 each; `darts_e0/e100` 100 each: **560 distinct keys, eight settings**. No partial rank or post hoc substitution.
- Protocol: `eval`, offset 0, 64 samples, seed 20260808, batch size 1, original `make_adversarial(..., "fgsm", epsilon=8/255, pgd_steps=1)`, clamp [0,1]. Freeze sample order/content and all source, manifest and checkpoint hashes before full run.
- Historical v3/E36/E37, both strict releases, RACL exclusions and original failed attempts are read-only. New attempts and release get unique directories; failures retain exact key, traceback and provenance. Retry only missing successful keys under identical hashes.
- No GPU run before source reconciliation, tests, non-ranking smoke, frozen 560 manifest, and a resource gate. Do not disturb other users' GPU processes.
- Main-grid victory is descriptive because previous candidates informed design. It requires 560/560 valid rows, eight-setting mean directional Spearman difference versus Φ > 0, and architecture-paired 3,000-draw percentile 95% difference interval lower > 0; report every setting and three-space equal mean. External success is governed by the companion plan.

## Review Focus

1. Missing, duplicate, nonfinite or failed key must block release; Task 4 tests each and exact stage membership.
2. Reused modules, zero, shape/call mismatch and functional conv must not silently change the formula; Task 2 tests each.
3. WRN-size activations must not accumulate as 64 full tensors; Task 2 tests bounded live activation references and Task 5 measures peak memory.
4. A failed retry must preserve the failed attempt, while a second success for one key blocks canonicalization; Task 3 tests both.
5. A favorable subset or per-stage-independent bootstrap must not produce a victory; Task 6 tests paired resampling and eight-setting gating.

---

## File map and fixed interfaces

| File | Responsibility |
|---|---|
| `src/robust_metric_lab/activation_sign_survival.py` | `sign_survival(model, clean, labels, *, epsilon=8/255) -> dict` with `metric_value`, `sample_count`, `call_count`, `diagnostics`. No file I/O. |
| `scripts/activation_survival_manifest.py` | `expected_main_keys(items_by_setting)` and `write_or_verify_manifest(path, payload)`; exact keys, source/environment/checkpoint/input hashes. |
| `scripts/evaluate_activation_survival_student.py` | Reuse the canonical student item/model/sample loaders; evaluate one setting, append an immutable attempt for each exact key. |
| `scripts/verify_activation_survival.py` | `verify_attempts(manifest, attempts_dir) -> dict`; canonical 560 CSV plus machine-readable exclusion/verification evidence only after full validation. |
| `scripts/analyze_activation_survival.py` | Join verified new rows with frozen Φ and target rows; paired main-grid comparison and structural diagnostics. |
| `tests/test_activation_sign_survival.py`, `tests/test_activation_survival_manifest.py`, `tests/test_activation_survival_attempts.py`, `tests/test_activation_survival_verification.py`, `tests/test_activation_survival_analysis.py` | Own the failure cases above. |

The server's current `student_benchmark.py`, `student_runner.py`, three `evaluate_*_student.py` model loaders, strict-release joiner and GPU monitor are **not present in this committed local worktree**. Task 1 must identify and hash the authoritative server versions, their dependencies, and their public interfaces before Task 3. Never invent compatible-looking replacements or copy untracked code from another worktree without matching the server provenance. If the server cannot be inspected, stop at Task 1 and report the gap.

### Task 1: Reconcile authoritative source, frozen baseline and environment

**Files:** Create `docs/experiments/activation_sign_survival_source_audit_20261027.md` and `tests/test_activation_survival_source_audit.py`; copy only verified server-owned runner/model-loader source into the isolated worktree if the agreed implementation actually needs a local mirror. Do not edit original E37 or Φ code.

**Inputs/outputs:** Consume the verified `strict_adv_excess_release_20261024T110422Z`, its `release_verification.json`, the current server bundle manifests and source hashes. Produce a source-to-hash/interface inventory, exact paths to the existing strict merger, GPU monitor, sample loader and the 560 baseline keys.

- [ ] Read-only in `wangjiahao`: confirm strict verification PASS; 84 ranked metrics × 560 rows, exact frozen IDs/epochs/targets, Φ rows and RACL exclusion evidence. Record source paths/hashes and relevant function signatures for `student_benchmark`, `student_runner` and the three model loaders. Inspect current `git status` and preserve user changes.
- [ ] Write a failing fixture test in `tests/test_activation_survival_source_audit.py`: a changed audited byte or missing required runner path raises `ValueError("source mismatch")`. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_survival_source_audit.py -q`; expect red before the checker exists.
- [ ] Add `verify_source_inventory(inventory, project_root)` to `scripts/activation_survival_manifest.py`, comparing SHA-256 and refusing missing files. Record the Python/PyTorch/NumPy/SciPy versions and source commit or dirty-file hashes. Use the audited paths, not an assumed local `student_benchmark.py`.
- [ ] Run the focused test to green. Run baseline tests with `PYTHONPATH=src` on the reconciled copy, record any pre-existing failure separately. If canonical source differs from historical strict source, record both and choose one explicit frozen version before calculating; no silent mixing. Commit only the audit/checker files, then pause for a provenance decision if hashes cannot be explained.

Minimal failing-test and implementation contract:

```python
def test_audited_source_rejects_mutation(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"trusted")
    inventory = {"runner.py": hashlib.sha256(b"trusted").hexdigest()}
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_source_inventory(inventory, tmp_path)

def verify_source_inventory(inventory: dict[str, str], root: Path) -> None:
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source mismatch: {name}")
```

Expected red: import of `verify_source_inventory` fails. Expected green: test passes. Targeted commit: `git add scripts/activation_survival_manifest.py tests/test_activation_survival_source_audit.py docs/experiments/activation_sign_survival_source_audit_20261027.md`, then `git commit -m "test: pin activation candidate source provenance"`.

### Task 2: Implement and test the frozen score with bounded live memory

**Files:** Create `src/robust_metric_lab/activation_sign_survival.py` and `tests/test_activation_sign_survival.py`. Reuse `src/robust_metric_lab/attacks.py` unchanged.

**Inputs/outputs:** Consume one model plus `(clean, labels)`, each exactly one sample at a time. Return the fixed result dict; nested diagnostics contain per-call flip and both-zero rates and elapsed/peak-memory scalars, never feature tensors.

- [ ] Write failing tests for `>0` semantics (including `0→0` retention and `0→positive` flip), exact call sum and 64-sample arithmetic mean, no Conv/Linear, empty/nonfinite output, shape mismatch, differing call order, repeated use of one module (`path#1` and `path#2`), and a functional `F.conv2d` that is intentionally not hooked. Assert `model.training is False` after entry and hooks removed on exception.
- [ ] Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_sign_survival.py -q` and record the expected missing-module red.
- [ ] Implement a context-managed tracer registering only `nn.Conv2d`/`nn.Linear` forward hooks. On the clean pass, retain per-call **boolean sign masks** on CPU or bounded device memory, shape and ordered `(module_path, invocation)` metadata; on the adversarial pass, compare one call at a time and release its clean mask immediately. Reject non-tensor, empty or nonfinite outputs before converting to signs. Guard one-sample batch shape. Compute `q=1-flips/N` in float64 scalar accumulation; `S_i=sum(q)`, `S=mean(S_i)`. Do not retain all 64 activation tensors, divide by call count, or alter the attack.
- [ ] Make the focused tests green. Add a stress test with many repeated calls and assert the tracer's maximum number of retained masks is at most one sample's call count, with no tensors in returned diagnostics; run it green. Commit only Task 2 files.

Minimal score oracle and hot-path contract:

```python
def test_sign_rule_and_raw_call_sum():
    clean = torch.tensor([[0.0, 2.0], [-1.0, 3.0]])
    adv = torch.tensor([[0.0, -2.0], [4.0, 3.0]])
    assert call_survival(clean, adv) == pytest.approx(0.5)
    assert mean_sample_scores([[0.5, 1.0], [1.0, 1.0]]) == pytest.approx(1.75)

def call_survival(clean: torch.Tensor, adv: torch.Tensor) -> float:
    if clean.shape != adv.shape or clean.numel() == 0:
        raise ValueError("activation shape mismatch or empty")
    if not torch.isfinite(clean).all() or not torch.isfinite(adv).all():
        raise ValueError("nonfinite activation")
    return 1.0 - float((clean.gt(0) != adv.gt(0)).sum().item()) / clean.numel()

def mean_sample_scores(calls_by_sample: list[list[float]]) -> float:
    if not calls_by_sample or any(not calls for calls in calls_by_sample):
        raise ValueError("no Conv2d/Linear calls")
    return math.fsum(math.fsum(calls) for calls in calls_by_sample) / len(calls_by_sample)
```

The tracer may use these pure functions for tests but must compare masks sequentially in production, rather than holding both full clean and adversarial outputs. Expected red: missing module/functions. Expected green: all focused tests pass. Targeted commit: `git add src/robust_metric_lab/activation_sign_survival.py tests/test_activation_sign_survival.py`; `git commit -m "feat: score bounded Conv/Linear sign retention"`.

### Task 3: Write immutable per-key attempts and exact-key resume

**Files:** Create `scripts/evaluate_activation_survival_student.py`, `tests/test_activation_survival_attempts.py`; extend `scripts/activation_survival_manifest.py` only for attestation fields.

**Inputs/outputs:** The audited student loaders yield a `setting` and canonical item with `model_or_arch_id`, `checkpoint_epoch`, `checkpoint_path` and `target_value`. An attempt JSON stores `key`, `status`, `metric_value` or full error, diagnostics, checkpoint/source/manifest SHA-256 and timestamps. Filename is a new UUID or exclusive-create timestamp per attempt, never a key-reused overwrite.

- [ ] Test first: successful attempt is preserved; failure retains exception text/traceback and key; retry creates a new file; only a key with no prior valid success is scheduled; changed manifest/source hash refuses resume; a second success is *not* silently chosen. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_survival_attempts.py -q`; expect red.
- [ ] Implement runner around the actual audited `load_*_items`, `make_*_loader`, `load_cifar10_samples` interfaces. Freeze `batch_size=1` and sample protocol at CLI validation; no `--item-limit` in formal mode. Write JSON via exclusive create + flush/fsync; convert failures to `status=failed` with full traceback but return a nonzero run status. No retry of already valid keys; retain every attempt.
- [ ] Run focused tests to green. Simulate one loader failure and safe exact-key retry on a CPU tiny model; confirm first evidence remains. Commit Task 3 files.

Attempt identity and refusal oracle:

```python
def test_retry_keeps_original_failure(tmp_path):
    key = ("darts_e0", "arch-1", 0)
    first = write_attempt(tmp_path, {"key": key, "status": "failed", "error": "OOM"})
    second = write_attempt(tmp_path, {"key": key, "status": "ok", "metric_value": 1.0})
    assert first != second
    assert json.loads(first.read_text())["error"] == "OOM"
    assert len(list(tmp_path.glob("*.json"))) == 2

def write_attempt(directory: Path, payload: dict) -> Path:
    path = directory / f"{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return path
```

The real writer additionally requires frozen hashes, timestamp, traceback on failure, and exact key membership. Expected red: `write_attempt` absent. Expected green: test and failure/resume fixture pass. Targeted commit: `git add scripts/evaluate_activation_survival_student.py tests/test_activation_survival_attempts.py`; `git commit -m "feat: preserve activation candidate attempts"`.

### Task 4: Freeze and verify the exact 560-key candidate

**Files:** Complete `scripts/activation_survival_manifest.py`, create `scripts/verify_activation_survival.py`, `tests/test_activation_survival_manifest.py` and `tests/test_activation_survival_verification.py`.

**Inputs/outputs:** Freeze all eight-setting IDs/epochs, targets, checkpoint SHA-256, bundle CSV SHA-256, CIFAR-10 sample file/list/hash, protocol, code/environment hashes, strict Φ CSV hash and run directory. Verifier emits `metric_values.csv` and `release_verification.json` only after 560 valid rows; otherwise emits a failure audit without a leaderboard-ready CSV.

- [ ] Red tests: one missing RB item, NB stage, DARTS stage, duplicate ID, changed target, nonfinite target, missing/mismatched checkpoint hash or changed sample hash all fail the manifest. A run with 559 successes, a NaN, `status!=ok`, wrong provenance, unknown key or two successes for one key fails verification. All 560 exact successes pass.
- [ ] Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_survival_manifest.py tests/test_activation_survival_verification.py -q`; expect red.
- [ ] Implement `expected_main_keys` with fixed counts {`rb40_mature`:40, each NB:64, each DARTS:100}, same architecture sets across each space's stages, canonical IDs, no duplicates. Implement `write_or_verify_manifest` with exclusive create and canonical JSON hash as in existing `scripts/e37_diagnostic_manifest.py`. Implement verifier independently of runner, re-hash source/inputs and read JSON with duplicate-key/nonfinite rejection. Failed attempts remain documented but only one success per expected key is canonical.
- [ ] Run focused tests green and independently spot-check keys/targets against the existing strict release. Commit Task 4 files. Stop if a frozen item is absent; do not remove it.

Exact-count and duplicate oracle:

```python
COUNTS = {"rb40_mature": 40, **{f"nb201_e{x}": 64 for x in (0, 5, 10, 25, 50)},
          **{f"darts_e{x}": 100 for x in (0, 100)}}

def test_missing_success_blocks_canonical_csv(valid_attempts, frozen_manifest):
    attempts = valid_attempts[:-1]
    with pytest.raises(ValueError, match="missing"):
        verify_records(frozen_manifest, attempts)

def verify_records(manifest: dict, attempts: list[dict]) -> list[dict]:
    expected = {(row["setting"], row["model_or_arch_id"], row["checkpoint_epoch"])
                for row in manifest["objects"]}
    success: dict[tuple[str, str, int], dict] = {}
    for row in attempts:
        if row["status"] != "ok":
            continue
        key = (row["setting"], row["model_or_arch_id"], row["checkpoint_epoch"])
        if key not in expected or key in success or not math.isfinite(float(row["metric_value"])):
            raise ValueError(f"duplicate, unknown or nonfinite success: {key}")
        success[key] = row
    if set(success) != expected or len(expected) != 560:
        raise ValueError("missing success keys")
    return [success[key] for key in sorted(success)]
```

The production verifier additionally validates every `status=ok` target/provenance/hash and JSON duplicate fields before this pure core. Expected red: missing verifier function. Expected green: all manifest and verifier fixtures. Targeted commit: `git add scripts/activation_survival_manifest.py scripts/verify_activation_survival.py tests/test_activation_survival_manifest.py tests/test_activation_survival_verification.py`; `git commit -m "feat: gate exact 560 activation keys"`.

### Task 5: Non-ranking smoke, resource gate and full server run

**Files:** Create a new run directory under `outputs/student_metrics/wangjiahao/activation_sign_survival_<UTC>/`; no historical edits. Add a runbook `docs/experiments/activation_sign_survival_run_20261027.md`.

- [ ] Reconfirm source and manifest hashes in `wangjiahao`. On one tiny model per space, check only clean/adv call identity, 64-sample completion, finite score, runtime and peak GPU memory; do **not** inspect correlation/rank or change formula. A failed smoke is append-only evidence and blocks the main run until a code-only defect is fixed in a **new** source hash/version. Keep old attempt bytes.
- [ ] Use the project's audited GPU-memory monitor and threshold. If capacity is insufficient, wait without touching other processes; record the wait. Before launch, record `nvidia-smi` free/used and active processes read-only. No forced kills.
- [ ] Run eight settings on the server from the frozen manifest, solely in the allowed container/project. Record command lines, logs, process status and every attempt. After each interruption, independently compute the exact missing-success key set and resume only those keys with unchanged hashes. Never report interim ranks.
- [ ] Run Task 4 verifier and require exactly 560 unique finite `status=ok` rows. If not, publish a failure report and stop before ranking. Commit runbook updates only; raw data stays immutable in the new output directory.

### Task 6: New strict join and paired Φ comparison

**Files:** Create `scripts/analyze_activation_survival.py` and `tests/test_activation_survival_analysis.py`; create a **new** strict release directory only after Task 5 passes. Do not modify the existing strict joiner or registry unless its audited interface requires the new immutable metric metadata; any necessary change receives a separate focused test.

- [ ] Red tests: direction-adjusted Spearman per setting; exact eight-setting arithmetic mean; architecture-paired bootstrap resamples one architecture across all its stages (NB and DARTS), not individual rows; fixed 3,000 draws/seed 20261027; negative or zero space delta blocks independent victory; main CI low ≤0 blocks main victory; duplicate/missing Φ or Ours rows block all analysis. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_survival_analysis.py -q`; expect red.
- [ ] Implement read-only comparison from the verified candidate CSV and pinned Φ/target rows. Emit full-precision machine CSV and a three-decimal display CSV, per-setting deltas, three-space equal mean, 3,000-draw paired difference CI, and the fixed L-only, S/L, zero-rate, parameter-control and same-architecture stage-stability diagnostics. Label this grid `development / repeated-candidate`.
- [ ] Run the focused tests green. Invoke the audited strict merge in a new destination, then run its own verifier plus an independent 560-key check for **every** admitted metric. Preserve the two RACL exclusions and their raw evidence. Hash old release directories before/after or otherwise verify no historical file changed.
- [ ] Record Φ rank, candidate rank, all eight differences, CI, structural-confound findings, failed-attempt evidence and prior-candidate selection risk in a Chinese report. The main gate is **passed** only if all preregistered conditions pass; otherwise report the failed condition and do not open the new independent package for this candidate. Commit analysis/test/report files, not historical outputs.

Decision and resampling oracle:

```python
def test_main_gate_refuses_favorable_subset():
    deltas = {"rb40_mature": 0.1, "nb201_e0": 0.1, "nb201_e5": 0.1,
              "nb201_e10": 0.1, "nb201_e25": 0.1, "nb201_e50": 0.1,
              "darts_e0": 0.1}
    with pytest.raises(ValueError, match="eight settings"):
        main_gate(deltas, ci_low=0.05, valid_rows=560)

def main_gate(deltas: dict[str, float], *, ci_low: float, valid_rows: int) -> bool:
    if set(deltas) != set(COUNTS):
        raise ValueError("eight settings required")
    return valid_rows == 560 and math.fsum(deltas.values()) / 8 > 0 and ci_low > 0
```

For the bootstrap, draw RB IDs once, NB IDs once and DARTS IDs once **per draw**, then attach all phases of each sampled NB/DARTS ID before recomputing all eight Spearman differences; do not sample rows independently. Expected red: `main_gate` absent or accepts seven settings. Expected green: full analysis suite. Targeted commit: `git add scripts/analyze_activation_survival.py tests/test_activation_survival_analysis.py docs/experiments/activation_sign_survival_run_20261027.md`; `git commit -m "analysis: compare complete activation candidate with TRNAS phi"`.

## Completion handoff

Deliver exact source/run/release paths, 560-row audit, full-precision and three-decimal tables, old-release immutability check, failed-attempt ledger and main-gate result. Even a main-gate pass is **not** a confirmed defeat of Φ; proceed to the separately reviewed independent-validation plan only when its unopened package has been frozen.
