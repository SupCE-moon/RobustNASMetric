# Activation Sign Survival: Independent Three-Space Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct an unopened, frozen three-space package and, only after the 560-key main-grid gate passes, test once whether `feature_activation_sign_survival_sum` exceeds TRNAS Φ independently.

**Architecture:** Treat holdout construction as a separate subsystem from metric computation. First audit candidate IDs and target sources without candidate scores; build and hash all weights and targets; seal 371 expected keys and analysis code; then evaluate Φ and Ours on identical keys, verify complete results and make one preregistered decision. Any opened package is retired for future candidates.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, SciPy, pytest, SHA-256, existing benchmark loaders/attack code; server work only inside `wangjiahao` and `/workspace/robust_metric_lab_nas`.

**Spec:** `docs/superpowers/specs/2026-10-27-activation-boundary-survival-design.md` revision `be2d3ea`. Depends on `docs/superpowers/plans/2026-10-27-activation-sign-survival-main-grid.md`; do not open holdout scores unless that plan's full 560-key gate passed. This plan does not itself authorize implementation or GPU experiments.

## Global Constraints

- Expected package: RB40 holdout **11** distinct loadable models; new unseen NB201 **64 architectures × five epochs = 320**; new unseen DARTS **20 architectures × two epochs = 40**; **371 total keys in eight settings**. No 10/11, missing stage, reduced space, or old seen architecture treated as confirmation.
- “Unseen” excludes all architecture IDs/signatures, checkpoints and result-bearing candidate lists used to design or compare prior Ours metrics, not merely the current main 560. Audit contamination before choosing; refuse ambiguous provenance.
- RB40 replacement checkpoint `cifar10_linf_wrn70-16_cutmix_ddpm_v2.pt` is a new overlay with SHA-256 `0769d395cbf77dc49caeab7e4c48fec5299e694937bd0b598293410a0e39a9bb`. Preserve corrupt original and recovery evidence. Recheck 11/11 loads; this is **not** a metric result.
- NB201 targets must use the official matching robustness definition and exact source. New DARTS e100 weights require a fresh full 10,000-image PGD20 target evaluation under a single frozen protocol; e0 targets require the same target definition. Old DARTS target CSVs do not transfer to newly trained models.
- Freeze selection seed, exact IDs/structural signatures, epoch, training seed/recipe, weights, target source/results, CIFAR-10 sample order, attack/target protocol, code/environment versions and hashes, expected keys, bootstrap code and decision rules **before** seeing Ours holdout scores.
- Φ and Ours are evaluated on the same 371 keys, each status ok and finite. Space-average directional Spearman differences must each be >0; eight-setting equal-weight difference must be >0; architecture-paired 3,000-draw percentile 95% interval lower >0, seed 20261027. Report all settings and intervals. Small-sample uncertainty is a non-pass, not a reason to shrink.
- Candidate attempts and errors are append-only, exact-key resume only under identical hashes; prior v3/E36/E37/strict and RACL exclusions remain untouched. If materials are missing, report `external_validation_incomplete`, not a rank or a victory.
- Once a holdout score is opened, the package is `seen`. If this candidate fails, future candidates need a fresh package or a sequential-testing scheme registered **before** reopening; never reuse this package as naïve independent confirmation.

## Review Focus

1. A historical NB201/DARTS architecture or altered structure alias must be rejected before freezing; Task 1 tests ID and signature overlap.
2. A loadable RB checkpoint alone is not evidence of a completed RB holdout; Task 2 tests the distinction and exact 11 keys.
3. Rebuilt DARTS weights with an old target, 9,999 test targets, mixed PGD protocols or missing checkpoint stage must block sealing; Task 3 tests each.
4. Looking at Ours holdout scores before package seal or after main-gate failure must block an independent claim; Task 4 tests event order.
5. Per-row bootstrap, favorable-setting omission or a zero/negative space delta must not pass; Task 5 tests all gates.

---

## File map and fixed interfaces

| File | Responsibility |
|---|---|
| `scripts/holdout_overlap_audit.py` | `audit_unseen(candidates, prior_ledgers) -> report`; check exact IDs, canonical graph signatures and hashes. |
| `scripts/build_activation_holdout_manifest.py` | `validate_holdout_package(items, provenance) -> dict`; immutable 371-key package and source/target/checkpoint hashes. |
| `scripts/verify_holdout_targets.py` | `verify_targets(manifest, target_attempts) -> dict`; exact target completeness and PGD protocol checks. |
| `scripts/run_activation_holdout.py` | Reuse Task 3 main-grid append-only evaluator for Φ and Ours, with a seal/main-gate authorization check; no score printing before opening event. |
| `scripts/verify_activation_holdout.py` | `verify_holdout(manifest, phi_attempts, ours_attempts) -> dict`; 371 matched valid pairs or incomplete audit. |
| `scripts/analyze_activation_holdout.py` | One-time preregistered rank/paired bootstrap decision and retirement ledger. |
| `tests/test_holdout_overlap_audit.py`, `tests/test_activation_holdout_manifest.py`, `tests/test_holdout_targets.py`, `tests/test_activation_holdout_gate.py`, `tests/test_activation_holdout_analysis.py` | Focused red/green tests for each boundary. |

The actual server-side training/evaluation scripts, old architecture ledgers, checkpoint bundle and robustness target authority must be discovered and hashed in Task 1; do not assume that absent local files are available, and do not fabricate targets or infer unseen status from filenames.

### Task 1: Audit candidate independence and target-source authority

**Files:** Create `scripts/holdout_overlap_audit.py`, `tests/test_holdout_overlap_audit.py` and `docs/experiments/activation_holdout_provenance_audit_20261027.md`.

- [ ] Read-only inventory in `wangjiahao` of prior main/development/diagnostic/holdout architecture lists, DARTS graph signatures, checkpoint hashes, candidate selection history, official NB201 robustness-target source, and canonical training/evaluation scripts. Also read local `docs/experiments/独立验证材料复建边界补充记录_20261027.md` and RB recovery report; label historical NB/DARTS CSVs “seen” unless proven untouched by research choices.
- [ ] Write failing tests: exact ID overlap, stage alias, graph-isomorphic DARTS signature or checkpoint hash overlap makes `audit_unseen` reject; candidate without traceable target source is `unresolved`, not “unseen”. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_holdout_overlap_audit.py -q`; expect red.
- [ ] Implement canonical ID/signature/hash comparison and a provenance report with source paths, SHA-256 and the complete exclusion set. Run focused tests green. If fewer than 64 NB or 20 DARTS uncontaminated candidates with obtainable weights/targets exist, stop and report the shortage; do not reduce counts or choose by performance.
- [ ] Commit audit and tests. Do not evaluate Ours on any candidate in this task.

Minimal overlap oracle and implementation contract:

```python
def test_prior_signature_is_not_unseen():
    candidates = [{"space": "darts", "id": "fresh-name", "signature": "edge:a-b"}]
    prior = [{"space": "darts", "id": "old-name", "signature": "edge:a-b"}]
    assert audit_unseen(candidates, prior)["unseen"] == []

def audit_unseen(candidates: list[dict], prior_ledgers: list[dict]) -> dict:
    used = {(row["space"], row["id"]) for row in prior_ledgers}
    signatures = {(row["space"], row.get("signature")) for row in prior_ledgers
                  if row.get("signature")}
    unseen = [row for row in candidates
              if (row["space"], row["id"]) not in used
              and (row["space"], row.get("signature")) not in signatures]
    return {"unseen": unseen, "excluded": [row for row in candidates if row not in unseen]}
```

Production audit must also compare checkpoint hashes and require traceable target-source IDs; an unresolved candidate is excluded. Expected red: missing module/function. Expected green: exact and signature-overlap tests pass. Targeted commit: `git add scripts/holdout_overlap_audit.py tests/test_holdout_overlap_audit.py docs/experiments/activation_holdout_provenance_audit_20261027.md`; `git commit -m "audit: exclude seen holdout architectures"`.

### Task 2: Freeze RB11 overlay and full holdout skeleton

**Files:** Create `scripts/build_activation_holdout_manifest.py` and `tests/test_activation_holdout_manifest.py`; write a new holdout-only manifest under `outputs/student_metrics/wangjiahao/activation_holdout_<UTC>/`. Never rewrite the original corrupted checkpoint.

- [ ] Red tests: exactly 11 RB IDs; one missing, duplicate, changed target or changed replacement SHA fails; a loadable checkpoint flag does not count as a Φ/Ours result. NB must have exactly 64 identical IDs across five stages, DARTS exactly 20 identical signatures across two stages, no prior-overlap item, 371 total keys. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_holdout_manifest.py -q`; expect red.
- [ ] Recheck original and replacement hashes, replacement's full forward load, and the ten other RB checkpoints in the permitted server project. Record RB target source and 11 exact keys. Construct a manifest skeleton with 64+20 selected unseen structures and fixed seeds, but leave package `unsealed` until Task 3 has all weight and target hashes.
- [ ] Implement `validate_holdout_package` as a pure pre-seal validator and a write-once canonical-JSON seal function. Run tests green and commit code; write the skeleton to a new directory only. No Ours/Φ holdout score calculation yet.

Count oracle and implementation contract:

```python
HOLDOUT_COUNTS = {"rb40_mature": 11,
                  **{f"nb201_e{x}": 64 for x in (0, 5, 10, 25, 50)},
                  **{f"darts_e{x}": 20 for x in (0, 100)}}

def test_rb_ten_of_eleven_rejected(package):
    package["objects"] = [x for x in package["objects"]
                          if not (x["setting"] == "rb40_mature" and x["model_or_arch_id"] == "missing")]
    with pytest.raises(ValueError, match="rb40_mature"):
        validate_holdout_package(package)

def validate_holdout_package(package: dict) -> dict:
    grouped = collections.Counter(row["setting"] for row in package["objects"])
    for setting, count in HOLDOUT_COUNTS.items():
        if grouped[setting] != count:
            raise ValueError(f"{setting}: expected {count}, got {grouped[setting]}")
    if sum(grouped.values()) != 371 or set(grouped) != set(HOLDOUT_COUNTS):
        raise ValueError("holdout key set mismatch")
    return package
```

Add the tested uniqueness, phase-set, source/target/weight-hash and overlap rules to this pure function before sealing. Expected red: validator absent. Expected green: entire manifest test suite. Targeted commit: `git add scripts/build_activation_holdout_manifest.py tests/test_activation_holdout_manifest.py`; `git commit -m "feat: seal complete three-space holdout manifest"`.

### Task 3: Produce all NB/DARTS weights and valid targets, then seal

**Files:** Extend `scripts/verify_holdout_targets.py`; create `tests/test_holdout_targets.py`; update the new package manifest and provenance report only through append-only records.

- [ ] Red tests: one absent NB checkpoint/official target, DARTS e0/e100 mismatch, target computed on 9,999 rather than 10,000 images, wrong attack budget/steps/restarts or reused old target for a new checkpoint fails. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_holdout_targets.py -q`; expect red.
- [ ] Implement exact source/weight/target-hash checks. Copy or regenerate only from verified permitted project sources. NB checkpoint provenance must establish the same 64 architecture identities at all five phases and matching official robustness target semantics; if impossible, stop. For DARTS, train both phases according to one fixed recipe and compute the full CIFAR-10 test-set PGD20 target under that recipe, logging all 10,000 sample IDs, attack settings and per-key failure evidence; if the benchmark's accepted target definition differs, stop and reconcile **before** opening scores.
- [ ] Use the audited GPU monitor and leave other processes alone. Failed jobs remain in attempt logs; retry only missing exact weight/target keys under unchanged recipe. Run target verifier to green and seal the package with 371 item/checkpoint/target hashes, selection seed, sample/protocol hashes, code/environment hashes and preregistered analysis hash. A changed input after seal requires a new package/version.
- [ ] Commit tests and provenance/report files. Do not open the holdout until the main 560 gate is documented PASS.

Target rejection oracle and implementation contract:

```python
def test_new_darts_weight_cannot_use_old_target():
    item = {"setting": "darts_e100", "checkpoint_sha256": "new", "target_checkpoint_sha256": "old",
            "target_sample_count": 10000, "target_attack": "pgd20"}
    with pytest.raises(ValueError, match="checkpoint"):
        verify_target_item(item)

def verify_target_item(item: dict) -> None:
    if item["checkpoint_sha256"] != item["target_checkpoint_sha256"]:
        raise ValueError("target checkpoint mismatch")
    if item["setting"].startswith("darts_"):
        if item["target_sample_count"] != 10000 or item["target_attack"] != "pgd20":
            raise ValueError("DARTS full-test PGD20 target required")
```

Production metadata must also match epsilon, step size, restarts, normalization, dataset IDs and the frozen benchmark definition. Expected red: verifier absent. Expected green: target suite. Targeted commit: `git add scripts/verify_holdout_targets.py tests/test_holdout_targets.py docs/experiments/activation_holdout_provenance_audit_20261027.md`; `git commit -m "test: bind holdout targets to exact checkpoints"`.

### Task 4: Guard one-time opening and collect matched Φ/Ours rows

**Files:** Create `scripts/run_activation_holdout.py`, `scripts/verify_activation_holdout.py`, `tests/test_activation_holdout_gate.py`.

- [ ] Red tests: opening with an unsealed package, a missing/failed main-grid gate, changed code/weights/targets/protocol, or an already-retired package fails; 370/371, duplicated, nonfinite or failed Φ/Ours pairs fail verification; failed attempts remain on retry. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_holdout_gate.py -q`; expect red.
- [ ] Implement an append-only `open_event.json` written exclusively only after reading the signed/hash-verified 560-row main-gate report. Reuse the exact frozen Ours implementation and the audited Φ implementation; run both on identical clean/attack samples and item keys. Never show interim correlations or select subsets. Preserve per-key tracebacks and resume only exact missing keys if hashes match.
- [ ] Verify 11 RB + 320 NB + 40 DARTS matched successes, targets and source hashes. If incomplete, emit `external_validation_incomplete` with exact missing/failed keys; no final claim. Commit gate/verifier tests and code.

Opening-order oracle and implementation contract:

```python
def test_open_refuses_unpassed_main_gate(sealed_manifest, failed_main_gate, tmp_path):
    with pytest.raises(ValueError, match="main-grid gate"):
        open_holdout(sealed_manifest, failed_main_gate, tmp_path / "open_event.json")

def open_holdout(manifest: dict, main_gate: dict, event_path: Path) -> None:
    if manifest.get("sealed") is not True or main_gate.get("passed") is not True:
        raise ValueError("sealed package and main-grid gate PASS required")
    if event_path.exists():
        raise ValueError("holdout already opened")
    with event_path.open("x", encoding="utf-8") as stream:
        json.dump({"manifest_sha256": manifest["sha256"],
                   "main_gate_sha256": main_gate["sha256"],
                   "opened_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
```

The real gate re-hashes the sealed payload and gate report, verifies their source/weight/protocol commitments, then refuses a retired or changed package. Expected red: opening function absent. Expected green: gate tests and exact-pair verifier tests. Targeted commit: `git add scripts/run_activation_holdout.py scripts/verify_activation_holdout.py tests/test_activation_holdout_gate.py`; `git commit -m "feat: guard one-time independent holdout opening"`.

### Task 5: Predeclared paired decision, controls and retirement

**Files:** Create `scripts/analyze_activation_holdout.py` and `tests/test_activation_holdout_analysis.py`; generate a Chinese validation report and a full-precision CSV plus three-decimal display CSV in the **new** package directory.

- [ ] Red tests: paired architecture resampling retains all stages together; 3,000 draws/seed 20261027 are reproducible; each space delta >0, eight-setting mean >0, and percentile CI lower >0 are all necessary; omit one setting or compare different targets and reject. A 371-key complete failure still produces a valid negative report and retires the package. Run `$env:PYTHONPATH="src"; python -m pytest tests/test_activation_holdout_analysis.py -q`; expect red.
- [ ] Implement directional Spearman for all eight settings and architecture-paired difference bootstrap. Report each setting, RB/NB/DARTS means, equal eight-setting mean, interval, ties/undefined correlations and L-only/parameter confound diagnostics. Do not alter Ours formula on these results.
- [ ] Write the one-time report and `holdout_retirement.json` even if Ours loses, with timestamps and all failed attempts. Run the focused tests, the 371-pair verifier, source-hash audit and an independent recalculation of decision values. If any check fails, no victory claim. Commit only code/tests/report; preserve raw outputs immutable.

Decision oracle and implementation contract:

```python
def test_one_negative_space_blocks_victory():
    deltas = {name: 0.1 for name in HOLDOUT_COUNTS}
    deltas["darts_e100"] = -0.2
    assert independent_gate(deltas, ci_low=0.01, valid_pairs=371) is False

def independent_gate(deltas: dict[str, float], *, ci_low: float, valid_pairs: int) -> bool:
    if set(deltas) != set(HOLDOUT_COUNTS):
        raise ValueError("eight holdout settings required")
    space_mean = {
        "rb40": deltas["rb40_mature"],
        "nb201": math.fsum(deltas[f"nb201_e{x}"] for x in (0, 5, 10, 25, 50)) / 5,
        "darts": math.fsum(deltas[f"darts_e{x}"] for x in (0, 100)) / 2,
    }
    return (valid_pairs == 371 and all(value > 0 for value in space_mean.values())
            and math.fsum(deltas.values()) / 8 > 0 and ci_low > 0)
```

This example's DARTS mean is negative (0.1-0.2)/2; the full tests also pin NB and RB negative cases, paired resampling by architecture, 3,000-draw seed reproducibility and missing-key refusal. Expected red: decision function absent. Expected green: focused analysis suite and independent recalculation. Targeted commit: `git add scripts/analyze_activation_holdout.py tests/test_activation_holdout_analysis.py`; `git commit -m "analysis: apply preregistered three-space holdout gate"`.

## Completion handoff

Report whether the package was never opened, opened but incomplete, complete and negative, or complete and passed. Deliver all 371-key evidence, exact excluded historical overlap, failed-attempt ledger, source/target/checkpoint hashes, per-setting/space differences and paired interval. “Ours exceeds TRNAS Φ with independent support” is allowed only for **complete and passed**; RB11 alone or 560 development rows alone never suffice.
