# DARTS Independent Holdout Weights and Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 20 unseen DARTS cells with e0/e100 weights and 20 full-test PGD-20 targets that can join the 371-key independent validation package, while retaining every historical exclusion and failure.

**Architecture:** Keep architecture provenance, weight training, and target evaluation in separate append-only artifacts. Training never computes metric scores or targets; target evaluation uses only frozen e100 weights. Seal and score the three-space package only after all identity and source checks pass.

**Tech Stack:** Python 3.10, PyTorch, torchvision, pytest, existing `robust_metric_lab.darts_space` and `scripts/run_darts_metric_trajectory.py` functions.

**Spec:** `docs/superpowers/specs/2026-10-27-activation-boundary-survival-design.md`, especially the independent validation and repeated-selection gates.

## Global Constraints

- Work only inside `wangjiahao:/workspace/robust_metric_lab_nas` and the specified local RobustMetric worktree. Never edit v3, E36, E37, strict releases, or another container.
- The provisional `darts_selection_unsealed.json` has 20 ID/signature pairs, but E32 and other ID-only sources remain unresolved. Do not call it an independent set or launch formal holdout scoring until the prior-structure ledger is complete.
- Keep the main candidate's complete 560-key strict experiment ahead of holdout GPU work. Respect the existing GPU monitor's three 60-second checks at <=1,000 MiB other-compute usage; do not interfere with PID 2108226 or its successors.
- All DARTS targets must use their own e100 weights and all ordered 10,000 CIFAR-10 test images. Both e0 and e100 rows for one cell reference the same e100 target evidence.
- No partial 20-cell DARTS claim, 371-key external claim, or victory over TRNAS Φ. All source files, failed attempts, and exclusions remain available by hash.

## Review Focus

- E32 provenance remains unresolved: a candidate cannot be sealed merely because its new ID or known signature differs from the 120 mapped old structures.
- The original DARTS trajectory calculates metrics at e0/e50/e100 and reseeds the global RNG; omitting those calls changes later stochastic training. Freeze a new training-only RNG schedule explicitly and report that protocol difference.
- A training crash after e0 must leave its attempt intact. Restart that architecture in a new attempt; do not silently resume from a checkpoint without saved RNG state.
- A complete receipt with missing or changed e0/e100 bytes must fail verification, never be skipped as complete.
- A full-test target file with 9,999 images, altered order, forged accuracy, wrong checkpoint hash, or altered attack settings must fail package verification.

---

### Task 1: Resolve and freeze prior DARTS identities

**Files:** `scripts/holdout_exposure_inventory.py`, `scripts/holdout_overlap_audit.py`, `tests/test_holdout_exposure_inventory.py`, `docs/experiments/activation_holdout_provenance_audit_20261027.md`.

**Interfaces:** Consume the source-hashed 120-signature/453 ID-only inventory and E32 raw provenance if obtainable. Produce a sealed prior ledger mapping every DARTS identity-bearing source to a canonical signature set or an explicit unresolved finding.

- [ ] **Step 1:** Add a failing test: an E32-like ID-only source cannot be marked resolved by matching `arch_id` or historical metric values alone.
- [ ] **Step 2:** Run the targeted test and confirm the intended failure.
- [ ] **Step 3:** Trace each source class to a frozen genotype manifest/checkpoint mapping, recording source SHA-256. Preserve E32 as unresolved until direct structure evidence exists; do not fabricate a mapping.
- [ ] **Step 4:** Run tests and independently rehash the resulting ledger. If E32 remains unresolved, stop before Task 2's seal and report the precise missing evidence.
- [ ] **Step 5:** Commit the audit and leave `darts_selection_unsealed.json` unchanged.

### Task 2: Seal the 20 architecture identities without opening outcomes

**Files:** Create `scripts/seal_activation_holdout_darts_selection.py`; test in `tests/test_seal_activation_holdout_darts_selection.py`.

**Interfaces:** Consume the exact provisional selection SHA-256 `f3e272c91dbe0ed2746f7d40ebc8cbc7f497519bf9d5286d1c48bdde81231e86`, complete prior ledger, generator SHA-256 `3904d99614bae0d8218603ddf966a07e572a54d32d5a0d33b9518aa2ee0bb572`, and source hashes. Produce an exclusive-create `darts_selection_sealed.json` only if all 20 canonical signatures and IDs are unique and disjoint from every known prior structure/checkpoint.

- [ ] **Step 1:** Write failing tests for an unresolved prior source, one signature collision, duplicate selected structure, changed selection hash, and success with a complete clean ledger.
- [ ] **Step 2:** Verify those failures, implement minimal sealing, rerun targeted tests.
- [ ] **Step 3:** On the server, independently verify all source hashes and create the sealed file once. Never overwrite or silently reselect a candidate after inspecting scores.
- [ ] **Step 4:** Commit code and the audited seal receipt; do not copy target values into selection artifacts.

### Task 3: Train e0/e100 weights in append-only attempts

**Files:** Create `scripts/train_activation_holdout_darts.py`; test in `tests/test_train_activation_holdout_darts.py`.

**Interfaces:** Consume only the sealed selection, project-local CIFAR-10 train files, and pinned source/environment hashes. Produce 20 architecture directories, each with one verified complete attempt containing `arch_<id>_epoch_0_seed_0.pt` and `arch_<id>_epoch_100_seed_0.pt`, or raw failed attempts.

- [ ] **Step 1:** Write failing tests for wrong/missing selection, source/data drift, duplicate complete receipts, tampered checkpoints, and exact-key skip on rerun.
- [ ] **Step 2:** Verify failures, implement a `--plan-only` freeze and exclusive training attempts; rerun targeted tests.
- [ ] **Step 3:** Pin C=16, 8 cells, CIFAR-10 50,000 train images, random crop/flip, TRADES PGD-10, epsilon 8/255, step 2/255, beta 6, batch 128, SGD lr 0.1/momentum 0.9/weight decay 2e-4, cosine 100 epochs, seed 0. Save e0 before updates and e100 after epoch 100. No `metric_row`, `pgd_accuracy`, Ours, or Φ call in this task.
- [ ] **Step 4:** To make the training-only trajectory deterministic and independent of omitted metric evaluations, call `set_seed(epoch)` before building each epoch's loader and performing that epoch's TRADES updates. Use `set_seed(0)` before each model initialization. Record this RNG schedule as an explicit difference from the historical trajectory, whose metric calls altered the RNG stream.
- [ ] **Step 5:** After main-grid GPU work and the project monitor grant capacity, train all 20. A failed model restarts from e0 in a new UUID attempt; do not reuse partial weights without full RNG state. Verify 40 exact weight keys, model metadata, finite tensors, and file SHA-256.
- [ ] **Step 6:** Commit code/tests and the non-outcome training plan. Preserve outputs append-only.

### Task 4: Evaluate complete e100 PGD-20 targets with per-image evidence

**Files:** Extend `scripts/evaluate_activation_holdout_darts_targets.py` and `scripts/verify_holdout_targets.py`; test in `tests/test_evaluate_activation_holdout_darts_targets.py` and `tests/test_verify_holdout_targets.py`.

**Interfaces:** Consume 20 verified e100 checkpoints, sealed genotype list, and the project-local CIFAR-10 `test_batch` SHA-256. Produce one exclusive JSON per cell with `sample_ids=0..9999`, 10,000 booleans, exact attack settings, checkpoint/source hashes, and `status=ok`; produce a separate 20-key target verification report.

- [ ] **Step 1:** Add failing tests for wrong model/genotype metadata, 9,999 or repeated image IDs, non-boolean correctness, forged aggregate, changed checkpoint/data hash, wrong PGD seed/batch/step, duplicate target record, and missing file.
- [ ] **Step 2:** Verify failures; add a CLI to the already-tested per-image PGD loop. Match the existing DARTS evaluator: `eval`, 10,000 ordered images, batch 128, PGD-20, epsilon 8/255, step 2/255, one random start, `set_seed(400000)` immediately before attack evaluation.
- [ ] **Step 3:** Verify each file's bytes, 10,000 ordered IDs, booleans, recomputed accuracy, protocol, model metadata and e100 checkpoint SHA with a verifier independent of the writer. Preserve failed attempts and rerun only missing cell targets.
- [ ] **Step 4:** Commit source/tests and append-only target evidence; do not calculate Ours/Φ holdout scores yet.

### Task 5: Join the three-space package and run a single independent comparison

**Files:** Extend `scripts/build_activation_holdout_manifest.py`, create `scripts/verify_activation_holdout_package.py` and `scripts/analyze_activation_holdout.py`; test corresponding `tests/test_*.py`.

**Interfaces:** Consume RB40 11, NB201 64×5, DARTS 20×2 fully verified weights/targets. Produce a sealed **371-key** manifest, a verified pair of Φ/Ours result tables, and pre-registered three-space gate report.

- [ ] **Step 1:** Write failing tests proving that a `targets_verified=true` flag alone cannot seal the package; require the actual target report SHA and all 371 exact unique finite keys.
- [ ] **Step 2:** Verify failures; implement strict package checks including old-exposure exclusion, weight/source hashes, stage-paired targets, and source code/environment hashes.
- [ ] **Step 3:** Only after the candidate passes its complete 560-key main gate, run Φ and Ours once on the same sealed 371 keys. Keep every failed attempt; resume only exact missing keys.
- [ ] **Step 4:** Compute the pre-registered space-wise and eight-setting differences plus paired 3,000-draw bootstrap seed 20261027. Require every space delta >0, overall delta >0 and interval lower bound >0; otherwise report failure. Mark this package exposed after the first look, even if the candidate loses.
- [ ] **Step 5:** Publish the full verification, exclusions, negative results and selection-history risk in a new release directory. Do not alter the old strict leaderboard or promote a partial result.
