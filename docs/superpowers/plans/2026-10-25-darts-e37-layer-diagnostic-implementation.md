# DARTS E37 Layer Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect and verify complete, paired per-layer E37 diagnostics for the frozen DARTS 100 architectures at e0/e100, without altering the nine published metrics or the strict leaderboard.

**Architecture:** A provenance gate freezes the 200-key manifest and pinned E37 implementation. A separate diagnostic module reuses the original feature/FGSM path, emits numeric-only components into immutable per-key attempts, then a verifier builds one complete canonical dataset and a read-only analysis script produces paired layer/scale tables. Existing E37 calculators, metric registry, strict releases and exclusions remain unchanged.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, pandas, pytest, CSV/JSON; server execution only inside `wangjiahao` at `/workspace/robust_metric_lab_nas`.

**Spec:** `docs/superpowers/specs/2026-10-25-darts-e37-layer-diagnostic-design.md` (approved by user on 2026-10-25).

## Global Constraints

- Exact full grid: 100 DARTS architecture IDs × `darts_e0`/`darts_e100` = **200** unique keys, with original checkpoint paths and labels; no object deletion after OOM or missing files.
- E37 protocol: 64 CIFAR-10 test images at offset 0, seed `20260808`, FGSM `epsilon=8/255`, `batch_size=1`, `all_leaf`, `_anchor_indices(..., max_layers=4)`; no modified E37 formula or direction.
- Every one of the nine E37 reconstructed scores must match its strict-release row within `rel=3e-5, abs=3e-8`. A mismatch stops interpretation, not merely a warning.
- The 200-key diagnostic is **not** a 560-row metric release. Do not register it, rank it, or use it as independent confirmation. Subsequent Ours candidates require separate preregistration, full 560 rows and independent validation.
- Preserve v3/E36/E37/TRNAS, RACL exclusions, strict releases, original failed rows and all earlier attempts. No writes outside the permitted RobustMetric project and no container except `wangjiahao`.
- All per-sample output is scalar numeric summaries, shape and module identity; never persist input images, perturbation arrays or activation tensors. GPU jobs must wait for free capacity and not disturb other processes.

## Review Focus

1. A manifest with one duplicated ID or one missing stage must fail before GPU work; Task 1 tests the exact key set.
2. A reused leaf module called twice must retain its invocation number, not silently collapse to one layer; Task 2 tests the trace.
3. Failed attempts must not be skipped or overwritten by a retry, and two successful attempts for one key must be rejected; Task 3 tests both.
4. A NaN nested inside a per-sample layer component must fail verification even if the nine model scores are finite; Task 4 tests recursive numeric validation.
5. e0/e100 module identities or tensor shapes may differ; Task 4 tests that this blocks paired interpretation without dropping an architecture.

---

## File map and fixed interfaces

| File | Responsibility |
|---|---|
| `src/robust_metric_lab/e37_metrics.py`, `src/robust_metric_lab/e37_batched.py`, `scripts/evaluate_e37_student.py` | Exact pinned E37 copies from `wangjiahao`; do not edit formulas; verify all three source hashes against the original E37 run manifest. |
| `src/robust_metric_lab/e37_layer_diagnostic.py` | Pure numeric component extraction and leaf call trace; returns JSON-safe per-sample records plus original nine reconstructed scores. |
| `scripts/e37_diagnostic_manifest.py` | Freeze/verify 200 IDs, checkpoint/source/input hashes and protocol; write-once manifest. |
| `scripts/run_e37_layer_diagnostic.py` | Load frozen DARTS item and samples, call diagnostic, write one immutable success/failure attempt per key. |
| `scripts/verify_e37_layer_diagnostic.py` | Merge attempts by exact key, enforce 200-key completeness and score/layer identity checks; output audit and canonical numeric dataset. |
| `scripts/analyze_e37_layer_diagnostic.py` | Read only verified dataset plus frozen structural control; emit all preregistered paired Spearman/scale tables. |
| `tests/test_e37_diagnostic_manifest.py`, `tests/test_e37_layer_diagnostic.py`, `tests/test_e37_diagnostic_attempts.py`, `tests/test_e37_diagnostic_verification.py`, `tests/test_e37_diagnostic_analysis.py` | Focused TDD checks for each interface. |

The new module exposes `leaf_call_trace(model, inputs) -> tuple[list[torch.Tensor], list[dict]]` and `diagnose_e37(model, clean, labels, *, epsilon: float, max_layers: int = 4) -> dict`. The returned dict contains `sample_components`, `anchor_metadata`, `metric_values`, `sample_count`, `feature_count`; `metric_values` maps the exact `E37_METRIC_NAMES`. The manifest helper exposes `expected_darts_keys(items) -> set[tuple[str,int]]` and `write_or_verify_manifest(path, payload) -> str`. The verifier exposes a pure `verify_records(records, *, expected_keys, strict_scores, frozen_hashes) -> dict` for tests and `verify_attempts(manifest, attempts_dir, strict_csv, control_csv) -> dict` for files; only the latter writes outputs after validation.

### Task 1: Freeze original E37 provenance and 200 input keys

**Files:** Create `scripts/e37_diagnostic_manifest.py`; test `tests/test_e37_diagnostic_manifest.py`; mechanically add exact server copies of `src/robust_metric_lab/e37_metrics.py`, `src/robust_metric_lab/e37_batched.py`, `scripts/evaluate_e37_student.py`, `tests/test_e37_batched.py` to the isolated worktree. The first three must match original E37 manifest hashes; the test file has its own recorded SHA-256. Do not alter server originals.

**Interfaces:** Consumes `load_darts_items(...)` and original `outputs/student_metrics/wangjiahao/e37_full_metric_design_v1/run_manifest.json`; produces immutable `frozen_manifest.json`, expected 200 keys and source/checkpoint hashes for Tasks 2–4.

- [ ] **Step 1: Test malformed DARTS coverage.** In `tests/test_e37_diagnostic_manifest.py`, create fake items with 100 IDs × two epochs, assert `len(expected_darts_keys(items)) == 200`; delete `(id='17', epoch=100)` and assert `ValueError('missing')`; duplicate `(id='17', epoch=0)` and assert `ValueError('duplicate')`.

```python
from types import SimpleNamespace
import pytest
from scripts.e37_diagnostic_manifest import expected_darts_keys

def item(identifier, epoch):
    return SimpleNamespace(model_or_arch_id=str(identifier), checkpoint_epoch=epoch)

def test_exact_200_keys():
    items = [item(i, epoch) for i in range(100) for epoch in (0, 100)]
    assert len(expected_darts_keys(items)) == 200
    with pytest.raises(ValueError, match="missing"):
        expected_darts_keys(items[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        expected_darts_keys(items + [item(17, 0)])
```
- [ ] **Step 2: Observe red.** Run `python -m pytest tests/test_e37_diagnostic_manifest.py -q`; expect an import/function failure for `expected_darts_keys`.
- [ ] **Step 3: Implement exact-key gate.** Implement `expected_darts_keys(items)` using `str(item.model_or_arch_id)` and `int(item.checkpoint_epoch)`, reject any epoch outside `{0,100}`, duplicates, any count other than 100 IDs per stage or unequal ID sets; return the complete set. Adapt the existing `write_or_verify_manifest` pattern from `scripts/adv_excess_run_manifest.py` with exclusive create, canonical JSON SHA-256 and refusal to replace differing content. Include protocol, 200 checkpoint SHA-256 values, original source/manifest SHA-256 values, bundle manifest SHA-256, Python/PyTorch/NumPy versions, strict-release CSV SHA-256 and control-table SHA-256.

```python
def expected_darts_keys(items) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    stages = {0: set(), 100: set()}
    for item in items:
        epoch = int(item.checkpoint_epoch)
        if epoch not in stages:
            raise ValueError(f"unsupported DARTS stage: {epoch}")
        key = (str(item.model_or_arch_id), epoch)
        if key in seen:
            raise ValueError(f"duplicate DARTS key: {key}")
        seen.add(key)
        stages[epoch].add(key[0])
    if len(stages[0]) != 100 or len(stages[100]) != 100 or stages[0] != stages[100]:
        raise ValueError("missing or unequal DARTS stage keys")
    return seen
```
- [ ] **Step 4: Pin original source.** In `wangjiahao` project, read the original E37 run manifest and compute `sha256sum` for the three E37 sources used by the original run; if any differs, stop and preserve a mismatch report. Transfer only those exact read-only source bytes plus the existing batch test into the isolated worktree, verify the three local SHA-256 values equal their original manifest values, and run `tests/test_e37_batched.py`. This is a provenance sync, not a source change. Do not deploy a changed original E37 file back to the server.

```sh
docker exec wangjiahao sh -lc 'cd /workspace/robust_metric_lab_nas && sha256sum src/robust_metric_lab/e37_metrics.py src/robust_metric_lab/e37_batched.py scripts/evaluate_e37_student.py && sed -n "1,120p" outputs/student_metrics/wangjiahao/e37_full_metric_design_v1/run_manifest.json'
```
- [ ] **Step 5: Green and commit.** Run `python -m pytest tests/test_e37_diagnostic_manifest.py tests/test_e37_batched.py -q`; expect PASS. Commit only Task 1 files as `test/feat: freeze E37 diagnostic provenance`.

### Task 2: Capture numeric components without changing E37 metrics

**Files:** Create `src/robust_metric_lab/e37_layer_diagnostic.py`; test `tests/test_e37_layer_diagnostic.py`. Do not edit `e37_metrics.py`, `e37_batched.py`, `metrics.py` or `metric_registry.py`.

**Interfaces:** Consumes original `make_adversarial`, `ActivationRecorder`, `_anchor_indices`, `compute_e37_per_sample`, and E37's `_flow_drift`, `_spectrum_probabilities`, `_centered_channel_drift`, `_spatial_cosine_drift`; produces the fixed `diagnose_e37` dict for Task 3.

- [ ] **Step 1: Write failing trace/component tests.** A tiny model calls one leaf convolution twice; assert `leaf_call_trace` emits separate `(module_path, invocation=1/2)` entries in output order and only records 4D outputs. With 5 deterministic samples, assert four selected anchors and three transitions; each per-sample row contains `clean_rms`, `adv_rms`, `clean_centered_l2`, `adv_centered_l2`, `channel`, `spatial`, `spectrum_fisher`, `spectrum_triangular`, `spectrum_ks`, and transition `flow` values, all finite; assert no tensor occurs recursively in returned data. Compare all nine returned `metric_values` to `compute_e37_metrics_batched(..., batch_size=1)` at `rel=3e-5, abs=3e-8`.

```python
class SharedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x):
        return self.shared(self.shared(x))

def test_repeated_leaf_invocation_is_distinct():
    model = SharedConvModel().eval()
    _, trace = leaf_call_trace(model, torch.ones(1, 3, 4, 4))
    shared = [row for row in trace if row["module_path"] == "shared"]
    assert [row["invocation"] for row in shared] == [1, 2]

def test_diagnostic_matches_original_nine_metrics():
    tiny_e37_net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(4, 6, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(6, 3),
    ).eval()
    clean = torch.linspace(0, 1, 5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1])
    actual = diagnose_e37(tiny_e37_net, clean, labels, epsilon=8 / 255)
    expected = compute_e37_metrics_batched(
        tiny_e37_net, clean, labels, epsilon=8 / 255, batch_size=1
    )
    assert set(actual["metric_values"]) == set(E37_METRIC_NAMES)
    for name in E37_METRIC_NAMES:
        assert actual["metric_values"][name] == pytest.approx(
            expected[name], rel=3e-5, abs=3e-8
        )
```
- [ ] **Step 2: Observe red.** Run `python -m pytest tests/test_e37_layer_diagnostic.py -q`; expect missing module/functions.
- [ ] **Step 3: Implement trace.** Register forward hooks on `model.named_modules()` leaves, append only 4D outputs, count invocations per qualified path, record `(path, invocation, shape)` in execution order, and remove hooks in `finally`. Verify its tensor count/order equals `ActivationRecorder(..., 'all_leaf')`; a difference raises `ValueError('trace mismatch')`.

```python
def leaf_call_trace(model, inputs):
    features, metadata, handles, counts = [], [], [], {}
    def make_hook(path):
        def hook(_module, _args, output):
            if torch.is_tensor(output) and output.ndim == 4:
                counts[path] = counts.get(path, 0) + 1
                features.append(output.detach())
                metadata.append({"module_path": path, "invocation": counts[path],
                                 "shape": list(output.shape)})
        return hook
    try:
        for path, module in model.named_modules():
            if not tuple(module.children()):
                handles.append(module.register_forward_hook(make_hook(path)))
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return features, metadata
```
- [ ] **Step 4: Implement diagnostic.** Follow `compute_e37_metrics_batched` exactly for FGSM, one-sample chunks, E37 anchor selection, `compute_e37_per_sample` and sum-then-divide model scores. For each selected layer call the original private E37 component functions; derive spectral Fisher/triangular/KS from the original `_spectrum_probabilities` expression, not a new normalization. Compute clean/adv RMS and centered L2 from flattened 4D features, and each adjacent `_flow_drift`. Convert only finite scalar values to Python floats. Compare clean/adv trace metadata for each sample and across 64 samples; mismatch raises with exact sample/anchor. Return `sample_components`, `anchor_metadata`, `metric_values`, `sample_count`, `feature_count`.

```python
def scalar(value):
    number = float(value.item())
    if not math.isfinite(number):
        raise ValueError("nonfinite layer component")
    return number

def anchor_values(clean_feature, adv_feature):
    p = _spectrum_probabilities(clean_feature)
    q = _spectrum_probabilities(adv_feature)
    affinity = (p.sqrt() * q.sqrt()).sum(dim=1)
    flat_clean = clean_feature.flatten(1)
    flat_adv = adv_feature.flatten(1)
    return {
        "clean_rms": scalar(flat_clean.square().mean(dim=1).sqrt()),
        "adv_rms": scalar(flat_adv.square().mean(dim=1).sqrt()),
        "clean_centered_l2": scalar((flat_clean - flat_clean.mean(dim=1, keepdim=True)).norm(dim=1)),
        "adv_centered_l2": scalar((flat_adv - flat_adv.mean(dim=1, keepdim=True)).norm(dim=1)),
        "channel": scalar(_centered_channel_drift(clean_feature, adv_feature)),
        "spatial": scalar(_spatial_cosine_drift(clean_feature, adv_feature)),
        "spectrum_fisher": scalar(2 * torch.acos(affinity.clamp(0, 1))),
        "spectrum_triangular": scalar(0.5 * ((p - q).square() / (p + q).clamp_min(1e-12)).sum(dim=1)),
        "spectrum_ks": scalar((p - q).cumsum(dim=1).abs().max(dim=1).values),
    }

totals = {name: 0.0 for name in E37_METRIC_NAMES}
sample_components = []
for index in range(clean.shape[0]):
    x, y = clean[index:index + 1], labels[index:index + 1]
    x_adv = make_adversarial(model, x, y, "fgsm", epsilon, pgd_steps=1)
    with torch.no_grad():
        clean_features, clean_trace = leaf_call_trace(model, x)
        adv_features, adv_trace = leaf_call_trace(model, x_adv)
    anchors = _anchor_indices(min(len(clean_features), len(adv_features)), max_layers=4)
    values = compute_e37_per_sample(clean_features, adv_features, anchors)
    for name in E37_METRIC_NAMES:
        totals[name] += float(values[name].sum().item())
    if clean_trace != adv_trace:
        raise ValueError(f"trace mismatch at sample {index}")
    layer_rows = [anchor_values(clean_features[a], adv_features[a]) for a in anchors]
    transition_rows = [
        {"flow": scalar(_flow_drift(clean_features[left], clean_features[right],
                                     adv_features[left], adv_features[right]))}
        for left, right in zip(anchors, anchors[1:])
    ]
    sample_components.append({"sample_index": index, "anchors": layer_rows,
                              "transitions": transition_rows})
metric_values = {name: totals[name] / clean.shape[0] for name in E37_METRIC_NAMES}
```
- [ ] **Step 5: Green and commit.** Run `python -m pytest tests/test_e37_layer_diagnostic.py tests/test_e37_batched.py -q`; expect PASS. Commit only Task 2 files as `feat: record E37 layer diagnostic components`.

### Task 3: Write immutable attempts and exact-key repair runner

**Files:** Create `scripts/run_e37_layer_diagnostic.py`; test `tests/test_e37_diagnostic_attempts.py`.

**Interfaces:** Consumes Task 1 `frozen_manifest.json`, Task 2 `diagnose_e37`, `make_model_loader` from `scripts/evaluate_darts100_student.py`, and `load_cifar10_samples`; produces `attempts/<attempt_id>/<setting>/<arch_id>.json` and `attempts/<attempt_id>/launcher.log`. An attempt ID is an explicit safe single directory name; all files are exclusive-create. `--repair-keys` is generated only by Task 4 verifier and must enumerate the exact missing/failed keys from all preceding attempts.

- [ ] **Step 1: Test immutable write and selection.** Use `tmp_path` and a stub `diagnose_e37`; one success writes a JSON record with exact key, source/protocol/checkpoint hashes, 64 samples and four anchors. A repeated write to the same path raises `FileExistsError`. A failing loader writes `status='failed'`, error class/message and traceback, without invented numeric values. `--repair-keys` containing an unlisted or previously successful key fails before any load; a file containing only a failed/missing key is accepted for a separate attempt.

```python
def test_exclusive_attempt_record(tmp_path):
    record = {"setting": "darts_e0", "model_or_arch_id": "17", "status": "failed",
              "error_type": "RuntimeError", "error": "out of memory", "traceback": "RuntimeError: out of memory"}
    path = tmp_path / "attempts" / "a1" / "darts_e0" / "17.json"
    write_attempt(path, record)
    assert json.loads(path.read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        write_attempt(path, record)

def test_repair_rejects_successful_key():
    expected = {("darts_e0", "17"), ("darts_e100", "17")}
    successful = {("darts_e0", "17")}
    with pytest.raises(ValueError, match="successful"):
        validate_repair_keys({("darts_e0", "17")}, expected, successful)
```
- [ ] **Step 2: Observe red.** Run `python -m pytest tests/test_e37_diagnostic_attempts.py -q`; expect runner writer/key-selection failures.
- [ ] **Step 3: Implement CLI and writer.** Require `--bundle-root`, `--data-root`, `--frozen-manifest`, `--output-root`, `--attempt-id`, `--device`; optional `--repair-keys` is the verifier-produced CSV of exact `(setting,model_or_arch_id)` keys. Reject `--item-limit` in formal mode. Revalidate frozen hashes and exact 200-key set before model load; for repairs, recompute missing/failed keys from previous immutable attempts and require exact equality with the requested subset. For each selected key, load the original checkpoint strictly, call `diagnose_e37`, record wall time/peak CUDA allocation, and write JSON with `open('x')`, `allow_nan=False`, `flush` and `fsync`. On exception write a separate failed record with `traceback.format_exc()` and continue only if safe; never overwrite previous attempt or call `evaluate_items` for retries.

```python
def validate_repair_keys(requested, expected, successful):
    if not requested <= expected:
        raise ValueError(f"unknown repair keys: {sorted(requested - expected)}")
    if requested & successful:
        raise ValueError(f"previously successful repair keys: {sorted(requested & successful)}")
    if requested != expected - successful:
        raise ValueError("repair keys must equal all missing/failed keys")

def write_attempt(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

try:
    model = load_model(item)
    diagnostic = diagnose_e37(model, clean, labels, epsilon=EPSILON)
    record = {"setting": setting, "model_or_arch_id": item.model_or_arch_id,
              "status": "ok", "diagnostic": diagnostic, "source_sha256": source_hashes,
              "checkpoint_sha256": checkpoint_hash}
except Exception as exc:
    record = {"setting": setting, "model_or_arch_id": item.model_or_arch_id,
              "status": "failed", "error_type": type(exc).__name__,
              "error": str(exc), "traceback": traceback.format_exc(),
              "source_sha256": source_hashes, "checkpoint_sha256": checkpoint_hash}
write_attempt(attempt_path, record)
```
- [ ] **Step 4: Green and commit.** Run `python -m pytest tests/test_e37_diagnostic_attempts.py -q`; expect PASS. Commit only Task 3 files as `feat: preserve E37 diagnostic attempts`.

### Task 4: Verify all 200 keys and analyze paired layers

**Files:** Create `scripts/verify_e37_layer_diagnostic.py`, `scripts/analyze_e37_layer_diagnostic.py`; test `tests/test_e37_diagnostic_verification.py`, `tests/test_e37_diagnostic_analysis.py`.

**Interfaces:** Consumes frozen manifest, all immutable attempts, strict `metric_values_merged.csv`, static `controls/darts_e0.csv`; produces `verification.json`, `canonical_numeric.jsonl`, `paired_layer.csv`, `component_correlations.csv`, `reconstruction_errors.csv`, and a Chinese interpretation/issue report. Do not create canonical analysis outputs on failed verification.

- [ ] **Step 1: Write failing verifier tests.** Fabricate complete 200-key records with nine metric names; confirm PASS only if each exact key appears once, has 64 finite sample records, required 4 anchors/3 transitions, matching hashes, and all nine strict-reference values satisfy `abs(actual-expected) <= 3e-8 + 3e-5*abs(expected)`. Check that missing key, extra key, duplicate successful key, nested NaN, failed-only key, changed checkpoint/source hash and one out-of-tolerance metric all yield FAIL with exact evidence. A prior failed attempt plus one later valid success must remain auditable and may PASS.

```python
import copy
from types import SimpleNamespace

@pytest.fixture
def valid_200_attempts():
    keys = {(f"darts_e{epoch}", str(i)) for epoch in (0, 100) for i in range(100)}
    scores = {key: {name: 0.5 for name in E37_METRIC_NAMES} for key in keys}
    hashes = {"source_sha256": {"e37": "sha"}, "checkpoint_sha256": "ck"}
    records = []
    for setting, identifier in sorted(keys):
        anchors = [{"module_path": f"layer{j}", "invocation": 1,
                    "shape": [1, 2, 4, 4]} for j in range(4)]
        samples = [{"sample_index": s,
                    "anchors": [{"clean_rms": 1.0, "adv_rms": 1.1,
                                 "channel": 0.1, "spatial": 0.2,
                                 "spectrum_fisher": 0.3, "spectrum_triangular": 0.4,
                                 "spectrum_ks": 0.5, "clean_centered_l2": 1.0,
                                 "adv_centered_l2": 1.1} for _ in range(4)],
                    "transitions": [{"flow": 0.1} for _ in range(3)]}
                   for s in range(64)]
        records.append({"setting": setting, "model_or_arch_id": identifier,
                        "status": "ok", "source_sha256": hashes["source_sha256"],
                        "checkpoint_sha256": "ck",
                        "diagnostic": {"sample_count": 64, "feature_count": 4,
                                       "anchor_metadata": anchors,
                                       "sample_components": samples,
                                       "metric_values": scores[(setting, identifier)]}})
    return SimpleNamespace(records=records, expected_keys=keys,
                           strict_scores=scores, hashes=hashes)

def mutate(base, kind):
    records = copy.deepcopy(base.records)
    if kind == "missing":
        records.pop()
    elif kind == "extra":
        records.append({**copy.deepcopy(records[0]), "model_or_arch_id": "not-frozen"})
    elif kind == "duplicate_ok":
        records.append(copy.deepcopy(records[0]))
    elif kind == "nested_nan":
        records[0]["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = float("nan")
    elif kind == "failed_only":
        records[0].update(status="failed", error="OOM", traceback="OOM")
        records[0].pop("diagnostic")
    elif kind == "changed_hash":
        records[0]["source_sha256"] = {"e37": "changed"}
    elif kind == "score_mismatch":
        records[0]["diagnostic"]["metric_values"][E37_METRIC_NAMES[0]] = 0.7
    return records

@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_ok", "nested_nan",
                                       "failed_only", "changed_hash", "score_mismatch"])
def test_verification_rejects_corrupt_grid(valid_200_attempts, mutation):
    attempts = mutate(valid_200_attempts, mutation)
    result = verify_records(attempts, expected_keys=valid_200_attempts.expected_keys,
                            strict_scores=valid_200_attempts.strict_scores,
                            frozen_hashes=valid_200_attempts.hashes)
    assert result["status"] == "FAIL"
    assert result["reasons"]

def test_old_failure_plus_new_success_keeps_both_attempts(valid_200_attempts):
    attempts = copy.deepcopy(valid_200_attempts.records)
    attempts.append({"setting": "darts_e0", "model_or_arch_id": "17",
                     "status": "failed", "error": "OOM", "traceback": "OOM",
                     "source_sha256": valid_200_attempts.hashes["source_sha256"],
                     "checkpoint_sha256": "ck"})
    result = verify_records(attempts, expected_keys=valid_200_attempts.expected_keys,
                            strict_scores=valid_200_attempts.strict_scores,
                            frozen_hashes=valid_200_attempts.hashes)
    assert result["status"] == "PASS"
    assert result["failed_attempt_count"] == 1
```
- [ ] **Step 2: Write failing pair/analysis tests.** Two paired fake stages with matching names/shapes yield 100 complete pairs; mutate one anchor shape/path and assert pairing fails without deleting that architecture. Check an all-constant component reports Spearman undefined rather than filtering rows, and an integer-vs-string ID control join is normalized to canonical strings and checked one-to-one.

```python
def test_layer_mismatch_blocks_full_pairing(valid_200_attempts):
    records = copy.deepcopy(valid_200_attempts.records)
    row = next(row for row in records if row["setting"] == "darts_e100"
               and row["model_or_arch_id"] == "17")
    row["diagnostic"]["anchor_metadata"][0]["shape"] = [1, 99, 8, 8]
    result = verify_records(records, expected_keys=valid_200_attempts.expected_keys,
                            strict_scores=valid_200_attempts.strict_scores,
                            frozen_hashes=valid_200_attempts.hashes)
    assert result["status"] == "FAIL"
    assert ("darts_e100", "17") in result["pair_mismatches"]

def test_constant_component_is_reported_undefined():
    rho = paired_spearman([2.0] * 100, [3.0] * 100)
    assert rho is None
```
- [ ] **Step 3: Observe red.** Run `python -m pytest tests/test_e37_diagnostic_verification.py tests/test_e37_diagnostic_analysis.py -q`; expect missing verifier/analysis functions.
- [ ] **Step 4: Implement verifier.** Parse JSON with duplicate-key detection; recursively reject bool-as-number, NaN/Inf and unexpected tensor-like values in numeric fields. Verify full attempt lineage, exact 200-key set, sample/anchor/transition counts, module identity pairing, nine score names and reference comparison. Write `verification.json` even on FAIL with all reasons; write canonical JSONL only on PASS using exclusive output paths. Compute and include max absolute/relative score errors.

```python
def require_finite_numeric(value, path="root"):
    if isinstance(value, bool):
        raise ValueError(f"boolean in numeric field: {path}")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite: {path}")
    elif isinstance(value, list):
        for index, element in enumerate(value):
            require_finite_numeric(element, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, element in value.items():
            if key in {"anchors", "transitions", "metric_values"} or isinstance(element, (int, float, list, dict)):
                require_finite_numeric(element, f"{path}.{key}")

def score_matches(actual: float, expected: float) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= 3e-8 + 3e-5 * abs(expected)
```
- [ ] **Step 5: Implement preregistered analysis.** Load only a PASS canonical dataset. Aggregate 64 samples equally per key; for each of four anchor slots and three flow transitions output both-stage component values, paired raw-rank change `rank(e100)-rank(e0)`, 100-architecture Spearman between stages, change distributions and Spearman of change with `log(parameter_count)`. Validate control join 100/100 one-to-one; report constant-input correlations as undefined. Emit every planned component (not only favorable layers) and label all structure comparisons exploratory/non-causal.

```python
def paired_spearman(e0_values, e100_values):
    if len(e0_values) != 100 or len(e100_values) != 100:
        raise ValueError("paired analysis requires all 100 DARTS architectures")
    if len(set(e0_values)) == 1 or len(set(e100_values)) == 1:
        return None
    rho = float(scipy.stats.spearmanr(e0_values, e100_values).statistic)
    return rho if math.isfinite(rho) else None

control["model_or_arch_id"] = control["model_or_arch_id"].astype(str)
paired["model_or_arch_id"] = paired["model_or_arch_id"].astype(str)
joined = paired.merge(control, on="model_or_arch_id", validate="one_to_one", indicator=True)
if len(joined) != 100 or not joined["_merge"].eq("both").all():
    raise ValueError("static structural control is not a complete 100/100 join")
```
- [ ] **Step 6: Green and commit.** Run `python -m pytest tests/test_e37_diagnostic_verification.py tests/test_e37_diagnostic_analysis.py -q`; expect PASS. Commit only Task 4 files as `feat: verify and analyze E37 layer diagnostic`.

### Task 5: End-to-end server run, evidence and handoff

**Files:** Create a new dated directory only under `/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/`; create `docs/experiments/DARTS_E37_逐层诊断结果与问题记录_20261025.md` after results. Do not edit strict release or old attempts.

**Interfaces:** Consumes Tasks 1–4, original frozen manifests/checkpoints and free GPU; produces a reproducible 200-key diagnostic release, not a ranked metric.

- [ ] **Step 1: Preflight provenance.** Inside `wangjiahao`, run the original E37 source SHA-256 checks, all five diagnostic pytest files plus original `tests/test_e37_batched.py`, frozen manifest generation, 200-key/checkpoint existence check, and strict-release hash check. On mismatch stop and save logs; do not patch published sources.
- [ ] **Step 2: Smoke for implementation/resources only.** Select two predeclared architecture IDs at both stages (four keys), run a separately named smoke attempt behind the existing GPU-memory gate. Check E37 reconstruction, output shape, finite values, runtime and peak memory. Do not modify formulas after looking at robustness correlations. If implementation bug is found, fix with a new source hash and rerun smoke in a new attempt.
- [ ] **Step 3: Full 200-key attempt.** Wait using the existing GPU gate (`scripts/run_e37_full.sh` as operational pattern) until enough memory is free; run only `wangjiahao` project. Record process command, start/end UTC, source/manifest hashes, exit status and GPU checks. Do not interfere with other jobs or other containers.
- [ ] **Step 4: Exact-key repair.** If failures occur, preserve initial files/logs, generate a keys CSV containing only missing/failed exact keys, run a new immutable attempt. Do not rerun a successful key or reduce the 100-architecture set. If retries require a definition change, stop and version the specification/manifests instead of merging silently.
- [ ] **Step 5: Verify and report.** Require `verification.json` PASS, 200/200 complete unique finite records, nine-score reconstruction PASS and 100/100 paired module identities before interpretation. Run full preregistered analysis, preserve any negative/undefined findings, write Chinese issue/evidence report with runtime and memory. If verification fails, report failure and do not produce a mechanism claim or candidate metric.
- [ ] **Step 6: Handoff.** Show the user diagnostic report and provenance hashes. Only then start a *separate* brainstorming/spec/preregistration cycle for the next Ours metric; do not add this diagnostic to the strict leaderboard. Full-goal completion still needs a later 560-row strict release beating TRNAS Φ plus independent validation; the missing RB40 holdout weight and three-space independent materials remain explicit blockers.

## Self-review checkpoints

Spec §§1–6 map to Tasks 1–5 respectively: provenance/grid (1), numeric capture (2), append-only failures (3), complete verification and fixed analysis (4), scoped GPU run and separation from 560-row ranking (5). Review Focus conditions are explicitly tested in Tasks 1–4. No task edits old releases, registry or exclusion policy. The first implementation action waits for user review of this plan and selection of execution method.
