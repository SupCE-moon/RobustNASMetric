# 鲁棒指标三空间基准实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `wangjiahao` 容器的 `/workspace/robust_metric_lab_nas` 中建立可审计的 RB40、NB201-64、DARTS-100 八 setting 评测流水线，补齐 E25 之后指标覆盖并生成冻结的跨空间总表。

**Architecture:** 本地代码库负责协议、manifest 解析、三空间评测适配器、统计分析和测试；老师容器只作为 benchmark 快照来源，目标容器只运行经过测试的代码。三个空间输出同一行级 schema，再由统一分析器计算各 setting 相关性和八 setting 均值。材料迁移、smoke test、完整评测和条件式新指标设计之间设置硬性关卡。

**Tech Stack:** Python 3.9+、PyTorch、torchvision、RobustBench、pandas、SciPy、PyYAML、pytest、Windows OpenSSH、Docker。

**Spec:** `docs/superpowers/specs/2026-09-17-three-space-benchmark-design.md`

## Global Constraints

- 只读取源容器 `huangyue:/root/robust_metric_lab_nas` 中老师提供的项目和 benchmark。
- 只修改目标容器 `wangjiahao` 的 `/workspace/robust_metric_lab_nas`；不得修改 `/workspace/DynamicBN20251203`。
- 不访问其他容器、其他用户目录、其他 GPU 或无关数据集。
- 密码不得写入文件、命令、日志、Git 或实验输出。
- 保留源 `TRANSFER_STATUS.json` 原文和验收结果；不得伪造 `COMPLETE` 或 `PASS`。
- 固定八个 setting：RB40 mature、NB201 e0/e5/e10/e25/e50、DARTS e0/e100。
- 固定样本协议：`sample_offset=0`、`num_samples=64`、`seed=20260808`、`epsilon=8/255`。
- 指标方向必须在读取最终 robust target 前注册；不得事后翻转方向。
- 只有八个 setting 全部有效时才计算 `All-setting mean`。
- 历史结果不得覆盖；所有运行写入新的 run 目录。

---

### Task 1：定义冻结 setting 和统一结果 schema

**Files:**
- Create: `src/robust_metric_lab/student_benchmark.py`
- Create: `tests/test_student_benchmark_protocol.py`

**Interfaces:**
- Consumes: `src/robust_metric_lab/metric_registry.py::METRIC_BY_NAME`
- Produces: `BenchmarkSetting`、`BENCHMARK_SETTINGS`、`MetricValueRow`、`validate_complete_setting_grid()`、`all_setting_mean()`

- [ ] **Step 1: 写失败测试，固定八 setting 和完整性规则**

```python
from robust_metric_lab.student_benchmark import (
    BENCHMARK_SETTINGS,
    all_setting_mean,
    validate_complete_setting_grid,
)


def test_frozen_benchmark_has_exactly_eight_settings():
    assert [s.key for s in BENCHMARK_SETTINGS] == [
        "rb40_mature",
        "nb201_e0",
        "nb201_e5",
        "nb201_e10",
        "nb201_e25",
        "nb201_e50",
        "darts_e0",
        "darts_e100",
    ]


def test_all_setting_mean_requires_all_eight_values():
    values = {s.key: 0.5 for s in BENCHMARK_SETTINGS}
    assert all_setting_mean(values) == 0.5
    values.pop("darts_e100")
    try:
        all_setting_mean(values)
    except ValueError as exc:
        assert "darts_e100" in str(exc)
    else:
        raise AssertionError("missing setting must fail")


def test_setting_grid_rejects_duplicate_ids():
    rows = [
        {"setting": "rb40_mature", "model_or_arch_id": "m0"},
        {"setting": "rb40_mature", "model_or_arch_id": "m0"},
    ]
    errors = validate_complete_setting_grid(rows, {"rb40_mature": 1})
    assert any("duplicate" in error.lower() for error in errors)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_student_benchmark_protocol.py -v`

Expected: collection fails because `robust_metric_lab.student_benchmark` does not exist.

- [ ] **Step 3: 实现最小协议对象和校验函数**

```python
from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, Mapping


@dataclass(frozen=True)
class BenchmarkSetting:
    key: str
    benchmark: str
    checkpoint_epoch: int | None
    target_column: str
    expected_rows: int


BENCHMARK_SETTINGS = (
    BenchmarkSetting("rb40_mature", "rb40", None, "robust_accuracy", 40),
    BenchmarkSetting("nb201_e0", "nb201", 0, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e5", "nb201", 5, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e10", "nb201", 10, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e25", "nb201", 25, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e50", "nb201", 50, "official_final_robust_accuracy", 64),
    BenchmarkSetting("darts_e0", "darts", 0, "full_pgd20_accuracy", 100),
    BenchmarkSetting("darts_e100", "darts", 100, "full_pgd20_accuracy", 100),
)


def all_setting_mean(values: Mapping[str, float]) -> float:
    keys = [setting.key for setting in BENCHMARK_SETTINGS]
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"missing settings: {', '.join(missing)}")
    return fsum(float(values[key]) for key in keys) / len(keys)
```

Implement `validate_complete_setting_grid()` with exact duplicate, missing-row, and unknown-setting diagnostics. Add a frozen `MetricValueRow` dataclass containing the columns required by the design.

- [ ] **Step 4: 运行测试并确认通过**

Run: `pytest tests/test_student_benchmark_protocol.py -v`

Expected: all tests pass.

- [ ] **Step 5: 提交**

```bash
git add src/robust_metric_lab/student_benchmark.py tests/test_student_benchmark_protocol.py
git commit -m "feat: define frozen student benchmark protocol"
```

### Task 2：实现 manifest 解析与路径安全检查

**Files:**
- Modify: `src/robust_metric_lab/student_benchmark.py`
- Create: `tests/test_student_benchmark_manifests.py`

**Interfaces:**
- Consumes: teacher manifests `rb40_models.csv`、`nb201_64_architectures.csv`、`darts100_architectures.csv`
- Produces: `load_rb40_items()`、`load_nb201_items()`、`load_darts_items()`、`resolve_under_root()`

- [ ] **Step 1: 写失败测试，覆盖数量、epoch 网格和目录逃逸**

```python
from pathlib import Path

import pytest

from robust_metric_lab.student_benchmark import resolve_under_root


def test_resolve_under_root_rejects_parent_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="outside benchmark root"):
        resolve_under_root(tmp_path, "../secret.pt")


def test_resolve_under_root_accepts_nested_artifact(tmp_path: Path):
    expected = tmp_path / "nb201_64" / "checkpoints" / "a.pt"
    assert resolve_under_root(tmp_path, "nb201_64/checkpoints/a.pt") == expected.resolve()
```

Add temporary fixture manifests with exactly two architectures and explicit e0/e5 paths. Assert that loaders preserve manifest order, parse JSON genotypes, and reject duplicate IDs or missing required columns.

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_student_benchmark_manifests.py -v`

Expected: import errors for the new loader functions.

- [ ] **Step 3: 实现 CSV loaders**

Use `csv.DictReader`, explicit required-column sets, and `Path.resolve()` containment checks. Do not infer checkpoint names when a manifest provides them. Represent each item as a frozen dataclass with `benchmark`、`model_or_arch_id`、`checkpoint_epoch`、`checkpoint_path`、`target_value` and `metadata`.

- [ ] **Step 4: 运行 manifest 测试**

Run: `pytest tests/test_student_benchmark_manifests.py -v`

Expected: all tests pass.

- [ ] **Step 5: 提交**

```bash
git add src/robust_metric_lab/student_benchmark.py tests/test_student_benchmark_manifests.py
git commit -m "feat: parse frozen benchmark manifests"
```

### Task 3：实现材料快照审计器

**Files:**
- Create: `scripts/audit_student_benchmark.py`
- Create: `tests/test_audit_student_benchmark.py`

**Interfaces:**
- Consumes: `--root`、`--output-dir`、原始 `TRANSFER_STATUS.json` 和 manifests
- Produces: `source_snapshot.json`、`manifest_sha256.csv`、`file_inventory.csv`，退出码 0 表示审计文件成功生成，不代表老师 verifier PASS

- [ ] **Step 1: 写失败测试，验证状态不会被改写**

```python
import json
from pathlib import Path

from scripts.audit_student_benchmark import build_snapshot


def test_snapshot_preserves_in_progress_state(tmp_path: Path):
    root = tmp_path / "bundle"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    (root / "TRANSFER_STATUS.json").write_text(
        json.dumps({"state": "IN_PROGRESS"}), encoding="utf-8"
    )
    (manifests / "bundle_metadata.json").write_text("{}", encoding="utf-8")
    snapshot = build_snapshot(root)
    assert snapshot["transfer_status"]["state"] == "IN_PROGRESS"
    assert snapshot["audit_result"] == "RECORDED_NOT_VERIFIED"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_audit_student_benchmark.py -v`

Expected: module import fails.

- [ ] **Step 3: 实现流式 SHA-256 和清单生成**

`sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str` must stream files and never read a checkpoint fully into RAM. `build_snapshot()` records UTC timestamp, resolved root, original status JSON, manifest digests, recursive file count, and byte total. The CLI writes atomically through a sibling `.tmp` file and `Path.replace()`.

- [ ] **Step 4: 运行测试和 CLI 参数检查**

Run: `pytest tests/test_audit_student_benchmark.py -v`

Run: `python scripts/audit_student_benchmark.py --help`

Expected: tests pass and the CLI lists required `--root` and `--output-dir` arguments.

- [ ] **Step 5: 提交**

```bash
git add scripts/audit_student_benchmark.py tests/test_audit_student_benchmark.py tests/fixtures/student_bundle
git commit -m "feat: audit student benchmark snapshots"
```

### Task 4：实现统一统计分析器

**Files:**
- Create: `src/robust_metric_lab/student_analysis.py`
- Create: `scripts/analyze_student_benchmark.py`
- Create: `tests/test_student_benchmark_analysis.py`

**Interfaces:**
- Consumes: unified `metric_values.csv`、`METRIC_BY_NAME` frozen directions
- Produces: per-setting `correlations.csv`、`all_setting_table.csv`、`analysis_report.md`

- [ ] **Step 1: 写失败测试，禁止缺 setting、NaN 和事后翻向**

```python
import pandas as pd
import pytest

from robust_metric_lab.student_analysis import summarize_metric


def test_summary_uses_registered_lower_direction():
    frame = pd.DataFrame({
        "setting": ["rb40_mature"] * 4,
        "metric_name": ["feature_delta"] * 4,
        "metric_value": [4.0, 3.0, 2.0, 1.0],
        "target_value": [1.0, 2.0, 3.0, 4.0],
        "status": ["ok"] * 4,
    })
    row = summarize_metric(frame, "feature_delta", "rb40_mature")
    assert row["raw_spearman"] == -1.0
    assert row["direction_adjusted_spearman"] == 1.0


def test_summary_rejects_unregistered_metric():
    with pytest.raises(KeyError, match="unregistered metric"):
        summarize_metric(pd.DataFrame(), "made_up", "rb40_mature")
```

Add tests for Kendall tau, Pearson, deterministic bootstrap seed `20260808`, eight-setting completeness, and failure-row accounting.

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_student_benchmark_analysis.py -v`

Expected: import error for `student_analysis`.

- [ ] **Step 3: 实现统计函数和 CLI**

Use `scipy.stats.spearmanr`, `pearsonr`, and `kendalltau`. Bootstrap architecture/model rows 2,000 times with `numpy.random.default_rng(20260808)`. Read direction only from `METRIC_BY_NAME`; unknown metrics fail. Compute `All-setting mean` through Task 1's `all_setting_mean()`.

- [ ] **Step 4: 运行测试并确认通过**

Run: `pytest tests/test_student_benchmark_analysis.py -v`

Expected: all tests pass.

- [ ] **Step 5: 提交**

```bash
git add src/robust_metric_lab/student_analysis.py scripts/analyze_student_benchmark.py tests/test_student_benchmark_analysis.py
git commit -m "feat: analyze frozen eight-setting benchmark"
```

### Task 5：实现 E25+ 覆盖矩阵

**Files:**
- Create: `scripts/build_e25_coverage_matrix.py`
- Create: `tests/test_e25_coverage_matrix.py`

**Interfaces:**
- Consumes: `metric_registry.METRIC_SPECS` and one or more `metric_values.csv`
- Produces: `e25_plus_coverage.csv` with one row per metric and one column per frozen setting, values `complete`、`partial`、`missing`、`failed`

- [ ] **Step 1: 写失败测试**

```python
from scripts.build_e25_coverage_matrix import coverage_state


def test_coverage_states_are_explicit():
    assert coverage_state(ok_rows=64, failed_rows=0, expected_rows=64) == "complete"
    assert coverage_state(ok_rows=63, failed_rows=1, expected_rows=64) == "failed"
    assert coverage_state(ok_rows=32, failed_rows=0, expected_rows=64) == "partial"
    assert coverage_state(ok_rows=0, failed_rows=0, expected_rows=64) == "missing"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_e25_coverage_matrix.py -v`

Expected: import error for the new script.

- [ ] **Step 3: 实现覆盖矩阵**

Require explicit `--input` paths and derive setting from the `setting` column, never from filenames. Preserve metric registry order and include attribution, direction, expected count, valid count, and failed count.

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_e25_coverage_matrix.py -v`

Expected: all tests pass.

- [ ] **Step 5: 提交**

```bash
git add scripts/build_e25_coverage_matrix.py tests/test_e25_coverage_matrix.py
git commit -m "feat: report E25 metric coverage"
```

### Task 6：实现三个只评测、不训练的 runner

**Files:**
- Create: `scripts/evaluate_rb40_student.py`
- Create: `scripts/evaluate_nb201_student.py`
- Create: `scripts/evaluate_darts100_student.py`
- Create: `src/robust_metric_lab/student_runner.py`
- Create: `tests/test_student_runner.py`

**Interfaces:**
- Consumes: parsed benchmark items、CIFAR-10 loader、`compute_feature_metrics()`、`compute_feature_geometry_metrics()`、TRNAS helpers
- Produces: a unified row writer `evaluate_items(items, load_model, loader, metrics, output_path, run_metadata)` and append-safe `metric_values.csv`

- [ ] **Step 1: 写失败测试，验证失败行继续、重复键拒绝和原子写入**

```python
from pathlib import Path

from robust_metric_lab.student_runner import append_rows_atomic


def test_append_rows_atomic_rejects_duplicate_keys(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    row = {
        "setting": "nb201_e0",
        "model_or_arch_id": "1",
        "metric_name": "feature_delta",
        "status": "ok",
    }
    append_rows_atomic(output, [row])
    try:
        append_rows_atomic(output, [row])
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate row must fail")
```

Add a fake loader where the second model raises `RuntimeError("broken fixture")`; assert the first and third rows are `ok`, the second is `failed`, and all three IDs remain in the CSV.

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_student_runner.py -v`

Expected: import error for `student_runner`.

- [ ] **Step 3: 实现共享 runner**

The unique key is `(setting, model_or_arch_id, checkpoint_epoch, metric_name, seed, sample_offset, num_samples)`. Write to `metric_values.csv.tmp`, flush and `os.fsync()`, then replace. Catch per-item exceptions, store the exception type/message in `error`, call `torch.cuda.empty_cache()`, and continue. Never catch `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 4: 实现空间适配器**

`evaluate_rb40_student.py` must call `load_rb_model()` with manifest `model_name` and the bundle's `rb40/models` root. `evaluate_nb201_student.py` must call `build_nb201_model()` using manifest `arch_id/arch_string`, fixed `C=16,N=5`, then load `model_state_dict`. `evaluate_darts100_student.py` must parse `genotype_json`, construct `DartsNetworkCIFAR(channels=16, layers=8)`, wrap `Normalize()`, and load `model_state_dict`.

Each CLI must accept exact common flags:

```text
--bundle-root PATH
--data-root PATH
--output-dir PATH
--run-id TEXT
--sample-offset INT
--num-samples INT
--seed INT
--item-limit INT
--metric NAME (repeatable)
--device TEXT
```

NB201 additionally accepts `--epoch {0,5,10,25,50}`; DARTS accepts `--epoch {0,100}`.

- [ ] **Step 5: 运行 runner 测试和全套快速测试**

Run: `pytest tests/test_student_runner.py tests/test_nb201_metric_sampling.py tests/test_darts_space.py -v`

Expected: all tests pass.

- [ ] **Step 6: 提交**

```bash
git add scripts/evaluate_rb40_student.py scripts/evaluate_nb201_student.py scripts/evaluate_darts100_student.py src/robust_metric_lab/student_runner.py tests/test_student_runner.py
git commit -m "feat: add manifest-driven benchmark runners"
```

### Task 7：迁移代码和 benchmark 快照到目标容器

**Files:**
- Create on server: `/workspace/robust_metric_lab_nas/docs/experiments/source_snapshot/`
- Preserve: `/workspace/DynamicBN20251203`

**Interfaces:**
- Consumes: local repository and `huangyue:/root/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1`
- Produces: `/workspace/robust_metric_lab_nas` with code and embedded `shared_benchmarks/robust_metric_v1`

- [ ] **Step 1: 记录目标容器现状**

Run through the `guzhengkai` SSH entry:

```bash
docker exec wangjiahao bash -lc '
  set -e
  test -d /workspace/DynamicBN20251203
  test ! -e /workspace/robust_metric_lab_nas
  df -h /workspace
'
```

Expected: old project exists, destination does not exist, and free space exceeds 40 GiB.

- [ ] **Step 2: 打包本地代码，排除平台环境与缓存**

Run locally from the repository root:

```powershell
tar.exe -czf D:\Project\codex\2026-09-16\nid\work\robust_metric_lab_nas.tar.gz `
  --exclude=.venv --exclude=.idea --exclude=__pycache__ --exclude=.pytest_cache `
  --exclude=models --exclude=data --exclude=.git .
```

Expected: archive is below 200 MiB and contains `src/`、`scripts/`、`tests/`、`docs/` and `outputs/`.

- [ ] **Step 3: 上传并原子展开代码**

Upload the archive to `/home/usertest/.staging/robust_metric_lab_nas.tar.gz`, extract into `/home/usertest/.staging/robust_metric_lab_nas`, verify required directories, then rename the staging directory to `/home/usertest/robust_metric_lab_nas`. Do not write into the final directory before staging validation succeeds.

- [ ] **Step 4: 从明确源容器复制 benchmark**

Run on the host through the `guzhengkai` account:

```bash
mkdir -p /home/usertest/robust_metric_lab_nas/shared_benchmarks
docker cp \
  huangyue:/root/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1 \
  /home/usertest/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1
```

Expected: only the named source path is read and only the destination project is written.

- [ ] **Step 5: 保存来源状态和文件清点**

Run Task 3's audit CLI inside `wangjiahao` and write to:

```text
/workspace/robust_metric_lab_nas/docs/experiments/source_snapshot/
```

Expected: original `IN_PROGRESS` state remains visible in `source_snapshot.json`.

### Task 8：建立服务器环境并运行材料验收

**Files:**
- Create on server: `/workspace/robust_metric_lab_nas/.venv/`
- Create: `docs/experiments/student_benchmark_environment.md`

**Interfaces:**
- Consumes: copied project, server CUDA/PyTorch installation, teacher verifier
- Produces: tested environment and preserved verifier log

- [ ] **Step 1: 检查已有 Python、CUDA 和依赖版本**

Run inside `wangjiahao`:

```bash
cd /workspace/robust_metric_lab_nas
python3 --version
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Record exact output in `docs/experiments/student_benchmark_environment.md`.

- [ ] **Step 2: 创建隔离环境并安装项目**

```bash
cd /workspace/robust_metric_lab_nas
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Expected: imports `torch`、`torchvision`、`robustbench`、`pandas`、`scipy` and `robust_metric_lab` succeed.

- [ ] **Step 3: 获取老师原版 verifier**

If `scripts/verify_student_benchmark_bundle.py` is absent from the local project copy, copy exactly that file from `huangyue:/root/robust_metric_lab_nas/scripts/verify_student_benchmark_bundle.py`, record its SHA-256 in source snapshot metadata, and do not edit it.

- [ ] **Step 4: 运行验收并保存真实退出码**

```bash
cd /workspace/robust_metric_lab_nas
. .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"
set +e
python scripts/verify_student_benchmark_bundle.py \
  --root "$PWD/shared_benchmarks/robust_metric_v1" \
  --manifest-dir "$PWD/shared_benchmarks/robust_metric_v1/manifests" \
  > docs/experiments/source_snapshot/verifier.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > docs/experiments/source_snapshot/verifier.exit_code
cat docs/experiments/source_snapshot/verifier.log
```

Expected: preserve actual result. A failure does not get relabeled as PASS.

- [ ] **Step 5: 运行本地单元测试**

Run: `pytest -q`

Expected: all tests pass before GPU smoke tests.

### Task 9：运行三空间 smoke test

**Files:**
- Create on server: `outputs/student_metrics/wangjiahao/smoke_v1/`
- Create: `docs/experiments/student_benchmark_smoke_report.md`

**Interfaces:**
- Consumes: verified per-space materials and Task 6 runners
- Produces: four-item outputs for RB40、NB201 e0/e50、DARTS e0/e100

- [ ] **Step 1: 固定 smoke 指标**

Use these five metrics because they exercise distinct paths:

```text
feature_delta
fsp_delta
perturbation_capacity_delta
jacobian_spectral_norm
trnas_r_score
```

- [ ] **Step 2: RB40 四模型 smoke**

Run `evaluate_rb40_student.py` with `--item-limit 4 --num-samples 64 --run-id smoke_v1` and the five repeated `--metric` flags.

Expected: four model IDs remain present; any failure row includes an error and does not abort the queue.

- [ ] **Step 3: NB201 两端点 smoke**

Run `evaluate_nb201_student.py` twice with epochs 0 and 50, `--item-limit 4` and the frozen sample arguments.

Expected: eight architecture/checkpoint groups, each with five metric rows.

- [ ] **Step 4: DARTS 两端点 smoke**

Run `evaluate_darts100_student.py` twice with epochs 0 and 100 and the same limits.

Expected: eight architecture/checkpoint groups, each with five metric rows.

- [ ] **Step 5: 验证 schema 和生成 smoke 报告**

Run Task 1 grid validation with smoke expected counts and Task 4 analysis. Record GPU peak memory, runtime, failures, normalization checks, sampled IDs and exact commands in `student_benchmark_smoke_report.md`.

- [ ] **Step 6: 提交 smoke 代码修复和报告**

Only commit source/test fixes and the Markdown report; do not commit model weights or large CSV outputs.

### Task 10：补齐现有指标并生成冻结总表

**Files:**
- Create on server: `outputs/student_metrics/wangjiahao/existing_metrics_full_v1/`
- Create: `docs/experiments/e25_plus_three_space_validation.md`

**Interfaces:**
- Consumes: all registered existing metrics, eight frozen settings
- Produces: complete `metric_values.csv`、`correlations.csv`、`e25_plus_coverage.csv`、`all_setting_table.csv`

- [ ] **Step 1: 冻结运行清单**

Export the current `METRIC_SPECS` names, directions, attributions and references to `run_manifest.json` before loading target columns. Include Git commit, environment versions, sample protocol and bundle snapshot hashes.

- [ ] **Step 2: 按 setting 顺序运行完整队列**

Run in this order to expose cheap failures early:

```text
NB201 e0 -> DARTS e0 -> RB40 mature -> NB201 e5 -> NB201 e10 ->
NB201 e25 -> NB201 e50 -> DARTS e100
```

Use append-safe outputs and resume only missing unique keys. Do not recompute successful rows.

- [ ] **Step 3: 验证完整网格**

Expected object counts are 40, 64, 64, 64, 64, 64, 100, 100. Every registered metric must have one explicit row per object per setting, either `ok` or `failed`.

- [ ] **Step 4: 生成统计和覆盖矩阵**

Run:

```bash
python scripts/analyze_student_benchmark.py \
  --input outputs/student_metrics/wangjiahao/existing_metrics_full_v1/metric_values.csv \
  --output-dir outputs/student_metrics/wangjiahao/existing_metrics_full_v1/analysis

python scripts/build_e25_coverage_matrix.py \
  --input outputs/student_metrics/wangjiahao/existing_metrics_full_v1/metric_values.csv \
  --output outputs/student_metrics/wangjiahao/existing_metrics_full_v1/analysis/e25_plus_coverage.csv
```

- [ ] **Step 5: 写正式报告**

`e25_plus_three_space_validation.md` must state the true bundle status, verifier result, failures, per-setting uncertainty, confound diagnostics, eight-setting ranking, and differences from the old E35 nine-setting table.

- [ ] **Step 6: 提交代码和报告**

```bash
git add docs/experiments/e25_plus_three_space_validation.md
git commit -m "docs: report frozen three-space validation"
```

### Task 11：条件式新指标阶段关卡

**Files:**
- Create only if triggered: `docs/superpowers/specs/2026-09-17-new-robust-metric-design.md`
- Create only after approval: a separate implementation plan for the frozen candidate family

**Interfaces:**
- Consumes: Task 10 complete ranking and development-only partitions
- Produces: either a documented stop decision or a separately approved new-metric design

- [ ] **Step 1: 检查是否已有指标进入前五**

If an existing Ours metric is top five by the valid eight-setting mean and does not fail the preregistered confound checks, stop metric creation and report success without adding candidates.

- [ ] **Step 2: 若未达到目标，冻结开发/留出划分**

Use `sha256(f"{benchmark}:{model_or_arch_id}:20260808")` to assign 70% development and 30% held-out entries independently per space. Persist assignments before computing candidate scores on held-out entries.

- [ ] **Step 3: 基于机制提出不超过三个候选**

Candidates must target the observed cross-space failure mechanism, have fixed lower/higher direction, add no target-trained coefficients, and reuse existing activations/gradients so compute cost remains comparable. Write exact formulas and ablations in the new design spec.

- [ ] **Step 4: 请求用户批准新指标设计**

Do not implement candidates before the separate design is approved. After approval, create a separate TDD implementation plan and run development evaluation before one held-out evaluation.

### Task 12：最终验证与交付

**Files:**
- Modify: `docs/experiments/实验管理.md`
- Create: `docs/experiments/student_three_space_final_report.md`

**Interfaces:**
- Consumes: all completed outputs and reports
- Produces: final reproducibility index and user-facing conclusion

- [ ] **Step 1: 运行完整测试和静态检查**

Run:

```bash
pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: 复核输出完整性**

Re-run grid validation, verify every report path exists, verify all CSVs parse, and compare `run_manifest.json` hashes with the copied bundle snapshot.

- [ ] **Step 3: 更新实验总控**

Add the new experiment ID, true status, eight-setting definition, exact source files, server location, commands, failures and conclusion to `docs/experiments/实验管理.md`.

- [ ] **Step 4: 写最终报告**

The final report must separate: verified facts, teacher-authorized integrity exception, existing-metric results, any conditional new-metric results, limitations, and exact reproduction commands.

- [ ] **Step 5: 提交最终文档**

```bash
git add docs/experiments/实验管理.md docs/experiments/student_three_space_final_report.md
git commit -m "docs: finalize student three-space benchmark"
```
