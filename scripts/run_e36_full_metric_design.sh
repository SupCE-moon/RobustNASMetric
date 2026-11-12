#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/workspace/robust_metric_lab_nas
PY="$PROJECT/.venv/bin/python"
BUNDLE="$PROJECT/shared_benchmarks/robust_metric_v1"
DATA="$PROJECT/data/cifar10"
OUTPUT_BASE="$PROJECT/outputs/student_metrics/wangjiahao"
RUN_ID=e36_full_metric_design_v1
RUN_DIR="$OUTPUT_BASE/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
STATUS="$RUN_DIR/STATUS.env"
CSV="$RUN_DIR/metric_values.csv"
ANALYSIS="$RUN_DIR/analysis"
SMOKE_ID=e36_metric_smoke_v1
SMOKE_DIR="$OUTPUT_BASE/$SMOKE_ID"
HELDOUT_ROOT="$OUTPUT_BASE/heldout_rb12_20260922"
HELDOUT_RUN=e36_metrics_v1
HELDOUT_CSV="$HELDOUT_ROOT/$HELDOUT_RUN/metric_values.csv"

METRICS=(
  feature_measure_channel_gram_cosine_delta
  feature_measure_channel_gram_centered_rel_delta
  feature_measure_spatial_gram_cosine_delta
  feature_measure_spectrum_js_delta
  feature_measure_spectrum_emd_delta
  feature_measure_spectrum_hellinger_delta
  feature_measure_flow_corr_angular_delta
  feature_measure_flow_corr_symmetric_rel_delta
)
metric_args=()
for metric in "${METRICS[@]}"; do metric_args+=(--metric "$metric"); done

finish() {
  rc=$?
  printf 'process_exit=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS" 2>/dev/null || true
  exit "$rc"
}
trap finish EXIT

test ! -e "$STATUS"
test ! -e "$CSV"
test ! -e "$SMOKE_DIR"
test ! -e "$HELDOUT_ROOT/$HELDOUT_RUN"
mkdir -p "$LOG_DIR" "$ANALYSIS"
cp "$0" "$RUN_DIR/run.sh"
chmod 700 "$RUN_DIR/run.sh"
printf 'state=CREATING_MANIFEST\nstarted_utc=%s\nmetric_count=%s\nexpected_rows=4480\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#METRICS[@]}" > "$STATUS"

PROJECT="$PROJECT" RUN_DIR="$RUN_DIR" "$PY" - <<'PY_EOF'
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import torch

from robust_metric_lab.metric_registry import METRIC_BY_NAME

names = [
    "feature_measure_channel_gram_cosine_delta",
    "feature_measure_channel_gram_centered_rel_delta",
    "feature_measure_spatial_gram_cosine_delta",
    "feature_measure_spectrum_js_delta",
    "feature_measure_spectrum_emd_delta",
    "feature_measure_spectrum_hellinger_delta",
    "feature_measure_flow_corr_angular_delta",
    "feature_measure_flow_corr_symmetric_rel_delta",
]
project = Path(os.environ["PROJECT"])
run_dir = Path(os.environ["RUN_DIR"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": "e36_full_metric_design_v1",
    "experiment": "E36 pre-registered relative-structure candidates",
    "metrics": [asdict(METRIC_BY_NAME[name]) for name in names],
    "metric_count": len(names),
    "settings": [
        "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
        "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
    ],
    "expected_objects": 560,
    "expected_rows": 4480,
    "protocol": {
        "seed": 20260808,
        "sample_offset": 0,
        "num_samples": 64,
        "epsilon": 8 / 255,
        "directions_frozen_before_targets": True,
    },
    "source_sha256": {
        "metrics.py": sha256(project / "src/robust_metric_lab/metrics.py"),
        "metric_registry.py": sha256(project / "src/robust_metric_lab/metric_registry.py"),
        "test_e36_relative_structure_metrics.py": sha256(
            project / "tests/test_e36_relative_structure_metrics.py"
        ),
    },
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "robustbench": importlib.metadata.version("robustbench"),
    },
    "heldout": {
        "bundle": "heldout_rb12_20260922/bundle",
        "declared_corrupt_model": "Rebuffi2021Fixing_70_16_cutmix_ddpm",
        "valid_models_expected": 10,
    },
}
target = run_dir / "run_manifest.json"
tmp = target.with_suffix(".json.tmp")
with tmp.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, target)
PY_EOF
printf 'run_manifest_sha256=%s\n' "$(sha256sum "$RUN_DIR/run_manifest.json" | awk '{print $1}')" >> "$STATUS"

wait_for_gpu() {
  local setting=$1 stable=0 used
  printf 'state=WAITING_GPU\nsetting=%s\n' "$setting" >> "$STATUS"
  while test "$stable" -lt 3; do
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    printf 'gpu_wait_utc=%s setting=%s used_mib=%s stable=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$setting" "$used" "$stable" >> "$STATUS"
    if test "$used" -le 6000; then stable=$((stable+1)); else stable=0; fi
    if test "$stable" -lt 3; then sleep 60; fi
  done
}

run_logged() {
  local phase=$1 setting=$2
  shift 2
  wait_for_gpu "$phase:$setting"
  printf 'state=RUNNING\nphase=%s\nsetting=%s\nsetting_started_utc=%s\n' \
    "$phase" "$setting" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  printf '%q ' "$@" > "$LOG_DIR/$phase.$setting.command.txt"
  printf '\n' >> "$LOG_DIR/$phase.$setting.command.txt"
  set +e
  PYTHONPATH="$PROJECT/src:$PROJECT" "$@" > "$LOG_DIR/$phase.$setting.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    rows=0
    if test "$phase" = full && test -f "$CSV"; then rows=$(($(wc -l < "$CSV") - 1)); fi
    printf 'heartbeat_utc=%s phase=%s setting=%s rows=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$setting" "$rows" >> "$STATUS"
    sleep 60
  done
  wait "$pid"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$LOG_DIR/$phase.$setting.exit_code"
  printf 'setting_finished_utc=%s phase=%s setting=%s exit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$setting" "$rc" >> "$STATUS"
  test "$rc" -eq 0
}

smoke_common=(--bundle-root "$BUNDLE" --data-root "$DATA" --output-dir "$OUTPUT_BASE" \
  --run-id "$SMOKE_ID" --sample-offset 0 --num-samples 8 --seed 20260808 \
  --item-limit 1 --device cuda)
run_logged smoke nb201_e0 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 0 "${smoke_common[@]}" "${metric_args[@]}"
run_logged smoke darts_e0 "$PY" "$PROJECT/scripts/evaluate_darts100_student.py" --epoch 0 "${smoke_common[@]}" "${metric_args[@]}"
run_logged smoke rb40_mature "$PY" "$PROJECT/scripts/evaluate_rb40_student.py" "${smoke_common[@]}" "${metric_args[@]}"
run_logged smoke darts_e100 "$PY" "$PROJECT/scripts/evaluate_darts100_student.py" --epoch 100 "${smoke_common[@]}" "${metric_args[@]}"

"$PY" - "$SMOKE_DIR/metric_values.csv" <<'PY_EOF' > "$RUN_DIR/smoke_validation.log"
import csv, collections, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
keys = [(r["setting"], r["model_or_arch_id"], r["metric_name"]) for r in rows]
assert len(rows) == 32, len(rows)
assert len(keys) == len(set(keys))
assert collections.Counter(r["setting"] for r in rows) == {
    "nb201_e0": 8, "darts_e0": 8, "rb40_mature": 8, "darts_e100": 8,
}
failures = [r for r in rows if r["status"] != "ok"]
assert not failures, failures
print("smoke_gate=PASS rows=32 unique_keys=32")
PY_EOF
printf 'smoke_gate=PASS\n' >> "$STATUS"

full_common=(--bundle-root "$BUNDLE" --data-root "$DATA" --output-dir "$OUTPUT_BASE" \
  --run-id "$RUN_ID" --sample-offset 0 --num-samples 64 --seed 20260808 --device cuda)
run_logged full nb201_e0 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 0 "${full_common[@]}" "${metric_args[@]}"
run_logged full darts_e0 "$PY" "$PROJECT/scripts/evaluate_darts100_student.py" --epoch 0 "${full_common[@]}" "${metric_args[@]}"
run_logged full rb40_mature "$PY" "$PROJECT/scripts/evaluate_rb40_student.py" "${full_common[@]}" "${metric_args[@]}"
run_logged full nb201_e5 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 5 "${full_common[@]}" "${metric_args[@]}"
run_logged full nb201_e10 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 10 "${full_common[@]}" "${metric_args[@]}"
run_logged full nb201_e25 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 25 "${full_common[@]}" "${metric_args[@]}"
run_logged full nb201_e50 "$PY" "$PROJECT/scripts/evaluate_nb201_student.py" --epoch 50 "${full_common[@]}" "${metric_args[@]}"
run_logged full darts_e100 "$PY" "$PROJECT/scripts/evaluate_darts100_student.py" --epoch 100 "${full_common[@]}" "${metric_args[@]}"

"$PY" - "$CSV" <<'PY_EOF' > "$RUN_DIR/grid_validation.log"
import collections, csv, sys
expected_objects = {
    "rb40_mature": 40, "nb201_e0": 64, "nb201_e5": 64,
    "nb201_e10": 64, "nb201_e25": 64, "nb201_e50": 64,
    "darts_e0": 100, "darts_e100": 100,
}
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 4480, len(rows)
keys = [
    (r["setting"], r["model_or_arch_id"], r["checkpoint_epoch"],
     r["metric_name"], r["seed"], r["sample_offset"], r["num_samples"])
    for r in rows
]
assert len(keys) == len(set(keys))
counts = collections.Counter(r["setting"] for r in rows)
assert counts == {name: count * 8 for name, count in expected_objects.items()}, counts
statuses = collections.Counter(r["status"] for r in rows)
failures = collections.Counter(r["metric_name"] for r in rows if r["status"] != "ok")
print("rows", len(rows), "unique_keys", len(set(keys)))
print("setting_counts", dict(counts))
print("status_counts", dict(statuses))
print("failure_metrics", dict(failures))
PY_EOF

PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/analyze_student_benchmark.py" \
  --input "$CSV" --output-dir "$ANALYSIS" > "$LOG_DIR/analysis.log" 2>&1
PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/build_e25_coverage_matrix.py" \
  --input "$CSV" --output "$ANALYSIS/e36_coverage.csv" > "$LOG_DIR/coverage.log" 2>&1

OLD_TABLE="$OUTPUT_BASE/existing_metrics_full_v3_cpu_repair/analysis/all_setting_table.csv"
"$PY" - "$OLD_TABLE" "$ANALYSIS/all_setting_table.csv" "$ANALYSIS/comparison_all_metrics.csv" <<'PY_EOF'
import csv, sys
old_path, new_path, output_path = sys.argv[1:]
def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
rows = read(old_path) + read(new_path)
complete = [r for r in rows if r["status"] == "complete" and r["all_setting_mean"]]
complete.sort(key=lambda r: float(r["all_setting_mean"]), reverse=True)
fields = ["rank"] + list(rows[0])
with open(output_path, "x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for rank, row in enumerate(complete, 1):
        writer.writerow({"rank": rank, **row})
print("complete_metrics", len(complete), "top5", [(r["metric_name"], r["all_setting_mean"]) for r in complete[:5]])
PY_EOF

wait_for_gpu heldout_rb11
printf 'state=RUNNING_HELDOUT\nheldout_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/evaluate_rb40_student.py" \
  --bundle-root "$HELDOUT_ROOT/bundle" --data-root "$DATA" --output-dir "$HELDOUT_ROOT" \
  --run-id "$HELDOUT_RUN" --sample-offset 0 --num-samples 64 --seed 20260808 \
  --device cuda "${metric_args[@]}" > "$LOG_DIR/heldout_rb11.log" 2>&1

"$PY" - "$HELDOUT_CSV" "$ANALYSIS/correlations.csv" "$ANALYSIS/heldout_correlations.csv" <<'PY_EOF'
import csv, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr

input_path, original_path, output_path = sys.argv[1:]
with open(input_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
keys = [(r["model_or_arch_id"], r["metric_name"]) for r in rows]
assert len(rows) == 88, len(rows)
assert len(keys) == len(set(keys))
failures = [r for r in rows if r["status"] != "ok"]
assert len(failures) == 8, len(failures)
assert Counter(r["model_or_arch_id"] for r in failures) == {
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 8
}
with open(original_path, newline="", encoding="utf-8") as handle:
    original_rows = list(csv.DictReader(handle))
original = {
    r["metric_name"]: float(r["direction_adjusted_spearman"])
    for r in original_rows if r["setting"] == "rb40_mature" and r["status"] == "ok"
}
by_metric = defaultdict(list)
for row in rows:
    if row["status"] == "ok":
        by_metric[row["metric_name"]].append(row)
rng = np.random.default_rng(20260927)
summary = []
for metric, valid in sorted(by_metric.items()):
    assert len(valid) == 10, (metric, len(valid))
    sign = -1.0 if valid[0]["direction"] == "lower" else 1.0
    x = np.asarray([float(r["metric_value"]) for r in valid])
    y = np.asarray([float(r["target_value"]) for r in valid])
    rho = float(spearmanr(sign * x, y).statistic)
    boots = []
    for _ in range(20_000):
        idx = rng.integers(0, len(valid), size=len(valid))
        value = float(spearmanr(sign * x[idx], y[idx]).statistic)
        if np.isfinite(value):
            boots.append(value)
    low, high = np.percentile(boots, [2.5, 97.5])
    summary.append({
        "metric_name": metric,
        "valid_models": 10,
        "failed_models": 1,
        "heldout_direction_adjusted_spearman": rho,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_repeats": 20_000,
        "original_rb40_direction_adjusted_spearman": original[metric],
        "heldout_minus_original": rho - original[metric],
    })
summary.sort(key=lambda r: r["heldout_direction_adjusted_spearman"], reverse=True)
with open(output_path, "x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print("heldout_rows", len(rows), "valid_per_metric", 10, "failed_rows", len(failures))
print("heldout_ranking", [(r["metric_name"], r["heldout_direction_adjusted_spearman"]) for r in summary])
PY_EOF

printf 'state=COMPLETE\nrows=4480\ncomparison=%s\nheldout_rows=88\n' \
  "$ANALYSIS/comparison_all_metrics.csv" >> "$STATUS"
