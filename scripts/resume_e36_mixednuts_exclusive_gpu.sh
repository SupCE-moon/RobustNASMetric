#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/workspace/robust_metric_lab_nas
PY="$PROJECT/.venv/bin/python"
OUTPUT_BASE="$PROJECT/outputs/student_metrics/wangjiahao"
RUN_ID=e36_full_metric_design_v1
RUN_DIR="$OUTPUT_BASE/$RUN_ID"
CSV="$RUN_DIR/metric_values.csv"
STATUS="$RUN_DIR/STATUS.env"
DATA="$PROJECT/data/cifar10"
OLD_TABLE="$OUTPUT_BASE/existing_metrics_full_v3_cpu_repair/analysis/all_setting_table.csv"
HELDOUT_CSV="$OUTPUT_BASE/heldout_rb12_20260922/e36_metrics_v1/metric_values.csv"
BUNDLE="$RUN_DIR/retry_bundles.20260929T105701Z/Bai2024MixedNUTS"

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

exec 9>"$RUN_DIR/.mixednuts_exclusive_gpu.lock"
flock -n 9 || { echo "MixedNUTS exclusive-GPU repair already running" >&2; exit 1; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir="$RUN_DIR/backups/mixednuts_exclusive_gpu.$stamp"
analysis_tmp="$RUN_DIR/analysis_rb40_complete.$stamp"
mkdir -p "$backup_dir" "$analysis_tmp"
cp -p "$CSV" "$backup_dir/metric_values.csv.before_retry"
cp -a "$RUN_DIR/analysis" "$backup_dir/analysis.before_retry"
cp -p "$STATUS" "$backup_dir/STATUS.env.before_retry"

"$PY" - "$CSV" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)
assert fields is not None
assert len(rows) == 4480, len(rows)
failed = [row for row in rows if row["status"] != "ok"]
assert len(failed) == 8, len(failed)
assert {row["model_or_arch_id"] for row in failed} == {"Bai2024MixedNUTS"}
assert all("OutOfMemoryError" in row["error"] for row in failed)
kept = [row for row in rows if row["status"] == "ok"]
assert len(kept) == 4472
tmp = path.with_suffix(path.suffix + ".mixednuts_retry.tmp")
if tmp.exists():
    tmp.unlink()
with tmp.open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(kept)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
print("preflight=PASS removed_mixednuts_oom_rows=8 kept_rows=4472")
PY

printf 'state=WAITING_EXCLUSIVE_GPU_FOR_MIXEDNUTS\nmixednuts_wait_started_utc=%s\nmixednuts_backup=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$backup_dir" >> "$STATUS"

stable=0
while test "$stable" -lt 3; do
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  printf 'gpu_wait_utc=%s setting=rb40:Bai2024MixedNUTS_exclusive used_mib=%s stable=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$used" "$stable" >> "$STATUS"
  if test "$used" -le 512; then stable=$((stable+1)); else stable=0; fi
  if test "$stable" -lt 3; then sleep 60; fi
done

printf 'state=RUNNING_MIXEDNUTS_EXCLUSIVE_GPU\nmixednuts_started_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
PYTHONPATH="$PROJECT/src:$PROJECT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" "$PROJECT/scripts/evaluate_rb40_student.py" \
  --bundle-root "$BUNDLE" --data-root "$DATA" --output-dir "$OUTPUT_BASE" \
  --run-id "$RUN_ID" --sample-offset 0 --num-samples 64 --seed 20260808 \
  --device cuda "${metric_args[@]}" > "$RUN_DIR/logs/mixednuts_exclusive_gpu.log" 2>&1

"$PY" - "$CSV" <<'PY'
import csv
import sys
from collections import Counter

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
keys = [
    (
        row["setting"], row["model_or_arch_id"], row["checkpoint_epoch"],
        row["metric_name"], row["seed"], row["sample_offset"], row["num_samples"],
    )
    for row in rows
]
assert len(rows) == 4480, len(rows)
assert len(keys) == len(set(keys)), (len(keys), len(set(keys)))
assert Counter(row["status"] for row in rows) == {"ok": 4480}
rb40 = [row for row in rows if row["setting"] == "rb40_mature"]
assert len(rb40) == 320
assert len({row["model_or_arch_id"] for row in rb40}) == 40
print("grid=PASS rows=4480 unique_keys=4480 ok=4480 rb40_rows=320 rb40_models=40")
PY

PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/analyze_student_benchmark.py" \
  --input "$CSV" --output-dir "$analysis_tmp"
PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/build_e25_coverage_matrix.py" \
  --input "$CSV" --output "$analysis_tmp/e36_coverage.csv"

"$PY" - "$OLD_TABLE" "$analysis_tmp/all_setting_table.csv" \
  "$analysis_tmp/comparison_all_metrics.csv" <<'PY'
import csv
import sys

old_path, new_path, output_path = sys.argv[1:]
def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
rows = read(old_path) + read(new_path)
complete = [row for row in rows if row["status"] == "complete" and row["all_setting_mean"]]
complete.sort(key=lambda row: float(row["all_setting_mean"]), reverse=True)
fields = ["rank"] + list(rows[0])
with open(output_path, "x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for rank, row in enumerate(complete, 1):
        writer.writerow({"rank": rank, **row})
print("complete_metrics", len(complete))
print("top5", [(row["metric_name"], row["all_setting_mean"]) for row in complete[:5]])
PY

PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/finalize_e36_heldout.py" \
  "$HELDOUT_CSV" "$analysis_tmp/correlations.csv" "$analysis_tmp/heldout_correlations.csv"

cp -p "$analysis_tmp/correlations.csv" "$RUN_DIR/analysis/correlations.csv"
cp -p "$analysis_tmp/all_setting_table.csv" "$RUN_DIR/analysis/all_setting_table.csv"
cp -p "$analysis_tmp/analysis_report.md" "$RUN_DIR/analysis/analysis_report.md"
cp -p "$analysis_tmp/e36_coverage.csv" "$RUN_DIR/analysis/e36_coverage.csv"
cp -p "$analysis_tmp/comparison_all_metrics.csv" "$RUN_DIR/analysis/comparison_all_metrics.csv"
cp -p "$analysis_tmp/heldout_correlations.csv" "$RUN_DIR/analysis/heldout_correlations.csv"

printf 'state=COMPLETE_RB40_FULL\nrows=4480\nok_rows=4480\nrb40_models=40\nrb40_repair_finished_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
echo "MIXEDNUTS_EXCLUSIVE_GPU_COMPLETE"
