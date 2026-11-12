#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/workspace/robust_metric_lab_nas
PY="$PROJECT/.venv/bin/python"
SOURCE_BUNDLE="$PROJECT/shared_benchmarks/robust_metric_v1"
DATA="$PROJECT/data/cifar10"
OUTPUT_BASE="$PROJECT/outputs/student_metrics/wangjiahao"
RUN_ID=e36_full_metric_design_v1
RUN_DIR="$OUTPUT_BASE/$RUN_ID"
CSV="$RUN_DIR/metric_values.csv"
STATUS="$RUN_DIR/STATUS.env"
OLD_TABLE="$OUTPUT_BASE/existing_metrics_full_v3_cpu_repair/analysis/all_setting_table.csv"
HELDOUT_CSV="$OUTPUT_BASE/heldout_rb12_20260922/e36_metrics_v1/metric_values.csv"

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

exec 9>"$RUN_DIR/.rb40_remaining_cpu.lock"
flock -n 9 || { echo "remaining-model repair already running" >&2; exit 1; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir="$RUN_DIR/backups/rb40_remaining_cpu.$stamp"
retry_root="$RUN_DIR/retry_bundles.$stamp"
analysis_tmp="$RUN_DIR/analysis_rb40_complete.$stamp"
mkdir -p "$backup_dir" "$retry_root" "$analysis_tmp"
cp -p "$CSV" "$backup_dir/metric_values.csv.before_retry"
cp -a "$RUN_DIR/analysis" "$backup_dir/analysis.before_retry"
cp -p "$STATUS" "$backup_dir/STATUS.env.before_retry"

printf 'state=RUNNING_RB40_REMAINING_CPU\nrb40_remaining_started_utc=%s\nrb40_remaining_backup=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$backup_dir" >> "$STATUS"

"$PY" - "$CSV" "$SOURCE_BUNDLE" "$retry_root" <<'PY'
import csv
import os
import sys
from pathlib import Path

csv_path, source_root, retry_root = map(Path, sys.argv[1:])
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 4456, len(rows)
assert all(row["status"] == "ok" for row in rows)
keys = [
    (
        row["setting"], row["model_or_arch_id"], row["checkpoint_epoch"],
        row["metric_name"], row["seed"], row["sample_offset"], row["num_samples"],
    )
    for row in rows
]
assert len(keys) == len(set(keys))

manifest_path = source_root / "manifests/rb40_models.csv"
with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    manifest_rows = list(reader)
assert fields is not None
present = {
    row["model_or_arch_id"] for row in rows if row["setting"] == "rb40_mature"
}
all_models = {row["model_name"] for row in manifest_rows}
missing = sorted(all_models - present)
expected = {
    "Amini2024MeanSparse_S-WRN-94-16",
    "Bai2024MixedNUTS",
    "Bartoldson2024Adversarial_WRN-94-16",
}
assert set(missing) == expected, missing

source_models = source_root / "rb40/models/cifar10/Linf"
for model_name in missing:
    row = next(row for row in manifest_rows if row["model_name"] == model_name)
    bundle = retry_root / model_name
    manifest_dir = bundle / "manifests"
    model_dir = bundle / "rb40/models/cifar10/Linf"
    manifest_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    with (manifest_dir / "rb40_models.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    artifacts = [name.strip() for name in row["artifact_files"].split(";") if name.strip()]
    assert artifacts, model_name
    for name in artifacts:
        source = source_models / name
        target = model_dir / name
        assert source.is_file(), source
        os.link(source, target)
print("preflight=PASS existing_rows=4456 remaining_models=" + ",".join(missing))
PY

for bundle in "$retry_root"/*; do
  model=$(basename "$bundle")
  log="$RUN_DIR/logs/rb40_remaining_cpu.$model.log"
  printf 'rb40_model_started_utc=%s model=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" >> "$STATUS"
  PYTHONPATH="$PROJECT/src:$PROJECT" "$PY" "$PROJECT/scripts/evaluate_rb40_student.py" \
    --bundle-root "$bundle" --data-root "$DATA" --output-dir "$OUTPUT_BASE" \
    --run-id "$RUN_ID" --sample-offset 0 --num-samples 64 --seed 20260808 \
    --device cpu "${metric_args[@]}" > "$log" 2>&1
  printf 'rb40_model_finished_utc=%s model=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" >> "$STATUS"
done

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
assert len(rb40) == 320, len(rb40)
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
echo "RB40_REMAINING_CPU_COMPLETE"
