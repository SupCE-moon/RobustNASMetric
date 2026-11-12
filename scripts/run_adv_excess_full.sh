#!/usr/bin/env bash
set -euo pipefail

project=/workspace/robust_metric_lab_nas
bundle="$project/shared_benchmarks/robust_metric_v1"
data_root=/workspace/DynamicBN20251203/data
output_base="$project/outputs/student_metrics/wangjiahao"
mode="${1:-full}"
device="${DEVICE:-cuda}"
batch_size=1

case "$mode" in
  smoke) expected_rows=32; item_args=(--item-limit 4) ;;
  full) expected_rows=560; item_args=() ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

run_id="${RUN_ID:-adv_excess_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "unsafe RUN_ID" >&2
  exit 2
fi
run_dir="$output_base/$run_id"
mkdir -p "$run_dir/logs"
export PYTHONPATH="$project/src:$project"

"$project/.venv/bin/python" "$project/scripts/adv_excess_run_manifest.py" \
  --mode "$mode" --project "$project" --bundle "$bundle" --run-dir "$run_dir" \
  --device "$device" --batch-size "$batch_size" \
  2>&1 | tee "$run_dir/logs/manifest.log"

printf 'state=WAITING_FOR_GPU\nmode=%s\nexpected_rows=%s\n' \
  "$mode" "$expected_rows" >> "$run_dir/STATUS.env"

record_failure() {
  local result=$?
  if (( result != 0 )); then
    printf 'state=FAILED\nexit_code=%s\nfailed_utc=%s\n' \
      "$result" "$(date -u +%FT%TZ)" >> "$run_dir/STATUS.env"
  fi
}
trap record_failure EXIT

wait_for_gpu() {
  if [[ "$device" == cpu ]]; then
    return 0
  fi
  local stable=0 used_mib
  while (( stable < 3 )); do
    if ! used_mib="$(nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>>"$run_dir/logs/gpu_gate.log" |
      awk '{sum += $1} END {print sum + 0}')"; then
      printf 'gpu_gate_utc=%s query_failed stable=0\n' \
        "$(date -u +%FT%TZ)" >> "$run_dir/STATUS.env"
      stable=0
      sleep 60
      continue
    fi
    printf 'gpu_gate_utc=%s used_mib=%s stable=%s\n' \
      "$(date -u +%FT%TZ)" "$used_mib" "$stable" >> "$run_dir/STATUS.env"
    if (( used_mib <= 1000 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    if (( stable < 3 )); then
      sleep 60
    fi
  done
}

run_setting() {
  local label="$1"
  shift
  wait_for_gpu
  printf 'state=RUNNING\nsetting=%s\nstarted_utc=%s\n' \
    "$label" "$(date -u +%FT%TZ)" >> "$run_dir/STATUS.env"
  "$project/.venv/bin/python" "$project/scripts/evaluate_adv_excess_student.py" \
    "$@" --bundle-root "$bundle" --data-root "$data_root" \
    --output-dir "$output_base" --run-id "$run_id" \
    --batch-size "$batch_size" --device "$device" "${item_args[@]}" \
    2>&1 | tee "$run_dir/logs/${label}.log"
}

run_setting rb40_mature --benchmark rb40
for epoch in 0 5 10 25 50; do
  run_setting "nb201_e$epoch" --benchmark nb201 --epoch "$epoch"
done
for epoch in 0 100; do
  run_setting "darts_e$epoch" --benchmark darts --epoch "$epoch"
done

CSV="$run_dir/metric_values.csv" MODE="$mode" EXPECTED_ROWS="$expected_rows" \
  "$project/.venv/bin/python" - <<'PY'
import csv
import math
import os
from collections import Counter
from robust_metric_lab.student_runner import UNIQUE_KEY_COLUMNS

with open(os.environ["CSV"], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected = int(os.environ["EXPECTED_ROWS"])
if len(rows) != expected:
    raise ValueError(f"raw row count {len(rows)} != {expected}")
keys = {tuple(row[column] for column in UNIQUE_KEY_COLUMNS) for row in rows}
if len(keys) != expected:
    raise ValueError("duplicate raw key")
if any(row["metric_name"] != "feature_adv_excess_centered_unit_l2" or
       row["direction"] != "lower" or row["status"] != "ok" or
       not math.isfinite(float(row["metric_value"])) for row in rows):
    raise ValueError("raw grid contains a failed, nonfinite, or wrong-metric row")
counts = Counter(row["setting"] for row in rows)
want = {"rb40_mature": 40, **{f"nb201_e{epoch}": 64 for epoch in (0, 5, 10, 25, 50)},
        **{f"darts_e{epoch}": 100 for epoch in (0, 100)}}
if os.environ["MODE"] == "smoke":
    want = {setting: 4 for setting in want}
if counts != want:
    raise ValueError(f"setting counts mismatch: {counts} != {want}")
print(f"raw_grid_verified={expected}")
PY

printf 'state=RAW_GRID_OK\nrows=%s\nfinished_utc=%s\n' \
  "$expected_rows" "$(date -u +%FT%TZ)" >> "$run_dir/STATUS.env"
