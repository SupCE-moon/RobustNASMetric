#!/usr/bin/env bash
set -euo pipefail

project=/workspace/robust_metric_lab_nas
status_dir="$project/outputs/student_metrics/wangjiahao/strict_e37_smoke_launcher"
mkdir -p "$status_dir"
stable=0
while (( stable < 3 )); do
  used_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
  printf 'wait_utc=%s used_mib=%s stable=%s\n' "$(date -u +%FT%TZ)" "$used_mib" "$stable" >> "$status_dir/STATUS.env"
  if (( used_mib <= 1000 )); then stable=$((stable + 1)); else stable=0; fi
  (( stable < 3 )) && sleep 60
done
cd "$project"
printf 'state=RUNNING_IMPORTANCE_SMOKE\n' >> "$status_dir/STATUS.env"
RUN_ID=strict_importance_smoke_e37_v1 bash scripts/run_strict_all_metrics.sh smoke
printf 'state=RUNNING_E37_SMOKE\n' >> "$status_dir/STATUS.env"
RUN_ID=e37_smoke_strict_v1 bash scripts/run_e37_full.sh smoke
printf 'state=SMOKE_COMPLETE\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status_dir/STATUS.env"
printf 'state=RUNNING_IMPORTANCE_FULL\n' >> "$status_dir/STATUS.env"
RUN_ID=strict_importance_v1 bash scripts/run_strict_all_metrics.sh full
printf 'state=RUNNING_E37_FULL\n' >> "$status_dir/STATUS.env"
RUN_ID=e37_full_metric_design_v1 bash scripts/run_e37_full.sh full
printf 'state=COMPLETE_MAIN_RUNS\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status_dir/STATUS.env"
