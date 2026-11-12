#!/usr/bin/env bash
set -euo pipefail

project=/workspace/robust_metric_lab_nas
run="$project/outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z"
launcher="$run/smoke_launcher"
mkdir "$launcher"
status="$launcher/STATUS.env"
trap 'printf "state=FAILED\nfailed_utc=%s\n" "$(date -u +%FT%TZ)" >> "$status"' ERR
printf 'state=WAITING_GPU\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$status"

stable=0
while (( stable < 3 )); do
  used_output="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits)"
  used_mib="$(printf '%s\n' "$used_output" | awk '{sum += $1} END {print sum + 0}')"
  printf 'wait_utc=%s used_mib=%s stable_before=%s\n' \
    "$(date -u +%FT%TZ)" "$used_mib" "$stable" >> "$status"
  if (( used_mib <= 1000 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  if (( stable < 3 )); then sleep 60; fi
done

printf 'state=RUNNING_SMOKE\nstarted_smoke_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
cd "$project"
python scripts/smoke_activation_survival.py \
  --bundle-root shared_benchmarks/robust_metric_v1 \
  --data-root data/cifar10 \
  --manifest outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z/frozen_manifest.json \
  --output-dir outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z/smoke
printf 'state=SMOKE_PASS\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
