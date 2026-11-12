#!/usr/bin/env bash
set -Eeuo pipefail

project=/workspace/robust_metric_lab_nas
run="$project/outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z"
launcher="$run/verification_launcher"
mkdir "$launcher"
status="$launcher/STATUS.env"
trap 'printf "state=FAILED\nfailed_utc=%s\n" "$(date -u +%FT%TZ)" >> "$status"' ERR
printf 'state=WAITING_FULL_GRID\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$status"

while true; do
  full_status="$run/full_launcher/STATUS.env"
  if test -f "$full_status" && grep -q '^state=FAILED$' "$full_status"; then
    printf 'state=FAILED\nreason=full_grid_failed\nfailed_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
    exit 1
  fi
  if test -f "$full_status" && grep -q '^state=ALL_SETTINGS_RUN_COMPLETE$' "$full_status"; then
    break
  fi
  sleep 60
done

cd "$project"
printf 'state=COLLECTING_ATTEMPTS\n' >> "$status"
ionice -c3 python scripts/collect_activation_survival_attempts.py \
  --manifest "$run/frozen_manifest.json" \
  --attempts-root "$run/attempts" \
  --output-dir "$run/attempts_flat" > "$launcher/collection.log" 2>&1

printf 'state=VERIFYING_ALL_INPUTS\n' >> "$status"
ionice -c3 python scripts/verify_activation_survival.py \
  --manifest "$run/frozen_manifest.json" \
  --attempts-dir "$run/attempts_flat" \
  --output-dir "$run/candidate_release" > "$launcher/verification.log" 2>&1

printf 'state=CANDIDATE_VERIFIED\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
