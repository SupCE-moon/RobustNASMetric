#!/usr/bin/env bash
set -Eeuo pipefail

project=/workspace/robust_metric_lab_nas
run="$project/outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z"
release="$project/outputs/student_metrics/wangjiahao/activation_sign_survival_strict_release_from_20261030T153734Z"
launcher="$run/release_launcher"
mkdir "$launcher"
status="$launcher/STATUS.env"
trap 'printf "state=FAILED\nfailed_utc=%s\n" "$(date -u +%FT%TZ)" >> "$status"' ERR
printf 'state=WAITING_VERIFIED_560\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$status"

while true; do
  verification_status="$run/verification_launcher/STATUS.env"
  if test -f "$verification_status" && grep -q '^state=FAILED$' "$verification_status"; then
    printf 'state=FAILED\nreason=candidate_verification_failed\nfailed_utc=%s\n' \
      "$(date -u +%FT%TZ)" >> "$status"
    exit 1
  fi
  if test -f "$verification_status" && grep -q '^state=CANDIDATE_VERIFIED$' "$verification_status"; then
    break
  fi
  sleep 60
done

cd "$project"
printf 'state=BUILDING_STRICT_RELEASE\n' >> "$status"
ionice -c3 python scripts/build_activation_survival_strict_release.py \
  --previous-release outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z \
  --candidate-release "$run/candidate_release" \
  --output-dir "$release" \
  --bundle-root shared_benchmarks/robust_metric_v1 > "$launcher/build.log" 2>&1

printf 'state=ANALYZING_PAIRED_PHI\n' >> "$status"
ionice -c3 python scripts/analyze_activation_survival.py \
  --candidate-csv "$run/candidate_release/metric_values.csv" \
  --strict-csv outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z/metric_values_merged.csv \
  --output-json "$run/analysis_main/paired_phi.json" > "$launcher/analysis.log" 2>&1

printf 'state=STRICT_RELEASE_PASS\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
