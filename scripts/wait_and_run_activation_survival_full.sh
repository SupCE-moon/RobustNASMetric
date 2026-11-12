#!/usr/bin/env bash
set -Eeuo pipefail

project=/workspace/robust_metric_lab_nas
run="$project/outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z"
launcher="$run/full_launcher"
mkdir "$launcher"
status="$launcher/STATUS.env"
trap 'printf "state=FAILED\nfailed_utc=%s\n" "$(date -u +%FT%TZ)" >> "$status"' ERR
printf 'state=WAITING_SMOKE\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$status"

while true; do
  smoke_status="$run/smoke_launcher/STATUS.env"
  if test -f "$smoke_status" && grep -q '^state=FAILED$' "$smoke_status"; then
    printf 'state=FAILED\nreason=smoke_failed\nfailed_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
    exit 1
  fi
  if test -f "$smoke_status" && grep -q '^state=SMOKE_PASS$' "$smoke_status"; then
    break
  fi
  sleep 60
done

cd "$project"
python - <<'PY'
import hashlib
import json
from pathlib import Path

from scripts.activation_survival_manifest import _canonical, verify_source_inventory

root = Path('/workspace/robust_metric_lab_nas')
run = root / 'outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z'
manifest = json.loads((run / 'frozen_manifest.json').read_text())
payload = manifest['frozen_payload']
smoke = json.loads((run / 'smoke/smoke_verification.json').read_text())
assert hashlib.sha256(_canonical(payload)).hexdigest() == manifest['frozen_sha256']
assert len(payload['objects']) == 560 and payload['expected_rows'] == 560
assert smoke['status'] == 'PASS' and smoke['manifest_sha256'] == manifest['frozen_sha256']
assert smoke['successful_settings'] == ['rb40_mature', 'nb201_e0', 'darts_e0']
verify_source_inventory(payload['source_sha256'], root)
PY

printf 'state=WAITING_GPU\n' >> "$status"
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

run_one() {
  local setting="$1" benchmark="$2" epoch="${3:-}"
  local command=(python scripts/evaluate_activation_survival_student.py
    --benchmark "$benchmark"
    --bundle-root shared_benchmarks/robust_metric_v1
    --data-root data/cifar10
    --manifest "$run/frozen_manifest.json"
    --attempts-dir "$run/attempts/$setting"
    --device cuda)
  if test -n "$epoch"; then command+=(--epoch "$epoch"); fi
  printf 'state=RUNNING\nsetting=%s\nstarted_setting_utc=%s\n' \
    "$setting" "$(date -u +%FT%TZ)" >> "$status"
  "${command[@]}" > "$launcher/$setting.log" 2>&1
  printf 'state=SETTING_COMPLETE\nsetting=%s\nfinished_setting_utc=%s\n' \
    "$setting" "$(date -u +%FT%TZ)" >> "$status"
}

run_one rb40_mature rb40
run_one nb201_e0 nb201 0
run_one nb201_e5 nb201 5
run_one nb201_e10 nb201 10
run_one nb201_e25 nb201 25
run_one nb201_e50 nb201 50
run_one darts_e0 darts 0
run_one darts_e100 darts 100

printf 'state=ALL_SETTINGS_RUN_COMPLETE\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$status"
