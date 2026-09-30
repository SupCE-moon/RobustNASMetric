#!/usr/bin/env bash
set -euo pipefail

project=/workspace/robust_metric_lab_nas
bundle="$project/shared_benchmarks/robust_metric_v1"
data_root=/workspace/DynamicBN20251203/data
output_base="$project/outputs/student_metrics/wangjiahao"
mode="${1:-full}"
device="${DEVICE:-cuda}"
batch_size="${BATCH_SIZE:-1}"

case "$mode" in
  full) run_id="${RUN_ID:-e37_full_metric_design_v1}"; item_args=(); expected_rows=5040 ;;
  smoke) run_id="${RUN_ID:-e37_smoke_v1}"; item_args=(--item-limit 4); expected_rows=288 ;;
  *) echo "usage: $0 [full|smoke]" >&2; exit 2 ;;
esac

run_dir="$output_base/$run_id"
mkdir -p "$run_dir/logs"
if [[ ! -e "$run_dir/run_manifest.json" ]]; then
  PROJECT="$project" BUNDLE="$bundle" RUN_DIR="$run_dir" RUN_ID="$run_id" MODE="$mode" EXPECTED_ROWS="$expected_rows" DEVICE="$device" BATCH_SIZE="$batch_size" \
  "$project/.venv/bin/python" - <<'PY'
from datetime import datetime, timezone
import hashlib, json, os, platform, subprocess
from pathlib import Path
import torch
from robust_metric_lab.e37_metrics import E37_METRIC_NAMES
from robust_metric_lab.student_benchmark import EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED, load_darts_items, load_nb201_items, load_rb40_items

project=Path(os.environ["PROJECT"]); bundle=Path(os.environ["BUNDLE"]); run_dir=Path(os.environ["RUN_DIR"])
paths=[project/"scripts/evaluate_e37_student.py",project/"src/robust_metric_lab/e37_metrics.py",project/"src/robust_metric_lab/e37_batched.py",project/"src/robust_metric_lab/metric_registry.py",project/"configs/metric_eligibility.json",bundle/"manifests/rb40_models.csv",bundle/"manifests/nb201_64_architectures.csv",bundle/"manifests/darts100_architectures.csv"]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
rb=load_rb40_items(bundle/"manifests/rb40_models.csv",bundle); nb=load_nb201_items(bundle/"manifests/nb201_64_architectures.csv",bundle); da=load_darts_items(bundle/"manifests/darts100_architectures.csv",bundle)
limit=4 if os.environ["MODE"]=="smoke" else None
objects={"rb40_mature":[x.model_or_arch_id for x in rb[:limit]],**{f"nb201_e{e}":[x.model_or_arch_id for x in [y for y in nb if y.checkpoint_epoch==e][:limit]] for e in (0,5,10,25,50)},**{f"darts_e{e}":[x.model_or_arch_id for x in [y for y in da if y.checkpoint_epoch==e][:limit]] for e in (0,100)}}
payload={"created_at_utc":datetime.now(timezone.utc).isoformat(),"run_id":os.environ["RUN_ID"],"mode":os.environ["MODE"],"metric_names":list(E37_METRIC_NAMES),"directions":{n:"lower" for n in E37_METRIC_NAMES},"formula_family":{"flow":"successive-anchor FSP correlation drift aggregated by worst/RMS/top-half mean","spectrum":"per-anchor normalized singular-energy drift by Fisher-Rao/triangular/KS","multiview":"bounded channel/spatial/flow drift aggregated by geometric/max/RMS"},"protocol":{"seed":SEED,"sample_offset":SAMPLE_OFFSET,"num_samples":NUM_SAMPLES,"epsilon":EPSILON,"settings":list(objects)},"objects":objects,"expected_rows":int(os.environ["EXPECTED_ROWS"]),"device":os.environ["DEVICE"],"batch_size":int(os.environ["BATCH_SIZE"]),"environment":{"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"pip_freeze":subprocess.check_output([str(project/".venv/bin/python"),"-m","pip","freeze"],text=True).splitlines()},"sha256":{str(p.relative_to(project) if p.is_relative_to(project) else p):sha(p) for p in paths}}
target=run_dir/"run_manifest.json"; target.open("x",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
fi

export PYTHONPATH="$project/src:$project"
wait_for_gpu() {
  local stable=0 used_mib
  while (( stable < 3 )); do
    used_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
    printf 'gpu_gate_utc=%s used_mib=%s stable=%s\n' "$(date -u +%FT%TZ)" "$used_mib" "$stable" >> "$run_dir/STATUS.env"
    if (( used_mib <= 1000 )); then stable=$((stable + 1)); else stable=0; fi
    (( stable < 3 )) && sleep 60
  done
}
run_setting() {
  local label="$1"; shift
  wait_for_gpu
  "$project/.venv/bin/python" "$project/scripts/evaluate_e37_student.py" \
    "$@" --bundle-root "$bundle" --data-root "$data_root" --output-dir "$output_base" \
    --run-id "$run_id" --batch-size "$batch_size" --device "$device" "${item_args[@]}" \
    2>&1 | tee "$run_dir/logs/${label}.log"
}

run_setting rb40_mature --benchmark rb40
for epoch in 0 5 10 25 50; do run_setting "nb201_e$epoch" --benchmark nb201 --epoch "$epoch"; done
for epoch in 0 100; do run_setting "darts_e$epoch" --benchmark darts --epoch "$epoch"; done

CSV="$run_dir/metric_values.csv" EXPECTED_ROWS="$expected_rows" "$project/.venv/bin/python" - <<'PY'
import os, pandas as pd
from robust_metric_lab.e37_metrics import E37_METRIC_NAMES
from robust_metric_lab.student_runner import UNIQUE_KEY_COLUMNS
frame=pd.read_csv(os.environ["CSV"])
expected=int(os.environ["EXPECTED_ROWS"])
assert len(frame)==expected, (len(frame),expected)
assert frame[list(UNIQUE_KEY_COLUMNS)].drop_duplicates().shape[0]==expected
assert set(frame.metric_name)==set(E37_METRIC_NAMES)
assert set(frame.status)=={"ok"}
assert frame.metric_value.map(lambda value: __import__("math").isfinite(float(value))).all()
PY
printf 'state=COMPLETE\nrows=%s\nfinished_utc=%s\n' "$expected_rows" "$(date -u +%FT%TZ)" >> "$run_dir/STATUS.env"

