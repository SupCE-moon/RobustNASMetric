from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robust_metric_lab.metric_eligibility import load_metric_eligibility, validate_registry
from robust_metric_lab.release import verify_strict_release
from robust_metric_lab.strict_ranking import load_unique_metric_rows, manifest_expectations
from robust_metric_lab.student_benchmark import load_darts_items, load_nb201_items, load_rb40_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a strict all-metric release from raw evidence")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--rb40-manifest", type=Path, required=True)
    parser.add_argument("--nb201-manifest", type=Path, required=True)
    parser.add_argument("--darts-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--eligibility", type=Path, default=Path("configs/metric_eligibility.json"))
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = load_metric_eligibility(args.eligibility)
    validate_registry(records)
    frame = load_unique_metric_rows(args.input)
    expected = manifest_expectations(
        load_rb40_items(args.rb40_manifest, args.benchmark_root),
        load_nb201_items(args.nb201_manifest, args.benchmark_root),
        load_darts_items(args.darts_manifest, args.benchmark_root),
    )
    payload = verify_strict_release(frame, expected, records, pd.read_csv(args.ranking))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
