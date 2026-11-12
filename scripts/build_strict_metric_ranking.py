from __future__ import annotations

import argparse
from pathlib import Path

from robust_metric_lab.metric_eligibility import load_metric_eligibility, validate_registry
from robust_metric_lab.strict_ranking import build_strict_ranking, load_unique_metric_rows, manifest_expectations
from robust_metric_lab.student_analysis import write_analysis_outputs
from robust_metric_lab.student_benchmark import load_darts_items, load_nb201_items, load_rb40_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strict ranking from raw metric rows only")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--rb40-manifest", type=Path, required=True)
    parser.add_argument("--nb201-manifest", type=Path, required=True)
    parser.add_argument("--darts-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--eligibility", type=Path, default=Path("configs/metric_eligibility.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_metric_eligibility(args.eligibility)
    validate_registry(records)
    frame = load_unique_metric_rows(args.input)
    expected = manifest_expectations(
        load_rb40_items(args.rb40_manifest, args.benchmark_root),
        load_nb201_items(args.nb201_manifest, args.benchmark_root),
        load_darts_items(args.darts_manifest, args.benchmark_root),
    )
    ranking, excluded, analysis = build_strict_ranking(frame, expected, records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(args.output_dir / "strict_ranking.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_metrics.csv", index=False)
    write_analysis_outputs(analysis, args.output_dir / "analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
