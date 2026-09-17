#!/usr/bin/env python3
"""Analyze unified student-benchmark metric CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robust_metric_lab.student_analysis import REQUIRED_COLUMNS, analyze_frame, write_analysis_outputs
from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.student_benchmark import SETTING_BY_KEY


def _parse_inputs(values: list[list[Path]]) -> list[Path]:
    return [path for group in values for path in group]


def _load_inputs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one --input CSV is required")
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for row_number, row in frame.iterrows():
            metric = str(row["metric_name"])
            setting = str(row["setting"])
            identifier = str(row["model_or_arch_id"])
            key = (metric, setting, identifier)
            if key in seen:
                raise ValueError(
                    "duplicate key across input CSVs: "
                    f"metric_name={metric}, setting={setting}, model_or_arch_id={identifier}"
                )
            seen.add(key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze unified student-benchmark metric CSVs using the frozen eight-setting protocol."
    )
    parser.add_argument(
        "--input", required=True, nargs="+", action="append", type=Path,
        help="input metric_values.csv (repeat --input or provide multiple paths)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for three analysis artifacts")
    args = parser.parse_args(argv)
    paths = _parse_inputs(args.input)
    try:
        output = args.output_dir
        targets = [output / "correlations.csv", output / "all_setting_table.csv", output / "analysis_report.md"]
        existing = [path.name for path in targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing output: " + ", ".join(existing))
        frame = _load_inputs(paths)
        result = analyze_frame(frame, input_files=[str(path) for path in paths])
        write_analysis_outputs(result, args.output_dir)
    except (OSError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
