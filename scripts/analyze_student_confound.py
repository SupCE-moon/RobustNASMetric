from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robust_metric_lab.strict_ranking import load_unique_metric_rows
from robust_metric_lab.student_confound import confound_diagnostics


def _control_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("control must be SETTING=CSV")
    setting, path = value.split("=", 1)
    return setting, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute frozen benchmark confound diagnostics")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--control", type=_control_argument, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    controls = {setting: pd.read_csv(path) for setting, path in args.control}
    if len(controls) != len(args.control):
        parser.error("duplicate control setting")
    frame = load_unique_metric_rows(args.input)
    result = confound_diagnostics(frame, controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
