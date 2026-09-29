from __future__ import annotations

import argparse
import csv
from pathlib import Path

from robust_metric_lab.metric_eligibility import load_metric_eligibility, validate_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and export metric eligibility")
    parser.add_argument("--config", type=Path, default=Path("configs/metric_eligibility.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_metric_eligibility(args.config)
    validate_registry(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric_name", "eligibility", "supported_settings", "reason"))
        for record in records:
            writer.writerow(
                (
                    record.metric_name,
                    record.eligibility.value,
                    ";".join(record.supported_settings),
                    record.reason,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
