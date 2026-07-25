from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


KEY_COLUMNS = {"model_name", "dataset", "threat_model", "batch_size", "num_samples", "num_layers"}
TEXT_COLUMNS = {"attack", "attack_source", "adv_cache"}


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Average per-model metrics across shared adversarial caches.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    fieldnames: list[str] | None = None
    for path in args.inputs:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                grouped[row["model_name"]].append(row)

    if not grouped or fieldnames is None:
        raise SystemExit("No rows found.")

    rows = []
    for model_name, model_rows in sorted(grouped.items()):
        out = dict(model_rows[0])
        out["attack_source"] = "multi_source_avg"
        out["adv_cache"] = ";".join(row.get("attack_source", "") for row in model_rows)
        for field in fieldnames:
            if field in TEXT_COLUMNS or field == "model_name":
                continue
            values = [parse_float(row.get(field, "")) for row in model_rows]
            numeric = [value for value in values if value is not None]
            if len(numeric) == len(model_rows):
                avg = sum(numeric) / len(numeric)
                out[field] = str(int(avg)) if field in KEY_COLUMNS and avg.is_integer() else f"{avg:.12g}"
        rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
