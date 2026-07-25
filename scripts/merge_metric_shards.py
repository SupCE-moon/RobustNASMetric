from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def merge_rows(
    paths: list[Path],
    key_columns: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    fields: list[str] | None = None
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        current_fields, rows = read_csv(path)
        if fields is None:
            fields = current_fields
        elif current_fields != fields:
            raise ValueError(f"CSV header mismatch: {path}")
        missing = set(key_columns) - set(current_fields)
        if missing:
            raise ValueError(f"Missing key columns in {path}: {', '.join(sorted(missing))}")
        for row in rows:
            key = tuple(row[column] for column in key_columns)
            if key in seen:
                raise ValueError(f"Duplicate key {key} found in {path}")
            seen.add(key)
            merged.append(row)
    if fields is None:
        raise ValueError("No input CSV files provided")
    return fields, merged


def validate_grid(
    rows: list[dict[str, str]],
    expected_architectures: int | None,
    expected_epochs: list[int] | None,
) -> None:
    if expected_architectures is None and not expected_epochs:
        return
    arch_ids = {row["arch_id"] for row in rows}
    epochs = {int(row["train_epochs"]) for row in rows}
    if expected_architectures is not None and len(arch_ids) != expected_architectures:
        raise ValueError(
            f"Expected {expected_architectures} architectures, found {len(arch_ids)}"
        )
    if expected_epochs and epochs != set(expected_epochs):
        raise ValueError(f"Expected epochs {sorted(expected_epochs)}, found {sorted(epochs)}")
    if expected_architectures is not None and expected_epochs:
        expected_rows = expected_architectures * len(expected_epochs)
        if len(rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge metric CSV shards with duplicate and grid checks.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-columns", nargs="+", default=["arch_id", "train_epochs"])
    parser.add_argument("--expected-architectures", type=int)
    parser.add_argument("--expected-epochs", nargs="+", type=int)
    args = parser.parse_args()

    fields, rows = merge_rows(args.inputs, args.key_columns)
    validate_grid(rows, args.expected_architectures, args.expected_epochs)
    rows.sort(key=lambda row: (int(row.get("arch_start", 0)), row["arch_id"], int(row["train_epochs"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"inputs: {len(args.inputs)}")
    print(f"rows: {len(rows)}")
    print(f"architectures: {len({row['arch_id'] for row in rows})}")
    print(f"epochs: {sorted({int(row['train_epochs']) for row in rows})}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
