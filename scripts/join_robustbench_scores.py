from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_by_model(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["model_name"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--scores", default="outputs/robustbench_cifar10_linf_scores.csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scores = read_by_model(args.scores)
    with args.metrics.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    merged = []
    for row in rows:
        score = scores.get(row["model_name"])
        if score is None:
            raise SystemExit(f"Missing RobustBench score for {row['model_name']}")
        merged.append({**row, **score})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    print(args.output)


if __name__ == "__main__":
    main()
