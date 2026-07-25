from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path


README_URL = "https://raw.githubusercontent.com/RobustBench/robustbench/master/README.md"
ROW_RE = re.compile(
    r"\|\s*<sub>\*\*(?P<rank>\d+)\*\*</sub>\s*"
    r"\|\s*<sub><sup>\*\*(?P<model_id>[^*]+)\*\*</sup></sub>\s*"
    r"\|.*?"
    r"\|\s*<sub>(?P<clean>\d+(?:\.\d+)?)%</sub>\s*"
    r"\|\s*<sub>(?P<robust>\d+(?:\.\d+)?)%</sub>\s*"
    r"\|",
)


DATASET_HEADING = {
    "cifar10": "### CIFAR-10",
    "cifar100": "### CIFAR-100",
    "imagenet": "### ImageNet",
}


def section_between(text: str, start_marker: str, next_pattern: str) -> str:
    start = text.index(start_marker)
    match = re.search(next_pattern, text[start + len(start_marker) :])
    if match is None:
        return text[start:]
    end = start + len(start_marker) + match.start()
    return text[start:end]


def extract_dataset_linf_section(text: str, dataset: str) -> str:
    dataset_section = section_between(text, DATASET_HEADING[dataset], r"\n### ")
    linf_match = re.search(r"#### Linf[^\n]*", dataset_section)
    if linf_match is None:
        raise SystemExit(f"No Linf section found for {dataset}")
    linf_section = section_between(dataset_section, linf_match.group(0), r"\n#### ")
    return linf_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_HEADING))
    parser.add_argument("--output", default="outputs/robustbench_cifar10_linf_scores.csv")
    args = parser.parse_args()

    with urllib.request.urlopen(README_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    text = extract_dataset_linf_section(text, args.dataset)

    rows = []
    for match in ROW_RE.finditer(text):
        rows.append(
            {
                "rank": int(match.group("rank")),
                "model_name": match.group("model_id").strip(),
                "rb_clean_acc": float(match.group("clean")) / 100.0,
                "rb_robust_acc": float(match.group("robust")) / 100.0,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "model_name", "rb_clean_acc", "rb_robust_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
