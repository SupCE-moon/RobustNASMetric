from __future__ import annotations

import argparse
from pathlib import Path

from robust_metric_lab.release import repair_existing_metric_grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely repair failed rows in an existing raw metric grid")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repair_existing_metric_grid(args.source, args.repair, args.target, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
