from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Join metric extension columns onto a base metric table.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--keys", required=True, nargs="+")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Extension columns to attach; defaults to every non-key column.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    extension = pd.read_csv(args.extension)
    missing = set(args.keys).difference(base.columns) | set(args.keys).difference(extension.columns)
    if missing:
        raise ValueError(f"Missing join keys: {sorted(missing)}")
    if extension.duplicated(args.keys).any():
        raise ValueError("Extension table contains duplicate join keys")
    new_columns = args.columns or [
        column for column in extension.columns if column not in args.keys
    ]
    if missing_columns := set(new_columns).difference(extension.columns):
        raise ValueError(f"Missing extension columns: {sorted(missing_columns)}")
    extension = extension[[*args.keys, *new_columns]]
    overlap = set(new_columns).intersection(base.columns)
    if overlap:
        base = base.drop(columns=sorted(overlap))
    joined = base.merge(extension, on=args.keys, how="left", validate="one_to_one")
    if joined[new_columns].isna().any().any():
        raise ValueError("At least one base row has no complete metric extension")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output, index=False)
    print(f"# rows={len(joined)} extension_columns={len(new_columns)}")
    print(args.output)


if __name__ == "__main__":
    main()
