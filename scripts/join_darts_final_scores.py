from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach one final DARTS robustness score to every checkpoint metric row."
    )
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    scores = pd.read_csv(args.scores)

    required_metric_columns = {"arch_id", "train_epochs"}
    required_score_columns = {"arch_id", "full_clean_accuracy", "full_pgd20_accuracy"}
    if missing := required_metric_columns.difference(metrics.columns):
        raise ValueError(f"Metric table is missing columns: {sorted(missing)}")
    if missing := required_score_columns.difference(scores.columns):
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")
    if scores["arch_id"].duplicated().any():
        duplicates = scores.loc[scores["arch_id"].duplicated(False), "arch_id"].tolist()
        raise ValueError(f"Final-score table contains duplicate arch_id values: {duplicates}")

    score_columns = [column for column in scores.columns if column != "checkpoint_path"]
    overlap = set(score_columns).intersection(metrics.columns).difference({"arch_id"})
    if overlap:
        metrics = metrics.drop(columns=sorted(overlap))

    joined = metrics.merge(scores[score_columns], on="arch_id", how="left", validate="many_to_one")
    missing_targets = joined["full_pgd20_accuracy"].isna()
    if missing_targets.any():
        missing_ids = sorted(joined.loc[missing_targets, "arch_id"].unique().tolist())
        raise ValueError(f"No final score found for arch_id values: {missing_ids}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output, index=False)
    print(
        f"# rows={len(joined)} architectures={joined['arch_id'].nunique()} "
        f"epochs={sorted(joined['train_epochs'].unique().tolist())}"
    )
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()
