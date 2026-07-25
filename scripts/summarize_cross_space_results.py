from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SETTING_ORDER = (
    "RobustBench20",
    "NB201_e0",
    "NB201_e5",
    "NB201_e10",
    "NB201_e25",
    "NB201_e50",
    "DARTS_e0",
    "DARTS_e50",
    "DARTS_e100",
)


def load_results(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["epoch"].isin(mapping)].copy()
    frame["setting"] = frame["epoch"].map(mapping)
    return frame


def fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.3g}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| Rank | Metric | " + " | ".join(columns) + " |",
        "|---:|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['metric']}` | "
            + " | ".join(fmt(row[column]) for column in columns)
            + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustbench", required=True, type=Path)
    parser.add_argument("--nb201", required=True, type=Path)
    parser.add_argument("--darts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frames = [
        load_results(args.robustbench, {"final": "RobustBench20"}),
        load_results(
            args.nb201,
            {
                "epoch_0": "NB201_e0",
                "epoch_5": "NB201_e5",
                "epoch_10": "NB201_e10",
                "epoch_25": "NB201_e25",
                "epoch_50": "NB201_e50",
            },
        ),
        load_results(
            args.darts,
            {"epoch_0": "DARTS_e0", "epoch_50": "DARTS_e50", "epoch_100": "DARTS_e100"},
        ),
    ]
    long = pd.concat(frames, ignore_index=True)
    long["setting"] = pd.Categorical(long["setting"], SETTING_ORDER, ordered=True)
    matrix = long.pivot_table(
        index="metric",
        columns="setting",
        values="direction_adjusted_spearman",
        observed=False,
    ).reindex(columns=SETTING_ORDER)
    matrix["zero_cost_mean"] = matrix[["NB201_e0", "DARTS_e0"]].mean(axis=1)
    matrix["mature_mean"] = matrix[["RobustBench20", "NB201_e50", "DARTS_e100"]].mean(axis=1)
    matrix["all_setting_mean"] = matrix[list(SETTING_ORDER)].mean(axis=1)
    matrix["cross_space_min"] = matrix[["RobustBench20", "NB201_e0", "DARTS_e0"]].min(axis=1)
    matrix = matrix.reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_dir / "cross_space_correlations_long.csv", index=False)
    matrix.sort_values("all_setting_mean", ascending=False).to_csv(
        args.output_dir / "cross_space_spearman_matrix.csv", index=False
    )

    zero = matrix.sort_values("zero_cost_mean", ascending=False).head(15)
    mature = matrix.sort_values("mature_mean", ascending=False).head(15)
    overall = matrix.sort_values("all_setting_mean", ascending=False).head(20)
    lines = [
        "# Cross-Space Spearman Summary",
        "",
        "All values are direction-adjusted Spearman scores: larger is better. Metric directions were fixed before analysis.",
        "",
        "## Zero-Cost Ranking",
        "",
        "Mean of NAS-RobBench-201 epoch 0 and DARTS epoch 0.",
        "",
        *markdown_table(zero, ["NB201_e0", "DARTS_e0", "zero_cost_mean"]),
        "",
        "## Mature-Model Ranking",
        "",
        "Mean of RobustBench-20, NAS-RobBench-201 epoch 50, and DARTS epoch 100.",
        "",
        *markdown_table(
            mature,
            ["RobustBench20", "NB201_e50", "DARTS_e100", "mature_mean"],
        ),
        "",
        "## All Nine Settings",
        "",
        *markdown_table(overall, [*SETTING_ORDER, "all_setting_mean"]),
        "",
        "A high average does not prove mechanism independence. Consult the DARTS structural-confound table and bootstrap intervals before making a paper claim.",
    ]
    (args.output_dir / "cross_space_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "cross_space_spearman_matrix.csv")
    print(args.output_dir / "cross_space_summary.md")


if __name__ == "__main__":
    main()
