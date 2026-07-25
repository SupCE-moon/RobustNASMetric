from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from robust_metric_lab.metric_registry import METRIC_BY_NAME, canonical_metric_name


def finite_correlation(function, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(function(x, y).statistic)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    lower_is_better: bool,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        value = finite_correlation(spearmanr, x[indices], y[indices])
        if np.isfinite(value):
            values.append(-value if lower_is_better else value)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def analyze_group(
    frame: pd.DataFrame,
    target: str,
    space: str,
    epoch: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    for column in frame.columns:
        if column not in METRIC_BY_NAME:
            continue
        spec = METRIC_BY_NAME[column]
        x_all = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        x = x_all[mask]
        y = y_all[mask]
        if len(x) < 3:
            continue
        pearson = finite_correlation(pearsonr, x, y)
        spearman = finite_correlation(spearmanr, x, y)
        kendall = finite_correlation(kendalltau, x, y)
        score = -spearman if spec.lower_is_better else spearman
        ci_low, ci_high = bootstrap_spearman(
            x,
            y,
            spec.lower_is_better,
            bootstrap_samples,
            seed + len(rows),
        )
        rows.append(
            {
                "space": space,
                "epoch": epoch,
                "metric": canonical_metric_name(column),
                "source_field": column,
                "attribution": spec.attribution,
                "reference": spec.reference,
                "direction": spec.direction,
                "n": len(x),
                "pearson": pearson,
                "spearman": spearman,
                "kendall_tau": kendall,
                "direction_adjusted_spearman": score,
                "bootstrap_95_low": ci_low,
                "bootstrap_95_high": ci_high,
            }
        )
    return sorted(rows, key=lambda row: float(row["direction_adjusted_spearman"]), reverse=True)


def format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(number) else f"{number:.3g}"


def write_report(path: Path, results: pd.DataFrame, target: str) -> None:
    lines = [
        "# Cross-Space Robust Metric Correlation",
        "",
        f"Target: `{target}`. Directions are fixed in `metric_registry.py` before analysis.",
        "All displayed numbers use three significant digits.",
        "",
    ]
    for (space, epoch), group in results.groupby(["space", "epoch"], sort=False):
        group = group.sort_values("direction_adjusted_spearman", ascending=False)
        lines.extend(
            [
                f"## {space}: {epoch}",
                "",
                "| Rank | Metric | Attribution | Direction | n | Spearman | Adjusted score | 95% bootstrap CI |",
                "|---:|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            interval = (
                f"[{format_number(row['bootstrap_95_low'])}, "
                f"{format_number(row['bootstrap_95_high'])}]"
            )
            lines.append(
                f"| {rank} | `{row['metric']}` | {row['attribution']} {row['reference']} | "
                f"{row['direction']} | {int(row['n'])} | {format_number(row['spearman'])} | "
                f"{format_number(row['direction_adjusted_spearman'])} | {interval} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--space", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch-column")
    parser.add_argument("--epochs", nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.target not in frame:
        raise SystemExit(f"Target column not found: {args.target}")
    groups: list[tuple[str, pd.DataFrame]] = [("final", frame)]
    if args.epoch_column:
        if args.epoch_column not in frame:
            raise SystemExit(f"Epoch column not found: {args.epoch_column}")
        values = args.epochs or [str(value) for value in frame[args.epoch_column].dropna().unique()]
        groups = []
        for value in values:
            numeric = pd.to_numeric(frame[args.epoch_column], errors="coerce")
            subset = frame[numeric == float(value)]
            if not subset.empty:
                groups.append((f"epoch_{value}", subset))

    rows: list[dict[str, object]] = []
    for epoch, subset in groups:
        rows.extend(
            analyze_group(
                subset,
                args.target,
                args.space,
                epoch,
                args.bootstrap_samples,
                args.seed,
            )
        )
    results = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    write_report(args.output_report, results, args.target)
    print(args.output_csv)
    print(args.output_report)


if __name__ == "__main__":
    main()
