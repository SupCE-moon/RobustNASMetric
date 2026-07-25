from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from robust_metric_lab.metric_registry import METRIC_BY_NAME, canonical_metric_name


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    ranked = rankdata(values)
    design = np.column_stack([np.ones(len(values)), covariates])
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    return float(spearmanr(residualize(x, covariates), residualize(y, covariates)).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="full_pgd20_accuracy")
    parser.add_argument("--epoch-column", default="train_epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=(0, 50, 100))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    operation_columns = sorted(column for column in frame if column.startswith("op_count_"))
    if not operation_columns:
        raise SystemExit("No op_count_* columns found.")
    # Operation counts sum to a fixed number, so one column is removed to avoid
    # an exactly singular design. Parameter count is retained as a separate size control.
    operation_design_columns = operation_columns[:-1]
    rows: list[dict[str, object]] = []
    for epoch in args.epochs:
        subset = frame[pd.to_numeric(frame[args.epoch_column], errors="coerce") == epoch].copy()
        target = pd.to_numeric(subset[args.target], errors="coerce").to_numpy(float)
        parameter_count = rankdata(pd.to_numeric(subset["parameter_count"]).to_numpy(float))[:, None]
        operations = np.column_stack(
            [rankdata(pd.to_numeric(subset[column]).to_numpy(float)) for column in operation_design_columns]
        )
        size_and_operations = np.column_stack([parameter_count, operations])
        for column in subset.columns:
            if column not in METRIC_BY_NAME:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(float)
            mask = np.isfinite(values) & np.isfinite(target)
            if mask.sum() < 5:
                continue
            spec = METRIC_BY_NAME[column]
            raw = float(spearmanr(values[mask], target[mask]).statistic)
            rows.append(
                {
                    "epoch": epoch,
                    "metric": canonical_metric_name(column),
                    "source_field": column,
                    "direction": spec.direction,
                    "raw_spearman": raw,
                    "partial_parameter_count": partial_spearman(
                        values[mask], target[mask], parameter_count[mask]
                    ),
                    "partial_parameter_and_operation_counts": partial_spearman(
                        values[mask], target[mask], size_and_operations[mask]
                    ),
                    "metric_vs_parameter_count": float(
                        spearmanr(values[mask], parameter_count[mask, 0]).statistic
                    ),
                    "n": int(mask.sum()),
                    "operation_controls": ",".join(operation_design_columns),
                }
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
