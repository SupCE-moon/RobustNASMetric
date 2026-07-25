from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from robust_metric_lab.metric_registry import METRIC_BY_NAME, canonical_metric_name


OPS = ("nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3")


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    ranked = rankdata(values)
    design = np.column_stack([np.ones(len(values)), covariates])
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    return float(
        spearmanr(residualize(x, covariates), residualize(y, covariates)).statistic
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="oracle_target")
    parser.add_argument("--epoch-column", default="train_epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=(0, 5, 10, 25, 50))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    for operation in OPS:
        frame[f"count_{operation}"] = frame["arch_string"].str.count(operation)
    # Operation counts sum to six in NAS-Bench-201, so remove one column.
    operation_columns = [f"count_{operation}" for operation in OPS[:-1]]
    rows: list[dict[str, object]] = []
    for epoch in args.epochs:
        subset = frame[pd.to_numeric(frame[args.epoch_column], errors="coerce") == epoch]
        target = pd.to_numeric(subset[args.target], errors="coerce").to_numpy(float)
        operations = np.column_stack(
            [rankdata(pd.to_numeric(subset[column]).to_numpy(float)) for column in operation_columns]
        )
        official_clean = rankdata(
            pd.to_numeric(subset["oracle_val_acc"], errors="coerce").to_numpy(float)
        )[:, None]
        operations_and_clean = np.column_stack([operations, official_clean])
        for column in subset.columns:
            if column not in METRIC_BY_NAME:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(float)
            mask = np.isfinite(values) & np.isfinite(target)
            if mask.sum() < 5:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "metric": canonical_metric_name(column),
                    "source_field": column,
                    "direction": METRIC_BY_NAME[column].direction,
                    "raw_spearman": float(spearmanr(values[mask], target[mask]).statistic),
                    "partial_operation_counts": partial_spearman(
                        values[mask], target[mask], operations[mask]
                    ),
                    "partial_official_clean_accuracy": partial_spearman(
                        values[mask], target[mask], official_clean[mask]
                    ),
                    "partial_operations_and_clean_accuracy": partial_spearman(
                        values[mask], target[mask], operations_and_clean[mask]
                    ),
                    "n": int(mask.sum()),
                    "operation_controls": ",".join(operation_columns),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
