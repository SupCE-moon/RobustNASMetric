from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_trnas_mmd_nb201 import OPS, partial_spearman, spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["trnas_activation_density"] = (
        frame["trnas_activation_psi"] / frame["trnas_num_relu_units"].clip(lower=1)
    )
    frame["trnas_consistency_mean"] = (
        frame["trnas_consistency_phi"] / frame["trnas_num_consistency_layers"].clip(lower=1)
    )
    frame["trnas_r_score_normalized"] = (
        frame["trnas_activation_density"].pow(0.5) * frame["trnas_consistency_mean"]
    )
    for operation in OPS:
        frame[f"count_{operation}"] = frame["arch_string"].str.count(operation)
    operation_counts = [f"count_{operation}" for operation in OPS]

    metrics = (
        "feature_measure_mmd",
        "trnas_activation_psi",
        "trnas_consistency_phi",
        "trnas_r_score",
        "trnas_activation_density",
        "trnas_consistency_mean",
        "trnas_r_score_normalized",
    )
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "spearman_target": spearman(frame[metric], frame["oracle_target"]),
                "spearman_layer_count": spearman(
                    frame[metric], frame["trnas_num_consistency_layers"]
                ),
                "partial_target_given_layer_count": partial_spearman(
                    frame, metric, ["trnas_num_consistency_layers"]
                ),
                "partial_target_given_operation_counts": partial_spearman(
                    frame, metric, operation_counts
                ),
                "partial_target_given_official_clean": partial_spearman(
                    frame, metric, ["oracle_val_acc"]
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
