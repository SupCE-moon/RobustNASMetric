from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRICS = {
    "feature_measure_mmd": ("lower", "primary"),
    "trnas_activation_psi": ("higher", "primary"),
    "trnas_consistency_phi": ("higher", "primary"),
    "trnas_r_score": ("higher", "primary"),
    "trnas_activation_density": ("higher", "normalization_diagnostic"),
    "trnas_consistency_mean": ("higher", "normalization_diagnostic"),
    "trnas_r_score_normalized": ("higher", "normalization_diagnostic"),
    "trnas_num_relu_units": ("higher", "structural_control"),
    "trnas_num_consistency_layers": ("higher", "structural_control"),
}
OPS = ("nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3")


def spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2:
        return float("nan")
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def utility(rho: float, direction: str) -> float:
    return -rho if direction == "lower" else rho


def partial_spearman(frame: pd.DataFrame, metric: str, controls: list[str]) -> float:
    columns = [metric, "oracle_target", *controls]
    ranked = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().rank()
    design = np.column_stack([np.ones(len(ranked)), *[ranked[c] for c in controls]])
    x = ranked[metric].to_numpy()
    y = ranked["oracle_target"].to_numpy()
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(repeats):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        values = {
            metric: utility(spearman(sampled[metric], sampled["oracle_target"]), metadata[0])
            for metric, metadata in METRICS.items()
        }
        for metric, value in values.items():
            samples[metric].append(value)

    ci_rows = []
    for metric, values in samples.items():
        low, high = np.nanquantile(values, [0.025, 0.975])
        ci_rows.append({"metric": metric, "utility_ci_low": low, "utility_ci_high": high})
    diff_rows = []
    for metric, (direction, _role) in METRICS.items():
        if metric == "feature_measure_mmd":
            continue
        paired = frame[["feature_measure_mmd", metric, "oracle_target"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        pair_rng = np.random.default_rng(seed + sum(ord(character) for character in metric))
        values = []
        for _ in range(repeats):
            sampled = paired.iloc[pair_rng.integers(0, len(paired), len(paired))]
            mmd_utility = -spearman(
                sampled["feature_measure_mmd"], sampled["oracle_target"]
            )
            metric_utility = utility(
                spearman(sampled[metric], sampled["oracle_target"]), direction
            )
            values.append(mmd_utility - metric_utility)
        low, high = np.nanquantile(values, [0.025, 0.975])
        diff_rows.append(
            {
                "comparison": f"feature_measure_mmd_minus_{metric}",
                "paired_n": len(paired),
                "utility_difference": np.nanmean(values),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sensitivity-inputs", type=Path, nargs="*")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    count_columns = [f"count_{operation}" for operation in OPS]
    ci, differences = bootstrap(frame, args.bootstrap_repeats, args.seed)

    rows = []
    for metric, (direction, role) in METRICS.items():
        rho = spearman(frame[metric], frame["oracle_target"])
        ranked = frame.sort_values(metric, ascending=direction == "lower")
        top_n = max(1, int(np.ceil(0.1 * len(ranked))))
        rows.append(
            {
                "metric": metric,
                "metric_role": role,
                "direction": direction,
                "n": len(frame),
                "finite_n": int(frame[metric].replace([np.inf, -np.inf], np.nan).notna().sum()),
                "spearman": rho,
                "ranking_utility": utility(rho, direction),
                "top10_target_mean": ranked.head(top_n)["oracle_target"].mean(),
                "population_target_mean": frame["oracle_target"].mean(),
                "partial_spearman_given_operation_counts": partial_spearman(
                    frame, metric, count_columns
                ),
                "partial_spearman_given_official_clean": partial_spearman(
                    frame, metric, ["oracle_val_acc"]
                ),
                "spearman_with_layer_count": spearman(
                    frame[metric], frame["trnas_num_consistency_layers"]
                ),
                "partial_spearman_given_layer_count": partial_spearman(
                    frame, metric, ["trnas_num_consistency_layers"]
                ),
            }
        )
    result = pd.DataFrame(rows).merge(ci, on="metric").sort_values(
        "ranking_utility", ascending=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "correlations.csv", index=False)
    differences.to_csv(args.output_dir / "paired_bootstrap_differences.csv", index=False)
    sensitivity_rows = []
    for path in args.sensitivity_inputs or []:
        sensitivity = pd.read_csv(path)
        step_size = float(sensitivity["trnas_weight_step_size"].iloc[0])
        for metric in ("trnas_activation_psi", "trnas_consistency_phi", "trnas_r_score"):
            sensitivity_rows.append(
                {
                    "input": str(path),
                    "weight_step_size": step_size,
                    "metric": metric,
                    "n": len(sensitivity),
                    "finite_n": int(sensitivity[metric].notna().sum()),
                    "spearman": spearman(sensitivity[metric], sensitivity["oracle_target"]),
                }
            )
    sensitivity_result = pd.DataFrame(sensitivity_rows)
    sensitivity_result.to_csv(args.output_dir / "weight_step_sensitivity.csv", index=False)
    print(result.to_string(index=False))
    print(differences.to_string(index=False))
    if not sensitivity_result.empty:
        print(sensitivity_result.to_string(index=False))


if __name__ == "__main__":
    main()
