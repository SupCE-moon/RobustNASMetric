from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LOWER_IS_BETTER = {
    "feature_delta",
    "feature_measure_swd",
    "feature_measure_mmd",
    "feature_measure_cov_delta",
    "feature_measure_moment_delta",
    "feature_measure_local_delta",
    "feature_measure_weighted_delta",
    "feature_measure_struct_delta",
    "feature_measure_struct01_delta",
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "logit_kl",
    "natural_loss",
    "adversarial_loss",
    "input_loss_delta",
    "input_grad_l2",
    "local_logit_lipschitz",
    "feature_measure_flow_delta",
    "feature_measure_flow_rel_delta",
    "feature_measure_spatial_gram_delta",
    "feature_measure_spatial_gram_rel_delta",
    "feature_measure_relu_active_delta",
    "feature_measure_relu_active_spatial_delta",
    "feature_measure_relu_active_spatial_rel_delta",
    "feature_measure_relu_active_hybrid_delta",
    "feature_measure_relu_active_hybrid_rel_delta",
    "feature_measure_robust_delta",
    "feature_measure_robust_rel_delta",
    "feature_measure_robust_spatial_delta",
    "feature_measure_robust_spatial_rel_delta",
    "feature_measure_robust_hybrid_delta",
    "feature_measure_robust_hybrid_rel_delta",
    "fsp_strict_delta",
    "fsp_delta",
    "racl_arch_lipschitz_log_upper",
    "racl_input_gradient_lower",
}
HIGHER_IS_BETTER = {
    "margin_input_grad_ratio",
    "clean_acc",
    "adv_acc",
    "trnas_activation_psi",
    "trnas_consistency_phi",
    "trnas_r_score",
}
FEATURE_SAMPLING_METRICS = LOWER_IS_BETTER - {
    "logit_kl",
    "natural_loss",
    "adversarial_loss",
    "input_loss_delta",
    "input_grad_l2",
    "local_logit_lipschitz",
    "racl_arch_lipschitz_log_upper",
    "racl_input_gradient_lower",
}


def spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def utility(rho: float, direction: str) -> float:
    return -rho if direction == "lower" else rho


def rank_bootstrap_ci(
    left: pd.Series,
    right: pd.Series,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna().rank()
    x = valid.iloc[:, 0].to_numpy(dtype=float)
    y = valid.iloc[:, 1].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.integers(0, len(valid), len(valid))
        xs = x[sampled]
        ys = y[sampled]
        if np.ptp(xs) < 1e-12 or np.ptp(ys) < 1e-12:
            values[index] = np.nan
        else:
            values[index] = np.corrcoef(xs, ys)[0, 1]
    return tuple(float(value) for value in np.nanquantile(values, [0.025, 0.975]))


def paired_utility_difference(
    frame: pd.DataFrame,
    entity_column: str,
    target_column: str,
    metric_left: str,
    mode_left: str,
    layers_left: int,
    metric_right: str,
    mode_right: str,
    layers_right: int,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    left = frame[
        (frame["activation_mode"] == mode_left) & (frame["max_layers"] == layers_left)
    ][[entity_column, target_column, metric_left]].rename(columns={metric_left: "left"})
    right = frame[
        (frame["activation_mode"] == mode_right) & (frame["max_layers"] == layers_right)
    ][[entity_column, metric_right]].rename(columns={metric_right: "right"})
    paired = left.merge(right, on=entity_column, validate="one_to_one").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    left_direction = "lower" if metric_left in LOWER_IS_BETTER else "higher"
    right_direction = "lower" if metric_right in LOWER_IS_BETTER else "higher"

    observed = utility(
        spearman(paired["left"], paired[target_column]), left_direction
    ) - utility(spearman(paired["right"], paired[target_column]), right_direction)
    ranked = paired[["left", "right", target_column]].rank().to_numpy(dtype=float)
    left_sign = -1.0 if left_direction == "lower" else 1.0
    right_sign = -1.0 if right_direction == "lower" else 1.0

    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denominator < 1e-12:
            return float("nan")
        return float(np.dot(x_centered, y_centered) / denominator)

    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = ranked[rng.integers(0, len(ranked), len(ranked))]
        values[index] = (
            left_sign * correlation(sampled[:, 0], sampled[:, 2])
            - right_sign * correlation(sampled[:, 1], sampled[:, 2])
        )
    low, high = np.nanquantile(values, [0.025, 0.975])
    return {
        "paired_n": len(paired),
        "utility_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def summarize(
    frame: pd.DataFrame,
    metrics: list[str],
    target_column: str,
    cohort: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["activation_mode", "max_layers"], sort=True)
    for condition_index, ((activation_mode, max_layers), condition) in enumerate(grouped):
        for metric_index, metric in enumerate(metrics):
            direction = "lower" if metric in LOWER_IS_BETTER else "higher"
            rho = spearman(condition[metric], condition[target_column])
            ci_low, ci_high = rank_bootstrap_ci(
                condition[metric],
                condition[target_column],
                repeats,
                seed + condition_index * 1000 + metric_index,
            )
            ranked = condition.sort_values(metric, ascending=direction == "lower")
            top_n = max(1, int(np.ceil(0.1 * len(ranked))))
            rows.append(
                {
                    "cohort": cohort,
                    "activation_mode": activation_mode,
                    "max_layers": int(max_layers),
                    "metric": metric,
                    "scope": (
                        "feature_sampling_ablation"
                        if metric in FEATURE_SAMPLING_METRICS
                        else "fixed_reference"
                    ),
                    "direction": direction,
                    "n": len(condition),
                    "finite_n": int(
                        condition[metric].replace([np.inf, -np.inf], np.nan).notna().sum()
                    ),
                    "spearman": rho,
                    "spearman_ci_low": ci_low,
                    "spearman_ci_high": ci_high,
                    "ranking_utility": utility(rho, direction),
                    "top10_target_mean": float(ranked.head(top_n)[target_column].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--entity-column", default="arch_id")
    parser.add_argument("--target-column", default="oracle_target")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.targets is not None:
        targets = pd.read_csv(args.targets)[[args.entity_column, args.target_column]].drop_duplicates()
        frame = frame.merge(targets, on=args.entity_column, how="left", validate="many_to_one")
    required = {args.entity_column, args.target_column, "activation_mode", "max_layers"}
    missing = required - set(frame)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    metrics = [
        metric
        for metric in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)
        if metric in frame and frame[metric].notna().sum() >= 3
    ]
    cohort_frames = [("all_models", frame[frame["activation_mode"] != "post_relu"])]
    if "post_relu" in set(frame["activation_mode"]):
        post_relu = frame[frame["activation_mode"] == "post_relu"]
        eligible = post_relu.assign(
            enough_candidates=post_relu["num_candidate_layers"] >= post_relu["max_layers"]
        ).groupby(args.entity_column)["enough_candidates"].all()
        relu_entities = eligible[eligible].index
        relu_common = frame[frame[args.entity_column].isin(relu_entities)]
        cohort_frames.append(("relu_common", relu_common))
    correlations = pd.concat(
        [
            summarize(
                cohort_frame,
                metrics,
                args.target_column,
                cohort,
                args.bootstrap_repeats,
                args.seed + cohort_index * 100_000,
            )
            for cohort_index, (cohort, cohort_frame) in enumerate(cohort_frames)
        ],
        ignore_index=True,
    )

    baseline = correlations[
        (correlations["activation_mode"] == "all_leaf")
        & (correlations["max_layers"] == 4)
    ][["cohort", "metric", "ranking_utility"]].rename(
        columns={"ranking_utility": "baseline_utility"}
    )
    correlations = correlations.merge(baseline, on=["cohort", "metric"], how="left")
    correlations["utility_delta_vs_all_leaf_k4"] = (
        correlations["ranking_utility"] - correlations["baseline_utility"]
    )
    correlations = correlations.sort_values(
        ["cohort", "activation_mode", "max_layers", "ranking_utility"],
        ascending=[True, True, True, False],
    )

    feature_rows = correlations[correlations["scope"] == "feature_sampling_ablation"]
    best = (
        feature_rows.sort_values("ranking_utility", ascending=False)
        .groupby(["cohort", "metric"], as_index=False)
        .first()
        .sort_values("ranking_utility", ascending=False)
    )
    hook_comparison = feature_rows.pivot_table(
        index=["cohort", "metric", "max_layers"],
        columns="activation_mode",
        values=["spearman", "ranking_utility"],
    )
    hook_comparison.columns = [f"{value}_{mode}" for value, mode in hook_comparison.columns]
    hook_comparison = hook_comparison.reset_index()
    if {"ranking_utility_all_leaf", "ranking_utility_post_relu"}.issubset(hook_comparison):
        hook_comparison["post_relu_utility_delta"] = (
            hook_comparison["ranking_utility_post_relu"]
            - hook_comparison["ranking_utility_all_leaf"]
        )
    if {"ranking_utility_all_leaf", "ranking_utility_post_activation"}.issubset(
        hook_comparison
    ):
        hook_comparison["post_activation_utility_delta"] = (
            hook_comparison["ranking_utility_post_activation"]
            - hook_comparison["ranking_utility_all_leaf"]
        )

    layer_comparison = feature_rows[
        feature_rows["activation_mode"] == "all_leaf"
    ].pivot(index=["cohort", "metric"], columns="max_layers", values="ranking_utility").reset_index()
    layer_comparison.columns = [
        column if column in {"cohort", "metric"} else f"utility_k{int(column)}"
        for column in layer_comparison.columns
    ]
    for column in [name for name in layer_comparison if name.startswith("utility_k")]:
        if column != "utility_k4" and "utility_k4" in layer_comparison:
            layer_comparison[f"delta_{column}_vs_k4"] = (
                layer_comparison[column] - layer_comparison["utility_k4"]
            )

    paired_rows: list[dict[str, object]] = []
    for row_index, row in best.iterrows():
        cohort_frame = dict(cohort_frames)[row["cohort"]]
        comparison = paired_utility_difference(
            cohort_frame,
            args.entity_column,
            args.target_column,
            row["metric"],
            row["activation_mode"],
            int(row["max_layers"]),
            row["metric"],
            "all_leaf",
            4,
            args.bootstrap_repeats,
            args.seed + 500_000 + row_index,
        )
        paired_rows.append(
            {
                "cohort": row["cohort"],
                "metric": row["metric"],
                "best_activation_mode": row["activation_mode"],
                "best_max_layers": int(row["max_layers"]),
                "baseline_activation_mode": "all_leaf",
                "baseline_max_layers": 4,
                **comparison,
            }
        )
    paired_best = pd.DataFrame(paired_rows)

    all_models = dict(cohort_frames)["all_models"]
    selected_specs = [
        (
            "capacity_k8_minus_swd_post_activation_k16",
            "perturbation_capacity_delta",
            "all_leaf",
            8,
            "feature_measure_swd",
            "post_activation",
            16,
        ),
        (
            "capacity_k8_minus_adversarial_loss",
            "perturbation_capacity_delta",
            "all_leaf",
            8,
            "adversarial_loss",
            "all_leaf",
            4,
        ),
        (
            "mmd_k8_minus_mmd_k4",
            "feature_measure_mmd",
            "all_leaf",
            8,
            "feature_measure_mmd",
            "all_leaf",
            4,
        ),
    ]
    selected_rows = []
    for index, spec in enumerate(selected_specs):
        name, left_metric, left_mode, left_layers, right_metric, right_mode, right_layers = spec
        selected_rows.append(
            {
                "comparison": name,
                **paired_utility_difference(
                    all_models,
                    args.entity_column,
                    args.target_column,
                    left_metric,
                    left_mode,
                    left_layers,
                    right_metric,
                    right_mode,
                    right_layers,
                    args.bootstrap_repeats,
                    args.seed + 700_000 + index,
                ),
            }
        )
    selected = pd.DataFrame(selected_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(args.output_dir / "correlations_all.csv", index=False)
    best.to_csv(args.output_dir / "best_configuration_per_metric.csv", index=False)
    hook_comparison.to_csv(args.output_dir / "matched_hook_comparison.csv", index=False)
    layer_comparison.to_csv(args.output_dir / "all_leaf_layer_count_comparison.csv", index=False)
    paired_best.to_csv(args.output_dir / "best_vs_baseline_paired_bootstrap.csv", index=False)
    selected.to_csv(args.output_dir / "selected_metric_comparisons.csv", index=False)
    correlations[correlations["metric"] == "feature_measure_mmd"].to_csv(
        args.output_dir / "mmd_parameter_summary.csv", index=False
    )
    manifest = {
        "input": str(args.input),
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
        "entity_column": args.entity_column,
        "target_column": args.target_column,
        "entity_count": int(frame[args.entity_column].nunique()),
        "conditions": int(frame[["activation_mode", "max_layers"]].drop_duplicates().shape[0]),
        "cohorts": {cohort: int(cohort_frame[args.entity_column].nunique()) for cohort, cohort_frame in cohort_frames},
        "metrics_analyzed": metrics,
        "ci_method": "bootstrap Pearson correlation of globally ranked variables",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
