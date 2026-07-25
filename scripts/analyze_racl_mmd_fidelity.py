from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRICS = {
    "feature_measure_mmd": "lower",
    "racl_arch_lipschitz_log_upper": "lower",
    "racl_input_gradient_lower": "lower",
    "trnas_activation_psi": "higher",
    "trnas_consistency_phi": "higher",
    "trnas_r_score": "higher",
}

OPERATION_COUNT_COLUMNS = (
    "count_nor_conv_3x3",
    "count_nor_conv_1x1",
    "count_skip_connect",
    "count_avg_pool_3x3",
)


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def partial_spearman(df: pd.DataFrame, metric: str, target: str, controls: list[str]) -> float:
    columns = [metric, target, *controls]
    valid = df[columns].replace([np.inf, -np.inf], np.nan).dropna().rank(method="average")
    design = np.column_stack([np.ones(len(valid)), *[valid[c].to_numpy() for c in controls]])
    x = valid[metric].to_numpy()
    y = valid[target].to_numpy()
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.ptp(x_res) < 1e-12 or np.ptp(y_res) < 1e-12:
        return float("nan")
    # Partial Spearman is Pearson correlation after residualizing rank variables.
    return float(np.corrcoef(x_res, y_res)[0, 1])


def bootstrap_spearman(
    df: pd.DataFrame,
    metric: str,
    target: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    valid = df[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for index in range(repeats):
        sample = valid.iloc[rng.integers(0, len(valid), len(valid))]
        values[index] = spearman(sample[metric], sample[target])
    return tuple(float(value) for value in np.nanquantile(values, [0.025, 0.975]))


def summarize_frame(
    frame: pd.DataFrame,
    experiment: str,
    epoch: int,
    subset: str,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, direction in METRICS.items():
        if metric not in frame or frame[metric].notna().sum() < 3:
            continue
        rho = spearman(frame[metric], frame["oracle_target"])
        ci_low, ci_high = bootstrap_spearman(
            frame, metric, "oracle_target", bootstrap_repeats, seed
        )
        ranked = frame.sort_values(metric, ascending=direction == "lower")
        top_n = max(1, int(np.ceil(len(ranked) * 0.1)))
        row: dict[str, object] = {
            "experiment": experiment,
            "epoch": epoch,
            "subset": subset,
            "n": len(frame),
            "metric": metric,
            "direction": direction,
            "spearman": rho,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "top10_target_mean": float(ranked.head(top_n)["oracle_target"].mean()),
            "population_target_mean": float(frame["oracle_target"].mean()),
        }
        if "oracle_val_acc" in frame:
            row["spearman_clean"] = spearman(frame[metric], frame["oracle_val_acc"])
            row["partial_spearman_given_oracle_clean"] = partial_spearman(
                frame, metric, "oracle_target", ["oracle_val_acc"]
            )
        if "clean_acc" in frame:
            row["partial_spearman_given_low_fidelity_clean"] = partial_spearman(
                frame, metric, "oracle_target", ["clean_acc"]
            )
        if all(column in frame for column in OPERATION_COUNT_COLUMNS):
            row["partial_spearman_given_operation_counts"] = partial_spearman(
                frame, metric, "oracle_target", list(OPERATION_COUNT_COLUMNS)
            )
            if "clean_acc" in frame:
                row["partial_spearman_given_ops_and_low_fidelity_clean"] = partial_spearman(
                    frame,
                    metric,
                    "oracle_target",
                    [*OPERATION_COUNT_COLUMNS, "clean_acc"],
                )
        rows.append(row)
    return rows


def add_experiment(
    rows: list[dict[str, object]],
    path: Path | None,
    name: str,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if "arch_string" in frame:
        for operation in (
            "nor_conv_3x3",
            "nor_conv_1x1",
            "skip_connect",
            "avg_pool_3x3",
        ):
            frame[f"count_{operation}"] = frame["arch_string"].str.count(operation)
    if "racl_input_gradient_lower" not in frame and "input_grad_l2" in frame:
        frame["racl_input_gradient_lower"] = frame["input_grad_l2"]
    for epoch, epoch_frame in frame.groupby("train_epochs"):
        rows.extend(
            summarize_frame(
                epoch_frame,
                name,
                int(epoch),
                "all",
                bootstrap_repeats,
                seed + int(epoch),
            )
        )
        if "racl_zero_node_sums" in epoch_frame:
            structural = epoch_frame[epoch_frame["racl_zero_node_sums"] == 0]
            if len(structural) >= 3:
                rows.extend(
                    summarize_frame(
                        structural,
                        name,
                        int(epoch),
                        "structurally_non_degenerate",
                        bootstrap_repeats,
                        seed + 10_000 + int(epoch),
                    )
                )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-cost", type=Path)
    parser.add_argument("--zero-name", default="E19-zero-cost")
    parser.add_argument("--early-fidelity", type=Path)
    parser.add_argument("--early-name", default="E16-early-fidelity")
    parser.add_argument("--mature-fidelity", type=Path)
    parser.add_argument("--mature-name", default="E17-mature-fidelity")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    frames = {
        args.zero_name: add_experiment(
            rows, args.zero_cost, args.zero_name, args.bootstrap_repeats, args.seed
        ),
        args.early_name: add_experiment(
            rows, args.early_fidelity, args.early_name, args.bootstrap_repeats, args.seed
        ),
        args.mature_name: add_experiment(
            rows, args.mature_fidelity, args.mature_name, args.bootstrap_repeats, args.seed
        ),
    }
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "racl_mmd_fidelity_correlations.csv", index=False)

    stability_rows: list[dict[str, object]] = []
    for name, frame in frames.items():
        if frame is None or "feature_measure_mmd" not in frame:
            continue
        wide = frame.pivot(index="arch_id", columns="train_epochs", values="feature_measure_mmd")
        for left in wide.columns:
            for right in wide.columns:
                if left >= right:
                    continue
                paired = wide[[left, right]].dropna()
                stability_rows.append(
                    {
                        "experiment": name,
                        "epoch_left": int(left),
                        "epoch_right": int(right),
                        "n": len(paired),
                        "mmd_rank_stability": spearman(paired[left], paired[right]),
                    }
                )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(args.output_dir / "mmd_rank_stability.csv", index=False)

    payload = {
        "inputs": {
            key: str(value) if value is not None else None
            for key, value in {
                "zero_cost": args.zero_cost,
                "early_fidelity": args.early_fidelity,
                "mature_fidelity": args.mature_fidelity,
            }.items()
        },
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
