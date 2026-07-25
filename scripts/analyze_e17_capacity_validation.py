from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


DEFAULT_METRICS = (
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "feature_measure_mmd",
    "adversarial_loss",
    "fsp_strict_delta",
    "local_logit_lipschitz",
    "input_grad_l2",
    "clean_acc",
    "adv_acc",
)


def load_complete_rows(paths: list[Path], epochs: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    expected = tuple(sorted(epochs))
    observed = raw.groupby(["source_file", "arch_id"])["train_epochs"].agg(
        lambda values: tuple(sorted(set(int(value) for value in values)))
    )
    complete_groups = set(observed[observed == expected].index)
    keep = [
        (source, int(arch_id)) in complete_groups
        for source, arch_id in zip(raw["source_file"], raw["arch_id"])
    ]
    complete = raw.loc[keep].copy()

    duplicated = complete.duplicated(["arch_id", "train_epochs"], keep=False)
    if duplicated.any():
        duplicate_ids = sorted(complete.loc[duplicated, "arch_id"].unique())
        raise ValueError(f"Complete architecture appears in multiple sources: {duplicate_ids}")
    complete = complete.sort_values(["arch_id", "train_epochs"]).reset_index(drop=True)
    return complete


def partial_rank_correlation(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    controls: list[str],
) -> tuple[float, float]:
    def standardized_rank(column: str) -> np.ndarray:
        values = rankdata(frame[column].to_numpy(dtype=float))
        return (values - values.mean()) / values.std(ddof=0)

    metric_rank = standardized_rank(metric)
    target_rank = standardized_rank(target)
    design = np.column_stack(
        [np.ones(len(frame)), *[standardized_rank(control) for control in controls]]
    )

    def residual(values: np.ndarray) -> np.ndarray:
        # The design has at most four columns. A tiny pivoted Gaussian solver
        # avoids an intermittent Windows MKL/PyTorch abort in np.linalg.lstsq.
        columns = design.shape[1]
        gram = [
            [float(np.dot(design[:, i], design[:, j])) for j in range(columns)]
            for i in range(columns)
        ]
        rhs = [float(np.dot(design[:, i], values)) for i in range(columns)]
        for column in range(columns):
            pivot = max(range(column, columns), key=lambda row: abs(gram[row][column]))
            gram[column], gram[pivot] = gram[pivot], gram[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
            if abs(gram[column][column]) < 1e-12:
                gram[column][column] = 1e-12
            scale = gram[column][column]
            gram[column] = [value / scale for value in gram[column]]
            rhs[column] /= scale
            for row in range(columns):
                if row == column:
                    continue
                factor = gram[row][column]
                gram[row] = [
                    gram[row][item] - factor * gram[column][item]
                    for item in range(columns)
                ]
                rhs[row] -= factor * rhs[column]
        fitted = sum(rhs[index] * design[:, index] for index in range(columns))
        return values - fitted

    metric_residual = residual(metric_rank)
    target_residual = residual(target_rank)
    result = pearsonr(metric_residual, target_residual)
    pvalue = min(1.0, max(0.0, float(result.pvalue)))
    return float(result.statistic), pvalue


def correlation_table(
    frame: pd.DataFrame,
    epochs: tuple[int, ...],
    metrics: tuple[str, ...],
    target: str,
) -> pd.DataFrame:
    rows = []
    for epoch in epochs:
        subset = frame[frame["train_epochs"] == epoch]
        for metric in metrics:
            if metric not in subset:
                continue
            pearson = pearsonr(subset[metric], subset[target])
            spearman = spearmanr(subset[metric], subset[target])
            kendall = kendalltau(subset[metric], subset[target], variant="b")
            rows.append(
                {
                    "train_epochs": epoch,
                    "metric": metric,
                    "architectures": len(subset),
                    "pearson": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "kendall_tau_b": kendall.statistic,
                    "kendall_p": kendall.pvalue,
                }
            )
    return pd.DataFrame(rows).sort_values(["train_epochs", "spearman"])


def bootstrap_capacity(
    frame: pd.DataFrame,
    epochs: tuple[int, ...],
    metric: str,
    target: str,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = frame.pivot(index="arch_id", columns="train_epochs")
    arch_ids = pivot.index.to_numpy()
    rng = np.random.default_rng(seed)

    def rho(epoch: int, indices: np.ndarray) -> float:
        sampled = arch_ids[indices]
        return float(
            spearmanr(
                pivot.loc[sampled, (metric, epoch)],
                pivot.loc[sampled, (target, epoch)],
            ).statistic
        )

    distributions: dict[int, np.ndarray] = {}
    rows = []
    observed_indices = np.arange(len(arch_ids))
    for epoch in epochs:
        values = np.asarray(
            [rho(epoch, rng.integers(0, len(arch_ids), len(arch_ids))) for _ in range(samples)]
        )
        distributions[epoch] = values
        rows.append(
            {
                "train_epochs": epoch,
                "metric": metric,
                "architectures": len(arch_ids),
                "spearman": rho(epoch, observed_indices),
                "ci95_low": np.quantile(values, 0.025),
                "ci95_high": np.quantile(values, 0.975),
                "bootstrap_samples": samples,
            }
        )

    changes = []
    first_epoch = epochs[0]
    for epoch in epochs[1:]:
        # Re-sample the same architecture indices at both checkpoints.
        values = []
        for _ in range(samples):
            indices = rng.integers(0, len(arch_ids), len(arch_ids))
            values.append(rho(epoch, indices) - rho(first_epoch, indices))
        observed = rho(epoch, observed_indices) - rho(first_epoch, observed_indices)
        changes.append(
            {
                "from_epoch": first_epoch,
                "to_epoch": epoch,
                "spearman_change": observed,
                "ci95_low": np.quantile(values, 0.025),
                "ci95_high": np.quantile(values, 0.975),
                "probability_change_below_zero": np.mean(np.asarray(values) < 0),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(changes)


def residual_table(
    frame: pd.DataFrame,
    epochs: tuple[int, ...],
    metric: str,
    target: str,
) -> pd.DataFrame:
    control_sets = {
        "same_run_clean_acc": ["clean_acc"],
        "official_val_acc_diagnostic": ["oracle_val_acc"],
        "training_maturity": ["clean_acc", "natural_loss", "train_epoch_loss"],
    }
    rows = []
    for epoch in epochs:
        subset = frame[frame["train_epochs"] == epoch]
        for name, controls in control_sets.items():
            value, pvalue = partial_rank_correlation(subset, metric, target, controls)
            rows.append(
                {
                    "train_epochs": epoch,
                    "metric": metric,
                    "control_set": name,
                    "controls": "+".join(controls),
                    "partial_spearman": value,
                    "p_value": pvalue,
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return f"{value:.3g}"


def write_report(
    path: Path,
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    bootstrap: pd.DataFrame,
    changes: pd.DataFrame,
    residuals: pd.DataFrame,
    expected_architectures: int,
    requested_status: str,
) -> None:
    architectures = frame["arch_id"].nunique()
    status = requested_status
    if status == "auto":
        status = "final" if architectures == expected_architectures else "interim"
    cap = correlations[correlations["metric"] == "perturbation_capacity_delta"]
    selected = correlations[
        correlations["metric"].isin(
            [
                "perturbation_capacity_delta",
                "perturbation_capacity_relu_active_delta",
                "perturbation_capacity_above_mean_delta",
                "feature_measure_mmd",
                "adversarial_loss",
                "fsp_strict_delta",
                "clean_acc",
            ]
        )
    ]

    lines = [
        "# E17 Full-Training Capacity Validation Report",
        "",
        f"Status: **{status}**, with {architectures}/{expected_architectures} planned architectures complete.",
        "",
        "## Capacity Result",
        "",
        "| Epoch | Spearman | p-value | Bootstrap 95% CI |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in cap.iterrows():
        boot = bootstrap[bootstrap["train_epochs"] == row["train_epochs"]].iloc[0]
        lines.append(
            f"| {int(row['train_epochs'])} | {fmt(row['spearman'])} | "
            f"{fmt(row['spearman_p'])} | [{fmt(boot['ci95_low'])}, {fmt(boot['ci95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## Selected Metric Comparison",
            "",
            "| Epoch | Metric | Spearman | p-value |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in selected.sort_values(["train_epochs", "spearman"]).iterrows():
        lines.append(
            f"| {int(row['train_epochs'])} | `{row['metric']}` | "
            f"{fmt(row['spearman'])} | {fmt(row['spearman_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Residual Diagnostic",
            "",
            "| Epoch | Control set | Partial Spearman | p-value |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in residuals.iterrows():
        lines.append(
            f"| {int(row['train_epochs'])} | `{row['control_set']}` | "
            f"{fmt(row['partial_spearman'])} | {fmt(row['p_value'])} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint Change",
            "",
            "| From | To | Spearman change | Bootstrap 95% CI |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in changes.iterrows():
        lines.append(
            f"| {int(row['from_epoch'])} | {int(row['to_epoch'])} | "
            f"{fmt(row['spearman_change'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "Raw negative correlation is directionally consistent with a lower-is-better capacity metric. "
            "The residual rows test whether that association remains after removing clean-performance "
            "or training-maturity rank information. The official-validation control is diagnostic only "
            "and must not be used as a search-time input.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E17 full-training capacity validation.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--target", default="oracle_target")
    parser.add_argument("--metric", default="perturbation_capacity_delta")
    parser.add_argument("--expected-architectures", type=int, default=96)
    parser.add_argument(
        "--status",
        choices=("auto", "interim", "final", "stopped"),
        default="auto",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    epochs = tuple(sorted(set(args.epochs)))
    frame = load_complete_rows(args.inputs, epochs)
    correlations = correlation_table(frame, epochs, DEFAULT_METRICS, args.target)
    bootstrap, changes = bootstrap_capacity(
        frame,
        epochs,
        args.metric,
        args.target,
        args.bootstrap_samples,
        args.seed,
    )
    residuals = residual_table(frame, epochs, args.metric, args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "complete_metrics.csv", index=False)
    correlations.to_csv(args.output_dir / "correlations.csv", index=False)
    bootstrap.to_csv(args.output_dir / "capacity_bootstrap.csv", index=False)
    changes.to_csv(args.output_dir / "capacity_checkpoint_changes.csv", index=False)
    residuals.to_csv(args.output_dir / "capacity_residuals.csv", index=False)
    write_report(
        args.output_dir / "e17_full50_capacity_report.md",
        frame,
        correlations,
        bootstrap,
        changes,
        residuals,
        args.expected_architectures,
        args.status,
    )
    print(f"complete_architectures={frame['arch_id'].nunique()}")
    print(args.output_dir / "e17_full50_capacity_report.md")


if __name__ == "__main__":
    main()
