from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


OURS_CANDIDATES = (
    "feature_measure_spatial_gram_delta",
    "feature_measure_spatial_gram_rel_delta",
    "feature_measure_weighted_delta",
    "feature_measure_robust_spatial_delta",
    "feature_measure_robust_spatial_rel_delta",
    "feature_measure_robust_hybrid_delta",
    "feature_measure_robust_hybrid_rel_delta",
)

BASELINES = ("local_logit_lipschitz", "input_grad_l2")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=float)


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std(ddof=0)
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty_like(x, dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        out[order[i:j]] = avg
        i = j
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt((x0 * x0).sum() * (y0 * y0).sum())
    if denom == 0:
        return float("nan")
    return float((x0 * y0).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(x), ranks(y))


def ols_fit(y: np.ndarray, predictors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(len(y)), *predictors])
    beta = np.linalg.pinv(x) @ y
    pred = x @ beta
    return beta, pred


def residualize(y: np.ndarray, predictors: list[np.ndarray]) -> np.ndarray:
    _, pred = ols_fit(y, predictors)
    return y - pred


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0:
        return float("nan")
    sse = float(((y - pred) ** 2).sum())
    return 1.0 - sse / sst


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def loocv_r2(y: np.ndarray, predictors: list[np.ndarray]) -> float:
    preds = np.empty_like(y)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        train_predictors = [x[mask] for x in predictors]
        beta, _ = ols_fit(y[mask], train_predictors)
        test_x = np.array([1.0, *[x[i] for x in predictors]], dtype=float)
        preds[i] = float(test_x @ beta)
    return r2_score(y, preds)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3g}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="rb_robust_acc")
    parser.add_argument("--output-dir", default="outputs/residual_value", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.input)
    y = col(rows, args.target)
    n = len(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_rows = []
    residual_export = []
    for baseline_name in BASELINES:
        baseline = col(rows, baseline_name)
        _, baseline_pred = ols_fit(y, [baseline])
        baseline_r2 = r2_score(y, baseline_pred)
        baseline_adj = adjusted_r2(baseline_r2, n, 1)
        baseline_loo = loocv_r2(y, [baseline])
        y_resid = y - baseline_pred
        rank_y_resid = residualize(ranks(y), [ranks(baseline)])

        for row, resid in zip(rows, y_resid):
            residual_export.append(
                {
                    "baseline": baseline_name,
                    "model_name": row["model_name"],
                    "target": row[args.target],
                    "baseline_value": row[baseline_name],
                    "target_residual": f"{resid:.12g}",
                }
            )

        for candidate_name in OURS_CANDIDATES:
            candidate = col(rows, candidate_name)
            candidate_resid = residualize(candidate, [baseline])
            rank_candidate_resid = residualize(ranks(candidate), [ranks(baseline)])
            _, combined_pred = ols_fit(y, [baseline, candidate])
            combined_r2 = r2_score(y, combined_pred)
            combined_adj = adjusted_r2(combined_r2, n, 2)
            combined_loo = loocv_r2(y, [baseline, candidate])
            residual_rows.append(
                {
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    "baseline_r2": fmt(baseline_r2),
                    "combined_r2": fmt(combined_r2),
                    "delta_r2": fmt(combined_r2 - baseline_r2),
                    "baseline_adj_r2": fmt(baseline_adj),
                    "combined_adj_r2": fmt(combined_adj),
                    "delta_adj_r2": fmt(combined_adj - baseline_adj),
                    "baseline_loocv_r2": fmt(baseline_loo),
                    "combined_loocv_r2": fmt(combined_loo),
                    "delta_loocv_r2": fmt(combined_loo - baseline_loo),
                    "target_residual_vs_candidate_pearson": fmt(pearson(y_resid, candidate)),
                    "partial_pearson": fmt(pearson(y_resid, candidate_resid)),
                    "partial_spearman": fmt(pearson(rank_y_resid, rank_candidate_resid)),
                }
            )

    write_csv(output_dir / "residual_value_summary.csv", residual_rows)
    write_csv(output_dir / "baseline_residuals.csv", residual_export)

    lambdas = [-3.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0]
    calibration_rows = []
    weighted = zscore(col(rows, "feature_measure_weighted_delta"))
    spatial = zscore(col(rows, "feature_measure_spatial_gram_delta"))
    spatial_rel = zscore(col(rows, "feature_measure_spatial_gram_rel_delta"))
    flow = zscore(col(rows, "feature_measure_flow_delta"))
    flow_rel = zscore(col(rows, "feature_measure_flow_rel_delta"))

    combo_specs = {
        "spatial_plus_lambda_weighted": (spatial, weighted),
        "spatial_rel_plus_lambda_weighted": (spatial_rel, weighted),
        "hybrid_plus_lambda_weighted": (0.5 * spatial + 0.5 * flow, weighted),
        "hybrid_rel_plus_lambda_weighted": (0.5 * spatial_rel + 0.5 * flow_rel, weighted),
    }
    for combo_name, (main_component, weighted_component) in combo_specs.items():
        for lam in lambdas:
            combo = main_component + lam * weighted_component
            calibration_rows.append(
                {
                    "combo": combo_name,
                    "baseline": "",
                    "lambda": lam,
                    "pearson": fmt(pearson(combo, y)),
                    "spearman": fmt(spearman(combo, y)),
                    "direction": "lower_is_better" if pearson(combo, y) < 0 else "higher_is_better",
                }
            )

    gammas = [-3.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0]
    for baseline_name in BASELINES:
        baseline_z = zscore(col(rows, baseline_name))
        for candidate_name in ("feature_measure_spatial_gram_delta", "feature_measure_spatial_gram_rel_delta"):
            candidate_z = zscore(col(rows, candidate_name))
            for gamma in gammas:
                combo = baseline_z + gamma * candidate_z
                calibration_rows.append(
                    {
                        "combo": f"{baseline_name}_plus_gamma_{candidate_name}",
                        "baseline": baseline_name,
                        "lambda": gamma,
                        "pearson": fmt(pearson(combo, y)),
                        "spearman": fmt(spearman(combo, y)),
                        "direction": "lower_is_better" if pearson(combo, y) < 0 else "higher_is_better",
                    }
                )

    write_csv(output_dir / "coefficient_calibration_grid.csv", calibration_rows)

    best_rows = []
    for combo_name in sorted({row["combo"] for row in calibration_rows}):
        group = [row for row in calibration_rows if row["combo"] == combo_name]
        best = max(group, key=lambda row: abs(float(row["spearman"])))
        best_rows.append(best)
    write_csv(output_dir / "coefficient_calibration_best.csv", best_rows)

    report = output_dir / "residual_value_calibration_report.md"
    report.write_text(
        build_report(args.input, residual_rows, best_rows),
        encoding="utf-8",
    )
    print(report)


def build_report(input_path: Path, residual_rows: list[dict[str, object]], best_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Residual-Value and Coefficient Calibration Analysis",
        "",
        "Date: 2026-07-07",
        "",
        f"Input: `{input_path}`",
        "",
        "Target: RobustBench official CIFAR-10 Linf robust accuracy.",
        "",
        "## Residual-Value Test",
        "",
        "This table asks whether an Ours metric improves a strong baseline model of robust accuracy.",
        "`delta_r2` is in-sample incremental explanatory power; `delta_loocv_r2` is leave-one-out predictive change and is more conservative for 20 models.",
        "",
        "| baseline | candidate | delta R2 | delta adjusted R2 | delta LOOCV R2 | partial Pearson | partial Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in residual_rows:
        if row["candidate"] not in {
            "feature_measure_spatial_gram_delta",
            "feature_measure_spatial_gram_rel_delta",
            "feature_measure_robust_spatial_delta",
            "feature_measure_robust_spatial_rel_delta",
        }:
            continue
        lines.append(
            "| {baseline} | `{candidate}` | {delta_r2} | {delta_adj_r2} | {delta_loocv_r2} | {partial_pearson} | {partial_spearman} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Best Exploratory Coefficient Grid Results",
            "",
            "The grid uses z-scored metric components. It is a sensitivity analysis, not a final learned metric.",
            "",
            "| combo | best coefficient | Pearson | Spearman | direction |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in best_rows:
        lines.append(
            "| `{combo}` | {lambda} | {pearson} | {spearman} | {direction} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful positive result would be: adding the spatial feature-relation metric increases adjusted R2 and leave-one-out R2 over `local_logit_lipschitz` or `input_grad_l2`, and the partial correlations are meaningfully nonzero.",
            "",
            "If in-sample R2 improves but leave-one-out R2 drops, the metric may contain overlapping or overfit information under the current 20-model setting. In that case, the research value shifts toward architecture interpretability and should be tested on same-training-different-architecture or early-training settings.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
