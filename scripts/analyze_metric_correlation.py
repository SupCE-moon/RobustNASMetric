from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_METRICS = (
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
    "margin_input_grad_ratio",
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
)


HIGHER_IS_BETTER_METRICS = {
    "clean_acc",
    "adv_acc",
    "rb_clean_acc",
    "rb_robust_acc",
    "margin_input_grad_ratio",
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def ranks(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda item: item[1])
    out = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = avg_rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def _tied_pair_count(xs: list[float]) -> int:
    counts: dict[float, int] = {}
    for value in xs:
        counts[value] = counts.get(value, 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    pair_count = len(xs) * (len(xs) - 1) // 2
    untied_x = pair_count - _tied_pair_count(xs)
    untied_y = pair_count - _tied_pair_count(ys)
    denominator = math.sqrt(untied_x * untied_y)
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float_column(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def filter_rows(
    rows: list[dict[str, str]], column: str | None, value: str | None
) -> list[dict[str, str]]:
    if (column is None) != (value is None):
        raise ValueError("--filter-column and --filter-value must be provided together")
    if column is None:
        return rows
    if rows and column not in rows[0]:
        raise ValueError(f"Filter column not found: {column}")
    return [row for row in rows if row.get(column) == value]


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3g}"


def expected_direction(metric: str) -> str:
    return "higher_is_better" if metric in HIGHER_IS_BETTER_METRICS else "lower_is_better"


def observed_correlation_sign(value: float) -> str:
    if math.isnan(value) or value == 0:
        return "flat_or_nan"
    return "negative" if value < 0 else "positive"


def direction_agreement(metric: str, pearson_value: float) -> str:
    expected = expected_direction(metric)
    observed = observed_correlation_sign(pearson_value)
    if observed == "flat_or_nan":
        return "unknown"
    if expected == "lower_is_better" and observed == "negative":
        return "yes"
    if expected == "higher_is_better" and observed == "positive":
        return "yes"
    return "no"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether candidate robustness metrics track attack accuracy."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="adv_acc")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    args = parser.parse_args()

    rows = read_rows(args.input)
    try:
        rows = filter_rows(rows, args.filter_column, args.filter_value)
    except ValueError as exc:
        parser.error(str(exc))
    if args.exclude_model:
        excluded = set(args.exclude_model)
        rows = [row for row in rows if row.get("model_name") not in excluded]
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    target = as_float_column(rows, args.target)
    print(f"input: {args.input}")
    print(f"models: {len(rows)}")
    if args.filter_column:
        print(f"filter: {args.filter_column}={args.filter_value}")
    if args.exclude_model:
        print(f"excluded: {', '.join(args.exclude_model)}")
    print(f"target: {args.target}")
    print()
    print("metric,pearson,spearman,kendall_tau_b,expected_direction,observed_correlation_sign,direction_agreement")

    skipped = []
    for metric in args.metrics:
        if metric not in rows[0]:
            skipped.append(metric)
            continue
        xs = as_float_column(rows, metric)
        pearson_value = pearson(xs, target)
        print(
            ",".join(
                (
                    metric,
                    fmt(pearson_value),
                    fmt(spearman(xs, target)),
                    fmt(kendall_tau_b(xs, target)),
                    expected_direction(metric),
                    observed_correlation_sign(pearson_value),
                    direction_agreement(metric, pearson_value),
                )
            )
        )
    if skipped:
        print()
        print("# skipped_missing_metrics=" + ",".join(skipped))


if __name__ == "__main__":
    main()
