from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


correlation = load_script("analyze_metric_correlation")
search = load_script("search_nasrobbench201")


DEFAULT_METRICS = (
    "perturbation_capacity_delta",
    "perturbation_capacity_relu_active_delta",
    "perturbation_capacity_above_mean_delta",
    "adversarial_loss",
    "input_grad_l2",
    "local_logit_lipschitz",
    "feature_measure_mmd",
    "fsp_strict_delta",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(
    rows: list[dict[str, str]],
    epochs: list[int],
    metrics: list[str],
    target: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for epoch in epochs:
        subset = [row for row in rows if row.get("train_epochs") == str(epoch)]
        if not subset:
            raise ValueError(f"No metric rows for train_epochs={epoch}")
        ys = correlation.as_float_column(subset, target)
        for metric in metrics:
            if metric not in subset[0]:
                continue
            xs = correlation.as_float_column(subset, metric)
            pearson_value = correlation.pearson(xs, ys)
            spearman_value = correlation.spearman(xs, ys)
            kendall_value = correlation.kendall_tau_b(xs, ys)
            expected = correlation.expected_direction(metric)
            aligned_spearman = spearman_value if expected == "higher_is_better" else -spearman_value
            output.append(
                {
                    "train_epochs": epoch,
                    "metric": metric,
                    "models": len(subset),
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                    "kendall_tau_b": kendall_value,
                    "expected_direction": expected,
                    "direction_agreement": correlation.direction_agreement(metric, pearson_value),
                    "aligned_spearman": aligned_spearman,
                }
            )
    return sorted(output, key=lambda row: (int(row["train_epochs"]), -float(row["aligned_spearman"])))


def metric_maps(
    rows: list[dict[str, str]], epoch: int, metric: str
) -> tuple[dict[str, float], dict[str, float]]:
    proxy: dict[str, float] = {}
    clean: dict[str, float] = {}
    for row in rows:
        if row.get("train_epochs") != str(epoch):
            continue
        arch_id = row["arch_id"]
        proxy[arch_id] = float(row[metric])
        clean[arch_id] = float(row["clean_acc"])
    return proxy, clean


def run_searches(
    benchmark_json: Path,
    meta_json: Path,
    metric_rows: list[dict[str, str]],
    dataset: str,
    target_metric: str,
    epochs: list[int],
    metrics: list[str],
    budget: int,
    trials: int,
    population: int,
    generations: int,
    seed: int,
) -> list[dict[str, object]]:
    benchmark = search.load_json(benchmark_json)
    meta = search.load_meta(meta_json)
    records = search.iter_records(benchmark, dataset, meta)
    output: list[dict[str, object]] = []
    for epoch in epochs:
        for metric in metrics:
            if metric not in metric_rows[0]:
                continue
            proxy, clean = metric_maps(metric_rows, epoch, metric)
            candidates = search.build_candidates(
                records,
                target_metric=target_metric,
                clean_metric=None,
                proxy_column=metric,
                proxy_csv=proxy,
                clean_csv=clean,
            )
            direction = "higher" if correlation.expected_direction(metric) == "higher_is_better" else "lower"
            global_best = max(item.target for item in candidates)
            rng = random.Random(seed)
            results = search.random_search(
                candidates,
                budget=budget,
                trials=trials,
                rng=rng,
                proxy_direction=direction,
                global_best_target=global_best,
            )
            results.extend(
                search.nsga2_search(
                    candidates,
                    budget=budget,
                    trials=trials,
                    population_size=population,
                    generations=generations,
                    rng=rng,
                    proxy_direction=direction,
                    global_best_target=global_best,
                )
            )
            for result in results:
                row = {
                    "train_epochs": epoch,
                    "metric": metric,
                    "proxy_direction": direction,
                }
                row.update(result.__dict__)
                output.append(row)
    return output


def summarize_search(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (int(row["train_epochs"]), str(row["metric"]), str(row["algorithm"]))
        groups[key].append(row)

    output: list[dict[str, object]] = []
    for (epoch, metric, algorithm), subset in sorted(groups.items()):
        targets = [float(row["selected_target"]) for row in subset]
        regrets = [float(row["regret_to_global"]) for row in subset]
        mean_target = sum(targets) / len(targets)
        mean_regret = sum(regrets) / len(regrets)
        std_target = math.sqrt(sum((value - mean_target) ** 2 for value in targets) / len(targets))
        output.append(
            {
                "train_epochs": epoch,
                "metric": metric,
                "algorithm": algorithm,
                "trials": len(subset),
                "mean_selected_target": mean_target,
                "std_selected_target": std_target,
                "mean_regret": mean_regret,
                "best_selected_target": max(targets),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze early-fidelity metrics and run matched NAS-RobBench searches."
    )
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--meta-json", required=True, type=Path)
    parser.add_argument("--metric-csv", required=True, type=Path)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--target", default="oracle_target")
    parser.add_argument("--target-metric", default="autoattack")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.metric_csv)
    correlations = correlation_rows(rows, args.epochs, args.metrics, args.target)
    raw_search = run_searches(
        args.benchmark_json,
        args.meta_json,
        rows,
        args.dataset,
        args.target_metric,
        args.epochs,
        args.metrics,
        args.budget,
        args.trials,
        args.population,
        args.generations,
        args.seed,
    )
    summary = summarize_search(raw_search)

    write_rows(args.output_dir / "correlations.csv", correlations)
    write_rows(args.output_dir / "search_trials.csv", raw_search)
    write_rows(args.output_dir / "search_summary.csv", summary)
    print(f"correlations: {args.output_dir / 'correlations.csv'}")
    print(f"search trials: {args.output_dir / 'search_trials.csv'}")
    print(f"search summary: {args.output_dir / 'search_summary.csv'}")


if __name__ == "__main__":
    main()
