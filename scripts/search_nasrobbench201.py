from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.nasrobbench201 import (  # noqa: E402
    NasRobRecord,
    iter_records,
    load_json,
    load_meta,
    make_nb201_arch,
    metric_names,
    parse_nb201_arch,
)


@dataclass(frozen=True)
class Candidate:
    arch_id: str
    canonical_id: str
    arch_string: str | None
    target: float
    proxy: float
    clean: float | None


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    trial: int
    selected_arch_id: str
    selected_canonical_id: str
    selected_target: float
    selected_proxy: float
    selected_clean: float | None
    sampled_best_target: float
    global_best_target: float
    regret_to_global: float
    budget: int
    notes: str


def _as_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def read_proxy_csv(
    path: Path,
    id_column: str,
    proxy_column: str,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in proxy CSV: {path}")
    required = {id_column, proxy_column}
    if filter_column:
        required.add(filter_column)
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Proxy CSV missing columns: {', '.join(sorted(missing))}")

    values: dict[str, float] = {}
    for row in rows:
        if filter_column and str(row.get(filter_column)) != str(filter_value):
            continue
        value = _as_float(row.get(proxy_column))
        arch_id = row.get(id_column)
        if arch_id and value is not None:
            values[str(arch_id)] = value
    return values


def build_candidates(
    records: Iterable[NasRobRecord],
    target_metric: str,
    clean_metric: str | None,
    proxy_column: str,
    proxy_csv: dict[str, float] | None,
    clean_csv: dict[str, float] | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    clean_requested = clean_metric is not None or clean_csv is not None
    for record in records:
        target = record.metrics.get(target_metric)
        if target is None:
            continue

        proxy = None
        if proxy_csv is not None:
            proxy = proxy_csv.get(record.arch_id)
            if proxy is None:
                proxy = proxy_csv.get(record.canonical_id)
        else:
            proxy = record.metrics.get(proxy_column)
        if proxy is None:
            continue

        if clean_csv is not None:
            clean = clean_csv.get(record.arch_id)
            if clean is None:
                clean = clean_csv.get(record.canonical_id)
        else:
            clean = record.metrics.get(clean_metric) if clean_metric else None
        if clean_requested and clean is None:
            continue
        candidates.append(
            Candidate(
                arch_id=record.arch_id,
                canonical_id=record.canonical_id,
                arch_string=record.arch_string,
                target=target,
                proxy=proxy,
                clean=clean,
            )
        )
    if not candidates:
        raise ValueError(
            "No usable candidates after joining target/proxy metrics. "
            "Check --target-metric, --proxy-column, --proxy-csv, and --proxy-id-column."
        )
    return candidates


def proxy_key(candidate: Candidate, proxy_direction: str) -> float:
    return candidate.proxy if proxy_direction == "higher" else -candidate.proxy


def best_by_proxy(candidates: Iterable[Candidate], proxy_direction: str) -> Candidate:
    return max(candidates, key=lambda item: proxy_key(item, proxy_direction))


def random_search(
    candidates: list[Candidate],
    budget: int,
    trials: int,
    rng: random.Random,
    proxy_direction: str,
    global_best_target: float,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    sample_size = min(budget, len(candidates))
    for trial in range(trials):
        sampled = rng.sample(candidates, sample_size)
        sampled_best = max(item.target for item in sampled)
        selections = [
            (
                "random_uniform",
                rng.choice(sampled),
                "uniformly select one queried architecture without proxy guidance",
            ),
            (
                "random_proxy",
                best_by_proxy(sampled, proxy_direction),
                "random sample, then select by proxy metric",
            ),
        ]
        if all(item.clean is not None for item in sampled):
            objectives = {
                item.arch_id: objective_values(item, proxy_direction, use_clean=True)
                for item in sampled
            }
            first_front = fast_nondominated_sort(sampled, objectives)[0]
            selections.append(
                (
                    "random_clean",
                    max(sampled, key=lambda item: float(item.clean)),
                    "random sample, then select by queried early clean accuracy",
                )
            )
            selections.append(
                (
                    "random_proxy_clean",
                    recommend_from_front(first_front, proxy_direction, use_clean=True),
                    "random sample, then apply the same proxy-clean Pareto recommendation as NSGA-II",
                )
            )
        for algorithm, selected, notes in selections:
            results.append(
                SearchResult(
                    algorithm=algorithm,
                    trial=trial,
                    selected_arch_id=selected.arch_id,
                    selected_canonical_id=selected.canonical_id,
                    selected_target=selected.target,
                    selected_proxy=selected.proxy,
                    selected_clean=selected.clean,
                    sampled_best_target=sampled_best,
                    global_best_target=global_best_target,
                    regret_to_global=global_best_target - selected.target,
                    budget=sample_size,
                    notes=notes,
                )
            )
    return results


def objective_values(candidate: Candidate, proxy_direction: str, use_clean: bool) -> tuple[float, ...]:
    values = [proxy_key(candidate, proxy_direction)]
    if use_clean and candidate.clean is not None:
        values.append(candidate.clean)
    return tuple(values)


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fast_nondominated_sort(population: list[Candidate], objectives: dict[str, tuple[float, ...]]) -> list[list[Candidate]]:
    domination_counts: dict[str, int] = {}
    dominated: dict[str, list[Candidate]] = {}
    fronts: list[list[Candidate]] = [[]]

    for p in population:
        dominated[p.arch_id] = []
        domination_counts[p.arch_id] = 0
        p_obj = objectives[p.arch_id]
        for q in population:
            if p.arch_id == q.arch_id:
                continue
            q_obj = objectives[q.arch_id]
            if dominates(p_obj, q_obj):
                dominated[p.arch_id].append(q)
            elif dominates(q_obj, p_obj):
                domination_counts[p.arch_id] += 1
        if domination_counts[p.arch_id] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[Candidate] = []
        for p in fronts[i]:
            for q in dominated[p.arch_id]:
                domination_counts[q.arch_id] -= 1
                if domination_counts[q.arch_id] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distances(front: list[Candidate], objectives: dict[str, tuple[float, ...]]) -> dict[str, float]:
    if not front:
        return {}
    distances = {item.arch_id: 0.0 for item in front}
    dims = len(next(iter(objectives.values())))
    if len(front) <= 2:
        for item in front:
            distances[item.arch_id] = float("inf")
        return distances

    for dim in range(dims):
        ordered = sorted(front, key=lambda item: objectives[item.arch_id][dim])
        distances[ordered[0].arch_id] = float("inf")
        distances[ordered[-1].arch_id] = float("inf")
        min_value = objectives[ordered[0].arch_id][dim]
        max_value = objectives[ordered[-1].arch_id][dim]
        span = max_value - min_value
        if span <= 0:
            continue
        for i in range(1, len(ordered) - 1):
            prev_value = objectives[ordered[i - 1].arch_id][dim]
            next_value = objectives[ordered[i + 1].arch_id][dim]
            distances[ordered[i].arch_id] += (next_value - prev_value) / span
    return distances


def nsga2_select(
    population: list[Candidate],
    size: int,
    proxy_direction: str,
    use_clean: bool,
) -> list[Candidate]:
    population = list({item.arch_id: item for item in population}.values())
    objectives = {item.arch_id: objective_values(item, proxy_direction, use_clean) for item in population}
    fronts = fast_nondominated_sort(population, objectives)
    selected: list[Candidate] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        distances = crowding_distances(front, objectives)
        selected.extend(sorted(front, key=lambda item: distances[item.arch_id], reverse=True)[: size - len(selected)])
        break
    return selected


def normalized_proxy_score(candidate: Candidate, proxy_direction: str, min_proxy: float, max_proxy: float) -> float:
    value = proxy_key(candidate, proxy_direction)
    lo = min_proxy if proxy_direction == "higher" else -max_proxy
    hi = max_proxy if proxy_direction == "higher" else -min_proxy
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def normalized_clean_score(candidate: Candidate, min_clean: float, max_clean: float) -> float:
    if candidate.clean is None or max_clean <= min_clean:
        return 0.5
    return (candidate.clean - min_clean) / (max_clean - min_clean)


def recommend_from_front(
    front: list[Candidate],
    proxy_direction: str,
    use_clean: bool,
) -> Candidate:
    # Normalize only values in the queried first front. Using all benchmark
    # candidates here would leak information outside the search budget.
    min_proxy = min(item.proxy for item in front)
    max_proxy = max(item.proxy for item in front)
    clean_values = [float(item.clean) for item in front if item.clean is not None]
    min_clean = min(clean_values) if clean_values else 0.0
    max_clean = max(clean_values) if clean_values else 1.0

    def score(candidate: Candidate) -> float:
        total = normalized_proxy_score(candidate, proxy_direction, min_proxy, max_proxy)
        count = 1
        if use_clean:
            total += normalized_clean_score(candidate, min_clean, max_clean)
            count += 1
        return total / count

    return max(front, key=score)


def mutate_ops(
    ops: tuple[str, ...],
    operation_set: list[str],
    rng: random.Random,
) -> tuple[str, ...]:
    out = list(ops)
    index = rng.randrange(len(out))
    choices = [op for op in operation_set if op != out[index]]
    if choices:
        out[index] = rng.choice(choices)
    return tuple(out)


def crossover_ops(
    left: tuple[str, ...],
    right: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    return tuple(a if rng.random() < 0.5 else b for a, b in zip(left, right))


def make_child(
    parents: list[Candidate],
    candidate_by_arch: dict[str, Candidate],
    operation_set: list[str],
    rng: random.Random,
) -> Candidate:
    attempts = 0
    while attempts < 32:
        attempts += 1
        a, b = rng.sample(parents, 2)
        a_ops = parse_nb201_arch(a.arch_string)
        b_ops = parse_nb201_arch(b.arch_string)
        if a_ops is None or b_ops is None:
            break
        child_ops = crossover_ops(a_ops, b_ops, rng)
        if rng.random() < 0.9:
            child_ops = mutate_ops(child_ops, operation_set, rng)
        child_arch = make_nb201_arch(child_ops)
        child = candidate_by_arch.get(child_arch)
        if child is not None:
            return child
    return rng.choice(list(candidate_by_arch.values()))


def nsga2_search(
    candidates: list[Candidate],
    budget: int,
    trials: int,
    population_size: int,
    generations: int,
    rng: random.Random,
    proxy_direction: str,
    global_best_target: float,
) -> list[SearchResult]:
    use_clean = all(item.clean is not None for item in candidates)
    candidate_by_arch = {
        item.arch_string: item
        for item in candidates
        if item.arch_string and parse_nb201_arch(item.arch_string) is not None
    }
    ops_seen = sorted(
        {
            op
            for item in candidates
            for ops in [parse_nb201_arch(item.arch_string)]
            if ops is not None
            for op in ops
        }
    )
    can_mutate_arch = len(candidate_by_arch) > 0 and len(ops_seen) > 1

    pop_size = min(population_size, budget, len(candidates))
    if pop_size < 2:
        raise ValueError("NSGA-II requires at least 2 candidates in the population.")
    generation_count = max(1, min(generations, max(1, budget // pop_size)))
    results: list[SearchResult] = []
    for trial in range(trials):
        population = rng.sample(candidates, pop_size)
        seen = {item.arch_id: item for item in population}
        evaluations = len(population)

        for _ in range(generation_count):
            offspring: list[Candidate] = []
            while len(offspring) < pop_size and evaluations < budget:
                if can_mutate_arch:
                    child = make_child(population, candidate_by_arch, ops_seen, rng)
                else:
                    child = rng.choice(candidates)
                offspring.append(child)
                seen.setdefault(child.arch_id, child)
                evaluations += 1
            population = nsga2_select(population + offspring, pop_size, proxy_direction, use_clean)
            if evaluations >= budget:
                break

        objectives = {item.arch_id: objective_values(item, proxy_direction, use_clean) for item in population}
        first_front = fast_nondominated_sort(population, objectives)[0]
        selected = recommend_from_front(
            first_front,
            proxy_direction,
            use_clean,
        )
        sampled_best = max(item.target for item in seen.values())
        results.append(
            SearchResult(
                algorithm="nsga2_proxy_clean" if use_clean else "nsga2_proxy",
                trial=trial,
                selected_arch_id=selected.arch_id,
                selected_canonical_id=selected.canonical_id,
                selected_target=selected.target,
                selected_proxy=selected.proxy,
                selected_clean=selected.clean,
                sampled_best_target=sampled_best,
                global_best_target=global_best_target,
                regret_to_global=global_best_target - selected.target,
                budget=evaluations,
                notes="NSGA-II over proxy and clean accuracy" if use_clean else "NSGA-II over proxy only",
            )
        )
    return results


def write_results(path: Path, rows: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algorithm",
        "trial",
        "selected_arch_id",
        "selected_canonical_id",
        "selected_target",
        "selected_proxy",
        "selected_clean",
        "sampled_best_target",
        "global_best_target",
        "regret_to_global",
        "budget",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def print_summary(rows: list[SearchResult]) -> None:
    print("algorithm,trials,mean_selected_target,mean_regret,best_selected_target")
    algorithms = sorted({row.algorithm for row in rows})
    for algorithm in algorithms:
        subset = [row for row in rows if row.algorithm == algorithm]
        mean_target = sum(row.selected_target for row in subset) / len(subset)
        mean_regret = sum(row.regret_to_global for row in subset) / len(subset)
        best_target = max(row.selected_target for row in subset)
        print(f"{algorithm},{len(subset)},{mean_target:.6f},{mean_regret:.6f},{best_target:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run tabular random search and NSGA-II on NAS-RobBench-201 to validate "
            "whether a robustness proxy metric selects truly robust architectures."
        )
    )
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--meta-json", type=Path)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--target-metric")
    parser.add_argument("--clean-metric")
    parser.add_argument(
        "--clean-proxy-column",
        help="Read early clean accuracy from --proxy-csv instead of the benchmark JSON.",
    )
    parser.add_argument("--proxy-csv", type=Path)
    parser.add_argument("--proxy-id-column", default="arch_id")
    parser.add_argument("--proxy-filter-column")
    parser.add_argument("--proxy-filter-value")
    parser.add_argument("--proxy-column")
    parser.add_argument("--proxy-direction", choices=("lower", "higher"), default="lower")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "nasrobbench201_search_results.csv")
    parser.add_argument("--list-metrics", action="store_true")
    args = parser.parse_args()

    if bool(args.proxy_filter_column) != bool(args.proxy_filter_value):
        raise SystemExit("Use --proxy-filter-column and --proxy-filter-value together.")

    benchmark = load_json(args.benchmark_json)
    meta = load_meta(args.meta_json)
    records = iter_records(benchmark, args.dataset, meta)
    if args.list_metrics:
        print("\n".join(metric_names(records)))
        return

    if not args.target_metric:
        raise SystemExit("Please provide --target-metric, or run --list-metrics first.")
    if not args.proxy_column:
        raise SystemExit("Please provide --proxy-column. Use --proxy-csv for our metric CSV, or a benchmark metric name.")

    proxy_csv = (
        read_proxy_csv(
            args.proxy_csv,
            args.proxy_id_column,
            args.proxy_column,
            args.proxy_filter_column,
            args.proxy_filter_value,
        )
        if args.proxy_csv
        else None
    )
    if args.clean_proxy_column and not args.proxy_csv:
        raise SystemExit("--clean-proxy-column requires --proxy-csv.")
    if args.clean_proxy_column and args.clean_metric:
        raise SystemExit("Use only one of --clean-proxy-column and --clean-metric.")
    clean_csv = (
        read_proxy_csv(
            args.proxy_csv,
            args.proxy_id_column,
            args.clean_proxy_column,
            args.proxy_filter_column,
            args.proxy_filter_value,
        )
        if args.proxy_csv and args.clean_proxy_column
        else None
    )
    candidates = build_candidates(
        records,
        target_metric=args.target_metric,
        clean_metric=args.clean_metric,
        proxy_column=args.proxy_column,
        proxy_csv=proxy_csv,
        clean_csv=clean_csv,
    )
    global_best_target = max(item.target for item in candidates)
    rng = random.Random(args.seed)

    rows: list[SearchResult] = []
    rows.extend(
        random_search(
            candidates,
            budget=args.budget,
            trials=args.trials,
            rng=rng,
            proxy_direction=args.proxy_direction,
            global_best_target=global_best_target,
        )
    )
    rows.extend(
        nsga2_search(
            candidates,
            budget=args.budget,
            trials=args.trials,
            population_size=args.population,
            generations=args.generations,
            rng=rng,
            proxy_direction=args.proxy_direction,
            global_best_target=global_best_target,
        )
    )
    write_results(args.output, rows)

    print(f"records: {len(records)}")
    print(f"usable candidates: {len(candidates)}")
    print(f"target metric: {args.target_metric}")
    print(f"proxy metric: {args.proxy_column} ({args.proxy_direction}_is_better)")
    if args.clean_metric:
        print(f"clean metric: {args.clean_metric} (benchmark-assisted)")
    if args.clean_proxy_column:
        print(f"clean metric: {args.clean_proxy_column} (queried proxy CSV)")
    print(f"output: {args.output}")
    print()
    print_summary(rows)


if __name__ == "__main__":
    main()
