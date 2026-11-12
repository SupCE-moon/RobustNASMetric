"""Preregistered paired resampling and holdout-material checks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def require_release_verified(verification: Mapping[str, object]) -> None:
    required = {
        "status": "PASS", "raw_rows": 48160, "unique_raw_keys": 48160,
        "ranked_metrics": 84, "excluded_metrics": 2,
    }
    if any(verification.get(key) != value for key, value in required.items()):
        raise ValueError("analysis requires the complete verified 48,160-row release")


def space_balanced_mean(setting_scores: Mapping[str, float]) -> float:
    nb = tuple(f"nb201_e{epoch}" for epoch in (0, 5, 10, 25, 50))
    da = ("darts_e0", "darts_e100")
    required = {"rb40_mature", *nb, *da}
    if set(setting_scores) != required:
        raise ValueError("three-space mean requires all eight settings")
    return float(np.mean([
        float(setting_scores["rb40_mature"]),
        float(np.mean([setting_scores[key] for key in nb])),
        float(np.mean([setting_scores[key] for key in da])),
    ]))


def interpret_predeclared_gates(
    *, mean_delta_ci_low: float, new_stage_ci_low: float,
    stability_delta_ci_low: float,
) -> dict[str, bool]:
    if not all(np.isfinite(value) for value in (
        mean_delta_ci_low, new_stage_ci_low, stability_delta_ci_low
    )):
        raise ValueError("gate confidence bounds must be finite")
    return {
        "beats_phi_on_frozen_grid": mean_delta_ci_low > 0,
        "improves_stage_stability": (
            new_stage_ci_low > 0 and stability_delta_ci_low > 0
        ),
    }


def _space(setting: str) -> str:
    if setting == "rb40_mature":
        return "rb40"
    if setting.startswith("nb201_e"):
        return "nb201"
    if setting.startswith("darts_e"):
        return "darts"
    raise ValueError(f"unknown setting: {setting}")


def paired_sample_indices(
    ids_by_setting: Mapping[str, Sequence[str]], *, draws: int, seed: int,
) -> list[dict[str, tuple[str, ...]]]:
    """Draw each architecture once per space and reuse it at every epoch."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    groups: dict[str, list[str]] = defaultdict(list)
    canonical = {}
    for setting, ids in ids_by_setting.items():
        space = _space(setting)
        ids = tuple(str(value) for value in ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"empty or duplicate architecture IDs: {setting}")
        groups[space].append(setting)
        existing = canonical.get(space)
        if existing is not None and set(existing) != set(ids):
            raise ValueError(f"epoch architecture set differs in {space}")
        canonical[space] = tuple(sorted(ids))
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(draws):
        draw = {}
        for space, settings in groups.items():
            choices = tuple(rng.choice(canonical[space], size=len(canonical[space]),
                                       replace=True).tolist())
            for setting in settings:
                draw[setting] = choices
        output.append(draw)
    return output


def stage_pairs(
    early: Sequence[tuple[str, float]],
    late: Sequence[tuple[str, float]],
) -> list[tuple[str, float, float]]:
    """Align the identical architecture population in early-stage order."""
    early_ids = [str(identifier) for identifier, _ in early]
    late_ids = [str(identifier) for identifier, _ in late]
    if (not early_ids or len(early_ids) != len(set(early_ids))
            or len(late_ids) != len(set(late_ids))
            or set(early_ids) != set(late_ids)):
        raise ValueError("stage architecture IDs are missing, extra, or duplicate")
    late_by_id = {str(identifier): float(value) for identifier, value in late}
    return [(str(identifier), float(value), late_by_id[str(identifier)])
            for identifier, value in early]


def holdout_material_status(
    expected_ids: Sequence[str], materials: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """No external-validity result unless all 11 frozen materials are valid."""
    expected = tuple(expected_ids)
    if len(expected) != 11 or len(set(expected)) != 11:
        raise ValueError("the frozen RB40 holdout must contain 11 unique IDs")
    invalid = [
        identifier for identifier in expected
        if not all(materials.get(identifier, {}).get(field) is True for field in
                   ("checkpoint_ok", "label_ok", "sample_hash_ok"))
    ]
    return {
        "ready": not invalid,
        "complete": len(expected) - len(invalid),
        "missing_or_invalid": invalid,
    }


def _prepared_rows(rows: Sequence[Mapping[str, object]], metrics: tuple[str, str],
                   *, with_target: bool):
    by_setting: dict[str, dict[str, dict[str, Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    directions = {}
    for row in rows:
        name = str(row["metric_name"])
        if name not in metrics:
            continue
        setting = str(row["setting"])
        identifier = str(row["model_or_arch_id"])
        if identifier in by_setting[setting][name]:
            raise ValueError(f"duplicate metric architecture: {setting}/{name}/{identifier}")
        by_setting[setting][name][identifier] = row
        if with_target:
            direction = str(row["direction"])
            if direction not in {"higher", "lower"}:
                raise ValueError(f"invalid direction: {name}")
            if name in directions and directions[name] != direction:
                raise ValueError(f"inconsistent direction: {name}")
            directions[name] = direction
    arrays = {}
    space_ids = {}
    for setting, metrics_here in by_setting.items():
        if set(metrics_here) != set(metrics):
            raise ValueError(f"metric missing from setting {setting}")
        ids = tuple(sorted(metrics_here[metrics[0]]))
        if not ids or set(ids) != set(metrics_here[metrics[1]]):
            raise ValueError(f"metric architecture mismatch: {setting}")
        space = _space(setting)
        if space in space_ids and space_ids[space] != ids:
            raise ValueError(f"epoch architecture mismatch: {space}")
        space_ids[space] = ids
        values = {
            name: np.asarray([float(metrics_here[name][identifier]["metric_value"])
                              for identifier in ids], dtype=float)
            for name in metrics
        }
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"nonfinite metric: {setting}")
        targets = None
        if with_target:
            targets = np.asarray([
                float(metrics_here[metrics[0]][identifier]["target_value"])
                for identifier in ids
            ], dtype=float)
            other = np.asarray([
                float(metrics_here[metrics[1]][identifier]["target_value"])
                for identifier in ids
            ], dtype=float)
            if not np.isfinite(targets).all() or not np.array_equal(targets, other):
                raise ValueError(f"target mismatch: {setting}")
        arrays[setting] = {"ids": ids, "values": values, "target": targets}
    if not arrays:
        raise ValueError("no matching rows")
    return arrays, directions, space_ids


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    result = float(spearmanr(first, second).statistic)
    return result


def _percentile(values: list[float]) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        raise ValueError("no finite bootstrap draws")
    return tuple(float(value) for value in np.quantile(valid, [0.025, 0.975]))


def bootstrap_mean_delta(
    rows: Sequence[Mapping[str, object]], new_metric: str, comparison_metric: str,
    *, draws: int = 20000, seed: int = 20261024,
) -> dict[str, object]:
    """Paired 8-setting oriented Spearman difference with shared epoch samples."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    metrics = (new_metric, comparison_metric)
    arrays, directions, space_ids = _prepared_rows(rows, metrics, with_target=True)
    required_settings = {
        "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
        "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
    }
    if set(arrays) != required_settings:
        raise ValueError("formal mean comparison requires all eight settings")
    settings = tuple(sorted(arrays))
    signs = {name: 1 if directions[name] == "higher" else -1 for name in metrics}

    def score(indices: Mapping[str, np.ndarray]) -> float:
        differences = []
        for setting in settings:
            record = arrays[setting]
            index = indices[_space(setting)]
            target = record["target"][index]
            rhos = [
                _rho(signs[name] * record["values"][name][index], target)
                for name in metrics
            ]
            differences.append(rhos[0] - rhos[1])
        return float(np.mean(differences))

    full = {space: np.arange(len(ids)) for space, ids in space_ids.items()}
    point = score(full)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        indices = {
            space: rng.integers(0, len(ids), size=len(ids))
            for space, ids in sorted(space_ids.items())
        }
        value = score(indices)
        if np.isfinite(value):
            deltas.append(value)
    low, high = _percentile(deltas)
    return {
        "point_delta": point, "ci95_low": low, "ci95_high": high,
        "valid_draws": len(deltas), "requested_draws": draws,
        "seed": seed, "shared_epoch_indices": True,
        "settings": list(settings),
    }


def bootstrap_stage_stability(
    rows: Sequence[Mapping[str, object]], new_metric: str, comparison_metric: str,
    *, draws: int = 20000, seed: int = 20261024,
) -> dict[str, object]:
    """Paired DARTS e0/e100 rank-stability comparison on identical models."""
    metrics = (new_metric, comparison_metric)
    arrays, _, space_ids = _prepared_rows(rows, metrics, with_target=False)
    if set(arrays) != {"darts_e0", "darts_e100"}:
        raise ValueError("stage stability requires exactly DARTS e0/e100")
    if draws <= 0:
        raise ValueError("draws must be positive")
    early, late = arrays["darts_e0"], arrays["darts_e100"]

    def score(index: np.ndarray) -> tuple[float, float]:
        return tuple(
            _rho(early["values"][name][index], late["values"][name][index])
            for name in metrics
        )

    n = len(space_ids["darts"])
    point_new, point_comparison = score(np.arange(n))
    rng = np.random.default_rng(seed)
    new_draws, delta_draws = [], []
    for _ in range(draws):
        new, comparison = score(rng.integers(0, n, size=n))
        if np.isfinite(new) and np.isfinite(comparison):
            new_draws.append(new)
            delta_draws.append(new - comparison)
    new_low, new_high = _percentile(new_draws)
    delta_low, delta_high = _percentile(delta_draws)
    return {
        "new_stability": point_new, "comparison_stability": point_comparison,
        "ci95_new_low": new_low, "ci95_new_high": new_high,
        "ci95_delta_low": delta_low, "ci95_delta_high": delta_high,
        "valid_draws": len(delta_draws), "requested_draws": draws,
        "seed": seed, "shared_architecture_indices": True,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_release(release: Path, *, draws: int) -> dict[str, object]:
    """Use only verified full-precision raw rows and the strict final ranking."""
    verification_path = release / "release_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require_release_verified(verification)
    raw_path = release / "metric_values_merged.csv"
    ranking_path = release / "ranking/strict_ranking.csv"
    names = (
        "feature_adv_excess_centered_unit_l2", "trnas_consistency_phi",
        "trnas_r_score", "feature_measure_multiview_max_delta",
        "feature_measure_multiview_rms_delta",
    )
    selected = [row for row in _read_csv(raw_path) if row["metric_name"] in names]
    for name in names:
        rows = [row for row in selected if row["metric_name"] == name]
        if (len(rows) != 560 or any(row["status"] != "ok" for row in rows)
                or len({(row["setting"], row["model_or_arch_id"]) for row in rows}) != 560):
            raise ValueError(f"comparison metric lacks full 560-key evidence: {name}")
    ranking = {row["metric_name"]: row for row in _read_csv(ranking_path)}
    if not set(names).issubset(ranking):
        raise ValueError("one prespecified comparison metric is not formally ranked")
    eight = (
        "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
        "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
    )
    score_rows = {}
    for name in names:
        row = ranking[name]
        settings = {setting: float(row[setting]) for setting in eight}
        score_rows[name] = {
            "strict_rank": int(row["rank"]),
            "settings": settings,
            "all_setting_mean": float(row["all_setting_mean"]),
            "three_space_equal_mean": space_balanced_mean(settings),
        }
    new_name = names[0]
    mean_differences = {
        name: bootstrap_mean_delta(selected, new_name, name,
                                   draws=draws, seed=20261024)
        for name in names[1:]
    }
    darts = [row for row in selected if row["setting"] in {"darts_e0", "darts_e100"}]
    stability = {
        name: bootstrap_stage_stability(darts, new_name, name,
                                        draws=draws, seed=20261024)
        for name in (
            "feature_measure_multiview_max_delta",
            "feature_measure_multiview_rms_delta",
        )
    }
    max_stability = stability["feature_measure_multiview_max_delta"]
    gates = interpret_predeclared_gates(
        mean_delta_ci_low=mean_differences["trnas_consistency_phi"]["ci95_low"],
        new_stage_ci_low=max_stability["ci95_new_low"],
        stability_delta_ci_low=max_stability["ci95_delta_low"],
    )
    runtime = defaultdict(list)
    for row in selected:
        if row["metric_name"] == new_name:
            runtime[row["setting"]].append(float(row["runtime_sec"]))
    costs = {
        setting: {"rows": len(values), "median_runtime_sec": statistics.median(values)}
        for setting, values in runtime.items()
    }
    confound_path = release / "ranking/confound_diagnostics.csv"
    confound = None
    if confound_path.is_file():
        confound = [
            row for row in _read_csv(confound_path) if row["metric_name"] == new_name
        ]
    return {
        "release": str(release.resolve()),
        "verification": verification,
        "source_sha256": {
            "raw": _sha256(raw_path), "ranking": _sha256(ranking_path),
            "verification": _sha256(verification_path),
        },
        "bootstrap": {"draws": draws, "seed": 20261024,
                      "method": "paired architecture resampling shared across stages"},
        "scores": score_rows,
        "mean_differences": mean_differences,
        "darts_stage_stability": stability,
        "predeclared_gates": gates,
        "runtime": costs,
        "structure_control": confound,
        "external_holdout": {
            "status": "pending_material_repair",
            "frozen_models": 11,
            "historical_loadable_models": 10,
            "failed_model": "Rebuffi2021Fixing_70_16_cutmix_ddpm",
            "reason": "historical E37 raw evidence reports an unreadable checkpoint",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze_release(args.release, draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
