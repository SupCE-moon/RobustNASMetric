"""Frozen, CPU-only TRNAS/Ours comparison on a verified strict release.

This analysis never changes a leaderboard. Structural matching uses controls
fixed by the original strict release; unmatched architectures stay unmatched.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


METRICS = (
    "trnas_consistency_phi",
    "trnas_r_score",
    "trnas_activation_psi",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
    "feature_measure_mmd",
)
SETTINGS = (
    "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
    "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
)
SPACE_SETTINGS = {
    "rb40": ("rb40_mature",),
    "nb201": SETTINGS[1:6],
    "darts": SETTINGS[6:8],
}


def spearman_utility(values: np.ndarray, target: np.ndarray, direction: str) -> float:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or not np.isfinite(target).all():
        return float("nan")
    x, y = rankdata(values), rankdata(target)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rho = float(np.corrcoef(x, y)[0, 1])
    if direction == "lower":
        return -rho
    if direction == "higher":
        return rho
    raise ValueError(f"Unexpected direction: {direction}")


def matched_pairs(controls: list[dict], setting: str) -> list[tuple[int, int]]:
    """Freeze matching before reading targets or metric values."""
    pairs = []
    for i in range(len(controls)):
        a = controls[i]
        for j in range(i + 1, len(controls)):
            b = controls[j]
            if setting.startswith("nb201_"):
                cols = sorted(k for k in a if k.startswith("op_count_"))
                keep = all(int(a[k]) == int(b[k]) for k in cols)
            elif setting.startswith("darts_"):
                cols = sorted(k for k in a if k.startswith("op_count_"))
                keep = all(int(a[k]) == int(b[k]) for k in cols)
                keep &= abs(float(a["log_parameter_count"]) - float(b["log_parameter_count"])) <= 0.1
            elif setting == "rb40_mature":
                keep = a["model_family"] == b["model_family"]
                keep &= abs(float(a["log_parameter_count"]) - float(b["log_parameter_count"])) <= 0.1
            else:
                raise ValueError(setting)
            if keep:
                pairs.append((i, j))
    return pairs


def pair_concordance(
    pairs: list[tuple[int, int]], target: np.ndarray, values: np.ndarray, direction: str
) -> tuple[int, int, float]:
    sign = -1 if direction == "lower" else 1 if direction == "higher" else None
    if sign is None:
        raise ValueError(direction)
    good = usable = 0
    for i, j in pairs:
        ty = np.sign(target[j] - target[i])
        vx = np.sign(sign * (values[j] - values[i]))
        if ty == 0 or vx == 0:
            continue
        usable += 1
        good += int(ty == vx)
    return good, usable, good / usable if usable else float("nan")


def load_release(release: Path, controls_root: Path):
    verification = json.loads((release / "release_verification.json").read_text())
    assert verification == {
        "status": "PASS", "raw_rows": 47600, "unique_raw_keys": 47600,
        "ranked_metrics": 83, "excluded_metrics": 2,
    }, verification
    raw = pd.read_csv(release / "metric_values_merged.csv", low_memory=False)
    assert len(raw) == 47600
    keys = ["metric_name", "setting", "model_or_arch_id"]
    assert not raw.duplicated(keys).any(), "Duplicated raw keys"
    rank = pd.read_csv(release / "ranking/strict_ranking.csv")
    assert set(METRICS) <= set(rank.metric_name)
    selected = raw[raw.metric_name.isin(METRICS)].copy()
    assert len(selected) == 560 * len(METRICS)
    assert selected.status.eq("ok").all()
    assert np.isfinite(pd.to_numeric(selected.metric_value)).all()
    assert set(selected.setting) == set(SETTINGS)
    directions = selected.groupby("metric_name").direction.unique()
    directions = {k: v[0] for k, v in directions.items() if len(v) == 1}
    assert len(directions) == len(METRICS)
    data = {}
    controls = {}
    for setting in SETTINGS:
        part = selected[selected.setting.eq(setting)]
        assert len(part) == len(METRICS) * (40 if setting == "rb40_mature" else 64 if setting.startswith("nb201") else 100)
        targets = part.groupby("model_or_arch_id").target_value.nunique()
        assert targets.eq(1).all(), f"Inconsistent targets: {setting}"
        wide = part.pivot(index="model_or_arch_id", columns="metric_name", values="metric_value")
        wide["target"] = part.groupby("model_or_arch_id").target_value.first()
        wide = wide.sort_index()
        assert not wide.isna().any().any()
        control = pd.read_csv(controls_root / f"{setting}.csv").set_index("model_or_arch_id")
        control.index = control.index.astype(str)
        wide.index = wide.index.astype(str)
        assert set(wide.index) == set(control.index), f"Controls mismatch: {setting}"
        data[setting] = wide.sort_index()
        controls[setting] = control.loc[data[setting].index]
        for metric in METRICS:
            observed = spearman_utility(data[setting][metric], data[setting].target, directions[metric])
            recorded = float(rank.set_index("metric_name").loc[metric, setting])
            assert abs(observed - recorded) < 1e-9, (setting, metric, observed, recorded)
    for space_settings in SPACE_SETTINGS.values():
        ids = data[space_settings[0]].index
        assert all(data[s].index.equals(ids) for s in space_settings)
    return data, controls, directions, verification


def analyze(data, controls, directions, draws: int, seed: int):
    raw_rows, match_rows, stage_rows = [], [], []
    for setting in SETTINGS:
        frame = data[setting]
        pairs = matched_pairs(controls[setting].to_dict("records"), setting)
        pair_models = len(set(k for pair in pairs for k in pair))
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            target = frame.target.to_numpy(float)
            raw_rows.append({"setting": setting, "metric_name": metric, "n": len(frame),
                             "spearman_utility": spearman_utility(values, target, directions[metric])})
            good, usable, fraction = pair_concordance(pairs, target, values, directions[metric])
            match_rows.append({"setting": setting, "metric_name": metric,
                               "eligible_pairs": len(pairs), "models_in_pairs": pair_models,
                               "usable_pairs": usable, "concordant_pairs": good,
                               "pair_concordance": fraction if usable >= 10 else float("nan"),
                               "interpretation": "descriptive" if usable >= 10 else "insufficient_pairs"})
    rng = np.random.default_rng(seed)
    boot_mean = {metric: [] for metric in METRICS}
    stage_defs = (("nb201_e0", "nb201_e50"), ("darts_e0", "darts_e100"))
    boot_stage = {(a, b, metric): [] for a, b in stage_defs for metric in METRICS}
    for _ in range(draws):
        indices = {space: rng.integers(0, len(data[settings[0]]), len(data[settings[0]]))
                   for space, settings in SPACE_SETTINGS.items()}
        sampled = {}
        for space, settings in SPACE_SETTINGS.items():
            for setting in settings:
                sampled[setting] = data[setting].iloc[indices[space]]
        for metric in METRICS:
            scores = [spearman_utility(sampled[s][metric], sampled[s].target, directions[metric])
                      for s in SETTINGS]
            if np.isfinite(scores).all():
                boot_mean[metric].append(float(np.mean(scores)))
            for a, b in stage_defs:
                early = spearman_utility(sampled[a][metric], sampled[a].target, directions[metric])
                late = spearman_utility(sampled[b][metric], sampled[b].target, directions[metric])
                if np.isfinite([early, late]).all():
                    boot_stage[a, b, metric].append(late - early)
    comparison_rows = []
    observed = pd.DataFrame(raw_rows).pivot(index="metric_name", columns="setting", values="spearman_utility")
    for ours in METRICS[3:]:
        for trnas in METRICS[:3]:
            # Identical valid draws are required for a paired contrast.
            a, b = np.array(boot_mean[ours]), np.array(boot_mean[trnas])
            assert len(a) == len(b) == draws
            diff = a - b
            comparison_rows.append({"ours": ours, "trnas": trnas, "n_bootstrap": draws,
                                    "observed_mean_difference": float(observed.loc[ours, list(SETTINGS)].mean() - observed.loc[trnas, list(SETTINGS)].mean()),
                                    "ci_low": float(np.quantile(diff, 0.025)),
                                    "ci_high": float(np.quantile(diff, 0.975))})
    for a, b in stage_defs:
        assert data[a].index.equals(data[b].index)
        for metric in METRICS:
            early = data[a][metric].to_numpy(float)
            late = data[b][metric].to_numpy(float)
            early_rho = float(observed.loc[metric, a])
            late_rho = float(observed.loc[metric, b])
            delta = np.array(boot_stage[a, b, metric])
            stage_rows.append({"metric_name": metric, "early": a, "late": b, "n": len(early),
                               "metric_rank_stability": spearman_utility(early, late, "higher"),
                               "early_utility": early_rho, "late_utility": late_rho,
                               "late_minus_early": late_rho - early_rho,
                               "delta_ci_low": float(np.quantile(delta, 0.025)),
                               "delta_ci_high": float(np.quantile(delta, 0.975))})
    return map(pd.DataFrame, (raw_rows, match_rows, stage_rows, comparison_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    assert args.draws >= 100
    assert not args.output.exists(), "Output exists; refusing to overwrite"
    data, controls, directions, verification = load_release(args.release, args.controls_root)
    raw, matched, stage, comparisons = analyze(data, controls, directions, args.draws, args.seed)
    args.output.mkdir(parents=True)
    raw.to_csv(args.output / "raw_setting_correlations.csv", index=False)
    matched.to_csv(args.output / "structure_matched_pairs.csv", index=False)
    stage.to_csv(args.output / "training_stage_stability.csv", index=False)
    comparisons.to_csv(args.output / "paired_bootstrap_mean_differences.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "release": str(args.release),
                "controls_root": str(args.controls_root), "release_verification": verification,
                "metrics": METRICS, "settings": SETTINGS, "bootstrap_draws": args.draws,
                "bootstrap_seed": args.seed,
                "matching": "NB201 exact operation counts; DARTS exact operation counts and |delta log params|<=0.1; RB40 same family and |delta log params|<=0.1. No relaxation.",
                "limitations": "Matched-pair concordance is descriptive; pair dependencies preclude naive pair-level confidence intervals. Bootstrap resamples architectures jointly across epochs within each space."}
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "ranked_metrics": len(METRICS),
                      "raw_cells": len(raw), "matched_cells": len(matched),
                      "stage_cells": len(stage), "comparisons": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
