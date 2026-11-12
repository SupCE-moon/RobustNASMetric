"""Attach confidence intervals to the RB40-only ranking.

Why this exists
---------------
``tools/analyze_setting_weights.py`` re-ranks the frozen board inside each
grid and finds that the eight-setting mean agrees with NB201-only
(Kendall tau +0.926) but not with RB40-only (tau -0.127).  On the 40 real
defended models ``trnas_r_score`` scores -0.184, i.e. it would order them
backwards.

That claim rests on 40 models.  A Spearman correlation over 40 points has a
wide sampling interval, so the point estimate alone cannot support
"R-Score fails on defended models".  This tool resamples the 40 models with
replacement and reports, for every metric, a 95% interval for its rank
correlation and for its RB40-only rank.  It also resamples the agreement
between the RB40-only ranking and the published eight-setting ranking, so
the tau = -0.127 itself gets an interval.

What it deliberately does not do
--------------------------------
It never re-ranks or rewrites the published table.  The output is a separate
audit next to it; D-021 explains why the frozen board stays as published.

Caveat that must travel with the numbers
----------------------------------------
Models are the resampling unit because the 40 RB40 entries are 40 distinct
trained networks.  Resampling by metric row instead would treat the 85
metrics of one model as independent evidence, which they are not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

RB40_SETTING = "rb40_mature"
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261112
CI = (2.5, 97.5)

# The release table stores three decimals, so a recomputation can only agree to
# within half a unit in the last place. Anything beyond that is a real mismatch.
REPRODUCTION_TOLERANCE = 5.5e-4

# Metrics worth naming in the summary because their ranks move the most
# between the eight-setting mean and the RB40-only view.
FOCUS_METRICS = (
    "trnas_consistency_phi",
    "trnas_r_score",
    "trnas_activation_psi",
    "adversarial_loss",
    "feature_adv_excess_centered_unit_l2",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rb40(
    rows_path: Path, repair_path: Path | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, str], list[str]]:
    """Pivot the RB40 slice into (models x metrics), with one target per model.

    ``repair_path`` matters: the three TRNAS metrics failed on Kang2021Stable
    with a float32 underflow and were recomputed into a separate
    ``repair_rows.csv``.  Without merging it, those three metrics are missing
    one model each and get dropped from the comparison — which would silently
    remove exactly the metrics this audit is about.

    Returns the raw matrix (before direction normalisation), the target vector,
    each metric's declared direction, and the names repaired from the patch.
    """
    frame = pd.read_csv(rows_path)
    repaired: list[str] = []
    if repair_path is not None and repair_path.exists():
        patch = pd.read_csv(repair_path)
        repaired = sorted(patch["metric_name"].unique().tolist())
        frame = pd.concat([frame, patch], ignore_index=True)

    subset = frame[(frame["setting"] == RB40_SETTING) & (frame["status"] == "ok")].copy()

    # A metric may be listed with several rows per model in messy exports; keep
    # the last non-null value (patched rows are appended last) rather than
    # summing, which would be meaningless.
    subset = subset.dropna(subset=["metric_value"])
    subset = subset.drop_duplicates(["model_or_arch_id", "metric_name"], keep="last")
    matrix = (
        subset.pivot_table(
            index="model_or_arch_id",
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
        )
        .sort_index()
    )

    targets = subset.groupby("model_or_arch_id")["target_value"].first().reindex(matrix.index)
    if targets.isna().any():
        raise ValueError("some RB40 models have no target_value; the ranking would be built on nothing")

    direction = (
        subset.drop_duplicates("metric_name").set_index("metric_name")["direction"].to_dict()
    )
    return matrix, targets, direction, repaired


def orient(matrix: pd.DataFrame, direction: dict[str, str]) -> pd.DataFrame:
    """Flip metrics declared 'lower' so that bigger always means more robust."""
    flipped = matrix.copy()
    for name in flipped.columns:
        if direction.get(name, "higher") == "lower":
            flipped[name] = -flipped[name]
    return flipped


def _rho_block(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman rho between a (n, m) block and a length-n target.

    Computed as Pearson correlation of average ranks, which is what ties
    require; the shortcut 1 - 6*sum(d^2)/(n(n^2-1)) silently misreports when
    two models share a value.
    """
    ranked_values = rankdata(values, axis=0)
    ranked_target = rankdata(target)
    centred_values = ranked_values - ranked_values.mean(axis=0)
    centred_target = ranked_target - ranked_target.mean()
    numerator = centred_values.T @ centred_target
    denominator = np.sqrt((centred_values ** 2).sum(axis=0) * (centred_target ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def bootstrap(
    matrix: pd.DataFrame,
    target: pd.Series,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Resample models and return per-metric rho intervals plus rank intervals."""
    values = matrix.to_numpy(dtype=float)
    labels = target.to_numpy(dtype=float)
    n_models = values.shape[0]
    if n_models < 3:
        raise ValueError("need at least three models to rank anything")

    # Metrics that are constant or absent across RB40 cannot be ranked; drop
    # them from the comparison instead of letting them sort as if they scored 0.
    usable = np.array([np.isfinite(values[:, j]).all() and np.nanstd(values[:, j]) > 0 for j in range(values.shape[1])])
    metrics = [matrix.columns[j] for j in range(len(usable)) if usable[j]]
    block = values[:, usable]

    point = _rho_block(block, labels)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, block.shape[1]), dtype=float)
    for draw in range(resamples):
        index = rng.integers(0, n_models, size=n_models)
        draws[draw] = _rho_block(block[index], labels[index])

    # Rank each draw independently; ties get the best rank (method='min') to
    # stay consistent with analyze_setting_weights.rank_grid.
    ranks = np.empty_like(draws)
    for draw in range(resamples):
        ranks[draw] = pd.Series(draws[draw]).rank(ascending=False, method="min").to_numpy()

    return {
        "n_models": int(n_models),
        "resamples": int(resamples),
        "seed": int(seed),
        "metrics": metrics,
        "dropped_metrics": [c for j, c in enumerate(matrix.columns) if not usable[j]],
        "point_rho": dict(zip(metrics, point.tolist())),
        "rho_low": dict(zip(metrics, np.nanpercentile(draws, CI[0], axis=0).tolist())),
        "rho_high": dict(zip(metrics, np.nanpercentile(draws, CI[1], axis=0).tolist())),
        "rank_median": dict(
            zip(metrics, np.nanmedian(ranks, axis=0).tolist())
        ),
        "rank_low": dict(zip(metrics, np.nanpercentile(ranks, CI[0], axis=0).tolist())),
        "rank_high": dict(zip(metrics, np.nanpercentile(ranks, CI[1], axis=0).tolist())),
        "point_rank": dict(
            zip(metrics, pd.Series(point).rank(ascending=False, method="min").tolist())
        ),
        "rho_draws_shape": list(draws.shape),
    }


def agreement_with_published(
    result: dict[str, Any],
    matrix: pd.DataFrame,
    published: pd.DataFrame,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Interval for Kendall tau between the RB40-only order and the published order.

    ``published`` must already be restricted to metrics present in the audit.
    """
    metrics = result["metrics"]
    published_rank = published.set_index("metric_name").loc[metrics, "strict_rank"].to_numpy(float)
    point_rank = np.array([result["point_rank"][m] for m in metrics], dtype=float)
    point_tau = kendalltau(point_rank, published_rank).statistic

    values = matrix[metrics].to_numpy(dtype=float)
    labels = result["_target"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed + 1)
    taus: list[float] = []
    n_models = values.shape[0]
    for _ in range(resamples):
        index = rng.integers(0, n_models, size=n_models)
        rho = _rho_block(values[index], labels[index])
        ranks = pd.Series(rho).rank(ascending=False, method="min").to_numpy(float)
        tau = kendalltau(ranks, published_rank).statistic
        if np.isfinite(tau):
            taus.append(float(tau))
    return {
        "point": float(point_tau),
        "low": float(np.percentile(taus, CI[0])),
        "high": float(np.percentile(taus, CI[1])),
        "draws": len(taus),
    }


def verdict(rho_low: float, rho_high: float) -> str:
    """Say only what an interval can support, and say which thing it supports."""
    if rho_high < 0:
        return "negative (whole interval below zero)"
    if rho_low > 0:
        return "positive (whole interval above zero)"
    return "inconclusive (interval covers zero)"


def analyze(
    rows_path: Path, table_path: Path, resamples: int, seed: int, repair_path: Path | None = None
) -> dict[str, Any]:
    matrix, targets, direction, repaired = load_rb40(rows_path, repair_path)
    oriented = orient(matrix, direction)
    result = bootstrap(oriented, targets, resamples=resamples, seed=seed)
    result["repaired_metrics"] = repaired

    published = pd.read_csv(table_path)
    published = published[published["strict_status"] == "ranked"].copy()
    common = [m for m in result["metrics"] if m in set(published["metric_name"])]
    result["metrics_not_on_board"] = [m for m in result["metrics"] if m not in set(published["metric_name"])]
    result["board_metrics_missing_rows"] = [
        m for m in published["metric_name"] if m not in set(result["metrics"])
    ]
    result["_target"] = targets
    result["agreement"] = agreement_with_published(
        {**result, "metrics": common}, oriented[common], published, resamples=resamples, seed=seed
    )
    del result["_target"]

    # Self-check: the recomputed point rho must land on the published RB40 column.
    # If it does not, everything downstream is describing a different dataset.
    errors = {}
    for name in common:
        row = published.loc[published["metric_name"] == name, "rb40_mature"]
        if len(row) and pd.notna(row.iloc[0]):
            errors[name] = abs(result["point_rho"][name] - float(row.iloc[0]))
    result["reproduction_max_abs_error"] = max(errors.values()) if errors else None
    result["reproduction_mismatches"] = sorted(
        name for name, err in errors.items() if err > REPRODUCTION_TOLERANCE
    )

    result["focus"] = {
        name: {
            "point_rho": result["point_rho"].get(name),
            "rho_interval": [result["rho_low"].get(name), result["rho_high"].get(name)],
            "point_rank": result["point_rank"].get(name),
            "rank_interval": [result["rank_low"].get(name), result["rank_high"].get(name)],
            "published_rank": (
                float(published.loc[published["metric_name"] == name, "strict_rank"].iloc[0])
                if (published["metric_name"] == name).any()
                else None
            ),
        }
        for name in FOCUS_METRICS
    }
    result["verdicts"] = {
        name: verdict(info["rho_interval"][0], info["rho_interval"][1])
        for name, info in result["focus"].items()
        if info["rho_interval"][0] is not None
    }
    result["inputs"] = {
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "table": str(table_path),
        "table_sha256": sha256_file(table_path),
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    agreement = result["agreement"]
    lines = [
        "# RB40 名次的置信区间（bootstrap，2026-11-12）",
        "",
        f"输入：`{result['inputs']['rows']}`（sha256 `{result['inputs']['rows_sha256'][:16]}…`）",
        f"，`{result['inputs']['table']}`。",
        "",
        f"重采样单位是**模型**而不是指标行：{result['n_models']} 个已训练防御模型，"
        f"有放回抽 {result['resamples']} 次，种子 {result['seed']}，取 2.5% / 97.5% 分位。",
        "",
        "## 最要紧的一句",
        "",
    ]
    tau_note = (
        f"RB40 单独排名与已发布八设置榜的一致度 τ = **{agreement['point']:+.3f}**，"
        f"95% 区间 [{agreement['low']:+.3f}, {agreement['high']:+.3f}]。"
    )
    if agreement["low"] < 0 < agreement["high"]:
        tau_note += "区间跨过 0，说明**不能断言两套排序反向**，只能说它们不一致。"
    elif agreement["high"] < 0:
        tau_note += "区间整体在 0 以下，**两套排序确实反向**。"
    else:
        tau_note += "区间整体在 0 以上。"
    lines += [tau_note, "", "## 重点指标", "", "| 指标 | RB40 点估计 | 95% 区间 | 结论 | RB40 名次（区间） | 已发布名次 |", "| --- | ---: | --- | --- | --- | ---: |"]
    for name, info in result["focus"].items():
        low, high = info["rho_interval"]
        if info["point_rho"] is None or low is None:
            continue
        lines.append(
            f"| `{name}` | {info['point_rho']:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            f"{result['verdicts'][name]} | {info['point_rank']:.0f} "
            f"({info['rank_interval'][0]:.0f}–{info['rank_interval'][1]:.0f}) | "
            f"{info['published_rank']:.0f} |"
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- **区间覆盖 0 = 不知道方向**。这时不能说\"该指标在已训练模型上失效\"，也不能说\"它有效\"，",
        "  只能说 40 个模型的证据不足以区分。要更窄的区间就得加模型，而不是加大重采样次数。",
        "- **名次区间比点名次重要**。若区间是 60–80，说\"第 73\"会给出并不存在的精度。",
        "- 这套区间只回答\"在 RB40 这 40 个模型上排得稳不稳\"，不回答\"换成另外 40 个模型会怎样\"。",
        "  后者需要新的模型集合，不是重采样能给的。",
        "- **区间宽不等于指标没用**。它反映的是\"40 个模型能提供多少信息\"：",
        "  一个真实相关 0.5 的指标，在 40 个模型上算出来，95% 区间大致就有 ±0.3 宽。",
        "",
        "## 为什么按模型重采样",
        "",
        f"每个模型对 84 个指标各贡献一行，这些行不是独立证据（同一张网络的不同测量）。"
        f"若按行重采样，等于把 1 个模型的 84 行当成 84 个独立样本，区间会被算得过窄。",
        "",
    ]
    if result.get("reproduction_max_abs_error") is not None:
        lines += [
            "## 复现核对",
            "",
            f"把算出的每个指标 RB40 相关系数与发布表 `rb40_mature` 列逐项相减："
            f"最大绝对误差 **{result['reproduction_max_abs_error']:.3e}**，"
            f"不一致指标 {len(result['reproduction_mismatches'])} 个。",
            "误差为 0 说明这份审计与冻结表算的是同一批数，不是另找了一份数据。",
            "",
        ]
    if result.get("board_metrics_missing_rows"):
        lines += [
            "## 榜上有、但本审计算不了的指标",
            "",
            "这些指标的原始行不在本机（只在服务器），所以没有区间，也不补数：",
            "",
        ]
        lines += [f"- `{name}`" for name in result["board_metrics_missing_rows"]]
        lines += [""]
    if result["dropped_metrics"]:
        lines += [
            "## 被排除的指标",
            "",
            "以下指标在 RB40 上没有可用数值（或全为常数），不参与排名，也不补零：",
            "",
        ]
        lines += [f"- `{name}`" for name in result["dropped_metrics"]]
        lines += [""]
    lines += [
        "## 这条检查能证明什么，不能证明什么",
        "",
        "**能**：给出\"RB40 上的名次\"这套说法的误差范围，从而判断哪些名次差别值得写进论文。",
        "",
        "**不能**：证明某个指标机制上更好；证明换一批模型后排序不变；"
        "替代已发布的八设置榜（那份协议没变，也不会变）。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="metric_values_original_strict.csv")
    parser.add_argument(
        "--repair",
        type=Path,
        default=None,
        help="repair_rows.csv with the recomputed TRNAS rows (default: next to --rows)",
    )
    parser.add_argument("--table", type=Path, required=True, help="86-row release table")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    repair = args.repair if args.repair is not None else args.rows.parent / "repair_rows.csv"
    result = analyze(args.rows, args.table, args.resamples, args.seed, repair)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(result), encoding="utf-8")

    agreement = result["agreement"]
    print(
        f"models={result['n_models']} metrics={len(result['metrics'])} "
        f"tau(rb40,published)={agreement['point']:+.3f} "
        f"[{agreement['low']:+.3f},{agreement['high']:+.3f}] "
        f"dropped={len(result['dropped_metrics'])}"
    )
    for name in ("trnas_r_score", "trnas_consistency_phi", "adversarial_loss"):
        info = result["focus"][name]
        if info["point_rho"] is not None:
            print(
                f"  {name}: rho={info['point_rho']:+.3f} "
                f"[{info['rho_interval'][0]:+.3f},{info['rho_interval'][1]:+.3f}] "
                f"rank={info['point_rank']:.0f} "
                f"({info['rank_interval'][0]:.0f}-{info['rank_interval'][1]:.0f}) "
                f"{result['verdicts'][name]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
