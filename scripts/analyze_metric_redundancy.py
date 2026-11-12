"""Measure how redundant the 85 strict-benchmark metrics are with each other.

Why this exists
---------------
The strict leaderboard ranks 84 metrics, but most of them are variations of the
same measurement (e.g. ``feature_measure_robust_hybrid_rel_delta`` and
``feature_measure_robust_hybrid_delta``). If the leaderboard is really driven by
only a handful of independent signals, then "84 metrics" overstates how much
evidence the ranking rests on -- and a reviewer will ask exactly that.

What it computes
----------------
1. For every setting, orient each metric so that "larger = more robust"
   (``direction == "lower"`` is multiplied by -1) and build the model x metric
   matrix, then the Spearman correlation matrix between metrics.
2. Average those matrices across the eight settings (``nanmean``).
3. Average-linkage hierarchical clustering on ``1 - rho``.
4. Report, per metric, its mean correlation with all other metrics, and the
   clusters at several cut heights.

This is a descriptive analysis of metrics that already exist; it introduces no
new metric and trains no model. It cannot tell you which metric is *correct* --
only which ones carry near-duplicate information.

Usage
-----
    python scripts/analyze_metric_redundancy.py \
        --rows outputs/.../sources/metric_values_original_strict.csv \
        --json results/audits/metric-redundancy-20261112.json \
        --report results/audits/metric-redundancy-20261112.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SETTINGS = (
    "rb40_mature",
    "nb201_e0",
    "nb201_e5",
    "nb201_e10",
    "nb201_e25",
    "nb201_e50",
    "darts_e0",
    "darts_e100",
)
MIN_MODELS = 8
CUT_HEIGHTS = (0.40, 0.50, 0.60)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = {"setting", "metric_name", "model_or_arch_id", "status", "metric_value", "direction"}
    missing -= set(frame.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    frame = frame[frame["status"].astype(str).str.strip() == "ok"].copy()
    signs = {"lower": -1.0, "higher": 1.0}
    direction = frame["direction"].astype(str).str.strip().str.lower()
    unknown = sorted(set(direction) - set(signs))
    if unknown:
        raise SystemExit(f"unknown direction values: {unknown}")
    frame["oriented"] = frame["metric_value"].astype(float) * direction.map(signs)
    return frame


def correlation_matrices(frame: pd.DataFrame) -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    """Return per-setting Spearman matrices on a shared metric axis."""
    metrics = sorted(frame["metric_name"].unique())
    matrices: list[np.ndarray] = []
    used: list[str] = []
    skipped: list[str] = []
    for setting in SETTINGS:
        subset = frame[frame["setting"] == setting]
        pivot = subset.pivot_table(index="model_or_arch_id", columns="metric_name", values="oriented")
        pivot = pivot.reindex(columns=metrics)
        if len(pivot) < MIN_MODELS:
            skipped.append(f"{setting} (only {len(pivot)} models)")
            continue
        matrices.append(pivot.corr(method="spearman").to_numpy())
        used.append(setting)
    if not matrices:
        raise SystemExit("no setting had enough models to correlate")
    return matrices, metrics, skipped, used


def coverage_by_metric(frame: pd.DataFrame) -> dict[str, int]:
    """How many of the eight settings carry a *finite* value for each metric.

    A row can be ``status == "ok"`` and still hold no number (the frozen protocol
    marks a metric ok but records a blank value when the raw run produced one).
    Counting rows alone would then treat "no data" as "a number near zero", which
    is exactly the failure mode the leaderboard audit had to guard against.
    """
    subset = frame[frame["setting"].isin(SETTINGS) & frame["metric_value"].notna()]
    counts = subset.groupby("metric_name")["setting"].nunique()
    return {str(name): int(count) for name, count in counts.items()}


def mean_matrix(matrices: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(matrices, axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    np.fill_diagonal(mean, 1.0)
    return (mean + mean.T) / 2.0


def cluster(mean: np.ndarray, metrics: list[str], height: float) -> list[dict]:
    distance = 1.0 - mean
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=height, criterion="distance")
    mean_rho = (mean.sum(axis=1) - 1.0) / (len(metrics) - 1)
    frame = pd.DataFrame({"metric": metrics, "cluster": labels, "mean_rho": mean_rho})
    groups = []
    for label, group in frame.groupby("cluster"):
        ordered = group.sort_values("mean_rho", ascending=False)
        groups.append(
            {
                "cluster": int(label),
                "size": int(len(ordered)),
                "members": ordered["metric"].tolist(),
                "mean_rho": {
                    row.metric: round(float(row.mean_rho), 4) for row in ordered.itertuples()
                },
            }
        )
    groups.sort(key=lambda item: (-item["size"], item["cluster"]))
    return groups


def tag(name: str, payload: dict) -> str:
    """Metric name, flagged when its correlation rests on fewer than eight settings."""
    coverage = payload["coverage"].get(name, 0)
    settings = len(payload["settings_used"])
    if coverage >= settings:
        return f"`{name}`"
    return f"`{name}` ⚠{coverage}/{settings}"


def render_markdown(payload: dict) -> str:
    lines = [
        "# 指标冗余度：84 项严格榜里到底有几个独立信号",
        "",
        f"- 输入：`{payload['input']['path']}`（sha256 `{payload['input']['sha256']}`）",
        f"- 参与统计的设置：{', '.join(payload['settings_used'])}",
        f"- 输入里出现的指标：{payload['metrics_in_input']} 个；"
        f"实际参与统计：{payload['metric_count']} 个",
        f"- 生成脚本：`scripts/analyze_metric_redundancy.py`",
        "",
        "## 方法",
        "",
        "先按 `direction` 把每个指标统一成「越大越好」，再在每个设置内算指标两两的 "
        "Spearman 秩相关，然后对八个设置取平均，最后用平均连接（average linkage）"
        "对 `1 - rho` 聚类。相关矩阵按指标对逐对平均，某个指标在某设置里缺值时"
        "不会污染其他指标对。",
        "",
    ]
    if payload["unavailable_metrics"]:
        lines += [
            "## 无可用数据、被排除的指标",
            "",
            "下面这些指标在八个设置里都没有有限值，因此不参与任何统计。"
            "**它们不是「算出来很低」，而是「没有数据」**，必须与低相关区分开。",
            "",
            "| 指标 | 有数据的设置数 |",
            "| --- | --- |",
        ]
        for item in payload["unavailable_metrics"]:
            lines.append(f"| `{item['metric']}` | {item['settings_with_rows']} |")
        lines += [""]

    partial = [
        (name, payload["coverage"].get(name, 0))
        for name, _ in payload["mean_rho_ranked"]
        if payload["coverage"].get(name, 0) < len(payload["settings_used"])
    ]
    if partial:
        lines += [
            "## 覆盖不足的指标（相关只基于部分设置）",
            "",
            f"冻结协议要求八个设置齐全才入榜，所以下面这些指标在正式榜单里被整项排除；"
            f"但它们在部分设置上确实有值，因此仍参与了本报告的冗余度统计。"
            f"**读它们的相关系数时必须记住样本少**——尤其只有 1 个设置的指标，"
            f"相关系数只由 40 或 100 个模型决定，和其余指标的 560 个模型不可直接比较。",
            "",
            "| 指标 | 有数据的设置数 |",
            "| --- | --- |",
        ]
        for name, count in partial:
            lines.append(f"| `{name}` | {count}/{len(payload['settings_used'])} |")
        lines += [""]
    others = payload["metric_count"] - 1
    lines += [
        "## 与其他指标的平均相关（最高的 12 个）",
        "",
        f"| 指标 | 与其他 {others} 项的平均 rho |",
        "| --- | --- |",
    ]
    top = payload["mean_rho_ranked"][:12]
    for name, value in top:
        lines.append(f"| {tag(name, payload)} | {value:+.3f} |")
    lines += [
        "",
        "平均相关越高，说明这个指标越「没有新信息」——它和榜上其他指标测的是同一件事。",
        "",
        "## 与其他指标的平均相关（最低的 8 个）",
        "",
        f"| 指标 | 与其他 {others} 项的平均 rho |",
        "| --- | --- |",
    ]
    for name, value in payload["mean_rho_ranked"][-8:][::-1]:
        lines.append(f"| {tag(name, payload)} | {value:+.3f} |")
    lines += ["", "负值意味着这个指标与榜单主流方向相反（低相关不等于它不好，可能恰恰是独立信息）。", ""]

    for cut in payload["cuts"]:
        lines += [
            f"## 切树高度 rho >= {cut['min_rho']:.2f} → {cut['cluster_count']} 个簇",
            "",
            "| 簇 | 成员数 | 代表指标（簇内平均相关最高） | 成员 |",
            "| --- | --- | --- | --- |",
        ]
        for group in cut["clusters"]:
            representative = group["members"][0]
            members = ", ".join(f"`{name}`" for name in group["members"])
            lines.append(
                f"| {group['cluster']} | {group['size']} | `{representative}` | {members} |"
            )
        lines.append("")
    lines += [
        "## 结论边界",
        "",
        "- 这是**描述性**分析：它说明哪些指标彼此携带重复信息，**不能**说明哪个指标"
        "「正确」或「更好」。",
        "- 高相关不等于因果。两个指标高度相关，可能是因为它们都由同一个更底层的性质"
        "（比如特征漂移的幅度）驱动。",
        "- 簇的数量依赖切树高度，报告里给了 0.40 / 0.50 / 0.60 三档，不做单一断言。",
        "- 相关是在「模型之间的排序」上算的，因此只对「用指标给模型排序」这个用途有效。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    frame = load_rows(args.rows)
    coverage = coverage_by_metric(frame)
    all_metrics = sorted(frame["metric_name"].unique())
    unavailable = [name for name in all_metrics if coverage.get(name, 0) == 0]
    if unavailable:
        frame = frame[~frame["metric_name"].isin(unavailable)]
    matrices, metrics, skipped, used = correlation_matrices(frame)
    mean = mean_matrix(matrices)
    mean_rho = (mean.sum(axis=1) - 1.0) / (len(metrics) - 1)
    ranked = sorted(zip(metrics, mean_rho.tolist()), key=lambda item: -item[1])

    payload = {
        "schema_version": 1,
        "input": {
            "path": str(args.rows),
            "sha256": sha256_file(args.rows),
            "rows_used": int(len(frame)),
        },
        "settings_used": used,
        "settings_skipped": skipped,
        "metric_count": len(metrics),
        "metrics_in_input": len(all_metrics),
        "unavailable_metrics": [
            {"metric": name, "settings_with_rows": coverage.get(name, 0)} for name in unavailable
        ],
        "coverage": coverage,
        "mean_rho_ranked": [[name, round(value, 4)] for name, value in ranked],
        "cuts": [
            {
                "min_rho": round(1.0 - height, 2),
                "height": height,
                "cluster_count": len(groups := cluster(mean, metrics, height)),
                "clusters": groups,
            }
            for height in CUT_HEIGHTS
        ],
        "correlation_matrix": {
            "metrics": metrics,
            "mean_rho": np.round(mean, 4).tolist(),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")

    print(
        f"metrics={len(metrics)} of {len(all_metrics)} settings={len(matrices)} "
        f"unavailable={len(unavailable)} skipped={skipped}"
    )
    print("most redundant:", ", ".join(f"{n} {v:+.3f}" for n, v in ranked[:3]))
    print("least redundant:", ", ".join(f"{n} {v:+.3f}" for n, v in ranked[-3:]))
    for cut in payload["cuts"]:
        print(f"rho>={cut['min_rho']:.2f}: {cut['cluster_count']} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
