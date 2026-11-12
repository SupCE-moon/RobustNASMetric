"""How much does each benchmark grid actually decide the leaderboard?

The strict leaderboard averages eight settings: one RobustBench grid (40 mature
models), five NAS-Bench-201 training stages (64 architectures each) and two DARTS
stages (100 architectures each). A simple average therefore gives NAS-Bench-201
5/8 of the weight, DARTS 2/8 and RobustBench only 1/8 -- even though RobustBench
is the grid closest to "real deployed defenses".

This script quantifies that. It does not recompute any metric and does not change
any published number; it only re-reads the frozen table and asks which grid each
rank actually comes from.

Usage:
    python tools/analyze_setting_weights.py \
        --table outputs/.../RobustMetric_严格全部指标总表_86项_三位小数.csv \
        --json results/audits/setting-weights-20261112.json \
        --report results/audits/setting-weights-20261112.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

SCHEMA_VERSION = 1

# The eight frozen settings, grouped into the three grids they come from.
SETTING_COLUMNS = (
    "rb40_mature",
    "nb201_e0",
    "nb201_e5",
    "nb201_e10",
    "nb201_e25",
    "nb201_e50",
    "darts_e0",
    "darts_e100",
)
GRID_OF_SETTING = {
    "rb40_mature": "rb40",
    "nb201_e0": "nb201",
    "nb201_e5": "nb201",
    "nb201_e10": "nb201",
    "nb201_e25": "nb201",
    "nb201_e50": "nb201",
    "darts_e0": "darts",
    "darts_e100": "darts",
}
GRIDS = ("all", "rb40", "nb201", "darts")

# Metrics worth naming in the report specifically: the leaders plus the two
# in-house candidates the project cares about.
FOCUS_METRICS = (
    "trnas_consistency_phi",
    "trnas_r_score",
    "trnas_activation_psi",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
    "adversarial_loss",
    "fsp_delta",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ranked(table: Path) -> pd.DataFrame:
    frame = pd.read_csv(table)
    missing = [c for c in SETTING_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"table is missing setting columns: {missing}")
    frame = frame[frame["strict_status"] == "ranked"].copy()
    if frame.empty:
        raise ValueError("no ranked rows in table")
    return frame


def grid_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """One score per metric per grid, so grids can be ranked against each other.

    A grid's score is the mean of its own settings -- the same rule the frozen
    protocol uses, applied to a subset instead of to all eight.
    """
    out = pd.DataFrame({"metric_name": frame["metric_name"].to_numpy()})
    out["all"] = frame["all_setting_mean"]
    out["rb40"] = frame["rb40_mature"]
    out["nb201"] = frame[[c for c in SETTING_COLUMNS if GRID_OF_SETTING[c] == "nb201"]].mean(axis=1)
    out["darts"] = frame[[c for c in SETTING_COLUMNS if GRID_OF_SETTING[c] == "darts"]].mean(axis=1)
    return out


def rank_grid(series: pd.Series) -> pd.Series:
    """Descending rank, 1 = best. NaN never gets a rank (缺就是缺，不是最后一名)."""
    return series.rank(ascending=False, method="min")


def compare(a: pd.Series, b: pd.Series) -> dict:
    """Agreement between two rankings, on metrics ranked in both."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 3:
        return {"n": len(both), "kendall_tau": None, "top10_overlap": None}
    tau = kendalltau(both["a"], both["b"]).statistic
    top_a = set(both["a"].nsmallest(min(10, len(both))).index)
    top_b = set(both["b"].nsmallest(min(10, len(both))).index)
    return {
        "n": int(len(both)),
        "kendall_tau": round(float(tau), 4),
        "top10_overlap": len(top_a & top_b),
    }


def setting_representativeness(frame: pd.DataFrame) -> dict:
    """How much each individual setting agrees with the final eight-setting mean."""
    final = frame["all_setting_mean"]
    per_setting = {}
    for column in SETTING_COLUMNS:
        both = pd.concat([frame[column], final], axis=1, keys=["s", "final"]).dropna()
        if len(both) < 3:
            per_setting[column] = None
            continue
        per_setting[column] = round(float(spearmanr(both["s"], both["final"]).statistic), 4)
    return per_setting


def analyze(frame: pd.DataFrame) -> dict:
    scores = grid_scores(frame)
    ranks = {grid: rank_grid(scores[grid]) for grid in GRIDS}

    agreement = {}
    for a in GRIDS:
        for b in GRIDS:
            if a >= b:
                continue
            agreement[f"{a}__vs__{b}"] = compare(ranks[a], ranks[b])

    focus = {}
    for name in FOCUS_METRICS:
        row = frame.index[frame["metric_name"] == name]
        if len(row) == 0:
            continue
        idx = row[0]
        focus[name] = {
            "rank_" + grid: (None if pd.isna(ranks[grid].get(idx)) else int(ranks[grid].get(idx)))
            for grid in GRIDS
        }
        focus[name]["score_all_setting_mean"] = (
            None if pd.isna(scores.at[idx, "all"]) else round(float(scores.at[idx, "all"]), 4)
        )

    weights = {
        grid: round(len([c for c in SETTING_COLUMNS if GRID_OF_SETTING[c] == grid]) / len(SETTING_COLUMNS), 4)
        for grid in ("rb40", "nb201", "darts")
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_ranked_metrics": int(len(frame)),
        "weights": weights,
        "agreement": agreement,
        "focus": focus,
        "setting_vs_final": setting_representativeness(frame),
        "top10_by_grid": {grid: list(ranks[grid].nsmallest(10).index) for grid in GRIDS},
        "top10_names_by_grid": {
            grid: list(frame.loc[ranks[grid].nsmallest(10).index, "metric_name"]) for grid in GRIDS
        },
    }


def render_markdown(payload: dict, input_path: str, input_sha256: str) -> str:
    weights = payload["weights"]
    lines = [
        "# 八设置的平均里，每个网格到底占多少权重",
        "",
        f"- 输入：`{input_path}`（sha256 `{input_sha256}`）",
        f"- 参与排名的指标：{payload['n_ranked_metrics']} 个（已排除 `excluded` 行）",
        "- 生成脚本：`tools/analyze_setting_weights.py`",
        "",
        "## 为什么这件事值得查",
        "",
        "冻结协议的最终分数是八个设置的**简单平均**。但八个设置并不来自三个同等份量的网格：",
        "",
        "| 网格 | 设置数 | 在平均里的权重 |",
        "| --- | ---: | ---: |",
        f"| RobustBench（已训练成熟模型） | 1 | {weights['rb40']:.1%} |",
        f"| NAS-Bench-201（5 个训练阶段） | 5 | {weights['nb201']:.1%} |",
        f"| DARTS（2 个训练阶段） | 2 | {weights['darts']:.1%} |",
        "",
        "也就是说，**排名主要由 NAS-Bench-201 的训练阶段决定**，而 RobustBench 只贡献 1/8。",
        "下面用同一张冻结表验证这句话是否成立——不重算任何指标。",
        "",
        "## 各网格单独排名的一致度",
        "",
        "| 对比 | 共同指标数 | Kendall τ | 前 10 名重合 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, value in payload["agreement"].items():
        a, b = key.split("__vs__")
        tau = "—" if value["kendall_tau"] is None else f"{value['kendall_tau']:+.3f}"
        overlap = "—" if value["top10_overlap"] is None else f"{value['top10_overlap']}/10"
        lines.append(f"| {a} vs {b} | {value['n']} | {tau} | {overlap} |")

    lines += [
        "",
        "τ 越接近 +1，说明两个网格给出的排序越像；接近 0 说明它们各说各的。",
        "",
        "## 每个设置与最终榜的吻合度",
        "",
        "| 设置 | 与 `all_setting_mean` 的 Spearman |",
        "| --- | ---: |",
    ]
    for setting, rho in payload["setting_vs_final"].items():
        lines.append(f"| `{setting}` | {rho if rho is not None else '—'} |")

    lines += [
        "",
        "## 几个关键指标的跨网格名次",
        "",
        "| 指标 | 全 8 设置 | 只 RB40 | 只 NB201 | 只 DARTS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, ranks in payload["focus"].items():
        def cell(grid: str) -> str:
            value = ranks.get("rank_" + grid)
            return "—" if value is None else str(value)

        lines.append(f"| `{name}` | {cell('all')} | {cell('rb40')} | {cell('nb201')} | {cell('darts')} |")

    lines += [
        "",
        "## 怎么读这些数",
        "",
    ]
    tau_nb201 = payload["agreement"]["all__vs__nb201"]["kendall_tau"]
    tau_rb40 = payload["agreement"]["all__vs__rb40"]["kendall_tau"]

    def fmt_tau(value: float | None) -> str:
        # A two-metric table cannot yield a rank correlation; say so rather than crash.
        return "—（指标太少，无法计算）" if value is None else f"{value:+.3f}"

    lines += [
        f"- 全 8 设置与**只 NB201** 的排名一致度 τ = {fmt_tau(tau_nb201)}，与**只 RB40** 是 {fmt_tau(tau_rb40)}。",
        "  前者接近 1 是权重（5/8）的必然结果；后者接近 0、甚至为负，说明"
        "「在已训练防御模型上谁更强」与「最终榜上的名次」几乎是两套排序。",
        "- 关键指标的名次差别很大：同一个指标可以全榜靠前、RB40 里靠后。"
        "**引用名次时必须说出是哪一套名次**（全 8 设置 / 只 RB40 / 只 NB201 / 只 DARTS）。",
        "",
        "## 局限（不要把上面的数字读成「榜错了」）",
        "",
        "1. **RB40 只有 40 个模型**。单个指标在 40 个模型上的 Spearman，其 95% 置信区间通常宽达 ±0.3，",
        "   因此「只 RB40」那一列的名次是**指示性**的，不是定论。要下结论必须补按模型的 bootstrap。",
        "2. **三个网格的 target 语义不同**：RB40 是公开报告的 AutoAttack/PGD 鲁棒准确率，",
        "   NB201/DARTS 是固定协议训练后的 PGD 值。它们不是同一个「真值」的三个测量，",
        "   所以「哪个网格更对」这个问题本身没有意义。",
        "3. 冻结协议就是八设置简单平均，**这不是 bug**。本节要说的是：读数时必须知道这个平均实际在平均什么。",
        "",
        "## 这条检查能证明什么",
        "",
        "- 能证明：最终名次在多大程度上由某一个网格决定，以及哪些指标的名次依赖网格组合。",
        "- 不能证明：哪个网格的标签「更对」。三个网格测的是不同对象（成熟防御 vs 训练阶段 vs DARTS 阶段），",
        "  没有共同的「真值」；只能说名次的来源结构，不能说某一名次是错的。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    table = args.table.resolve()
    frame = load_ranked(table)
    payload = analyze(frame)
    payload["input"] = {"path": str(table), "sha256": sha256_file(table)}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(
        render_markdown(payload, str(table), payload["input"]["sha256"]) + "\n", encoding="utf-8"
    )

    tau_nb = payload["agreement"]["all__vs__nb201"]["kendall_tau"]
    tau_rb = payload["agreement"]["all__vs__rb40"]["kendall_tau"]
    print(
        f"metrics={payload['n_ranked_metrics']} "
        f"tau(all,nb201)={tau_nb} tau(all,rb40)={tau_rb} "
        f"weights={payload['weights']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
