"""Draw the one-page picture behind docs/experiments/Ours指标分方向优势_20261118.md.

The table in that document answers "who wins each question".  A picture has to
answer the second half as well: no single metric wins everywhere.  So the
heatmap keeps the shortlist visible, normalised *inside each column* to that
column's best value (the eight questions have different scales and different
n), and the small bar panel says where each winner lands on the official
84-metric board.

Run:  python scripts/plot_ours_directions.py
Out:  figures/ours_directions.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "audits" / "ours-directions-20261118.json"
OUT = ROOT / "figures" / "ours_directions.png"

COLUMNS = [
    ("rb40_mature", "RB40\n已训练"),
    ("nb201_e0", "NB201 e0\n单次前向"),
    ("nb201_mean", "NB201\n五阶段"),
    ("darts_e100", "DARTS\ne100"),
    ("all_setting_mean", "八设置\n总榜"),
    ("rq2_partial", "真实预测\n400 架构"),
    ("seed_stability", "跨初始化\n稳定性"),
    ("selection_delta", "决策价值\n前 10%"),
]


def load(path: Path = REPORT) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def value_lookup(block: dict) -> dict[str, tuple[float, int, int]]:
    """metric -> (value, rank_ours, rank_all) for one direction."""
    return {row["metric"]: (row["value"], row["rank_ours"], row["rank_all"]) for row in block["top"]}


def short(name: str) -> str:
    """Keep ``feature_`` so ``feature_delta`` does not collapse into a bare ``delta``."""
    return name.replace("feature_measure_", "").replace("perturbation_capacity_", "cap:")


def select_metrics(report: dict) -> list[str]:
    """Top-2 of every question plus anything that reached four or more top-tens."""
    chosen: dict[str, int] = {}
    for block in report["directions"].values():
        for row in block["top"][:2]:
            chosen[row["metric"]] = chosen.get(row["metric"], 0) + 1
    for row in report["multi_direction"]:
        if row["n_directions"] >= 4:
            chosen.setdefault(row["metric"], row["n_directions"])
    return list(chosen)


def build_matrix(report: dict, metrics: list[str]) -> np.ndarray:
    values = np.full((len(metrics), len(COLUMNS)), np.nan)
    for j, (key, _) in enumerate(COLUMNS):
        lookup = value_lookup(report["directions"][key])
        for i, metric in enumerate(metrics):
            if metric in lookup:
                values[i, j] = lookup[metric][0]
    return values


def column_relative(values: np.ndarray) -> np.ndarray:
    """Divide every column by its own maximum.

    Ranking inside a column was the obvious choice, but a column with only two
    entries would then paint 0.96 as the palest cell of all.  Scaling to the
    column maximum keeps "close to that question's best" dark, whatever n is.
    """
    rel = np.full_like(values, np.nan)
    for j in range(values.shape[1]):
        column = values[:, j]
        mask = ~np.isnan(column)
        if not mask.any():
            continue
        top = float(np.max(column[mask]))
        if top > 0:
            rel[mask, j] = column[mask] / top
    return rel


def draw(out: Path = OUT) -> Path:
    report = load()
    metrics = select_metrics(report)
    values = build_matrix(report, metrics)
    rel = column_relative(values)

    score = np.nanmean(rel, axis=1)
    score = np.where(np.isnan(score), -1.0, score)
    order = np.argsort(-score)
    metrics = [metrics[i] for i in order]
    values, rel = values[order], rel[order]

    fig = plt.figure(figsize=(15.2, 8.4), dpi=190)
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=(2.15, 1.0), height_ratios=(1.0, 0.62),
        left=0.245, right=0.985, top=0.845, bottom=0.115,
        wspace=0.60, hspace=0.55,
    )

    ax = fig.add_subplot(grid[:, 0])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ededed")
    im = ax.imshow(np.ma.masked_invalid(rel), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([label for _, label in COLUMNS], fontsize=8.6, linespacing=1.4)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([short(m) for m in metrics], fontsize=8.6)
    ax.tick_params(axis="x", pad=6)

    for i in range(len(metrics)):
        for j in range(len(COLUMNS)):
            value = values[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8.2, color="#a0a0a0")
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.0,
                    color="white" if rel[i, j] > 0.72 else "#1f2f3d")

    ax.set_xticks(np.arange(-0.5, len(COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label("相对该方向第一名（深 = 接近该问的最强）", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Right top: where each winner lands on the official board.
    axr = fig.add_subplot(grid[0, 1])
    winners = []
    for key, label in COLUMNS:
        champion = report["directions"][key]["champion"]
        if champion:
            winners.append((label, champion["metric"], champion["rank_all"],
                            report["directions"][key]["n_all"]))
    winners.sort(key=lambda row: row[2])
    ys = np.arange(len(winners))
    ranks = [row[2] for row in winners]
    axr.barh(ys, ranks, color="#3f6d9e", height=0.58)
    axr.set_yticks(ys)
    axr.set_yticklabels([f"{label.replace(chr(10), ' ')}\n{short(metric)}" for label, metric, _, _ in winners],
                        fontsize=7.3, linespacing=1.5)
    axr.invert_yaxis()
    axr.set_xlabel("自研冠军在「全榜」上的名次（越靠左越好）", fontsize=8.4, labelpad=6)
    axr.set_title("同是「自研第一」，全榜位置差很多", fontsize=9.8, pad=9, loc="left")
    axr.set_xlim(0, max(ranks) + 3.4)
    for y, row in zip(ys, winners):
        axr.text(row[2] + 0.3, y, f"#{row[2]}/{row[3]}", va="center", fontsize=8.0, color="#22384d")
    axr.tick_params(axis="x", length=0, labelsize=7.8)
    axr.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        axr.spines[side].set_visible(False)
    axr.grid(axis="x", color="#dcdcdc", linewidth=0.7)
    axr.set_axisbelow(True)

    # Right bottom: the point of the exercise.
    axn = fig.add_subplot(grid[1, 1])
    axn.axis("off")
    axn.text(0.0, 1.02,
             "怎么读\n"
             "① 每列以该方向第一名归一成 1.0。八个问题的目标量、\n"
             "    样本量都不同，跨列比深浅没有意义。\n"
             "② 没有指标在八列都深：RB40 冠军是 adv_excess，\n"
             "    真实预测冠军是 robust_delta，稳定性冠军是\n"
             "    flow_rel_allpairs —— 三个方向三套答案。\n"
             "③ 右上图才是关键：同是「自研第一」，\n"
             "    有的落在全榜第 1，有的只到第 7。\n"
             "    总榜 62.5% 的权重给了 NB201，\n"
             "    落差就是「平均掩盖专长」的证据。",
             ha="left", va="top", fontsize=8.4, linespacing=1.72,
             color="#1f2f3d", transform=axn.transAxes)

    fig.suptitle("自研指标的分方向优势：换一个问题，就换一个冠军",
                 fontsize=16, x=0.245, ha="left", y=0.962)
    fig.text(0.245, 0.902,
             "69 个自研指标 × 8 个问题。单元格数字 = 该指标在该方向的分数（越大越好，lower 类已翻号）；"
             "「—」= 该方向未覆盖，不是 0。",
             fontsize=9.3, color="#44586a", ha="left")
    fig.text(0.245, 0.028,
             "数据：results/audits/ours-directions-20261118.json —— 只重排已有结果，不含新测量。"
             "边界：RB40 的 target 混 Linf/L2；e0 三列用的是发布表，而发布表与当前代码重算不一致，只能作相对参考。",
             fontsize=7.8, color="#6b7a88", ha="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}  metrics={len(metrics)}")
    return out


if __name__ == "__main__":
    draw()
