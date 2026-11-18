"""按方向给自研（Ours）指标排座次。

动机：84 项总榜是"八设置平均"，它回答的是"总体谁最强"，会把
"在某一类条件下最好"的指标埋掉。导师的意见是——不必样样都强，
找出 Ours 在哪些方向上领先，本身就是一个可写的点。

本脚本把已有的六份证据拼在一起，对每个指标问八个不同的问题：
  D1 已训练防御模型（RB40）上排序能力如何
  D2 早期架构单次前向（NB201 e0）上如何
  D3 早期架构五阶段平均如何
  D4 DARTS 搜索空间上如何
  D5 八设置总榜如何
  D6 400 个架构上控掉 clean 与参数量之后还剩多少真实预测力
  D7 换随机初始化种子后还稳不稳
  D8 按它挑前 10% 去训练，比随机挑高多少

D6/D7/D8 只覆盖当前 MetricResult 能算出来的 69 个指标，其余标 n/a。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = 1
TOP_K = 10
OURS = "Ours"

# name: (列名或来源, 中文标签, 一句话说明)
DIRECTIONS: tuple[tuple[str, str, str], ...] = (
    ("rb40_mature", "已训练防御模型（RB40，40 个 RobustBench 模型）", "对已经训练好的防御模型排序"),
    ("nb201_e0", "早期架构·单次前向（NB201 e0）", "只跑一次随机初始化的前向"),
    ("nb201_mean", "早期架构·五阶段平均（NB201）", "跨训练阶段的平均表现"),
    ("darts_e100", "DARTS 搜索空间（e100）", "换一个搜索空间是否还成立"),
    ("all_setting_mean", "八设置总榜", "官方总榜口径"),
    ("rq2_partial", "400 架构真实预测力（控 clean + 参数量）", "最接近“能不能预测”的那个数"),
    ("seed_stability", "跨初始化稳定性（值稳定 rho）", "换个随机种子序还一样吗"),
    ("selection_delta", "决策价值（挑前 10% 的提升）", "真拿它当筛选器能赚多少"),
)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ours_names(table: pd.DataFrame) -> set[str]:
    return set(table.loc[table["source_type"] == OURS, "metric_name"].astype(str))


def scores_from_table(table: pd.DataFrame, column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in zip(table["metric_name"].astype(str), table[column], strict=False):
        if pd.isna(value):
            continue
        values[name] = float(value)
    return values


def scores_from_confounders(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in report.get("metrics", []):
        value = row.get("rho_clean_params")
        if value is None:
            continue
        out[str(row["metric"])] = float(value)
    return out


def scores_from_seed(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in report.get("metrics", []):
        value = (row.get("value") or {}).get("mean_rho")
        if value is None:
            continue
        out[str(row["metric"])] = float(value)
    return out


def scores_from_selection(report: dict[str, Any], k_percent: int = 10) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in report.get("rows", []):
        if int(row.get("k_percent", -1)) != k_percent:
            continue
        value = row.get("delta")
        if value is None:
            continue
        out[str(row["metric"])] = float(value)
    return out


def rank_block(scores: dict[str, float], ours: set[str], top_k: int = TOP_K) -> dict[str, Any]:
    """降序排名；方向已在数据侧统一成“越大越好”。"""
    items = sorted(((m, v) for m, v in scores.items() if v == v), key=lambda kv: kv[1], reverse=True)
    all_rank = {m: i + 1 for i, (m, _) in enumerate(items)}
    ours_items = [(m, v) for m, v in items if m in ours]
    ours_rank = {m: i + 1 for i, (m, _) in enumerate(ours_items)}
    top = [
        {
            "metric": m,
            "value": round(v, 4),
            "rank_ours": ours_rank[m],
            "rank_all": all_rank[m],
        }
        for m, v in ours_items[:top_k]
    ]
    return {
        "n_all": len(items),
        "n_ours": len(ours_items),
        "top": top,
        "champion": top[0] if top else None,
        "best_all_rank": min((r["rank_all"] for r in top), default=None),
    }


def analyse(
    table: pd.DataFrame,
    confounders: dict[str, Any],
    seed: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    ours = ours_names(table)
    sources: dict[str, dict[str, float]] = {
        "rq2_partial": scores_from_confounders(confounders),
        "seed_stability": scores_from_seed(seed),
        "selection_delta": scores_from_selection(selection),
    }
    directions: dict[str, Any] = {}
    for name, label, why in DIRECTIONS:
        scores = sources.get(name)
        if scores is None:
            scores = scores_from_table(table, name)
        block = rank_block(scores, ours)
        block["label"] = label
        block["why"] = why
        directions[name] = block

    # 多面手：在几个方向上进过 Ours 前 10
    hits: dict[str, list[str]] = {}
    for name, block in directions.items():
        for row in block["top"]:
            hits.setdefault(row["metric"], []).append(name)
    multi = sorted(
        (
            {"metric": m, "n_directions": len(ds), "directions": ds}
            for m, ds in hits.items()
        ),
        key=lambda r: (-r["n_directions"], r["metric"]),
    )

    # 每个方向的空白：Ours 最好成绩在全榜的位置
    gaps = [
        {
            "direction": name,
            "label": block["label"],
            "best_ours": (block["champion"] or {}).get("metric"),
            "best_ours_value": (block["champion"] or {}).get("value"),
            "best_ours_rank_all": block["best_all_rank"],
            "n_all": block["n_all"],
        }
        for name, block in directions.items()
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "n_ours": len(ours),
        "n_metrics": int(len(table)),
        "top_k": TOP_K,
        "directions": directions,
        "multi_direction": multi,
        "gaps": gaps,
        "sources": {
            "table": str(table.attrs.get("path", "")),
            "confounders_n": confounders.get("n_architectures"),
            "seed_n": seed.get("n_architectures"),
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 自研指标的分方向对比（2026-11-18）\n")
    lines.append(
        f"总榜 84 项里自研 {report['n_ours']} 项、已有方法 15 项。"
        "总榜是八设置平均，会把“只在某一类条件下强”的指标埋掉；"
        f"下面按 {len(DIRECTIONS)} 个方向分别排座次（每列只在小范围内比较）。\n"
    )
    lines.append("## 1 每个方向的冠军\n")
    lines.append("| 方向 | 自研冠军 | 分数 | 自研内名次 | 全榜名次 | 全榜数量 |")
    lines.append("|---|---|---|---|---|---|")
    for gap in report["gaps"]:
        lines.append(
            f"| {gap['label']} | `{gap['best_ours']}` | {_fmt(gap['best_ours_value'])} | "
            f"1 | {gap['best_ours_rank_all']} | {gap['n_all']} |"
        )
    lines.append("")

    for name, label, why in DIRECTIONS:
        block = report["directions"][name]
        lines.append(f"## {label}\n")
        lines.append(f"> {why}。共 {block['n_all']} 个指标有分数，其中自研 {block['n_ours']} 个。\n")
        if not block["top"]:
            lines.append("无数据。\n")
            continue
        lines.append("| 自研内名次 | 指标 | 分数 | 全榜名次 |")
        lines.append("|---|---|---|---|")
        for row in block["top"]:
            lines.append(
                f"| {row['rank_ours']} | `{row['metric']}` | {_fmt(row['value'])} | {row['rank_all']} |"
            )
        lines.append("")

    lines.append("## 多面手：在几个方向上都进过自研前 10\n")
    lines.append("| 指标 | 方向数 | 命中的方向 |")
    lines.append("|---|---|---|")
    for row in report["multi_direction"][:15]:
        lines.append(f"| `{row['metric']}` | {row['n_directions']} | {', '.join(row['directions'])} |")
    lines.append("")
    lines.append("## 怎么读\n")
    lines.append(
        "- 每张表都是“降序、越大越好”，方向符号已经统一过（`lower` 类指标在数据侧翻成正号）。\n"
        "- “全榜名次”是 84 项里的位置，可以看出自研在这个方向有没有输给外部方法。\n"
        "- 只覆盖当前 `MetricResult` 能算出的 69 个指标的三个方向（D6/D7/D8），"
        "其余标 `—`，不是 0。\n"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--confounders", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    table = load_table(args.table)
    table.attrs["path"] = str(args.table)
    report = analyse(
        table,
        load_json(args.confounders),
        load_json(args.seed),
        load_json(args.selection),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
