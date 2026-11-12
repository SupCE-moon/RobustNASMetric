"""Compare the signed and absolute capacity-drift metrics.

`metrics.py:1265` and `:1286` compute two numbers from the same quantity:

    perturbation_capacity_delta        mean(|adv - clean| / |clean|)   -> always >= 0
    perturbation_capacity_signed_delta mean((adv - clean) / |clean|)   -> may be either sign

They differ by one `abs()`, so it is natural to assume they rank models the same
way up to a sign.  This audit checks that assumption against the released rows,
because both appear on the leaderboard.

The interesting failure mode is sign cancellation: averaging signed relative
changes lets samples where capacity grew cancel those where it shrank, so the
signed value measures "how many samples moved down minus how many moved up",
not "how far they moved".  When that happens the two metrics are not two views
of one thing, and the signed one's declared `lower` direction means something
different from its neighbours'.

Usage:

    python tools/audit_signed_capacity.py \
        --rows  outputs/.../metric_values_original_strict.csv \
        --repair outputs/.../repair_rows.csv \
        --json  results/audits/signed-capacity.json \
        --report results/audits/signed-capacity.md

Exit code 1 means the two metrics disagree more than the tolerance allows,
i.e. the "same metric, different sign" reading is not supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SIGNED = "perturbation_capacity_signed_delta"
ABSOLUTE = "perturbation_capacity_delta"
SETTING_KEYS = (
    "rb40_mature",
    "nb201_e0",
    "nb201_e5",
    "nb201_e10",
    "nb201_e25",
    "nb201_e50",
    "darts_e0",
    "darts_e100",
)
EPS = 1e-12
SAME_METRIC_RHO = 0.8
FLIP_TOLERANCE = 0.10


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(rows_path: Path, repair_path: Path | None = None) -> pd.DataFrame:
    rows = pd.read_csv(rows_path)
    keep = rows[rows["metric_name"].isin([SIGNED, ABSOLUTE])]
    panel = keep.pivot_table(
        index=["setting", "model_or_arch_id"],
        columns="metric_name",
        values="metric_value",
        aggfunc="mean",
        dropna=False,  # keep rows whose value is NaN; repair may fill them later
    )
    target = (
        rows.groupby(["setting", "model_or_arch_id"])["target_value"].first().rename("target")
    )
    panel = panel.join(target)
    # Repair before dropping: the whole point of repair_rows.csv is to fill
    # values whose status is "failed", and dropping first would delete exactly
    # the rows the repair is meant to rescue.
    if repair_path is not None and repair_path.exists():
        repair = pd.read_csv(repair_path)
        repair = repair[repair["metric_name"].isin([SIGNED, ABSOLUTE])]
        for _, row in repair.iterrows():
            key = (row["setting"], row["model_or_arch_id"])
            if key in panel.index:
                panel.loc[key, row["metric_name"]] = float(row["metric_value"])
    return panel.dropna(subset=[SIGNED, ABSOLUTE, "target"])


def ratio(panel: pd.DataFrame) -> pd.Series:
    """Signed value divided by its absolute counterpart, in [-1, 1].

    A ratio near -1 means every sample moved the same way; near 0 means the
    samples cancelled each other out almost perfectly.
    """
    return panel[SIGNED] / (panel[ABSOLUTE] + EPS)


def _rho(x, y) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return None if value != value else float(value)


def per_setting(panel: pd.DataFrame) -> list[dict]:
    out = []
    for setting, group in panel.groupby(level=0):
        frame = group.droplevel(0)
        r = ratio(group)
        out.append(
            {
                "setting": setting,
                "n": int(len(frame)),
                "rho_signed_target": _rho(frame[SIGNED], frame["target"]),
                "rho_absolute_target": _rho(frame[ABSOLUTE], frame["target"]),
                "rho_signed_absolute": _rho(frame[SIGNED], frame[ABSOLUTE]),
                "rho_ratio_target": _rho(r, frame["target"]),
                "ratio_mean": float(r.mean()),
                "ratio_std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                "negative_share": float((frame[SIGNED] < 0).mean()),
            }
        )
    order = {key: i for i, key in enumerate(SETTING_KEYS)}
    out.sort(key=lambda row: order.get(row["setting"], 99))
    return out


def summarise(table: list[dict], panel: pd.DataFrame) -> dict:
    usable = [row for row in table if row["rho_signed_absolute"] is not None]
    flips = [
        row
        for row in usable
        if row["rho_signed_target"] is not None
        and row["rho_absolute_target"] is not None
        and row["rho_signed_target"] * row["rho_absolute_target"] < 0
        and abs(row["rho_signed_target"]) > FLIP_TOLERANCE
        and abs(row["rho_absolute_target"]) > FLIP_TOLERANCE
    ]
    rhos = [row["rho_signed_absolute"] for row in usable]
    # |mean(x)| <= mean(|x|) must hold for every row.  If it does not, the two
    # columns were not computed from the same samples and every conclusion
    # below is built on a broken join.
    violations = int((panel[SIGNED].abs() > panel[ABSOLUTE] + 1e-9).sum())
    return {
        "abs_le_abs_violations": violations,
        "n_settings": len(usable),
        "n_rank_flips": len(flips),
        "flipped_settings": [row["setting"] for row in flips],
        "median_rho_signed_absolute": float(np.median(rhos)) if rhos else None,
        "min_rho_signed_absolute": float(min(rhos)) if rhos else None,
        "median_negative_share": float(np.median([row["negative_share"] for row in usable]))
        if usable
        else None,
    }


def verdict(summary: dict) -> str:
    median = summary["median_rho_signed_absolute"]
    if median is None:
        return "insufficient data"
    if median < SAME_METRIC_RHO or summary["n_rank_flips"] >= 2:
        return "not the same metric"
    return "consistent up to sign"


VERDICT_ZH = {
    "not the same metric": "不是同一个指标的两种写法（有符号与绝对值是两个不同指标，设计如此）",
    "consistent up to sign": "只差一个符号，可以视为同一指标",
    "insufficient data": "数据不足，无法判断",
}


def analyse(panel: pd.DataFrame) -> dict:
    table = per_setting(panel)
    summary = summarise(table, panel)
    return {
        "schema_version": 1,
        "signed_metric": SIGNED,
        "absolute_metric": ABSOLUTE,
        "n_rows": int(len(panel)),
        "thresholds": {
            "same_metric_rho": SAME_METRIC_RHO,
            "flip_tolerance": FLIP_TOLERANCE,
        },
        "summary": summary,
        "verdict": verdict(summary),
        "settings": table,
    }


def _fmt(value) -> str:
    return "—" if value is None else f"{value:+.3f}"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# 有符号 vs 绝对 capacity 漂移",
        "",
        f"- 两个指标只差一个 `abs()`：`{report['absolute_metric']}` 与 `{report['signed_metric']}`",
        f"- 样本：{report['n_rows']} 个（设置 × 模型/架构），目标是发布表里的鲁棒准确率",
        "",
        "## 结论",
        "",
        f"**{VERDICT_ZH[report['verdict']]}**",
        "",
        f"- 两者在设置内的秩相关中位数：**{_fmt(summary['median_rho_signed_absolute'])}**"
        f"（最低 {_fmt(summary['min_rho_signed_absolute'])}）",
        f"- 与 target 的相关**符号相反**的设置：**{summary['n_rank_flips']} / {summary['n_settings']}**"
        + (f"（{'、'.join(summary['flipped_settings'])}）" if summary["flipped_settings"] else ""),
        f"- 有符号指标为负的样本占比（中位数）：{_fmt(summary['median_negative_share'])}",
        f"- 自检：`|带符号| <= 绝对` 不成立的样本数 **{summary['abs_le_abs_violations']} / {report['n_rows']}**"
        "（数学上必然为 0；不为 0 说明两个指标不是同一批样本，配对出错）",
        "",
        "## 逐设置",
        "",
        "| 设置 | n | ρ(带符号, target) | ρ(绝对, target) | ρ(带符号, 绝对) | ρ(比值, target) | 负值占比 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["settings"]:
        lines.append(
            f"| `{row['setting']}` | {row['n']} | {_fmt(row['rho_signed_target'])} | "
            f"{_fmt(row['rho_absolute_target'])} | {_fmt(row['rho_signed_absolute'])} | "
            f"{_fmt(row['rho_ratio_target'])} | {row['negative_share']:.2f} |"
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- `比值` = 带符号值 / 绝对值。它接近 ±1 表示所有样本朝同一方向变，接近 0 表示"
        "上升的样本和下降的样本几乎互相抵消。",
        "- 如果两个指标真是「只差符号」，`ρ(带符号, 绝对)` 应该接近 ±1，"
        "且两者与 target 的相关只在符号上不同。表中若出现「相关符号相反」，"
        "就说明排序不是简单翻转，而是**换了内容**。",
        "- 这套比较不改变任何已发布数字；它只是说明这两个指标能不能放在一起解读。",
        "- 两个指标**本来就该不同**：一个有符号、一个取绝对值。前者保留「capacity 是升还是降」，"
        "后者只问「变了多少」。它们同时出现在榜单上时，名次不能互相印证，必须分别说明。",
        "",
        "## 局限",
        "",
        "- RB40 只有 40 个模型，单指标 Spearman 的 95% 区间约 ±0.3，该行只能当指示。",
        "- NB201 同一架构有 5 个阶段，样本不独立；这里按设置分别算，没有合并。",
        "- 没有控制模型容量，因此「带符号值与 target 负相关」仍可能是容量混杂。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    panel = load_panel(args.rows, args.repair)
    report = analyse(panel)
    report["inputs"] = {
        "rows": {"path": str(args.rows), "sha256": sha256_file(args.rows)},
    }
    if args.repair:
        report["inputs"]["repair"] = {"path": str(args.repair), "sha256": sha256_file(args.repair)}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print(
        f"verdict={report['verdict']!r} median_rho={summary['median_rho_signed_absolute']} "
        f"flips={summary['n_rank_flips']}/{summary['n_settings']} "
        f"pairing_violations={summary['abs_le_abs_violations']}"
    )
    return 1 if report["verdict"] == "not the same metric" else 0


if __name__ == "__main__":
    raise SystemExit(main())
