"""Build one index of every metric the project knows about.

Why this exists
---------------
There are 86 registered metrics. Their definitions live in
``src/robust_metric_lab/metric_registry.py``, their three-space eligibility in
``configs/metric_eligibility.json``, their scores in the release table, and their
prose in a dozen reports. Answering "is metric X on the leaderboard, which
direction does it point, is it ours or borrowed, does it work in all three
spaces?" currently means opening four files.

This tool joins those four sources into one Markdown table, regenerable from the
repository. It is an index, not an analysis: it moves no numbers and computes no
new scores.

What it refuses to do
---------------------
It does not fill in a coverage count when the raw rows are not supplied. A metric
with "no data" and a metric with "low value" are different things; printing 0
here would repeat the exact mistake the redundancy audit had to fix.

Usage
-----
    python tools/build_metric_index.py \
        --table outputs/.../86项_三位小数.csv \
        --markdown docs/theory/METRIC_INDEX.md \
        --json results/audits/metric-index-20261112.json
    # optional: --rows <metric_values_original_strict.csv> to add coverage counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

# Heuristic families, by name prefix. Deliberately labelled as a guess in the
# output: the registry has no family field, and inventing a strict taxonomy here
# would be a claim the data does not support.
FAMILIES = (
    ("trnas_", "TRNAS 移植（Φ/R/Ψ）"),
    ("racl_", "RACL 理论界"),
    ("feature_measure_", "特征漂移度量家族"),
    ("perturbation_capacity_", "扰动容量"),
    ("fsp_", "FSP / Gram 结构"),
    ("jacobian_", "雅可比谱"),
    ("spectrum_", "谱结构"),
    ("spectral_", "谱结构"),
    ("cov_", "协方差结构"),
    ("feature_", "特征空间（其他）"),
    ("input_", "输入/输出敏感性"),
    ("local_", "输入/输出敏感性"),
    ("logit_", "输入/输出敏感性"),
    ("margin_", "输入/输出敏感性"),
    ("natural_", "输出空间"),
    ("adversarial_", "输出空间"),
)


def family_of(name: str) -> str:
    for prefix, label in FAMILIES:
        if name.startswith(prefix):
            return label
    return "其他"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry() -> dict[str, dict]:
    from robust_metric_lab.metric_registry import METRIC_SPECS

    return {
        spec.name: {
            "direction": spec.direction,
            "attribution": spec.attribution,
            "reference": spec.reference,
        }
        for spec in METRIC_SPECS
    }


def load_eligibility(path: Path) -> dict[str, dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["metric_name"]: {
            "eligibility": item["eligibility"],
            "supported_settings": item["supported_settings"],
            "reason": item["reason"],
        }
        for item in entries
    }


def coverage_from_rows(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    frame = pd.read_csv(path, usecols=["metric_name", "setting", "metric_value"])
    finite = frame[pd.to_numeric(frame["metric_value"], errors="coerce").notna()]
    return finite.groupby("metric_name")["setting"].nunique().to_dict()


def build(table: pd.DataFrame, registry: dict, eligibility: dict, coverage: dict | None) -> dict:
    known = set(table["metric_name"])
    rows = []
    for name, spec in registry.items():
        hit = table[table["metric_name"] == name]
        row = hit.iloc[0] if len(hit) == 1 else None
        rank = None if row is None else row["strict_rank"]
        rows.append(
            {
                "metric": name,
                "family": family_of(name),
                "direction": spec["direction"],
                "attribution": spec["attribution"],
                "reference": spec["reference"],
                "rank": None if pd.isna(rank) else int(rank),
                "status": None if row is None else str(row["strict_status"]),
                "score": None if row is None else float(row["all_setting_mean"]),
                "raw_rows": None if row is None else int(row["raw_rows"]),
                "exclusion_reason": None
                if row is None or pd.isna(row["exclusion_reason"])
                else str(row["exclusion_reason"]),
                "eligibility": eligibility.get(name, {}).get("eligibility"),
                "coverage": None if coverage is None else coverage.get(name, 0),
            }
        )

    # Set-level cross-checks: the three files must describe the same metrics.
    problems = []
    if known - set(registry):
        problems.append(f"发布表有而注册表没有：{sorted(known - set(registry))}")
    if set(registry) - known:
        problems.append(f"注册表有而发布表没有：{sorted(set(registry) - known)}")
    if set(eligibility) - set(registry):
        problems.append(f"资格表有而注册表没有：{sorted(set(eligibility) - set(registry))}")
    if set(registry) - set(eligibility):
        problems.append(f"注册表有而资格表没有：{sorted(set(registry) - set(eligibility))}")

    # Soft contradiction: a metric declared unusable still has a rank. Both facts
    # can be legitimate (a repair landed after the declaration), but nobody wrote
    # down which one is current, so a human has to look.
    warnings = sorted(
        (
            {
                "metric": row["metric"],
                "rank": row["rank"],
                "eligibility": row["eligibility"],
                "reason": eligibility.get(row["metric"], {}).get("reason"),
            }
            for row in rows
            if row["rank"] is not None and row["eligibility"] == "implementation_error"
        ),
        key=lambda item: item["rank"],
    )

    return {
        "rows": rows,
        "problems": problems,
        "warnings": warnings,
        "coverage_supplied": coverage is not None,
    }


def sorted_rows(rows: list[dict]) -> list[dict]:
    ranked = sorted((r for r in rows if r["rank"] is not None), key=lambda r: r["rank"])
    unranked = sorted((r for r in rows if r["rank"] is None), key=lambda r: r["metric"])
    return ranked + unranked


def render_markdown(payload: dict) -> str:
    rows = sorted_rows(payload["rows"])
    ranked = [r for r in rows if r["rank"] is not None]
    excluded = [r for r in rows if r["rank"] is None]
    lines = [
        "# 指标总索引",
        "",
        "> 本文件由 `tools/build_metric_index.py` 自动生成，**不要手改**。",
        "> 改指标请改 `src/robust_metric_lab/metric_registry.py` 或 `configs/metric_eligibility.json`，然后重跑脚本。",
        "",
        f"- 注册表指标数：{len(rows)}；入榜 {len(ranked)}；无名次 {len(excluded)}",
        f"- 覆盖列：{'已按原始行统计' if payload['coverage_supplied'] else '**未提供原始行，留空**（不是 0）'}",
        "",
        "## 入榜指标（按名次）",
        "",
        "| 名次 | 指标 | 家族 | 方向 | 归属 | 八设置均值 | 覆盖 | 资格 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in ranked:
        coverage = "—" if r["coverage"] is None else f"{r['coverage']}/8"
        lines.append(
            f"| {r['rank']} | `{r['metric']}` | {r['family']} | {r['direction']} | "
            f"{r['attribution']} | {r['score']:.3f} | {coverage} | {r['eligibility'] or '—'} |"
        )
    lines += [
        "",
        "## 无名次的指标",
        "",
        "| 指标 | 家族 | 方向 | 状态 | 原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in excluded:
        status = r["status"] or "未出现在发布表"
        reason = r["exclusion_reason"] or "—"
        lines.append(f"| `{r['metric']}` | {r['family']} | {r['direction']} | {status} | {reason} |")
    lines += [
        "",
        "## 怎么读这张表",
        "",
        "- **方向**：`higher` 表示指标越大越鲁棒，`lower` 表示越小越鲁棒。冻结协议在算秩相关时"
        "按这个方向统一符号，所以方向写错会让一个指标的排名整体翻转。",
        "- **资格**：`cross_space` 表示三个空间定义一致、可以跨空间比较；其余取值只在单一空间有意义。",
        "- **覆盖**：该指标在 8 个设置中有几个给出有限值。冻结协议要求 8/8 才入榜，"
        "所以覆盖不足的指标不会出现在名次列。",
        "- **家族**是**按名字前缀猜的**，不是注册表里的正式字段。它只用于快速浏览，"
        "不要拿它当统计分组的依据。",
    ]
    if payload["warnings"]:
        lines += [
            "",
            "## ⚠ 需要人看的不一致：声明为不可用，却在榜上有名",
            "",
            "`configs/metric_eligibility.json` 把这些指标标成 `implementation_error`，"
            "但发布表给了它们名次。两件事可能都对（先声明有错、后来修好了），"
            "但没人写下哪一条是当前口径，所以需要人确认。",
            "",
            "| 名次 | 指标 | 配置里的理由 |",
            "| --- | --- | --- |",
        ]
        for item in payload["warnings"]:
            lines.append(f"| {item['rank']} | `{item['metric']}` | {item['reason'] or '—'} |")
    if payload["problems"]:
        lines += ["", "## ⚠ 数据源不一致", ""] + [f"- {p}" for p in payload["problems"]]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--eligibility", type=Path, default=ROOT / "configs" / "metric_eligibility.json")
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.table, keep_default_na=False, na_values=[""])
    table["strict_rank"] = pd.to_numeric(table["strict_rank"], errors="coerce")

    payload = build(
        table,
        load_registry(),
        load_eligibility(args.eligibility),
        coverage_from_rows(args.rows),
    )
    payload.update(
        {
            "schema_version": 1,
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "eligibility": str(args.eligibility),
            "rows_source": None if args.rows is None else str(args.rows),
        }
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")

    print(f"metrics={len(payload['rows'])} problems={len(payload['problems'])}")
    for problem in payload["problems"]:
        print(f"  PROBLEM {problem}")
    return 0 if not payload["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
