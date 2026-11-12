"""Check that numbers written in the docs still match the release tables.

Why this exists
---------------
Over months the same conclusion gets restated in reports, in PROJECT_STATUS, and
in hand-over notes. Each restatement is a chance to introduce a typo or to quote
an older release. A leaderboard whose prose disagrees with its CSV is worse than
no leaderboard, because readers trust the prose.

This tool holds a short, explicit list of claims ("PROJECT_STATUS says Phi's
eight-setting mean is 0.7822") and checks each one against the release table.

Why the claim list is hand-written
----------------------------------
Auto-extracting every decimal from Markdown looks cheaper, but it misfires: the
docs also contain stability correlations (0.914, 0.995), timing figures (3.08s),
and numbers from the superseded 80-item release. A number that is not on the
leaderboard is not wrong -- it just belongs to a different claim. So the list
stays explicit and reviewable, and every entry names where it came from.

Usage
-----
    python tools/verify_doc_claims.py \
        --table outputs/.../RobustMetric_..._86项_三位小数.csv \
        --json results/audits/doc-claims-20261112.json \
        --report results/audits/doc-claims-20261112.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

# The table is rounded to three decimals; docs usually quote four.
# 0.0006 accepts that rounding and rejects a real disagreement.
TOLERANCE = 0.0006

# Settings in the order the project reports them.
NB201_STAGES = ("nb201_e0", "nb201_e5", "nb201_e10", "nb201_e25", "nb201_e50")

PHI = "trnas_consistency_phi"
RSCORE = "trnas_r_score"
PSI = "trnas_activation_psi"
MAXD = "feature_measure_multiview_max_delta"
RMSD = "feature_measure_multiview_rms_delta"


def claims() -> list[dict]:
    out: list[dict] = []

    def field(doc, where, metric, name, expected, note):
        out.append(
            {
                "doc": doc,
                "where": where,
                "kind": "field",
                "metric": metric,
                "field": name,
                "expected": expected,
                "note": note,
            }
        )

    def diff(doc, where, left, right, expected, note):
        out.append(
            {
                "doc": doc,
                "where": where,
                "kind": "diff",
                "left": left,
                "right": right,
                "field": "all_setting_mean",
                "expected": expected,
                "note": note,
            }
        )

    status = "docs/PROJECT_STATUS.md"
    compare = "docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md"
    position = "docs/experiments/自研指标定位与TRNAS对照进度_20261030.md"

    # 1. Top of the leaderboard: score and rank.
    field(status, "§3 / §4 榜单", PHI, "all_setting_mean", 0.7822, "Φ 八设置均值，第 1 名")
    field(status, "§3 / §4 榜单", PHI, "strict_rank", 1, "Φ 名次")
    field(status, "§3 / §4 榜单", RSCORE, "all_setting_mean", 0.7378, "R-Score 第 2 名")
    field(status, "§3 / §4 榜单", RSCORE, "strict_rank", 2, "R-Score 名次")
    field(status, "§3 / §4 榜单", "adversarial_loss", "all_setting_mean", 0.614, "第 3 名")
    field(status, "§3 / §4 榜单", PSI, "all_setting_mean", 0.5981, "Ψ 第 4 名")
    field(status, "§3 / §4 榜单", "natural_loss", "all_setting_mean", 0.591, "第 5 名")
    field(status, "§3 / §4 榜单", MAXD, "all_setting_mean", 0.5241, "自研 max 第 6 名")
    field(status, "§3 / §4 榜单", MAXD, "strict_rank", 6, "自研 max 名次")
    field(status, "§3 / §4 榜单", RMSD, "all_setting_mean", 0.4950, "自研 RMS 第 7 名")
    field(status, "§3 / §4 榜单", "fsp_delta", "all_setting_mean", 0.479, "第 8 名")

    # 2. Pairwise gaps quoted in the comparison report.
    diff(compare, "八设置均值差", MAXD, PHI, -0.258, "max − Φ")
    diff(compare, "八设置均值差", MAXD, RSCORE, -0.214, "max − R-Score")
    diff(compare, "八设置均值差", MAXD, PSI, -0.074, "max − Ψ")
    diff(compare, "八设置均值差", RMSD, PHI, -0.287, "RMS − Φ")

    # 3. Per-setting correlations quoted in the positioning note.
    for stage, value in zip(NB201_STAGES, (0.849, 0.825, 0.797, 0.753, 0.771)):
        field(position, "NB201 五阶段 Φ", PHI, stage, value, f"Φ 在 {stage}")
    for stage, value in zip(NB201_STAGES, (0.686, 0.735, 0.750, 0.677, 0.657)):
        field(position, "NB201 五阶段 max", MAXD, stage, value, f"max 在 {stage}")
    for stage, value in zip(NB201_STAGES, (0.663, 0.716, 0.761, 0.653, 0.630)):
        field(position, "NB201 五阶段 RMS", RMSD, stage, value, f"RMS 在 {stage}")
    field(position, "DARTS 塌陷", MAXD, "darts_e100", 0.006, "max 在 darts_e100")
    field(position, "DARTS 塌陷", RMSD, "darts_e100", -0.036, "RMS 在 darts_e100")
    field(position, "DARTS 对照", PHI, "darts_e100", 0.847, "Φ 在 darts_e100")
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup(table: pd.DataFrame, metric: str, column: str) -> float:
    rows = table[table["metric_name"] == metric]
    if len(rows) != 1:
        raise SystemExit(f"{metric}: expected exactly one row, found {len(rows)}")
    value = rows.iloc[0][column]
    if pd.isna(value):
        raise SystemExit(f"{metric}.{column}: empty value in the release table")
    return float(value)


def evaluate(table: pd.DataFrame, claim: dict) -> dict:
    if claim["kind"] == "field":
        actual = lookup(table, claim["metric"], claim["field"])
    elif claim["kind"] == "diff":
        actual = lookup(table, claim["left"], claim["field"]) - lookup(
            table, claim["right"], claim["field"]
        )
    else:  # pragma: no cover - guarded by the claim list itself
        raise SystemExit(f"unknown claim kind: {claim['kind']}")
    error = actual - float(claim["expected"])
    return {**claim, "actual": round(actual, 6), "error": round(error, 6), "ok": abs(error) <= TOLERANCE}


def check_invariants(table: pd.DataFrame) -> list[dict]:
    """Structural facts the docs rely on, checked against the table itself."""
    checks = []

    def record(name, expected, actual, note):
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "ok": expected == actual,
                "note": note,
            }
        )

    record("总行数", 86, len(table), "84 入榜 + 2 排除")
    record(
        "入榜行数",
        84,
        int((table["strict_status"].astype(str) == "ranked").sum()),
        "strict_status == ranked",
    )
    record(
        "排除行数",
        2,
        int((table["strict_status"].astype(str) == "excluded").sum()),
        "strict_status == excluded",
    )
    record("每行原始行数唯一值", [560], sorted(table["raw_rows"].unique().tolist()), "raw_rows 应恒为 560")
    ranks = sorted(table["strict_rank"].dropna().astype(int).tolist())
    record("名次连续 1..84", list(range(1, 85)), ranks, "无并列、无跳号")
    record(
        "release_id 唯一",
        1,
        table["release_id"].nunique(),
        "全表来自同一次发布",
    )
    return checks


def render_markdown(payload: dict) -> str:
    failed = [item for item in payload["claims"] if not item["ok"]]
    lines = [
        "# 文档声称 vs 发布表：数字对账",
        "",
        f"- 发布表：`{payload['table']['path']}`（sha256 `{payload['table']['sha256']}`）",
        f"- 声称条数：{len(payload['claims'])}，通过 {len(payload['claims']) - len(failed)}，"
        f"失败 {len(failed)}",
        f"- 容差：{payload['tolerance']}（表保留三位小数，文档常写四位）",
        "",
        "## 逐条核对",
        "",
        "| 文档 | 位置 | 声称 | 表里的值 | 差值 | 结果 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["claims"]:
        if item["kind"] == "field":
            claim = f"`{item['metric']}` 的 `{item['field']}` = {item['expected']}"
        else:
            claim = (
                f"`{item['left']}` − `{item['right']}` = {item['expected']}"
            )
        verdict = "通过" if item["ok"] else "**不一致**"
        lines.append(
            f"| `{item['doc']}` | {item['where']} | {claim} | {item['actual']} | "
            f"{item['error']:+.6f} | {verdict} |"
        )
    lines += ["", "## 结构性检查", "", "| 检查项 | 期望 | 实际 | 结果 |", "| --- | --- | --- | --- |"]
    for check in payload["invariants"]:
        verdict = "通过" if check["ok"] else "**不一致**"
        lines.append(
            f"| {check['name']} | {check['expected']} | {check['actual']} | {verdict} |"
        )
    lines += [
        "",
        "## 这条检查能证明什么、不能证明什么",
        "",
        "- 能证明：文档里写的这些数字，与当前发布表**逐个对得上**；没有出现「引用旧版发布」或"
        "抄写错误。",
        "- 不能证明：这些数字**本身是对的**。它只保证文档与表一致；表是否正确，"
        "要看 `results/audits/strict-ranking-recompute-20261112.md` 的独立复算。",
        "- 声称清单是**手工维护**的，只覆盖被反复引用、最容易出错的那批数字；"
        "它不代表全部文档都已核对。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.table, keep_default_na=False, na_values=[""])
    table["strict_rank"] = pd.to_numeric(table["strict_rank"], errors="coerce")
    evaluated = [evaluate(table, claim) for claim in claims()]
    invariants = check_invariants(table)

    payload = {
        "schema_version": 1,
        "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
        "tolerance": TOLERANCE,
        "claims": evaluated,
        "invariants": invariants,
        "failed": [item for item in evaluated if not item["ok"]],
        "failed_invariants": [check for check in invariants if not check["ok"]],
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")

    total = len(evaluated)
    bad = len(payload["failed"]) + len(payload["failed_invariants"])
    print(f"claims={total} failed={len(payload['failed'])} invariant_failures={len(payload['failed_invariants'])}")
    for item in payload["failed"]:
        print(f"  MISMATCH {item['metric']}.{item['field']}: doc={item['expected']} table={item['actual']}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
