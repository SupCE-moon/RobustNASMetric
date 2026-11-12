"""Does the RQ2 signal come only from metrics that survive a change of weights?

Every RQ2 number so far was computed on a mixed bag of metrics. Some of them
barely change when the untrained network is re-initialised; others (the whole
output-space family) are essentially a random number per seed. Pooling the two
groups hides which one is doing the work.

This script joins two things that already exist on disk:

1. results/audits/e0-weight-seed-20261112.json -- value stability (rank
   correlation of the same metric across weight seeds) and the signal verdict.
2. outputs/e28_capacity_family/nb201_e0_n400_20261112.csv -- 400 architectures
   with their trained robustness target.

and asks, for each metric, three questions at once: how stable is its value,
how stable is its signal, and how strong is its partial correlation with the
target. The report then splits the metrics into four groups and shows which
group the strong results live in.

The point is a sanity check, not a new finding: if the strongest association
sat in the "unstable" group, the RQ2 conclusion would be resting on noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_metric_lab.metric_registry import METRIC_SPECS

SCHEMA_VERSION = 1
VALUE_THRESHOLD = 0.8
CONTROL_COLUMNS = ("clean", "params")
DIRECTION = {spec.name: spec.direction for spec in METRIC_SPECS}
METADATA = {"arch_id", "arch_string", "target", "clean", "params", "init_clean"}
GROUPS = (
    "stable_both",
    "stable_value_only",
    "stable_signal_only",
    "unstable_both",
)
GROUP_LABEL = {
    "stable_both": "值与信号都稳定",
    "stable_value_only": "只有值稳定",
    "stable_signal_only": "只有信号稳定",
    "unstable_both": "都不稳定",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(arch_string: str) -> float | None:
    """Parameter count of the tiny network this repository uses for NB201."""
    from robust_metric_lab.nb201_space import Nb201TinyNetwork
    from robust_metric_lab.nasrobbench201 import parse_nb201_arch

    ops = parse_nb201_arch(str(arch_string))
    if ops is None:
        return None
    return float(sum(p.numel() for p in Nb201TinyNetwork(ops).parameters()))


def load_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"arch_id": str})
    required = {"arch_id", "arch_string", "target", "clean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"wide table is missing columns: {sorted(missing)}")
    if "params" not in frame.columns:
        counts = frame["arch_string"].map(parameter_count)
        frame["params"] = counts.astype(float)
    frame = frame.dropna(subset=["target", "clean", "params"])
    return frame.reset_index(drop=True)


def load_seed_report(path: Path) -> dict[str, dict]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, dict] = {}
    for row in report.get("metrics", []):
        table[str(row["metric"])] = {
            "value_rho": float(row["value"]["mean_rho"]),
            "signal_verdict": str(row.get("signal_verdict", "")),
        }
    return table


def group_of(metric: str, seeds: dict[str, dict], threshold: float = VALUE_THRESHOLD) -> str:
    """Which of the four groups a metric falls into, or '' when unmeasured."""
    entry = seeds.get(metric)
    if entry is None:
        return ""
    value_ok = entry["value_rho"] >= threshold
    signal_ok = entry["signal_verdict"] == "stable_signal"
    if value_ok and signal_ok:
        return "stable_both"
    if value_ok:
        return "stable_value_only"
    if signal_ok:
        return "stable_signal_only"
    return "unstable_both"


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def partial_rho(x: pd.Series, y: pd.Series, controls: list[pd.Series]) -> float | None:
    """Rank partial correlation: rank everything, then correlate the residuals."""
    x = pd.to_numeric(x, errors="coerce").reset_index(drop=True)
    y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
    mask = x.notna() & y.notna()
    matrix = []
    for control in controls:
        column = pd.to_numeric(control, errors="coerce").reset_index(drop=True)
        mask &= column.notna()
        matrix.append(column)
    if mask.sum() < 20:
        return None
    ranked_x = _rank(x[mask])
    ranked_y = _rank(y[mask])
    if ranked_x.nunique() < 3 or ranked_y.nunique() < 3:
        return None
    design = np.column_stack(
        [np.ones(int(mask.sum()))] + [_rank(c[mask]).to_numpy() for c in matrix]
    )
    residual_x = ranked_x.to_numpy() - design @ np.linalg.lstsq(design, ranked_x.to_numpy(), rcond=None)[0]
    residual_y = ranked_y.to_numpy() - design @ np.linalg.lstsq(design, ranked_y.to_numpy(), rcond=None)[0]
    if residual_x.std() == 0 or residual_y.std() == 0:
        return None
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def scores(frame: pd.DataFrame, seeds: dict[str, dict]) -> list[dict]:
    """Partial correlation with the target for every metric, plus its group."""
    target = frame["target"]
    controls = [frame[c] for c in CONTROL_COLUMNS if c in frame.columns]
    rows: list[dict] = []
    for column in frame.columns:
        if column in METADATA:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() < 20:
            continue
        flipped = series
        if DIRECTION.get(column) == "lower":
            flipped = -series
        rho = partial_rho(flipped, target, controls)
        rows.append(
            {
                "metric": column,
                "partial_rho": rho,
                "value_rho": (seeds.get(column) or {}).get("value_rho"),
                "signal_verdict": (seeds.get(column) or {}).get("signal_verdict", ""),
                "group": group_of(column, seeds),
            }
        )
    rows.sort(key=lambda row: -abs(row["partial_rho"]) if row["partial_rho"] is not None else 0)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Per-group counts and the strongest |rho| in each group."""
    out: dict[str, dict] = {}
    for group in GROUPS + ("unmeasured",):
        members = [r for r in rows if (r["group"] or "unmeasured") == group]
        finite = [r for r in members if r["partial_rho"] is not None]
        strongest = max(finite, key=lambda r: abs(r["partial_rho"]), default=None)
        rhos = [abs(r["partial_rho"]) for r in finite]
        out[group] = {
            "count": len(members),
            "measured": len(finite),
            "strongest": strongest["metric"] if strongest else None,
            "strongest_rho": strongest["partial_rho"] if strongest else None,
            "median_abs_rho": float(np.median(rhos)) if rhos else None,
            "above_0_30": sum(1 for value in rhos if value >= 0.30),
        }
    return out


def verdict(rows: list[dict], summary: dict) -> str:
    """Is the top of the table made of metrics that survive re-initialisation?

    The useful question is not "how many of the top five are perfect" -- a
    metric whose value moves a little but whose signal is steady is still a
    legitimate early predictor (sample it a few times and average). What would
    invalidate the reading is a metric whose value and signal both move
    appearing near the top, because then the number is a seed artefact.
    """
    measured = [r for r in rows if r["partial_rho"] is not None]
    if not measured:
        return "no measurable metric"
    top5 = measured[:5]
    top10 = measured[:10]
    unstable_in_top10 = [r["metric"] for r in top10 if r["group"] == "unstable_both"]
    if unstable_in_top10:
        return (
            "前十混入了值不稳的指标（{}），这一段的结论要打折".format(
                ", ".join(unstable_in_top10)
            )
        )
    usable = {"stable_both", "stable_signal_only"}
    if all(r["group"] in usable for r in top10):
        if sum(1 for r in top5 if r["group"] == "stable_both") >= 4:
            return "最强的关联来自值与信号都稳定的指标"
        return "前十全部可复现，但前五里有指标只是信号稳定（值需多次采样）"
    return "前十混入了只有值稳定、信号不稳的指标"


def analyse(wide: Path, seed_report: Path) -> dict:
    frame = load_wide(wide)
    seeds = load_seed_report(seed_report)
    rows = scores(frame, seeds)
    summary = summarise(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "wide": {"path": str(wide), "sha256": sha256_file(wide), "n": int(len(frame))},
        "seed_report": {"path": str(seed_report), "sha256": sha256_file(seed_report)},
        "value_threshold": VALUE_THRESHOLD,
        "controls": list(CONTROL_COLUMNS),
        "verdict": verdict(rows, summary),
        "groups": summary,
        "metrics": rows,
    }


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}"


def render_markdown(report: dict) -> str:
    groups = report["groups"]
    lines = [
        "# RQ2 的正面结论是否只来自「稳得住」的指标（2026-11-12）",
        "",
        f"宽表：`{report['wide']['path']}`（{report['wide']['n']} 个架构，sha256 `{report['wide']['sha256'][:16]}…`）  ",
        f"稳定性来源：`{report['seed_report']['path']}`  ",
        f"控制变量：{', '.join('`' + c + '`' for c in report['controls'])}；"
        f"值稳定阈值 {report['value_threshold']}",
        "",
        "## 结论",
        "",
        f"**{report['verdict']}**",
        "",
        "## 分组统计",
        "",
        "| 组 | 指标数 | 最强指标 | 最强 \\|rho\\| | 中位 \\|rho\\| | \\|rho\\|≥0.30 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group in GROUPS + ("unmeasured",):
        block = groups.get(group)
        if not block or not block["count"]:
            continue
        label = GROUP_LABEL.get(group, "未测稳定性")
        lines.append(
            "| {} | {} | `{}` | {} | {} | {} |".format(
                label,
                block["count"],
                block["strongest"] or "—",
                _fmt(block["strongest_rho"]),
                _fmt(block["median_abs_rho"]),
                block["above_0_30"],
            )
        )
    lines += [
        "",
        "## 前二十（按 |偏相关| 降序）",
        "",
        "| 指标 | 偏相关 | 值稳定 | 信号 | 组 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["metrics"][:20]:
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row["metric"],
                _fmt(row["partial_rho"]),
                _fmt(row["value_rho"]),
                row["signal_verdict"] or "—",
                GROUP_LABEL.get(row["group"], "未测稳定性"),
            )
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- 偏相关是**控制训练后 clean 与参数量之后**的秩偏相关，`lower` 指标已翻号（正数=方向对）。",
        "- 值稳定与信号稳定来自 `e0-weight-seed-20261112`（100 架构 × 3 个权重种子）。",
        "  稳定性是指标的性质，不依赖样本量，所以拿它给 400 架构的结果分组是成立的。",
        "- 分组不是「好坏」评级：`只有信号稳定` 的指标仍可用于预测（多次采样取均值），只是不能当固定特征。",
        "",
        "## 这份报告不能说明什么",
        "",
        "- 稳定性只在 100 个架构（字典序前 100）上测过，没有覆盖全部 400 个。",
        "- 输出空间一族（`adversarial_loss` 等）在这里被判为不稳定，但榜单里它们排得很前——",
        "  那评的是**已训练模型**，与这里的**未训练网络**不是同一个对象。两者不能互相引用。",
        "- 相关不是因果；分组只说明这个数有多可信，不说明它测的是鲁棒性的成因。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts\\audit_rq2_stable_subset.py `",
        "  --wide <n400 宽表>.csv `",
        "  --seed-report results\\audits\\e0-weight-seed-20261112.json `",
        "  --json results\\audits\\rq2-stable-subset-20261112.json `",
        "  --report results\\audits\\rq2-stable-subset-20261112.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wide", required=True, type=Path)
    parser.add_argument("--seed-report", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    report = analyse(args.wide, args.seed_report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_markdown(report), encoding="utf-8")
    print(f"verdict: {report['verdict']}")
    for group in GROUPS:
        block = report["groups"][group]
        print(f"  {group}: count={block['count']} strongest={block['strongest']} {_fmt(block['strongest_rho'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
