"""Does the leftover early-signal survive controlling for parameter count?

The RQ2 check controls trained clean accuracy, because clean and robust accuracy
rank together at +0.98.  A second confounder is untested: bigger architectures may
both drift more under an attack and train to higher robust accuracy.  Parameter
count is computable from the NB201 string, so this audit adds it as a second
control and reports how much of the leftover signal it explains away.

  ρ(m, y)              raw
  ρ(m, y | clean)      what c9d9d3c and D-028 reported
  ρ(m, y | clean, C)   this audit; C = parameter count

A metric is only worth following if the third number is still non-trivial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from robust_metric_lab.metric_registry import METRIC_SPECS
from robust_metric_lab.nasrobbench201 import parse_nb201_arch
from robust_metric_lab.nb201_space import Nb201TinyNetwork

SCHEMA_VERSION = 1
MIN_ARCH = 20
SURVIVES_RHO = 0.30
SHRINK_LIMIT = 0.10
RESAMPLES = 2000
SEED = 20261112
CI = (2.5, 97.5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["arch_id"] = frame["arch_id"].astype(str)
    missing = {"arch_id", "arch_string", "target", "clean"} - set(frame.columns)
    if missing:
        raise ValueError(f"wide table is missing {sorted(missing)}")
    return frame


def parameter_counts(arch_strings: pd.Series) -> dict[str, int]:
    """Parameter count of the tiny NB201 network each string describes."""
    counts: dict[str, int] = {}
    for arch_id, text in arch_strings.items():
        ops = parse_nb201_arch(str(text))
        if ops is None:
            continue
        model = Nb201TinyNetwork(ops)
        counts[str(arch_id)] = int(sum(p.numel() for p in model.parameters()))
    return counts


def _rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values).astype(float)


def _residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_rho(x, y, controls) -> float | None:
    """Rank partial correlation with one or more controls.

    Everything is ranked first, then ``x`` and ``y`` are regressed on an intercept
    plus the (already ranked) controls and the residual correlation is taken.  This
    is the multi-control generalisation of ``partial_spearman`` in
    ``audit_early_metrics_vs_robbench``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if len(controls) == 0:
        control_matrix = np.empty((len(x), 0))
    else:
        control_matrix = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    for column in control_matrix.T:
        mask &= np.isfinite(column)
    if mask.sum() < MIN_ARCH:
        return None

    # Rank the controls too.  Regressing a ranked x on raw control values is not
    # a rank statistic for any known estimator, and it disagrees with
    # `audit_early_metrics_vs_robbench.partial_spearman`, which ranks everything.
    # On the 400-architecture table the difference is not small: `feature_delta`
    # controlling clean and parameters comes out +0.422 with raw controls and
    # +0.145 with ranked ones.
    ranked_controls = [_rank(column) for column in control_matrix[mask].T]
    design = np.column_stack([np.ones(int(mask.sum())), *ranked_controls])
    rx = _residualise(_rank(x[mask]), design)
    ry = _residualise(_rank(y[mask]), design)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(x, y, controls, resamples: int = RESAMPLES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(controls) == 0:
        control_matrix = np.empty((len(x), 0))
    else:
        control_matrix = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    n = len(x)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        value = partial_rho(x[idx], y[idx], [c[idx] for c in control_matrix.T])
        if value is not None:
            draws.append(value)
    if not draws:
        return None, None, 0
    low, high = np.percentile(draws, CI)
    return float(low), float(high), len(draws)


def classify(rho_clean, rho_both) -> str:
    if rho_clean is None or rho_both is None:
        return "uncovered"
    # A metric that was already weak without the capacity control cannot be said
    # to have been "explained" by it, so it gets its own bucket.
    if abs(rho_clean) < SURVIVES_RHO:
        return "weak_before_capacity"
    if abs(rho_both) < SURVIVES_RHO:
        return "explained_by_capacity"
    if abs(rho_both) < abs(rho_clean) - SHRINK_LIMIT:
        return "shrinks"
    if np.sign(rho_both) != np.sign(rho_clean):
        return "flips"
    return "survives"


def analyse(wide: pd.DataFrame, resamples: int = RESAMPLES) -> dict:
    params = parameter_counts(pd.Series(wide["arch_string"].to_numpy(), index=wide["arch_id"]))
    frame = wide.copy()
    frame["params"] = frame["arch_id"].map(params)
    frame = frame.dropna(subset=["params", "target", "clean"])

    registered = {spec.name: spec.direction for spec in METRIC_SPECS}
    rows = []
    for name, direction in registered.items():
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < MIN_ARCH:
            continue
        x = values[mask]
        y = frame["target"].to_numpy(dtype=float)[mask]
        clean = frame["clean"].to_numpy(dtype=float)[mask]
        capacity = frame["params"].to_numpy(dtype=float)[mask]
        if direction == "lower":
            x = -x
        both = partial_rho(x, y, [clean, capacity])
        only_clean = partial_rho(x, y, [clean])
        low, high, draws = bootstrap_ci(x, y, [clean, capacity], resamples=resamples)
        rows.append(
            {
                "metric": name,
                "n": int(mask.sum()),
                "direction": direction,
                "rho_target": partial_rho(x, y, []),
                "rho_clean": only_clean,
                "rho_clean_params": both,
                "ci_low": low,
                "ci_high": high,
                "delta_from_clean": (
                    None if both is None or only_clean is None else both - only_clean
                ),
                "verdict": classify(only_clean, both),
            }
        )
    rows.sort(key=lambda row: -abs(row["rho_clean_params"] or 0.0))
    return {
        "schema_version": SCHEMA_VERSION,
        "n_architectures": int(len(frame)),
        "parameters_range": [int(frame["params"].min()), int(frame["params"].max())],
        "rho_target_clean": partial_rho(frame["target"], frame["clean"], []),
        "rho_target_params": partial_rho(frame["target"], frame["params"], []),
        "rho_clean_params": partial_rho(frame["clean"], frame["params"], []),
        "thresholds": {"survives_rho": SURVIVES_RHO, "shrink_limit": SHRINK_LIMIT},
        "resamples": resamples,
        "seed": SEED,
        "metrics": rows,
    }


def _fmt(value) -> str:
    return "—" if value is None else f"{value:+.3f}"


def render_markdown(report: dict, wide_path: Path, wide_sha: str) -> str:
    counts: dict[str, int] = {}
    for row in report["metrics"]:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    lines = [
        "# 早期信号在控制「参数量」之后还剩多少（RQ2 第二个混杂）",
        "",
        f"输入：`{wide_path}`（sha256 `{wide_sha}`）",
        f"架构数：{report['n_architectures']}；参数量范围 {report['parameters_range'][0]}–{report['parameters_range'][1]}",
        f"相关：ρ(target,clean) = {_fmt(report['rho_target_clean'])}、"
        f"ρ(target,params) = {_fmt(report['rho_target_params'])}、"
        f"ρ(clean,params) = {_fmt(report['rho_clean_params'])}",
        "",
        "## 结论",
        "",
    ]
    if counts.get("survives", 0):
        lines.append(
            f"- **{counts['survives']} 个指标在同时控制 clean 与参数量后仍有 |偏相关| ≥ "
            f"{SURVIVES_RHO:.2f}**：它们没有因为「架构更大」而消失。"
        )
    if counts.get("shrinks", 0):
        lines.append(
            f"- {counts['shrinks']} 个指标在加入参数量后明显缩小（> {SHRINK_LIMIT:.2f}）："
            "原来的一部分信号其实是「更大的架构更容易同时漂移得多、也训得更好」。"
        )
    if counts.get("explained_by_capacity", 0):
        lines.append(
            f"- {counts['explained_by_capacity']} 个指标原本 ≥ {SURVIVES_RHO:.2f}、"
            f"加控制后掉到 {SURVIVES_RHO:.2f} 以下：它们的独立信息基本就是规模。"
        )
    if counts.get("weak_before_capacity", 0):
        lines.append(
            f"- {counts['weak_before_capacity']} 个指标在只控 clean 时就已经 < {SURVIVES_RHO:.2f}，"
            "谈不上被参数量解释——它们本来就没有超出 clean 的信号。"
        )
    if counts.get("flips", 0):
        lines.append(f"- {counts['flips']} 个指标符号翻转，说明它的相关方向不稳定。")
    unresolved = counts.get("uncovered", 0)
    if unresolved:
        lines.append(f"- {unresolved} 个指标在加控制后无法计算（架构数不足 20 或方差为 0）。")
    if report["metrics"]:
        best = report["metrics"][0]
        if best["rho_clean"] is not None and best["rho_clean_params"] is not None:
            moved = "上升" if abs(best["rho_clean_params"]) > abs(best["rho_clean"]) else "下降"
            lines.append(
                f"- 最强的一个是 `{best['metric']}`：只控 clean 时 {_fmt(best['rho_clean'])}，"
                f"再加参数量后 {_fmt(best['rho_clean_params'])}（**{moved}**）。"
                "参数量在这里更像一个抑制变量——它与 clean 同向、与漂移类指标不同向，"
                "控掉它反而让剩余变异与 target 的关系更清楚。这只是解释，不是已验证的机制。"
            )
    lines += [
        "",
        f"判定：`rho_clean` 已经 < {SURVIVES_RHO:.2f} → weak_before_capacity（本来就不强，不算被解释）；"
        f"否则 `rho_clean_params` < {SURVIVES_RHO:.2f} → explained_by_capacity；"
        f"比 `rho_clean` 低 {SHRINK_LIMIT:.2f} 以上 → shrinks；符号反了 → flips；其余 → survives。",
        "",
        "## 前 15 个指标",
        "",
        "| 指标 | ρ(target) | ρ(仅控 clean) | ρ(控 clean+参数) | 95% 区间 | 判定 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["metrics"][:15]:
        interval = (
            "—"
            if row["ci_low"] is None
            else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        )
        lines.append(
            f"| `{row['metric']}` | {_fmt(row['rho_target'])} | {_fmt(row['rho_clean'])} | "
            f"**{_fmt(row['rho_clean_params'])}** | {interval} | {row['verdict']} |"
        )
    lines += ["", "## 怎么读", ""]
    lines += [
        "- 参数量的计算来自 `Nb201TinyNetwork(ops)` 的实际参数个数，不是估计值；C=16、N=5 下取值很有限，",
        "  所以参数量实际上是一个「op 类型组合」的分组变量，控制它等于在同一档规模的架构内部比较。",
        "- 参数量与 clean、target 本身相关（见上方三个数）。如果后两者都随规模上升，那么",
        "  「控参数量」会同时拿掉一部分真实信号与一部分混杂，读到接近 0 不等于指标无用，",
        "  只说明在这批架构里它没有超出规模的独立信息。",
        "- 这里仍然只是相关，不是因果；也没有控制训练难度或 RobBench 自身的协议差异。",
        "",
        "## 复现",
        "",
        "```powershell",
        f"python scripts/audit_rq2_confounders.py --wide {wide_path} --json <out.json> --report <out.md>",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True, help="wide e0 table with arch_string/target/clean")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()

    wide = load_wide(args.wide)
    report = analyse(wide, resamples=args.resamples)
    report["inputs"] = {"wide": str(args.wide), "wide_sha256": sha256_file(args.wide)}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(
        render_markdown(report, args.wide, report["inputs"]["wide_sha256"]), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for row in report["metrics"]:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    best = report["metrics"][0]
    print(
        f"n={report['n_architectures']} best={best['metric']} "
        f"rho_clean={_fmt(best['rho_clean'])} rho_both={_fmt(best['rho_clean_params'])} "
        f"counts={counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
