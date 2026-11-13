"""Ask whether an early indicator would actually help pick architectures.

Every RQ2 number so far is a rank correlation. A correlation tells you that the
orderings agree; it does not tell you whether acting on the ordering pays off.
The decision a practitioner faces is concrete: measure the early indicator on N
candidates, keep the best k, train those, and hope their robustness is above
average. This script measures exactly that payoff.

The unit is the architecture. For each candidate indicator we sort the 400
architectures, take the top k, and report how much more robust that group is than
the full set. The comparison baselines are deliberate:

  * a random subset of the same size -- the null that any indicator must beat;
  * the trained clean accuracy -- the best cheap proxy, but it needs training;
  * the parameter count -- free, and a known confounder.

If the early indicator cannot beat a random subset, the rank correlation is not
worth acting on, however significant it looks.
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
METADATA = ("arch_id", "arch_string", "target", "clean", "params", "oracle_target")
KS = (10, 20, 30)  # percent of the pool kept
DEFAULT_RESAMPLES = 2000
DEFAULT_RANDOM_DRAWS = 5000
SEED = 20261112
ALPHA = 0.05
USEFUL_DELTA = 0.02  # a top decile must beat the pool by two accuracy points

DIRECTION = {spec.name: spec.direction for spec in METRIC_SPECS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(arch_string: str) -> int | None:
    """Count parameters the way every other RQ2 script does."""
    from robust_metric_lab.nasrobbench201 import parse_nb201_arch
    from robust_metric_lab.nb201_space import Nb201TinyNetwork

    ops = parse_nb201_arch(str(arch_string))
    if ops is None:
        return None
    return sum(p.numel() for p in Nb201TinyNetwork(ops).parameters())


def load_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "arch_id" not in frame.columns:
        raise ValueError("wide table is missing columns: ['arch_id']")
    frame["arch_id"] = frame["arch_id"].astype(str)
    missing = [c for c in ("arch_id", "target", "clean") if c not in frame.columns]
    if "params" not in frame.columns:
        if "arch_string" not in frame.columns:
            missing.append("params or arch_string")
        else:
            frame["params"] = frame["arch_string"].map(parameter_count).astype(float)
    if missing:
        raise ValueError(f"wide table is missing columns: {missing}")
    return frame.dropna(subset=["target", "clean", "params"]).reset_index(drop=True)


def load_seed_report(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("metrics", []):
        name = row.get("metric")
        if not name:
            continue
        value = row.get("value") or {}
        out[name] = {
            "value_rho": value.get("mean_rho"),
            "signal_verdict": row.get("signal_verdict"),
        }
    return out


def higher_is_better(metric: str) -> bool:
    """Clean accuracy and parameter count are not registered metrics.

    They are not in METRIC_SPECS, so the fallback would treat them as "lower is
    better" and the selection would keep the *least* accurate and smallest
    architectures -- a silent inversion that makes both baselines look harmful.
    """
    if metric in ("clean", "params"):
        return True
    return DIRECTION.get(metric, "lower") == "higher"


def metric_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Partial correlation against the target, controlling clean accuracy and size."""
    from audit_rq2_confounders import partial_rho

    rows = []
    for name in frame.columns:
        if name in METADATA or name in ("params", "clean"):
            continue
        column = frame[name]
        if column.isna().any() or column.std() == 0:
            continue
        rho = partial_rho(column, frame["target"], [frame["clean"], frame["params"]])
        if rho is None:
            continue
        rows.append({"metric": name, "partial_rho": rho, "abs_partial": abs(rho)})
    return pd.DataFrame(rows)


def pick_candidates(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    seeds: dict[str, dict],
    limit: int,
) -> list[str]:
    """Keep the strongest metrics that are stable in both senses, then add baselines."""
    ranked = scores.sort_values("abs_partial", ascending=False)["metric"].tolist()
    chosen: list[str] = []
    for name in ranked:
        info = seeds.get(name)
        if not info or info.get("signal_verdict") != "stable_signal":
            continue
        if (info.get("value_rho") or 0.0) < 0.8:
            continue
        if name in frame.columns and name not in chosen:
            chosen.append(name)
        if len(chosen) >= limit:
            break
    for name in ("clean", "params"):
        if name in frame.columns and name not in chosen:
            chosen.append(name)
    return chosen


def curve(frame: pd.DataFrame, metric: str, ks: tuple[int, ...]) -> list[dict]:
    target = frame["target"].to_numpy(float)
    values = frame[metric].to_numpy(float)
    order = np.argsort(-values if higher_is_better(metric) else values)
    rows = []
    for k in ks:
        chosen = order[: k * len(target) // 100]
        mean = float(target[chosen].mean())
        rows.append(
            {
                "metric": metric,
                "k_percent": k,
                "n_selected": int(len(chosen)),
                "mean_target": mean,
                "delta": mean - float(target.mean()),
            }
        )
    return rows


def bootstrap(frame: pd.DataFrame, metric: str, ks: tuple[int, ...], resamples: int) -> dict:
    """Resample architectures, re-selecting the top k inside every draw.

    Re-selecting matters: the uncertainty we care about includes the instability
    of the selection itself, not only the noise in the means of a fixed set.
    """
    target = frame["target"].to_numpy(float)
    values = frame[metric].to_numpy(float)
    n = len(target)
    rng = np.random.default_rng(SEED)
    draws = {k: np.empty(resamples) for k in ks}
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        tb, xb = target[idx], values[idx]
        order = np.argsort(-xb if higher_is_better(metric) else xb)
        for k in ks:
            cut = max(1, k * n // 100)
            draws[k][b] = tb[order[:cut]].mean() - tb.mean()
    return {
        k: {
            "low": float(np.percentile(draws[k], 2.5)),
            "high": float(np.percentile(draws[k], 97.5)),
        }
        for k in ks
    }


def random_baseline(target: np.ndarray, k: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    n = len(target)
    cut = max(1, k * n // 100)
    out = np.empty(draws)
    for i in range(draws):
        out[i] = target[rng.choice(n, size=cut, replace=False)].mean() - target.mean()
    return out


def p_against_random(observed: float, null: np.ndarray) -> float:
    """P(a random subset does at least as well as the selected one)."""
    exceed = int(np.count_nonzero(null >= observed))
    return (exceed + 1) / (len(null) + 1)


def verdict(rows: list[dict], deltas: dict) -> str:
    """Judge the early indicators, and report where the free baselines stand.

    `clean` and `params` are references, not candidates: clean needs a trained
    model, params needs nothing. Calling clean the winner would answer a
    different question than the one this script asks, so the verdict is computed
    over the early indicators only and the baselines are quoted alongside.
    """
    top = [r for r in rows if r["k_percent"] == 10]
    early = [r for r in top if r["metric"] not in ("clean", "params")]
    if not early:
        return "没有可判定的早期指标"

    best = max(early, key=lambda r: r["delta"])
    interval = deltas.get((best["metric"], 10), {})
    low = interval.get("low", -1.0)
    p = best.get("p_random", 1.0)

    def _look(name: str) -> str:
        row = next((r for r in top if r["metric"] == name), None)
        return f"{row['delta']:+.3f}" if row else "—"

    reference = f"参数量（免费）{_look('params')}，clean（需训练）{_look('clean')}"
    params_row = next((r for r in top if r["metric"] == "params"), None)
    params_delta = params_row["delta"] if params_row else None

    if low > 0 and p < ALPHA and best["delta"] >= USEFUL_DELTA:
        if params_delta is not None and best["delta"] <= params_delta:
            return (
                "早期指标有信号但不及免费基线：%s 前 10%% 高 %+.3f（显著，p=%.4f），"
                "而参数量高 %+.3f。既然数参数就能做得更好，"
                "这些指标的实用价值需要用「增量超过参数量」来证明，不能只报显著性。"
                % (best["metric"], best["delta"], p, params_delta)
            )
        return "早期指标可用于筛选：%s 前 10%% 高 %+.3f；参照 %s" % (
            best["metric"],
            best["delta"],
            reference,
        )
    if low <= 0 <= interval.get("high", 1.0):
        return (
            "早期指标不足以筛选架构：最好的 %s 前 10%% 只高 %+.3f（区间跨 0，p=%.3f）；"
            "参照 %s。免费基线更强时，这些指标的实用价值还没有被证明。"
            % (best["metric"], best["delta"], p, reference)
        )
    return (
        "早期指标方向为正但未达门槛：%s 前 10%% 高 %+.3f；参照 %s"
        % (best["metric"], best["delta"], reference)
    )


def analyse(wide: pd.DataFrame, seed_report: dict, limit: int, resamples: int) -> dict:
    scores = metric_scores(wide)
    candidates = pick_candidates(wide, scores, seed_report, limit)
    target = wide["target"].to_numpy(float)
    rng = np.random.default_rng(SEED + 1)
    deltas: dict[tuple[str, int], dict] = {}

    rows: list[dict] = []
    for metric in candidates:
        rows += curve(wide, metric, KS)
        intervals = bootstrap(wide, metric, KS, resamples)
        for k in KS:
            deltas[(metric, k)] = intervals[k]

    for k in KS:
        null = random_baseline(target, k, DEFAULT_RANDOM_DRAWS, rng)
        for row in rows:
            if row["k_percent"] != k:
                continue
            row["random_sd"] = float(null.std())
            row["p_random"] = p_against_random(row["delta"], null)

    for row in rows:
        interval = deltas.get((row["metric"], row["k_percent"]), {})
        row["ci_low"] = interval.get("low")
        row["ci_high"] = interval.get("high")

    return {
        "schema_version": SCHEMA_VERSION,
        "wide": {"path": str(wide.attrs.get("path", "")), "sha256": wide.attrs.get("sha256")},
        "ks": list(KS),
        "resamples": resamples,
        "verdict": verdict(rows, deltas),
        "candidates": candidates,
        "rows": rows,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def render_markdown(report: dict) -> str:
    lines = ["# 早期指标能不能用来挑架构（可操作性检验）", ""]
    lines += ["## 结论", "", report["verdict"], ""]
    lines += [
        "这不是相关分析，而是决策分析：按指标排序取前 k%，看这 k% 的真实鲁棒性比全体高多少。",
        "随机基线是同样大小的随机子集——任何指标都要先赢它。",
        "",
        "## 逐指标结果",
        "",
        "| 指标 | 前 k% | 选中数 | 平均鲁棒性 | 相对全体 | 95% 区间 | 随机基线 p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(report["rows"], key=lambda r: (r["k_percent"], -r["delta"])):
        lines.append(
            "| `%s` | %d%% | %d | %.3f | %s | [%s, %s] | %.4f |"
            % (
                row["metric"],
                row["k_percent"],
                row["n_selected"],
                row["mean_target"],
                fmt(row["delta"]),
                fmt(row.get("ci_low")),
                fmt(row.get("ci_high")),
                row.get("p_random", 1.0),
            )
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "`相对全体` = 选中组的平均鲁棒性 减去 400 个架构的平均鲁棒性。",
        "`随机基线 p` = 随机抽同样多架构、平均鲁棒性不低于当前选择的概率；",
        "它同时考虑了类别不平衡与抽样波动，比只看区间更严格。",
        "`clean` 与 `params` 是对照：前者要训练完才有，后者免费但与鲁棒性强相关（第 19 条）。",
        "早期指标的价值在于**不用训练**，所以它的参照对象是 `params` 与随机，不是 `clean`。",
        "",
        "## 这份报告不能说明什么",
        "",
        "1. 400 个架构是从 6466 个里抽的，不是全集；换一批架构，前 10% 的具体数值会变。",
        "2. 只有一套训练协议（RobBench 官方），target 之间的差异可能来自训练随机性。",
        "3. k 取 10/20/30 是人为的；对 k 的取值敏感说明结论不稳，报告里三档都列出来。",
        "4. 这是相关性证据：指标高与鲁棒性高同时出现，不表示按指标挑就能得到鲁棒模型。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, help="stably-ranked metrics to test")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    wide = load_wide(args.wide)
    wide.attrs["path"] = str(args.wide)
    wide.attrs["sha256"] = sha256_file(args.wide)
    seeds = load_seed_report(args.seed_report)
    report = analyse(wide, seeds, args.limit, args.resamples)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_markdown(report), encoding="utf-8")
    print(report["verdict"])
    for row in report["rows"]:
        if row["k_percent"] == 10:
            print(
                "%-48s delta=%+.3f [%s,%s] p=%.4f"
                % (
                    row["metric"],
                    row["delta"],
                    fmt(row.get("ci_low")),
                    fmt(row.get("ci_high")),
                    row.get("p_random", 1.0),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
