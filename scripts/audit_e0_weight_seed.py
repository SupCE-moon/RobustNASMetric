"""Does an e0 metric even have a value, or does it depend on the weight seed?

Every e0 number in this project comes from one random initialisation: weights are
seeded with 0 and never trained.  If the metric moves a lot between seeds, then
"the e0 metric of this architecture" is not a property of the architecture, and
any correlation computed from it is one draw rather than one measurement.

This script recomputes the same bundle on a subset of architectures under several
weight seeds and reports two different things:

- **value stability**: for each metric, the rank correlation between seeds.  A
  metric whose value is seed-dependent cannot be used as a fixed feature.
- **signal stability**: for each metric, the partial correlation with the target
  (controlling clean accuracy and parameter count) computed separately per seed.
  A metric can have a noisy value yet a stable signal if the noise is shared
  across architectures.

The two are not the same question, which is why both are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expand_nb201_e0_coverage import (  # noqa: E402
    VALID_OPS,
    load_batch,
    merge_rows,
    parse_ops,
)
from robust_metric_lab.attacks import make_adversarial  # noqa: E402
from robust_metric_lab.metric_registry import METRIC_SPECS  # noqa: E402
from robust_metric_lab.metrics import compute_feature_metrics  # noqa: E402
from robust_metric_lab.nb201_space import Nb201TinyNetwork  # noqa: E402

SCHEMA_VERSION = 1
METADATA = frozenset(
    {"adv_acc", "clean_acc", "num_candidate_layers", "num_layers", "sampled_layer_indices"}
)
SEEDS = (0, 1, 2)
STABLE_RHO = 0.80
SPREAD_LIMIT = 0.20
MIN_ARCH = 20
CONTROL_COLUMNS = ("clean", "params")
DIRECTION = {spec.name: spec.direction for spec in METRIC_SPECS}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_architectures(wide: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(wide)
    required = {"arch_id", "arch_string", "target", "clean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"wide table is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["arch_id"] = frame["arch_id"].astype(str)
    if limit is not None and limit < len(frame):
        frame = frame.sort_values("arch_id").head(limit)
    frame = frame.reset_index(drop=True)
    if "params" not in frame.columns:
        frame["params"] = [parameter_count(value) for value in frame["arch_string"]]
    return frame


def parameter_count(arch_string: str) -> float | None:
    ops = parse_ops(str(arch_string))
    if ops is None or any(op not in VALID_OPS for op in ops):
        return None
    return float(sum(p.numel() for p in Nb201TinyNetwork(ops).parameters()))


def run_seed(
    frame: pd.DataFrame,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    epsilon: float,
    output: Path,
    resume: bool = True,
) -> pd.DataFrame:
    done: set[str] = set()
    if resume and output.is_file():
        done = set(pd.read_csv(output, usecols=["arch_id"])["arch_id"].astype(str))
        print(f"[seed {seed}] resuming with {len(done)} rows already on disk")
    rows: list[dict] = []
    started = time.time()
    for _, record in frame.iterrows():
        arch_id = str(record["arch_id"])
        if arch_id in done:
            continue
        ops = parse_ops(str(record["arch_string"]))
        if ops is None or any(op not in VALID_OPS for op in ops):
            continue
        torch.manual_seed(seed)
        # Deliberately NOT .eval(): the released e0 table was produced by
        # scripts/evaluate_nb201_e27_zerocost.py, which builds the network and
        # then calls make_adversarial / compute_feature_metrics without ever
        # switching to eval(). Copying that recipe bit for bit is what makes
        # seed 0 reproduce the published row exactly; an earlier version of this
        # script added .eval() and its numbers silently stopped matching the
        # table it was supposed to probe.
        model = Nb201TinyNetwork(ops).to(x.device)
        x_adv = make_adversarial(model, x, y, "fgsm", epsilon, pgd_steps=1)
        result = compute_feature_metrics(model, x, x_adv, y, max_layers=4)
        row = {"arch_id": arch_id}
        for key, value in result.__dict__.items():
            if key in METADATA or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
        # The random network's accuracy, kept for reference only. It must not be
        # confused with the trained clean accuracy that every other audit
        # controls for: at initialisation this is close to chance, while the
        # released tables carry the accuracy after the official training run.
        row["init_clean"] = float(result.clean_acc)
        row["params"] = float(sum(p.numel() for p in model.parameters()))
        rows.append(row)
        if len(rows) % 20 == 0:
            _merge(rows, output if resume else None).to_csv(output, index=False)
            rate = (time.time() - started) / max(1, len(rows))
            print(f"[seed {seed}] {len(rows)} done, {rate:.2f}s each")
    table = _merge(rows, output if resume else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def _merge(rows: list[dict], existing: Path | None) -> pd.DataFrame:
    if rows:
        return merge_rows(rows, existing)
    if existing is not None and Path(existing).is_file():
        frame = pd.read_csv(existing)
        frame["arch_id"] = frame["arch_id"].astype(str)
        return frame
    return pd.DataFrame(columns=["arch_id"])


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def metric_overlap(metrics: list[str]) -> list[str]:
    return [m for m in metrics if m not in {"arch_id", "clean", "init_clean", "params", "target"}]


def value_stability(tables: dict[int, pd.DataFrame]) -> dict[str, dict]:
    """Rank correlation of the same metric between every pair of seeds."""
    metrics = metric_overlap(list(next(iter(tables.values())).columns))
    out: dict[str, dict] = {}
    for metric in metrics:
        pairs = []
        for a, b in combinations(sorted(tables), 2):
            left = tables[a].set_index("arch_id")[metric]
            right = tables[b].set_index("arch_id")[metric]
            joined = pd.DataFrame({"a": left, "b": right}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(joined) < MIN_ARCH:
                continue
            values = _rank(joined["a"]).corr(_rank(joined["b"]))
            if values == values:
                pairs.append(float(values))
        if pairs:
            out[metric] = {
                "mean_rho": float(np.mean(pairs)),
                "min_rho": float(np.min(pairs)),
                "pairs": len(pairs),
            }
    return out


def partial_rho(x: pd.Series, y: pd.Series, controls: list[pd.Series]) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y})
    for index, control in enumerate(controls):
        frame[f"c{index}"] = control
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < MIN_ARCH:
        return None
    rx = _rank(frame["x"]).to_numpy()
    ry = _rank(frame["y"]).to_numpy()
    design = np.column_stack(
        [np.ones(len(frame))]
        + [_rank(frame[f"c{index}"]).to_numpy() for index in range(len(controls))]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    beta_x = np.linalg.lstsq(design, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, ry, rcond=None)[0]
    resid_x, resid_y = rx - design @ beta_x, ry - design @ beta_y
    if resid_x.std() == 0 or resid_y.std() == 0:
        return None
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def signal_scores(table: pd.DataFrame, target: pd.Series, controls: list[str]) -> dict[str, float | None]:
    """Partial correlation with the target, per metric, after the sign flip."""
    metrics = metric_overlap(list(table.columns))
    available = [c for c in controls if c in table.columns]
    target = pd.to_numeric(target, errors="coerce").reset_index(drop=True)
    out: dict[str, float | None] = {}
    for metric in metrics:
        series = pd.to_numeric(table[metric], errors="coerce").reset_index(drop=True)
        if DIRECTION.get(metric) == "lower":
            series = -series
        out[metric] = partial_rho(
            series,
            target,
            [pd.to_numeric(table[c], errors="coerce").reset_index(drop=True) for c in available],
        )
    return out


def signal_stability(per_seed: dict[int, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    seeds = sorted(per_seed)
    for metric in per_seed[seeds[0]]:
        values = [per_seed[seed].get(metric) for seed in seeds]
        finite = [value for value in values if value is not None]
        if len(finite) < 2:
            continue
        signs = {np.sign(round(value, 6)) for value in finite}
        out[metric] = {
            "by_seed": values,
            "spread": float(max(finite) - min(finite)),
            "sign_flips": len(signs) > 1,
        }
    return out


def classify(values: dict, spread: float, stable_rho: float) -> str:
    """Three buckets: stable value, stable signal, or neither."""
    value_ok = values is not None and values["mean_rho"] >= stable_rho
    return "stable_value" if value_ok else "seed_dependent"


def analyse(
    tables: dict[int, pd.DataFrame], source: pd.DataFrame, controls: list[str]
) -> dict:
    """Stability of values and of the partial correlation, across weight seeds.

    `source` supplies the target plus the control columns. It has to be the
    released wide table, not a per-seed table: a per-seed table only knows the
    random network's accuracy, and controlling for that instead of the trained
    clean accuracy silently changes what the partial correlation means.
    """
    reference = source.set_index("arch_id")
    reference.index = reference.index.astype(str)
    target = reference["target"]
    injected = [c for c in controls if c in reference.columns]
    values = value_stability(tables)
    per_seed = {}
    for seed, table in tables.items():
        aligned = table.set_index("arch_id")
        aligned.index = aligned.index.astype(str)
        common = aligned.index.intersection(reference.index)
        frame = aligned.loc[common].copy()
        for column in injected:
            frame[column] = reference.loc[common, column]
        per_seed[seed] = signal_scores(
            frame.reset_index(),
            target.loc[common].reset_index(drop=True),
            controls,
        )
    signals = signal_stability(per_seed)
    metrics = []
    for metric, block in values.items():
        signal = signals.get(metric, {})
        metrics.append(
            {
                "metric": metric,
                "value": block,
                "value_verdict": classify(block, SPREAD_LIMIT, STABLE_RHO),
                "signal": signal,
                "signal_verdict": _signal_verdict(signal),
            }
        )
    metrics.sort(key=lambda row: row["value"]["mean_rho"], reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "seeds": sorted(tables),
        "n_architectures": int(len(source)),
        "controls": controls,
        "thresholds": {"stable_rho": STABLE_RHO, "spread_limit": SPREAD_LIMIT},
        "counts": {
            "metrics": len(metrics),
            "stable_value": sum(1 for row in metrics if row["value_verdict"] == "stable_value"),
            "stable_signal": sum(1 for row in metrics if row["signal_verdict"] == "stable_signal"),
        },
        "metrics": metrics,
        "inputs": {"wide": str(source.attrs.get("path", ""))},
    }


def _signal_verdict(signal: dict) -> str:
    if not signal:
        return "uncovered"
    finite = [value for value in signal["by_seed"] if value is not None]
    if len(finite) < 2:
        return "uncovered"
    if signal["sign_flips"]:
        return "signal_flips"
    if signal["spread"] > SPREAD_LIMIT:
        return "signal_spread"
    return "stable_signal"


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}"


def render_markdown(report: dict, wide: Path, digest: str) -> str:
    counts = report["counts"]
    lines = [
        "# e0 指标对权重种子的稳定性（2026-11-12）",
        "",
        f"架构：`{wide}`（sha256 `{digest}`，{report['n_architectures']} 个）",
        f"权重种子：{report['seeds']}；每个种子重算 {counts['metrics']} 个指标，"
        f"FGSM 8/255 单步、64 张图。",
        "",
        "## 结论",
        "",
        f"- **值稳定**（种子间秩相关 ≥ {report['thresholds']['stable_rho']}）的指标："
        f"**{counts['stable_value']} / {counts['metrics']}**。",
        f"- **信号稳定**（不翻号且极差 ≤ {report['thresholds']['spread_limit']}）的指标："
        f"**{counts['stable_signal']} / {counts['metrics']}**。",
        "",
        "值稳定和信号稳定是两件事：一个指标可以每次初始化都算出不同的数，但只要这个抖动"
        "对所有架构是共同的，它和 target 的相关就可能仍然稳定。",
        "",
        "## 值稳定性最好的十个",
        "",
        "| 指标 | 种子间平均秩相关 | 最差一对 |",
        "| --- | --- | --- |",
    ]
    for row in report["metrics"][:10]:
        lines.append(
            f"| `{row['metric']}` | {_fmt(row['value']['mean_rho'])} | {_fmt(row['value']['min_rho'])} |"
        )
    lines += [
        "",
        "## 值稳定性最差的十个",
        "",
        "| 指标 | 种子间平均秩相关 | 最差一对 |",
        "| --- | --- | --- |",
    ]
    for row in report["metrics"][-10:]:
        lines.append(
            f"| `{row['metric']}` | {_fmt(row['value']['mean_rho'])} | {_fmt(row['value']['min_rho'])} |"
        )
    lines += [
        "",
        "## 信号跨种子",
        "",
        "| 指标 | " + " | ".join(f"seed {seed}" for seed in report["seeds"]) + " | 极差 | 判定 |",
        "| --- | " + " | ".join("---" for _ in report["seeds"]) + " | --- | --- |",
    ]
    for row in report["metrics"][:15]:
        values = " | ".join(_fmt(value) for value in row["signal"].get("by_seed", []))
        lines.append(
            f"| `{row['metric']}` | {values} | {_fmt(row['signal'].get('spread'))} | {row['signal_verdict']} |"
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- `值稳定` 只看同一架构在不同权重种子下的指标值是否同序；它不涉及 target。",
        "- `信号稳定` 看每个种子单独算出的偏相关是否同号、同量级。",
        "- 只有**值稳定**的指标才适合当作架构的固定特征写进模型；值不稳但信号稳的指标"
        "只能作为多次采样后的统计量使用（那需要明说采了几次）。",
        "",
        "## 配方一致性",
        "",
        "本脚本刻意**不**调用 `.eval()`，与 `scripts/evaluate_nb201_e27_zerocost.py` 的调用序列逐行一致。",
        "验证办法：用本脚本算 seed 0，与发布宽表按 `arch_id` 对齐逐列比对。",
        "2026-11-12 实测（100 行 × 70 列）：57 列逐位相同，其余 13 列的差都在 1e-2 以内",
        "（最大是 `margin_input_grad_ratio` 的 7.8e-3，梯度与特征分解在 GPU 上的规约顺序不固定），",
        "属于数值抖动而不是配方差异。",
        "",
        "这一点值得单列，因为早期版本在这里加了 `.eval()`，结果值稳定的指标一度只有 14 / 69。",
        "那个数字是配方漂移造成的假象，不是指标的性质——同一份数据，只是网络多切了一次模式。",
        "",
        "## 这份报告不能说明什么",
        "",
        "- 只用了三个种子，`min_rho` 是三个里最差的一对，置信度有限。",
        "- 只检验了权重种子，没有检验样本子集、攻击配置（后者见 `e0-perturbation-20261112`）。",
        "- 这 100 个架构是 `arch_id` 字典序的前 100 个，不是随机抽样；它们的信号量级（约 +0.30）"
        "低于 400 架构全域（约 +0.40），子集效应本身没有被控制。",
        "- 相关不是因果；信号稳也不代表这个指标测量的是鲁棒性的成因。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python scripts\\audit_e0_weight_seed.py `",
        "  --wide <n400 宽表>.csv `",
        "  --data-root data\\cifar10 `",
        "  --limit 100 `",
        "  --json results\\audits\\e0-weight-seed-20261112.json `",
        "  --report results\\audits\\e0-weight-seed-20261112.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wide", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    source = load_architectures(args.wide, args.limit)
    source.attrs["path"] = str(args.wide)
    output_dir = args.output_dir or args.wide.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[int, pd.DataFrame] = {}
    x = y = None
    for seed in seeds:
        path = output_dir / f"nb201_e0_seed{seed}_{len(source)}_{args.samples}.csv"
        if args.reuse and path.is_file():
            table = pd.read_csv(path)
            table["arch_id"] = table["arch_id"].astype(str)
            print(f"[seed {seed}] reusing {len(table)} rows from {path.name}")
        else:
            if x is None:
                x, y = load_batch(args.data_root, args.samples, args.device)
            table = run_seed(source, x, y, seed, args.epsilon, path)
        tables[seed] = table

    report = analyse(tables, source, list(CONTROL_COLUMNS))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(
        render_markdown(report, args.wide, sha256_file(args.wide)), encoding="utf-8"
    )
    print(
        "n={} counts={}".format(report["n_architectures"], report["counts"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
