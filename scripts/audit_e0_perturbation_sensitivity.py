"""Does the early-signal result survive a different perturbation?

The RQ2 number reported so far (+0.42 on 400 architectures, controlling clean
accuracy and parameter count) comes from one configuration: one FGSM step at
8/255 on 64 test images, with the same weight seed for every architecture.  A
reviewer's first question is whether that number is a property of the
architecture or a property of that knob.

This script recomputes the same bundle on the same architectures under two other
perturbations and reports, per metric, how much the correlation with post-training
robustness moves:

- ``weak``: one FGSM step at 4/255 -- half the radius, so the features barely move.
- ``pgd``: three PGD steps at 8/255 -- same radius, but a much stronger attack.

It does not touch the published tables and does not train anything.  It writes one
table per variant plus a report that puts the three side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    _value,
)
from robust_metric_lab.attacks import make_adversarial  # noqa: E402
from robust_metric_lab.metric_registry import METRIC_SPECS  # noqa: E402
from robust_metric_lab.metrics import compute_feature_metrics  # noqa: E402
from robust_metric_lab.nb201_space import Nb201TinyNetwork  # noqa: E402

# A drift metric registered as "lower" improves when it falls, while the target
# improves when it rises.  Every correlation in this report is computed after the
# sign flip so that a positive number always means "this metric points the right
# way".  Without it the baseline column comes out negative and cannot be compared
# with `audit_rq2_confounders`, which does flip.
DIRECTION = {spec.name: spec.direction for spec in METRIC_SPECS}

SCHEMA_VERSION = 1
METADATA = frozenset(
    {"adv_acc", "clean_acc", "num_candidate_layers", "num_layers", "sampled_layer_indices"}
)
VARIANTS = {
    "weak": {"attack": "fgsm", "epsilon": 4 / 255, "steps": 1},
    "pgd": {"attack": "pgd", "epsilon": 8 / 255, "steps": 3},
}
CONTROL_COLUMNS = ("clean", "params")
MIN_ARCH = 20


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
    # The wide table records parameter count nowhere, but parameter count is a
    # confounder strong enough to be worth controlling (rho +0.71 with the target).
    # Rebuilding each network just to count parameters costs milliseconds.
    if "params" not in frame.columns:
        frame["params"] = [parameter_count(value) for value in frame["arch_string"]]
    return frame


def parameter_count(arch_string: str) -> float | None:
    ops = parse_ops(str(arch_string))
    if ops is None or any(op not in VALID_OPS for op in ops):
        return None
    return float(sum(p.numel() for p in Nb201TinyNetwork(ops).parameters()))


def evaluate_variant(model, x, y, attack: str, epsilon: float, steps: int):
    x_adv = make_adversarial(model, x, y, attack, epsilon, pgd_steps=steps)
    return compute_feature_metrics(model, x, x_adv, y, max_layers=4)


def _merge_safely(rows: list[dict], existing: Path | None) -> pd.DataFrame:
    """`merge_rows` assumes at least one new row; a resumed run may add none."""
    if rows:
        return merge_rows(rows, existing)
    if existing is not None and Path(existing).is_file():
        frame = pd.read_csv(existing)
        frame["arch_id"] = frame["arch_id"].astype(str)
        return frame
    return pd.DataFrame(columns=["arch_id"])


def run_variant(
    frame: pd.DataFrame,
    x: torch.Tensor,
    y: torch.Tensor,
    variant: str,
    weight_seed: int,
    device: str,
    output: Path,
    resume: bool = True,
) -> pd.DataFrame:
    spec = VARIANTS[variant]
    done: set[str] = set()
    if resume and output.is_file():
        done = set(pd.read_csv(output, usecols=["arch_id"])["arch_id"].astype(str))
        print(f"[{variant}] resuming with {len(done)} rows already on disk")
    rows: list[dict] = []
    started = time.time()
    for index, record in frame.iterrows():
        arch_id = str(record["arch_id"])
        if arch_id in done:
            continue
        ops = parse_ops(str(record["arch_string"]))
        if ops is None or any(op not in VALID_OPS for op in ops):
            continue
        torch.manual_seed(weight_seed)
        model = Nb201TinyNetwork(ops).to(device).eval()
        result = evaluate_variant(
            model, x, y, spec["attack"], spec["epsilon"], spec["steps"]
        )
        row = {"arch_id": arch_id}
        for key, value in result.__dict__.items():
            if key in METADATA:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
        row["clean"] = float(result.clean_acc)
        row["params"] = float(sum(p.numel() for p in model.parameters()))
        rows.append(row)
        if len(rows) % 20 == 0:
            _merge_safely(rows, output if resume else None).to_csv(output, index=False)
            elapsed = time.time() - started
            rate = elapsed / max(1, len(rows))
            print(
                f"[{variant}] {len(rows)} done, {rate:.2f}s each, "
                f"{rate * (len(frame) - len(rows) - len(done)) / 60:.1f} min left"
            )
    table = _merge_safely(rows, output if resume else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


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
    resid_x = rx - design @ beta_x
    resid_y = ry - design @ beta_y
    if resid_x.std() == 0 or resid_y.std() == 0:
        return None
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def per_metric(table: pd.DataFrame, target: pd.Series) -> dict[str, dict]:
    """Correlation with the target, raw and after the two controls.

    ``lower`` metrics are negated first, so a positive number always means the
    metric points the same way as robustness.
    """
    metrics = [
        column
        for column in table.columns
        if column not in {"arch_id", "clean", "params", "target"}
    ]
    out = {}
    target = pd.to_numeric(target, errors="coerce").reset_index(drop=True)
    for metric in metrics:
        series = pd.to_numeric(table[metric], errors="coerce").reset_index(drop=True)
        if DIRECTION.get(metric) == "lower":
            series = -series
        pair = pd.DataFrame({"x": series, "y": target}).replace([np.inf, -np.inf], np.nan).dropna()
        raw = None
        if len(pair) >= MIN_ARCH and pair["x"].nunique() > 1:
            raw = float(_rank(pair["x"]).corr(_rank(pair["y"])))
        available = [c for c in CONTROL_COLUMNS if c in table.columns]
        controlled = partial_rho(
            series,
            target,
            [pd.to_numeric(table[c], errors="coerce").reset_index(drop=True) for c in available],
        )
        out[metric] = {"raw": raw, "controlled": controlled}
    return out


def top_metrics(scores: dict, k: int = 5) -> list[str]:
    """The k metrics with the largest controlled correlation, strongest first."""
    ranked = [(name, block["controlled"]) for name, block in scores.items() if block["controlled"] is not None]
    ranked.sort(key=lambda pair: -pair[1])
    return [name for name, _ in ranked[:k]]


def compare_variants(baseline: dict, variant: dict) -> dict:
    """How much does each metric's controlled correlation move between variants?"""
    common = sorted(set(baseline) & set(variant))
    base_vals = [baseline[m]["controlled"] for m in common if baseline[m]["controlled"] is not None]
    var_vals = [variant[m]["controlled"] for m in common if variant[m]["controlled"] is not None]
    pairs = [
        (baseline[m]["controlled"], variant[m]["controlled"])
        for m in common
        if baseline[m]["controlled"] is not None and variant[m]["controlled"] is not None
    ]
    rho = None
    if len(pairs) >= 5:
        xs = pd.Series([p[0] for p in pairs])
        ys = pd.Series([p[1] for p in pairs])
        rho = float(_rank(xs).corr(_rank(ys)))
    moves = [
        {
            "metric": m,
            "baseline": baseline[m]["controlled"],
            "variant": variant[m]["controlled"],
            "delta": (
                None
                if baseline[m]["controlled"] is None or variant[m]["controlled"] is None
                else variant[m]["controlled"] - baseline[m]["controlled"]
            ),
        }
        for m in common
    ]
    moves.sort(key=lambda r: -abs(r["delta"]) if r["delta"] is not None else 1e9)
    base_top = top_metrics(baseline)
    var_top = top_metrics(variant)
    return {
        "metrics": len(common),
        "rho_between_variants": None if rho is None else round(rho, 4),
        "top5_baseline": base_top,
        "top5_variant": var_top,
        "top5_overlap": len(set(base_top) & set(var_top)),
        "best_baseline": max(
            (m for m in common if baseline[m]["controlled"] is not None),
            key=lambda m: baseline[m]["controlled"],
            default=None,
        ),
        "best_variant": max(
            (m for m in common if variant[m]["controlled"] is not None),
            key=lambda m: variant[m]["controlled"],
            default=None,
        ),
        "largest_moves": moves[:10],
    }


def _target_for(table: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Look the target up by architecture id, whatever index the caller passed.

    `target` may be indexed by arch_id (the normal case) or by position (in tests);
    matching on `str(key)` accepts both without silently producing all-NaN columns.
    """
    lookup = {str(key): value for key, value in target.items()}
    return pd.Series([lookup.get(str(arch), np.nan) for arch in table["arch_id"]])


def analyse(baseline_table: pd.DataFrame, variant_tables: dict[str, pd.DataFrame], target: pd.Series) -> dict:
    baseline = per_metric(baseline_table, _target_for(baseline_table, target))
    report = {
        "schema_version": SCHEMA_VERSION,
        "n_architectures": int(len(baseline_table)),
        "variants": {},
        "baseline": baseline,
    }
    for name, table in variant_tables.items():
        scores = per_metric(table, _target_for(table, target))
        report["variants"][name] = {
            "config": VARIANTS[name],
            "n": int(len(table)),
            "scores": scores,
            "comparison": compare_variants(baseline, scores),
        }
    return report


def _fmt(value, spec="%+.3f") -> str:
    return "—" if value is None else spec % value


def render_markdown(report: dict, baseline_path: Path, sha: str) -> str:
    lines = [
        "# 早期信号换一个扰动还在不在（2026-11-12）",
        "",
        f"基准表：`{baseline_path}`（sha256 `{sha}`，{report['n_architectures']} 个架构）",
        "两个变体：`weak` = FGSM 4/255（半径减半）、`pgd` = PGD-3 8/255（同半径、更强攻击）。",
        "所有数字都是**控制 clean 准确率与参数量之后**的偏秩相关。",
        "",
        "## 结论",
        "",
    ]
    for name, block in report["variants"].items():
        comparison = block["comparison"]
        base_best = report["baseline"].get(comparison["best_baseline"], {}).get("controlled")
        var_best = block["scores"].get(comparison["best_variant"], {}).get("controlled")
        lines.append(
            "**{}**：最强指标从 `{}`（{:.3f}）变成 `{}`（{:.3f}）；前五名重合 **{}/5**；"
            "所有指标的偏相关排序相关 {:.3f}。".format(
                name,
                comparison["best_baseline"],
                base_best if base_best is not None else float("nan"),
                comparison["best_variant"],
                var_best if var_best is not None else float("nan"),
                comparison.get("top5_overlap", 0),
                comparison["rho_between_variants"] or float("nan"),
            )
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- `rho_between_variants` 衡量的是「哪些指标在这种扰动下强」这件事有没有变。接近 1 说明排序不变，"
        "接近 0 说明换一个扰动就是另一批指标赢。",
        "- **单指标本身有抽样误差**：n=150 时约 ±0.16，n=400 时约 ±0.10（1.96/√(n−3)）。"
        "所以两个变体之间差 0.2 以内，不要当成差别。",
        "- 真正该看的是**前五名重合度**：重合高说明「换扰动后还是同一批指标赢」，"
        "重合低说明这批指标的强弱是配置的产物。",
        "- 单个指标的偏相关本身有抽样误差（这里只有几百个架构），所以**不要**拿两个变体的差值当效应量，"
        "要看的是整体排序是否稳定。",
        "- 这只检验一种稳健性：扰动强度。没有检验别的随机初始化、别的样本子集、别的训练协议。",
        "",
        "## 移动最大的十个指标",
        "",
        "| 变体 | 指标 | 基准 | 变体 | 差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, block in report["variants"].items():
        for move in block["comparison"]["largest_moves"]:
            lines.append(
                "| {} | `{}` | {} | {} | {} |".format(
                    name, move["metric"], _fmt(move["baseline"]), _fmt(move["variant"]), _fmt(move["delta"])
                )
            )
    lines += ["", "## 复现", "", "```powershell", "python scripts/audit_e0_perturbation_sensitivity.py `"]
    lines += [
        "  --wide <n400 宽表>.csv `",
        "  --data-root <cifar10 目录> `",
        "  --limit 150 `",
        "  --json results\\audits\\e0-perturbation-20261112.json `",
        "  --report results\\audits\\e0-perturbation-20261112.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wide", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse the per-variant tables already on disk instead of running the forward passes",
    )
    args = parser.parse_args(argv)

    source = load_architectures(args.wide, args.limit)
    output_dir = args.output_dir or args.wide.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    x = y = None
    for variant in VARIANTS:
        path = output_dir / f"nb201_e0_{variant}_{len(source)}_{args.samples}.csv"
        if args.reuse and path.is_file():
            table = pd.read_csv(path)
            table["arch_id"] = table["arch_id"].astype(str)
            print(f"[{variant}] reusing {len(table)} rows from {path.name}")
        else:
            if x is None:
                x, y = load_batch(args.data_root, args.samples, args.device)
            table = run_variant(
                source, x, y, variant, args.weight_seed, args.device, path
            )
        tables[variant] = table

    target = source.set_index("arch_id")["target"]
    baseline = pd.read_csv(args.wide)
    baseline["arch_id"] = baseline["arch_id"].astype(str)
    baseline = baseline[baseline["arch_id"].isin(source["arch_id"])]
    # The baseline table predates this script and has no parameter column, but the
    # control set has to be identical across variants or the comparison is meaningless.
    if "params" not in baseline.columns:
        baseline = baseline.merge(
            tables["weak"][["arch_id", "params"]], on="arch_id", how="left"
        )
    report = analyse(baseline, tables, target)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(
        render_markdown(report, args.wide, sha256_file(args.wide)), encoding="utf-8"
    )
    print(
        "n={} variants={}".format(
            report["n_architectures"], {k: v["n"] for k, v in report["variants"].items()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
