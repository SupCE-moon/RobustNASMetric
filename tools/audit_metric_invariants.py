"""Check physical sanity conditions on the one-pass metric bundle.

The leaderboard ranks 86 numbers, but nothing so far checked that a single one
of them behaves like a distance at all.  A metric can be registered, have a
code path, and still be wrong in a way that only shows up as "it ranks models
oddly" months later.

This audit asks three questions of every delta-style metric, using a tiny
deterministic network so the answers are reproducible on a laptop:

    zero      does it return 0 when the adversarial input equals the clean one?
    finite    is every value finite across the perturbation grid?
    monotone  does it grow with epsilon (Spearman >= MONOTONE_RHO)?

The first two are hard requirements.  The third is a *sign* check: a distance
between clean and perturbed features should increase as the perturbation gets
stronger.  A metric that decreases means either its sign convention is the
opposite of its neighbours', or it is measuring something else entirely --
both worth a human look, neither automatically a bug.

Deliberate non-monotonicity is recorded in EXPECTED_NON_MONOTONE with the
reason, so the report distinguishes "we know and accept this" from "nobody has
looked at this before".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F

SETTING = 8.0 / 255.0
EPS_FRACTIONS = (0.0, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0)
MONOTONE_RHO = 0.7
DECREASING_RHO = -0.7
ZERO_TOLERANCE = 1e-6

# Metrics whose value at zero perturbation is a property of the clean input,
# not a distance, so "should be 0" does not apply.
NOT_A_DISTANCE = frozenset(
    {
        "input_grad_l2",
        "margin_input_grad_ratio",
        "jacobian_spectral_norm",
        "jacobian_effective_rank",
        "jacobian_spectral_decay",
        "feature_spectral_erank_static",
        "feature_spectral_participation_ratio_static",
        "feature_spectral_decay_exponent_static",
        "natural_loss",
        "adversarial_loss",
        "clean_acc",
        "adv_acc",
        "num_layers",
        "num_candidate_layers",
    }
)

# Still expected to vanish at zero perturbation, but not named "*delta".
ZERO_EXTRA = {"logit_kl": 1e-6}

# Known placeholder: needs a two-pass importance vector, written as NaN by the
# one-pass bundle.  See tools/audit_metric_implementation.py.
KNOWN_NAN = frozenset({"perturbation_capacity_importance_delta"})

# Non-monotonicity we have looked at and accept, with the reason on record.
EXPECTED_NON_MONOTONE = {
    "feature_box_dimension_delta": (
        "box-counting dimension is a quantised geometric quantity; below "
        "one pixel of change the count jumps between integers"
    ),
}

METADATA = frozenset({"num_layers", "num_candidate_layers", "sampled_layer_indices"})


class TinyNet(torch.nn.Module):
    """Two conv layers, deterministic, small enough to run in a test."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).mean(dim=(2, 3)))


def build_inputs(seed: int = 0, batch: int = 6) -> tuple[TinyNet, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyNet().eval()
    x = torch.rand(batch, 3, 8, 8)
    y = torch.randint(0, 2, (batch,))
    return model, x, y


def fgsm_sign(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unit-sign input gradient of the cross-entropy loss.

    Note the `.grad.sign()`: `x.detach().sign()` would silently give the clean
    input's sign, which is a mistake this module's first draft made and which
    produced seven identical rows before it was caught.
    """
    xg = x.clone().requires_grad_(True)
    F.cross_entropy(model(xg), y).backward()
    assert xg.grad is not None
    return xg.grad.sign()


def sweep(model, x, y, fractions=EPS_FRACTIONS, setting: float = SETTING) -> dict:
    """Metric values at each epsilon, keyed by field name."""
    from robust_metric_lab.metrics import MetricResult, compute_feature_metrics

    sign = fgsm_sign(model, x, y)
    epsilons = [setting * f for f in fractions]
    values: dict[str, list] = {}
    for eps in epsilons:
        adv = (x + eps * sign).clamp(0, 1)
        result = compute_feature_metrics(model, x, adv, y)
        for field in dataclasses.fields(MetricResult):
            values.setdefault(field.name, []).append(getattr(result, field.name))
    return {"epsilons": epsilons, "values": values, "sign_magnitude": float(sign.abs().sum())}


def spearman(xs, ys) -> float | None:
    n = len(xs)
    if n < 3 or len(set(ys)) == 1:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else max(-1.0, min(1.0, num / den))


def _average_ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def is_distance(name: str) -> bool:
    if name in NOT_A_DISTANCE or name in METADATA:
        return False
    if name in ZERO_EXTRA:
        return True
    return name.endswith("_delta") or name == "feature_delta"


def classify(name: str, series: list, epsilons: list) -> dict:
    row = {"name": name, "values": series}

    if name in METADATA:
        row["finite"] = True
        row["status"] = "metadata"
        row["detail"] = "bookkeeping field, not a metric"
        return row

    finite = all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf") for v in series)
    row["finite"] = finite

    if name in KNOWN_NAN:
        row["status"] = "known_nan"
        row["detail"] = "NaN placeholder; real value comes from the two-pass path"
        return row
    if not finite:
        row["status"] = "non_finite"
        row["detail"] = "value is NaN or inf"
        return row

    numeric = [float(v) for v in series]
    if is_distance(name):
        tol = ZERO_EXTRA.get(name, ZERO_TOLERANCE)
        if abs(numeric[0]) > tol:
            row["status"] = "nonzero_at_zero"
            row["detail"] = f"expected ~0 at eps=0, got {numeric[0]:.3e}"
            return row

    if not is_distance(name):
        row["status"] = "ok"
        row["detail"] = "not a distance; only finiteness checked"
        row["rho"] = None
        return row

    rho = spearman(epsilons, numeric)
    row["rho"] = rho
    if rho is None:
        row["status"] = "constant"
        row["detail"] = "identical at every epsilon; cannot rank"
        return row
    if rho >= MONOTONE_RHO:
        row["status"] = "ok"
        row["detail"] = "increases with epsilon"
        return row
    if name in EXPECTED_NON_MONOTONE:
        row["status"] = "ok_expected"
        row["detail"] = EXPECTED_NON_MONOTONE[name]
        return row
    if rho <= DECREASING_RHO:
        row["status"] = "decreasing"
        row["detail"] = "decreases with epsilon: sign opposite to the other deltas"
        return row
    row["status"] = "unordered"
    row["detail"] = "does not track epsilon; neither increasing nor decreasing"
    return row


def analyse(model, x, y) -> dict:
    data = sweep(model, x, y)
    rows = [classify(n, s, data["epsilons"]) for n, s in sorted(data["values"].items())]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": 1,
        "setting": SETTING,
        "epsilons": data["epsilons"],
        "sign_magnitude": data["sign_magnitude"],
        "thresholds": {"monotone_rho": MONOTONE_RHO, "zero_tolerance": ZERO_TOLERANCE},
        "counts": counts,
        "rows": rows,
    }


ZH = {
    "ok": "通过",
    "ok_expected": "已知非单调（有记录）",
    "known_nan": "已知占位 NaN",
    "nonzero_at_zero": "零扰动时不为 0",
    "non_finite": "出现 NaN/inf",
    "decreasing": "随扰动**下降**（方向与其他 delta 相反）",
    "unordered": "不随扰动有序变化",
    "constant": "在所有扰动下恒定",
    "metadata": "记录字段（不是指标）",
}


def render_markdown(report: dict) -> str:
    counts = report["counts"]
    total = sum(counts.values())
    eps = ", ".join(f"{e*255:.2f}/255" for e in report["epsilons"])
    lines = [
        "# 指标基本不变量审计",
        "",
        f"- 小网络：两层卷积（`tools/audit_metric_invariants.py::TinyNet`），种子 0，6 张 8×8 图",
        f"- 扰动网格：{eps}",
        f"- 判据：`eps=0` 时距离类应为 0（容差 {report['thresholds']['zero_tolerance']:g}）；"
        f"随 `eps` 单调上升要求 Spearman ≥ {report['thresholds']['monotone_rho']}",
        f"- 共检查 **{total}** 个字段",
        "",
        "## 结论",
        "",
        "| 判定 | 数量 |",
        "| --- | --- |",
    ]
    for status in ["ok", "ok_expected", "known_nan", "nonzero_at_zero", "non_finite", "decreasing", "unordered", "constant", "metadata"]:
        if counts.get(status):
            lines.append(f"| {ZH[status]} | {counts[status]} |")
    lines.append("")
    lines.append("## 需要人看的")
    lines.append("")
    flagged = [r for r in report["rows"] if r["status"] not in {"ok", "ok_expected", "metadata"}]
    if not flagged:
        lines.append("无。")
    else:
        lines.append("| 指标 | 判定 | Spearman | 起点值 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in flagged:
            rho = row.get("rho")
            rho_text = "—" if rho is None else f"{rho:+.3f}"
            first = row["values"][0]
            first_text = f"{first:.3e}" if isinstance(first, float) else str(first)
            lines.append(f"| `{row['name']}` | {ZH[row['status']]} | {rho_text} | {first_text} | {row['detail']} |")
    lines += [
        "",
        "## 怎么读",
        "",
        "- `通过` 只说明这个指标在**这一个随机小网络**上守住了三条基本性质，"
        "不说明它测的东西有意义，也不说明它在真实 CIFAR 模型上同样表现。",
        "- `随扰动下降` 不等于 bug：可能它本来就有符号（例如相对变化量），"
        "但注册表把它声明成 `lower` 时，它与其他 55 个 delta 的方向约定就不一致了，"
        "跨指标比较排名时要单独说明。",
        "- `不随扰动有序变化` 常见于量化/离散量（盒计数维度）或饱和量，"
        "这类指标在小扰动区间的排序能力要单独验证。",
        "- 这个审计**不检查**指标与鲁棒准确率的相关性——那是榜单在做的事。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument("--report", type=Path, help="write a Markdown report")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model, x, y = build_inputs(seed=args.seed)
    report = analyse(model, x, y)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(report), encoding="utf-8")

    counts = report["counts"]
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    failures = counts.get("nonzero_at_zero", 0) + counts.get("non_finite", 0)
    for row in report["rows"]:
        if row["status"] in {"nonzero_at_zero", "non_finite", "decreasing", "unordered"}:
            print(f"{row['status']:16s} {row['name']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
