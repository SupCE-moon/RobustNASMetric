"""Test whether epoch-0 metrics predict a trained model's robust accuracy.

This is the first direct check of RQ2 ("do early-training indicators predict the
robustness an architecture would reach after training?"), and it needs no GPU:
the NAS-RobBench-201 release already contains, for 6466 architectures, clean
accuracy and robust accuracy after their training protocol.

The trap this audit is built to avoid: in that release, clean accuracy and
AutoAttack accuracy correlate at rho = +0.98.  Any epoch-0 metric that merely
predicts clean accuracy will therefore look like a robustness predictor.  So every
metric is reported three times -- raw correlation with robust accuracy, its
correlation with clean accuracy, and the partial rank correlation that removes
clean accuracy.  A metric only counts as "robustness" evidence if the partial
value survives.

Outputs a verdict per metric:
    direct      partial |rho| >= DIRECT_RHO
    via_clean   raw |rho| >= DIRECT_RHO but partial |rho| < VIA_CLEAN_RHO
    suppressed  partial flips sign relative to the raw correlation
    weak        neither
    uncovered   too few finite pairs

Caveats that the report states and that the numbers cannot fix: the epoch-0
metrics come from one initialisation, the target comes from someone else's
training protocol, the architecture sample is the 64 official ones, and the target
is not a causal quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

E0_SETTING = "nb201_e0"
TARGET_FIELD = "autoattack"
CONTROL_FIELD = "val_acc"
RESAMPLES = 2000
SEED = 20261112
CI = (2.5, 97.5)
DIRECT_RHO = 0.30
VIA_CLEAN_RHO = 0.10
MIN_PAIRS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, setting: str = E0_SETTING) -> pd.DataFrame:
    """Read the long-format CSV and hand it to :func:`load_rows_from_frame`."""
    return load_rows_from_frame(pd.read_csv(path), setting=setting)


def load_rows_from_frame(rows: pd.DataFrame, setting: str = E0_SETTING) -> pd.DataFrame:
    """Return a models x metrics frame with every metric oriented so higher is better."""
    frame = rows[rows["setting"] == setting].copy()
    lower = frame["direction"].eq("lower")
    frame.loc[lower, "metric_value"] = -frame.loc[lower, "metric_value"]
    return frame.pivot_table(
        index="model_or_arch_id", columns="metric_name", values="metric_value", dropna=False
    )


def _three_seed(value):
    if isinstance(value, dict):
        return value.get("threeseed", np.nan)
    return value


def load_robbench(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))["cifar10"]
    frame = pd.DataFrame(
        {
            arch: {
                "target": _three_seed(entry.get(TARGET_FIELD)),
                "clean": _three_seed(entry.get(CONTROL_FIELD)),
            }
            for arch, entry in payload.items()
        }
    ).T
    return frame.astype(float)


def _residualise(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(control), control])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def _rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values).astype(float)


def partial_spearman(x, y, control) -> float | None:
    """Rank partial correlation of x and y after removing control.

    Ranking before regressing keeps this a Spearman-family statistic; regressing
    raw values and then taking a Pearson correlation would quietly become a
    different estimator.
    """
    if len(x) < MIN_PAIRS:
        return None
    rx, ry, rc = _rank(x), _rank(y), _rank(control)
    ex, ey = _residualise(rx, rc), _residualise(ry, rc)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return None
    return float(np.corrcoef(ex, ey)[0, 1])


def _rho(x, y) -> float | None:
    if len(x) < MIN_PAIRS or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return None if value != value else float(value)


def classify(raw: float | None, partial: float | None) -> str:
    if raw is None or partial is None:
        return "uncovered"
    if abs(raw) >= DIRECT_RHO and abs(partial) < VIA_CLEAN_RHO:
        return "via_clean"
    if raw * partial < 0 and abs(raw) >= DIRECT_RHO and abs(partial) >= VIA_CLEAN_RHO:
        return "suppressed"
    if abs(partial) >= DIRECT_RHO:
        return "direct"
    return "weak"


def bootstrap_ci(x, y, control, resamples: int = RESAMPLES, seed: int = SEED):
    """Percentile CI for the partial correlation, resampling architectures."""
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        value = partial_spearman(x[idx], y[idx], control[idx])
        if value is not None:
            draws.append(value)
    if not draws:
        return None, None, 0
    return (
        float(np.percentile(draws, CI[0])),
        float(np.percentile(draws, CI[1])),
        len(draws),
    )


def permutation_p(x, y, control, observed: float | None, resamples: int = RESAMPLES, seed: int = SEED):
    """p = fraction of shuffled targets whose |partial| reaches the observed one.

    Uses (exceed + 1) / (resamples + 1): with B shuffles the smallest reportable
    p is 1/(B+1), and printing 0.0000 is false precision.
    """
    if observed is None:
        return None
    rng = np.random.default_rng(seed + 1)
    exceed = 0
    for _ in range(resamples):
        shuffled = rng.permutation(y)
        value = partial_spearman(x, shuffled, control)
        if value is not None and abs(value) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (resamples + 1)


def analyse(metrics: pd.DataFrame, robbench: pd.DataFrame, resamples: int = RESAMPLES) -> dict:
    joined = metrics.join(robbench, how="inner")
    rows = []
    for name in metrics.columns:
        frame = joined[[name, "target", "clean"]].dropna()
        x = frame[name].to_numpy(dtype=float)
        y = frame["target"].to_numpy(dtype=float)
        c = frame["clean"].to_numpy(dtype=float)
        raw = _rho(x, y)
        control = _rho(x, c)
        partial = partial_spearman(x, y, c)
        low, high, draws = bootstrap_ci(x, y, c, resamples=resamples)
        p = permutation_p(x, y, c, partial, resamples=resamples)
        rows.append(
            {
                "metric": name,
                "n": int(len(frame)),
                "rho_target": raw,
                "rho_clean": control,
                "partial_clean": partial,
                "partial_ci_low": low,
                "partial_ci_high": high,
                "bootstrap_draws": draws,
                "permutation_p": p,
                "verdict": classify(raw, partial),
            }
        )
    rows.sort(key=lambda row: -abs(row["partial_clean"] or 0.0))
    return {
        "schema_version": 1,
        "setting": E0_SETTING,
        "target_field": TARGET_FIELD,
        "control_field": CONTROL_FIELD,
        "thresholds": {"direct_rho": DIRECT_RHO, "via_clean_rho": VIA_CLEAN_RHO},
        "n_architectures": int(len(joined)),
        "rho_target_clean": _rho(joined["target"], joined["clean"]),
        "resamples": resamples,
        "seed": SEED,
        "metrics": rows,
    }


VERDICT_ZH = {
    "direct": "控制 clean 准确率后仍有相关",
    "via_clean": "与 clean 准确率高度共线，控制后基本消失",
    "suppressed": "控制后符号翻转（抑制效应）",
    "weak": "两边都弱",
    "uncovered": "可用样本不足",
}


def _fmt(value) -> str:
    return "—" if value is None else f"{value:+.3f}"


def render_markdown(report: dict) -> str:
    counts: dict[str, int] = {}
    for row in report["metrics"]:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    lines = [
        "# 早期指标能否预测训练后的鲁棒准确率（第一版证据）",
        "",
        f"- 早期指标：`{report['setting']}`（训练第 0 个 epoch，即随机初始化后的前向）",
        f"- 目标：NAS-RobBench-201 的 `{report['target_field']}`（三 seed 均值，训练后的数据）",
        f"- 控制变量：`{report['control_field']}`（同一份数据的训练后 clean 准确率）",
        f"- 架构数：**{report['n_architectures']}**；重采样 {report['resamples']} 次，种子 {report['seed']}",
        "",
        "## 为什么必须控制 clean 准确率",
        "",
        f"在这份数据里，训练后 clean 准确率与 AutoAttack 准确率的秩相关是 "
        f"**{_fmt(report['rho_target_clean'])}**。也就是说，任何能预测 clean 准确率的早期指标，"
        "都会看起来像在预测鲁棒性。下面的「偏相关」列把 clean 准确率的影响去掉之后再看。",
        "",
        "## 判定分布",
        "",
        "| 判定 | 数量 | 含义 |",
        "| --- | --- | --- |",
    ]
    for key in ("direct", "via_clean", "suppressed", "weak", "uncovered"):
        if key in counts:
            lines.append(f"| `{key}` | {counts[key]} | {VERDICT_ZH[key]} |")
    lines += [
        "",
        "## 偏相关最强的 12 个指标",
        "",
        "| 指标 | n | ρ(target) | ρ(clean) | 偏相关 | 95% 区间 | 置换 p | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["metrics"][:12]:
        ci = f"[{_fmt(row['partial_ci_low'])}, {_fmt(row['partial_ci_high'])}]"
        p = "—" if row["permutation_p"] is None else f"{row['permutation_p']:.4f}"
        lines.append(
            f"| `{row['metric']}` | {row['n']} | {_fmt(row['rho_target'])} | "
            f"{_fmt(row['rho_clean'])} | {_fmt(row['partial_clean'])} | {ci} | {p} | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- `ρ(target)` 是原始相关：早期指标越好、训练后越鲁棒吗。",
        "- `ρ(clean)` 是同一指标与训练后 clean 准确率的相关。它高，说明这个指标主要在测"
        "「这个架构好不好训」，而不是「这个架构抗不抗攻击」。",
        "- `偏相关` 把 clean 准确率去掉之后剩下的相关。**这一列才是 RQ2 的证据**。",
        "- `via_clean` 不是「指标没用」，而是「它预测鲁棒性的方式，是预测 clean 准确率」。"
        "要谈抗攻击能力，得另找证据。",
        "",
        "## 这份报告不能说明什么",
        "",
        "- **跨协议**：目标是别人固定训练协议下的结果，早期指标是本仓库的 e0 前向；"
        "两者不是一个训练流程的产物，所以不能说「训练早期就能预测」，只能说"
        "「架构层面的早期指标与训练后结果相关」。",
        "- **单次初始化**：e0 指标只有一个随机种子，初始化噪声没有估计。",
        "- **n = 64**：Spearman 的 95% 区间约 ±0.25；置换 p 只检验「不相关」这一零假设。",
        "- **不是因果**：控制 clean 准确率只能排除一条混杂路径，参数量、可训练性等仍可能同时影响两者。",
        "- **目标不是 AutoAttack 的最强设置**：RobBench 记录的是他们的评估配置，"
        "与发布榜里的 AutoAttack 数字不可直接比较。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--robbench", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)

    metrics = load_rows(args.rows)
    robbench = load_robbench(args.robbench)
    report = analyse(metrics, robbench, resamples=args.resamples)
    report["inputs"] = {
        "rows": {"path": str(args.rows), "sha256": sha256_file(args.rows)},
        "robbench": {"path": str(args.robbench), "sha256": sha256_file(args.robbench)},
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(report), encoding="utf-8")

    counts: dict[str, int] = {}
    for row in report["metrics"]:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    best = report["metrics"][0]
    print(
        f"n={report['n_architectures']} rho(target,clean)={report['rho_target_clean']:+.3f} "
        f"best={best['metric']} partial={best['partial_clean']:+.3f} "
        f"counts={counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
