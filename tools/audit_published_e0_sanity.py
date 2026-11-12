"""Check whether the published e0 numbers could have come from the code in this repo.

The strict release stores its e0 values but not the code that produced them.  The
current code clamps every relative feature metric at 1e3 (`rel = ... .clamp_max(1e3)`
in `metrics.py`), so a published value above 1e3 cannot have been produced by it.
This tool measures, per metric, how far the published column is from the same column
recomputed with the current code, and counts rows that violate that ceiling.

It does not change anything.  It answers one question: which published e0 columns can
be quoted, and which ones need a recomputation before anyone builds on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCHEMA_VERSION = 1
SETTING = "nb201_e0"
SPIKE_CEILING = 1e3  # metrics.py clamps relative feature metrics at this value
RHO_OK = 0.8
MIN_PAIRS = 10
TARGET_COLUMN = "oracle_target"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_published(path: Path, setting: str = SETTING) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"setting", "model_or_arch_id", "metric_name", "metric_value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"published table is missing columns: {sorted(missing)}")
    frame = frame[frame["setting"] == setting]
    frame = frame.copy()
    frame["model_or_arch_id"] = frame["model_or_arch_id"].astype(str)
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    return frame.pivot_table(
        index="model_or_arch_id", columns="metric_name", values="metric_value"
    )


def load_current(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "arch_id" not in frame.columns:
        raise ValueError("current table needs an arch_id column")
    frame = frame.copy()
    frame["arch_id"] = frame["arch_id"].astype(str)
    return frame.set_index("arch_id")


def _rho(x, y):
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(joined) < MIN_PAIRS:
        return None, len(joined)
    if joined["x"].nunique() < 2 or joined["y"].nunique() < 2:
        return None, len(joined)
    return float(spearmanr(joined["x"], joined["y"])[0]), len(joined)


def compare_column(published: pd.Series, current: pd.Series) -> dict:
    published = pd.to_numeric(published, errors="coerce")
    current = pd.to_numeric(current, errors="coerce")
    rho, n = _rho(published, current)
    spikes = int((published.abs() > SPIKE_CEILING).sum())
    pub_median = float(published.median()) if published.notna().any() else None
    cur_median = float(current.median()) if current.notna().any() else None
    ratio = None
    if pub_median is not None and cur_median not in (None, 0):
        ratio = float(pub_median / cur_median)
    return {
        "n": n,
        "rho": None if rho is None else round(rho, 4),
        "spikes": spikes,
        "published_max_abs": None if published.notna().sum() == 0 else float(published.abs().max()),
        "current_max_abs": None if current.notna().sum() == 0 else float(current.abs().max()),
        "median_ratio": None if ratio is None else round(ratio, 4),
    }


def classify(entry: dict) -> str:
    if entry["rho"] is None:
        return "uncomparable"
    if entry["spikes"]:
        return "spiked"
    if entry["rho"] < RHO_OK:
        return "divergent"
    return "clean"


def analyse(published: pd.DataFrame, current: pd.DataFrame) -> dict:
    rows = []
    for metric in sorted(published.columns):
        if metric not in current.columns:
            continue
        entry = compare_column(published[metric], current[metric])
        entry["metric"] = metric
        entry["verdict"] = classify(entry)
        rows.append(entry)
    rows.sort(key=lambda r: (-r["spikes"], r["rho"] if r["rho"] is not None else 2.0))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "setting": SETTING,
        "metrics": len(rows),
        "spike_ceiling": SPIKE_CEILING,
        "rho_ok": RHO_OK,
        "counts": counts,
        "columns": rows,
    }


def _fmt(value, spec="%+.3f") -> str:
    if value is None:
        return "—"
    return spec % value


def render_markdown(report: dict, published_path: Path, current_path: Path, sha_pub: str, sha_cur: str) -> str:
    counts = report["counts"]
    lines = [
        f"# 发布表的 {report['setting']} 能不能直接引用（2026-11-12）",
        "",
        f"发布表：`{published_path}`（sha256 `{sha_pub}`）",
        f"当前代码重算：`{current_path}`（sha256 `{sha_cur}`）",
        f"共比较 {report['metrics']} 个指标；判定阈值：相对特征指标的代码上限 {report['spike_ceiling']:.0f}、秩相关 {report['rho_ok']}",
        "",
        "## 结论",
        "",
    ]
    clean = counts.get("clean", 0)
    spiked = counts.get("spiked", 0)
    divergent = counts.get("divergent", 0)
    uncomparable = counts.get("uncomparable", 0)
    lines.append(
        f"- **{clean} 个指标可以引用**：没有超过 {report['spike_ceiling']:.0f} 的值，且与当前代码秩相关 ≥ {report['rho_ok']}。"
    )
    if spiked:
        lines.append(
            f"- **{spiked} 个指标出现过超过 {report['spike_ceiling']:.0f} 的值**：当前代码在 `metrics.py` 里对相对类指标做了 "
            "`clamp_max(1e3)`，所以这些值**不可能**由本仓库的代码产出。它们的分母接近 0，是数值伪影，不是真实度量。"
        )
    if divergent:
        lines.append(
            f"- **{divergent} 个指标没有极端值，但与当前代码秩相关 < {report['rho_ok']}**：同一名字、不同的量。"
        )
    if uncomparable:
        lines.append(f"- {uncomparable} 个指标无法比较（共同架构数 < {MIN_PAIRS}，或某一侧方差为 0）。")
    lines += [
        "",
        "## 怎么读",
        "",
        "- 「超过 1e3」这条判据来自代码本身，不是我们挑的阈值：`metrics.py` 里 "
        "`rel = ((am - cm).abs() / cm.abs().clamp_min(1e-6)).clamp_max(1e3)`，输出上界就是 1e3。",
        "- 伪影行会**主导 Spearman**：秩相关只看大小顺序，一个 1e8 的值会把它所在的架构顶到最前或最后，"
        "所以受影响的不是「数值不好看」，而是**排序结论**。",
        "- 这不说明发布表整体是错的：它是那次流水线的真实输出。它说明的是**那一版代码与本仓库不同**，"
        "因此在引用发布表 e0 得出的任何幅度之前，必须先看这一列判成什么。",
        "",
        "## 需要人看的列（前 15）",
        "",
        "| 指标 | 判定 | 秩相关 | 超过 1e3 的行 | 发布最大绝对值 | 当前最大绝对值 | 中位数比 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    shown = [r for r in report["columns"] if r["verdict"] in ("spiked", "divergent", "uncomparable")][:15]
    if not shown:
        lines.append("| （无） | — | — | — | — | — | — |")
    for row in shown:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["metric"],
                row["verdict"],
                _fmt(row["rho"]),
                row["spikes"],
                _fmt(row["published_max_abs"], "%.4g"),
                _fmt(row["current_max_abs"], "%.4g"),
                _fmt(row["median_ratio"], "%.3g"),
            )
        )
    lines += [
        "",
        "## 复现",
        "",
        "```powershell",
        "python tools/audit_published_e0_sanity.py `",
        "  --published <发布表.csv> `",
        "  --current <当前代码 e0 表.csv> `",
        "  --json results\\audits\\published-e0-20261112.json `",
        "  --report results\\audits\\published-e0-20261112.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    published = load_published(args.published)
    current = load_current(args.current)
    report = analyse(published, current)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(
        render_markdown(
            report,
            args.published,
            args.current,
            sha256_file(args.published),
            sha256_file(args.current),
        ),
        encoding="utf-8",
    )
    print(
        "metrics={metrics} counts={counts}".format(
            metrics=report["metrics"], counts=report["counts"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
