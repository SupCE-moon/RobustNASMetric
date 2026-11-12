"""Recompute the frozen eight-setting strict ranking directly from raw metric rows.

Why this exists
---------------
The authoritative leaderboard lives only on the server
(`outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z/`).
When the server is unreachable, the only way to tell whether a quoted rank is real
is to rebuild it from the raw per-model rows and compare.

This module deliberately does **not** import `robust_metric_lab`. It re-implements
the frozen protocol in ~40 lines of pandas/scipy so that a coding bug in the
pipeline cannot hide behind itself:

    per metric x setting -> Spearman(metric_value, target_value)
    direction == "lower" -> negate the correlation
    all_setting_mean     -> plain mean over the eight settings
    rank                 -> all_setting_mean descending, metric_name ascending

Protocol constants are copied from the project docs, not imported:
`docs/experiments/实验管理.md` and `src/robust_metric_lab/student_analysis.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

SETTING_KEYS: tuple[str, ...] = (
    "rb40_mature",
    "nb201_e0",
    "nb201_e5",
    "nb201_e10",
    "nb201_e25",
    "nb201_e50",
    "darts_e0",
    "darts_e100",
)
REPAIR_KEY = ("metric_name", "setting", "model_or_arch_id")
# Frozen grid sizes: 40 RB40 models, 64 NB201 architectures x 5 epochs, 100 DARTS architectures x 2 epochs.
# A metric/setting with fewer `ok` rows than this is not silently scored here: the frozen
# protocol would reject it, so we reject it too.
EXPECTED_ROWS: dict[str, int] = {
    "rb40_mature": 40,
    "nb201_e0": 64,
    "nb201_e5": 64,
    "nb201_e10": 64,
    "nb201_e25": 64,
    "nb201_e50": 64,
    "darts_e0": 100,
    "darts_e100": 100,
}
TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(rows_path: Path, repair_path: Path | None) -> pd.DataFrame:
    """Load raw rows and apply the TRNAS repair rows, if any."""
    frame = pd.read_csv(rows_path, keep_default_na=False, na_values=[""])
    if repair_path is None:
        return frame
    repair = pd.read_csv(repair_path, keep_default_na=False, na_values=[""])
    if repair.empty:
        return frame
    index = list(REPAIR_KEY)
    replaced = frame.set_index(index).drop(index=repair.set_index(index).index, errors="ignore")
    merged = pd.concat([replaced, repair.set_index(index)])
    return merged.reset_index()


def oriented_spearman(values: pd.Series, targets: pd.Series, direction: str) -> float | None:
    """Spearman correlation with the metric's declared direction applied."""
    mask = values.notna() & targets.notna()
    x = values[mask].to_numpy(dtype=float)
    y = targets[mask].to_numpy(dtype=float)
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho = float(stats.spearmanr(x, y).statistic)
    if rho != rho:  # NaN
        return None
    return -rho if str(direction).strip().lower() == "lower" else rho


def recompute(
    frame: pd.DataFrame,
    expected_rows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return one row per metric with the eight settings and their mean.

    ``expected_rows`` defaults to the frozen grid sizes; tests pass a smaller map
    so they can use a handful of synthetic rows per setting.
    """
    expected_map = EXPECTED_ROWS if expected_rows is None else expected_rows
    direction = (
        frame.groupby("metric_name")["direction"].agg(lambda s: str(s.iloc[0]).strip().lower())
    )
    records: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for metric_name, group in frame.groupby("metric_name"):
        row: dict[str, Any] = {"metric_name": metric_name}
        missing: list[str] = []
        for setting in SETTING_KEYS:
            subset = group[group["setting"].astype(str) == setting]
            ok = subset[subset["status"].astype(str) == "ok"]
            expected = expected_map.get(setting)
            if expected is not None and len(ok) != expected:
                missing.append(f"{setting} ({len(ok)}/{expected} ok rows)")
                row[setting] = None
                continue
            rho = oriented_spearman(
                pd.to_numeric(ok["metric_value"], errors="coerce"),
                pd.to_numeric(ok["target_value"], errors="coerce"),
                direction.get(metric_name, "lower"),
            )
            if rho is None:
                missing.append(setting)
                row[setting] = None
            else:
                row[setting] = rho
        if missing:
            rejected[metric_name] = "missing or degenerate settings: " + ", ".join(missing)
            continue
        row["all_setting_mean"] = sum(row[s] for s in SETTING_KEYS) / len(SETTING_KEYS)
        records[metric_name] = row
    table = pd.DataFrame(records.values())
    if not table.empty:
        table = table.sort_values(
            ["all_setting_mean", "metric_name"], ascending=[False, True]
        ).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
    return table, rejected


def compare(mine: pd.DataFrame, official: pd.DataFrame) -> dict[str, Any]:
    """Compare recomputed values against an official ranking table."""
    official = official.set_index("metric_name")
    common = [m for m in mine["metric_name"] if m in official.index]
    mine_idx = mine.set_index("metric_name")
    diffs: dict[str, float] = {}
    for metric in common:
        for column in (*SETTING_KEYS, "all_setting_mean"):
            if column not in official.columns:
                continue
            expected = float(official.loc[metric, column])
            got = float(mine_idx.loc[metric, column])
            diffs[f"{metric}::{column}"] = abs(got - expected)
    worst = max(diffs.items(), key=lambda kv: kv[1]) if diffs else (None, 0.0)

    # Rank order must be compared inside the shared subset: the official table may
    # contain extra metrics (e.g. one added after these rows were frozen).
    official_order = [m for m in official.loc[common].sort_values("all_setting_mean", ascending=False).index]
    mine_order = [m for m in mine.loc[mine["metric_name"].isin(common), "metric_name"]]
    return {
        "metrics_compared": len(common),
        "metrics_official_only": sorted(set(official.index) - set(mine["metric_name"])),
        "metrics_recomputed_only": sorted(set(mine["metric_name"]) - set(official.index)),
        "max_abs_error": round(worst[1], 12),
        "max_abs_error_at": worst[0] if worst[1] > 0 else "(all compared cells identical)",
        "exact_match": all(v <= TOLERANCE for v in diffs.values()),
        "rank_order_identical": official_order == mine_order,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, type=Path, help="merged raw metric rows CSV")
    parser.add_argument("--repair", type=Path, default=None, help="optional repair rows CSV")
    parser.add_argument("--ranking", required=True, type=Path, help="official strict_ranking.csv")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--report", type=Path, default=None, help="optional Markdown summary")
    args = parser.parse_args(argv)

    frame = load_rows(args.rows, args.repair)
    mine, rejected = recompute(frame)
    official = pd.read_csv(args.ranking)
    result = compare(mine, official)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "rows": {"path": str(args.rows), "sha256": sha256_file(args.rows), "data_rows": len(frame)},
            "repair": (
                {"path": str(args.repair), "sha256": sha256_file(args.repair)}
                if args.repair
                else None
            ),
            "ranking": {"path": str(args.ranking), "sha256": sha256_file(args.ranking)},
        },
        "protocol": {
            "settings": list(SETTING_KEYS),
            "direction": "negate Spearman when direction == lower",
            "aggregate": "mean over the eight settings",
            "rank": "all_setting_mean desc, metric_name asc",
        },
        "recomputed_metrics": len(mine),
        "rejected_metrics": rejected,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"recomputed={len(mine)} compared={result['metrics_compared']} "
        f"max_abs_error={result['max_abs_error']:.3e} exact={result['exact_match']} "
        f"order_identical={result['rank_order_identical']}"
    )
    if rejected:
        print("rejected:", json.dumps(rejected, ensure_ascii=False))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 严格排名离线复算（从原始行重建）",
            "",
            f"生成时间（UTC）：{payload['generated_at_utc']}",
            "",
            f"- 原始行：`{args.rows}`（{len(frame):,} 行，sha256 `{payload['inputs']['rows']['sha256'][:16]}…`）",
            f"- 官方排名表：`{args.ranking}`",
            f"- 复算成功指标：**{len(mine)}**；被排除：{len(rejected)}",
            f"- 与官方逐值比较：{result['metrics_compared']} 项，最大绝对误差 `{result['max_abs_error']:.3e}`"
            f"（{result['max_abs_error_at']}）",
            f"- 逐值完全一致（≤{TOLERANCE:g}）：**{result['exact_match']}**",
            f"- 共享子集内名次顺序一致：**{result['rank_order_identical']}**",
            "",
        ]
        if result["metrics_official_only"]:
            lines += [
                "## 官方表里有、这批原始行里没有的指标",
                "",
                *[f"- `{m}`" for m in result["metrics_official_only"]],
                "",
                "这些指标是在这批原始行冻结之后才加入的，其原始行只存在于服务器侧，离线无法复算。",
                "",
            ]
        if rejected:
            lines += ["## 被排除的指标", "", *[f"- `{k}`：{v}" for k, v in rejected.items()], ""]
        args.report.write_text("\n".join(lines), encoding="utf-8")
    return 0 if result["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
