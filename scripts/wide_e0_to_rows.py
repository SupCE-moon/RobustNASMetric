"""Convert the wide e0 table into the long row format the audits read.

The published `metric_values_original_strict.csv` stores one row per
(metric, setting, architecture).  The e0 expansion
(`scripts/expand_nb201_e0_coverage.py`) naturally produces the transposed shape:
one row per architecture, one column per metric.  Rather than teaching every
audit a second input shape, this script performs the one conversion.

Directions come from `metric_registry.METRIC_SPECS` and are not inferred from the
data: `direction` decides whether a metric is negated before correlating, so a
guess here would silently flip a conclusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robust_metric_lab.metric_registry import METRIC_SPECS

SETTING = "nb201_e0"
TARGET_COLUMNS = ("target", "oracle_target")


def directions() -> dict[str, str]:
    return {spec.name: spec.direction for spec in METRIC_SPECS}


def to_long(wide: pd.DataFrame, direction_map: dict[str, str]) -> pd.DataFrame:
    """One row per (metric, architecture), keeping only registered metrics."""
    if "arch_id" not in wide.columns:
        raise ValueError("wide table needs an arch_id column")
    target_column = next((c for c in TARGET_COLUMNS if c in wide.columns), None)
    if target_column is None:
        raise ValueError("wide table needs a target column")

    metrics = [c for c in wide.columns if c in direction_map]
    if not metrics:
        raise ValueError("no column of the wide table matches a registered metric")

    ids = wide["arch_id"].astype(str)
    target = pd.to_numeric(wide[target_column], errors="coerce")
    frames = []
    for name in metrics:
        values = pd.to_numeric(wide[name], errors="coerce")
        frames.append(
            pd.DataFrame(
                {
                    "setting": SETTING,
                    "model_or_arch_id": ids,
                    "metric_name": name,
                    "metric_value": values,
                    # A non-finite value means the metric could not be computed
                    # for that architecture; it must be recorded as failed rather
                    # than dropped, so the coverage is visible downstream.
                    "status": ["ok" if pd.notna(v) else "failed" for v in values],
                    "direction": direction_map[name],
                    "target_value": target,
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["metric_name", "model_or_arch_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wide = pd.read_csv(args.wide)
    long = to_long(wide, directions())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output, index=False)

    per_metric = long.groupby("metric_name")["status"].apply(lambda s: (s == "ok").sum())
    print(
        f"architectures={wide['arch_id'].nunique()} metrics={long['metric_name'].nunique()} "
        f"rows={len(long)} min_ok_coverage={int(per_metric.min())}"
    )
    weak = per_metric[per_metric < wide["arch_id"].nunique()]
    if len(weak):
        print(f"metrics with incomplete coverage: {len(weak)} (worst {int(weak.min())})")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
