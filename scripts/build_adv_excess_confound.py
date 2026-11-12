"""Recompute frozen architecture controls for the one new metric."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


NB_OPERATIONS = (
    "none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3",
    "avg_pool_3x3",
)
COUNTS = {"rb40_mature": 40, **{f"nb201_e{e}": 64 for e in (0, 5, 10, 25, 50)},
          **{f"darts_e{e}": 100 for e in (0, 100)}}
METRIC = "feature_adv_excess_centered_unit_l2"


def _log_parameters(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["parameter_count"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any():
        raise ValueError("parameter counts must be positive and finite")
    return np.log(values.astype(float))


def _check_ids(frame: pd.DataFrame) -> None:
    identifiers = frame["model_or_arch_id"]
    if identifiers.isna().any() or identifiers.astype(str).duplicated().any():
        raise ValueError("architecture controls have missing or duplicate IDs")


def controls_from_manifests(
    rb40: pd.DataFrame, nb201: pd.DataFrame, darts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Keep exactly the original preregistered covariates in all eight settings."""
    rb = pd.DataFrame({
        "model_or_arch_id": rb40["model_name"].astype(str),
        "log_parameter_count": _log_parameters(rb40),
        "model_family": rb40["architecture_family"].astype(str),
    })
    if rb["model_family"].isna().any() or (rb["model_family"] == "").any():
        raise ValueError("missing RB40 model family")
    nb = pd.DataFrame({"model_or_arch_id": nb201["arch_id"].astype(str)})
    for operation in NB_OPERATIONS:
        column = f"op_{operation}"
        values = pd.to_numeric(nb201[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"nonfinite NB201 control {column}")
        nb[f"op_count_{operation}"] = values
    da = pd.DataFrame({
        "model_or_arch_id": darts["arch_id"].astype(str),
        "log_parameter_count": _log_parameters(darts),
    })
    operation_columns = [column for column in darts if column.startswith("op_")]
    if not operation_columns:
        raise ValueError("DARTS manifest has no operation counts")
    for column in operation_columns:
        values = pd.to_numeric(darts[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"nonfinite DARTS control {column}")
        da[f"op_count_{column[3:]}"] = values
    for frame in (rb, nb, da):
        _check_ids(frame)
    return {
        "rb40_mature": rb,
        **{f"nb201_e{epoch}": nb.copy() for epoch in (0, 5, 10, 25, 50)},
        **{f"darts_e{epoch}": da.copy() for epoch in (0, 100)},
    }


def build_confound(
    release: Path, historical_release: Path, bundle: Path, output: Path,
) -> pd.DataFrame:
    """Append eight new descriptive rows to the untouched 680-row old audit."""
    if output.exists():
        raise FileExistsError(output)
    verification = json.loads((release / "release_verification.json").read_text())
    required = {
        "status": "PASS", "raw_rows": 48160, "unique_raw_keys": 48160,
        "ranked_metrics": 84, "excluded_metrics": 2,
    }
    if any(verification.get(key) != value for key, value in required.items()):
        raise ValueError("strict release must PASS before structure analysis")
    manifests = bundle / "manifests"
    controls = controls_from_manifests(
        pd.read_csv(manifests / "rb40_models.csv"),
        pd.read_csv(manifests / "nb201_64_architectures.csv"),
        pd.read_csv(manifests / "darts100_architectures.csv"),
    )
    for setting, expected in COUNTS.items():
        if len(controls[setting]) != expected:
            raise ValueError(f"incomplete controls: {setting}")
    from robust_metric_lab.student_confound import confound_diagnostics

    raw = pd.read_csv(release / "metric_values_merged.csv")
    new_rows = raw[raw["metric_name"] == METRIC].copy()
    if len(new_rows) != 560 or not new_rows["status"].eq("ok").all():
        raise ValueError("new metric is not a complete successful 560-row grid")
    new_result = confound_diagnostics(new_rows, controls)
    if (len(new_result) != 8
            or set(new_result["setting"]) != set(COUNTS)
            or any(int(row["count"]) != COUNTS[row["setting"]]
                   for _, row in new_result.iterrows())):
        raise ValueError("structure diagnostics did not cover the full grid")
    old_path = historical_release / "ranking/confound_diagnostics.csv"
    old = pd.read_csv(old_path)
    if (len(old) != 680
            or len(old[["metric_name", "setting"]].drop_duplicates()) != 680
            or METRIC in set(old["metric_name"])):
        raise ValueError("historical 680 structure rows changed")
    combined = pd.concat([old, new_result], ignore_index=True)
    if len(combined) != 688:
        raise ValueError("combined structure rows are not 688")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        combined.to_csv(handle, index=False)
    return new_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--historical-release", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_confound(
        args.release, args.historical_release, args.bundle, args.output
    )
    print(f"new_structure_rows={len(result)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
