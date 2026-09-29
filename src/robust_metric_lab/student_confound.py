from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metric_registry import METRIC_BY_NAME


@dataclass(frozen=True)
class PartialSpearmanResult:
    value: float | None
    status: str
    reason: str
    count: int
    controls: tuple[str, ...]


NB201_CONTROL_COLUMNS = {
    "model_or_arch_id",
    "op_count_none",
    "op_count_skip_connect",
    "op_count_nor_conv_1x1",
    "op_count_nor_conv_3x3",
    "op_count_avg_pool_3x3",
}


def validate_frozen_control_schema(setting: str, controls: pd.DataFrame) -> None:
    columns = set(controls.columns)
    if setting == "rb40_mature":
        expected = {"model_or_arch_id", "log_parameter_count", "model_family"}
        if columns != expected:
            raise ValueError(f"RB40 controls must be exactly {sorted(expected)}")
        return
    if setting.startswith("nb201_e"):
        if columns != NB201_CONTROL_COLUMNS:
            raise ValueError(f"NB201 controls must be exactly {sorted(NB201_CONTROL_COLUMNS)}")
        return
    if setting.startswith("darts_e"):
        required = {"model_or_arch_id", "log_parameter_count"}
        operation_columns = {column for column in columns if column.startswith("op_count_")}
        if not required <= columns or not operation_columns or columns != required | operation_columns:
            raise ValueError("DARTS controls require log_parameter_count and manifest op_count_* columns")
        return
    raise ValueError(f"unknown benchmark setting for controls: {setting}")


def partial_spearman(
    metric: Sequence[float],
    target: Sequence[float],
    controls: pd.DataFrame,
) -> PartialSpearmanResult:
    metric_values = np.asarray(metric, dtype=float)
    target_values = np.asarray(target, dtype=float)
    count = len(metric_values)
    names = tuple(str(column) for column in controls.columns)
    if count != len(target_values) or count != len(controls):
        raise ValueError("metric, target and controls must have equal length")
    if count < 4:
        return PartialSpearmanResult(None, "invalid", "fewer than four objects", count, names)
    if not np.isfinite(metric_values).all() or not np.isfinite(target_values).all():
        return PartialSpearmanResult(None, "invalid", "nonfinite metric or target", count, names)

    encoded = pd.get_dummies(controls.reset_index(drop=True), drop_first=False, dtype=float)
    if encoded.shape[1] and not np.isfinite(encoded.to_numpy(dtype=float)).all():
        return PartialSpearmanResult(None, "invalid", "nonfinite control", count, names)
    design = np.column_stack((np.ones(count), encoded.to_numpy(dtype=float)))
    metric_rank = stats.rankdata(metric_values, method="average")
    target_rank = stats.rankdata(target_values, method="average")
    metric_fit = design @ np.linalg.lstsq(design, metric_rank, rcond=None)[0]
    target_fit = design @ np.linalg.lstsq(design, target_rank, rcond=None)[0]
    metric_residual = metric_rank - metric_fit
    target_residual = target_rank - target_fit
    if np.allclose(metric_residual, metric_residual[0]) or np.allclose(target_residual, target_residual[0]):
        return PartialSpearmanResult(None, "invalid", "constant residual", count, names)
    value = float(stats.pearsonr(metric_residual, target_residual).statistic)
    if not math.isfinite(value):
        return PartialSpearmanResult(None, "invalid", "undefined residual correlation", count, names)
    return PartialSpearmanResult(value, "ok", "", count, names)


def confound_diagnostics(
    frame: pd.DataFrame,
    controls_by_setting: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    required = {"metric_name", "setting", "model_or_arch_id", "metric_value", "target_value", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metric rows missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for (metric_name, setting), group in frame.groupby(["metric_name", "setting"], sort=True):
        if metric_name not in METRIC_BY_NAME:
            raise ValueError(f"unknown metric: {metric_name}")
        if setting not in controls_by_setting:
            raise ValueError(f"missing frozen controls for {setting}")
        controls = controls_by_setting[setting]
        validate_frozen_control_schema(setting, controls)
        if "model_or_arch_id" not in controls.columns:
            raise ValueError(f"controls for {setting} lack model_or_arch_id")
        if controls["model_or_arch_id"].duplicated().any():
            raise ValueError(f"controls for {setting} contain duplicate IDs")
        selected = group[group["status"] == "ok"].copy()
        selected["metric_value"] = pd.to_numeric(selected["metric_value"], errors="coerce")
        selected["target_value"] = pd.to_numeric(selected["target_value"], errors="coerce")
        selected = selected[np.isfinite(selected["metric_value"]) & np.isfinite(selected["target_value"])]
        merged = selected.merge(controls, on="model_or_arch_id", how="inner", validate="one_to_one")
        control_columns = [column for column in controls.columns if column != "model_or_arch_id"]
        raw = float(stats.spearmanr(merged["metric_value"], merged["target_value"]).statistic) if len(merged) >= 2 else float("nan")
        partial = partial_spearman(
            merged["metric_value"].to_numpy(),
            merged["target_value"].to_numpy(),
            merged[control_columns],
        )
        direction = METRIC_BY_NAME[metric_name].direction
        rows.append(
            {
                "metric_name": metric_name,
                "setting": setting,
                "count": len(merged),
                "controls": ";".join(control_columns),
                "raw_spearman": raw if math.isfinite(raw) else None,
                "direction_adjusted_raw_spearman": (-raw if direction == "lower" else raw) if math.isfinite(raw) else None,
                "partial_spearman": partial.value,
                "direction_adjusted_partial_spearman": (
                    -partial.value if partial.value is not None and direction == "lower" else partial.value
                ),
                "status": partial.status,
                "reason": partial.reason,
            }
        )
    return pd.DataFrame(rows)
