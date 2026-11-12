from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.metric_eligibility import MetricEligibility, MetricEligibilityRecord
from robust_metric_lab.strict_ranking import (
    RAW_KEY,
    ManifestExpectation,
    build_strict_ranking,
    load_unique_metric_rows,
    validate_metric_grid,
)
from robust_metric_lab.student_benchmark import BENCHMARK_SETTINGS, EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED


METRIC = "feature_delta"


def _manifests() -> dict[str, ManifestExpectation]:
    result = {}
    for setting in BENCHMARK_SETTINGS:
        ids = tuple(f"{setting.key}_id_{i:03d}" for i in range(setting.expected_rows))
        targets = {identifier: float(i) / setting.expected_rows for i, identifier in enumerate(ids)}
        result[setting.key] = ManifestExpectation(
            benchmark=setting.benchmark,
            checkpoint_epoch=setting.checkpoint_epoch,
            targets=targets,
        )
    return result


def _frame() -> pd.DataFrame:
    rows = []
    for setting in BENCHMARK_SETTINGS:
        for i in range(setting.expected_rows):
            identifier = f"{setting.key}_id_{i:03d}"
            rows.append(
                {
                    "run_id": "test",
                    "benchmark": setting.benchmark,
                    "setting": setting.key,
                    "model_or_arch_id": identifier,
                    "checkpoint_epoch": setting.checkpoint_epoch,
                    "metric_name": METRIC,
                    "direction": METRIC_BY_NAME[METRIC].direction,
                    "metric_value": float(i + 1),
                    "target_value": float(i) / setting.expected_rows,
                    "status": "ok",
                    "error": "",
                    "runtime_sec": 0.1,
                    "seed": SEED,
                    "sample_offset": SAMPLE_OFFSET,
                    "num_samples": NUM_SAMPLES,
                    "epsilon": EPSILON,
                    "checkpoint_path": f"artifacts/{identifier}.pt",
                }
            )
    return pd.DataFrame(rows)


def test_complete_grid_has_exactly_560_rows() -> None:
    validated = validate_metric_grid(_frame(), METRIC, _manifests())
    assert len(validated) == 560
    assert not validated.duplicated(list(RAW_KEY)).any()


def test_29_of_40_rb40_grid_is_rejected() -> None:
    frame = _frame()
    drop = frame[(frame.setting == "rb40_mature")].index[-1]
    with pytest.raises(ValueError, match="rb40_mature.*expected 40.*found 39"):
        validate_metric_grid(frame.drop(drop), METRIC, _manifests())


def test_duplicate_raw_key_aborts_before_ranking(tmp_path: Path) -> None:
    frame = _frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "raw.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate raw key"):
        load_unique_metric_rows([path])


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("metric_value", np.inf, "nonfinite metric_value"),
        ("seed", SEED + 1, "seed"),
        ("num_samples", NUM_SAMPLES - 1, "num_samples"),
        ("epsilon", 0.1, "epsilon"),
    ],
)
def test_grid_rejects_nonfinite_or_protocol_mismatch(column, value, match: str) -> None:
    frame = _frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_metric_grid(frame, METRIC, _manifests())


def test_grid_rejects_manifest_id_mismatch() -> None:
    frame = _frame()
    frame.loc[0, "model_or_arch_id"] = "unknown_rb40_model"
    with pytest.raises(ValueError, match="manifest ID mismatch"):
        validate_metric_grid(frame, METRIC, _manifests())


def test_grid_rejects_manifest_target_mismatch() -> None:
    frame = _frame()
    frame.loc[0, "target_value"] += 0.25
    with pytest.raises(ValueError, match="target mismatch"):
        validate_metric_grid(frame, METRIC, _manifests())


def test_ranking_uses_only_complete_eligible_raw_grids() -> None:
    eligibility = (
        MetricEligibilityRecord(
            METRIC,
            MetricEligibility.CROSS_SPACE,
            tuple(setting.key for setting in BENCHMARK_SETTINGS),
            "complete test metric",
        ),
        MetricEligibilityRecord(
            "feature_measure_moment_delta",
            MetricEligibility.CROSS_SPACE,
            tuple(setting.key for setting in BENCHMARK_SETTINGS),
            "missing test metric",
        ),
        MetricEligibilityRecord(
            "racl_arch_lipschitz_log_upper",
            MetricEligibility.SPACE_SPECIFIC,
            tuple(setting.key for setting in BENCHMARK_SETTINGS if setting.benchmark != "rb40"),
            "space-specific test metric",
        ),
    )
    ranking, excluded, analysis = build_strict_ranking(_frame(), _manifests(), eligibility)
    assert ranking.metric_name.tolist() == [METRIC]
    assert ranking["rank"].tolist() == [1]
    assert np.isfinite(ranking["all_setting_mean"]).all()
    excluded_by_name = excluded.set_index("metric_name")
    assert excluded_by_name.loc["feature_measure_moment_delta", "eligibility"] == "incomplete"
    assert excluded_by_name.loc["racl_arch_lipschitz_log_upper", "eligibility"] == "space_specific"
    assert len(analysis["correlations"]) == 8
