from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_metric_lab.student_confound import partial_spearman, validate_frozen_control_schema


def test_partial_spearman_removes_parameter_count_confound() -> None:
    log_params = np.linspace(1.0, 8.0, 80)
    generator = np.random.default_rng(11)
    metric = log_params + 0.2 * generator.normal(size=80)
    target = log_params + 0.2 * generator.normal(size=80)
    assert pd.Series(metric).corr(pd.Series(target), method="spearman") > 0.9
    result = partial_spearman(metric, target, pd.DataFrame({"log_params": log_params}))
    assert result.status == "ok"
    assert abs(result.value) < 0.1


def test_partial_spearman_encodes_categorical_family() -> None:
    family = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    family_effect = np.repeat([0.0, 10.0, 20.0], 20)
    jitter = np.tile(np.linspace(-0.2, 0.2, 20), 3)
    result = partial_spearman(
        family_effect + jitter,
        family_effect - jitter,
        pd.DataFrame({"family": family}),
    )
    assert result.status == "ok"
    assert result.controls == ("family",)
    assert result.value < -0.9


def test_partial_spearman_handles_singular_controls() -> None:
    x = np.arange(10, dtype=float)
    controls = pd.DataFrame({"one": x, "duplicate": x})
    result = partial_spearman(x + 2.0 * np.sin(x), x + 2.0 * np.cos(x), controls)
    assert result.status == "ok"
    assert np.isfinite(result.value)


@pytest.mark.parametrize(
    ("metric", "target", "controls", "reason"),
    [
        ([1, 2, 3], [1, 2, 3], pd.DataFrame({"x": [0, 1, 2]}), "fewer than four"),
        ([1, 1, 1, 1], [1, 2, 3, 4], pd.DataFrame({"x": [0, 0, 0, 0]}), "constant residual"),
    ],
)
def test_partial_spearman_reports_invalid_reason(metric, target, controls, reason) -> None:
    result = partial_spearman(metric, target, controls)
    assert result.status == "invalid"
    assert result.value is None
    assert reason in result.reason


def test_frozen_control_schema_rejects_wrong_benchmark_controls() -> None:
    with pytest.raises(ValueError, match="RB40 controls"):
        validate_frozen_control_schema(
            "rb40_mature", pd.DataFrame(columns=["model_or_arch_id", "log_parameter_count"])
        )
    with pytest.raises(ValueError, match="NB201 controls"):
        validate_frozen_control_schema(
            "nb201_e25", pd.DataFrame(columns=["model_or_arch_id", "op_count_skip_connect"])
        )
    with pytest.raises(ValueError, match="DARTS controls"):
        validate_frozen_control_schema(
            "darts_e100", pd.DataFrame(columns=["model_or_arch_id", "log_parameter_count"])
        )
