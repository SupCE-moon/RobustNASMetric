import pytest

from robust_metric_lab.student_benchmark import (
    BENCHMARK_SETTINGS,
    MetricValueRow,
    all_setting_mean,
    validate_complete_setting_grid,
)


def test_frozen_benchmark_has_exactly_eight_settings():
    assert [s.key for s in BENCHMARK_SETTINGS] == [
        "rb40_mature",
        "nb201_e0",
        "nb201_e5",
        "nb201_e10",
        "nb201_e25",
        "nb201_e50",
        "darts_e0",
        "darts_e100",
    ]


def test_frozen_benchmark_includes_exact_metadata():
    assert [
        (s.benchmark, s.checkpoint_epoch, s.target_column, s.expected_rows)
        for s in BENCHMARK_SETTINGS
    ] == [
        ("rb40", None, "robust_accuracy", 40),
        ("nb201", 0, "official_final_robust_accuracy", 64),
        ("nb201", 5, "official_final_robust_accuracy", 64),
        ("nb201", 10, "official_final_robust_accuracy", 64),
        ("nb201", 25, "official_final_robust_accuracy", 64),
        ("nb201", 50, "official_final_robust_accuracy", 64),
        ("darts", 0, "full_pgd20_accuracy", 100),
        ("darts", 100, "full_pgd20_accuracy", 100),
    ]


def _valid_metric_row(**overrides):
    values = dict(
        setting="nb201_e5",
        model_or_arch_id="m0",
        checkpoint_epoch=5,
        metric_name="feature_delta",
        metric_value=1.0,
        target_value=0.5,
        seed=20260808,
        sample_offset=0,
        num_samples=64,
        epsilon=8 / 255,
        direction="lower",
    )
    values.update(overrides)
    return MetricValueRow(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"setting": "not_frozen"},
        {"setting": "nb201_e5", "checkpoint_epoch": 10},
        {"seed": 1},
        {"sample_offset": 1},
        {"num_samples": 32},
        {"epsilon": 4 / 255},
    ],
)
def test_metric_row_rejects_non_frozen_protocol_values(overrides):
    with pytest.raises(ValueError):
        _valid_metric_row(**overrides)


def test_all_setting_mean_requires_all_eight_values():
    values = {s.key: 0.5 for s in BENCHMARK_SETTINGS}
    assert all_setting_mean(values) == 0.5
    values.pop("darts_e100")
    try:
        all_setting_mean(values)
    except ValueError as exc:
        assert "darts_e100" in str(exc)
    else:
        raise AssertionError("missing setting must fail")


def test_setting_grid_rejects_duplicate_ids():
    rows = [
        {"setting": "rb40_mature", "model_or_arch_id": "m0"},
        {"setting": "rb40_mature", "model_or_arch_id": "m0"},
    ]
    errors = validate_complete_setting_grid(rows, {"rb40_mature": 1})
    assert any("duplicate" in error.lower() for error in errors)


def test_setting_grid_reports_missing_rows():
    errors = validate_complete_setting_grid([], {"rb40_mature": 1})
    assert any("missing" in error.lower() and "rb40_mature" in error for error in errors)


def test_setting_grid_reports_unknown_settings():
    errors = validate_complete_setting_grid(
        [{"setting": "not_frozen", "model_or_arch_id": "m0"}], {"rb40_mature": 0}
    )
    assert any("unknown" in error.lower() and "not_frozen" in error for error in errors)
