from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum
from typing import Iterable, Mapping

from .metric_registry import METRIC_BY_NAME


@dataclass(frozen=True)
class BenchmarkSetting:
    key: str
    benchmark: str
    checkpoint_epoch: int | None
    target_column: str
    expected_rows: int


BENCHMARK_SETTINGS = (
    BenchmarkSetting("rb40_mature", "rb40", None, "robust_accuracy", 40),
    BenchmarkSetting("nb201_e0", "nb201", 0, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e5", "nb201", 5, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e10", "nb201", 10, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e25", "nb201", 25, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e50", "nb201", 50, "official_final_robust_accuracy", 64),
    BenchmarkSetting("darts_e0", "darts", 0, "full_pgd20_accuracy", 100),
    BenchmarkSetting("darts_e100", "darts", 100, "full_pgd20_accuracy", 100),
)
SETTING_BY_KEY = {setting.key: setting for setting in BENCHMARK_SETTINGS}

SAMPLE_OFFSET = 0
NUM_SAMPLES = 64
SEED = 20260808
EPSILON = 8 / 255


@dataclass(frozen=True)
class MetricValueRow:
    """One metric observation under the shared student-benchmark protocol."""

    setting: str
    model_or_arch_id: str
    checkpoint_epoch: int | None
    metric_name: str
    metric_value: float | None
    target_value: float | None
    seed: int
    sample_offset: int
    num_samples: int
    epsilon: float
    direction: str
    status: str = "ok"
    error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        setting = SETTING_BY_KEY.get(self.setting)
        if setting is None:
            raise ValueError(f"unknown benchmark setting: {self.setting}")
        if self.checkpoint_epoch != setting.checkpoint_epoch:
            raise ValueError(
                f"checkpoint epoch mismatch for {self.setting}: "
                f"expected {setting.checkpoint_epoch}, got {self.checkpoint_epoch}"
            )
        protocol_values = (
            ("seed", self.seed, SEED),
            ("sample_offset", self.sample_offset, SAMPLE_OFFSET),
            ("num_samples", self.num_samples, NUM_SAMPLES),
            ("epsilon", self.epsilon, EPSILON),
        )
        for name, value, expected in protocol_values:
            if value != expected:
                raise ValueError(f"{name} must be frozen at {expected!r}, got {value!r}")
        spec = METRIC_BY_NAME.get(self.metric_name)
        if spec is None:
            raise ValueError(f"unknown metric: {self.metric_name}")
        if self.direction != spec.direction:
            raise ValueError(
                f"metric direction mismatch for {self.metric_name}: "
                f"expected {spec.direction}, got {self.direction}"
            )


def validate_complete_setting_grid(
    rows: Iterable[Mapping[str, object]], expected_counts: Mapping[str, int]
) -> list[str]:
    """Return deterministic diagnostics for unknown, duplicate, and incomplete rows."""

    errors: list[str] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, object]] = set()
    known = set(expected_counts)
    for row in rows:
        setting = str(row.get("setting", ""))
        identifier = row.get("model_or_arch_id")
        if setting not in known:
            errors.append(f"unknown setting: {setting}")
            continue
        key = (setting, identifier)
        if key in seen:
            errors.append(
                f"duplicate row: setting={setting}, model_or_arch_id={identifier}"
            )
        seen.add(key)
        counts[setting] = counts.get(setting, 0) + 1
    for setting, expected in expected_counts.items():
        actual = counts.get(setting, 0)
        if actual < expected:
            errors.append(
                f"missing rows for setting {setting}: expected {expected}, found {actual}"
            )
    return errors


def all_setting_mean(values: Mapping[str, float]) -> float:
    keys = [setting.key for setting in BENCHMARK_SETTINGS]
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"missing settings: {', '.join(missing)}")
    return fsum(float(values[key]) for key in keys) / len(keys)
