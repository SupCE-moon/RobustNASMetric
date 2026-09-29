from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Sequence

from .metric_registry import METRIC_SPECS
from .student_benchmark import BENCHMARK_SETTINGS


class MetricEligibility(str, Enum):
    CROSS_SPACE = "cross_space"
    SPACE_SPECIFIC = "space_specific"
    IMPLEMENTATION_ERROR = "implementation_error"


@dataclass(frozen=True)
class MetricEligibilityRecord:
    metric_name: str
    eligibility: MetricEligibility
    supported_settings: tuple[str, ...]
    reason: str


SETTING_NAMES = tuple(setting.key for setting in BENCHMARK_SETTINGS)
SETTING_NAME_SET = frozenset(SETTING_NAMES)


def _record_from_json(value: object, index: int) -> MetricEligibilityRecord:
    if not isinstance(value, dict):
        raise ValueError(f"eligibility record {index} must be an object")
    expected = {"metric_name", "eligibility", "supported_settings", "reason"}
    extra = set(value) - expected
    missing = expected - set(value)
    if missing or extra:
        raise ValueError(
            f"eligibility record {index} schema mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    name = value["metric_name"]
    reason = value["reason"]
    settings = value["supported_settings"]
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError(f"eligibility record {index} has invalid metric_name")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"eligibility record {index} has empty reason")
    if not isinstance(settings, list) or not all(isinstance(item, str) for item in settings):
        raise ValueError(f"eligibility record {index} supported_settings must be a string list")
    if len(settings) != len(set(settings)):
        raise ValueError(f"eligibility record {index} has duplicate supported setting")
    unknown = set(settings) - SETTING_NAME_SET
    if unknown:
        raise ValueError(f"eligibility record {index} has unknown setting: {sorted(unknown)}")
    try:
        eligibility = MetricEligibility(value["eligibility"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"eligibility record {index} has invalid eligibility") from exc
    if eligibility is MetricEligibility.CROSS_SPACE and tuple(settings) != SETTING_NAMES:
        raise ValueError(f"cross_space metric {name} must support all eight frozen settings in order")
    return MetricEligibilityRecord(name, eligibility, tuple(settings), reason.strip())


def load_metric_eligibility(path: str | Path) -> tuple[MetricEligibilityRecord, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("metric eligibility root must be a list")
    records = tuple(_record_from_json(value, index) for index, value in enumerate(raw))
    names = [record.metric_name for record in records]
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise ValueError(f"duplicate metric eligibility records: {duplicate}")
    return records


def validate_registry(
    records: Sequence[MetricEligibilityRecord],
    *,
    registered_names: Iterable[str] | None = None,
) -> None:
    expected = list(registered_names if registered_names is not None else (spec.name for spec in METRIC_SPECS))
    if len(expected) != len(set(expected)):
        raise ValueError("registered metric names contain duplicates")
    actual = [record.metric_name for record in records]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate metric eligibility records")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"missing registered metrics: {missing}")
    if extra:
        raise ValueError(f"unregistered metrics in eligibility registry: {extra}")


def eligible_cross_space_names(
    records: Sequence[MetricEligibilityRecord],
) -> frozenset[str]:
    validate_registry(records)
    return frozenset(
        record.metric_name
        for record in records
        if record.eligibility is MetricEligibility.CROSS_SPACE
    )
