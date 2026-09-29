from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_metric_lab.metric_eligibility import (
    MetricEligibility,
    eligible_cross_space_names,
    load_metric_eligibility,
    validate_registry,
)
from robust_metric_lab.metric_registry import METRIC_SPECS


CONFIG = Path("configs/metric_eligibility.json")


def _payload() -> list[dict[str, object]]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_registry_classifies_every_metric_exactly_once() -> None:
    records = load_metric_eligibility(CONFIG)
    registered = [spec.name for spec in METRIC_SPECS]
    classified = [record.metric_name for record in records]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(registered)
    validate_registry(records, registered_names=registered)


def test_space_specific_racl_proxies_never_enter_cross_space_list() -> None:
    records = load_metric_eligibility(CONFIG)
    eligible = eligible_cross_space_names(records)
    assert "racl_arch_lipschitz_log_upper" not in eligible
    assert "racl_fixed_network_log_upper_proxy" not in eligible
    by_name = {record.metric_name: record for record in records}
    assert by_name["racl_arch_lipschitz_log_upper"].eligibility is MetricEligibility.SPACE_SPECIFIC
    assert by_name["racl_fixed_network_log_upper_proxy"].eligibility is MetricEligibility.SPACE_SPECIFIC


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows + [dict(rows[0])], "duplicate"),
        (lambda rows: [{**rows[0], "reason": ""}, *rows[1:]], "reason"),
        (
            lambda rows: [
                {**rows[0], "supported_settings": ["not_a_setting"]},
                *rows[1:],
            ],
            "unknown setting",
        ),
        (lambda rows: rows[1:], "missing"),
        (
            lambda rows: [
                *rows,
                {
                    "metric_name": "unregistered_metric",
                    "eligibility": "cross_space",
                    "supported_settings": [],
                    "reason": "test-only extra",
                },
            ],
            "unregistered",
        ),
    ],
)
def test_registry_rejects_invalid_records(tmp_path, mutate, match: str) -> None:
    rows = mutate(_payload())
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        records = load_metric_eligibility(path)
        validate_registry(
            records,
            registered_names=[spec.name for spec in METRIC_SPECS],
        )


def test_cross_space_record_requires_all_eight_settings(tmp_path) -> None:
    rows = _payload()
    rows[0] = {**rows[0], "supported_settings": rows[0]["supported_settings"][:-1]}
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="all eight"):
        load_metric_eligibility(path)
