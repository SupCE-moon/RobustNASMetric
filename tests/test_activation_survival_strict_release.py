"""New release may extend the registry only inside its own process."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from robust_metric_lab.metric_eligibility import (
    MetricEligibility, MetricEligibilityRecord, SETTING_NAMES, validate_registry,
)
from robust_metric_lab.metric_registry import METRIC_BY_NAME, METRIC_SPECS
from scripts.build_activation_survival_strict_release import (
    CANDIDATE, _read_raw, _snapshot_tree, append_eligibility, candidate_registration,
    require_release_inputs,
)
from robust_metric_lab.student_runner import CSV_COLUMNS
from robust_metric_lab.strict_ranking import ManifestExpectation, validate_metric_grid


def _historical() -> list[MetricEligibilityRecord]:
    return [MetricEligibilityRecord(spec.name, MetricEligibility.SPACE_SPECIFIC, (), "existing")
            for spec in METRIC_SPECS if spec.name != CANDIDATE]


def test_candidate_registration_is_temporary_and_direction_is_fixed() -> None:
    original_count = len(METRIC_SPECS)
    assert CANDIDATE not in METRIC_BY_NAME
    with candidate_registration():
        assert METRIC_BY_NAME[CANDIDATE].direction == "higher"
        assert METRIC_BY_NAME[CANDIDATE].attribution.startswith("Ours")
        assert len(METRIC_SPECS) == original_count + 1
        validate_registry(append_eligibility(_historical()))
        with pytest.raises(ValueError, match="already registered"):
            with candidate_registration():
                pass
    assert CANDIDATE not in METRIC_BY_NAME
    assert len(METRIC_SPECS) == original_count


def test_eligibility_append_preserves_all_exclusions_and_prior_rows() -> None:
    historical = _historical()
    result = append_eligibility(historical)
    assert result[:-1] == historical
    assert result[-1].metric_name == CANDIDATE
    assert result[-1].eligibility is MetricEligibility.CROSS_SPACE
    assert result[-1].supported_settings == SETTING_NAMES
    with pytest.raises(ValueError, match="already eligible"):
        append_eligibility(result)


def test_release_input_gate_requires_full_previous_and_new_releases() -> None:
    old = {"status": "PASS", "raw_rows": 48160, "unique_raw_keys": 48160,
           "ranked_metrics": 84, "excluded_metrics": 2}
    candidate = {"status": "PASS", "unique_success_keys": 560,
                 "metric": CANDIDATE}
    require_release_inputs(old, candidate)
    for altered in ({**old, "ranked_metrics": 83}, {**old, "excluded_metrics": 1}):
        with pytest.raises(ValueError, match="previous strict release"):
            require_release_inputs(altered, candidate)
    with pytest.raises(ValueError, match="candidate verification"):
        require_release_inputs(old, {**candidate, "unique_success_keys": 559})


def test_candidate_csv_column_order_may_differ_but_fields_must_match(tmp_path) -> None:
    path = tmp_path / "candidate.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reversed(CSV_COLUMNS)))
        writer.writeheader()
        writer.writerow({name: "" for name in CSV_COLUMNS})
    assert len(_read_raw(path)) == 1
    path.write_text("unexpected\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        _read_raw(path)


def test_historical_snapshot_detects_changed_exclusion_evidence(tmp_path) -> None:
    old = tmp_path / "strict"
    old.mkdir()
    excluded = old / "ranking" / "excluded_metrics.csv"
    excluded.parent.mkdir()
    excluded.write_text("RACL excluded\n", encoding="utf-8")
    before = _snapshot_tree(old)
    excluded.write_text("RACL included\n", encoding="utf-8")
    assert _snapshot_tree(old) != before


def test_temporary_registration_accepts_only_complete_higher_direction_grid() -> None:
    counts = {"rb40_mature": 40, **{f"nb201_e{epoch}": 64 for epoch in (0, 5, 10, 25, 50)},
              **{f"darts_e{epoch}": 100 for epoch in (0, 100)}}
    rows = []
    expected = {}
    for setting, count in counts.items():
        benchmark = "rb40" if setting == "rb40_mature" else setting.split("_e")[0]
        epoch = None if setting == "rb40_mature" else int(setting.rsplit("e", 1)[1])
        expected[setting] = ManifestExpectation(benchmark, epoch,
                                                {str(index): 0.5 for index in range(count)})
        for index in range(count):
            rows.append({"run_id": "test", "benchmark": benchmark, "setting": setting,
                         "model_or_arch_id": str(index), "checkpoint_epoch": epoch,
                         "metric_name": CANDIDATE, "direction": "higher",
                         "metric_value": 1.0, "target_value": 0.5, "status": "ok",
                         "error": "", "runtime_sec": 1.0, "seed": 20260808,
                         "sample_offset": 0, "num_samples": 64,
                         "epsilon": 8 / 255, "checkpoint_path": "weight.pt"})
    frame = pd.DataFrame(rows)
    with candidate_registration():
        assert len(validate_metric_grid(frame, CANDIDATE, expected)) == 560
        frame.loc[0, "direction"] = "lower"
        with pytest.raises(ValueError, match="direction mismatch"):
            validate_metric_grid(frame, CANDIDATE, expected)
