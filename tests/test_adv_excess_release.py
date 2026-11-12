"""Strict 560-key release and failed-key repair gates."""

import copy
import math

import pytest

from scripts.build_adv_excess_strict_release import (
    IncompleteGridError,
    append_eligibility,
    validate_historical_eligibility,
    validate_old_grid,
    validate_new_grid,
)
from scripts.merge_adv_excess_repair import merge_failed_rows


METRIC = "feature_adv_excess_centered_unit_l2"
SETTINGS = {"rb40_mature": 40, **{f"nb201_e{x}": 64 for x in (0, 5, 10, 25, 50)},
            **{f"darts_e{x}": 100 for x in (0, 100)}}


def frozen_grid():
    objects = {}
    rows = []
    for setting, count in SETTINGS.items():
        epoch = int(setting.split("_e")[1]) if "_e" in setting else None
        benchmark = setting.split("_")[0]
        objects[setting] = []
        for index in range(count):
            object_id = f"{setting}-{index:03d}"
            checkpoint = f"/workspace/robust_metric_lab_nas/checkpoints/{object_id}.pt"
            objects[setting].append({
                "model_or_arch_id": object_id, "checkpoint_epoch": epoch,
                "target_value": 0.5, "checkpoint_path": checkpoint,
                "checkpoint_sha256": f"sha-{object_id}",
            })
            rows.append({
                "run_id": "new-run", "benchmark": benchmark, "setting": setting,
                "model_or_arch_id": object_id,
                "checkpoint_epoch": "" if epoch is None else str(epoch),
                "metric_name": METRIC, "direction": "lower", "metric_value": "0.1",
                "target_value": "0.5", "status": "ok", "error": "",
                "runtime_sec": "1", "seed": "20260808", "sample_offset": "0",
                "num_samples": "64", "epsilon": str(8 / 255),
                "checkpoint_path": checkpoint,
            })
    return {"frozen_payload": {
        "metric_name": METRIC, "expected_rows": 560,
        "expected_counts": SETTINGS, "objects": objects,
        "protocol": {"seed": 20260808, "sample_offset": 0,
                     "num_samples": 64, "epsilon": 8 / 255},
    }}, rows


def test_complete_grid_accepts_only_frozen_560_keys():
    manifest, rows = frozen_grid()
    validate_new_grid(rows, manifest)
    with pytest.raises(IncompleteGridError):
        validate_new_grid(rows[:-1], manifest)
    with pytest.raises(IncompleteGridError):
        validate_new_grid(rows[:-1] + [rows[0]], manifest)


@pytest.mark.parametrize("field,value", [
    ("target_value", "0.6"), ("checkpoint_path", "/wrong.pt"),
    ("seed", "3"), ("sample_offset", "1"), ("num_samples", "32"),
    ("metric_name", "other"), ("setting", "other"),
    ("metric_value", str(math.inf)), ("status", "failed"),
])
def test_changed_protocol_or_invalid_result_is_rejected(field, value):
    manifest, rows = frozen_grid()
    modified = copy.deepcopy(rows)
    modified[0][field] = value
    with pytest.raises(IncompleteGridError):
        validate_new_grid(modified, manifest)


def test_repair_replaces_only_identical_failed_key():
    _, rows = frozen_grid()
    original = copy.deepcopy(rows[:2])
    original[0]["status"] = "failed"
    original[0]["metric_value"] = ""
    repair = copy.deepcopy(original[0])
    repair.update(status="ok", metric_value="0.2", error="")
    merged = merge_failed_rows(original, [repair])
    assert merged[0] == repair
    assert merged[1] == original[1]
    with pytest.raises(ValueError):
        merge_failed_rows(original, [repair, repair])
    bad = copy.deepcopy(repair)
    bad["checkpoint_path"] = "/changed.pt"
    with pytest.raises(ValueError):
        merge_failed_rows(original, [bad])
    with pytest.raises(ValueError):
        merge_failed_rows(rows[:2], [repair])


def test_historical_eligibility_is_preserved_exactly():
    historical = [
        {"metric_name": "trnas_activation_psi", "eligibility": "cross_space",
         "supported_settings": list(SETTINGS), "reason": "repaired"},
        {"metric_name": "racl_arch_lipschitz_log_upper", "eligibility": "excluded",
         "supported_settings": [], "reason": "missing implementation"},
    ]
    added = append_eligibility(historical)
    assert added[:-1] == historical
    assert len(added) == 3
    assert added[-1]["metric_name"] == METRIC
    assert added[-1]["supported_settings"] == list(SETTINGS)
    with pytest.raises(ValueError):
        append_eligibility(added)


def test_historical_raw_requires_original_47600_unique_keys():
    _, rows = frozen_grid()
    with pytest.raises(IncompleteGridError):
        validate_old_grid(rows, expected_names={"historic"})


def test_only_two_historically_excluded_racl_names_are_pinned():
    records = [
        {"metric_name": "racl_arch_lipschitz_log_upper",
         "eligibility": "space_specific"},
        {"metric_name": "racl_fixed_network_log_upper_proxy",
         "eligibility": "space_specific"},
        {"metric_name": "racl_input_gradient_lower",
         "eligibility": "cross_space"},
        {"metric_name": "trnas_activation_psi", "eligibility": "cross_space"},
        {"metric_name": "trnas_consistency_phi", "eligibility": "cross_space"},
        {"metric_name": "trnas_r_score", "eligibility": "cross_space"},
    ]
    validate_historical_eligibility(records)
    changed = copy.deepcopy(records)
    changed[0]["eligibility"] = "cross_space"
    with pytest.raises(IncompleteGridError):
        validate_historical_eligibility(changed)
