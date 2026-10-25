from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from robust_metric_lab.e37_metrics import E37_METRIC_NAMES
from scripts.verify_e37_layer_diagnostic import read_json_strict, verify_attempts, verify_records, write_verification_output


@pytest.fixture
def valid_grid():
    keys = {(f"darts_e{stage}", str(i)) for stage in (0, 100) for i in range(100)}
    scores = {key: {name: 0.5 for name in E37_METRIC_NAMES} for key in keys}
    anchors = [
        {"module_path": f"layer{slot}", "invocation": 1, "shape": [1, 2, 4, 4], "anchor_index": slot}
        for slot in range(4)
    ]
    sample = {
        "anchors": [{
            "clean_rms": 1.0, "adv_rms": 1.1, "clean_centered_l2": 1.0,
            "adv_centered_l2": 1.1, "channel": 0.1, "spatial": 0.2,
            "spectrum_fisher": 0.3, "spectrum_triangular": 0.4, "spectrum_ks": 0.5,
        } for _ in range(4)],
        "transitions": [{"flow": 0.1} for _ in range(3)],
    }
    samples = [dict(sample, sample_index=index) for index in range(64)]
    records = [
        {
            "setting": setting, "model_or_arch_id": identifier,
            "status": "ok", "source_sha256": {"e37": "source"},
            "checkpoint_sha256": "ck", "manifest_sha256": "manifest",
            "diagnostic": {
                "sample_count": 64, "feature_count": 4,
                "anchor_metadata": copy.deepcopy(anchors), "sample_components": samples,
                "metric_values": scores[(setting, identifier)],
            },
        }
        for setting, identifier in sorted(keys)
    ]
    return records, keys, scores, {
        "source_sha256": {"e37": "source"},
        "checkpoint_sha256": "ck", "manifest_sha256": "manifest",
    }


def _verify(grid, records):
    _, keys, scores, hashes = grid
    return verify_records(records, expected_keys=keys, strict_scores=scores, frozen_hashes=hashes)


def test_exact_finite_200_grid_passes(valid_grid) -> None:
    result = _verify(valid_grid, valid_grid[0])
    assert result["status"] == "PASS"
    assert result["successful_key_count"] == 200
    assert result["paired_architecture_count"] == 100
    assert result["reconstruction_max_abs_error"] == 0
    assert len(result["reconstruction_rows"]) == 1800


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "duplicate_ok", "nested_nan", "failed_only",
    "changed_hash", "score_mismatch", "pair_mismatch", "bad_sample_count",
])
def test_corrupt_grid_fails_with_evidence(valid_grid, mutation) -> None:
    records = copy.deepcopy(valid_grid[0])
    row = records[0]
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append({**copy.deepcopy(row), "model_or_arch_id": "not-frozen"})
    elif mutation == "duplicate_ok":
        records.append(copy.deepcopy(row))
    elif mutation == "nested_nan":
        row["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = float("nan")
    elif mutation == "failed_only":
        row.update(status="failed", error="OOM", traceback="OOM")
        row.pop("diagnostic")
    elif mutation == "changed_hash":
        row["source_sha256"] = {"e37": "changed"}
    elif mutation == "score_mismatch":
        row["diagnostic"]["metric_values"][E37_METRIC_NAMES[0]] = 0.7
    elif mutation == "pair_mismatch":
        counterpart = next(item for item in records if item["setting"] == "darts_e100" and item["model_or_arch_id"] == row["model_or_arch_id"])
        counterpart["diagnostic"]["anchor_metadata"][0]["shape"] = [1, 99, 8, 8]
    elif mutation == "bad_sample_count":
        row["diagnostic"]["sample_components"].pop()
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL"
    assert result["reasons"]
    if mutation == "pair_mismatch":
        assert ("darts_e100", row["model_or_arch_id"]) in result["pair_mismatches"]


def test_failed_attempt_then_success_keeps_audit_and_passes(valid_grid) -> None:
    records = list(valid_grid[0])
    records.append({
        "setting": "darts_e0", "model_or_arch_id": "17", "status": "failed",
        "error_type": "RuntimeError", "error": "OOM", "traceback": "RuntimeError: OOM",
        "source_sha256": {"e37": "source"}, "checkpoint_sha256": "ck",
        "manifest_sha256": "manifest",
    })
    result = _verify(valid_grid, records)
    assert result["status"] == "PASS"
    assert result["failed_attempt_count"] == 1


def test_duplicate_json_field_and_nonfinite_token_are_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"status":"ok","status":"failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_json_strict(path)
    path.write_text('{"metric_value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON"):
        read_json_strict(path)


def test_fail_verification_writes_evidence_but_no_canonical(tmp_path) -> None:
    output = tmp_path / "v1"
    result = {
        "status": "FAIL", "reasons": ["record 3: nonfinite channel"],
        "repair_keys": [("darts_e0", "17")], "canonical_records": [],
    }
    write_verification_output(output, result)
    assert (output / "verification.json").exists()
    assert (output / "repair_keys.csv").exists()
    assert not (output / "canonical_numeric.jsonl").exists()
    with pytest.raises(FileExistsError):
        write_verification_output(output, result)


def test_corrupt_manifest_still_yields_fail_evidence(tmp_path) -> None:
    manifest = tmp_path / "corrupt.json"
    manifest.write_text('{"frozen_payload":NaN}', encoding="utf-8")
    output = tmp_path / "v1"
    result = verify_attempts(manifest, tmp_path / "attempts", tmp_path / "strict.csv", output)
    assert result["status"] == "FAIL"
    assert "nonfinite JSON token" in result["reasons"][0]
    assert (output / "verification.json").exists()
    assert not (output / "canonical_numeric.jsonl").exists()


def test_verifier_cli_starts_without_external_pythonpath() -> None:
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(project / "scripts/verify_e37_layer_diagnostic.py"), "--help"],
        cwd=project, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
