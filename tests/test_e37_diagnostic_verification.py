from __future__ import annotations

import copy
import hashlib
import json
import math
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
    bounded = [value / (1.0 + value) for value in (0.1, 0.2, 0.1)]
    consistent_scores = dict(zip(E37_METRIC_NAMES, (
        0.1, 0.1, 0.1, 0.3, 0.4, 0.5,
        math.prod(bounded) ** (1.0 / 3.0),
        max(bounded), math.sqrt(sum(value * value for value in bounded) / 3.0),
    ), strict=True))
    scores = {key: dict(consistent_scores) for key in keys}
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
            "diagnostic_code_sha256": {"collector": "collector-hash", "runner": "runner-hash"},
            "protocol": {"epsilon": 8 / 255, "sample_count": 64, "batch_size": 1},
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
        "diagnostic_code_sha256": {"collector": "collector-hash", "runner": "runner-hash"},
        "protocol": {"epsilon": 8 / 255, "sample_count": 64, "batch_size": 1},
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
        "diagnostic_code_sha256": {"collector": "collector-hash", "runner": "runner-hash"},
        "protocol": {"epsilon": 8 / 255, "sample_count": 64, "batch_size": 1},
    })
    result = _verify(valid_grid, records)
    assert result["status"] == "PASS"
    assert result["failed_attempt_count"] == 1


def test_invalid_ok_record_remains_exact_repair_key(valid_grid) -> None:
    records = copy.deepcopy(valid_grid[0])
    row = records[0]
    row["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = 123456.0
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL"
    assert (row["setting"], row["model_or_arch_id"]) in result["repair_keys"]


@pytest.mark.parametrize("bad_first", [True, False])
def test_invalid_ok_then_one_valid_repair_passes_with_rejection_evidence(valid_grid, bad_first) -> None:
    records = copy.deepcopy(valid_grid[0])
    bad = copy.deepcopy(records[0])
    bad["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = 123456.0
    records.insert(0 if bad_first else 1, bad)
    result = _verify(valid_grid, records)
    assert result["status"] == "PASS"
    assert result["successful_key_count"] == 200
    assert result["rejected_attempt_count"] == 1
    assert result["rejected_attempts"][0]["key"] == [bad["setting"], bad["model_or_arch_id"]]
    assert not result["repair_keys"]


def test_provenance_mismatch_remains_fatal_after_valid_repair(valid_grid) -> None:
    records = copy.deepcopy(valid_grid[0])
    bad = copy.deepcopy(records[0])
    bad["source_sha256"] = {"e37": "wrong"}
    records.insert(0, bad)
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL"
    assert any("source hash mismatch" in reason for reason in result["reasons"])


def test_component_reconstruction_must_match_strict_directly(valid_grid) -> None:
    records = copy.deepcopy(valid_grid[0])
    row = records[0]
    row["diagnostic"]["sample_components"] = copy.deepcopy(row["diagnostic"]["sample_components"])
    name = "feature_measure_spectrum_fisher_rao_delta"
    row["diagnostic"]["metric_values"] = dict(row["diagnostic"]["metric_values"])
    row["diagnostic"]["metric_values"][name] = 0.300007
    for sample in row["diagnostic"]["sample_components"]:
        for anchor in sample["anchors"]:
            anchor["spectrum_fisher"] = 0.300014
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL"
    assert (row["setting"], row["model_or_arch_id"]) in result["repair_keys"]


@pytest.mark.parametrize("mutation", ["diagnostic", "anchor", "sample", "huge_feature_count", "huge_component"])
def test_null_nested_record_yields_fail_evidence(valid_grid, mutation) -> None:
    records = copy.deepcopy(valid_grid[0])
    row = records[0]
    if mutation == "diagnostic":
        row["diagnostic"] = None
    elif mutation == "anchor":
        row["diagnostic"]["anchor_metadata"][0] = None
    elif mutation == "sample":
        row["diagnostic"]["sample_components"] = [None] * 64
    elif mutation == "huge_feature_count":
        row["diagnostic"]["feature_count"] = 10 ** 100
    else:
        row["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = 10 ** 1000
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL"
    assert result["reasons"]
    assert (row["setting"], row["model_or_arch_id"]) in result["repair_keys"]


@pytest.mark.parametrize("mutation", [
    "finite_channel", "finite_flow", "singular_channel", "missing_anchor_index", "duplicate_anchor_index",
    "wrong_anchor_index", "wrong_protocol", "wrong_diagnostic_code", "missing_diagnostic_code",
])
def test_finite_component_or_anchor_or_protocol_corruption_fails(valid_grid, mutation) -> None:
    records = copy.deepcopy(valid_grid[0])
    row = records[0]
    if mutation == "finite_channel":
        row["diagnostic"]["sample_components"][0]["anchors"][0]["channel"] = 123456.0
    elif mutation == "singular_channel":
        for sample in row["diagnostic"]["sample_components"]:
            for anchor in sample["anchors"]:
                anchor["channel"] = -1.0
    elif mutation == "finite_flow":
        row["diagnostic"]["sample_components"][0]["transitions"][1]["flow"] = 2.0
    elif mutation == "missing_anchor_index":
        for item in records:
            for anchor in item["diagnostic"]["anchor_metadata"]:
                anchor.pop("anchor_index")
    elif mutation == "duplicate_anchor_index":
        for item in records:
            for anchor in item["diagnostic"]["anchor_metadata"]:
                anchor["anchor_index"] = 0
    elif mutation == "wrong_anchor_index":
        row["diagnostic"]["anchor_metadata"][2]["anchor_index"] = 99
    elif mutation == "wrong_protocol":
        row["protocol"] = {"epsilon": 0, "sample_count": 64, "batch_size": 64}
    elif mutation == "wrong_diagnostic_code":
        row["diagnostic_code_sha256"]["collector"] = "changed"
    elif mutation == "missing_diagnostic_code":
        row.pop("diagnostic_code_sha256")
    result = _verify(valid_grid, records)
    assert result["status"] == "FAIL", mutation
    assert result["reasons"]


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


def test_legacy_attestation_binds_exact_original_attempt_files(tmp_path: Path) -> None:
    from scripts.verify_e37_layer_diagnostic import load_legacy_attestation

    attempts = tmp_path / "attempts"
    record = attempts / "attempt_01" / "darts_e0" / "17.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"status":"ok"}\n', encoding="utf-8")
    launcher = tmp_path / "full_01_launcher.log"
    launcher.write_text("runner_sha256=runner-hash\n", encoding="utf-8")
    code_manifest = tmp_path / "diagnostic_code_sha256_v4.json"
    code_manifest.write_text(json.dumps({"sha256": {
        "scripts/run_e37_layer_diagnostic.py": "runner-hash",
        "src/robust_metric_lab/e37_layer_diagnostic.py": "collector-hash",
    }}) + "\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    attestation = tmp_path / "legacy_attestation_v1.json"
    attestation.write_text(json.dumps({
        "schema": "legacy_e37_diag_batch_v1",
        "frozen_manifest_sha256": "frozen-hash",
        "diagnostic_code_sha256": {"collector": "collector-hash", "runner": "runner-hash"},
        "code_manifest_sha256": digest(code_manifest),
        "launcher_log_sha256": digest(launcher),
        "attempt_file_sha256": {"attempt_01/darts_e0/17.json": digest(record)},
    }), encoding="utf-8")
    kwargs = dict(
        attempts_dir=attempts, manifest_hash="frozen-hash",
        code_hashes={"collector": "collector-hash", "runner": "runner-hash"},
        code_manifest_path=code_manifest, launcher_log_path=launcher,
    )
    assert load_legacy_attestation(attestation, **kwargs) == {"attempt_01/darts_e0/17.json"}
    record.write_text('{"status":"failed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="attempt.*hash"):
        load_legacy_attestation(attestation, **kwargs)
    record.write_text('{"status":"ok"}\n', encoding="utf-8")
    extra = attempts / "unexpected" / "extra.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attempt.*hash"):
        load_legacy_attestation(attestation, **kwargs)
