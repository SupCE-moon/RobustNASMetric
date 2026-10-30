import copy
import math
from pathlib import Path

import pytest

from scripts.verify_activation_survival import main, verify_frozen_files, verify_records


SOURCE = {"candidate.py": "a" * 64}


def full_fixture():
    objects = []
    for setting, count, epoch in [
        ("rb40_mature", 40, None),
        *[(f"nb201_e{e}", 64, e) for e in (0, 5, 10, 25, 50)],
        *[(f"darts_e{e}", 100, e) for e in (0, 100)],
    ]:
        for i in range(count):
            objects.append({
                "setting": setting, "model_or_arch_id": str(i),
                "checkpoint_epoch": epoch, "target_value": 0.5,
                "checkpoint_sha256": "b" * 64,
                "checkpoint_path": str((Path.cwd() / "weight.pt").resolve()),
                "artifact_sha256": {"weight.pt": "b" * 64},
            })
    manifest = {
        "frozen_sha256": "c" * 64,
        "frozen_payload": {"expected_rows": 560, "objects": objects,
                           "source_sha256": SOURCE},
    }
    attempts = [{
        "key": [obj["setting"], obj["model_or_arch_id"], obj["checkpoint_epoch"]],
        "status": "ok", "metric_value": 1.0, "target_value": 0.5,
        "sample_count": 64, "call_count": 1,
        "checkpoint_sha256": "b" * 64,
        "artifact_sha256": {"weight.pt": "b" * 64},
        "manifest_sha256": "c" * 64, "source_sha256": SOURCE,
        "diagnostics": {"mean_retention": [0.5], "elapsed_sec": 1.0},
    } for obj in objects]
    return manifest, attempts


def test_exact_560_successes_pass():
    manifest, attempts = full_fixture()
    result = verify_records(manifest, attempts)
    assert len(result["rows"]) == 560
    assert result["failed_attempts"] == 0
    assert {"run_id", "benchmark", "error", "runtime_sec"} <= set(result["rows"][0])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "failed",
                                       "nan", "target", "source", "checkpoint", "sample"])
def test_invalid_attempts_block_release(mutation):
    manifest, attempts = full_fixture()
    if mutation == "missing":
        attempts.pop()
    elif mutation == "duplicate":
        attempts.append(copy.deepcopy(attempts[0]))
    elif mutation == "unknown":
        attempts[0]["key"][1] = "unknown"
    elif mutation == "failed":
        attempts[0]["status"] = "failed"
        attempts[0]["error"] = "OOM"
    elif mutation == "nan":
        attempts[0]["metric_value"] = math.nan
    elif mutation == "target":
        attempts[0]["target_value"] = 0.6
    elif mutation == "source":
        attempts[0]["source_sha256"] = {"candidate.py": "d" * 64}
    elif mutation == "checkpoint":
        attempts[0]["checkpoint_sha256"] = "d" * 64
    elif mutation == "sample":
        attempts[0]["sample_count"] = 63
    with pytest.raises(ValueError):
        verify_records(manifest, attempts)


def test_nested_nonfinite_diagnostic_blocks_release():
    manifest, attempts = full_fixture()
    attempts[0]["diagnostics"]["mean_retention"] = [math.inf]
    with pytest.raises(ValueError, match="nonfinite"):
        verify_records(manifest, attempts)


def test_prior_failed_attempt_retained_but_later_exact_success_valid():
    manifest, attempts = full_fixture()
    failed = copy.deepcopy(attempts[0])
    failed.update({"status": "failed", "error": "old OOM"})
    attempts.append(failed)
    result = verify_records(manifest, attempts)
    assert len(result["rows"]) == 560
    assert result["failed_attempts"] == 1


def test_forged_manifest_with_unequal_nb_stage_ids_is_rejected():
    manifest, attempts = full_fixture()
    changed = next(row for row in manifest["frozen_payload"]["objects"]
                   if row["setting"] == "nb201_e50" and row["model_or_arch_id"] == "0")
    changed["model_or_arch_id"] = "extra"
    with pytest.raises(ValueError, match="unequal|stage"):
        verify_records(manifest, attempts)


def test_frozen_checkpoint_bytes_are_rehashed_before_release(tmp_path):
    manifest, _ = full_fixture()
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"frozen")
    digest = __import__("hashlib").sha256(b"frozen").hexdigest()
    for obj in manifest["frozen_payload"]["objects"]:
        obj["checkpoint_path"] = str(weight)
        obj["checkpoint_sha256"] = digest
        obj["artifact_sha256"] = {str(weight): digest}
    source = tmp_path / "candidate.py"
    source.write_bytes(b"source")
    source_hash = __import__("hashlib").sha256(b"source").hexdigest()
    manifest["frozen_payload"]["source_sha256"] = {"candidate.py": source_hash}
    manifest["frozen_payload"]["input_paths"] = {"bundle": {}, "sample": {}, "strict_csv": str(source)}
    manifest["frozen_payload"]["bundle_manifest_sha256"] = {}
    manifest["frozen_payload"]["sample_sha256"] = {}
    manifest["frozen_payload"]["strict_csv_sha256"] = source_hash
    verify_frozen_files(manifest, tmp_path)
    weight.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checkpoint|artifact"):
        verify_frozen_files(manifest, tmp_path)


def test_verifier_cli_refuses_output_outside_project(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--manifest", str(tmp_path / "manifest.json"),
              "--attempts-dir", str(tmp_path / "attempts"),
              "--output-dir", str(tmp_path / "release")])
    assert error.value.code == 2
