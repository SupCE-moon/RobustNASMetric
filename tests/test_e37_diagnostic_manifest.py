from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import pytest

from scripts.e37_diagnostic_manifest import (
    build_payload,
    expected_darts_keys,
    pinned_source_hashes,
    write_or_verify_manifest,
)


def item(identifier: object, epoch: int) -> SimpleNamespace:
    return SimpleNamespace(model_or_arch_id=str(identifier), checkpoint_epoch=epoch)


def test_exact_200_darts_keys_and_missing_stage() -> None:
    items = [item(i, epoch) for i in range(100) for epoch in (0, 100)]
    assert len(expected_darts_keys(items)) == 200
    assert ("17", 0) in expected_darts_keys(items)
    assert ("17", 100) in expected_darts_keys(items)
    with pytest.raises(ValueError, match="missing"):
        expected_darts_keys(items[:-1])


def test_duplicate_or_unknown_stage_is_rejected() -> None:
    items = [item(i, epoch) for i in range(100) for epoch in (0, 100)]
    with pytest.raises(ValueError, match="duplicate"):
        expected_darts_keys(items + [item(17, 0)])
    with pytest.raises(ValueError, match="unsupported"):
        expected_darts_keys(items + [item(17, 50)])


def test_frozen_manifest_is_idempotent_but_never_overwritten(tmp_path) -> None:
    path = tmp_path / "run" / "frozen_manifest.json"
    original = {"expected_rows": 200, "seed": 20260808}
    first_digest = write_or_verify_manifest(path, original)
    original_bytes = path.read_bytes()
    assert json.loads(original_bytes)["frozen_payload"] == original
    assert write_or_verify_manifest(path, original) == first_digest
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="mismatch"):
        write_or_verify_manifest(path, {"expected_rows": 199, "seed": 20260808})
    assert path.read_bytes() == original_bytes


def test_original_e37_manifest_sha256_map_pins_all_sources(tmp_path) -> None:
    from scripts.e37_diagnostic_manifest import PINNED_SOURCES

    path = tmp_path / "run_manifest.json"
    expected = {name: "a" * 64 for name in PINNED_SOURCES}
    path.write_text(json.dumps({"sha256": expected}), encoding="utf-8")
    assert pinned_source_hashes(path) == expected


def test_payload_hashes_all_200_checkpoints_and_pinned_sources(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "e37_metrics.py"
    source.write_bytes(b"original source")
    bundle_manifest = tmp_path / "darts100_architectures.csv"
    bundle_manifest.write_bytes(b"arch_id,epoch\n")
    strict_csv = tmp_path / "metric_values_merged.csv"
    strict_csv.write_bytes(b"metric_name,metric_value\n")
    control_csv = tmp_path / "darts_e0.csv"
    control_csv.write_bytes(b"model_or_arch_id,log_params\n")
    original_manifest = tmp_path / "original_run_manifest.json"
    original_manifest.write_bytes(b"{}")
    items = [
        SimpleNamespace(
            model_or_arch_id=str(i), checkpoint_epoch=epoch,
            checkpoint_path=checkpoint, target_value=0.5,
        )
        for i in range(100) for epoch in (0, 100)
    ]
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = build_payload(
        items,
        source_paths={"src/robust_metric_lab/e37_metrics.py": source},
        pinned_source_sha256={"src/robust_metric_lab/e37_metrics.py": source_digest},
        original_manifest=original_manifest,
        bundle_manifest=bundle_manifest,
        strict_csv=strict_csv,
        control_csv=control_csv,
        environment={"python": "test", "torch": "test", "numpy": "test"},
    )
    assert payload["expected_rows"] == 200
    assert len(payload["objects"]) == 200
    assert payload["source_sha256"]["src/robust_metric_lab/e37_metrics.py"] == source_digest
    assert payload["objects"][0]["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert payload["protocol"]["num_samples"] == 64
    assert payload["input_paths"]["original_e37_manifest"] == str(original_manifest.resolve())
    assert payload["input_paths"]["strict_csv"] == str(strict_csv.resolve())
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="pinned source mismatch"):
        build_payload(
            items,
            source_paths={"src/robust_metric_lab/e37_metrics.py": source},
            pinned_source_sha256={"src/robust_metric_lab/e37_metrics.py": source_digest},
            original_manifest=original_manifest,
            bundle_manifest=bundle_manifest,
            strict_csv=strict_csv,
            control_csv=control_csv,
            environment={"python": "test", "torch": "test", "numpy": "test"},
        )
