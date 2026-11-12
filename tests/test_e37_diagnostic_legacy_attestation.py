from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_legacy_attestation_freezes_exact_200_attempt_bytes(tmp_path: Path) -> None:
    from scripts.attest_e37_legacy_batch import build_attestation

    project = Path(__file__).resolve().parents[1]
    collector = project / "src/robust_metric_lab/e37_layer_diagnostic.py"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"expected_rows": 200}
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    manifest = tmp_path / "frozen_manifest.json"
    manifest.write_text(json.dumps({"frozen_payload": payload, "frozen_sha256": payload_hash}), encoding="utf-8")
    code_manifest = tmp_path / "diagnostic_code_sha256_v4.json"
    code_manifest.write_text(json.dumps({"sha256": {
        "src/robust_metric_lab/e37_layer_diagnostic.py": digest(collector),
        "scripts/run_e37_layer_diagnostic.py": "old-runner-hash",
    }}), encoding="utf-8")
    launcher = tmp_path / "full_01_launcher.log"
    launcher.write_text("runner_sha256=old-runner-hash\n", encoding="utf-8")
    for stage in (0, 100):
        for identifier in range(100):
            path = tmp_path / "attempts" / "attempt_01" / f"darts_e{stage}" / f"{identifier}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": "ok", "id": identifier}) + "\n", encoding="utf-8")
    output = tmp_path / "legacy_attestation_v1.json"
    document = build_attestation(tmp_path, output)
    assert document["frozen_manifest_sha256"] == payload_hash
    assert len(document["attempt_file_sha256"]) == 200
    assert document["posthoc_binding_not_execution_proof"] is True
    with pytest.raises(FileExistsError):
        build_attestation(tmp_path, output)
