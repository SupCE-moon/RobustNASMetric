"""Post-hoc, write-once byte binding for the pre-provenance E37 attempt_01 batch.

This binds preserved files to the contemporaneous code manifest and launcher
log. It does not independently prove which code executed each model forward.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_attestation(run_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["frozen_payload"]
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if digest != manifest.get("frozen_sha256") or payload.get("expected_rows") != 200:
        raise ValueError("frozen 200-key manifest mismatch")
    code_path = run_root / "diagnostic_code_sha256_v4.json"
    code_hashes = json.loads(code_path.read_text(encoding="utf-8"))["sha256"]
    collector_hash = code_hashes["src/robust_metric_lab/e37_layer_diagnostic.py"]
    runner_hash = code_hashes["scripts/run_e37_layer_diagnostic.py"]
    if _sha256_file(ROOT / "src/robust_metric_lab/e37_layer_diagnostic.py") != collector_hash:
        raise ValueError("diagnostic collector differs from pre-run source hash")
    launcher = run_root / "full_01_launcher.log"
    if runner_hash not in launcher.read_text(encoding="utf-8"):
        raise ValueError("old runner hash absent from formal launcher log")
    attempts_root = run_root / "attempts"
    files = sorted(attempts_root.rglob("*.json"))
    relative = [file.relative_to(attempts_root).as_posix() for file in files]
    if len(files) != 200 or any(not name.startswith("attempt_01/darts_e") or len(Path(name).parts) != 3 for name in relative):
        raise ValueError("legacy attestation requires exactly 200 original attempt_01 JSON files")
    document = {
        "schema": "legacy_e37_diag_batch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "posthoc_binding_not_execution_proof": True,
        "frozen_manifest_sha256": digest,
        "diagnostic_code_sha256": {"collector": collector_hash, "runner": runner_hash},
        "code_manifest_sha256": _sha256_file(code_path),
        "launcher_log_sha256": _sha256_file(launcher),
        "attempt_file_sha256": dict(zip(relative, (_sha256_file(file) for file in files), strict=True)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to(ROOT / "outputs/student_metrics/wangjiahao"):
        raise ValueError("legacy attestation must stay inside wangjiahao's RobustMetric outputs")
    output = run_root / "legacy_attestation_v1.json"
    document = build_attestation(run_root, output)
    print(f"attested_original_files={len(document['attempt_file_sha256'])} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
