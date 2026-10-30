"""Independent exact-key verifier for sign-survival attempts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.activation_survival_manifest import (  # noqa: E402
    SETTING_COUNTS, _canonical, expected_main_keys, require_project_path, sha256_file,
)


def _finite_tree(value: object, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {where}")
        return
    if isinstance(value, list):
        for child in value:
            _finite_tree(child, where)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{where}.{key}")
        return
    raise ValueError(f"unsupported diagnostic type at {where}")


def _key(row: Mapping) -> tuple[str, str, int | None]:
    setting, identifier, epoch = row["key"]
    return str(setting), str(identifier), epoch


def verify_records(manifest: dict, attempts: list[dict]) -> dict:
    """Only a complete 560-success set can become a canonical release CSV."""
    payload = manifest["frozen_payload"]
    objects = payload["objects"]
    grouped = {setting: [] for setting in SETTING_COUNTS}
    for obj in objects:
        if obj.get("setting") not in grouped:
            raise ValueError("unknown manifest setting")
        grouped[obj["setting"]].append(SimpleNamespace(**obj))
    expected_main_keys(grouped)
    expected: dict[tuple[str, str, int | None], dict] = {}
    counts = {setting: 0 for setting in SETTING_COUNTS}
    for obj in objects:
        key = (obj["setting"], str(obj["model_or_arch_id"]), obj["checkpoint_epoch"])
        if key in expected or key[0] not in counts:
            raise ValueError(f"duplicate or unknown manifest key: {key}")
        expected[key] = obj
        counts[key[0]] += 1
    if counts != SETTING_COUNTS or len(expected) != 560 or payload.get("expected_rows") != 560:
        raise ValueError("missing exact 560-key manifest")
    source = payload["source_sha256"]
    manifest_hash = manifest["frozen_sha256"]
    successes: dict[tuple[str, str, int | None], dict] = {}
    failed = 0
    for row in attempts:
        key = _key(row)
        if key not in expected:
            raise ValueError(f"unknown attempt key: {key}")
        if row.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"manifest mismatch: {key}")
        if row.get("source_sha256") != source:
            raise ValueError(f"source mismatch: {key}")
        status = row.get("status")
        if status == "failed":
            failed += 1
            if not row.get("error"):
                raise ValueError(f"failed attempt missing error: {key}")
            continue
        if status != "ok":
            raise ValueError(f"invalid status: {key}")
        if key in successes:
            raise ValueError(f"duplicate success: {key}")
        obj = expected[key]
        if row.get("target_value") != obj["target_value"]:
            raise ValueError(f"target mismatch: {key}")
        if row.get("checkpoint_sha256") != obj["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch: {key}")
        if row.get("artifact_sha256", {}) != obj.get("artifact_sha256", {}):
            raise ValueError(f"artifact mismatch: {key}")
        if row.get("sample_count") != 64 or not isinstance(row.get("call_count"), int) or row["call_count"] < 1:
            raise ValueError(f"sample or call count mismatch: {key}")
        value = row.get("metric_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"nonfinite or nonnumeric score: {key}")
        _finite_tree(row.get("diagnostics", {}), f"diagnostics {key}")
        runtime = row.get("diagnostics", {}).get("elapsed_sec")
        if not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
            raise ValueError(f"missing or invalid runtime: {key}")
        successes[key] = row
    if set(successes) != set(expected):
        missing = sorted(set(expected) - set(successes), key=str)
        raise ValueError(f"missing success keys: {missing[:10]} ({len(missing)} total)")
    rows = []
    for key in sorted(successes, key=lambda value: (list(SETTING_COUNTS).index(value[0]), value[1])):
        obj = expected[key]
        result = successes[key]
        benchmark = "rb40" if key[0] == "rb40_mature" else key[0].split("_e")[0]
        rows.append({
            "run_id": payload.get("run_id", "activation_sign_survival"),
            "benchmark": benchmark,
            "setting": key[0], "model_or_arch_id": key[1], "checkpoint_epoch": key[2],
            "metric_name": "feature_activation_sign_survival_sum",
            "metric_value": result["metric_value"], "target_value": obj["target_value"],
            "status": "ok", "direction": "higher", "seed": 20260808,
            "sample_offset": 0, "num_samples": 64, "epsilon": 8 / 255,
            "error": "", "runtime_sec": result["diagnostics"]["elapsed_sec"],
            "checkpoint_path": obj.get("checkpoint_path", ""),
        })
    return {"rows": rows, "failed_attempts": failed, "unique_success_keys": len(successes)}


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _strict_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(
            stream, object_pairs_hook=_strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")),
        )


def verify_frozen_files(manifest: dict, project_root: Path) -> None:
    """Re-read every code, input and weight byte before admitting a release."""
    payload = manifest["frozen_payload"]
    root = project_root.resolve()
    cache: dict[str, str] = {}

    def checked_hash(path_value: str | Path) -> str:
        path = Path(path_value).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"frozen path outside permitted project: {path}")
        path_key = str(path)
        if path_key not in cache:
            cache[path_key] = sha256_file(path)
        return cache[path_key]

    for relative, expected_hash in payload["source_sha256"].items():
        if checked_hash(root / relative) != expected_hash:
            raise ValueError(f"source mismatch: {relative}")
    for category, hashes in (
        ("bundle", payload.get("bundle_manifest_sha256", {})),
        ("sample", payload.get("sample_sha256", {})),
    ):
        for name, expected_hash in hashes.items():
            if checked_hash(payload["input_paths"][category][name]) != expected_hash:
                raise ValueError(f"{category} mismatch: {name}")
    if checked_hash(payload["input_paths"]["strict_csv"]) != payload["strict_csv_sha256"]:
        raise ValueError("strict source mismatch")
    for obj in payload["objects"]:
        checkpoint = obj["checkpoint_path"]
        if checked_hash(checkpoint) != obj["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch: {checkpoint}")
        artifacts = obj["artifact_sha256"]
        if checkpoint not in artifacts:
            raise ValueError("primary checkpoint absent from artifacts")
        for path, expected_hash in artifacts.items():
            if checked_hash(path) != expected_hash:
                raise ValueError(f"artifact mismatch: {path}")


def verify_attempts(
    manifest_path: Path, attempts_dir: Path, output_dir: Path, project_root: Path,
) -> dict:
    """Re-hash frozen inputs and write a canonical CSV only after full validation."""
    manifest = _strict_json(manifest_path)
    payload = manifest["frozen_payload"]
    if hashlib.sha256(_canonical(payload)).hexdigest() != manifest["frozen_sha256"]:
        raise ValueError("frozen manifest hash mismatch")
    verify_frozen_files(manifest, project_root)
    attempts = [_strict_json(path) for path in sorted(attempts_dir.glob("*.json"))]
    result = verify_records(manifest, attempts)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metric_values.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to replace canonical CSV: {csv_path}")
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
        stream.flush()
        os.fsync(stream.fileno())
    verification = {
        "status": "PASS", "metric": "feature_activation_sign_survival_sum",
        "unique_success_keys": result["unique_success_keys"],
        "failed_attempts_retained": result["failed_attempts"],
        "manifest_sha256": manifest["frozen_sha256"],
        "canonical_csv_sha256": sha256_file(csv_path),
    }
    report_path = output_dir / "release_verification.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--attempts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest_path = require_project_path(args.manifest, ROOT)
        attempts_dir = require_project_path(args.attempts_dir, ROOT)
        output_dir = require_project_path(args.output_dir, ROOT)
        result = verify_attempts(manifest_path, attempts_dir, output_dir, ROOT)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
