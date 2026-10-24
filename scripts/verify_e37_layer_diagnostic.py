"""Fail closed on incomplete or inconsistent 200-key E37 layer diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.e37_metrics import E37_METRIC_NAMES  # noqa: E402
from scripts.run_e37_layer_diagnostic import load_frozen_manifest  # noqa: E402

ANCHOR_COMPONENTS = {
    "clean_rms", "adv_rms", "clean_centered_l2", "adv_centered_l2",
    "channel", "spatial", "spectrum_fisher", "spectrum_triangular", "spectrum_ks",
}


def _finite_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric {where}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {where}")
    return number


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field: {key}")
        output[key] = value
    return output


def _parse_constant(value: str):
    raise ValueError(f"nonfinite JSON token: {value}")


def read_json_strict(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_pairs,
        parse_constant=_parse_constant,
    )


def _record_key(record: Mapping) -> tuple[str, str]:
    return str(record["setting"]), str(record["model_or_arch_id"])


def _expected_checkpoint(frozen_hashes: Mapping, key: tuple[str, str]) -> str:
    value = frozen_hashes["checkpoint_sha256"]
    return str(value[key] if isinstance(value, dict) else value)


def _check_success(record: dict, strict_values: Mapping[str, float]) -> tuple[float, float]:
    diagnostic = record["diagnostic"]
    if diagnostic.get("sample_count") != 64:
        raise ValueError("sample_count is not 64")
    anchors = diagnostic.get("anchor_metadata")
    if not isinstance(anchors, list) or len(anchors) != 4:
        raise ValueError("anchor_metadata does not contain four anchors")
    if not isinstance(diagnostic.get("feature_count"), int) or diagnostic["feature_count"] < 4:
        raise ValueError("invalid feature_count")
    for slot, anchor in enumerate(anchors):
        if not isinstance(anchor.get("module_path"), str) or not anchor["module_path"]:
            raise ValueError(f"anchor {slot} missing module_path")
        if not isinstance(anchor.get("invocation"), int) or anchor["invocation"] < 1:
            raise ValueError(f"anchor {slot} invalid invocation")
        shape = anchor.get("shape")
        if not isinstance(shape, list) or len(shape) != 4 or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in shape
        ) or shape[0] != 1:
            raise ValueError(f"anchor {slot} invalid shape")
    samples = diagnostic.get("sample_components")
    if not isinstance(samples, list) or len(samples) != 64:
        raise ValueError("sample_components does not contain 64 samples")
    for index, sample in enumerate(samples):
        if sample.get("sample_index") != index:
            raise ValueError(f"sample index mismatch at {index}")
        layers, transitions = sample.get("anchors"), sample.get("transitions")
        if not isinstance(layers, list) or len(layers) != 4:
            raise ValueError(f"sample {index} lacks four anchors")
        if not isinstance(transitions, list) or len(transitions) != 3:
            raise ValueError(f"sample {index} lacks three transitions")
        for slot, layer in enumerate(layers):
            if set(layer) != ANCHOR_COMPONENTS:
                raise ValueError(f"sample {index} anchor {slot} components differ")
            for name, value in layer.items():
                _finite_number(value, f"sample {index} anchor {slot} {name}")
        for slot, transition in enumerate(transitions):
            if set(transition) != {"flow"}:
                raise ValueError(f"sample {index} transition {slot} differs")
            _finite_number(transition["flow"], f"sample {index} transition {slot}")
    values = diagnostic.get("metric_values")
    if not isinstance(values, dict) or set(values) != set(E37_METRIC_NAMES):
        raise ValueError("nine E37 metric names differ")
    if set(strict_values) != set(E37_METRIC_NAMES):
        raise ValueError("strict reference lacks nine E37 metrics")
    max_abs = 0.0
    max_rel = 0.0
    for name in E37_METRIC_NAMES:
        actual = _finite_number(values[name], name)
        expected = _finite_number(strict_values[name], f"strict {name}")
        difference = abs(actual - expected)
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, difference / max(abs(expected), 1e-12))
        if difference > 3e-8 + 3e-5 * abs(expected):
            raise ValueError(f"strict reconstruction mismatch: {name} abs={difference}")
    return max_abs, max_rel


def verify_records(
    records: list[dict], *,
    expected_keys: set[tuple[str, str]],
    strict_scores: Mapping[tuple[str, str], Mapping[str, float]],
    frozen_hashes: Mapping,
) -> dict:
    reasons: list[str] = []
    successes: dict[tuple[str, str], dict] = {}
    failed_count = 0
    max_abs = 0.0
    max_rel = 0.0
    for index, record in enumerate(records):
        try:
            key = _record_key(record)
            if key not in expected_keys:
                raise ValueError(f"unexpected key {key}")
            if record.get("source_sha256") != frozen_hashes["source_sha256"]:
                raise ValueError("source hash mismatch")
            if record.get("checkpoint_sha256") != _expected_checkpoint(frozen_hashes, key):
                raise ValueError("checkpoint hash mismatch")
            if "manifest_sha256" in frozen_hashes and record.get("manifest_sha256") != frozen_hashes["manifest_sha256"]:
                raise ValueError("manifest hash mismatch")
            status = record.get("status")
            if status == "failed":
                failed_count += 1
                if "diagnostic" in record or not record.get("traceback"):
                    raise ValueError("failed attempt has invented values or no traceback")
            elif status == "ok":
                if key in successes:
                    raise ValueError(f"duplicate successful key {key}")
                successes[key] = record
                abs_error, rel_error = _check_success(record, strict_scores[key])
                max_abs = max(max_abs, abs_error)
                max_rel = max(max_rel, rel_error)
            else:
                raise ValueError(f"invalid status {status}")
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"record {index}: {exc}")
    missing = sorted(expected_keys - set(successes))
    if missing:
        reasons.append(f"missing successful keys: {len(missing)}; examples={missing[:10]}")
    pair_mismatches: list[tuple[str, str]] = []
    paired_count = 0
    identifiers = {identifier for _, identifier in expected_keys}
    for identifier in sorted(identifiers):
        e0 = successes.get(("darts_e0", identifier))
        e100 = successes.get(("darts_e100", identifier))
        if e0 is None or e100 is None:
            continue
        if e0["diagnostic"].get("anchor_metadata") != e100["diagnostic"].get("anchor_metadata"):
            pair_mismatches.append(("darts_e100", identifier))
        else:
            paired_count += 1
    if pair_mismatches:
        reasons.append(f"paired anchor identity mismatch: {len(pair_mismatches)}; examples={pair_mismatches[:10]}")
    if len(expected_keys) != 200 or len(identifiers) != 100:
        reasons.append("expected grid is not 100 paired architectures / 200 keys")
    reconstruction_rows: list[dict] = []
    if not reasons and len(successes) == 200 and paired_count == 100:
        for key, record in sorted(successes.items()):
            for name in E37_METRIC_NAMES:
                actual = float(record["diagnostic"]["metric_values"][name])
                reference = float(strict_scores[key][name])
                reconstruction_rows.append({
                    "setting": key[0], "model_or_arch_id": key[1], "metric_name": name,
                    "reconstructed": actual, "strict_reference": reference,
                    "absolute_error": abs(actual - reference),
                    "relative_error": abs(actual - reference) / max(abs(reference), 1e-12),
                })
    return {
        "status": "PASS" if not reasons and len(successes) == 200 and paired_count == 100 else "FAIL",
        "reasons": reasons,
        "expected_key_count": len(expected_keys),
        "successful_key_count": len(successes),
        "failed_attempt_count": failed_count,
        "paired_architecture_count": paired_count,
        "pair_mismatches": pair_mismatches,
        "repair_keys": missing,
        "reconstruction_max_abs_error": max_abs,
        "reconstruction_max_relative_error": max_rel,
        "reconstruction_rows": reconstruction_rows,
        "canonical_records": [successes[key] for key in sorted(successes)] if not reasons else [],
    }


def load_strict_scores(path: Path, expected_keys: set[tuple[str, str]]) -> dict:
    scores: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["setting"], row["model_or_arch_id"])
            name = row["metric_name"]
            if key not in expected_keys or name not in E37_METRIC_NAMES:
                continue
            if row["status"] != "ok":
                raise ValueError(f"strict E37 row not ok: {key} {name}")
            metric_scores = scores.setdefault(key, {})
            if name in metric_scores:
                raise ValueError(f"duplicate strict E37 row: {key} {name}")
            metric_scores[name] = _finite_number(float(row["metric_value"]), f"strict {key} {name}")
    if set(scores) != expected_keys or any(set(values) != set(E37_METRIC_NAMES) for values in scores.values()):
        raise ValueError("strict E37 reference is not complete for 200 keys and nine metrics")
    return scores


def write_verification_output(output_dir: Path, result: dict) -> None:
    if output_dir.exists():
        raise FileExistsError(f"verification output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in result.items() if key not in {"canonical_records", "reconstruction_rows"}}
    with (output_dir / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if result["repair_keys"]:
        with (output_dir / "repair_keys.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["setting", "model_or_arch_id"])
            writer.writerows(result["repair_keys"])
    if result["status"] == "PASS":
        with (output_dir / "canonical_numeric.jsonl").open("x", encoding="utf-8") as handle:
            for record in result["canonical_records"]:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        with (output_dir / "reconstruction_errors.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["reconstruction_rows"][0]))
            writer.writeheader()
            writer.writerows(result["reconstruction_rows"])


def verify_attempts(manifest: Path, attempts_dir: Path, strict_csv: Path, output_dir: Path) -> dict:
    try:
        manifest_document = read_json_strict(manifest)
        bundle_manifest = Path(manifest_document["frozen_payload"]["input_paths"]["bundle_manifest"])
        payload, manifest_hash = load_frozen_manifest(manifest, bundle_manifest.parents[1])
        expected = {(row["setting"], row["model_or_arch_id"]) for row in payload["objects"]}
        hashes = {
            "source_sha256": payload["source_sha256"],
            "checkpoint_sha256": {(row["setting"], row["model_or_arch_id"]): row["checkpoint_sha256"] for row in payload["objects"]},
            "manifest_sha256": manifest_hash,
        }
        if strict_csv.resolve() != Path(payload["input_paths"]["strict_csv"]).resolve():
            raise ValueError("strict CSV path differs from frozen input")
        records = [read_json_strict(path) for path in sorted(attempts_dir.glob("*/*/*.json"))]
        strict_scores = load_strict_scores(strict_csv, expected)
        result = verify_records(records, expected_keys=expected, strict_scores=strict_scores, frozen_hashes=hashes)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "FAIL", "reasons": [f"verification input error: {type(exc).__name__}: {exc}"],
            "repair_keys": [], "canonical_records": [],
            "successful_key_count": 0, "paired_architecture_count": 0,
        }
    write_verification_output(output_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--attempts-dir", required=True, type=Path)
    parser.add_argument("--strict-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.output_dir.name):
        raise ValueError("verification directory name must be safe")
    if not args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("verification output must stay inside RobustMetric project")
    result = verify_attempts(args.frozen_manifest, args.attempts_dir, args.strict_csv, args.output_dir)
    print(f"status={result['status']} successes={result['successful_key_count']} pairs={result['paired_architecture_count']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
