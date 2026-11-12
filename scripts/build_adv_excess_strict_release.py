"""Validation gates for an immutable adversarial-excess strict release."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from robust_metric_lab.student_runner import CSV_COLUMNS, UNIQUE_KEY_COLUMNS


METRIC = "feature_adv_excess_centered_unit_l2"
SETTINGS = (
    "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
    "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
)
COUNTS = {"rb40_mature": 40, **{f"nb201_e{e}": 64 for e in (0, 5, 10, 25, 50)},
          **{f"darts_e{e}": 100 for e in (0, 100)}}


class IncompleteGridError(ValueError):
    """A frozen model-setting key is missing or not valid for formal ranking."""


def _row_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in UNIQUE_KEY_COLUMNS)


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite value")
    return result


def validate_new_grid(
    rows: Sequence[Mapping[str, str]], manifest: Mapping[str, object],
) -> None:
    """Match every successful row to the frozen 560 model/checkpoint objects."""
    payload = manifest["frozen_payload"]
    if (payload["metric_name"] != METRIC or payload["expected_rows"] != 560
            or payload["expected_counts"] != COUNTS):
        raise IncompleteGridError("frozen manifest is not the full preregistered grid")
    objects = payload["objects"]
    if set(objects) != set(SETTINGS):
        raise IncompleteGridError("manifest settings mismatch")
    expected = {}
    protocol = payload["protocol"]
    for setting in SETTINGS:
        if len(objects[setting]) != COUNTS[setting]:
            raise IncompleteGridError(f"incomplete manifest setting: {setting}")
        for item in objects[setting]:
            epoch = item["checkpoint_epoch"]
            key = (
                setting, str(item["model_or_arch_id"]),
                "" if epoch is None else str(epoch), METRIC,
                str(protocol["seed"]), str(protocol["sample_offset"]),
                str(protocol["num_samples"]),
            )
            if key in expected:
                raise IncompleteGridError(f"duplicate frozen object: {key}")
            expected[key] = item
    if len(expected) != 560 or len(rows) != 560:
        raise IncompleteGridError(f"row count {len(rows)}/560")
    seen = set()
    counts = Counter()
    for row in rows:
        try:
            key = _row_key(row)
            if key not in expected or key in seen:
                raise IncompleteGridError(f"missing/duplicate/unknown key: {key}")
            seen.add(key)
            setting = row["setting"]
            item = expected[key]
            counts[setting] += 1
            benchmark = "rb40" if setting == "rb40_mature" else setting.split("_e")[0]
            if row["benchmark"] != benchmark or row["direction"] != "lower":
                raise ValueError("wrong benchmark or direction")
            if row["status"] != "ok" or row["error"]:
                raise ValueError("failed row")
            _finite(row["metric_value"])
            if _finite(row["target_value"]) != _finite(item["target_value"]):
                raise ValueError("changed target")
            if row["checkpoint_path"] != item["checkpoint_path"]:
                raise ValueError("changed checkpoint")
            if _finite(row["epsilon"]) != _finite(protocol["epsilon"]):
                raise ValueError("changed epsilon")
        except (KeyError, TypeError, ValueError) as exc:
            raise IncompleteGridError(f"invalid raw row: {exc}") from exc
    if seen != set(expected) or counts != COUNTS:
        raise IncompleteGridError("frozen model-setting grid is incomplete")


def append_eligibility(historical: Sequence[Mapping[str, object]]) -> list[dict]:
    """Append one preregistered record without altering the repaired TRNAS overlay."""
    existing = [dict(record) for record in historical]
    names = [record["metric_name"] for record in existing]
    if METRIC in names or len(names) != len(set(names)):
        raise ValueError("duplicate eligibility metric")
    existing.append({
        "metric_name": METRIC,
        "eligibility": "cross_space",
        "supported_settings": list(SETTINGS),
        "reason": (
            "Pre-registered adversarial-minus-deterministic-random centered-unit "
            "L2 feature drift uses one definition in all three spaces."
        ),
    })
    return existing


def validate_old_grid(
    rows: Sequence[Mapping[str, str]], *, expected_names: set[str],
) -> None:
    """Keep all 85 historical metric grids, including excluded raw evidence."""
    if len(rows) != 47600 or len(expected_names) != 85 or METRIC in expected_names:
        raise IncompleteGridError("historical release is not the 47,600-row baseline")
    seen = set()
    counts = Counter()
    for row in rows:
        try:
            key = _row_key(row)
            if key in seen or row["metric_name"] not in expected_names:
                raise ValueError(f"duplicate or unknown historical key: {key}")
            seen.add(key)
            counts[(row["metric_name"], row["setting"])] += 1
        except (KeyError, ValueError) as exc:
            raise IncompleteGridError(f"invalid historical raw row: {exc}") from exc
    if len(seen) != 47600:
        raise IncompleteGridError("historical raw keys are not unique")
    for name in expected_names:
        for setting, count in COUNTS.items():
            if counts[(name, setting)] != count:
                raise IncompleteGridError(f"historical grid missing {name}/{setting}")


def validate_historical_eligibility(
    records: Sequence[Mapping[str, object]],
) -> None:
    by_name = {str(record["metric_name"]): record for record in records}
    trnas = {"trnas_activation_psi", "trnas_consistency_phi", "trnas_r_score"}
    if not trnas.issubset(by_name) or any(
        by_name[name]["eligibility"] != "cross_space" for name in trnas
    ):
        raise IncompleteGridError("historical repaired TRNAS eligibility changed")
    excluded_racl = {
        "racl_arch_lipschitz_log_upper",
        "racl_fixed_network_log_upper_proxy",
    }
    if not excluded_racl.issubset(by_name) or any(
        by_name[name]["eligibility"] != "space_specific" for name in excluded_racl
    ):
        raise IncompleteGridError("historical excluded RACL eligibility was relaxed")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise IncompleteGridError(f"wrong raw schema: {path}")
        return list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_frozen_manifest(document: Mapping[str, object], project: Path) -> None:
    payload = document["frozen_payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != document["frozen_sha256"]:
        raise IncompleteGridError("new run manifest digest mismatch")
    for relative, digest in payload["source_sha256"].items():
        if _sha256(project / relative) != digest:
            raise IncompleteGridError(f"source changed since run: {relative}")
    for relative, digest in payload["bundle_manifest_sha256"].items():
        if _sha256(project / relative) != digest:
            raise IncompleteGridError(f"benchmark manifest changed: {relative}")


def _write_csv_create_only(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_release(old_release: Path, new_run: Path, output_dir: Path,
                  project: Path, bundle: Path) -> dict:
    """Build a fresh strict release or leave a visibly failed, non-PASS directory."""
    old_release, new_run = old_release.resolve(), new_run.resolve()
    output_dir, project, bundle = output_dir.resolve(), project.resolve(), bundle.resolve()
    old_verification = json.loads((old_release / "release_verification.json").read_text())
    required_old = {
        "status": "PASS", "raw_rows": 47600, "unique_raw_keys": 47600,
        "ranked_metrics": 83, "excluded_metrics": 2,
    }
    if any(old_verification.get(key) != value for key, value in required_old.items()):
        raise IncompleteGridError("historical repaired TRNAS release is not verified")
    old_eligibility = json.loads((old_release / "metric_eligibility.json").read_text())
    if len(old_eligibility) != 85:
        raise IncompleteGridError("historical eligibility count is not 85")
    validate_historical_eligibility(old_eligibility)
    by_name = {record["metric_name"]: record for record in old_eligibility}
    old_raw_path = old_release / "metric_values_merged.csv"
    old_hash = _sha256(old_raw_path)
    old_rows = _read_csv(old_raw_path)
    validate_old_grid(old_rows, expected_names=set(by_name))
    status = (new_run / "STATUS.env").read_text(encoding="utf-8")
    if "state=RAW_GRID_OK" not in status:
        raise IncompleteGridError("new run has no successful 560-row gate")
    manifest = json.loads((new_run / "run_manifest.json").read_text(encoding="utf-8"))
    _check_frozen_manifest(manifest, project)
    new_rows = _read_csv(new_run / "metric_values.csv")
    validate_new_grid(new_rows, manifest)
    eligibility = append_eligibility(old_eligibility)
    if eligibility[:-1] != old_eligibility:
        raise IncompleteGridError("historical eligibility was changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "STATUS.env").write_text("state=BUILDING\n", encoding="utf-8")
    try:
        merged_path = output_dir / "metric_values_merged.csv"
        _write_csv_create_only(merged_path, old_rows + new_rows)
        eligibility_path = output_dir / "metric_eligibility.json"
        with eligibility_path.open("x", encoding="utf-8") as handle:
            json.dump(eligibility, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        common = [
            "--input", str(merged_path),
            "--rb40-manifest", str(bundle / "manifests/rb40_models.csv"),
            "--nb201-manifest", str(bundle / "manifests/nb201_64_architectures.csv"),
            "--darts-manifest", str(bundle / "manifests/darts100_architectures.csv"),
            "--benchmark-root", str(bundle),
            "--eligibility", str(eligibility_path),
        ]
        ranking_dir = output_dir / "ranking"
        with (output_dir / "ranking.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(project / "scripts/build_strict_metric_ranking.py"),
                 *common, "--output-dir", str(ranking_dir)],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        verification_path = output_dir / "release_verification.json"
        with (output_dir / "verification.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(project / "scripts/verify_strict_release.py"),
                 *common, "--ranking", str(ranking_dir / "strict_ranking.csv"),
                 "--output", str(verification_path)],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        required_new = {
            "status": "PASS", "raw_rows": 48160, "unique_raw_keys": 48160,
            "ranked_metrics": 84, "excluded_metrics": 2,
        }
        if any(verification.get(key) != value for key, value in required_new.items()):
            raise IncompleteGridError(f"strict verifier did not PASS: {verification}")
        if _sha256(old_raw_path) != old_hash:
            raise IncompleteGridError("historical raw file changed during release")
        provenance = {
            "old_release": str(old_release),
            "old_raw_sha256": old_hash,
            "new_run": str(new_run),
            "new_run_manifest_sha256": _sha256(new_run / "run_manifest.json"),
            "new_raw_sha256": _sha256(new_run / "metric_values.csv"),
            "merged_raw_sha256": _sha256(merged_path),
            "verification_sha256": _sha256(verification_path),
        }
        with (output_dir / "provenance.json").open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with (output_dir / "STATUS.env").open("a", encoding="utf-8") as handle:
            handle.write("state=PASS\n")
        return verification
    except BaseException:
        with (output_dir / "STATUS.env").open("a", encoding="utf-8") as handle:
            handle.write("state=FAILED\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_release(args.old_release, args.new_run, args.output_dir,
                        args.project, args.bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
