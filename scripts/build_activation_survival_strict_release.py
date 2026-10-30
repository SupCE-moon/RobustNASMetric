"""Build a new strict release from 560 verified sign-survival rows.

The score source and historical releases are read only. Candidate registration
exists only in this process, so the frozen score-source bytes remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from robust_metric_lab.metric_eligibility import (
    MetricEligibility, MetricEligibilityRecord, SETTING_NAMES,
    load_metric_eligibility, validate_registry,
)
from robust_metric_lab.metric_registry import METRIC_BY_NAME, METRIC_SPECS, MetricSpec
from robust_metric_lab.release import verify_strict_release
from robust_metric_lab.strict_ranking import (
    build_strict_ranking, load_unique_metric_rows, manifest_expectations,
)
from robust_metric_lab.student_analysis import write_analysis_outputs
from robust_metric_lab.student_benchmark import (
    load_darts_items, load_nb201_items, load_rb40_items, resolve_under_root,
)
from robust_metric_lab.student_runner import CSV_COLUMNS
from scripts.activation_survival_manifest import require_project_path, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "feature_activation_sign_survival_sum"
RACL_EXCLUSIONS = frozenset({"racl_arch_lipschitz_log_upper",
                             "racl_fixed_network_log_upper_proxy"})


@contextmanager
def candidate_registration() -> Iterator[None]:
    """Expose one audited score definition to the existing strict validators."""
    if CANDIDATE in METRIC_BY_NAME or any(spec.name == CANDIDATE for spec in METRIC_SPECS):
        raise ValueError("candidate already registered")
    spec = MetricSpec(CANDIDATE, "higher", "Ours preregistered sign survival", "Ours")
    METRIC_SPECS.append(spec)
    METRIC_BY_NAME[CANDIDATE] = spec
    try:
        yield
    finally:
        removed = METRIC_SPECS.pop()
        METRIC_BY_NAME.pop(CANDIDATE)
        if removed is not spec:
            raise RuntimeError("metric registry was modified during strict release")


def append_eligibility(historical: Sequence[MetricEligibilityRecord]) -> list[MetricEligibilityRecord]:
    if any(record.metric_name == CANDIDATE for record in historical):
        raise ValueError("candidate already eligible")
    return [*historical, MetricEligibilityRecord(
        CANDIDATE, MetricEligibility.CROSS_SPACE, SETTING_NAMES,
        "Preregistered Conv/Linear clean-to-FGSM sign-survival call sum on all eight settings.",
    )]


def require_release_inputs(previous: dict, candidate: dict) -> None:
    expected_previous = {"status": "PASS", "raw_rows": 48160,
                         "unique_raw_keys": 48160, "ranked_metrics": 84,
                         "excluded_metrics": 2}
    if any(previous.get(key) != value for key, value in expected_previous.items()):
        raise ValueError("previous strict release is not the pinned 84-metric PASS release")
    expected_candidate = {"status": "PASS", "unique_success_keys": 560,
                          "metric": CANDIDATE}
    if any(candidate.get(key) != value for key, value in expected_candidate.items()):
        raise ValueError("candidate verification is not complete 560-key PASS")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(CSV_COLUMNS) or set(fields) != set(CSV_COLUMNS):
            raise ValueError(f"raw CSV schema differs from frozen runner: {path}")
        return list(reader)


def _write_json_x(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _eligibility_dict(record: MetricEligibilityRecord) -> dict:
    return {"metric_name": record.metric_name, "eligibility": record.eligibility.value,
            "supported_settings": list(record.supported_settings), "reason": record.reason}


def _snapshot_tree(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def build_release(previous_dir: Path, candidate_dir: Path, output_dir: Path,
                  bundle_root: Path) -> dict:
    previous_dir = require_project_path(previous_dir, ROOT)
    candidate_dir = require_project_path(candidate_dir, ROOT)
    output_dir = require_project_path(output_dir, ROOT)
    bundle_root = require_project_path(bundle_root, ROOT)
    if output_dir.exists():
        raise FileExistsError(f"new strict release already exists: {output_dir}")
    previous_snapshot = _snapshot_tree(previous_dir)
    previous_verification = _read_json(previous_dir / "release_verification.json")
    candidate_verification = _read_json(candidate_dir / "release_verification.json")
    require_release_inputs(previous_verification, candidate_verification)
    previous_csv = previous_dir / "metric_values_merged.csv"
    candidate_csv = candidate_dir / "metric_values.csv"
    previous_hash = sha256_file(previous_csv)
    candidate_hash = sha256_file(candidate_csv)
    if candidate_hash != candidate_verification.get("canonical_csv_sha256"):
        raise ValueError("candidate CSV differs from its verification")
    old_rows = _read_raw(previous_csv)
    new_rows = _read_raw(candidate_csv)
    if len(old_rows) != 48160 or len(new_rows) != 560:
        raise ValueError("old or candidate raw row count mismatch")
    if {row["metric_name"] for row in new_rows} != {CANDIDATE}:
        raise ValueError("candidate raw CSV contains another metric")
    historical_eligibility = load_metric_eligibility(previous_dir / "metric_eligibility.json")
    if len(historical_eligibility) != 86:
        raise ValueError("previous eligibility does not contain 86 metrics")
    by_name = {record.metric_name: record for record in historical_eligibility}
    if any(by_name[name].eligibility is not MetricEligibility.SPACE_SPECIFIC
           for name in RACL_EXCLUSIONS):
        raise ValueError("historical RACL exclusion changed")
    if by_name["trnas_consistency_phi"].eligibility is not MetricEligibility.CROSS_SPACE:
        raise ValueError("TRNAS Phi eligibility changed")
    validate_registry(historical_eligibility)
    eligibility = append_eligibility(historical_eligibility)
    with candidate_registration():
        validate_registry(eligibility)
        frame = load_unique_metric_rows([previous_csv, candidate_csv])
        expected = manifest_expectations(
            load_rb40_items(resolve_under_root(bundle_root, "manifests/rb40_models.csv"), bundle_root),
            load_nb201_items(resolve_under_root(bundle_root, "manifests/nb201_64_architectures.csv"), bundle_root),
            load_darts_items(resolve_under_root(bundle_root, "manifests/darts100_architectures.csv"), bundle_root),
        )
        ranking, excluded, analysis = build_strict_ranking(frame, expected, eligibility)
        verification = verify_strict_release(frame, expected, eligibility, ranking)
    required = {"status": "PASS", "raw_rows": 48720, "unique_raw_keys": 48720,
                "ranked_metrics": 85, "excluded_metrics": 2}
    if any(verification.get(key) != value for key, value in required.items()):
        raise ValueError(f"new strict release failed complete-grid gate: {verification}")
    if set(excluded["metric_name"]) != RACL_EXCLUSIONS:
        raise ValueError("RACL exclusion set changed")
    if _snapshot_tree(previous_dir) != previous_snapshot:
        raise ValueError("previous strict release changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "STATUS.env").write_text("state=BUILDING\n", encoding="utf-8")
    try:
        merged_csv = output_dir / "metric_values_merged.csv"
        with merged_csv.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows([*old_rows, *new_rows])
            stream.flush()
            os.fsync(stream.fileno())
        _write_json_x(output_dir / "metric_eligibility.json",
                      [_eligibility_dict(record) for record in eligibility])
        ranking_dir = output_dir / "ranking"
        ranking_dir.mkdir()
        ranking.to_csv(ranking_dir / "strict_ranking.csv", index=False)
        excluded.to_csv(ranking_dir / "excluded_metrics.csv", index=False)
        write_analysis_outputs(analysis, ranking_dir / "analysis")
        with candidate_registration():
            written = load_unique_metric_rows([merged_csv])
            verify_strict_release(written, expected, eligibility,
                                  pd.read_csv(ranking_dir / "strict_ranking.csv"))
        _write_json_x(output_dir / "release_verification.json", verification)
        _write_json_x(output_dir / "provenance.json", {
            "previous_release": str(previous_dir),
            "previous_raw_sha256_before_after": previous_hash,
            "previous_release_file_sha256": previous_snapshot,
            "candidate_release": str(candidate_dir),
            "candidate_raw_sha256": candidate_hash,
            "merged_raw_sha256": sha256_file(merged_csv),
            "builder_sha256": sha256_file(Path(__file__)),
            "candidate_manifest_sha256": candidate_verification["manifest_sha256"],
            "registration": _eligibility_dict(eligibility[-1]),
        })
        if _snapshot_tree(previous_dir) != previous_snapshot:
            raise ValueError("previous strict release changed during release")
        if sha256_file(candidate_csv) != candidate_hash:
            raise ValueError("candidate raw evidence changed during release")
        with (output_dir / "STATUS.env").open("a", encoding="utf-8") as stream:
            stream.write("state=PASS\n")
        return verification
    except BaseException:
        with (output_dir / "STATUS.env").open("a", encoding="utf-8") as stream:
            stream.write("state=FAILED\n")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-release", type=Path, required=True)
    parser.add_argument("--candidate-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_release(args.previous_release, args.candidate_release,
                           args.output_dir, args.bundle_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
