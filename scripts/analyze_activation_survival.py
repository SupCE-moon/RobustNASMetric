"""Complete-grid, architecture-paired comparison with pinned TRNAS Phi.

The 560-row main grid has already informed candidate selection. Its bootstrap
interval is descriptive; independent confirmation requires a separate package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


COUNTS = {
    "rb40_mature": 40,
    "nb201_e0": 64,
    "nb201_e5": 64,
    "nb201_e10": 64,
    "nb201_e25": 64,
    "nb201_e50": 64,
    "darts_e0": 100,
    "darts_e100": 100,
}
CANDIDATE = "feature_activation_sign_survival_sum"
REFERENCE = "trnas_consistency_phi"


def _space(setting: str) -> str:
    return "rb40" if setting == "rb40_mature" else setting.split("_e", 1)[0]


def _epoch(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("invalid checkpoint epoch")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid checkpoint epoch") from exc
    if str(value) not in {str(result), f"{result}.0"}:
        raise ValueError("invalid checkpoint epoch")
    return result


def _finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _indexed(rows: Iterable[Mapping], metric: str) -> dict[tuple[str, str, int | None], dict]:
    indexed = {}
    for row in rows:
        if row.get("metric_name") != metric or row.get("status") != "ok":
            raise ValueError(f"unexpected metric or status for {metric}")
        setting = str(row.get("setting", ""))
        if setting not in COUNTS:
            raise ValueError(f"unknown setting: {setting}")
        identifier = str(row.get("model_or_arch_id", ""))
        if not identifier:
            raise ValueError("empty model or architecture ID")
        epoch = _epoch(row.get("checkpoint_epoch"))
        expected_epoch = None if setting == "rb40_mature" else int(setting.rsplit("e", 1)[1])
        if epoch != expected_epoch:
            raise ValueError(f"checkpoint epoch differs from setting: {setting}")
        key = (setting, identifier, epoch)
        if key in indexed:
            raise ValueError(f"duplicate {metric} key: {key}")
        indexed[key] = {
            "score": _finite(row.get("metric_value"), "metric_value"),
            "target": _finite(row.get("target_value"), "target_value"),
        }
    if len(indexed) != 560:
        raise ValueError(f"missing complete 560 {metric} rows: {len(indexed)}")
    for setting, count in COUNTS.items():
        if sum(key[0] == setting for key in indexed) != count:
            raise ValueError(f"missing {setting} rows")
    for space in ("nb201", "darts"):
        stages = [setting for setting in COUNTS if _space(setting) == space]
        id_sets = [{key[1] for key in indexed if key[0] == setting} for setting in stages]
        if any(ids != id_sets[0] for ids in id_sets[1:]):
            raise ValueError(f"missing shared {space} architecture across stages")
    return indexed


def _paired(candidate: Iterable[Mapping], phi: Iterable[Mapping]) -> dict:
    ours = _indexed(candidate, CANDIDATE)
    reference = _indexed(phi, REFERENCE)
    if set(ours) != set(reference):
        raise ValueError("missing paired Phi or candidate keys")
    if any(ours[key]["target"] != reference[key]["target"] for key in ours):
        raise ValueError("target differs between candidate and Phi")
    return {key: (ours[key]["score"], reference[key]["score"], ours[key]["target"])
            for key in ours}


def _rho(values: list[tuple[float, float, float]], column: int) -> float:
    statistic = float(spearmanr([row[column] for row in values],
                                [row[2] for row in values]).statistic)
    if not math.isfinite(statistic):
        raise ValueError("nonfinite Spearman correlation")
    return statistic


def _deltas(paired: Mapping, sampled_ids: Mapping[str, list[str]] | None = None) -> dict[str, float]:
    deltas = {}
    for setting in COUNTS:
        space = _space(setting)
        ids = (sampled_ids[space] if sampled_ids is not None
               else sorted(key[1] for key in paired if key[0] == setting))
        epoch = None if space == "rb40" else int(setting.rsplit("e", 1)[1])
        values = [paired[(setting, identifier, epoch)] for identifier in ids]
        deltas[setting] = _rho(values, 0) - _rho(values, 1)
    return deltas


def main_gate(deltas: Mapping[str, float], *, ci_low: float, valid_rows: int) -> bool:
    if set(deltas) != set(COUNTS):
        raise ValueError("eight settings required")
    values = [_finite(deltas[setting], "setting delta") for setting in COUNTS]
    return valid_rows == 560 and math.fsum(values) / 8 > 0 and _finite(ci_low, "CI low") > 0


def compare_complete_grid(candidate: Iterable[Mapping], phi: Iterable[Mapping]) -> dict:
    paired = _paired(candidate, phi)
    deltas = _deltas(paired)
    space_deltas = {
        space: math.fsum(deltas[setting] for setting in COUNTS if _space(setting) == space)
        / sum(_space(setting) == space for setting in COUNTS)
        for space in ("rb40", "nb201", "darts")
    }
    return {
        "valid_rows": len(paired),
        "setting_deltas": deltas,
        "eight_setting_delta": math.fsum(deltas.values()) / 8,
        "three_space_equal_delta": math.fsum(space_deltas.values()) / 3,
        "space_deltas": space_deltas,
    }


def architecture_bootstrap(
    candidate: Iterable[Mapping], phi: Iterable[Mapping], *, draws: int = 3000,
    seed: int = 20261027, return_sampled_ids: bool = False,
) -> dict | list[dict]:
    """Resample IDs once per space; reuse those IDs for every stage of that space."""
    if draws < 1:
        raise ValueError("draws must be positive")
    paired = _paired(candidate, phi)
    ids_by_space = {
        space: sorted(key[1] for key in paired if key[0] == next(
            setting for setting in COUNTS if _space(setting) == space))
        for space in ("rb40", "nb201", "darts")
    }
    generator = np.random.default_rng(seed)
    results = []
    for _ in range(draws):
        sampled = {
            space: [ids[index] for index in generator.integers(0, len(ids), size=len(ids))]
            for space, ids in ids_by_space.items()
        }
        deltas = _deltas(paired, sampled)
        result = {"eight_setting_delta": math.fsum(deltas.values()) / 8,
                  "setting_deltas": deltas}
        if return_sampled_ids:
            result["sampled_ids"] = sampled
        results.append(result)
    if return_sampled_ids:
        return results
    interval = np.percentile([row["eight_setting_delta"] for row in results], [2.5, 97.5])
    return {"draws": draws, "seed": seed, "ci_low": float(interval[0]),
            "ci_high": float(interval[1])}


def _read_csv(path: Path, metric: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("metric_name") == metric]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--strict-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    candidate = _read_csv(args.candidate_csv, CANDIDATE)
    phi = _read_csv(args.strict_csv, REFERENCE)
    result = compare_complete_grid(candidate, phi)
    result["bootstrap"] = architecture_bootstrap(candidate, phi)
    result["main_gate_passed"] = main_gate(
        result["setting_deltas"], ci_low=result["bootstrap"]["ci_low"],
        valid_rows=result["valid_rows"])
    result["evidence_role"] = "development / repeated-candidate"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output_json), "main_gate_passed": result["main_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
