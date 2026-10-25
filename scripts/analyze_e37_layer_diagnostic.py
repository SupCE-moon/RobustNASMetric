"""Analyze all preregistered E37 layer components only after 200-key PASS."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.e37_metrics import E37_METRIC_NAMES  # noqa: E402


def paired_spearman(e0_values: list[float], e100_values: list[float]) -> float | None:
    if len(e0_values) != 100 or len(e100_values) != 100:
        raise ValueError("paired analysis requires all 100 DARTS architectures")
    if any(not math.isfinite(float(value)) for value in e0_values + e100_values):
        raise ValueError("nonfinite paired component")
    if len(set(e0_values)) == 1 or len(set(e100_values)) == 1:
        return None
    rho = float(spearmanr(e0_values, e100_values).statistic)
    return rho if math.isfinite(rho) else None


def join_controls(identifiers: list[str], controls: list[dict]) -> dict[str, float]:
    ids = [str(value) for value in identifiers]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("paired IDs are not 100 unique architectures")
    output: dict[str, float] = {}
    for row in controls:
        identifier = str(row["model_or_arch_id"])
        if identifier in output:
            raise ValueError(f"duplicate structural control ID: {identifier}")
        value = row.get("log_parameter_count", row.get("log_params"))
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"nonfinite log parameter count: {identifier}")
        output[identifier] = number
    if set(output) != set(ids):
        raise ValueError("static structural control is not a complete 100/100 join")
    return output


def _mean(values: list[float]) -> float:
    if len(values) != 64 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("component requires 64 finite samples")
    return float(np.mean(values))


def _components(record: dict) -> dict[tuple[str, int, str], float]:
    samples = record["diagnostic"]["sample_components"]
    output = {}
    for slot in range(4):
        names = samples[0]["anchors"][slot]
        for name in sorted(names):
            output[("anchor", slot, name)] = _mean([
                sample["anchors"][slot][name] for sample in samples
            ])
    for slot in range(3):
        output[("transition", slot, "flow")] = _mean([
            sample["transitions"][slot]["flow"] for sample in samples
        ])
    return output


def metric_stage_correlations(records: list[dict]) -> list[dict]:
    """Report every frozen E37 score's e0/e100 rank stability, without selection."""
    by_key = {(row["setting"], str(row["model_or_arch_id"])): row for row in records}
    identifiers = sorted({identifier for _, identifier in by_key})
    expected = {(f"darts_e{stage}", identifier) for stage in (0, 100) for identifier in identifiers}
    if len(records) != 200 or len(by_key) != 200 or len(identifiers) != 100 or set(by_key) != expected:
        raise ValueError("nine-score analysis requires exact 200-key paired grid")
    rows: list[dict] = []
    for name in E37_METRIC_NAMES:
        e0 = [float(by_key[("darts_e0", identifier)]["diagnostic"]["metric_values"][name]) for identifier in identifiers]
        e100 = [float(by_key[("darts_e100", identifier)]["diagnostic"]["metric_values"][name]) for identifier in identifiers]
        rows.append({
            "metric_name": name,
            "direction": "lower",
            "pair_count": 100,
            "stage_spearman": paired_spearman(e0, e100),
        })
    return rows


def analyze_records(records: list[dict], controls: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {(row["setting"], str(row["model_or_arch_id"])): row for row in records}
    ids = sorted({identifier for _, identifier in by_key})
    expected = {(f"darts_e{stage}", identifier) for stage in (0, 100) for identifier in ids}
    if len(records) != 200 or len(by_key) != 200 or len(ids) != 100 or set(by_key) != expected:
        raise ValueError("analysis requires exact 200-key PASS grid")
    control = join_controls(ids, controls)
    components = {key: _components(row) for key, row in by_key.items()}
    names = set(components[("darts_e0", ids[0])])
    if any(set(value) != names for value in components.values()):
        raise ValueError("component sets differ across 200 keys")
    paired_rows: list[dict] = []
    correlations: list[dict] = []
    for family, slot, component in sorted(names):
        e0 = [components[("darts_e0", identifier)][(family, slot, component)] for identifier in ids]
        e100 = [components[("darts_e100", identifier)][(family, slot, component)] for identifier in ids]
        changes = [late - early for early, late in zip(e0, e100)]
        early_ranks = rankdata(e0, method="average")
        late_ranks = rankdata(e100, method="average")
        for index, identifier in enumerate(ids):
            paired_rows.append({
                "model_or_arch_id": identifier, "family": family, "slot": slot,
                "component": component, "e0_mean": e0[index], "e100_mean": e100[index],
                "stage_change": changes[index],
                "rank_change_e100_minus_e0": float(late_ranks[index] - early_ranks[index]),
                "log_parameter_count": control[identifier],
            })
        correlations.append({
            "family": family, "slot": slot, "component": component,
            "stage_spearman": paired_spearman(e0, e100),
            "change_log_parameter_spearman": paired_spearman(
                changes, [control[identifier] for identifier in ids]
            ),
            "change_median": float(np.median(changes)),
            "change_q25": float(np.quantile(changes, 0.25)),
            "change_q75": float(np.quantile(changes, 0.75)),
            "pair_count": 100,
            "structure_interpretation": "exploratory_noncausal",
        })
    return paired_rows, correlations


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--control-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.verification_dir.resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("analysis output must stay inside RobustMetric project")
    verification = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise ValueError("analysis requires verification PASS")
    with (directory / "canonical_numeric.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    with args.control_csv.open(newline="", encoding="utf-8-sig") as handle:
        controls = list(csv.DictReader(handle))
    rows, correlations = analyze_records(records, controls)
    metric_correlations = metric_stage_correlations(records)
    _write_csv(directory / "paired_layer.csv", rows)
    _write_csv(directory / "component_correlations.csv", correlations)
    _write_csv(directory / "metric_stage_correlations.csv", metric_correlations)
    print(f"paired_rows={len(rows)} components={len(correlations)} metrics={len(metric_correlations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
