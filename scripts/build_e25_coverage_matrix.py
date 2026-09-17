"""Build the registry- and protocol-frozen E25+ metric coverage matrix."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

# Permit direct invocation from a source checkout as required by the CLI.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from robust_metric_lab.metric_registry import METRIC_SPECS, METRIC_BY_NAME
from robust_metric_lab.student_benchmark import BENCHMARK_SETTINGS, SETTING_BY_KEY

REQUIRED = {"setting", "metric_name", "model_or_arch_id", "checkpoint_epoch", "status"}
PROTOCOL = ("seed", "sample_offset", "num_samples")


def coverage_state(ok_rows: int, failed_rows: int, expected_rows: int) -> str:
    if ok_rows < 0 or failed_rows < 0 or expected_rows <= 0 or ok_rows > expected_rows:
        raise ValueError("invalid coverage counts")
    if ok_rows == 0 and failed_rows == 0:
        return "missing"
    if failed_rows > 0:
        return "failed"
    if ok_rows == expected_rows:
        return "complete"
    return "partial"


def _canonical(value: str, label: str, source: Path, line: int) -> str:
    if value is None or not value or value != value.strip():
        raise ValueError(f"noncanonical {label} at {source}:{line}")
    return value


def _epoch(value: str, setting_key: str, source: Path, line: int) -> int | None:
    value = "" if value is None else value.strip()
    if value == "":
        result = None
    else:
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint_epoch at {source}:{line}") from exc
    expected = SETTING_BY_KEY[setting_key].checkpoint_epoch
    if result != expected:
        raise ValueError(f"checkpoint_epoch mismatch at {source}:{line}")
    return result


def build_coverage_matrix(inputs: Iterable[str | Path]) -> OrderedDict[str, dict[str, object]]:
    metrics = OrderedDict()
    for spec in METRIC_SPECS:
        row: dict[str, object] = {"metric_name": spec.name, "attribution": spec.attribution,
                                  "direction": spec.direction, "reference": spec.reference,
                                  "protocol_key_mode": "full"}
        for setting in BENCHMARK_SETTINGS:
            row[f"{setting.key}_state"] = "missing"
            row[f"{setting.key}_expected_count"] = setting.expected_rows
            row[f"{setting.key}_valid_count"] = 0
            row[f"{setting.key}_failed_count"] = 0
        metrics[spec.name] = row

    # (setting, metric) -> id -> status; this makes mixed status for one ID fail closed.
    observations: dict[tuple[str, str], dict[str, str]] = {}
    seen_keys: set[tuple[object, ...]] = set()
    degraded = False
    for input_name in inputs:
        source = Path(input_name)
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = sorted(REQUIRED - fields)
            if missing:
                raise ValueError(f"missing required columns in {source}: {', '.join(missing)}")
            full_key = all(field in fields for field in PROTOCOL)
            if not full_key:
                degraded = True
            for line, raw in enumerate(reader, 2):
                setting = _canonical(raw.get("setting"), "setting", source, line)
                metric = _canonical(raw.get("metric_name"), "metric_name", source, line)
                model_id = _canonical(raw.get("model_or_arch_id"), "model_or_arch_id", source, line)
                if setting not in SETTING_BY_KEY:
                    raise ValueError(f"unknown setting at {source}:{line}: {setting}")
                if metric not in METRIC_BY_NAME:
                    raise ValueError(f"unknown metric at {source}:{line}: {metric}")
                epoch = _epoch(raw.get("checkpoint_epoch"), setting, source, line)
                status = raw.get("status")
                if status is None or status == "" or status != status.strip():
                    raise ValueError(f"empty/noncanonical status at {source}:{line}")
                if full_key:
                    protocol = tuple(raw.get(field) for field in PROTOCOL)
                    if any(value is None or value == "" or value != value.strip() for value in protocol):
                        raise ValueError(f"invalid protocol key at {source}:{line}")
                    key = (setting, model_id, epoch, metric, *protocol)
                else:
                    key = (setting, model_id, epoch, metric)
                if key in seen_keys:
                    raise ValueError(f"duplicate unique key at {source}:{line}")
                seen_keys.add(key)
                state = "ok" if status == "ok" else "failed"
                by_id = observations.setdefault((setting, metric), {})
                # Any failed observation makes that model ID fail closed.
                if by_id.get(model_id) == "failed" or state == "failed":
                    by_id[model_id] = "failed"
                else:
                    by_id[model_id] = "ok"

    mode = "degraded_prefix" if degraded else "full"
    for row in metrics.values():
        row["protocol_key_mode"] = mode
    for (setting, metric), ids in observations.items():
        expected = SETTING_BY_KEY[setting].expected_rows
        ok = sum(value == "ok" for value in ids.values())
        failed = sum(value == "failed" for value in ids.values())
        if len(ids) > expected:
            raise ValueError(f"excess model IDs for {setting}/{metric}: {len(ids)} > {expected}")
        row = metrics[metric]
        row[f"{setting}_state"] = coverage_state(ok, failed, expected)
        row[f"{setting}_valid_count"] = ok
        row[f"{setting}_failed_count"] = failed
    return metrics


def write_matrix(matrix: OrderedDict[str, dict[str, object]], output: str | Path) -> None:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric_name", "attribution", "direction", "reference", "protocol_key_mode"]
    for setting in BENCHMARK_SETTINGS:
        fields.extend([f"{setting.key}_state", f"{setting.key}_expected_count", f"{setting.key}_valid_count", f"{setting.key}_failed_count"])
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(matrix.values())
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(destination)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="CSV", help="metric_values.csv input (repeatable)")
    parser.add_argument("--output", required=True, metavar="CSV", help="output e25_plus_coverage.csv path")
    args = parser.parse_args(argv)
    try:
        write_matrix(build_coverage_matrix(args.input), args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
