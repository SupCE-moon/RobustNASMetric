import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_e25_coverage_matrix import coverage_state, build_coverage_matrix, write_matrix
from robust_metric_lab.metric_registry import METRIC_SPECS
from robust_metric_lab.student_benchmark import BENCHMARK_SETTINGS


def test_coverage_state_rules_and_invalid_counts():
    assert coverage_state(0, 0, 3) == "missing"
    assert coverage_state(1, 0, 3) == "partial"
    assert coverage_state(3, 0, 3) == "complete"
    assert coverage_state(1, 2, 3) == "failed"
    for args in [(-1, 0, 3), (0, -1, 3), (0, 0, 0), (4, 0, 3)]:
        with pytest.raises(ValueError):
            coverage_state(*args)


def _write(path: Path, rows, fields=None):
    fields = fields or ["setting", "metric_name", "model_or_arch_id", "checkpoint_epoch", "status", "seed", "sample_offset", "num_samples"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_registry_and_setting_order_and_counts(tmp_path):
    name = METRIC_SPECS[0].name
    p = tmp_path / "misleading_e25_space.csv"
    _write(p, [
        {"setting": "rb40_mature", "metric_name": name, "model_or_arch_id": "m1", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"},
        {"setting": "rb40_mature", "metric_name": name, "model_or_arch_id": "m2", "checkpoint_epoch": "", "status": "failed", "seed": "1", "sample_offset": "0", "num_samples": "64"},
    ])
    out = build_coverage_matrix([p])
    assert list(out) == [s.name for s in METRIC_SPECS]
    row = out[name]
    assert row["rb40_mature_state"] == "failed"
    assert row["rb40_mature_expected_count"] == 40
    assert row["rb40_mature_valid_count"] == 1
    assert row["rb40_mature_failed_count"] == 1
    assert row["attribution"] == METRIC_SPECS[0].attribution
    assert row["reference"] == METRIC_SPECS[0].reference


def test_unknown_bad_duplicate_and_empty_status_fail(tmp_path):
    base = {"setting": "rb40_mature", "metric_name": METRIC_SPECS[0].name, "model_or_arch_id": "m1", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"}
    for change in ({"metric_name": "unknown"}, {"setting": " rb40_mature"}, {"status": ""}):
        p = tmp_path / "bad.csv"; row = dict(base); row.update(change); _write(p, [row])
        with pytest.raises(ValueError): build_coverage_matrix([p])
    p1 = tmp_path / "a.csv"; p2 = tmp_path / "b.csv"
    _write(p1, [base]); _write(p2, [base])
    with pytest.raises(ValueError, match="duplicate"):
        build_coverage_matrix([p1, p2])


def test_cli_success_and_existing_output_rejected(tmp_path):
    p = tmp_path / "input.csv"; out = tmp_path / "nested" / "e25_plus_coverage.csv"
    _write(p, [{"setting": "rb40_mature", "metric_name": METRIC_SPECS[0].name, "model_or_arch_id": "m1", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"}])
    script = Path(__file__).parents[1] / "scripts" / "build_e25_coverage_matrix.py"
    ok = subprocess.run([sys.executable, str(script), "--input", str(p), "--output", str(out)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert out.exists()
    bad = subprocess.run([sys.executable, str(script), "--input", str(p), "--output", str(out)], capture_output=True, text=True)
    assert bad.returncode != 0


def test_degraded_mode_and_full_degraded_conflict(tmp_path):
    base = {"setting": "rb40_mature", "metric_name": METRIC_SPECS[0].name, "model_or_arch_id": "m1", "checkpoint_epoch": "", "status": "ok"}
    degraded = tmp_path / "degraded.csv"; _write(degraded, [base], fields=list(base))
    assert build_coverage_matrix([degraded])[METRIC_SPECS[0].name]["protocol_key_mode"] == "degraded_prefix"
    full = dict(base, seed="1", sample_offset="0", num_samples="64")
    complete = tmp_path / "full.csv"; _write(complete, [full])
    with pytest.raises(ValueError, match="first|duplicate|conflict"):
        build_coverage_matrix([degraded, complete])


def test_setting_global_excess_across_metrics_and_epoch_whitespace(tmp_path):
    rows = []
    for i in range(41):
        rows.append({"setting": "rb40_mature", "metric_name": METRIC_SPECS[i % 2].name, "model_or_arch_id": f"m{i}", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"})
    p = tmp_path / "excess.csv"; _write(p, rows)
    with pytest.raises(ValueError, match="excess"):
        build_coverage_matrix([p])
    bad = dict(rows[0], checkpoint_epoch=" ")
    p = tmp_path / "space.csv"; _write(p, [bad])
    with pytest.raises(ValueError, match="canonical"):
        build_coverage_matrix([p])


def test_atomic_publish_failure_cleans_tmp_and_does_not_overwrite(tmp_path, monkeypatch):
    p = tmp_path / "input.csv"
    _write(p, [{"setting": "rb40_mature", "metric_name": METRIC_SPECS[0].name, "model_or_arch_id": "m1", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"}])
    matrix = build_coverage_matrix([p]); out = tmp_path / "out.csv"
    monkeypatch.setattr("scripts.build_e25_coverage_matrix.os.link", lambda *_: (_ for _ in ()).throw(OSError("publish")))
    with pytest.raises(OSError): write_matrix(matrix, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.csv.*.tmp"))
    out.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError): write_matrix(matrix, out)
    assert out.read_text(encoding="utf-8") == "original"


def test_missing_metric_and_mixed_states_and_output_header(tmp_path):
    name = METRIC_SPECS[0].name
    rows = [
        {"setting": "rb40_mature", "metric_name": name, "model_or_arch_id": "a", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"},
        {"setting": "nb201_e0", "metric_name": name, "model_or_arch_id": "b", "checkpoint_epoch": "0", "status": "bad", "seed": "1", "sample_offset": "0", "num_samples": "64"},
        {"setting": "nb201_e5", "metric_name": name, "model_or_arch_id": "c", "checkpoint_epoch": "5", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"},
    ]
    p = tmp_path / "mix.csv"; _write(p, rows)
    matrix = build_coverage_matrix([p]); row = matrix[name]
    assert row["rb40_mature_state"] == "partial"
    assert row["nb201_e0_state"] == "failed"
    assert row["nb201_e5_state"] == "partial"
    absent = matrix[METRIC_SPECS[-1].name]
    assert all(absent[f"{s.key}_state"] == "missing" and absent[f"{s.key}_valid_count"] == 0 for s in BENCHMARK_SETTINGS)
    out = tmp_path / "matrix.csv"; write_matrix(matrix, out)
    with out.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    expected = ["metric_name", "attribution", "direction", "reference", "protocol_key_mode"]
    for s in BENCHMARK_SETTINGS: expected += [f"{s.key}_state", f"{s.key}_expected_count", f"{s.key}_valid_count", f"{s.key}_failed_count"]
    assert header == expected


def test_missing_schema_and_noncanonical_model_fail(tmp_path):
    p = tmp_path / "schema.csv"
    _write(p, [{"setting": "rb40_mature"}], fields=["setting"])
    with pytest.raises(ValueError, match="missing required"):
        build_coverage_matrix([p])
    row = {"setting": "rb40_mature", "metric_name": METRIC_SPECS[0].name, "model_or_arch_id": " m1", "checkpoint_epoch": "", "status": "ok", "seed": "1", "sample_offset": "0", "num_samples": "64"}
    _write(p, [row])
    with pytest.raises(ValueError, match="canonical"):
        build_coverage_matrix([p])
