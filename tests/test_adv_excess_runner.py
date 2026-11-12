from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest
import torch

from robust_metric_lab.metric_registry import METRIC_BY_NAME
from robust_metric_lab.student_benchmark import BenchmarkManifestItem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_adv_excess_student as runner  # noqa: E402
from adv_excess_run_manifest import expected_counts, write_or_verify_manifest  # noqa: E402


METRIC = "feature_adv_excess_centered_unit_l2"


def test_registry_and_eligibility_describe_one_lower_cross_space_metric():
    eligibility = json.loads((ROOT / "configs/metric_eligibility.json").read_text(encoding="utf-8"))
    records = [record for record in eligibility if record["metric_name"] == METRIC]
    assert METRIC_BY_NAME[METRIC].direction == "lower"
    assert len(records) == 1
    assert records[0]["eligibility"] == "cross_space"
    assert len(records[0]["supported_settings"]) == 8


def test_runner_rejects_changed_sample_protocol(tmp_path: Path):
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--benchmark", "nb201", "--epoch", "0", "--sample-offset", "1",
                "--bundle-root", str(tmp_path), "--data-root", str(tmp_path),
                "--output-dir", str(tmp_path), "--run-id", "adv_excess_smoke",
            ]
        )


def test_one_item_row_is_frozen_and_resume_does_not_recompute(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    item = BenchmarkManifestItem(
        benchmark="rb40", model_or_arch_id="toy-1", checkpoint_epoch=None,
        checkpoint_path=checkpoint, target_value=0.375,
    )
    calls = {"load": 0, "metric": 0}

    def load_model(_item):
        calls["load"] += 1
        return torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.Flatten(),
                                   torch.nn.Linear(2 * 2 * 2, 2))

    def load_samples(_root, _device):
        return torch.full((64, 3, 2, 2), 0.5), torch.zeros(64, dtype=torch.long)

    def compute(*_args, **_kwargs):
        calls["metric"] += 1
        return runner.AdvExcessResult(value=0.125, diagnostics={"samples": 64})

    monkeypatch.setattr(runner, "_setting_and_items", lambda _args: ("rb40_mature", [item], load_model))
    monkeypatch.setattr(runner, "load_cifar10_samples", load_samples)
    monkeypatch.setattr(runner, "compute_adv_excess_batched", compute)
    argv = [
        "--benchmark", "rb40", "--bundle-root", str(tmp_path),
        "--data-root", str(tmp_path), "--output-dir", str(tmp_path),
        "--run-id", "adv_excess_smoke", "--device", "cpu",
    ]
    assert runner.main(argv) == 0
    assert runner.main(argv) == 0
    assert calls == {"load": 1, "metric": 1}
    with (tmp_path / "adv_excess_smoke" / "metric_values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["setting"] == "rb40_mature"
    assert row["target_value"] == "0.375"
    assert row["metric_name"] == METRIC
    assert row["direction"] == "lower"
    assert row["metric_value"] == "0.125"
    assert row["status"] == "ok"
    assert (row["seed"], row["sample_offset"], row["num_samples"]) == (
        "20260808", "0", "64"
    )
    diagnostics = list((tmp_path / "adv_excess_smoke" / "diagnostics").glob("*.json"))
    assert len(diagnostics) == 1
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert payload["model_or_arch_id"] == "toy-1"
    assert payload["samples"] == 64


def test_launcher_modes_preserve_exact_eight_setting_grid():
    assert expected_counts("smoke") == {
        "rb40_mature": 4,
        "nb201_e0": 4, "nb201_e5": 4, "nb201_e10": 4,
        "nb201_e25": 4, "nb201_e50": 4,
        "darts_e0": 4, "darts_e100": 4,
    }
    full = expected_counts("full")
    assert full == {
        "rb40_mature": 40,
        "nb201_e0": 64, "nb201_e5": 64, "nb201_e10": 64,
        "nb201_e25": 64, "nb201_e50": 64,
        "darts_e0": 100, "darts_e100": 100,
    }
    assert sum(expected_counts("smoke").values()) == 32
    assert sum(full.values()) == 560


def test_manifest_mismatch_aborts_without_overwriting_first_manifest(tmp_path: Path):
    path = tmp_path / "run_manifest.json"
    first = {"mode": "smoke", "objects": ["a", "b"], "source_sha256": "123"}
    write_or_verify_manifest(path, first)
    original_bytes = path.read_bytes()
    write_or_verify_manifest(path, dict(first))
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="manifest mismatch"):
        write_or_verify_manifest(path, {**first, "source_sha256": "changed"})
    assert path.read_bytes() == original_bytes
