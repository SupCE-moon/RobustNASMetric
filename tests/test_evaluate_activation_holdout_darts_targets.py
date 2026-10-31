"""DARTS target evidence must reproduce the existing PGD-20 batch loop."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from scripts.evaluate_activation_holdout_darts_targets import (
    build_target_evidence,
    collect_pgd_correctness,
    write_target_evidence,
)
from scripts.verify_holdout_targets import verify_target_file
from scripts.run_darts_metric_trajectory import pgd_accuracy


def test_per_image_correctness_matches_existing_aggregate_pgd():
    torch.manual_seed(44)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    x = torch.rand(4, 3, 2, 2)
    y = torch.tensor([0, 1, 1, 0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
    torch.manual_seed(400000)
    correct = collect_pgd_correctness(model, loader, "cpu", 8 / 255, 2)
    torch.manual_seed(400000)
    aggregate = pgd_accuracy(model, loader, "cpu", 8 / 255, 2)
    assert len(correct) == 4 and all(type(value) is bool for value in correct)
    assert sum(correct) / 4 == aggregate


def test_evidence_is_exclusive_and_verified_against_all_10000_images(tmp_path):
    correct = [True] * 7000 + [False] * 3000
    record, item = build_target_evidence("1000", "a" * 64, correct)
    source = tmp_path / "target.json"
    digest = write_target_evidence(source, record)
    item["target_source_path"] = "target.json"
    item["target_source_sha256"] = digest
    assert verify_target_file(item, tmp_path) == 0.7
    with pytest.raises(FileExistsError):
        write_target_evidence(source, record)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(source.read_text())["sample_ids"] == list(range(10000))


def test_evidence_rejects_missing_or_nonboolean_image_result():
    with pytest.raises(ValueError, match="10,000"):
        build_target_evidence("1000", "a" * 64, [True] * 9999)
    with pytest.raises(ValueError, match="boolean"):
        build_target_evidence("1000", "a" * 64, [True] * 9999 + [1])
