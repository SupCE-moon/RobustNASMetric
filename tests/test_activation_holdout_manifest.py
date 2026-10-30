"""A three-space independent package cannot be sealed with fewer than 371 keys."""

from __future__ import annotations

import copy

import pytest

from scripts.build_activation_holdout_manifest import (
    HOLDOUT_COUNTS, seal_holdout_package, validate_holdout_package,
)


def complete_package() -> dict:
    objects = []
    for setting, count in HOLDOUT_COUNTS.items():
        space = "rb40" if setting == "rb40_mature" else setting.split("_e")[0]
        epoch = None if space == "rb40" else int(setting.rsplit("e", 1)[1])
        for index in range(count):
            checkpoint_hash = f"{index + 1 + (epoch or 0) * 1000:064x}"
            target_checkpoint_hash = (
                f"{index + 1 + 100 * 1000:064x}" if space == "darts" else None
            )
            objects.append({
                "setting": setting, "model_or_arch_id": str(index),
                "checkpoint_epoch": epoch, "signature": f"{space}_structure_{index}",
                "checkpoint_path": f"{space}/{setting}/{index}.pt",
                "checkpoint_sha256": checkpoint_hash,
                "target_value": 0.5, "target_source_id": f"{space}_frozen_target",
                "target_checkpoint_sha256": target_checkpoint_hash,
                "prior_overlap": False,
            })
    return {"objects": objects, "expected_rows": 371,
            "selection_seed": 20261027, "sample_sha256": "f" * 64}


def test_complete_package_has_exact_counts_and_shared_architectures() -> None:
    result = validate_holdout_package(complete_package())
    assert result["expected_rows"] == 371
    assert len(result["objects"]) == 371


@pytest.mark.parametrize("mutation, message", [
    ("missing_rb", "rb40_mature"),
    ("duplicate", "duplicate"),
    ("nb_stage_alias", "NB201"),
    ("darts_target_weight", "target checkpoint"),
    ("darts_e0_self_target", "target checkpoint"),
    ("changed_target", "target differs"),
    ("no_weight_hash", "checkpoint SHA"),
    ("prior_overlap", "prior overlap"),
    ("nonfinite_target", "finite"),
])
def test_incomplete_or_contaminated_package_rejected(mutation: str, message: str) -> None:
    package = copy.deepcopy(complete_package())
    rows = package["objects"]
    if mutation == "missing_rb":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "nb_stage_alias":
        next(row for row in rows if row["setting"] == "nb201_e50")["model_or_arch_id"] = "extra"
    elif mutation == "darts_target_weight":
        next(row for row in rows if row["setting"] == "darts_e100")["target_checkpoint_sha256"] = "a" * 64
    elif mutation == "darts_e0_self_target":
        row = next(row for row in rows if row["setting"] == "darts_e0")
        row["target_checkpoint_sha256"] = row["checkpoint_sha256"]
    elif mutation == "changed_target":
        next(row for row in rows if row["setting"] == "nb201_e50")["target_value"] = 0.6
    elif mutation == "no_weight_hash":
        rows[0]["checkpoint_sha256"] = ""
    elif mutation == "prior_overlap":
        rows[0]["prior_overlap"] = True
    elif mutation == "nonfinite_target":
        rows[0]["target_value"] = float("nan")
    with pytest.raises(ValueError, match=message):
        validate_holdout_package(package)


def test_unverified_package_cannot_be_sealed(tmp_path) -> None:
    with pytest.raises(ValueError, match="verified targets"):
        seal_holdout_package(complete_package(), tmp_path / "sealed.json")


def test_sealed_package_is_canonical_and_write_once(tmp_path) -> None:
    package = complete_package()
    package["source_audit_sha256"] = "a" * 64
    package["target_verification_sha256"] = "b" * 64
    package["protocol_sha256"] = "c" * 64
    package["analysis_sha256"] = "d" * 64
    package["code_sha256"] = "e" * 64
    package["environment_sha256"] = "1" * 64
    package["targets_verified"] = True
    path = tmp_path / "sealed.json"
    sealed = seal_holdout_package(package, path)
    assert sealed["sealed"] is True
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert len(sealed["sha256"]) == 64
    with pytest.raises(FileExistsError):
        seal_holdout_package(package, path)
