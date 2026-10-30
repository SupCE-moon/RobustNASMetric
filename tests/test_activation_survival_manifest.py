from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.activation_survival_manifest import (
    build_payload, expected_main_keys, require_project_path, write_or_verify_manifest,
)


def item(identifier, epoch, target=0.5):
    return SimpleNamespace(
        model_or_arch_id=str(identifier), checkpoint_epoch=epoch,
        target_value=target, checkpoint_path=Path("unused.pt"), metadata={},
    )


def full_grid():
    return {
        "rb40_mature": [item(i, None) for i in range(40)],
        **{f"nb201_e{epoch}": [item(i, epoch) for i in range(64)]
           for epoch in (0, 5, 10, 25, 50)},
        **{f"darts_e{epoch}": [item(i, epoch) for i in range(100)]
           for epoch in (0, 100)},
    }


def test_exact_560_grid_has_eight_settings():
    assert len(expected_main_keys(full_grid())) == 560


@pytest.mark.parametrize("setting", ["rb40_mature", "nb201_e25", "darts_e100"])
def test_missing_one_key_fails(setting):
    grid = full_grid()
    grid[setting].pop()
    with pytest.raises(ValueError, match="missing|count"):
        expected_main_keys(grid)


def test_duplicate_and_stage_mismatch_fail():
    grid = full_grid()
    grid["nb201_e0"][1] = item(0, 0)
    with pytest.raises(ValueError, match="duplicate"):
        expected_main_keys(grid)
    grid = full_grid()
    grid["darts_e100"][0] = item(101, 100)
    with pytest.raises(ValueError, match="stage|unequal"):
        expected_main_keys(grid)


def test_nonfinite_target_and_changed_stage_target_fail():
    grid = full_grid()
    grid["rb40_mature"][0] = item(0, None, float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        expected_main_keys(grid)
    grid = full_grid()
    grid["nb201_e50"][0] = item(0, 50, 0.6)
    with pytest.raises(ValueError, match="target"):
        expected_main_keys(grid)


def test_write_once_manifest_refuses_changed_payload(tmp_path):
    path = tmp_path / "frozen.json"
    payload = {"objects": [{"setting": "rb40_mature", "model_or_arch_id": "a"}]}
    digest = write_or_verify_manifest(path, payload)
    assert write_or_verify_manifest(path, payload) == digest
    with pytest.raises(ValueError, match="manifest mismatch"):
        write_or_verify_manifest(path, {"objects": []})


def test_build_payload_hashes_all_rb_artifacts_and_refuses_missing(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    grid = full_grid()
    for values in grid.values():
        for value in values:
            value.checkpoint_path = first
    grid["rb40_mature"][0].metadata = {"artifact_paths": [str(first), str(second)]}
    source = tmp_path / "source.py"
    source.write_bytes(b"source")
    bundle = tmp_path / "bundle.csv"
    bundle.write_bytes(b"bundle")
    sample = tmp_path / "test_batch"
    sample.write_bytes(b"sample")
    strict = tmp_path / "strict.csv"
    strict.write_bytes(b"strict")
    payload = build_payload(
        grid, source_paths={"src/robust_metric_lab/activation_sign_survival.py": source},
        bundle_manifests={"rb40": bundle, "nb201": bundle, "darts": bundle},
        sample_paths={"test_batch": sample}, strict_csv=strict,
        environment={"python": "test"},
    )
    row = next(x for x in payload["objects"]
               if x["setting"] == "rb40_mature" and x["model_or_arch_id"] == "0")
    assert len(row["artifact_sha256"]) == 2
    assert payload["input_paths"]["sample"]["test_batch"] == str(sample.resolve())
    second.unlink()
    with pytest.raises(FileNotFoundError):
        build_payload(
            grid, source_paths={"src/robust_metric_lab/activation_sign_survival.py": source},
            bundle_manifests={"rb40": bundle, "nb201": bundle, "darts": bundle},
            sample_paths={"test_batch": sample}, strict_csv=strict,
            environment={"python": "test"},
        )


def test_freeze_rejects_paths_outside_permitted_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    assert require_project_path(project / "bundle", project) == (project / "bundle").resolve()
    with pytest.raises(ValueError, match="outside permitted project"):
        require_project_path(tmp_path / "other", project)
