import csv
import json
from pathlib import Path

import pytest

from robust_metric_lab.student_benchmark import (
    load_darts_items,
    load_nb201_items,
    load_rb40_items,
    resolve_under_root,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_resolve_under_root_accepts_nested_artifact(tmp_path: Path):
    expected = tmp_path / "nb201_64" / "checkpoints" / "a.pt"
    assert resolve_under_root(tmp_path, "nb201_64/checkpoints/a.pt") == expected.resolve()


@pytest.mark.parametrize("relative", ["../secret.pt", "C:/secret.pt", "/tmp/secret.pt"])
def test_resolve_under_root_rejects_escape(tmp_path: Path, relative: str):
    with pytest.raises(ValueError, match="outside benchmark root"):
        resolve_under_root(tmp_path, relative)


def test_nb201_loader_preserves_order_and_parses_genotype(tmp_path: Path):
    manifest = tmp_path / "nb201.csv"
    rows = []
    for arch_id in ("b", "a"):
        for epoch in (0, 5, 10, 25, 50):
            rows.append({"arch_id": arch_id, "epoch": epoch, "checkpoint_path": f"ckpt/{arch_id}-e{epoch}.pt",
                         "official_final_robust_accuracy": f"0.{epoch + 1}",
                         "genotype_json": json.dumps({"nodes": [["skip_connect", 0]]}),
                         "metadata_json": json.dumps({"source": "teacher"})})
    write_csv(manifest, rows)
    items = load_nb201_items(manifest, tmp_path)
    assert [(item.model_or_arch_id, item.checkpoint_epoch) for item in items] == [
        (arch, epoch) for arch in ("b", "a") for epoch in (0, 5, 10, 25, 50)
    ]
    assert items[0].checkpoint_path == (tmp_path / "ckpt/b-e0.pt").resolve()
    assert items[0].metadata["genotype"] == {"nodes": [["skip_connect", 0]]}


def test_loaders_reject_missing_columns_and_duplicate_ids(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    write_csv(missing, [{"arch_id": "a", "epoch": "0"}])
    with pytest.raises(ValueError, match="missing required columns"):
        load_nb201_items(missing, tmp_path)

    duplicate = tmp_path / "duplicate.csv"
    row = {"model_id": "m", "checkpoint_path": "m.pt", "robust_accuracy": "0.5", "metadata_json": "{}"}
    write_csv(duplicate, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        load_rb40_items(duplicate, tmp_path)


def test_darts_loader_covers_epoch_grid_and_json(tmp_path: Path):
    manifest = tmp_path / "darts.csv"
    write_csv(manifest, [{"arch_id": "a", "epoch": epoch, "checkpoint_path": f"a-{epoch}.pt",
                         "full_pgd20_accuracy": "0.25", "genotype_json": json.dumps({"normal": []}),
                         "metadata_json": "{}"} for epoch in (0, 100)])
    items = load_darts_items(manifest, tmp_path)
    assert [item.checkpoint_epoch for item in items] == [0, 100]
    assert items[1].target_value == 0.25
