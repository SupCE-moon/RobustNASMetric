"""Only a shared checkpoint identity may resolve an ID-only DARTS source."""

from __future__ import annotations

import csv

from scripts.holdout_exposure_inventory import collect_exposure
from scripts.resolve_darts_id_only_sources import resolve_id_only_sources


SIG = ("normal[skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;"
       "skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1]|"
       "reduce[skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;"
       "skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1]")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_checkpoint_path_links_one_source_but_id_alone_stays_unresolved(tmp_path) -> None:
    root = tmp_path / "outputs" / "darts"
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "a.pt").write_bytes(b"weight")
    write_csv(root / "with_signature.csv", [{
        "arch_id": "0", "signature": SIG, "checkpoint_path": "checkpoints/a.pt",
    }])
    write_csv(root / "same_checkpoint.csv", [{
        "arch_id": "0", "checkpoint_path": "checkpoints/a.pt", "metric": "0.7",
    }])
    write_csv(root / "e32_id_only.csv", [{"arch_id": "0", "metric": "0.7"}])
    inventory = collect_exposure(tmp_path, {"ids": {}})
    report = resolve_id_only_sources(tmp_path, inventory)
    assert report["resolved_sources"] == ["outputs/darts/same_checkpoint.csv"]
    assert report["unresolved_sources"] == ["outputs/darts/e32_id_only.csv"]
    assert report["status"] == "UNRESOLVED"


def test_changed_source_bytes_cannot_resolve(tmp_path) -> None:
    root = tmp_path / "outputs" / "darts"
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "a.pt").write_bytes(b"weight")
    write_csv(root / "with_signature.csv", [{
        "arch_id": "0", "signature": SIG, "checkpoint_path": "checkpoints/a.pt",
    }])
    write_csv(root / "same_checkpoint.csv", [{
        "arch_id": "0", "checkpoint_path": "checkpoints/a.pt",
    }])
    inventory = collect_exposure(tmp_path, {"ids": {}})
    with (root / "same_checkpoint.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    try:
        resolve_id_only_sources(tmp_path, inventory)
    except ValueError as exc:
        assert "SHA" in str(exc)
    else:
        raise AssertionError("changed source accepted")


def test_nb201_signature_source_is_not_parsed_as_darts(tmp_path) -> None:
    root = tmp_path / "outputs"
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "a.pt").write_bytes(b"weight")
    write_csv(root / "darts" / "with.csv", [{
        "arch_id": "0", "signature": SIG, "checkpoint_path": "checkpoints/a.pt",
    }])
    write_csv(root / "darts" / "without.csv", [{
        "arch_id": "0", "checkpoint_path": "checkpoints/a.pt",
    }])
    write_csv(root / "nb201" / "old.csv", [{
        "arch_id": "1", "signature": "|nor_conv_3x3~0|", "checkpoint_path": "other.pt",
    }])
    inventory = collect_exposure(tmp_path, {"ids": {"1": {"isomorph": 1}}})
    report = resolve_id_only_sources(tmp_path, inventory)
    assert report["status"] == "PASS"


def test_missing_checkpoint_keeps_path_link_unresolved(tmp_path) -> None:
    root = tmp_path / "outputs" / "darts"
    write_csv(root / "with.csv", [{
        "arch_id": "0", "signature": SIG, "checkpoint_path": "checkpoints/missing.pt",
    }])
    write_csv(root / "without.csv", [{
        "arch_id": "0", "checkpoint_path": "checkpoints/missing.pt",
    }])
    inventory = collect_exposure(tmp_path, {"ids": {}})
    report = resolve_id_only_sources(tmp_path, inventory)
    assert report["status"] == "UNRESOLVED"
    assert report["unresolved_sources"] == ["outputs/darts/without.csv"]
