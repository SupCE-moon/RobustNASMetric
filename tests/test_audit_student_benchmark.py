import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_student_benchmark import build_snapshot, main, sha256_file


def make_bundle(tmp_path: Path, status="IN_PROGRESS") -> Path:
    root = tmp_path / "bundle"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "z.txt").write_bytes(b"z")
    (root / "manifests" / "a.txt").write_bytes(b"abc")
    (root / "nested.bin").write_bytes(b"12")
    (root / "TRANSFER_STATUS.json").write_text(json.dumps({"status": status, "note": "raw"}), encoding="utf-8")
    return root


def test_in_progress_is_preserved_and_not_verified(tmp_path):
    root = make_bundle(tmp_path)
    snapshot = build_snapshot(root)
    assert snapshot["raw_status"] == {"status": "IN_PROGRESS", "note": "raw"}
    assert snapshot["audit_result"] == "RECORDED_NOT_VERIFIED"


def test_sha256_reads_in_chunks(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    path.write_bytes(b"x" * 17)
    calls = []
    original = open

    class Reader:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def read(self, size=-1):
            calls.append(size)
            return self.handle.read(size)

    def tracked_open(*args, **kwargs):
        return Reader(original(*args, **kwargs))

    monkeypatch.setattr("builtins.open", tracked_open)
    assert sha256_file(path, chunk_size=8) == __import__("hashlib").sha256(b"x" * 17).hexdigest()
    assert calls[:2] == [8, 8]


def test_snapshot_and_cli_have_stable_inventory_and_manifest(tmp_path):
    root = make_bundle(tmp_path, "COMPLETE")
    output = tmp_path / "out"
    assert main(["--root", str(root), "--output-dir", str(output)]) == 0
    snapshot = json.loads((output / "source_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["audit_result"] == "RECORDED_NOT_VERIFIED"
    assert snapshot["file_count"] == 4
    assert snapshot["total_bytes"] == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    with (output / "manifest_sha256.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["relative_path"] for row in rows] == ["a.txt", "z.txt"]
    with (output / "file_inventory.csv").open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    assert [row["relative_path"] for row in inventory] == ["TRANSFER_STATUS.json", "manifests/a.txt", "manifests/z.txt", "nested.bin"]


def test_atomic_failure_leaves_no_completed_targets(tmp_path, monkeypatch):
    root = make_bundle(tmp_path)
    output = tmp_path / "out"
    original_replace = Path.replace

    def fail_replace(self, target):
        if str(target).endswith("manifest_sha256.csv"):
            raise OSError("injected failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected failure"):
        main(["--root", str(root), "--output-dir", str(output)])
    assert not any((output / name).exists() for name in ("source_snapshot.json", "manifest_sha256.csv", "file_inventory.csv"))


def test_missing_status_is_explicitly_unverified(tmp_path):
    root = make_bundle(tmp_path)
    (root / "TRANSFER_STATUS.json").unlink()
    snapshot = build_snapshot(root)
    assert snapshot["raw_status"] is None
    assert snapshot["audit_result"] == "RECORDED_NOT_VERIFIED"
    assert snapshot["status_state"] == "MISSING"


def test_output_inside_root_is_excluded_from_snapshot(tmp_path):
    root = make_bundle(tmp_path)
    output = root / "audit"
    output.mkdir()
    (output / "old-evidence.txt").write_text("must not count", encoding="utf-8")
    snapshot = build_snapshot(root, output_dir=output)
    assert snapshot["file_count"] == 4
    assert "audit" not in snapshot["root"] or snapshot["total_bytes"] == sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file() and output not in p.parents
    )


def test_corrupt_status_fails_clearly(tmp_path):
    root = make_bundle(tmp_path)
    (root / "TRANSFER_STATUS.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="TRANSFER_STATUS.json"):
        build_snapshot(root)
