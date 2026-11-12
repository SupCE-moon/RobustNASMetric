"""Tests for tools/compare_inventories.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from compare_inventories import compare, load_inventory, render_markdown


def write_inventory(path: Path, files: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_reports_a_file_only_the_other_tree_holds(tmp_path: Path) -> None:
    target = {"a.py": {"path": "a.py", "size": 1, "sha256": "11", "classification": "allow"}}
    other = dict(target)
    other["b.py"] = {"path": "b.py", "size": 2, "sha256": "22", "classification": "allow"}

    result = compare(target, other, "other")

    assert [item["path"] for item in result["only_in_other"]] == ["b.py"]
    assert result["differs"] == []
    assert result["only_in_target_count"] == 0


def test_treats_same_path_different_bytes_as_a_conflict_not_a_copy(tmp_path: Path) -> None:
    target = {"a.py": {"path": "a.py", "size": 1, "sha256": "11", "classification": "allow"}}
    other = {"a.py": {"path": "a.py", "size": 9, "sha256": "ff", "classification": "allow"}}

    result = compare(target, other, "other")

    assert len(result["differs"]) == 1
    assert result["differs"][0]["other_sha256"] == "ff"
    assert result["differs"][0]["target_sha256"] == "11"
    assert result["only_in_other"] == []


def test_ignores_files_the_policy_rejects() -> None:
    target: dict = {}
    other = {"weights.pt": {"path": "weights.pt", "size": 10, "sha256": "aa", "classification": "reject"}}

    result = compare(target, other, "other")

    assert result["only_in_other"] == []
    assert result["only_in_other_bytes"] == 0


def test_sums_the_size_of_unique_files() -> None:
    target: dict = {}
    other = {
        "a.py": {"path": "a.py", "size": 1_000, "sha256": "11", "classification": "allow"},
        "b.py": {"path": "b.py", "size": 2_000, "sha256": "22", "classification": "allow"},
    }

    result = compare(target, other, "other")

    assert result["only_in_other_bytes"] == 3_000


def test_round_trips_through_disk(tmp_path: Path) -> None:
    path = write_inventory(
        tmp_path / "inv.json",
        [{"path": "a.py", "size": 1, "sha256": "11", "classification": "allow"}],
    )

    loaded = load_inventory(path)

    assert list(loaded) == ["a.py"]


def test_markdown_table_has_one_row_per_tree() -> None:
    markdown = render_markdown(
        [
            {"label": "one", "only_in_other": [], "only_in_other_bytes": 0, "differs": [],
             "unhashed": [], "only_in_target_count": 0},
            {"label": "two", "only_in_other": [{}], "only_in_other_bytes": 1048576, "differs": [{}],
             "unhashed": [], "only_in_target_count": 3},
        ]
    )

    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(rows) == 4  # header, separator, two data rows
    assert "| one | 0 | 0.00 MiB | 0 | 0 | 0 |" in markdown
    assert "| two | 1 | 1.00 MiB | 1 | 0 | 3 |" in markdown


def test_a_file_above_the_hash_limit_is_reported_separately_from_a_conflict() -> None:
    target = {"big.npz": {"path": "big.npz", "size": 100, "sha256": None, "classification": "allow"}}
    other = {"big.npz": {"path": "big.npz", "size": 200, "sha256": None, "classification": "allow"}}

    result = compare(target, other, "other")

    assert result["differs"] == []
    assert len(result["unhashed"]) == 1
    assert result["unhashed"][0]["other_size"] == 200


def test_accepts_the_entries_key_that_inventory_tree_writes(tmp_path: Path) -> None:
    path = tmp_path / "tree.json"
    path.write_text(
        json.dumps({"schema_version": 1, "entries": [{"path": "a.py", "size": 1, "sha256": "11"}]}),
        encoding="utf-8",
    )

    assert list(load_inventory(path)) == ["a.py"]
