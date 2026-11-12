"""Tests for the repository content policy and the tree inventory tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import inventory_tree  # noqa: E402
import repository_policy  # noqa: E402

MIB = 1024 * 1024


@pytest.mark.parametrize(
    "path",
    [
        "models/wrn.pt",
        "outputs/cache/adv_cache.pt",
        "checkpoints/e0.ckpt",
        "data/cifar10.tar.gz",
        "secrets/token.txt",
        ".env",
        "weights.npz",
    ],
)
def test_policy_rejects_weights_datasets_and_secrets(path: str) -> None:
    assert repository_policy.classify_path(path, 1024) == "reject"


def test_policy_rejects_oversized_file_even_when_text() -> None:
    assert repository_policy.classify_path("outputs/big.csv", 51 * MIB) == "reject"


def test_policy_warns_between_thresholds() -> None:
    assert repository_policy.classify_path("outputs/medium.csv", 12 * MIB) == "warn"


def test_policy_allows_small_source_and_summary() -> None:
    assert repository_policy.classify_path("src/robust_metric_lab/metrics.py", 2048) == "allow"
    assert repository_policy.classify_path("outputs/summary.csv", 100_000) == "allow"


def test_iter_violations_reports_only_rejects() -> None:
    violations = repository_policy.iter_violations(
        [("src/a.py", 10), ("models/w.pt", 10), ("outputs/big.csv", 60 * MIB)]
    )
    assert [path for path, _ in violations] == ["models/w.pt", "outputs/big.csv"]


def test_scan_tree_records_hash_and_policy(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "weights.pt").write_bytes(b"\x00" * 16)
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")

    inventory = inventory_tree.scan_tree(tmp_path, hash_limit=-1)
    by_path = {entry["path"]: entry for entry in inventory["entries"]}

    assert inventory["summary"]["files"] == 3
    assert by_path["src/a.py"]["sha256"] == inventory_tree.sha256_file(
        tmp_path / "src" / "a.py", -1
    )
    assert by_path["weights.pt"]["classification"] == "reject"
    assert by_path["notes.md"]["classification"] == "allow"
    assert inventory["summary"]["rejected"] == 1


def test_scan_tree_skips_configured_names(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")

    inventory = inventory_tree.scan_tree(tmp_path)
    assert [entry["path"] for entry in inventory["entries"]] == ["keep.txt"]


def test_cli_writes_valid_json(tmp_path: Path) -> None:
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "file.txt").write_text("data\n", encoding="utf-8")
    output = tmp_path / "manifest.json"

    exit_code = inventory_tree.main(
        ["--root", str(tmp_path / "tree"), "--output", str(output)]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == inventory_tree.SCHEMA_VERSION
    assert payload["entries"][0]["path"] == "file.txt"
