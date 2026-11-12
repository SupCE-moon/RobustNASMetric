"""Tests for the remote census program embedded in tools/server_inventory.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import server_inventory  # noqa: E402


def test_remote_program_reports_files_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"0" * 2048)

    completed = subprocess.run(
        [sys.executable, "-", str(tmp_path), "1024"],
        input=server_inventory.REMOTE_PROGRAM,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    payload = json.loads(completed.stdout)
    by_path = {entry["path"]: entry for entry in payload["entries"]}

    assert set(by_path) == {"src/module.py", "big.bin"}
    assert by_path["src/module.py"]["sha256"] is not None
    assert by_path["big.bin"]["absolute_path"].endswith("big.bin")
    # Files above the hash limit are recorded but not hashed.
    assert by_path["big.bin"]["sha256"] is None
    expected_total = (tmp_path / "src" / "module.py").stat().st_size + 2048
    assert payload["summary"] == {
        "files": 2,
        "total_bytes": expected_total,
        "hashed": 1,
    }
