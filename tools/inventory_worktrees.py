"""Snapshot every Git worktree of the repository without modifying it.

Produces the ``local-worktrees.json`` evidence required by the repository
reconciliation plan: HEAD, branch, commit date, tracked modifications,
untracked files with size and SHA-256, and a policy classification.

The snapshot is read-only.  It never stages, commits or checks out anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_tree import sha256_file  # noqa: E402
from repository_policy import classify_path  # noqa: E402

SCHEMA_VERSION = 1


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}"
        )
    return completed.stdout


def parse_worktree_list(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        blocks.append(current)
    return blocks


def parse_status_z(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    tokens = [token for token in text.split("\0") if token]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        status, path = token[:2], token[3:]
        if "R" in status or "C" in status:
            index += 1  # the following token is the rename/copy source
        entries.append((status, path))
        index += 1
    return entries


def describe_worktree(repo_root: Path, block: dict[str, str]) -> dict:
    path = Path(block["worktree"])
    head = _git(path, "rev-parse", "HEAD").strip()
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    committed_at = _git(path, "log", "-1", "--format=%cI").strip()
    status_text = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    status_entries = parse_status_z(status_text)

    changes = []
    for status, relative in status_entries:
        absolute = path / relative
        size = absolute.stat().st_size if absolute.is_file() else None
        changes.append(
            {
                "status": status,
                "path": relative,
                "size": size,
                "sha256": sha256_file(absolute) if size is not None else None,
                "classification": classify_path(relative, size or 0),
            }
        )

    return {
        "path": str(path),
        "head": head,
        "branch": branch,
        "committed_at": committed_at,
        "detached": block.get("detached") == "detached",
        "locked": "locked" in block,
        "changes": changes,
        "change_count": len(changes),
        "tracked_change_count": sum(1 for c in changes if c["status"].strip() != "??"),
        "untracked_count": sum(1 for c in changes if c["status"].strip() == "??"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    blocks = parse_worktree_list(_git(repo, "worktree", "list", "--porcelain"))
    worktrees = []
    for block in blocks:
        try:
            worktrees.append(describe_worktree(repo, block))
        except (OSError, RuntimeError) as error:  # keep the snapshot complete
            worktrees.append({"path": block.get("worktree"), "error": str(error)})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repository": str(repo),
        "worktree_count": len(worktrees),
        "dirty_worktrees": sum(
            1 for worktree in worktrees if worktree.get("change_count", 0)
        ),
        "worktrees": worktrees,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"worktrees: {payload['worktree_count']}, dirty: {payload['dirty_worktrees']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
