"""Build a deterministic, hash-bearing inventory of a directory tree.

Used for two jobs:

1. record what exists in a local worktree;
2. record what exists in the server project before anything is imported.

The inventory never modifies the tree.  Hashing is bounded per file so a
multi-gigabyte checkpoint can be recorded by size/mtime without spending
minutes on I/O; callers choose the bound.

Example::

    python tools/inventory_tree.py --root .worktrees/server-mainline \
        --output results/manifests/reconciliation-20261112/local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repository_policy import classify_path  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".worktrees",
    ".idea",
    ".superpowers",
    "node_modules",
)
DEFAULT_HASH_LIMIT = 8 * 1024 * 1024


def sha256_file(path: Path, limit: int = DEFAULT_HASH_LIMIT) -> str | None:
    """Return the SHA-256 of ``path``, or ``None`` when it exceeds ``limit``."""

    size = path.stat().st_size
    if limit >= 0 and size > limit:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(relative: Path, excludes: tuple[str, ...]) -> bool:
    return any(part in excludes for part in relative.parts)


def scan_tree(
    root: Path,
    *,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    hash_limit: int = DEFAULT_HASH_LIMIT,
) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    entries: list[dict] = []
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        dirnames[:] = sorted(d for d in dirnames if d not in excludes)
        for filename in sorted(filenames):
            path = current_path / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            if _excluded(Path(relative), excludes):
                continue
            entries.append(
                {
                    "path": relative,
                    "size": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": sha256_file(path, hash_limit),
                    "classification": classify_path(relative, stat.st_size),
                }
            )

    total = sum(entry["size"] for entry in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "root": str(root),
        "hash_limit_bytes": hash_limit,
        "excluded_names": list(excludes),
        "summary": {
            "files": len(entries),
            "total_bytes": total,
            "hashed": sum(1 for entry in entries if entry["sha256"]),
            "rejected": sum(1 for entry in entries if entry["classification"] == "reject"),
            "warned": sum(1 for entry in entries if entry["classification"] == "warn"),
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="extra directory name to skip (repeatable)",
    )
    parser.add_argument(
        "--hash-limit-mb",
        type=int,
        default=DEFAULT_HASH_LIMIT // (1024 * 1024),
        help="skip hashing files larger than this many MiB (-1 hashes everything)",
    )
    args = parser.parse_args(argv)

    excludes = DEFAULT_EXCLUDES + tuple(args.exclude or ())
    inventory = scan_tree(
        args.root,
        excludes=excludes,
        hash_limit=args.hash_limit_mb * 1024 * 1024,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = inventory["summary"]
    print(
        f"inventory: {summary['files']} files, "
        f"{summary['total_bytes'] / (1024 * 1024):.1f} MiB, "
        f"{summary['hashed']} hashed, "
        f"{summary['rejected']} rejected, {summary['warned']} warned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
