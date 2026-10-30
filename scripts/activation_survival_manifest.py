"""Provenance checks for the preregistered activation-survival run."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_inventory(inventory: dict[str, str], root: Path) -> None:
    """Refuse a missing or altered audited source file."""
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source mismatch: {name}")
