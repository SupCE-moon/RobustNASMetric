"""Find direct checkpoint-path links for DARTS sources lacking signatures.

This is an evidence aid, not a substitute for a complete prior-source audit.
Sources without a direct path link remain unresolved, including E32.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath

from scripts.holdout_exposure_inventory import _space
from scripts.holdout_overlap_audit import canonical_darts_signature


def _read_csv(root: Path, relative: str, expected_sha256: str) -> list[dict]:
    if (not isinstance(relative, str) or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts):
        raise ValueError("historical source path outside project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("historical source missing or outside project")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"historical source SHA mismatch: {relative}")
    if path.suffix.lower() != ".csv":
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate historical CSV column: {relative}")
        return list(reader)


def _checkpoint_key(row: dict) -> tuple[str, str] | None:
    identifier = row.get("arch_id") or row.get("model_or_arch_id")
    path = row.get("checkpoint_path")
    if not identifier or not isinstance(path, str) or not path:
        return None
    normalized = path.replace("\\", "/").removeprefix("./")
    if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        return None
    return str(identifier), normalized


def resolve_id_only_sources(project_root: Path, inventory: dict) -> dict:
    """Resolve only rows whose ID and project-relative checkpoint path match."""
    root = Path(project_root).resolve()
    source_sha = inventory.get("source_sha256")
    id_only = inventory.get("darts_id_only_sources")
    if not isinstance(source_sha, dict) or not isinstance(id_only, list):
        raise ValueError("historical exposure inventory schema invalid")
    if len(set(id_only)) != len(id_only) or not set(id_only) <= set(source_sha):
        raise ValueError("historical ID-only source inventory mismatch")
    by_checkpoint = defaultdict(set)
    rows_by_source = {}
    for relative, expected_sha256 in sorted(source_sha.items()):
        rows = _read_csv(root, relative, expected_sha256)
        rows_by_source[relative] = rows
        for row in rows:
            if _space(relative, row) != "darts":
                continue
            key = _checkpoint_key(row)
            signature = row.get("signature")
            if key is not None and signature:
                by_checkpoint[key].add((canonical_darts_signature(signature), relative))
    resolved = []
    unresolved = []
    details = []
    checkpoint_hashes = {}
    for relative in sorted(id_only):
        rows = rows_by_source[relative]
        if not rows:
            unresolved.append(relative)
            details.append({"source": relative, "reason": "no CSV identity rows"})
            continue
        links = []
        reasons = []
        for row in rows:
            if row.get("signature"):
                continue
            identifier = row.get("arch_id") or row.get("model_or_arch_id")
            if not identifier:
                continue
            key = _checkpoint_key(row)
            if key is None:
                reasons.append("missing checkpoint path")
                continue
            matches = by_checkpoint.get(key, set())
            signatures = {signature for signature, _ in matches}
            if len(signatures) != 1:
                reasons.append("checkpoint has no unique signature")
                continue
            checkpoint = (root / key[1]).resolve()
            if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
                reasons.append("checkpoint file absent or outside project")
                continue
            if checkpoint not in checkpoint_hashes:
                digest = hashlib.sha256()
                with checkpoint.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                checkpoint_hashes[checkpoint] = digest.hexdigest()
            signature = next(iter(signatures))
            links.append({"arch_id": key[0], "checkpoint_path": key[1],
                          "checkpoint_sha256": checkpoint_hashes[checkpoint],
                          "canonical_signature": signature,
                          "signature_sources": sorted(source for sig, source in matches
                                                      if sig == signature)})
        if not links or reasons:
            unresolved.append(relative)
            details.append({"source": relative, "reason": sorted(set(reasons)) or ["no ID-only rows"]})
        else:
            resolved.append(relative)
            details.append({"source": relative, "matched_rows": len(links), "links": links})
    return {"status": "PASS" if not unresolved else "UNRESOLVED",
            "resolved_sources": resolved, "unresolved_sources": unresolved,
            "details": details}
