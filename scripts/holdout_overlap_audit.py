"""Conservatively exclude seen NB201 and DARTS validation architectures.

This module handles identities only. It never reads candidate scores or targets.
An architecture with uncertain provenance is unresolved, not independent.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Mapping


_CELL = re.compile(r"normal\[([^]]+)\]\|reduce\[([^]]+)\]")
_EDGE = re.compile(r"([A-Za-z0-9_]+)~([0-9]+)")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def canonical_darts_signature(signature: str) -> str:
    """Canonicalize the order of the two incoming edges at every DARTS node."""
    compact = re.sub(r"\s+", "", str(signature))
    match = _CELL.fullmatch(compact)
    if match is None:
        raise ValueError("invalid DARTS signature")
    cells = {}
    for cell_name, value in zip(("normal", "reduce"), match.groups()):
        raw_edges = value.split(";")
        if len(raw_edges) != 8:
            raise ValueError("DARTS signature requires eight edges per cell")
        edges = []
        for edge in raw_edges:
            parsed = _EDGE.fullmatch(edge)
            if parsed is None:
                raise ValueError("invalid DARTS signature edge")
            edges.append((parsed.group(1), int(parsed.group(2))))
        nodes = []
        for node in range(4):
            pair = edges[2 * node:2 * node + 2]
            if any(source > node + 1 for _, source in pair):
                raise ValueError("invalid DARTS signature source index")
            nodes.append(sorted(pair))
        cells[cell_name] = nodes
    return json.dumps(cells, sort_keys=True, separators=(",", ":"))


def _identity(row: Mapping) -> tuple[str, str, str | None, str | None]:
    space = str(row.get("space", "")).strip().lower()
    identifier = str(row.get("id", "")).strip()
    if space not in {"nb201", "darts", "rb40"} or not identifier:
        raise ValueError("missing space or architecture ID")
    raw_signature = row.get("signature")
    signature = None
    if raw_signature:
        signature = (canonical_darts_signature(str(raw_signature)) if space == "darts"
                     else re.sub(r"\s+", "", str(raw_signature)))
    raw_hash = row.get("checkpoint_sha256")
    digest = None
    if raw_hash:
        if _SHA256.fullmatch(str(raw_hash)) is None:
            raise ValueError("invalid checkpoint SHA-256")
        digest = str(raw_hash).lower()
    return space, identifier, signature, digest


def audit_unseen(candidates: Iterable[Mapping], prior_ledgers: Iterable[Mapping]) -> dict:
    """Return only candidates proven distinct by ID, structure and weight hash."""
    prior_ids = set()
    prior_signatures = set()
    prior_hashes = set()
    incomplete_prior_spaces = set()
    for row in prior_ledgers:
        space, identifier, signature, digest = _identity(row)
        prior_ids.add((space, identifier))
        if signature:
            prior_signatures.add((space, signature))
        elif space in {"nb201", "darts"}:
            incomplete_prior_spaces.add(space)
        if digest:
            prior_hashes.add(digest)
    report: dict[str, list] = {"unseen": [], "excluded": [], "unresolved": []}
    selected_ids = set()
    selected_signatures = set()
    selected_hashes = set()
    for row in candidates:
        candidate = dict(row)
        try:
            space, identifier, signature, digest = _identity(candidate)
        except ValueError as exc:
            report["unresolved"].append({"candidate": candidate, "reason": str(exc)})
            continue
        reason = None
        if (space, identifier) in prior_ids:
            reason = "prior_id"
        elif signature and (space, signature) in prior_signatures:
            reason = "prior_signature"
        elif digest and digest in prior_hashes:
            reason = "prior_checkpoint"
        elif (space, identifier) in selected_ids or (signature and (space, signature) in selected_signatures):
            reason = "candidate_duplicate"
        elif digest and digest in selected_hashes:
            reason = "candidate_duplicate_checkpoint"
        if reason:
            report["excluded"].append({"candidate": candidate, "reason": reason})
            continue
        if not signature:
            reason = "missing_structure_signature"
        elif space in incomplete_prior_spaces:
            reason = "incomplete_prior_structure_ledger"
        elif not str(candidate.get("target_source_id", "")).strip():
            reason = "untraceable_target_source"
        if reason:
            report["unresolved"].append({"candidate": candidate, "reason": reason})
            continue
        report["unseen"].append(candidate)
        selected_ids.add((space, identifier))
        selected_signatures.add((space, signature))
        if digest:
            selected_hashes.add(digest)
    return report
