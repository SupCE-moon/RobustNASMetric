"""Inventory historical NB201/DARTS architecture exposure without using scores."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.holdout_overlap_audit import canonical_darts_signature


def _space(path: str, row: dict) -> str | None:
    setting = str(row.get("setting", row.get("space", ""))).lower()
    if "nb201" in setting or "nasrobbench201" in setting:
        return "nb201"
    if "darts" in setting:
        return "darts"
    lowered = path.lower()
    if "nb201" in lowered or "nasrobbench201" in lowered:
        return "nb201"
    if "darts" in lowered:
        return "darts"
    return None


def _dicts(value: object):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            yield item
            stack.extend(v for v in item.values() if isinstance(v, (dict, list)))


def collect_exposure(project_root: Path, official_meta: dict) -> dict:
    """Return source-hashed prior IDs/signatures and unresolved identity rows.

    CSV/JSON values outside identity fields are never used for selection.
    The caller must still audit non-tabular provenance before sealing a holdout.
    """
    root = Path(project_root).resolve()
    meta_ids = official_meta["ids"]
    nb_ids: set[str] = set()
    darts_signatures: set[str] = set()
    darts_id_only: set[str] = set()
    source_sha: dict[str, str] = {}
    unresolved: list[dict[str, str]] = []

    csv_paths = list((root / "outputs").rglob("*.csv"))
    csv_paths += list((root / "shared_benchmarks").rglob("*.csv"))
    json_paths = list((root / "outputs").rglob("*.json"))
    for path in sorted([*csv_paths, *json_paths]):
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        if path.suffix == ".csv":
            try:
                rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
            except (UnicodeDecodeError, csv.Error):
                unresolved.append({"source": rel, "reason": "CSV parse failure"})
                continue
        else:
            try:
                rows = list(_dicts(json.loads(raw)))
            except (ValueError, UnicodeDecodeError):
                if any(token in raw for token in (b"arch_id", b"model_or_arch_id", b"signature")):
                    unresolved.append({"source": rel, "reason": "identity-bearing JSON parse failure"})
                continue
        used = False
        id_only = False
        for row in rows:
            space = _space(rel, row)
            if space == "nb201":
                identifier = row.get("canonical_id") or row.get("arch_id") or row.get("model_or_arch_id")
                if identifier is None or str(identifier).strip() == "":
                    continue
                used = True
                identifier = str(identifier).strip()
                if identifier not in meta_ids:
                    unresolved.append({"source": rel, "reason": "NB201 ID absent from official meta", "id": identifier})
                    continue
                nb_ids.add(str(meta_ids[identifier].get("isomorph", identifier)))
            elif space == "darts":
                identifier = row.get("arch_id") or row.get("model_or_arch_id")
                signature = row.get("signature")
                if identifier is None and not signature:
                    continue
                used = True
                if signature:
                    try:
                        darts_signatures.add(canonical_darts_signature(str(signature)))
                    except ValueError:
                        unresolved.append({"source": rel, "reason": "DARTS signature invalid"})
                else:
                    id_only = True
        if used:
            source_sha[rel] = hashlib.sha256(raw).hexdigest()
            if id_only:
                darts_id_only.add(rel)
    return {
        "nb201_canonical_ids": sorted(nb_ids, key=lambda value: int(value)),
        "darts_canonical_signatures": sorted(darts_signatures),
        "darts_id_only_sources": sorted(darts_id_only),
        "source_sha256": dict(sorted(source_sha.items())),
        "unresolved": unresolved,
    }
