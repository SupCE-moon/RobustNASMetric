"""Recheck a frozen preselection history ledger before a holdout can be sealed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_activation_holdout_manifest import validate_holdout_package
from scripts.holdout_overlap_audit import canonical_darts_signature
from scripts.resolve_darts_id_only_sources import resolve_id_only_sources


def verify_holdout_provenance(package: dict, project_root: Path,
                              inventory_path: Path) -> dict:
    """Require source bytes, complete DARTS identity linkage and no overlap."""
    validate_holdout_package(package)
    root = Path(project_root).resolve()
    path = Path(inventory_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prior inventory missing or outside project")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if package.get("prior_inventory_sha256") != digest:
        raise ValueError("frozen prior inventory SHA mismatch")
    def no_duplicate_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate prior inventory field: {key}")
            result[key] = value
        return result
    inventory = json.loads(
        raw, object_pairs_hook=no_duplicate_fields,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(inventory, dict) or inventory.get("unresolved") != []:
        raise ValueError("unresolved prior inventory entries")
    prior_nb = inventory.get("nb201_canonical_ids")
    prior_darts = inventory.get("darts_canonical_signatures")
    sources = inventory.get("source_sha256")
    if (not isinstance(prior_nb, list) or not isinstance(prior_darts, list)
            or not isinstance(sources, dict) or not sources
            or len(set(prior_nb)) != len(prior_nb)
            or len(set(prior_darts)) != len(prior_darts)):
        raise ValueError("prior inventory identities or sources invalid")
    resolution = resolve_id_only_sources(root, inventory)
    if resolution["status"] != "PASS":
        raise ValueError("unresolved DARTS prior sources")
    linked_darts = {
        link["canonical_signature"]
        for detail in resolution["details"] for link in detail.get("links", [])
    }
    rows = package["objects"]
    nb_ids = {row["model_or_arch_id"] for row in rows if row["setting"] == "nb201_e0"}
    darts_signatures = {
        canonical_darts_signature(row["signature"])
        for row in rows if row["setting"] == "darts_e0"
    }
    if nb_ids & set(prior_nb) or darts_signatures & (set(prior_darts) | linked_darts):
        raise ValueError("holdout architecture overlap with frozen history")
    objects_bytes = json.dumps(rows, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"kind": "activation_holdout_prior_provenance_verification",
            "status": "PASS", "expected_rows": 371,
            "objects_sha256": hashlib.sha256(objects_bytes).hexdigest(),
            "prior_inventory_sha256": digest,
            "verified_prior_sources": len(sources),
            "resolved_id_only_sources": len(resolution["resolved_sources"]),
            "unresolved_prior_sources": 0}
