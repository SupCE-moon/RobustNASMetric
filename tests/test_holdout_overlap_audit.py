"""Holdout selection must reject every previously seen architecture alias."""

from __future__ import annotations

import pytest

from scripts.holdout_overlap_audit import audit_unseen, canonical_darts_signature


def _candidate(**changes) -> dict:
    row = {"space": "darts", "id": "fresh", "signature":
           "normal[skip_connect~0;sep_conv_3x3~1;max_pool_3x3~0;skip_connect~2;"
           "dil_conv_3x3~1;sep_conv_5x5~2;skip_connect~0;max_pool_3x3~3]|"
           "reduce[skip_connect~0;sep_conv_3x3~1;max_pool_3x3~0;skip_connect~2;"
           "dil_conv_3x3~1;sep_conv_5x5~2;skip_connect~0;max_pool_3x3~3]",
           "target_source_id": "new_full_test_pgd20_10000"}
    row.update(changes)
    return row


def test_exact_id_and_stage_alias_are_excluded() -> None:
    candidate = _candidate(id="old")
    prior = [{"space": "darts", "id": "old", "setting": "darts_e0"}]
    report = audit_unseen([candidate], prior)
    assert report["unseen"] == []
    assert report["excluded"][0]["reason"] == "prior_id"


def test_darts_pair_swap_has_same_graph_signature() -> None:
    first = _candidate()
    swapped = _candidate(id="old", signature=first["signature"].replace(
        "skip_connect~0;sep_conv_3x3~1", "sep_conv_3x3~1;skip_connect~0"))
    assert canonical_darts_signature(first["signature"]) == canonical_darts_signature(swapped["signature"])
    report = audit_unseen([first], [swapped])
    assert report["unseen"] == []
    assert report["excluded"][0]["reason"] == "prior_signature"


def test_reused_checkpoint_hash_and_untraceable_target_are_not_unseen() -> None:
    digest = "a" * 64
    candidate = _candidate(checkpoint_sha256=digest)
    prior = [{"space": "darts", "id": "old", "checkpoint_sha256": digest}]
    assert audit_unseen([candidate], prior)["excluded"][0]["reason"] == "prior_checkpoint"
    unresolved = _candidate(target_source_id="")
    report = audit_unseen([unresolved], [])
    assert report["unseen"] == []
    assert report["unresolved"][0]["reason"] == "untraceable_target_source"


def test_invalid_darts_signature_refused() -> None:
    with pytest.raises(ValueError, match="signature"):
        canonical_darts_signature("normal[one] | reduce[two]")


def test_incomplete_prior_ledger_and_duplicate_candidate_block_unseen() -> None:
    candidate = _candidate()
    incomplete = audit_unseen([candidate], [{"space": "darts", "id": "older"}])
    assert incomplete["unseen"] == []
    assert incomplete["unresolved"][0]["reason"] == "incomplete_prior_structure_ledger"
    repeated = audit_unseen([candidate, _candidate(id="second")], [])
    assert len(repeated["unseen"]) == 1
    assert repeated["excluded"][0]["reason"] == "candidate_duplicate"
