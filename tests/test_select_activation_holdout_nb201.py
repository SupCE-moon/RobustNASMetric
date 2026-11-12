"""Holdout architecture selection must use identities, never target values."""

from __future__ import annotations

import pytest

from scripts.select_activation_holdout_nb201 import select_nb201_architectures


def meta_for(ids):
    return {str(i): {"isomorph": i, "nb201-string": f"structure-{i}"} for i in ids}


def test_selection_excludes_prior_and_is_order_independent():
    ids = [str(i) for i in range(100)]
    prior = {str(i) for i in range(20)}
    a = select_nb201_architectures(ids, meta_for(range(100)), prior, count=64)
    b = select_nb201_architectures(reversed(ids), meta_for(range(100)), prior, count=64)
    assert a == b
    assert len(a) == 64
    assert not {row["arch_id"] for row in a} & prior


def test_selection_refuses_alias_or_missing_structure():
    ids = [str(i) for i in range(65)]
    meta = meta_for(range(65))
    meta["64"]["isomorph"] = 1
    with pytest.raises(ValueError, match="noncanonical"):
        select_nb201_architectures(ids, meta, set(), count=64)
    meta["64"]["isomorph"] = 64
    del meta["64"]["nb201-string"]
    with pytest.raises(ValueError, match="structure"):
        select_nb201_architectures(ids, meta, set(), count=64)


def test_selection_refuses_incomplete_pool():
    with pytest.raises(ValueError, match="eligible"):
        select_nb201_architectures(["1", "2"], meta_for(range(3)), {"1"}, count=2)
