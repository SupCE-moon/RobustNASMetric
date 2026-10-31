"""DARTS holdout selection is deterministic and stays unsealed while provenance is incomplete."""

from __future__ import annotations

import pytest

from robust_metric_lab.darts_space import genotype_signature, sample_darts_genotypes
from scripts.holdout_overlap_audit import canonical_darts_signature
from scripts.select_activation_holdout_darts import select_darts_architectures


def test_selection_excludes_canonical_prior_and_is_reproducible():
    prior = {
        canonical_darts_signature(genotype_signature(g))
        for g in sample_darts_genotypes(10, seed=20261027)
    }
    a = select_darts_architectures(prior, count=20, seed=20261027)
    b = select_darts_architectures(reversed(sorted(prior)), count=20, seed=20261027)
    assert a == b
    assert len(a) == 20
    assert len({row["canonical_signature"] for row in a}) == 20
    assert not {row["canonical_signature"] for row in a} & prior
    assert [row["arch_id"] for row in a] == [str(1000 + i) for i in range(20)]


def test_selection_rejects_insufficient_pool():
    with pytest.raises(ValueError, match="eligible"):
        select_darts_architectures(set(), count=20, pool_size=5)
