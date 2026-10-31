"""Select provisional, unseen DARTS cells using identities only.

Selection remains unsealed until all historical ID-only DARTS sources have
been mapped to signatures. No target or metric values are accepted.
"""

from __future__ import annotations

from collections.abc import Iterable

from robust_metric_lab.darts_space import genotype_signature, sample_darts_genotypes
from scripts.holdout_overlap_audit import canonical_darts_signature


SELECTION_SEED = 20261027
HOLDOUT_ARCHITECTURES = 20
FIRST_HOLDOUT_ID = 1000


def select_darts_architectures(
    prior_canonical_signatures: Iterable[str], *,
    count: int = HOLDOUT_ARCHITECTURES,
    seed: int = SELECTION_SEED,
    pool_size: int = 4096,
) -> list[dict]:
    """Take the first `count` canonical-unseen cells in a fixed random stream."""
    if count < 1 or pool_size < count:
        raise ValueError("eligible DARTS architecture pool too small")
    prior = set(prior_canonical_signatures)
    selected: list[dict] = []
    seen = set()
    for genotype in sample_darts_genotypes(pool_size, seed=seed):
        signature = genotype_signature(genotype)
        canonical = canonical_darts_signature(signature)
        if canonical in prior or canonical in seen:
            continue
        seen.add(canonical)
        selected.append({
            "arch_id": str(FIRST_HOLDOUT_ID + len(selected)),
            "signature": signature,
            "canonical_signature": canonical,
            "genotype": genotype.to_dict(),
        })
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"eligible DARTS architecture pool has {len(selected)} < {count}")
    return selected
