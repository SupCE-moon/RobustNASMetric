"""Uniform, target-blind selection of unseen NAS-RobBench-201 architectures."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping


SELECTION_SEED = 20261027
HOLDOUT_ARCHITECTURES = 64
HOLDOUT_EPOCHS = (0, 5, 10, 25, 50)


def select_nb201_architectures(
    benchmark_ids: Iterable[str],
    official_meta_ids: Mapping[str, Mapping],
    prior_canonical_ids: Iterable[str],
    *,
    count: int = HOLDOUT_ARCHITECTURES,
    seed: int = SELECTION_SEED,
) -> list[dict[str, str]]:
    """Select from official canonical IDs without accepting any target values.

    Numeric sorting precedes ``random.sample`` so file order cannot change the
    selection. The returned list order is the frozen sample order.
    """
    ids = [str(value) for value in benchmark_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate official benchmark ID")
    for identifier in ids:
        if identifier not in official_meta_ids:
            raise ValueError(f"official meta missing ID: {identifier}")
        record = official_meta_ids[identifier]
        if str(record.get("isomorph", identifier)) != identifier:
            raise ValueError(f"noncanonical benchmark ID: {identifier}")
        if not isinstance(record.get("nb201-string"), str) or not record["nb201-string"]:
            raise ValueError(f"structure missing for benchmark ID: {identifier}")
    prior = {str(value) for value in prior_canonical_ids}
    if prior - set(ids):
        raise ValueError("prior ID absent from official benchmark")
    eligible = sorted(set(ids) - prior, key=int)
    if len(eligible) < count:
        raise ValueError(f"eligible architecture pool has {len(eligible)} < {count}")
    selected = random.Random(seed).sample(eligible, count)
    signatures = [official_meta_ids[identifier]["nb201-string"] for identifier in selected]
    if len(set(signatures)) != count:
        raise ValueError("duplicate selected structure")
    return [
        {"arch_id": identifier, "canonical_id": identifier, "signature": signature}
        for identifier, signature in zip(selected, signatures)
    ]
