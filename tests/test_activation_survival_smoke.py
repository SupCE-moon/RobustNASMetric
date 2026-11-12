"""Non-ranking smoke selection must stay within the frozen 560-key grid."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.smoke_activation_survival import select_smallest_item


def test_selects_smallest_frozen_model_by_weight_bytes(tmp_path: Path) -> None:
    small = tmp_path / "small.pt"
    large = tmp_path / "large.pt"
    small.write_bytes(b"x")
    large.write_bytes(b"xxxx")
    items = [SimpleNamespace(model_or_arch_id="large", checkpoint_epoch=0,
                             checkpoint_path=large),
             SimpleNamespace(model_or_arch_id="small", checkpoint_epoch=0,
                             checkpoint_path=small)]
    frozen = {("nb201_e0", "large", 0), ("nb201_e0", "small", 0)}
    assert select_smallest_item(items, setting="nb201_e0", frozen_keys=frozen).model_or_arch_id == "small"
    with pytest.raises(ValueError, match="frozen"):
        select_smallest_item(items, setting="darts_e0", frozen_keys=frozen)
