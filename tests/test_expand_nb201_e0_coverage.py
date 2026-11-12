"""Tests for the e0 coverage expansion script.

The expensive part (a forward pass per architecture) is not tested here; what is
tested is the bookkeeping that decides *which* architectures get computed, because
a bug there would silently change the sample and therefore the correlation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import expand_nb201_e0_coverage as expand  # noqa: E402

GOOD = "|nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|"
BAD = "|not_a_real_op~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|"


def write_sources(tmp_path: Path, ids: list[str], strings: dict[str, str]) -> tuple[Path, Path]:
    meta = {"ids": {i: {"nb201-string": strings.get(i, GOOD), "isomorph": i} for i in ids}}
    robbench = {
        "cifar10": {
            i: {
                "val_acc": {"threeseed": 0.9},
                "autoattack": {"threeseed": 0.2 + 0.001 * int(i)},
            }
            for i in ids
        }
    }
    meta_path = tmp_path / "meta.json"
    robbench_path = tmp_path / "robbench.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    robbench_path.write_text(json.dumps(robbench), encoding="utf-8")
    return meta_path, robbench_path


def test_load_pool_joins_strings_and_targets(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, ["0", "1", "2"], {})
    pool = expand.load_pool(meta_path, robbench_path)
    assert list(pool.columns) == ["arch_id", "arch_string", "target", "clean"]
    assert len(pool) == 3
    assert pool["target"].iloc[0] == pytest.approx(0.2)


def test_load_pool_accepts_a_scalar_autoattack(tmp_path):
    # The release stores `autoattack` as a bare float while `val_acc` is a dict.
    meta_path, robbench_path = write_sources(tmp_path, ["0"], {})
    payload = json.loads(robbench_path.read_text(encoding="utf-8"))
    payload["cifar10"]["0"]["autoattack"] = 0.4567
    robbench_path.write_text(json.dumps(payload), encoding="utf-8")

    pool = expand.load_pool(meta_path, robbench_path)
    assert pool["target"].iloc[0] == pytest.approx(0.4567)


def test_value_rejects_a_missing_seed_entry():
    assert expand._value({"val_acc": {"0": 0.5}}, "val_acc") is None
    assert expand._value({}, "autoattack") is None
    assert expand._value({"autoattack": "high"}, "autoattack") is None


def test_load_pool_drops_unparseable_strings(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, ["0", "1"], {"1": BAD})
    pool = expand.load_pool(meta_path, robbench_path)
    assert list(pool["arch_id"]) == ["0"]


def test_parse_ops_rejects_the_wrong_number_of_edges():
    short = "|nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_1x1~1|"
    assert expand.parse_ops(short) is None
    assert expand.parse_ops(GOOD) == (
        "nor_conv_1x1",
        "nor_conv_1x1",
        "nor_conv_1x1",
        "nor_conv_1x1",
        "nor_conv_1x1",
        "nor_conv_1x1",
    )


def test_load_pool_drops_architectures_missing_a_target(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, ["0", "1"], {})
    payload = json.loads(robbench_path.read_text(encoding="utf-8"))
    del payload["cifar10"]["1"]["autoattack"]
    robbench_path.write_text(json.dumps(payload), encoding="utf-8")

    pool = expand.load_pool(meta_path, robbench_path)
    assert list(pool["arch_id"]) == ["0"]


def test_sample_always_keeps_the_published_ids(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, [str(i) for i in range(6)], {})
    pool = expand.load_pool(meta_path, robbench_path)

    # limit is smaller than the published set: the published rows must survive,
    # otherwise old and new results stop being comparable.
    chosen = expand.sample(pool, limit=2, seed=1, include=["0", "1", "2"])
    assert sorted(chosen["arch_id"]) >= ["0", "1", "2"]
    assert len(chosen) == 3


def test_sample_fills_up_to_the_limit_without_duplicates(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, [str(i) for i in range(20)], {})
    pool = expand.load_pool(meta_path, robbench_path)

    chosen = expand.sample(pool, limit=10, seed=7, include=["0", "3"])
    assert len(chosen) == 10
    assert chosen["arch_id"].is_unique
    assert {"0", "3"} <= set(chosen["arch_id"])


def test_sample_is_deterministic_for_a_seed(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, [str(i) for i in range(20)], {})
    pool = expand.load_pool(meta_path, robbench_path)

    first = expand.sample(pool, limit=8, seed=20261112, include=[])
    second = expand.sample(pool, limit=8, seed=20261112, include=[])
    assert list(first["arch_id"]) == list(second["arch_id"])


def test_sample_returns_the_whole_pool_when_limit_exceeds_it(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, [str(i) for i in range(4)], {})
    pool = expand.load_pool(meta_path, robbench_path)

    chosen = expand.sample(pool, limit=99, seed=1, include=["1"])
    assert len(chosen) == 4


def test_sample_rows_carry_the_target_so_no_relookup_is_needed(tmp_path):
    meta_path, robbench_path = write_sources(tmp_path, [str(i) for i in range(3)], {})
    pool = expand.load_pool(meta_path, robbench_path)

    chosen = expand.sample(pool, limit=3, seed=1, include=[])
    assert set(chosen.columns) >= {"arch_id", "arch_string", "target", "clean"}
    assert chosen["target"].notna().all()


def test_merge_rows_does_not_duplicate_when_the_csv_turns_ids_into_ints(tmp_path):
    # The regression: pandas reads "0" back as 0, and drop_duplicates then treats
    # the existing row and the new row as different keys, doubling the file.
    path = tmp_path / "out.csv"
    pd.DataFrame(
        {"arch_id": [0, 1], "target": [0.1, 0.2]}
    ).to_csv(path, index=False)

    merged = expand.merge_rows(
        [{"arch_id": "0", "target": 0.5}, {"arch_id": "2", "target": 0.3}], path
    )
    assert len(merged) == 3
    assert merged["arch_id"].is_unique
    assert set(merged["arch_id"]) == {"0", "1", "2"}


def test_merge_rows_keeps_the_newer_value(tmp_path):
    path = tmp_path / "out.csv"
    pd.DataFrame({"arch_id": ["0"], "target": [0.1]}).to_csv(path, index=False)

    merged = expand.merge_rows([{"arch_id": "0", "target": 0.9}], path)
    assert len(merged) == 1
    assert merged["target"].iloc[0] == pytest.approx(0.9)


def test_merge_rows_works_without_an_existing_file(tmp_path):
    merged = expand.merge_rows([{"arch_id": "0", "target": 0.1}], tmp_path / "nope.csv")
    assert len(merged) == 1
