"""Historical exposure comes from source bytes, not a hand-picked ID list."""

from __future__ import annotations

import csv
import json

from scripts.holdout_exposure_inventory import collect_exposure


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_csv_and_json_aliases_are_all_excluded_with_source_hashes(tmp_path):
    source = tmp_path / "outputs" / "nasrobbench201_server" / "old.csv"
    write_csv(source, [{"arch_id": "1", "metric": "0.9"}])
    extra = tmp_path / "outputs" / "student_metrics" / "nb201_attempt.json"
    extra.parent.mkdir(parents=True)
    extra.write_text(json.dumps({"setting": "nb201_e0", "arch_id": "2"}), encoding="utf-8")
    meta = {"ids": {
        "1": {"isomorph": 10, "nb201-string": "arch-one"},
        "2": {"isomorph": 20, "nb201-string": "arch-two"},
    }}
    report = collect_exposure(tmp_path, meta)
    assert report["nb201_canonical_ids"] == ["10", "20"]
    assert set(report["source_sha256"]) == {
        "outputs/nasrobbench201_server/old.csv",
        "outputs/student_metrics/nb201_attempt.json",
    }


def test_unmapped_nb_id_is_reported_not_silently_dropped(tmp_path):
    source = tmp_path / "outputs" / "nb201" / "old.csv"
    write_csv(source, [{"arch_id": "missing", "metric": "0.9"}])
    report = collect_exposure(tmp_path, {"ids": {}})
    assert report["unresolved"][0]["reason"] == "NB201 ID absent from official meta"


def test_darts_signature_and_id_only_sources_are_separate(tmp_path):
    signature = "normal[skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1]|reduce[skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1]"
    with_signature = tmp_path / "outputs" / "darts" / "with.csv"
    without = tmp_path / "outputs" / "darts" / "without.csv"
    write_csv(with_signature, [{"arch_id": "0", "signature": signature}])
    write_csv(without, [{"arch_id": "0", "metric": "0.2"}])
    report = collect_exposure(tmp_path, {"ids": {}})
    assert len(report["darts_canonical_signatures"]) == 1
    assert report["darts_id_only_sources"] == ["outputs/darts/without.csv"]


def test_frozen_source_paths_exclude_later_outputs(tmp_path):
    old = tmp_path / "outputs" / "nb201" / "old.csv"
    later = tmp_path / "outputs" / "nb201" / "later.csv"
    write_csv(old, [{"arch_id": "1"}])
    write_csv(later, [{"arch_id": "2"}])
    meta = {"ids": {"1": {"isomorph": 1}, "2": {"isomorph": 2}}}
    report = collect_exposure(
        tmp_path, meta, source_paths=["outputs/nb201/old.csv"])
    assert report["nb201_canonical_ids"] == ["1"]
    assert list(report["source_sha256"]) == ["outputs/nb201/old.csv"]
