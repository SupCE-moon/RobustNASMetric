"""A three-space independent package cannot be sealed with fewer than 371 keys."""

from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from scripts.build_activation_holdout_manifest import (
    HOLDOUT_COUNTS, seal_holdout_package, validate_holdout_package,
)
from scripts.verify_holdout_targets import verify_holdout_package_targets
from scripts.verify_holdout_provenance import verify_holdout_provenance


def complete_package() -> dict:
    objects = []
    darts_ops = ("skip_connect", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
                 "dil_conv_5x5", "max_pool_3x3", "avg_pool_3x3")
    for setting, count in HOLDOUT_COUNTS.items():
        space = "rb40" if setting == "rb40_mature" else setting.split("_e")[0]
        epoch = None if space == "rb40" else int(setting.rsplit("e", 1)[1])
        for index in range(count):
            if space == "darts":
                first = darts_ops[index % len(darts_ops)]
                second = darts_ops[index // len(darts_ops)]
                cell = (f"{first}~0;{second}~1;skip_connect~0;sep_conv_3x3~1;"
                        "skip_connect~0;sep_conv_3x3~1;skip_connect~0;sep_conv_3x3~1")
                signature = f"normal[{cell}]|reduce[{cell}]"
            else:
                signature = f"{space}_structure_{index}"
            checkpoint_hash = f"{index + 1 + (epoch or 0) * 1000:064x}"
            target_checkpoint_hash = (
                f"{index + 1 + 100 * 1000:064x}" if space == "darts" else None
            )
            objects.append({
                "setting": setting, "model_or_arch_id": str(index),
                "checkpoint_epoch": epoch, "signature": signature,
                "checkpoint_path": f"{space}/{setting}/{index}.pt",
                "checkpoint_sha256": checkpoint_hash,
                "target_value": 0.5, "target_source_id": f"{space}_frozen_target",
                "target_checkpoint_sha256": target_checkpoint_hash,
                "prior_overlap": False,
            })
    return {"objects": objects, "expected_rows": 371,
            "selection_seed": 20261027, "sample_sha256": "f" * 64}


def test_complete_package_has_exact_counts_and_shared_architectures() -> None:
    result = validate_holdout_package(complete_package())
    assert result["expected_rows"] == 371
    assert len(result["objects"]) == 371


@pytest.mark.parametrize("mutation, message", [
    ("missing_rb", "rb40_mature"),
    ("duplicate", "duplicate"),
    ("nb_stage_alias", "NB201"),
    ("darts_target_weight", "target checkpoint"),
    ("darts_e0_self_target", "target checkpoint"),
    ("changed_target", "target differs"),
    ("no_weight_hash", "checkpoint SHA"),
    ("prior_overlap", "prior overlap"),
    ("nonfinite_target", "finite"),
    ("out_of_range_target", "target accuracy"),
    ("duplicate_nb_structure", "duplicate NB201 structure"),
    ("duplicate_darts_structure", "duplicate DARTS structure"),
])
def test_incomplete_or_contaminated_package_rejected(mutation: str, message: str) -> None:
    package = copy.deepcopy(complete_package())
    rows = package["objects"]
    if mutation == "missing_rb":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "nb_stage_alias":
        next(row for row in rows if row["setting"] == "nb201_e50")["model_or_arch_id"] = "extra"
    elif mutation == "darts_target_weight":
        next(row for row in rows if row["setting"] == "darts_e100")["target_checkpoint_sha256"] = "a" * 64
    elif mutation == "darts_e0_self_target":
        row = next(row for row in rows if row["setting"] == "darts_e0")
        row["target_checkpoint_sha256"] = row["checkpoint_sha256"]
    elif mutation == "changed_target":
        next(row for row in rows if row["setting"] == "nb201_e50")["target_value"] = 0.6
    elif mutation == "no_weight_hash":
        rows[0]["checkpoint_sha256"] = ""
    elif mutation == "prior_overlap":
        rows[0]["prior_overlap"] = True
    elif mutation == "nonfinite_target":
        rows[0]["target_value"] = float("nan")
    elif mutation == "out_of_range_target":
        rows[0]["target_value"] = 1.1
    elif mutation == "duplicate_nb_structure":
        for row in rows:
            if row["setting"].startswith("nb201_") and row["model_or_arch_id"] == "1":
                row["signature"] = "nb201_structure_0"
    elif mutation == "duplicate_darts_structure":
        first = next(row["signature"] for row in rows if row["setting"] == "darts_e0")
        for row in rows:
            if row["setting"].startswith("darts_") and row["model_or_arch_id"] == "1":
                row["signature"] = first
    with pytest.raises(ValueError, match=message):
        validate_holdout_package(package)


def test_unverified_package_cannot_be_sealed(tmp_path) -> None:
    with pytest.raises(ValueError, match="verified targets"):
        seal_holdout_package(complete_package(), tmp_path / "sealed.json")


def test_darts_edge_order_alias_is_duplicate_structure() -> None:
    package = complete_package()
    first = next(row["signature"] for row in package["objects"] if row["setting"] == "darts_e0")
    alias = first.replace("skip_connect~0;skip_connect~1", "skip_connect~1;skip_connect~0")
    assert alias != first
    for row in package["objects"]:
        if row["setting"].startswith("darts_") and row["model_or_arch_id"] == "1":
            row["signature"] = alias
    with pytest.raises(ValueError, match="duplicate DARTS structure"):
        validate_holdout_package(package)


def test_sealed_package_is_canonical_and_write_once(tmp_path, monkeypatch) -> None:
    package = complete_package()
    package["source_audit_sha256"] = "a" * 64
    package["protocol_sha256"] = "c" * 64
    package["analysis_sha256"] = "d" * 64
    package["code_sha256"] = "e" * 64
    package["environment_sha256"] = "1" * 64
    package["targets_verified"] = True
    report = tmp_path / "target_verification.json"
    report.write_text(json.dumps({
        "status": "PASS", "expected_rows": 371,
        "objects_sha256": hashlib.sha256(json.dumps(
            package["objects"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "verified_counts": HOLDOUT_COUNTS,
    }, sort_keys=True), encoding="utf-8")
    package["target_verification_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    path = tmp_path / "sealed.json"
    with pytest.raises(ValueError, match="verification report"):
        seal_holdout_package(package, path)
    with pytest.raises(ValueError, match="live target verification"):
        seal_holdout_package(package, path, verification_report_path=report)
    import scripts.verify_holdout_targets as verifier
    monkeypatch.setattr(verifier, "verify_holdout_package_targets",
                        lambda objects, **context: {**json.loads(report.read_text()), "status": "FAIL"})
    with pytest.raises(ValueError, match="differs from live"):
        seal_holdout_package(package, path, verification_report_path=report,
                             verification_context={"project_root": tmp_path})
    monkeypatch.setattr(verifier, "verify_holdout_package_targets",
                        lambda objects, **context: json.loads(report.read_text()))
    with pytest.raises(ValueError, match="source audit report"):
        seal_holdout_package(package, path, verification_report_path=report,
                             verification_context={"project_root": tmp_path})
    provenance = tmp_path / "source_audit.json"
    source_report = {"status": "PASS", "expected_rows": 371,
                     "objects_sha256": json.loads(report.read_text())["objects_sha256"],
                     "unresolved_prior_sources": 0}
    provenance.write_text(json.dumps(source_report), encoding="utf-8")
    package["source_audit_sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    import scripts.verify_holdout_provenance as provenance_verifier
    monkeypatch.setattr(provenance_verifier, "verify_holdout_provenance",
                        lambda objects, **context: source_report)
    sealed = seal_holdout_package(package, path, verification_report_path=report,
                                  verification_context={"project_root": tmp_path},
                                  source_audit_report_path=provenance,
                                  source_audit_context={"project_root": tmp_path})
    assert sealed["sealed"] is True
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert len(sealed["sha256"]) == 64
    with pytest.raises(FileExistsError):
        seal_holdout_package(package, path, verification_report_path=report,
                             verification_context={"project_root": tmp_path},
                             source_audit_report_path=provenance,
                             source_audit_context={"project_root": tmp_path})


def test_mismatched_target_report_cannot_seal(tmp_path) -> None:
    package = complete_package()
    package.update({field: "a" * 64 for field in (
        "source_audit_sha256", "target_verification_sha256", "protocol_sha256",
        "analysis_sha256", "code_sha256", "environment_sha256",
    )})
    package["targets_verified"] = True
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "PASS", "expected_rows": 371,
                                  "objects_sha256": "0" * 64,
                                  "verified_counts": HOLDOUT_COUNTS}))
    package["target_verification_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="objects"):
        seal_holdout_package(package, tmp_path / "sealed.json", verification_report_path=report)
    package["target_verification_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="report SHA"):
        seal_holdout_package(package, tmp_path / "sealed.json", verification_report_path=report)


def test_target_verification_report_binds_all_371_objects(tmp_path, monkeypatch) -> None:
    package = complete_package()
    expected = {"rb40_mature": 11, **{f"nb201_e{x}": 64 for x in (0, 5, 10, 25, 50)},
                "darts_e0": 20, "darts_e100": 20}
    import scripts.verify_holdout_targets as verifier
    monkeypatch.setattr(verifier, "verify_checkpoint_files", lambda rows, root:
                        {"status": "PASS", "verified_rows": len(rows), "verified_files": len(rows)})
    monkeypatch.setattr(verifier, "load_frozen_rb11_manifest", lambda *args: {str(i): {} for i in range(11)})
    monkeypatch.setattr(verifier, "load_rb11_recovery_report", lambda *args: {})
    monkeypatch.setattr(verifier, "load_official_nb201_sources", lambda *args: ({}, {}))
    monkeypatch.setattr(verifier, "verify_rb11_target_rows", lambda rows, *args:
                        {"status": "PASS", "verified_rows": len(rows)})
    monkeypatch.setattr(verifier, "verify_rb11_weight_hash_list", lambda rows, *args:
                        {"status": "PASS", "verified_rows": len(rows)})
    monkeypatch.setattr(verifier, "verify_nb201_target_rows", lambda rows, *args:
                        {"status": "PASS", "verified_rows": len(rows)})
    monkeypatch.setattr(verifier, "verify_darts_target_rows", lambda rows, root:
                        {"status": "PASS", "verified_rows": len(rows)})
    report = verify_holdout_package_targets(
        package, tmp_path, rb_manifest_path=tmp_path / "rb.csv", rb_manifest_sha256="a" * 64,
        rb_recovery_path=tmp_path / "recovery.json", rb_recovery_sha256="b" * 64,
        rb_weight_hash_list_path=tmp_path / "old.sha256", rb_weight_hash_list_sha256="e" * 64,
        nb_result_path=tmp_path / "official.json", nb_result_sha256="c" * 64,
        nb_meta_path=tmp_path / "meta.json", nb_meta_sha256="d" * 64,
    )
    assert report["status"] == "PASS" and report["expected_rows"] == 371
    assert report["verified_counts"] == expected
    assert report["objects_sha256"] == hashlib.sha256(json.dumps(
        package["objects"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    assert "target_value" not in json.dumps(report)
    package["objects"].pop()
    with pytest.raises(ValueError, match="darts_e100"):
        verify_holdout_package_targets(
            package, tmp_path, rb_manifest_path=tmp_path / "rb.csv", rb_manifest_sha256="a" * 64,
            rb_recovery_path=tmp_path / "recovery.json", rb_recovery_sha256="b" * 64,
            rb_weight_hash_list_path=tmp_path / "old.sha256", rb_weight_hash_list_sha256="e" * 64,
            nb_result_path=tmp_path / "official.json", nb_result_sha256="c" * 64,
            nb_meta_path=tmp_path / "meta.json", nb_meta_sha256="d" * 64,
        )


def test_full_371_target_verification_reads_source_and_weight_bytes(tmp_path) -> None:
    package = complete_package()
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"fixture checkpoint")
    weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    rb_path = tmp_path / "rb.csv"
    with rb_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "robust_accuracy"])
        writer.writeheader()
        writer.writerows({"model_name": str(i), "robust_accuracy": "0.5"}
                         for i in range(11))
    rb_sha = hashlib.sha256(rb_path.read_bytes()).hexdigest()
    frozen_hashes_path = tmp_path / "old.sha256"
    frozen_hashes_path.write_text("".join(
        f"{weight_sha}  old_model_{i}.pt\n" for i in range(11)
    ), encoding="ascii")
    frozen_hashes_sha = hashlib.sha256(frozen_hashes_path.read_bytes()).hexdigest()
    for i in range(11):
        (tmp_path / f"old_model_{i}.pt").write_bytes(weight.read_bytes())
    (tmp_path / "replacement.pt").write_bytes(weight.read_bytes())
    recovery = {"frozen_overlay": {"base_manifest_sha256": rb_sha,
                                    "expected_unique_models": 11,
                                    "base_weight_hash_list_sha256": frozen_hashes_sha,
                                    "all_original_weight_hashes_still_match_old_hash_list": True},
                "fresh_container_load_audit": {"status": "PASS", "loadable_unique_models": 11},
                "model_name": "0", "replacement_weight": {"sha256": weight_sha},
                "old_damaged_weight": {"sha256": weight_sha}}
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    recovery_sha = hashlib.sha256(recovery_path.read_bytes()).hexdigest()
    official_path = tmp_path / "official.json"
    official_path.write_text(json.dumps({"cifar10": {
        str(i): {"autoattack": 0.5} for i in range(64)
    }}), encoding="utf-8")
    official_sha = hashlib.sha256(official_path.read_bytes()).hexdigest()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"ids": {
        str(i): {"isomorph": i, "nb201-string": f"nb201_structure_{i}"}
        for i in range(64)
    }}), encoding="utf-8")
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    protocol = {
        "target_attack": "pgd20", "target_dataset": "cifar10_test",
        "target_epsilon": 8 / 255, "target_step_size": 2 / 255,
        "target_steps": 20, "target_restarts": 1, "target_random_start": True,
        "target_batch_size": 128, "target_random_seed": 400000,
        "target_normalization": "cifar10_125.3_123.0_113.9__63.0_62.1_66.7",
    }
    darts_evidence = {}
    for i in range(20):
        path = tmp_path / f"darts_{i}.json"
        path.write_text(json.dumps({
            "status": "ok", "model_or_arch_id": str(i),
            "checkpoint_sha256": weight_sha, "protocol": protocol,
            "sample_ids": list(range(10000)),
            "correct": [True] * 5000 + [False] * 5000,
        }), encoding="utf-8")
        darts_evidence[str(i)] = (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    ids_sha = hashlib.sha256(json.dumps(list(range(10000)), separators=(",", ":")).encode()).hexdigest()
    for row in package["objects"]:
        row["checkpoint_path"] = weight.name
        row["checkpoint_sha256"] = weight_sha
        space = row["setting"].split("_")[0]
        if space == "rb40":
            row["checkpoint_path"] = ("replacement.pt" if row["model_or_arch_id"] == "0"
                                      else f"old_model_{row['model_or_arch_id']}.pt")
            row.update(target_source_id="frozen_rb11_manifest_20260922",
                       target_source_sha256=rb_sha)
        elif space == "nb201":
            row.update(target_source_id="official_nasrobbench201_cifar10_autoattack",
                       target_source_sha256=official_sha)
        else:
            path, digest = darts_evidence[row["model_or_arch_id"]]
            row.update(protocol)
            row.update(target_source_id="new_darts_full_test_pgd20",
                       target_source_path=path, target_source_sha256=digest,
                       target_sample_count=10000, target_sample_ids_sha256=ids_sha,
                       target_checkpoint_sha256=weight_sha,
                       final_checkpoint_sha256=weight_sha)
    kwargs = dict(rb_manifest_path=rb_path, rb_manifest_sha256=rb_sha,
                  rb_recovery_path=recovery_path, rb_recovery_sha256=recovery_sha,
                  rb_weight_hash_list_path=frozen_hashes_path,
                  rb_weight_hash_list_sha256=frozen_hashes_sha,
                  nb_result_path=official_path, nb_result_sha256=official_sha,
                  nb_meta_path=meta_path, nb_meta_sha256=meta_sha)
    report = verify_holdout_package_targets(package, tmp_path, **kwargs)
    assert report["status"] == "PASS"
    assert report["checkpoint_verification"]["verified_files"] == 12
    assert report["darts_target_verification"]["verified_architectures"] == 20
    outside = tmp_path.parent / "outside_rb_manifest.csv"
    outside.write_bytes(rb_path.read_bytes())
    with pytest.raises(ValueError, match="outside project"):
        verify_holdout_package_targets(
            package, tmp_path, **{**kwargs, "rb_manifest_path": outside})
    package.update({field: "a" * 64 for field in (
        "protocol_sha256", "analysis_sha256",
        "code_sha256", "environment_sha256",
    )})
    package["targets_verified"] = True
    old = tmp_path / "outputs" / "nb201" / "old.csv"
    old.parent.mkdir(parents=True)
    old.write_text("arch_id\n9000\n", encoding="utf-8")
    prior_inventory = {"nb201_canonical_ids": ["9000"],
                       "darts_canonical_signatures": [], "darts_id_only_sources": [],
                       "unresolved": [], "source_sha256": {
                           "outputs/nb201/old.csv": hashlib.sha256(old.read_bytes()).hexdigest()}}
    prior_path = tmp_path / "prior_inventory.json"
    prior_path.write_text(json.dumps(prior_inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    prior_meta = tmp_path / "prior_meta.json"
    prior_meta.write_text(json.dumps({"ids": {"9000": {"isomorph": 9000}}}), encoding="utf-8")
    prior_meta_sha = hashlib.sha256(prior_meta.read_bytes()).hexdigest()
    source_report = verify_holdout_provenance(
        package, tmp_path, prior_path,
        official_meta_path=prior_meta, official_meta_sha256=prior_meta_sha)
    source_report_path = tmp_path / "source_audit.json"
    source_report_path.write_text(json.dumps(source_report, sort_keys=True), encoding="utf-8")
    package["source_audit_sha256"] = hashlib.sha256(source_report_path.read_bytes()).hexdigest()
    report_path = tmp_path / "target_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    package["target_verification_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    sealed = seal_holdout_package(
        package, tmp_path / "sealed.json", verification_report_path=report_path,
        verification_context={"project_root": tmp_path, **kwargs},
        source_audit_report_path=source_report_path,
        source_audit_context={"project_root": tmp_path, "inventory_path": prior_path,
                              "official_meta_path": prior_meta,
                              "official_meta_sha256": prior_meta_sha},
    )
    assert sealed["sealed"] is True
    old.write_text("arch_id\n9001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="historical source SHA"):
        seal_holdout_package(
            package, tmp_path / "stale_history.json", verification_report_path=report_path,
            verification_context={"project_root": tmp_path, **kwargs},
            source_audit_report_path=source_report_path,
            source_audit_context={"project_root": tmp_path, "inventory_path": prior_path,
                                  "official_meta_path": prior_meta,
                                  "official_meta_sha256": prior_meta_sha},
        )
    assert not (tmp_path / "stale_history.json").exists()
    old.write_text("arch_id\n9000\n", encoding="utf-8")
    weight.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        verify_holdout_package_targets(package, tmp_path, **kwargs)
    with pytest.raises(ValueError, match="checkpoint SHA"):
        seal_holdout_package(
            package, tmp_path / "should_not_seal.json", verification_report_path=report_path,
            verification_context={"project_root": tmp_path, **kwargs},
            source_audit_report_path=source_report_path,
            source_audit_context={"project_root": tmp_path, "inventory_path": prior_path,
                                  "official_meta_path": prior_meta,
                                  "official_meta_sha256": prior_meta_sha},
        )
    assert not (tmp_path / "should_not_seal.json").exists()


def test_frozen_prior_inventory_blocks_overlap_and_id_only_sources(tmp_path) -> None:
    package = complete_package()
    source = tmp_path / "outputs" / "nb201" / "old.csv"
    source.parent.mkdir(parents=True)
    source.write_text("arch_id\n9000\n", encoding="utf-8")
    relative = "outputs/nb201/old.csv"
    inventory = {"nb201_canonical_ids": ["9000"],
                 "darts_canonical_signatures": [], "darts_id_only_sources": [],
                 "unresolved": [],
                 "source_sha256": {relative: hashlib.sha256(source.read_bytes()).hexdigest()}}
    path = tmp_path / "prior_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"ids": {"9000": {"isomorph": 9000},
                                             "0": {"isomorph": 0}}}), encoding="utf-8")
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    context = {"official_meta_path": meta_path, "official_meta_sha256": meta_sha}
    report = verify_holdout_provenance(package, tmp_path, path, **context)
    assert report["status"] == "PASS" and report["expected_rows"] == 371
    assert report["unresolved_prior_sources"] == 0
    inventory["nb201_canonical_ids"] = ["0"]
    path.write_text(json.dumps(inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        verify_holdout_provenance(package, tmp_path, path, **context)
    source.write_text("arch_id\n0\n", encoding="utf-8")
    inventory["source_sha256"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="overlap"):
        verify_holdout_provenance(package, tmp_path, path, **context)
    inventory["nb201_canonical_ids"] = []
    path.write_text(json.dumps(inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="identit"):
        verify_holdout_provenance(package, tmp_path, path, **context)
    source.write_text("arch_id\n9000\n", encoding="utf-8")
    inventory["source_sha256"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    id_only = tmp_path / "outputs" / "darts" / "e32.csv"
    id_only.parent.mkdir(parents=True)
    id_only.write_text("arch_id,metric\n0,0.7\n", encoding="utf-8")
    inventory["nb201_canonical_ids"] = ["9000"]
    inventory["darts_id_only_sources"] = ["outputs/darts/e32.csv"]
    inventory["source_sha256"]["outputs/darts/e32.csv"] = hashlib.sha256(id_only.read_bytes()).hexdigest()
    path.write_text(json.dumps(inventory), encoding="utf-8")
    package["prior_inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unresolved"):
        verify_holdout_provenance(package, tmp_path, path, **context)
