"""Train only the frozen, unseen NB201 holdout weights; never score metrics.

Each architecture has an append-only attempt directory. An incomplete attempt
is retained as evidence, and rerunning starts a new attempt for that exact
missing architecture. This task must run after the main-grid GPU task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import traceback
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTION_SHA256 = "07eeafcc8dbc245ffab169173678b9bf53f7382b41a73c4690783a694ef012a1"
TRAINER_SHA256 = "452c9d84e011d589f7dc1c4b23b482aec4f54e18a20620904dfbcdbabadb22be"
ATTACK_SHA256 = "65b348cedccfb4fcea1aafc322c2f9984ec6c23d01b0bef232ebe760c8aeee13"
META_NODE_SHA256 = "275f76034ee3a4df01c39176101c036642520380d771c61a287b8f75daad02e0"
OFFICIAL_REPO_COMMIT = "f933a51be7026985a6b5fbf9ab82956acb14e3d4"
EPOCHS = (0, 5, 10, 25, 50)
TRAINING = {
    "dataset": "cifar10", "epochs": 50, "checkpoint_epochs": list(EPOCHS),
    "train_samples": 50000, "batch_size": 256, "train_attack": "pgd",
    "train_pgd_steps": 7, "epsilon": 8 / 255, "pgd_alpha": 2 / 255,
    "pgd_random_start": True, "lr": 0.1, "lr_schedule": "onecycle",
    "momentum": 0.9, "weight_decay": 5e-4, "seed": 0,
    "architecture": "NAS-Bench-201 infer.tiny C=16 N=5 classes=10",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def no_constants(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates,
                       parse_constant=no_constants)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def load_frozen_selection(path: Path, expected_sha: str = SELECTION_SHA256) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError("selection hash mismatch")
    selection = _strict_json(path)
    if (selection.get("kind") != "activation_holdout_nb201_selection"
            or selection.get("status") != "unsealed_weights_and_targets_missing"
            or selection.get("expected_architectures") != 64
            or selection.get("expected_keys") != 320
            or selection.get("checkpoint_epochs") != list(EPOCHS)):
        raise ValueError("frozen selection header mismatch")
    rows = selection.get("architectures")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("frozen selection must have 64 architectures")
    ids, signatures = [], []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"arch_id", "canonical_id", "signature"}:
            raise ValueError("selection architecture contains target or unknown field")
        identifier = row["arch_id"]
        if (not isinstance(identifier, str) or not identifier.isdecimal()
                or row["canonical_id"] != identifier
                or not isinstance(row["signature"], str) or not row["signature"]):
            raise ValueError("invalid architecture identity")
        ids.append(identifier)
        signatures.append(row["signature"])
    if len(set(ids)) != 64 or len(set(signatures)) != 64:
        raise ValueError("duplicate selected architecture or signature")
    return selection


def expected_training_plan(selection: dict, *, selection_sha: str, trainer_sha: str,
                           attack_sha: str, meta_node_sha: str,
                           cifar10_hashes: dict[str, str] | None = None) -> dict:
    """Construct the immutable, score-free training plan."""
    return {
        "kind": "activation_holdout_nb201_weight_training",
        "selection_sha256": selection_sha,
        "training_source_sha256": trainer_sha,
        "attack_source_sha256": attack_sha,
        "meta_node_sha256": meta_node_sha,
        "official_repo_commit": OFFICIAL_REPO_COMMIT,
        "cifar10_file_sha256": cifar10_hashes or {},
        "training": dict(TRAINING),
        "architectures": [{k: row[k] for k in ("arch_id", "canonical_id", "signature")}
                          for row in selection["architectures"]],
        "expected_checkpoint_keys": 320,
    }


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def write_or_verify_plan(path: Path, plan: dict) -> str:
    content = _canonical_bytes(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
        return "created"
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("frozen training plan differs from this run")
        return "verified"


def _project_path(value: Path) -> Path:
    resolved = value.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"path must be inside RobustMetric project: {value}")
    return resolved


def _data_hashes(data_root: Path) -> dict[str, str]:
    folder = data_root / "cifar-10-batches-py"
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["batches.meta"]
    return {name: sha256(folder / name) for name in names}


def _verified_complete(arch_dir: Path, identifier: str, plan_sha: str) -> bool:
    receipts = list(arch_dir.glob("attempt_*/complete.json"))
    if len(receipts) > 1:
        raise ValueError(f"duplicate complete attempts for {identifier}")
    if not receipts:
        return False
    receipt = _strict_json(receipts[0])
    if receipt.get("frozen_training_plan_sha256") != plan_sha:
        raise ValueError(f"completed checkpoint plan hash mismatch: {identifier}")
    if receipt.get("arch_id") != identifier or set(receipt.get("checkpoints", {})) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"invalid completed checkpoint receipt: {identifier}")
    for epoch, expected in receipt["checkpoints"].items():
        path = receipts[0].parent / f"arch_{identifier}_epoch_{epoch}_seed_0.pt"
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checkpoint receipt hash mismatch: {path}")
    return True


def _train_one_architecture(row: dict, out: Path, plan_sha: str, repo: Path,
                            data_root: Path) -> None:
    import torch
    from scripts.compute_nb201_metric_subset import (
        build_nb201_model, make_loader, save_training_checkpoint, set_seed,
        train_one_epoch,
    )

    identifier = row["arch_id"]
    arch_dir = out / f"arch_{identifier}"
    if _verified_complete(arch_dir, identifier, plan_sha):
        print(f"arch {identifier}: complete receipt verified; skip", flush=True)
        return
    attempt = arch_dir / f"attempt_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        set_seed(0)
        torch.backends.cudnn.benchmark = True
        loader = make_loader("cifar10", data_root, train=True, batch_size=256,
                             sample_limit=50000, seed=0, download=False)
        model = build_nb201_model(identifier, "cifar10", repo, "cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, epochs=50, steps_per_epoch=len(loader))
        checkpoint_hashes = {}
        for epoch in range(51):
            if epoch:
                loss = train_one_epoch(model, loader, optimizer, "cuda", "pgd",
                                       8 / 255, 7, scheduler)
                print(f"arch {identifier} epoch {epoch} loss={loss:.6f}", flush=True)
            if epoch in EPOCHS:
                path = attempt / f"arch_{identifier}_epoch_{epoch}_seed_0.pt"
                save_training_checkpoint(path, model, optimizer, scheduler, {
                    "arch_id": identifier, "canonical_id": identifier,
                    "arch_string": row["signature"], "dataset": "cifar10",
                    "epoch": epoch, "training_seed": 0,
                    "frozen_training_plan_sha256": plan_sha,
                })
                checkpoint_hashes[str(epoch)] = sha256(path)
        with (attempt / "complete.json").open("xb") as stream:
            stream.write(_canonical_bytes({"arch_id": identifier,
                                           "checkpoints": checkpoint_hashes,
                                           "frozen_training_plan_sha256": plan_sha}))
    except BaseException:
        with (attempt / "failed.json").open("xb") as stream:
            stream.write(_canonical_bytes({"arch_id": identifier,
                                           "traceback": traceback.format_exc()}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    selection_path = _project_path(args.selection)
    repo = _project_path(args.repo)
    data_root = _project_path(args.data_root)
    out = _project_path(args.output_dir)
    selection = load_frozen_selection(selection_path)
    training_path = ROOT / "scripts" / "compute_nb201_metric_subset.py"
    attack_path = ROOT / "src" / "robust_metric_lab" / "attacks.py"
    meta_node = repo / "libnas" / "output" / "NAS-BENCH-201-4" / "meta-node-4.pth"
    if (sha256(training_path) != TRAINER_SHA256 or sha256(attack_path) != ATTACK_SHA256
            or sha256(meta_node) != META_NODE_SHA256):
        raise ValueError("training implementation or architecture source hash changed")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != OFFICIAL_REPO_COMMIT:
        raise ValueError("official architecture repository commit changed")
    plan = expected_training_plan(selection, selection_sha=SELECTION_SHA256,
                                  trainer_sha=TRAINER_SHA256, attack_sha=ATTACK_SHA256,
                                  meta_node_sha=META_NODE_SHA256,
                                  cifar10_hashes=_data_hashes(data_root))
    plan_path = out / "training_plan.json"
    write_or_verify_plan(plan_path, plan)
    print(f"frozen plan: {plan_path} sha256={sha256(plan_path)}", flush=True)
    if args.plan_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA")
    plan_sha = sha256(plan_path)
    for index, row in enumerate(selection["architectures"], start=1):
        print(f"[{index}/64] arch {row['arch_id']}", flush=True)
        _train_one_architecture(row, out, plan_sha, repo, data_root)


if __name__ == "__main__":
    main()
