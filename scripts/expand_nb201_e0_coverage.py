"""Expand the NB201 e0 coverage from 64 architectures to a few hundred.

Why this script exists
----------------------
The RQ2 audit (`scripts/audit_early_metrics_vs_robbench.py`) correlates e0
metrics with the post-training robustness of the same architecture.  It could
only use the 64 architectures of `outputs/e28_capacity_family/nb201_e0_v3.csv`,
which leaves a 95% interval of roughly +-0.25 on Spearman's rho.  NAS-RobBench-201
publishes results for 6466 architectures, so the coverage can be raised without
training anything.

Why the recipe must not change
------------------------------
The purpose of expanding coverage is a narrower interval on the *same* quantity.
Changing the sample count, the attack, the epsilon or the weight seed would make
the new rows incomparable with the published 64, and those 64 cannot be recomputed
away.  So every knob here defaults to what
`scripts/evaluate_nb201_e27_zerocost.py` used, and `--samples`/`--attack`/
`--weight-seed` are documented as "do not change without a decision record".

What is repeated per architecture
---------------------------------
- `torch.manual_seed(weight_seed)` then `Nb201TinyNetwork(ops)`: random init,
  identical seed for every architecture (that is what "e0" means here).
- one FGSM step at 8/255 on the first `--samples` test images.
- `compute_feature_metrics(model, x_clean, x_adv, y, max_layers=4)`.

Resumability
------------
The run takes tens of minutes.  `--resume` (the default) reads the existing
output and skips architectures already present, so an interrupted run continues
instead of starting over.  Rows are flushed every 20 architectures for the same
reason.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.nasrobbench201 import parse_nb201_arch
from robust_metric_lab.nb201_space import Nb201TinyNetwork

EPSILON = 8 / 255
FLUSH_EVERY = 20
VALID_OPS = frozenset({"none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"})


def parse_ops(string: str) -> tuple[str, ...] | None:
    """Parse a cell string, rejecting anything the cell builder would throw on.

    `parse_nb201_arch` only checks the number of edges; an unknown op name passes
    it and then raises deep inside `make_nb201_op`.  Filtering here keeps a single
    bad string from aborting a run that is already minutes in.
    """
    ops = parse_nb201_arch(string)
    if ops is None or any(op not in VALID_OPS for op in ops):
        return None
    return ops


def _value(payload: dict, field: str) -> float | None:
    """Read one RobBench field.

    The release is not uniform: `val_acc` is a per-seed dict with a "threeseed"
    mean, while `autoattack` is already a float.  Both shapes occur, so the reader
    has to accept both instead of assuming the dict form.
    """
    raw = payload.get(field)
    if isinstance(raw, dict):
        value = raw.get("threeseed")
        return None if value is None else float(value)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def load_pool(meta_path: Path, robbench_path: Path) -> pd.DataFrame:
    """Architectures that have both an official string and a RobBench result."""
    ids = json.loads(Path(meta_path).read_text(encoding="utf-8"))["ids"]
    data = json.loads(Path(robbench_path).read_text(encoding="utf-8"))["cifar10"]
    rows = []
    for arch_id, payload in data.items():
        record = ids.get(str(arch_id))
        if not record:
            continue
        string = record.get("nb201-string")
        if not string:
            continue
        target = _value(payload, "autoattack")
        clean = _value(payload, "val_acc")
        if target is None or clean is None:
            continue
        rows.append(
            {
                "arch_id": str(arch_id),
                "arch_string": string,
                "target": target,
                "clean": clean,
            }
        )
    pool = pd.DataFrame(rows)
    # Some strings do not parse into the 6 ops the cell needs; drop them here so
    # the loop never has to decide what to do mid-run.
    pool = pool[pool["arch_string"].map(lambda s: parse_ops(s) is not None)]
    return pool.sort_values("arch_id").reset_index(drop=True)


def sample(pool: pd.DataFrame, limit: int, seed: int, include: list[str]) -> pd.DataFrame:
    """Random sample, plus the already-published ids so old and new rows overlap."""
    rng = np.random.RandomState(seed)
    keep = set(include)
    candidates = pool[~pool["arch_id"].isin(keep)]
    take = max(0, limit - len(keep))
    if take >= len(candidates):
        drawn = candidates
    else:
        idx = rng.choice(len(candidates), size=take, replace=False)
        drawn = candidates.iloc[np.sort(idx)]
    extra = pool[pool["arch_id"].isin(keep)]
    return (
        pd.concat([extra, drawn])
        .drop_duplicates(subset=["arch_id"])
        .sort_values("arch_id")
        .reset_index(drop=True)
    )


def evaluate(model, x, y, attack: str, device: str):
    x_adv = make_adversarial(model, x, y, attack, EPSILON, pgd_steps=1)
    return compute_feature_metrics(model, x, x_adv, y, max_layers=4)


def load_batch(data_root: str, samples: int, device: str):
    loader = get_dataset_loader(
        dataset_name="cifar10",
        data_root=data_root,
        batch_size=samples,
        num_workers=0,
        limit=samples,
    )
    x, y = next(iter(loader))
    return x.to(device), y.to(device)


def merge_rows(rows: list[dict], existing: Path | None = None) -> pd.DataFrame:
    """Combine this run's rows with what is already on disk, one row per architecture.

    A CSV round-trip turns "0" back into 0, and `drop_duplicates` treats 0 and "0"
    as different keys.  Both sides are therefore cast to text before merging;
    without that cast every flush doubles the file.
    """
    frame = pd.DataFrame(rows)
    frame["arch_id"] = frame["arch_id"].astype(str)
    if existing is not None and Path(existing).is_file():
        old = pd.read_csv(existing)
        old["arch_id"] = old["arch_id"].astype(str)
        frame = pd.concat([old, frame], ignore_index=True)
    return frame.drop_duplicates(subset=["arch_id"], keep="last")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="official meta.json (id -> nb201-string)")
    parser.add_argument("--robbench", required=True, help="NAS-RobBench-201_all.json")
    parser.add_argument("--data-root", required=True, help="directory containing cifar-10-batches-py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing", default=None, help="csv whose arch_ids must be included")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261112, help="which architectures, not weights")
    parser.add_argument("--weight-seed", type=int, default=0, help="do not change; defines e0")
    parser.add_argument("--samples", type=int, default=64, help="do not change; defines e0")
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "fixed"], help="do not change; defines e0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    pool = load_pool(Path(args.meta), Path(args.robbench))
    include: list[str] = []
    if args.existing:
        include = pd.read_csv(args.existing)["arch_id"].astype(str).tolist()
    chosen = sample(pool, args.limit, args.seed, include)
    print(f"pool={len(pool)} chosen={len(chosen)} (published={len(include)})")

    out_path = Path(args.output)
    done: set[str] = set()
    if out_path.is_file() and not args.no_resume:
        done = set(pd.read_csv(out_path)["arch_id"].astype(str))
        print(f"resume: {len(done)} rows already present")
    todo = chosen[~chosen["arch_id"].isin(done)]
    if todo.empty:
        print("nothing to do")
        return 0

    x, y = load_batch(args.data_root, args.samples, args.device)
    print(f"batch: x={tuple(x.shape)} device={args.device}")

    rows: list[dict] = []
    started = time.perf_counter()
    for index, record in enumerate(todo.itertuples(index=False), start=1):
        ops = parse_ops(str(record.arch_string))
        torch.manual_seed(args.weight_seed)
        model = Nb201TinyNetwork(ops).to(args.device)
        result = evaluate(model, x, y, args.attack, args.device)
        row = {
            "arch_id": str(record.arch_id),
            "arch_string": str(record.arch_string),
            "target": float(record.target),
            "clean": float(record.clean),
        }
        for key, value in result.__dict__.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
        rows.append(row)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if index % FLUSH_EVERY == 0 or index == len(todo):
            frame = merge_rows(rows, out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out_path, index=False)
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            print(
                f"[{index}/{len(todo)}] {rate:.1f}s/arch "
                f"eta={rate * (len(todo) - index) / 60:.1f}min total={len(frame)}",
                flush=True,
            )

    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
