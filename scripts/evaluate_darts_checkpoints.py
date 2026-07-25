from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.darts_space import DartsGenotype, DartsNetworkCIFAR  # noqa: E402
from run_darts_metric_trajectory import (  # noqa: E402
    Normalize,
    clean_accuracy,
    make_loader,
    pgd_accuracy,
    set_seed,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--arch-start", type=int, default=0)
    parser.add_argument("--arch-stop", type=int)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--checkpoint-epoch", type=int, default=100)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    architectures = payload["architectures"]
    stop = len(architectures) if args.arch_stop is None else args.arch_stop
    architectures = architectures[args.arch_start:stop]
    if not architectures:
        raise SystemExit("Architecture slice is empty.")

    loader = make_loader(
        args.data_root,
        train=False,
        batch_size=args.batch_size,
        sample_limit=args.samples,
        seed=args.weight_seed + 1,
    )
    rows: list[dict[str, object]] = []
    for position, architecture in enumerate(architectures, start=1):
        arch_id = int(architecture["arch_id"])
        print(f"[{position}/{len(architectures)}] arch_id={arch_id}", flush=True)
        set_seed(args.weight_seed)
        genotype = DartsGenotype.from_dict(architecture["genotype"])
        model = nn.Sequential(
            Normalize(),
            DartsNetworkCIFAR(
                genotype,
                channels=args.channels,
                layers=args.layers,
                num_classes=10,
            ),
        ).to(args.device)
        checkpoint = args.checkpoint_dir / (
            f"arch_{arch_id}_epoch_{args.checkpoint_epoch}_seed_{args.weight_seed}.pt"
        )
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        set_seed(args.weight_seed + 300_000)
        clean = clean_accuracy(model, loader, args.device)
        set_seed(args.weight_seed + 400_000)
        robust = pgd_accuracy(
            model,
            loader,
            args.device,
            args.epsilon,
            args.pgd_steps,
        )
        rows.append(
            {
                "arch_id": arch_id,
                "checkpoint_epoch": args.checkpoint_epoch,
                "checkpoint_path": str(checkpoint),
                "evaluation_samples": args.samples,
                "pgd_steps": args.pgd_steps,
                "epsilon": args.epsilon,
                "full_clean_accuracy": clean,
                "full_pgd20_accuracy": robust,
            }
        )
        write_rows(args.output, rows)
        print(f"  clean={clean:.4f} pgd{args.pgd_steps}={robust:.4f}", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
