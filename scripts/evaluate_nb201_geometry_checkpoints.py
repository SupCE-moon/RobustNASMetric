from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from compute_nb201_metric_subset import build_nb201_model, make_loader, set_seed
from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.metrics import GEOMETRY_METRIC_FIELDS, compute_feature_geometry_metrics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--nasrobbench-repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100"))
    parser.add_argument("--metric-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.metrics_csv.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows.sort(key=lambda row: (int(row["arch_id"]), int(row["train_epochs"])))
    loader = make_loader(
        args.dataset,
        args.data_root,
        train=False,
        batch_size=args.batch_size,
        sample_limit=args.metric_samples,
        seed=args.seed,
        download=False,
    )
    rows: list[dict[str, object]] = []
    for position, source in enumerate(source_rows, start=1):
        arch_id = source["arch_id"]
        epoch = int(source["train_epochs"])
        print(f"[{position}/{len(source_rows)}] arch_id={arch_id} epoch={epoch}", flush=True)
        set_seed(args.seed)
        model = build_nb201_model(arch_id, args.dataset, args.nasrobbench_repo, args.device)
        checkpoint = Path(source["checkpoint_path"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)

        totals = {field: 0.0 for field in GEOMETRY_METRIC_FIELDS}
        total = 0
        audit: dict[str, object] = {}
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            x_adv = make_adversarial(model, x, y, "fgsm", args.epsilon, pgd_steps=1)
            result = compute_feature_geometry_metrics(
                model, x, x_adv, max_layers=args.max_layers
            )
            values = asdict(result)
            for field in GEOMETRY_METRIC_FIELDS:
                totals[field] += float(values[field]) * x.shape[0]
            total += x.shape[0]
            audit = {
                "geometry_num_layers": result.num_layers,
                "geometry_num_candidate_layers": result.num_candidate_layers,
                "geometry_sampled_layer_indices": result.sampled_layer_indices,
            }
        rows.append(
            {
                "arch_id": int(arch_id),
                "train_epochs": epoch,
                **{field: totals[field] / total for field in GEOMETRY_METRIC_FIELDS},
                **audit,
            }
        )
        write_rows(args.output, rows)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
