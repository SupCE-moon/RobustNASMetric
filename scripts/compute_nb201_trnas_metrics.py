from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compute_nb201_metric_subset import build_nb201_model, make_loader, set_seed  # noqa: E402
from robust_metric_lab.attacks import make_adversarial  # noqa: E402
from robust_metric_lab.trnas import compute_trnas_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-metrics", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--nasrobbench-repo", required=True, type=Path)
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metric-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--weight-step-size", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = pd.read_csv(args.input_metrics)
    if "train_epochs" in source:
        source = source[source["train_epochs"] == 0]
    source = source.drop_duplicates("arch_id", keep="first").reset_index(drop=True)
    stop = len(source) if args.stop is None else args.stop
    if args.start < 0 or stop < args.start or stop > len(source):
        raise SystemExit(f"Invalid row slice [{args.start}:{stop}] for {len(source)} architectures.")
    source = source.iloc[args.start:stop].copy()
    if source.empty:
        raise SystemExit("Architecture slice is empty.")

    loader = make_loader(
        args.dataset,
        args.data_root,
        train=False,
        batch_size=args.batch_size,
        sample_limit=args.metric_samples,
        seed=args.seed,
        download=False,
    )
    batches = [(x.to(args.device), y.to(args.device)) for x, y in loader]
    if len(batches) != 1:
        raise SystemExit(
            "TRNAS Psi is defined over one joint sample batch; set --batch-size equal to --metric-samples."
        )
    clean_inputs, labels = batches[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, source_row in source.iterrows():
        arch_id = str(int(source_row["arch_id"]))
        print(f"[{len(rows) + 1}/{len(source)}] arch_id={arch_id}", flush=True)
        set_seed(args.seed)
        model = build_nb201_model(arch_id, args.dataset, args.nasrobbench_repo, args.device)
        checkpoint = args.checkpoint_dir / f"arch_{arch_id}_epoch_0_seed_{args.seed}.pt"
        payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        set_seed(args.seed + 100_000)
        adversarial_inputs = make_adversarial(
            model, clean_inputs, labels, "fgsm", args.epsilon, pgd_steps=1
        )
        result = compute_trnas_metrics(
            model,
            clean_inputs,
            adversarial_inputs,
            labels,
            weight_step_size=args.weight_step_size,
        )

        row = source_row.to_dict()
        row.update(
            {
                "trnas_activation_psi": result.activation_psi,
                "trnas_consistency_phi": result.consistency_phi,
                "trnas_r_score": result.r_score,
                "trnas_num_relu_units": result.num_relu_units,
                "trnas_num_consistency_layers": result.num_consistency_layers,
                "trnas_weight_step_size": args.weight_step_size,
                "trnas_epsilon": args.epsilon,
                "trnas_implementation": "paper_eq2_to_eq6_swap_cardinality_one_step_ascent",
            }
        )
        rows.append(row)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        del model, adversarial_inputs
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
