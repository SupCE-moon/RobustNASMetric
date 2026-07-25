from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.config import load_config
from robust_metric_lab.datasets import get_dataset_loader
from robust_metric_lab.robustbench_io import load_rb_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reusable adversarial-example cache from one source model."
    )
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "pgd-lite"])
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    loader = get_dataset_loader(
        dataset_name=config.dataset,
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        limit=args.sample_limit,
    )
    model = load_rb_model(
        args.source_model,
        config.dataset,
        config.threat_model,
        args.device,
        model_dir=args.model_dir or config.model_dir or "models",
    )

    xs, ys, x_advs = [], [], []
    for x, y in tqdm(loader, desc=f"{args.source_model}:{args.attack}"):
        x = x.to(args.device)
        y = y.to(args.device)
        x_adv = make_adversarial(
            model,
            x,
            y,
            args.attack,
            config.epsilon,
            pgd_steps=args.pgd_steps,
        )
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        x_advs.append(x_adv.detach().cpu())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": torch.cat(xs, dim=0),
            "y": torch.cat(ys, dim=0),
            "x_adv": torch.cat(x_advs, dim=0),
            "dataset": config.dataset,
            "threat_model": config.threat_model,
            "epsilon": config.epsilon,
            "source_model": args.source_model,
            "attack": args.attack,
            "pgd_steps": args.pgd_steps,
            "seed": args.seed,
        },
        output,
    )
    print(output)


if __name__ == "__main__":
    main()
