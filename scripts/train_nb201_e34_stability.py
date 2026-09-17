"""E34: train NB201 architectures adversarially to e5/e10 and test whether the
new capacity variants flip ranking at e5 (magnitude) while gram stays stable.

Local reimplementation (Nb201TinyNetwork) with a CIFAR-10 Normalize head, to
approximate the e23 protocol. This is the empirical confirmation of E33's
prediction that all dispersion-magnitude capacity variants share plain
capacity's training-dynamics instability.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.nasrobbench201 import parse_nb201_arch
from robust_metric_lab.nb201_space import Nb201TinyNetwork

EPSILON = 8 / 255
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_model(ops: tuple[str, ...], device: str) -> nn.Module:
    net = Nb201TinyNetwork(ops)
    return nn.Sequential(Normalize(CIFAR10_MEAN, CIFAR10_STD), net).to(device)


def make_train_loader(data_root: Path, samples: int, batch: int, seed: int):
    transform = T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
    )
    ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=transform)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:samples].tolist()
    ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, generator=gen)


def make_eval_loader(data_root: Path, samples: int, seed: int):
    ds = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=T.ToTensor())
    gen = torch.Generator().manual_seed(seed + 31)
    idx = torch.randperm(len(ds), generator=gen)[:samples].tolist()
    ds = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=samples, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    return x, y


def eval_metrics(model, x, y, device) -> dict:
    model.eval()
    x, y = x.to(device), y.to(device)
    x_adv = make_adversarial(model, x, y, "fgsm", EPSILON, pgd_steps=1)
    result = compute_feature_metrics(model, x, x_adv, y, max_layers=4)
    return {k: v for k, v in result.__dict__.items() if isinstance(v, (int, float))}


def train_epoch(model, loader, opt, device, pgd_steps):
    model.train()
    total, total_loss = 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = make_adversarial(model, x, y, "pgd", EPSILON, pgd_steps=pgd_steps)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        opt.step()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)
    return total_loss / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv")
    parser.add_argument("--data-root", default="D:/Project/RobustNAS/DynamicBN/DynamicBN20251203/data")
    parser.add_argument("--output", default="outputs/e34_nb201_e5_stability.csv")
    parser.add_argument("--arch-limit", type=int, default=None)
    parser.add_argument("--arch-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--metric-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pgd-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.csv)
    archs = df[df.train_epochs == 0].drop_duplicates("arch_id").sort_values("arch_id")
    if args.arch_limit:
        archs = archs.iloc[args.arch_start : args.arch_start + args.arch_limit]
    else:
        archs = archs.iloc[args.arch_start:]
    print(f"training {len(archs)} architectures to epoch {args.epochs}", flush=True)

    x_eval, y_eval = make_eval_loader(Path(args.data_root), args.metric_samples, args.seed)
    x_eval, y_eval = x_eval.to(args.device), y_eval.to(args.device)

    rows = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(archs.iterrows()):
        ops = parse_nb201_arch(str(row["arch_string"]))
        if ops is None:
            continue
        torch.manual_seed(args.weight_seed)
        model = build_model(ops, args.device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        def record(epoch: int):
            m = eval_metrics(model, x_eval, y_eval, args.device)
            r = {"arch_id": row["arch_id"], "oracle_target": row["oracle_target"], "train_epochs": epoch}
            r.update(m)
            rows.append(r)

        record(0)
        loader = make_train_loader(Path(args.data_root), args.train_samples, args.batch_size, args.seed)
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, opt, args.device, args.train_pgd_steps)
            print(f"  [{i+1}/{len(archs)}] arch {row['arch_id']} epoch {epoch} loss={loss:.3f}", flush=True)
            if epoch in (5, 10, args.epochs):
                record(epoch)

        del model, opt, loader
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # incremental write
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"saved {out_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
