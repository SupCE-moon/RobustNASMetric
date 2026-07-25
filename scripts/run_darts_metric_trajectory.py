from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.attacks import make_adversarial  # noqa: E402
from robust_metric_lab.darts_space import (  # noqa: E402
    DartsGenotype,
    DartsNetworkCIFAR,
    operation_counts,
)
from robust_metric_lab.metrics import compute_feature_metrics  # noqa: E402
from robust_metric_lab.nasrobbench201 import (  # noqa: E402
    racl_fixed_network_lipschitz_stats,
)
from robust_metric_lab.trnas import compute_trnas_metrics  # noqa: E402


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([125.3 / 255, 123.0 / 255, 113.9 / 255]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([63.0 / 255, 62.1 / 255, 66.7 / 255]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    root: Path,
    train: bool,
    batch_size: int,
    sample_limit: int | None,
    seed: int,
) -> torch.utils.data.DataLoader:
    transform = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        if train
        else transforms.ToTensor()
    )
    dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=train,
        download=False,
        transform=transform,
    )
    if sample_limit is not None and sample_limit < len(dataset):
        generator = torch.Generator().manual_seed(seed + (17 if train else 31))
        indices = torch.randperm(len(dataset), generator=generator)[:sample_limit].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed + (101 if train else 211))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def trades_adversarial(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean_prob = F.softmax(model(x), dim=1)
    x_adv = (x.detach() + 0.001 * torch.randn_like(x)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.kl_div(
            F.log_softmax(model(x_adv), dim=1),
            clean_prob,
            reduction="batchmean",
        )
        gradient = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * gradient.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon).clamp(0.0, 1.0)
    model.train(was_training)
    return x_adv.detach()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epsilon: float,
    step_size: float,
    attack_steps: int,
    beta: float,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = trades_adversarial(model, x, epsilon, step_size, attack_steps)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        clean_logits = model(x)
        adv_logits = model(x_adv)
        natural = F.cross_entropy(clean_logits, y)
        robust = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(clean_logits, dim=1),
            reduction="batchmean",
        )
        loss = natural + beta * robust
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * x.shape[0]
        total += x.shape[0]
    return total_loss / max(total, 1)


def metric_row(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    epsilon: float,
    max_layers: int,
    trnas_weight_step_size: float,
    trnas_samples: int,
) -> dict[str, object]:
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    if len(batches) != 1:
        raise ValueError("Metric and TRNAS computation require one joint batch.")
    x, y = batches[0]
    x_adv = make_adversarial(model, x, y, "fgsm", epsilon, pgd_steps=1)
    result = compute_feature_metrics(model, x, x_adv, y, max_layers=max_layers)
    row = asdict(result)
    racl = racl_fixed_network_lipschitz_stats(model)
    trnas = compute_trnas_metrics(
        model,
        x[:trnas_samples],
        x_adv[:trnas_samples],
        y[:trnas_samples],
        weight_step_size=trnas_weight_step_size,
    )
    row.update(
        {
            "racl_arch_lipschitz_log_upper": racl.log_upper,
            "racl_input_gradient_lower": result.input_grad_l2,
            "racl_upper_mode": racl.mode,
            "racl_zero_node_sums": racl.zero_node_sums,
            "trnas_activation_psi": trnas.activation_psi,
            "trnas_consistency_phi": trnas.consistency_phi,
            "trnas_r_score": trnas.r_score,
            "trnas_num_relu_units": trnas.num_relu_units,
            "trnas_num_consistency_layers": trnas.num_consistency_layers,
            "trnas_weight_step_size": trnas_weight_step_size,
            "trnas_samples": min(trnas_samples, x.shape[0]),
        }
    )
    return row


@torch.no_grad()
def clean_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += x.shape[0]
    return correct / max(total, 1)


def pgd_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    epsilon: float,
    steps: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = make_adversarial(model, x, y, "pgd", epsilon, pgd_steps=steps)
        with torch.no_grad():
            correct += int((model(x_adv).argmax(1) == y).sum().item())
        total += x.shape[0]
    return correct / max(total, 1)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--arch-start", type=int, default=0)
    parser.add_argument("--arch-stop", type=int)
    parser.add_argument("--channels", type=int, default=36)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=(0, 50, 100))
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--metric-samples", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--robust-eval-samples", type=int, default=1000)
    parser.add_argument("--robust-eval-batch-size", type=int, default=128)
    parser.add_argument("--robust-eval-pgd-steps", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-step-size", type=float, default=2 / 255)
    parser.add_argument("--train-attack-steps", type=int, default=10)
    parser.add_argument("--trades-beta", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--trnas-weight-step-size", type=float, default=1e-6)
    parser.add_argument("--trnas-samples", type=int, default=16)
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    architectures = manifest["architectures"]
    stop = len(architectures) if args.arch_stop is None else args.arch_stop
    architectures = architectures[args.arch_start:stop]
    if not architectures:
        raise SystemExit("Architecture slice is empty.")
    eval_epochs = sorted(set(args.eval_epochs))
    if eval_epochs[0] != 0 or eval_epochs[-1] != args.epochs:
        raise SystemExit("--eval-epochs must include 0 and --epochs.")

    set_seed(args.weight_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    metric_loader = make_loader(
        args.data_root,
        train=False,
        batch_size=args.metric_batch_size,
        sample_limit=args.metric_samples,
        seed=args.weight_seed,
    )
    robust_loader = make_loader(
        args.data_root,
        train=False,
        batch_size=args.robust_eval_batch_size,
        sample_limit=args.robust_eval_samples,
        seed=args.weight_seed + 1,
    )

    rows: list[dict[str, object]] = []
    if args.output.exists():
        with args.output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {
        int(row["arch_id"])
        for row in rows
        if row.get("final_pgd_accuracy") not in {None, ""}
    }

    for position, architecture in enumerate(architectures, start=1):
        arch_id = int(architecture["arch_id"])
        if arch_id in completed:
            print(f"[{position}/{len(architectures)}] arch_id={arch_id} already complete", flush=True)
            continue
        print(f"[{position}/{len(architectures)}] arch_id={arch_id}", flush=True)
        set_seed(args.weight_seed)
        genotype = DartsGenotype.from_dict(architecture["genotype"])
        network = DartsNetworkCIFAR(
            genotype,
            channels=args.channels,
            layers=args.layers,
            num_classes=10,
        )
        model = nn.Sequential(Normalize(), network).to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        counts = operation_counts(genotype)
        protocol_scope = (
            "racl_paper_final_macro"
            if args.channels == 36 and args.layers == 20
            else "racl_search_space_exploratory_macro"
        )
        common: dict[str, object] = {
            "arch_id": arch_id,
            "signature": architecture["signature"],
            "architecture_seed": manifest["architecture_seed"],
            "weight_seed": args.weight_seed,
            "channels": args.channels,
            "layers": args.layers,
            "parameter_count": parameter_count,
            "protocol_scope": protocol_scope,
            "train_samples": args.train_samples or 50000,
            "train_attack": "TRADES_PGD",
            "train_attack_steps": args.train_attack_steps,
            "trades_beta": args.trades_beta,
            "epsilon": args.epsilon,
            **{f"op_count_{name}": value for name, value in counts.items()},
        }

        rows = [row for row in rows if int(row["arch_id"]) != arch_id]
        start = time.perf_counter()
        set_seed(args.weight_seed + 100_000)
        initial_metric = metric_row(
            model,
            metric_loader,
            args.device,
            args.epsilon,
            args.max_layers,
            args.trnas_weight_step_size,
            args.trnas_samples,
        )
        initial_checkpoint = args.checkpoint_dir / (
            f"arch_{arch_id}_epoch_0_seed_{args.weight_seed}.pt"
        )
        rows.append(
            {
                **common,
                "train_epochs": 0,
                "train_epoch_loss": "",
                "checkpoint_path": str(initial_checkpoint),
                **initial_metric,
            }
        )
        save_checkpoint(
            initial_checkpoint,
            model,
            optimizer,
            scheduler,
            {**common, "epoch": 0, "genotype": genotype.to_dict()},
        )
        write_rows(args.output, rows)

        for epoch in range(1, args.epochs + 1):
            train_loader = make_loader(
                args.data_root,
                train=True,
                batch_size=args.batch_size,
                sample_limit=args.train_samples,
                seed=args.weight_seed + epoch,
            )
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                args.epsilon,
                args.attack_step_size,
                args.train_attack_steps,
                args.trades_beta,
            )
            scheduler.step()
            print(f"  epoch={epoch}/{args.epochs} loss={loss:.4f}", flush=True)
            if epoch in eval_epochs:
                set_seed(args.weight_seed + 100_000 + epoch)
                metrics = metric_row(
                    model,
                    metric_loader,
                    args.device,
                    args.epsilon,
                    args.max_layers,
                    args.trnas_weight_step_size,
                    args.trnas_samples,
                )
                checkpoint = args.checkpoint_dir / (
                    f"arch_{arch_id}_epoch_{epoch}_seed_{args.weight_seed}.pt"
                )
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    {**common, "epoch": epoch, "genotype": genotype.to_dict()},
                )
                rows.append(
                    {
                        **common,
                        "train_epochs": epoch,
                        "train_epoch_loss": loss,
                        "checkpoint_path": str(checkpoint),
                        **metrics,
                    }
                )
                write_rows(args.output, rows)

        set_seed(args.weight_seed + 200_000)
        final_clean = clean_accuracy(model, robust_loader, args.device)
        final_pgd = pgd_accuracy(
            model,
            robust_loader,
            args.device,
            args.epsilon,
            args.robust_eval_pgd_steps,
        )
        elapsed = time.perf_counter() - start
        for row in rows:
            if int(row["arch_id"]) == arch_id:
                row["final_clean_accuracy"] = final_clean
                row["final_pgd_accuracy"] = final_pgd
                row["robust_eval_samples"] = args.robust_eval_samples
                row["robust_eval_pgd_steps"] = args.robust_eval_pgd_steps
                row["architecture_runtime_seconds"] = elapsed
        write_rows(args.output, rows)
        print(
            f"  final clean={final_clean:.4f} pgd{args.robust_eval_pgd_steps}={final_pgd:.4f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        del model, network
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
