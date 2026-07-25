from __future__ import annotations

import argparse
import csv
import random
import sys
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
from robust_metric_lab.metrics import compute_feature_metrics  # noqa: E402
from robust_metric_lab.nasrobbench201 import (  # noqa: E402
    iter_records,
    load_json,
    load_meta,
    racl_fixed_network_lipschitz_stats,
)
from robust_metric_lab.trnas import compute_trnas_metrics  # noqa: E402


NORMALIZERS = {
    "cifar10": ([125.3 / 255, 123.0 / 255, 113.9 / 255], [63.0 / 255, 62.1 / 255, 66.7 / 255]),
    "cifar100": ([129.3 / 255, 124.1 / 255, 112.4 / 255], [68.2 / 255, 65.4 / 255, 70.4 / 255]),
}


class Normalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def import_nb201(nasrobbench_repo: Path):
    lib = nasrobbench_repo / "libnas" / "lib"
    if not lib.exists():
        raise FileNotFoundError(f"NAS-Bench-201 lib not found: {lib}")
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    from config_utils import dict2config
    from models import CellStructure, get_cell_based_tiny_net

    return dict2config, CellStructure, get_cell_based_tiny_net


def build_nb201_model(
    arch_id: str,
    dataset: str,
    nasrobbench_repo: Path,
    device: str,
) -> nn.Module:
    dict2config, CellStructure, get_cell_based_tiny_net = import_nb201(nasrobbench_repo)
    meta_path = nasrobbench_repo / "libnas" / "output" / "NAS-BENCH-201-4" / "meta-node-4.pth"
    meta_info = torch.load(meta_path, map_location="cpu", weights_only=False)
    arch = meta_info["archs"][int(arch_id)]
    num_classes = 10 if dataset == "cifar10" else 100
    net = get_cell_based_tiny_net(
        dict2config(
            {
                "name": "infer.tiny",
                "C": 16,
                "N": 5,
                "genotype": CellStructure.str2structure(arch),
                "num_classes": num_classes,
            },
            None,
        )
    )
    normalizer = Normalize(*NORMALIZERS[dataset])
    return nn.Sequential(normalizer, net).to(device)


def make_loader(
    dataset: str,
    root: Path,
    train: bool,
    batch_size: int,
    sample_limit: int | None,
    seed: int,
    download: bool,
) -> torch.utils.data.DataLoader:
    if dataset == "cifar10":
        cls = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError("This quick metric subset runner currently supports cifar10/cifar100.")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor()])
    data = cls(root=str(root), train=train, download=download, transform=train_transform if train else eval_transform)
    if sample_limit is not None and sample_limit < len(data):
        generator = torch.Generator()
        generator.manual_seed(seed + (17 if train else 31))
        indices = torch.randperm(len(data), generator=generator)[:sample_limit].tolist()
        data = torch.utils.data.Subset(data, indices)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + (101 if train else 211))
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )


def choose_architectures(
    benchmark_json: dict,
    meta_json: dict,
    dataset: str,
    target_metric: str,
    sample_size: int,
    seed: int,
    sampling: str = "target_stratified",
) -> list[str]:
    records = [
        record
        for record in iter_records(benchmark_json, dataset, meta_json)
        if target_metric in record.metrics and record.arch_id == record.canonical_id
    ]
    records = sorted(records, key=lambda item: int(item.arch_id))
    rng = random.Random(seed)
    if sample_size >= len(records):
        selected = [record.arch_id for record in records]
    elif sampling == "uniform":
        selected = [record.arch_id for record in rng.sample(records, sample_size)]
    elif sampling == "target_stratified":
        records = sorted(records, key=lambda item: item.metrics[target_metric])
        selected = []
        for i in range(sample_size):
            lo = int(i * len(records) / sample_size)
            hi = int((i + 1) * len(records) / sample_size)
            bucket = records[lo:max(hi, lo + 1)]
            selected.append(rng.choice(bucket).arch_id)
    else:
        raise ValueError(f"Unsupported architecture sampling mode: {sampling}")
    # Slices may run on different but comparable GPUs. Shuffle the sample so
    # no shard is tied to its original ordering or a target interval.
    rng.shuffle(selected)
    return selected


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    attack: str,
    epsilon: float,
    pgd_steps: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if attack != "none":
            model.eval()
            x_train = make_adversarial(model, x, y, attack, epsilon, pgd_steps=pgd_steps)
            model.train()
        else:
            x_train = x
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def compute_metric_row(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    attack: str,
    epsilon: float,
    pgd_steps: int,
    max_layers: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = make_adversarial(model, x, y, attack, epsilon, pgd_steps=pgd_steps)
        result = compute_feature_metrics(model, x, x_adv, y, max_layers=max_layers)
        row = asdict(result)
        batch_n = x.size(0)
        for key, value in row.items():
            if key in {"num_layers", "num_candidate_layers", "sampled_layer_indices"}:
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_n
        total += batch_n
    return {key: value / max(total, 1) for key, value in totals.items()}


def compute_trnas_row(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    epsilon: float,
    weight_step_size: float,
    sample_limit: int,
) -> dict[str, float]:
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    clean_inputs = torch.cat([item[0] for item in batches])[:sample_limit]
    labels = torch.cat([item[1] for item in batches])[:sample_limit]
    adversarial_inputs = make_adversarial(
        model,
        clean_inputs,
        labels,
        "fgsm",
        epsilon,
        pgd_steps=1,
    )
    result = compute_trnas_metrics(
        model,
        clean_inputs,
        adversarial_inputs,
        labels,
        weight_step_size=weight_step_size,
    )
    return {
        "trnas_activation_psi": result.activation_psi,
        "trnas_consistency_phi": result.consistency_phi,
        "trnas_r_score": result.r_score,
        "trnas_num_relu_units": float(result.num_relu_units),
        "trnas_num_consistency_layers": float(result.num_consistency_layers),
        "trnas_samples": float(clean_inputs.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--meta-json", required=True, type=Path)
    parser.add_argument("--nasrobbench-repo", required=True, type=Path)
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--target-metric", default="autoattack")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument(
        "--sampling",
        choices=("uniform", "target_stratified"),
        default="target_stratified",
        help="Architecture sampling rule. Use uniform for oracle-free validation subsets.",
    )
    parser.add_argument(
        "--architecture-seed",
        type=int,
        help="Seed used only for architecture sampling; defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--arch-start",
        type=int,
        default=0,
        help="Zero-based start index in the deterministic sampled architecture list.",
    )
    parser.add_argument(
        "--arch-stop",
        type=int,
        help="Exclusive stop index in the deterministic sampled architecture list.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--eval-epochs",
        type=int,
        nargs="+",
        help="Compute metrics at these epochs in one training run, e.g. --eval-epochs 5 10.",
    )
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-attack", default="pgd", choices=("none", "fgsm", "pgd"))
    parser.add_argument("--metric-attack", default="pgd", choices=("fgsm", "pgd"))
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--train-pgd-steps", type=int, default=3)
    parser.add_argument("--metric-pgd-steps", type=int, default=5)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--include-trnas", action="store_true")
    parser.add_argument("--trnas-weight-step-size", type=float, default=1e-6)
    parser.add_argument("--trnas-samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=("constant", "onecycle"), default="constant")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint directory. Defaults to <output-stem>_checkpoints beside the CSV.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        help="Epochs to save; defaults to every --eval-epochs checkpoint.",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable the default checkpoint saving behavior.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    eval_epochs = sorted(set(args.eval_epochs or [args.epochs]))
    if not eval_epochs or eval_epochs[0] < 0:
        raise SystemExit("--eval-epochs values must be non-negative.")
    checkpoint_epochs = set(args.checkpoint_epochs or eval_epochs)
    if any(epoch < 0 for epoch in checkpoint_epochs):
        raise SystemExit("--checkpoint-epochs values must be non-negative.")
    if args.no_save_checkpoints:
        checkpoint_epochs.clear()
    checkpoint_dir = args.checkpoint_dir or (
        args.output.parent / f"{args.output.stem}_checkpoints"
    )
    max_epochs = max([*eval_epochs, *checkpoint_epochs])

    benchmark = load_json(args.benchmark_json)
    meta = load_meta(args.meta_json)
    architecture_seed = args.seed if args.architecture_seed is None else args.architecture_seed
    arch_ids = choose_architectures(
        benchmark,
        meta,
        dataset=args.dataset,
        target_metric=args.target_metric,
        sample_size=args.sample_size,
        seed=architecture_seed,
        sampling=args.sampling,
    )
    full_arch_count = len(arch_ids)
    arch_stop = args.arch_stop if args.arch_stop is not None else full_arch_count
    if args.arch_start < 0 or arch_stop < args.arch_start or arch_stop > full_arch_count:
        raise SystemExit(
            f"Invalid architecture slice [{args.arch_start}:{arch_stop}] for {full_arch_count} sampled architectures."
        )
    arch_ids = arch_ids[args.arch_start:arch_stop]
    if not arch_ids:
        raise SystemExit("Architecture slice is empty.")
    records = {record.arch_id: record for record in iter_records(benchmark, args.dataset, meta)}

    metric_loader = make_loader(
        args.dataset,
        args.data_root,
        train=False,
        batch_size=args.batch_size,
        sample_limit=args.metric_samples,
        seed=args.seed,
        download=args.download_data,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, arch_id in enumerate(arch_ids, start=1):
        print(f"[{index}/{len(arch_ids)}] arch_id={arch_id}", flush=True)
        # Every architecture receives the same initialization seed, training
        # subset, augmentation RNG start, and batch order.
        set_seed(args.seed)
        train_loader = make_loader(
            args.dataset,
            args.data_root,
            train=True,
            batch_size=args.batch_size,
            sample_limit=args.train_samples,
            seed=args.seed,
            download=args.download_data,
        )
        model = build_nb201_model(arch_id, args.dataset, args.nasrobbench_repo, args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = None
        if args.lr_schedule == "onecycle" and max_epochs > 0:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=max_epochs,
                steps_per_epoch=len(train_loader),
            )

        record = records[arch_id]

        def checkpoint_path(epoch: int) -> Path:
            return checkpoint_dir / f"arch_{arch_id}_epoch_{epoch}_seed_{args.seed}.pt"

        def save_checkpoint(epoch: int) -> None:
            config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            save_training_checkpoint(
                checkpoint_path(epoch),
                model,
                optimizer,
                scheduler,
                {
                    "arch_id": arch_id,
                    "canonical_id": record.canonical_id,
                    "arch_string": record.arch_string or "",
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "oracle_metrics": dict(record.metrics),
                    "training_seed": args.seed,
                    "architecture_seed": architecture_seed,
                    "run_config": config,
                },
            )

        def append_metric_row(epoch: int, train_epoch_loss: float | None = None) -> None:
            # Keep PGD random starts identical across architectures at each
            # fidelity while using a distinct stream for each evaluation epoch.
            # Restore all training RNG streams afterwards so intermediate
            # metric evaluation cannot change the remaining training trajectory.
            python_rng_state = random.getstate()
            torch_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                set_seed(args.seed + 100_000 + epoch)
                metric_row = compute_metric_row(
                    model,
                    metric_loader,
                    args.device,
                    attack=args.metric_attack,
                    epsilon=args.epsilon,
                    pgd_steps=args.metric_pgd_steps,
                    max_layers=args.max_layers,
                )
                racl_stats = racl_fixed_network_lipschitz_stats(model)
                metric_row["racl_arch_lipschitz_log_upper"] = racl_stats.log_upper
                metric_row["racl_input_gradient_lower"] = metric_row["input_grad_l2"]
                metric_row["racl_zero_node_sums"] = float(racl_stats.zero_node_sums)
                metric_row["racl_upper_mode"] = racl_stats.mode
                if args.include_trnas:
                    metric_row.update(
                        compute_trnas_row(
                            model,
                            metric_loader,
                            args.device,
                            epsilon=args.epsilon,
                            weight_step_size=args.trnas_weight_step_size,
                            sample_limit=args.trnas_samples,
                        )
                    )
                    metric_row["trnas_weight_step_size"] = args.trnas_weight_step_size
            finally:
                random.setstate(python_rng_state)
                torch.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            row: dict[str, object] = {
                "arch_id": arch_id,
                "canonical_id": record.canonical_id,
                "arch_string": record.arch_string or "",
                "dataset": args.dataset,
                "target_metric": args.target_metric,
                "oracle_target": record.metrics.get(args.target_metric, ""),
                "oracle_val_acc": record.metrics.get("val_acc", ""),
                "sampled_arch_count": full_arch_count,
                "sampling": args.sampling,
                "architecture_seed": architecture_seed,
                "training_seed": args.seed,
                "train_epoch_loss": "" if train_epoch_loss is None else train_epoch_loss,
                "checkpoint_path": (
                    str(checkpoint_path(epoch)) if epoch in checkpoint_epochs else ""
                ),
                "arch_start": args.arch_start,
                "arch_stop": arch_stop,
                "train_epochs": epoch,
                "train_samples": args.train_samples,
                "metric_samples": args.metric_samples,
                "train_attack": args.train_attack,
                "train_pgd_steps": args.train_pgd_steps,
                "metric_attack": args.metric_attack,
                "metric_pgd_steps": args.metric_pgd_steps,
                "lr_schedule": args.lr_schedule,
            }
            row.update(metric_row)
            rows.append(row)
            with args.output.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        if 0 in eval_epochs:
            if 0 in checkpoint_epochs:
                save_checkpoint(0)
            append_metric_row(0)

        for epoch in range(1, max_epochs + 1):
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                attack=args.train_attack,
                epsilon=args.epsilon,
                pgd_steps=args.train_pgd_steps,
                scheduler=scheduler,
            )
            print(f"  epoch={epoch} loss={loss:.4f}", flush=True)
            if epoch in checkpoint_epochs:
                save_checkpoint(epoch)
            if epoch in eval_epochs:
                append_metric_row(epoch, train_epoch_loss=loss)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
