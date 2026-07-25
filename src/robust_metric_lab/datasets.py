from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASET_ROOT_CANDIDATES = {
    "cifar10": [
        "cifar10",
        "CIFAR10",
        "cifar-10",
    ],
    "cifar100": [
        "cifar100",
        "CIFAR100",
        "cifar-100",
    ],
    "mnist": [
        "MNIST",
        "mnist",
    ],
    "svhn": [
        "svhn",
        "SVHN",
    ],
    "imagenet": [
        "imagenet",
        "ImageNet",
        "ILSVRC2012",
        "ILSVRC",
    ],
}


def _base_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("DATA_ROOT", "DATASETS_ROOT", "TORCH_HOME"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]).expanduser())
    roots.extend(
        [
            Path.home() / "data",
            Path.home() / "datasets",
            Path("/data"),
            Path("/datasets"),
            Path("/home/ubuntu/hy/data"),
            Path("/root/data"),
        ]
    )
    return roots


def looks_like_dataset_root(dataset: str, path: Path) -> bool:
    key = dataset.lower()
    if not path.exists():
        return False
    if key == "cifar10":
        return any(
            candidate.exists()
            for candidate in (
                path / "cifar-10-batches-py",
                path / "CIFAR10" / "cifar-10-batches-py",
                path / "cifar10" / "cifar-10-batches-py",
            )
        )
    if key == "cifar100":
        return any(
            candidate.exists()
            for candidate in (
                path / "cifar-100-python",
                path / "CIFAR100" / "cifar-100-python",
                path / "cifar100" / "cifar-100-python",
            )
        )
    if key == "mnist":
        return any(
            candidate.exists()
            for candidate in (
                path / "MNIST" / "raw",
                path / "MNIST" / "processed",
                path / "raw",
                path / "processed",
            )
        )
    if key == "svhn":
        return any(
            candidate.exists()
            for candidate in (
                path / "test_32x32.mat",
                path / "train_32x32.mat",
                path / "SVHN" / "test_32x32.mat",
                path / "svhn" / "test_32x32.mat",
            )
        )
    if key == "imagenet":
        return _resolve_imagenet_val_root(path) is not None
    raise ValueError(f"Unsupported dataset: {dataset}")


def find_dataset_roots(
    dataset: str,
    extra_roots: list[str] | None = None,
    max_depth: int = 4,
) -> list[Path]:
    key = dataset.lower()
    roots: list[Path] = []
    for base in _base_roots():
        for name in DATASET_ROOT_CANDIDATES.get(key, [key]):
            candidate = base / name
            if looks_like_dataset_root(key, candidate):
                roots.append(candidate)
        if looks_like_dataset_root(key, base):
            roots.append(base)

    for root_text in extra_roots or []:
        root = Path(root_text).expanduser()
        if not root.exists():
            continue
        if looks_like_dataset_root(key, root):
            roots.append(root)
        for path in root.rglob("*"):
            try:
                if len(path.relative_to(root).parts) > max_depth:
                    continue
            except ValueError:
                continue
            if looks_like_dataset_root(key, path):
                roots.append(path)

    deduped = []
    seen = set()
    for path in roots:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def find_cifar10_roots(extra_roots: list[str] | None = None, max_depth: int = 4) -> list[Path]:
    return find_dataset_roots("cifar10", extra_roots=extra_roots, max_depth=max_depth)


def _resolve_imagenet_val_root(path: Path) -> Path | None:
    candidates = [
        path / "val",
        path / "ILSVRC2012_img_val",
        path / "imagenet" / "val",
        path / "ImageNet" / "val",
        path,
    ]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            has_class_dirs = any(child.is_dir() for child in candidate.iterdir())
        except PermissionError:
            has_class_dirs = False
        if has_class_dirs:
            return candidate
    return None


def _subset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def get_dataset_loader(
    dataset_name: str,
    data_root: Path,
    batch_size: int,
    num_workers: int = 2,
    limit: int | None = None,
    image_size: int = 224,
) -> DataLoader:
    key = dataset_name.lower()
    if key == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(data_root),
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
    elif key == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(data_root),
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
    elif key == "mnist":
        dataset = datasets.MNIST(
            root=str(data_root),
            train=False,
            download=False,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]),
        )
    elif key == "svhn":
        dataset = datasets.SVHN(
            root=str(data_root),
            split="test",
            download=False,
            transform=transforms.ToTensor(),
        )
    elif key == "imagenet":
        val_root = _resolve_imagenet_val_root(data_root)
        if val_root is None:
            raise SystemExit(f"No ImageNet validation ImageFolder found under {data_root}")
        dataset = datasets.ImageFolder(
            root=str(val_root),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = _subset(dataset, limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_cifar10_loader(
    data_root: Path,
    batch_size: int,
    num_workers: int = 2,
    limit: int | None = None,
) -> DataLoader:
    return get_dataset_loader("cifar10", data_root, batch_size, num_workers, limit)
