from __future__ import annotations

from typing import Iterable

import torch


def normalize_threat_model(threat_model: str) -> str:
    aliases = {
        "linf": "Linf",
        "l_inf": "Linf",
        "l∞": "Linf",
        "l2": "L2",
        "corruptions": "corruptions",
    }
    return aliases.get(threat_model.lower(), threat_model)


def load_rb_model(
    model_name: str,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    device: str | torch.device = "cuda",
    model_dir: str = "models",
) -> torch.nn.Module:
    from robustbench.utils import load_model

    model = load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset=dataset,
        threat_model=normalize_threat_model(threat_model),
    )
    model.eval()
    return model.to(device)


def available_model_names(dataset: str = "cifar10", threat_model: str = "Linf") -> list[str]:
    from robustbench.model_zoo.enums import BenchmarkDataset
    from robustbench.model_zoo.enums import ThreatModel
    from robustbench.model_zoo import model_dicts

    threat = normalize_threat_model(threat_model)
    threat_enum = getattr(ThreatModel, threat)
    dataset_key = dataset
    if dataset_key not in model_dicts:
        dataset_key = next(
            key for key in BenchmarkDataset if key.value == dataset or key.name == dataset
        )
    return list(model_dicts[dataset_key][threat_enum].keys())


def validate_model_names(names: Iterable[str], dataset: str, threat_model: str) -> tuple[list[str], list[str]]:
    available = set(available_model_names(dataset, threat_model))
    valid, missing = [], []
    for name in names:
        if name in available:
            valid.append(name)
        else:
            missing.append(name)
    return valid, missing
