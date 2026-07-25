from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    threat_model: str
    models: list[str]
    num_classes: int = 10
    epsilon: float = 8 / 255
    data_root: str | None = None
    model_dir: str | None = None
    batch_size: int = 128
    num_workers: int = 2


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(**data)


def resolve_data_root(config: ExperimentConfig, override: str | None = None) -> Path | None:
    if override:
        return Path(override).expanduser()
    if config.data_root:
        return Path(config.data_root).expanduser()
    return None
