from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from robust_metric_lab.config import load_config
from robust_metric_lab.robustbench_io import load_rb_model, validate_model_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    names = config.models[: args.limit] if args.limit else config.models
    valid, missing = validate_model_names(names, config.dataset, config.threat_model)
    if missing:
        raise SystemExit(f"Unknown RobustBench model names: {missing}")

    for name in tqdm(valid, desc="loading models"):
        model = load_rb_model(
            name,
            config.dataset,
            config.threat_model,
            args.device,
            model_dir=args.model_dir or config.model_dir or "models",
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"loaded={name}")


if __name__ == "__main__":
    main()
