from __future__ import annotations

import argparse

from robust_metric_lab.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_linf_20.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"dataset={config.dataset}")
    print(f"threat_model={config.threat_model}")
    print(f"models={len(config.models)}")
    print(f"first_model={config.models[0] if config.models else 'NONE'}")

    try:
        import torch
        import torchvision
        import robustbench

        print(f"torch={torch.__version__}")
        print(f"torchvision={torchvision.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"robustbench={getattr(robustbench, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"dependency_check_failed={exc}")
        raise


if __name__ == "__main__":
    main()

