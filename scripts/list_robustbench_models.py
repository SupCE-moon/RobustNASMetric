from __future__ import annotations

import argparse

from robust_metric_lab.robustbench_io import available_model_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--threat-model", default="Linf")
    args = parser.parse_args()

    names = available_model_names(args.dataset, args.threat_model)
    for i, name in enumerate(names, start=1):
        print(f"{i:03d} {name}")


if __name__ == "__main__":
    main()

