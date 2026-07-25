from __future__ import annotations

import argparse

from robust_metric_lab.datasets import find_cifar10_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=[])
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    roots = find_cifar10_roots(args.roots, max_depth=args.max_depth)
    if not roots:
        print("No existing CIFAR-10 root found. Set CIFAR10_ROOT or DATA_ROOT; do not redownload blindly.")
        return
    for root in roots:
        print(root)


if __name__ == "__main__":
    main()

