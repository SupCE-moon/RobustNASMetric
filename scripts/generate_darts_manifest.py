from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_metric_lab.darts_space import genotype_signature, sample_darts_genotypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    genotypes = sample_darts_genotypes(args.count, args.seed)
    payload = {
        "search_space": "RACL discrete DARTS/PC-DARTS cell space",
        "architecture_seed": args.seed,
        "count": args.count,
        "zero_edge_policy": "exclude none/zero during final top-two edge discretization",
        "architectures": [
            {
                "arch_id": index,
                "signature": genotype_signature(genotype),
                "genotype": genotype.to_dict(),
            }
            for index, genotype in enumerate(genotypes)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
