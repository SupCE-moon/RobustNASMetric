from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON,
    BenchmarkManifestItem,
    load_darts_items,
    resolve_under_root,
)
from robust_metric_lab.student_runner import (  # noqa: E402
    add_common_arguments,
    evaluate_items,
    load_cifar10_samples,
    validate_common_arguments,
)


def _wrap_darts_model(network):
    import torch
    import torch.nn as nn

    class Normalize(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "mean",
                torch.tensor([125.3 / 255, 123.0 / 255, 113.9 / 255]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([63.0 / 255, 62.1 / 255, 66.7 / 255]).view(1, 3, 1, 1),
            )

        def forward(self, inputs):
            return (inputs - self.mean) / self.std

    return nn.Sequential(Normalize(), network)


def make_model_loader(
    device: str,
    *,
    genotype_cls: object | None = None,
    network_cls: Callable[..., object] | None = None,
    wrap_model_fn: Callable[[object], object] | None = None,
    torch_module: object | None = None,
):
    def load(item: BenchmarkManifestItem):
        nonlocal genotype_cls, network_cls, wrap_model_fn, torch_module
        if genotype_cls is None or network_cls is None:
            from robust_metric_lab.darts_space import DartsGenotype, DartsNetworkCIFAR

            genotype_cls = genotype_cls or DartsGenotype
            network_cls = network_cls or DartsNetworkCIFAR
        if wrap_model_fn is None:
            wrap_model_fn = _wrap_darts_model
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        genotype_value = item.metadata.get("genotype")
        if not isinstance(genotype_value, dict):
            raise ValueError(f"manifest item {item.model_or_arch_id} has invalid genotype_json")
        genotype = genotype_cls.from_dict(genotype_value)
        network = network_cls(genotype, channels=16, layers=8, num_classes=10)
        model = wrap_model_fn(network)
        checkpoint = torch_module.load(
            item.checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("checkpoint is missing model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(device)

    return load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DARTS-100 manifest checkpoints without training."
    )
    add_common_arguments(parser)
    parser.add_argument("--epoch", required=True, type=int, choices=(0, 100))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)

    bundle_root = args.bundle_root.resolve()
    manifest = resolve_under_root(bundle_root, "manifests/darts100_architectures.csv")
    items = [
        item
        for item in load_darts_items(manifest, bundle_root)
        if item.checkpoint_epoch == args.epoch
    ]
    if args.item_limit is not None:
        items = items[: args.item_limit]
    output = args.output_dir / args.run_id / "metric_values.csv"
    result = evaluate_items(
        items,
        make_model_loader(args.device),
        lambda: load_cifar10_samples(args.data_root, args.device),
        args.metric,
        output,
        {
            "run_id": args.run_id,
            "setting": f"darts_e{args.epoch}",
            "seed": args.seed,
            "sample_offset": args.sample_offset,
            "num_samples": args.num_samples,
            "epsilon": EPSILON,
        },
    )
    print(
        f"output={output} written={result.written} failed={result.failed} "
        f"skipped={result.skipped} checkpoint_load=strict"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
