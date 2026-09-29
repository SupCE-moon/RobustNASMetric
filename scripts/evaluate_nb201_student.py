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
    load_nb201_items,
    resolve_under_root,
)
from robust_metric_lab.student_runner import (  # noqa: E402
    add_common_arguments,
    evaluate_items,
    load_cifar10_samples,
    validate_common_arguments,
)


def build_nb201_model(
    arch_string: str,
    C: int = 16,
    N: int = 5,
    *,
    library_root: str | Path | None = None,
):
    """Build the teacher checkpoint's official NAS-Bench-201 network."""

    import torch
    import torch.nn as nn

    lib = Path(library_root) if library_root is not None else ROOT / "third_party" / "nasrobbench201" / "libnas" / "lib"
    if not lib.is_dir():
        raise FileNotFoundError(f"official NAS-Bench-201 library not found: {lib}")
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    from config_utils import dict2config
    from models import CellStructure, get_cell_based_tiny_net

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

    network = get_cell_based_tiny_net(
        dict2config(
            {
                "name": "infer.tiny",
                "C": C,
                "N": N,
                "genotype": CellStructure.str2structure(arch_string),
                "num_classes": 10,
            },
            None,
        )
    )
    return nn.Sequential(Normalize(), network)


def _manifest_arch_string(item: BenchmarkManifestItem) -> str:
    value = item.metadata.get("arch_string")
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        f"manifest item {item.model_or_arch_id} has no standardized NB201 arch_string"
    )


def make_model_loader(
    device: str,
    *,
    build_model_fn: Callable[..., object] | None = None,
    torch_module: object | None = None,
):
    def load(item: BenchmarkManifestItem):
        nonlocal build_model_fn, torch_module
        if build_model_fn is None:
            build_model_fn = build_nb201_model
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        model = build_model_fn(_manifest_arch_string(item), C=16, N=5)
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
        description="Evaluate frozen NAS-Bench-201 manifest checkpoints without training."
    )
    add_common_arguments(parser)
    parser.add_argument("--epoch", required=True, type=int, choices=(0, 5, 10, 25, 50))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)

    bundle_root = args.bundle_root.resolve()
    manifest = resolve_under_root(bundle_root, "manifests/nb201_64_architectures.csv")
    items = [
        item
        for item in load_nb201_items(manifest, bundle_root)
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
            "setting": f"nb201_e{args.epoch}",
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
