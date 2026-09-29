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
    load_rb40_items,
    resolve_under_root,
)
from robust_metric_lab.student_runner import (  # noqa: E402
    add_common_arguments,
    evaluate_items,
    load_cifar10_samples,
    validate_common_arguments,
)


def make_model_loader(
    bundle_root: str | Path,
    device: str,
    *,
    load_rb_model_fn: Callable[..., object] | None = None,
):
    root = Path(bundle_root).resolve()
    model_root = resolve_under_root(root, "rb40/models")

    def load(item: BenchmarkManifestItem):
        nonlocal load_rb_model_fn
        if load_rb_model_fn is None:
            from robust_metric_lab.robustbench_io import load_rb_model

            load_rb_model_fn = load_rb_model
        return load_rb_model_fn(
            model_name=item.model_or_arch_id,
            dataset="cifar10",
            threat_model="Linf",
            device=device,
            model_dir=str(model_root),
        )

    return load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen RB40 manifest checkpoints without training."
    )
    add_common_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)

    bundle_root = args.bundle_root.resolve()
    manifest = resolve_under_root(bundle_root, "manifests/rb40_models.csv")
    items = load_rb40_items(manifest, bundle_root)
    if args.item_limit is not None:
        items = items[: args.item_limit]
    output = args.output_dir / args.run_id / "metric_values.csv"
    result = evaluate_items(
        items,
        make_model_loader(bundle_root, args.device),
        lambda: load_cifar10_samples(args.data_root, args.device),
        args.metric,
        output,
        {
            "run_id": args.run_id,
            "setting": "rb40_mature",
            "seed": args.seed,
            "sample_offset": args.sample_offset,
            "num_samples": args.num_samples,
            "epsilon": EPSILON,
        },
    )
    print(
        f"output={output} written={result.written} failed={result.failed} "
        f"skipped={result.skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
