"""Audit that every registered metric has somewhere to come from.

The registry in ``src/robust_metric_lab/metric_registry.py`` is the single
list of metrics the project claims to compute.  The one-pass bundle
``compute_feature_metrics`` only fills a subset of them: metrics that need a
second view, a per-architecture Jacobian bound, an ODE solve or a two-pass
importance vector live in their own modules.

Nothing so far checked that the two sides agree.  A name could be registered,
appear in the eligibility config and even in the release table while no code
path emits it -- a typo in the registry would then only surface as a missing
column in a later run, possibly hours into a job.

This tool answers one question per registered metric: *where does this number
come from?*  It resolves each name to the modules that mention it and, with
``--dynamic``, also runs the one-pass bundle once on a small random ResNet to
record which fields it really returns.

Three outcomes matter, and they are not the same thing:

* ``other module`` -- the bundle returns it as NaN on purpose (a placeholder)
  and a dedicated script computes the real value.  NaN here means "not this
  entry point", exactly like an empty coverage cell is not a zero.
* ``registry only`` -- the name exists in the registry and nowhere else.  That
  is a real gap: either the implementation never landed or it writes a
  different key.
* ``MISSING`` -- not even the registry mentions it (should be impossible).

Usage::

    python tools/audit_metric_implementation.py --json out.json --report out.md
    python tools/audit_metric_implementation.py --dynamic --json out.json --report out.md

Exit code is 1 when any metric is ``registry only`` or ``MISSING``, or when the
dynamic run returns a bundle field the registry does not know.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "robust_metric_lab"
REGISTRY = SRC / "metric_registry.py"
BUNDLE_FILE = "metrics.py"
BUNDLE_CLASS = "MetricResult"
REGISTRY_NAME = "metric_registry.py"

# Fields the bundle returns that are metadata (sample counts, layer ids), not
# metrics.  They are allowed to be absent from the registry.
METADATA_FIELDS = {
    "adv_acc",
    "clean_acc",
    "num_candidate_layers",
    "num_layers",
    "sampled_layer_indices",
}


def registered_names(path: pathlib.Path = REGISTRY) -> list[str]:
    """Read ``MetricSpec("name", ...)`` calls without importing torch.

    Parsing the source instead of importing it keeps the audit runnable on a
    machine with pandas but no torch, which is also the machine that reads the
    released CSVs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        func_name = getattr(func, "id", None) or getattr(func, "attr", None)
        first = node.args[0]
        if func_name == "MetricSpec":
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.append(first.value)
        elif func_name == "_specs":
            # The registry groups most metrics through a helper that takes a
            # tuple of names, so a literal-only scan would miss 39 of 86.
            if isinstance(first, (ast.Tuple, ast.List)):
                for element in first.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        names.append(element.value)
    if not names:
        raise SystemExit(f"no MetricSpec(...) calls found in {path}")
    return sorted(set(names))


def bundle_fields(path: pathlib.Path = SRC / BUNDLE_FILE) -> set[str]:
    """Names of the annotated attributes of ``MetricResult`` in metrics.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == BUNDLE_CLASS:
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise SystemExit(f"{BUNDLE_CLASS} not found in {path}")


def source_files(src: pathlib.Path = SRC) -> list[pathlib.Path]:
    return sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts)


def mentions(files: list[pathlib.Path], name: str) -> list[str]:
    """Files whose text contains ``name`` as a whole token.

    A plain substring match would let ``feature_delta`` match
    ``feature_delta_extra``, so each hit is checked against the surrounding
    characters.  Dynamic key construction (``f"{name}_mean"``) is invisible to
    this method; that is why a hit in any non-registry file downgrades the
    verdict from "registry only" to "other module".
    """
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        start = 0
        while True:
            index = text.find(name, start)
            if index < 0:
                break
            before = text[index - 1] if index else ""
            after_index = index + len(name)
            after = text[after_index] if after_index < len(text) else ""
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                hits.append(path.name)
                break
            start = index + len(name)
    return hits


@dataclass
class Row:
    name: str
    in_bundle: bool
    files: list[str]
    emitted_finite: bool | None = None

    @property
    def other_modules(self) -> list[str]:
        return [name for name in self.files if name != REGISTRY_NAME]

    @property
    def status(self) -> str:
        if self.in_bundle:
            return "one-pass bundle"
        if self.other_modules:
            return "other module"
        if self.files:
            return "registry only"
        return "MISSING"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "in_bundle": self.in_bundle,
            "files": self.files,
            "emitted_finite": self.emitted_finite,
        }


def dynamic_fields() -> dict[str, object]:
    """Run the bundle once and return ``{field: value}``.

    Kept out of the default path because it needs torch and torchvision.
    """
    import torch
    import torchvision

    from robust_metric_lab.metrics import compute_feature_metrics

    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=10).eval()
    clean = torch.rand(8, 3, 32, 32)
    adv = (clean + 0.03 * torch.sign(torch.randn_like(clean))).clamp(0, 1)
    labels = torch.randint(0, 10, (8,))
    result = compute_feature_metrics(model, clean, adv, labels)
    return {field: getattr(result, field) for field in vars(result)}


def _finite(value: object) -> bool | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return math.isfinite(number)


def analyse(
    dynamic: bool = False,
    src: pathlib.Path = SRC,
    registry: pathlib.Path = REGISTRY,
) -> dict:
    files = source_files(src)
    fields = bundle_fields(src / BUNDLE_FILE)
    names = registered_names(registry)
    values = dynamic_fields() if dynamic else {}

    rows = []
    for name in names:
        finite = _finite(values[name]) if name in values else None
        rows.append(
            Row(
                name=name,
                in_bundle=name in fields,
                files=mentions(files, name),
                emitted_finite=finite,
            )
        )

    unknown = sorted(set(values) - set(names) - METADATA_FIELDS) if dynamic else []
    registry_only = [row.name for row in rows if row.status == "registry only"]
    missing = [row.name for row in rows if row.status == "MISSING"]
    placeholders = [
        row.name for row in rows if row.emitted_finite is False and row.other_modules
    ]
    unexplained = [
        row.name for row in rows if row.emitted_finite is False and not row.other_modules
    ]

    return {
        "registered": len(names),
        "bundle_fields": len(fields),
        "in_bundle": sum(1 for row in rows if row.in_bundle),
        "other_module": sum(1 for row in rows if row.status == "other module"),
        "registry_only": registry_only,
        "missing": missing,
        "nan_placeholders": placeholders,
        "unexplained_nonfinite": unexplained,
        "unknown_bundle_fields": unknown,
        "dynamic": dynamic,
        "rows": [row.as_dict() for row in rows],
    }


def render_markdown(audit: dict) -> str:
    lines = [
        "# 注册指标 ↔ 实现入口 对照审计（自动生成）",
        "",
        "由 `tools/audit_metric_implementation.py` 生成，不要手改。",
        "",
        f"- 注册指标：**{audit['registered']}**",
        f"- 一次性 bundle（`compute_feature_metrics`）里的指标字段：**{audit['in_bundle']}**"
        f"（`MetricResult` 共 {audit['bundle_fields']} 个字段，其余是元数据）",
        f"- 由其他模块专门计算：**{audit['other_module']}**",
        f"- 只有声明、找不到实现痕迹：**{len(audit['registry_only'])}**",
        f"- 哪儿都找不到：**{len(audit['missing'])}**",
    ]
    if audit["dynamic"]:
        lines.append(
            f"- 动态复跑：bundle 里刻意留空的 NaN **{len(audit['nan_placeholders'])}** 个"
            f"（都由专门脚本补算），无解释的非有限值 **{len(audit['unexplained_nonfinite'])}** 个"
        )
    lines += ["", "## 由其他模块计算的指标", "", "| 指标 | 出现的文件 |", "| --- | --- |"]
    for row in audit["rows"]:
        if row["status"] == "other module":
            files = [f for f in row["files"] if f != REGISTRY_NAME]
            lines.append(f"| `{row['name']}` | {', '.join(f'`{f}`' for f in files)} |")
    if audit["registry_only"]:
        lines += [
            "",
            "## ⚠ 只有声明、没有实现痕迹",
            "",
            "这些名字只在 `metric_registry.py` 里出现。要么实现从没落地，"
            "要么它写出去的键名和注册名不一致（例如用 f-string 拼出来的名字）。",
            "",
        ]
        lines += [f"- `{name}`" for name in audit["registry_only"]]
    if audit["missing"]:
        lines += ["", "## ⚠ 注册表里有、但任何文件都没提过", ""]
        lines += [f"- `{name}`" for name in audit["missing"]]
    if audit["unknown_bundle_fields"]:
        lines += ["", "## ⚠ bundle 返回了注册表不知道的字段", ""]
        lines += [f"- `{name}`" for name in audit["unknown_bundle_fields"]]
    if audit["dynamic"] and audit["nan_placeholders"]:
        lines += [
            "",
            "## bundle 里留空（NaN）、由专门脚本补算的指标",
            "",
            "这是设计，不是缺陷：`metrics.py` 在拿不到重要性向量时写 `float(\"nan\")`，"
            "真正计算在 `importance_batched.py` / `evaluate_importance_student.py`。",
            "",
        ]
        lines += [f"- `{name}`" for name in audit["nan_placeholders"]]
    lines += [
        "",
        "## 怎么读",
        "",
        "- 「一次性 bundle」= 跑 `compute_feature_metrics` 就能拿到，主线流水线每步都算它。",
        "- 「其他模块」= 需要额外条件（多视图采样、逐架构 Jacobian 上界、ODE 求解、"
        "两遍重要性向量），由专门脚本调用；不代表它算不出来，只代表它不在那个 bundle 里。",
        "- 「只有声明」才是问题：注册了一个没人写出来的名字。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="where to write the audit JSON")
    parser.add_argument("--report", required=True, help="where to write the Markdown report")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="also run the one-pass bundle once (needs torch + torchvision)",
    )
    parser.add_argument("--src", default=str(SRC), help="package directory to scan")
    parser.add_argument("--registry", default=str(REGISTRY), help="metric_registry.py path")
    args = parser.parse_args(argv)

    audit = analyse(
        dynamic=args.dynamic,
        src=pathlib.Path(args.src),
        registry=pathlib.Path(args.registry),
    )

    json_path = pathlib.Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(audit), encoding="utf-8")

    print(
        f"registered={audit['registered']} bundle={audit['in_bundle']} "
        f"other={audit['other_module']} registry_only={len(audit['registry_only'])} "
        f"missing={len(audit['missing'])}"
    )
    if audit["dynamic"]:
        print(
            f"nan_placeholders={len(audit['nan_placeholders'])} "
            f"unexplained_nonfinite={len(audit['unexplained_nonfinite'])}"
        )
    return (
        1
        if audit["registry_only"]
        or audit["missing"]
        or audit["unknown_bundle_fields"]
        or audit["unexplained_nonfinite"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
