from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


NB201_NODE_SIZES = (1, 2, 3)


@dataclass(frozen=True)
class RaclLipschitzStats:
    """Fixed-architecture counterpart of the architecture term in RACL Eq. (10)."""

    log_upper: float
    num_cells: int
    num_nodes: int
    zero_node_sums: int
    mode: str = "nb201_cell"


def _power_iteration_spectral_norm(weight: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    matrix = weight.detach().reshape(weight.shape[0], -1).float()
    if matrix.numel() == 0:
        return matrix.new_tensor(0.0)
    u = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    u = u / u.norm(p=2).clamp_min(1e-12)
    for _ in range(iterations):
        v = matrix.t().mv(u)
        v = v / v.norm(p=2).clamp_min(1e-12)
        u = matrix.mv(v)
        u = u / u.norm(p=2).clamp_min(1e-12)
    return u.dot(matrix.mv(v)).abs()


def _racl_operation_lipschitz(operation: nn.Module, power_iterations: int) -> torch.Tensor:
    """Operation constants used by Dong et al., with conv norms from power iteration."""

    device = next(operation.parameters(), torch.empty(0)).device
    one = torch.tensor(1.0, device=device)
    name = type(operation).__name__
    if name == "Zero" or getattr(operation, "is_zero", False):
        return one.new_tensor(0.0)
    if name == "Identity":
        return one

    pool_factor = one
    if name == "POOLING":
        stride = getattr(getattr(operation, "op", None), "stride", 1)
        if isinstance(stride, tuple):
            stride = max(stride)
        pool_factor = one * float(stride) ** -0.5

    convs = [module for module in operation.modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        return pool_factor
    conv_factor = torch.stack(
        [_power_iteration_spectral_norm(conv.weight, power_iterations) for conv in convs]
    ).prod()
    return pool_factor * conv_factor


def racl_architecture_lipschitz_stats(
    model: nn.Module,
    power_iterations: int = 20,
    epsilon: float = 1e-12,
) -> RaclLipschitzStats:
    """Compute the searched-cell log upper-bound term from RACL Eq. (10).

    For a discrete NAS-Bench-201 cell, operation and connection weights are one
    on selected edges. Shared macro-network factors are intentionally omitted,
    matching the architecture-dependent part optimized in the paper.
    """

    log_upper = 0.0
    num_cells = 0
    num_nodes = 0
    zero_node_sums = 0
    with torch.no_grad():
        for cell in model.modules():
            if type(cell).__name__ != "InferCell":
                continue
            if not all(hasattr(cell, field) for field in ("layers", "node_IX")):
                continue
            num_cells += 1
            for node_layers in cell.node_IX:
                node_sum = sum(
                    (
                        _racl_operation_lipschitz(cell.layers[layer_index], power_iterations)
                        for layer_index in node_layers
                    ),
                    start=torch.tensor(0.0, device=next(cell.parameters(), torch.empty(0)).device),
                )
                value = float(node_sum.item())
                num_nodes += 1
                if value <= epsilon:
                    zero_node_sums += 1
                log_upper += math.log(max(value, epsilon))
    if num_cells == 0:
        raise ValueError("No NAS-Bench-201 InferCell modules found in model.")
    return RaclLipschitzStats(log_upper, num_cells, num_nodes, zero_node_sums)


def racl_fixed_network_lipschitz_stats(
    model: nn.Module,
    power_iterations: int = 20,
    epsilon: float = 1e-12,
) -> RaclLipschitzStats:
    """Compute an auditable fixed-network counterpart of RACL Eq. (10).

    NAS-Bench-201 ``InferCell`` modules use the exact discrete-cell mapping
    implemented by :func:`racl_architecture_lipschitz_stats`. DARTS-style
    cells are recognized by their two retained operations per intermediate
    node. For conventional networks without searched cells, the function
    returns a sequential spectral-product proxy over Conv2d/Linear leaves.
    The fallback is not a strict global bound for branched residual graphs;
    its ``mode`` field makes that limitation explicit in every result row.
    """

    try:
        return racl_architecture_lipschitz_stats(
            model,
            power_iterations=power_iterations,
            epsilon=epsilon,
        )
    except ValueError:
        pass

    log_upper = 0.0
    num_cells = 0
    num_nodes = 0
    zero_node_sums = 0
    with torch.no_grad():
        for cell in model.modules():
            operations = getattr(cell, "_ops", None)
            steps = getattr(cell, "_steps", None)
            if not isinstance(operations, nn.ModuleList) or not isinstance(steps, int):
                continue
            if len(operations) != 2 * steps:
                continue
            num_cells += 1
            for node_index in range(steps):
                node_operations = operations[2 * node_index : 2 * node_index + 2]
                node_sum = sum(
                    (
                        _racl_operation_lipschitz(operation, power_iterations)
                        for operation in node_operations
                    ),
                    start=torch.tensor(
                        0.0,
                        device=next(cell.parameters(), torch.empty(0)).device,
                    ),
                )
                value = float(node_sum.item())
                num_nodes += 1
                if value <= epsilon:
                    zero_node_sums += 1
                log_upper += math.log(max(value, epsilon))
    if num_cells:
        return RaclLipschitzStats(
            log_upper,
            num_cells,
            num_nodes,
            zero_node_sums,
            mode="darts_cell",
        )

    factors: list[float] = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                factors.append(
                    float(_power_iteration_spectral_norm(module.weight, power_iterations).item())
                )
    if not factors:
        raise ValueError("No supported searched cells or Conv2d/Linear modules found in model.")
    log_upper = sum(math.log(max(value, epsilon)) for value in factors)
    return RaclLipschitzStats(
        log_upper,
        num_cells=0,
        num_nodes=len(factors),
        zero_node_sums=sum(value <= epsilon for value in factors),
        mode="generic_chain_proxy",
    )


@dataclass(frozen=True)
class NasRobRecord:
    arch_id: str
    canonical_id: str
    arch_string: str | None
    dataset: str
    metrics: dict[str, float]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        try:
            x = float(value)
        except ValueError:
            return None
    elif isinstance(value, dict):
        if "1" in value:
            return _as_float(value["1"])
        for candidate in ("mean", "value", "acc", "accuracy"):
            if candidate in value:
                return _as_float(value[candidate])
        return None
    else:
        return None
    if not math.isfinite(x):
        return None
    return x


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(meta_path: str | Path | None) -> dict[str, Any]:
    if not meta_path:
        return {"ids": {}}
    return load_json(meta_path)


def canonical_id(meta: dict[str, Any], arch_id: str | int) -> str:
    ids = meta.get("ids", {})
    item = ids.get(str(arch_id))
    if isinstance(item, dict) and "isomorph" in item:
        return str(item["isomorph"])
    return str(arch_id)


def arch_string(meta: dict[str, Any], arch_id: str | int) -> str | None:
    ids = meta.get("ids", {})
    item = ids.get(str(arch_id))
    if isinstance(item, dict):
        value = item.get("nb201-string") or item.get("arch") or item.get("arch_str")
        if value:
            return str(value)
    return None


def iter_records(
    dataset_json: dict[str, Any],
    dataset: str,
    meta: dict[str, Any] | None = None,
) -> list[NasRobRecord]:
    meta = meta or {"ids": {}}
    if dataset not in dataset_json:
        available = ", ".join(sorted(dataset_json.keys()))
        raise KeyError(f"Dataset {dataset!r} not found. Available datasets: {available}")

    dataset_blob = dataset_json[dataset]
    records: list[NasRobRecord] = []
    for raw_arch_id, raw_metrics in dataset_blob.items():
        metrics: dict[str, float] = {}
        if not isinstance(raw_metrics, dict):
            continue
        for key, value in raw_metrics.items():
            number = _as_float(value)
            if number is not None:
                metrics[key] = number
        cid = canonical_id(meta, raw_arch_id)
        records.append(
            NasRobRecord(
                arch_id=str(raw_arch_id),
                canonical_id=cid,
                arch_string=arch_string(meta, raw_arch_id) or arch_string(meta, cid),
                dataset=dataset,
                metrics=metrics,
            )
        )
    return records


def metric_names(records: Iterable[NasRobRecord]) -> list[str]:
    names: set[str] = set()
    for record in records:
        names.update(record.metrics.keys())
    return sorted(names)


def parse_nb201_arch(arch: str | None) -> tuple[str, ...] | None:
    if not arch:
        return None
    ops = re.findall(r"\|([^|~]+)~\d+", arch)
    if len(ops) != 6:
        return None
    return tuple(ops)


def make_nb201_arch(ops: Iterable[str]) -> str:
    ops = list(ops)
    if len(ops) != 6:
        raise ValueError("NAS-Bench-201 architecture encoding must contain 6 operations.")
    parts = []
    offset = 0
    for node, size in enumerate(NB201_NODE_SIZES, start=1):
        incoming = []
        for src in range(size):
            incoming.append(f"|{ops[offset]}~{src}")
            offset += 1
        parts.append("".join(incoming) + "|")
    return "+".join(parts)


def arch_map_from_meta(meta: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for arch_id, item in meta.get("ids", {}).items():
        if not isinstance(item, dict):
            continue
        arch = item.get("nb201-string") or item.get("arch") or item.get("arch_str")
        if arch:
            mapping[str(arch)] = str(arch_id)
    return mapping
