from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MetricResult:
    feature_delta: float
    feature_measure_swd: float
    feature_measure_mmd: float
    feature_measure_cov_delta: float
    feature_hausdorff_delta: float
    feature_quantile_hausdorff_delta: float
    feature_box_dimension_delta: float
    feature_graph_fundamental_rank_delta: float
    feature_measure_moment_delta: float
    feature_measure_local_delta: float
    feature_measure_weighted_delta: float
    feature_measure_struct_delta: float
    feature_measure_struct01_delta: float
    perturbation_capacity_delta: float
    perturbation_capacity_relu_active_delta: float
    perturbation_capacity_above_mean_delta: float
    logit_kl: float
    natural_loss: float
    adversarial_loss: float
    input_loss_delta: float
    input_grad_l2: float
    local_logit_lipschitz: float
    margin_input_grad_ratio: float
    feature_measure_flow_delta: float
    feature_measure_flow_rel_delta: float
    feature_measure_spatial_gram_delta: float
    feature_measure_spatial_gram_rel_delta: float
    feature_measure_relu_active_delta: float
    feature_measure_relu_active_spatial_delta: float
    feature_measure_relu_active_spatial_rel_delta: float
    feature_measure_relu_active_hybrid_delta: float
    feature_measure_relu_active_hybrid_rel_delta: float
    feature_measure_robust_delta: float
    feature_measure_robust_rel_delta: float
    feature_measure_robust_spatial_delta: float
    feature_measure_robust_spatial_rel_delta: float
    feature_measure_robust_hybrid_delta: float
    feature_measure_robust_hybrid_rel_delta: float
    fsp_strict_delta: float
    fsp_delta: float
    clean_acc: float
    adv_acc: float
    num_layers: int
    num_candidate_layers: int
    sampled_layer_indices: str


GEOMETRY_METRIC_FIELDS = (
    "feature_hausdorff_delta",
    "feature_quantile_hausdorff_delta",
    "feature_box_dimension_delta",
    "feature_graph_fundamental_rank_delta",
)


@dataclass
class GeometryMetricResult:
    feature_hausdorff_delta: float
    feature_quantile_hausdorff_delta: float
    feature_box_dimension_delta: float
    feature_graph_fundamental_rank_delta: float
    num_layers: int
    num_candidate_layers: int
    sampled_layer_indices: str


class ActivationRecorder:
    def __init__(self, model: torch.nn.Module, activation_mode: str = "all_leaf"):
        if activation_mode not in {"all_leaf", "post_relu", "post_activation"}:
            raise ValueError(f"Unsupported activation mode: {activation_mode}")
        self.model = model
        self.activation_mode = activation_mode
        self.handles = []
        self.outputs: list[torch.Tensor] = []
        self._functional_relu = None
        self._torch_relu = None
        self._relu_call_depth = 0

    def __enter__(self) -> "ActivationRecorder":
        def hook(_module, _inputs, output):
            if torch.is_tensor(output) and output.dim() == 4:
                self.outputs.append(output.detach())

        if self.activation_mode in {"post_relu", "post_activation"}:
            self._functional_relu = F.relu
            self._torch_relu = torch.relu

            def record_relu_call(function, *args, **kwargs):
                outermost = self._relu_call_depth == 0
                self._relu_call_depth += 1
                try:
                    output = function(*args, **kwargs)
                finally:
                    self._relu_call_depth -= 1
                if outermost:
                    hook(None, None, output)
                return output

            def functional_relu(input_tensor, inplace=False):
                return record_relu_call(self._functional_relu, input_tensor, inplace=inplace)

            def torch_relu(input_tensor):
                return record_relu_call(self._torch_relu, input_tensor)

            F.relu = functional_relu
            torch.relu = torch_relu
            if self.activation_mode == "post_relu":
                return self

        # 这儿可以考虑只给特定层级的feature maps记录 ReLU之后的
        for module in self.model.modules():
            is_leaf = len(list(module.children())) == 0
            non_relu_activation_types = (
                torch.nn.ReLU6,
                torch.nn.LeakyReLU,
                torch.nn.PReLU,
                torch.nn.ELU,
                torch.nn.CELU,
                torch.nn.GELU,
                torch.nn.SiLU,
                torch.nn.Mish,
                torch.nn.Hardswish,
            )
            is_native_non_relu_activation = (
                isinstance(module, non_relu_activation_types)
                or type(module).__name__.lower() in {"swish", "mish"}
            )
            if is_leaf and (
                self.activation_mode == "all_leaf" or is_native_non_relu_activation
            ):
                self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        if self._functional_relu is not None:
            F.relu = self._functional_relu
        if self._torch_relu is not None:
            torch.relu = self._torch_relu

# 
def _anchor_indices(num_outputs: int, max_layers: int = 4) -> list[int]:
    if num_outputs <= max_layers:
        return list(range(num_outputs))
    
    # 均匀取层数的feature maps
    fractions = torch.linspace(0, num_outputs - 1, steps=max_layers)
    return sorted({int(round(v.item())) for v in fractions})


def _geometry_anchor_indices(
    features: list[torch.Tensor],
    max_layers: int = 4,
    min_positions: int = 16,
) -> tuple[list[int], int]:
    candidates = [
        index
        for index, feature in enumerate(features)
        if feature.shape[2] * feature.shape[3] >= min_positions
    ]
    selected = _anchor_indices(len(candidates), max_layers=max_layers)
    return [candidates[index] for index in selected], len(candidates)


def _pooled_normalized(feature: torch.Tensor) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
    return F.normalize(pooled, p=2, dim=1, eps=1e-12)

# => [B, C, H*W] => [B, H*W, C]
def _spatial_atoms(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2)


def _fixed_projections(channels: int, num_projections: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1729 + channels * 31 + num_projections)
    projections = torch.randn(channels, num_projections, generator=generator, dtype=torch.float32)
    projections = F.normalize(projections, p=2, dim=0, eps=1e-12)
    return projections.to(device=device, dtype=dtype)


def _fixed_rff(
    channels: int,
    num_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7919 + channels * 17 + num_features)
    omega = torch.randn(channels, num_features, generator=generator, dtype=torch.float32)
    bias = torch.empty(num_features, dtype=torch.float32).uniform_(0.0, 2.0 * torch.pi, generator=generator)
    return omega.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)


def _standardized_atoms(clean: torch.Tensor, adv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 这个过程中将空间维度=>1, channel维度=>2
    clean_atoms = _spatial_atoms(clean)
    adv_atoms = _spatial_atoms(adv)
    # channel 维度
    clean_mean = clean_atoms.mean(dim=1, keepdim=True)
    clean_std = clean_atoms.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (clean_atoms - clean_mean) / clean_std, (adv_atoms - clean_mean) / clean_std

# 把channel维度投影到 16 个方向上，然后计算 Wasserstein 距离
def _feature_measure_swd(clean: torch.Tensor, adv: torch.Tensor, num_projections: int = 16) -> torch.Tensor:
    clean_atoms, adv_atoms = _standardized_atoms(clean, adv)
    projections = _fixed_projections(clean_atoms.shape[-1], num_projections, clean_atoms.device, clean_atoms.dtype)
    clean_proj = clean_atoms @ projections
    adv_proj = adv_atoms @ projections
    # 空间维度进行排序
    clean_sorted = clean_proj.sort(dim=1).values
    adv_sorted = adv_proj.sort(dim=1).values
    return (clean_sorted - adv_sorted).pow(2).mean(dim=(1, 2)).sqrt().mean()


def _feature_measure_mmd(clean: torch.Tensor, adv: torch.Tensor, num_features: int = 64) -> torch.Tensor:
    clean_atoms, adv_atoms = _standardized_atoms(clean, adv)
    omega, bias = _fixed_rff(clean_atoms.shape[-1], num_features, clean_atoms.device, clean_atoms.dtype)
    scale = (2.0 / num_features) ** 0.5
    clean_phi = torch.cos(clean_atoms @ omega + bias) * scale
    adv_phi = torch.cos(adv_atoms @ omega + bias) * scale
    return (adv_phi.mean(dim=1) - clean_phi.mean(dim=1)).norm(p=2, dim=1).mean()


def _feature_measure_cov_delta(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    clean_atoms = _spatial_atoms(clean)
    adv_atoms = _spatial_atoms(adv)
    clean_centered = clean_atoms - clean_atoms.mean(dim=1, keepdim=True)
    adv_centered = adv_atoms - adv_atoms.mean(dim=1, keepdim=True)
    clean_cov = torch.bmm(clean_centered.transpose(1, 2), clean_centered) / clean_centered.shape[1]
    adv_cov = torch.bmm(adv_centered.transpose(1, 2), adv_centered) / adv_centered.shape[1]
    scale = clean_cov.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)
    return ((adv_cov - clean_cov).flatten(1).norm(p=2, dim=1) / scale).mean()


def _geometry_point_clouds(
    clean: torch.Tensor,
    adv: torch.Tensor,
    max_positions: int = 64,
    projection_dim: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map corresponding feature maps to one clean-normalized Euclidean frame."""
    if clean.shape != adv.shape:
        raise ValueError("clean and adv feature maps must have the same shape")
    spatial_size = clean.shape[2] * clean.shape[3]
    if spatial_size > max_positions:
        side = max(1, int(max_positions**0.5))
        clean = F.adaptive_avg_pool2d(clean, (side, side))
        adv = F.adaptive_avg_pool2d(adv, (side, side))

    clean_atoms = _spatial_atoms(clean)
    adv_atoms = _spatial_atoms(adv)
    clean_centered = clean_atoms - clean_atoms.mean(dim=1, keepdim=True)
    adv_centered = adv_atoms - adv_atoms.mean(dim=1, keepdim=True)

    dimensions = min(projection_dim, clean_atoms.shape[-1])
    projections = _fixed_projections(
        clean_atoms.shape[-1],
        dimensions,
        clean_atoms.device,
        clean_atoms.dtype,
    )
    clean_projected = clean_centered @ projections
    adv_projected = adv_centered @ projections
    clean_energy = clean_projected.pow(2).sum(dim=2).mean(dim=1, keepdim=True)
    adv_energy = adv_projected.pow(2).sum(dim=2).mean(dim=1, keepdim=True)
    shared_scale = (0.5 * (clean_energy + adv_energy)).sqrt().clamp_min(1e-6)
    return clean_projected / shared_scale.unsqueeze(2), adv_projected / shared_scale.unsqueeze(2)


def _box_counting_dimension(
    points: torch.Tensor,
    origins: torch.Tensor,
    base_scales: torch.Tensor,
    ratios: tuple[float, ...] = (0.125, 0.2, 0.32, 0.5),
) -> torch.Tensor:
    """Finite-scale grid estimate of Minkowski/box-counting dimension."""
    counts_by_scale = []
    for ratio in ratios:
        width = (base_scales * ratio).clamp_min(1e-6)
        coordinates = torch.floor((points - origins) / width.unsqueeze(2)).to(torch.int64)
        counts = [
            torch.unique(coordinates[index], dim=0).shape[0]
            for index in range(coordinates.shape[0])
        ]
        counts_by_scale.append(
            torch.tensor(counts, device=points.device, dtype=points.dtype).clamp_min(1.0)
        )

    x = points.new_tensor([torch.log(torch.tensor(1.0 / ratio)) for ratio in ratios])
    x = x - x.mean()
    y = torch.stack(counts_by_scale, dim=1).log()
    y = y - y.mean(dim=1, keepdim=True)
    return (y * x).sum(dim=1) / x.pow(2).sum().clamp_min(1e-8)


def _minimum_spanning_tree_adjacency(distances: torch.Tensor) -> torch.Tensor:
    """Batched Prim trees used to connect each finite neighborhood graph."""
    batch, points, _ = distances.shape
    selected = torch.zeros(batch, points, dtype=torch.bool, device=distances.device)
    selected[:, 0] = True
    min_distance = distances[:, 0].clone()
    parent = torch.zeros(batch, points, dtype=torch.long, device=distances.device)
    tree = torch.zeros(batch, points, points, dtype=torch.bool, device=distances.device)
    batch_indices = torch.arange(batch, device=distances.device)

    for _ in range(points - 1):
        candidates = min_distance.masked_fill(selected, float("inf"))
        next_point = candidates.argmin(dim=1)
        next_parent = parent.gather(1, next_point.unsqueeze(1)).squeeze(1)
        tree[batch_indices, next_point, next_parent] = True
        tree[batch_indices, next_parent, next_point] = True
        selected[batch_indices, next_point] = True
        new_distance = distances[batch_indices, next_point]
        improve = (~selected) & (new_distance < min_distance)
        min_distance = torch.where(improve, new_distance, min_distance)
        parent = torch.where(improve, next_point.unsqueeze(1).expand_as(parent), parent)
    return tree


def _graph_fundamental_rank_delta(
    clean_points: torch.Tensor,
    adv_points: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> torch.Tensor:
    """Compare ranks of free fundamental groups of multiscale proximity graphs."""
    if clean_points.shape[1] > 32:
        indices = torch.linspace(
            0,
            clean_points.shape[1] - 1,
            steps=32,
            device=clean_points.device,
        ).round().long()
        clean_points = clean_points.index_select(1, indices)
        adv_points = adv_points.index_select(1, indices)

    clean_distances = torch.cdist(
        clean_points, clean_points, compute_mode="donot_use_mm_for_euclid_dist"
    )
    adv_distances = torch.cdist(
        adv_points, adv_points, compute_mode="donot_use_mm_for_euclid_dist"
    )
    points = clean_points.shape[1]
    if points < 2:
        return clean_points.new_zeros(clean_points.shape[0])
    diagonal = torch.eye(points, dtype=torch.bool, device=clean_points.device).unsqueeze(0)
    upper = torch.triu(
        torch.ones(points, points, dtype=torch.bool, device=clean_points.device), diagonal=1
    )
    clean_pairwise = clean_distances[:, upper]
    thresholds = torch.quantile(
        clean_pairwise,
        clean_points.new_tensor(quantiles),
        dim=1,
    ).transpose(0, 1)
    clean_tree = _minimum_spanning_tree_adjacency(clean_distances)
    adv_tree = _minimum_spanning_tree_adjacency(adv_distances)

    discrepancies = []
    for scale_index in range(len(quantiles)):
        threshold = thresholds[:, scale_index].view(-1, 1, 1)
        clean_graph = ((clean_distances <= threshold) & ~diagonal) | clean_tree
        adv_graph = ((adv_distances <= threshold) & ~diagonal) | adv_tree
        clean_edges = clean_graph[:, upper].sum(dim=1).to(clean_points.dtype)
        adv_edges = adv_graph[:, upper].sum(dim=1).to(clean_points.dtype)
        clean_rank = clean_edges - points + 1
        adv_rank = adv_edges - points + 1
        discrepancies.append((adv_rank - clean_rank).abs() / clean_rank.clamp_min(1.0))
    return torch.stack(discrepancies, dim=1).mean(dim=1)


def _feature_geometry_stats(
    clean: torch.Tensor,
    adv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_points, adv_points = _geometry_point_clouds(clean, adv)
    cross_distances = torch.cdist(
        clean_points, adv_points, compute_mode="donot_use_mm_for_euclid_dist"
    )
    clean_to_adv = cross_distances.min(dim=2).values
    adv_to_clean = cross_distances.min(dim=1).values
    hausdorff = torch.maximum(clean_to_adv.max(dim=1).values, adv_to_clean.max(dim=1).values)
    quantile_hausdorff = torch.maximum(
        torch.quantile(clean_to_adv, 0.95, dim=1),
        torch.quantile(adv_to_clean, 0.95, dim=1),
    )

    clean_box, adv_box = _geometry_point_clouds(clean, adv, projection_dim=3)
    origin = clean_box.amin(dim=1, keepdim=True)
    extent = clean_box.amax(dim=1, keepdim=True) - origin
    base_scale = extent.amax(dim=2).clamp_min(1e-6)
    clean_dimension = _box_counting_dimension(clean_box, origin, base_scale)
    adv_dimension = _box_counting_dimension(adv_box, origin, base_scale)
    box_delta = (adv_dimension - clean_dimension).abs()
    graph_delta = _graph_fundamental_rank_delta(clean_points, adv_points)
    return (
        hausdorff.mean(),
        quantile_hausdorff.mean(),
        box_delta.mean(),
        graph_delta.mean(),
    )


def compute_feature_geometry_metrics(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    max_layers: int = 4,
    activation_mode: str = "all_leaf",
) -> GeometryMetricResult:
    """Compute only the geometric/fractal metrics for checkpoint re-evaluation."""
    model.eval()
    with torch.no_grad():
        with ActivationRecorder(model, activation_mode=activation_mode) as clean_recorder:
            model(x_clean)
        with ActivationRecorder(model, activation_mode=activation_mode) as adv_recorder:
            model(x_adv)

    n = min(len(clean_recorder.outputs), len(adv_recorder.outputs))
    clean_features = clean_recorder.outputs[:n]
    adv_features = adv_recorder.outputs[:n]
    indices, candidate_count = _geometry_anchor_indices(
        clean_features, max_layers=max_layers
    )
    values = [[] for _ in GEOMETRY_METRIC_FIELDS]
    for index in indices:
        layer_values = _feature_geometry_stats(
            clean_features[index], adv_features[index]
        )
        for bucket, value in zip(values, layer_values):
            bucket.append(value)

    means = [
        torch.stack(bucket).mean().item() if bucket else float("nan") for bucket in values
    ]
    return GeometryMetricResult(
        **dict(zip(GEOMETRY_METRIC_FIELDS, means)),
        num_layers=len(indices),
        num_candidate_layers=candidate_count,
        sampled_layer_indices=",".join(str(index) for index in indices),
    )

# feature_measure_local_delta
def _feature_measure_local_delta(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    # => [B, C, H*W] => [B, H*W, C]
    clean_atoms = F.normalize(_spatial_atoms(clean), p=2, dim=2, eps=1e-12)
    adv_atoms = F.normalize(_spatial_atoms(adv), p=2, dim=2, eps=1e-12)
    return (adv_atoms - clean_atoms).norm(p=2, dim=2).mean()


def _feature_measure_weighted_delta(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    clean_atoms_raw = _spatial_atoms(clean)
    weights = clean_atoms_raw.norm(p=2, dim=2)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    clean_atoms = F.normalize(clean_atoms_raw, p=2, dim=2, eps=1e-12)
    adv_atoms = F.normalize(_spatial_atoms(adv), p=2, dim=2, eps=1e-12)
    local_delta = (adv_atoms - clean_atoms).norm(p=2, dim=2)
    return (weights * local_delta).sum(dim=1).mean()


def _perturbation_capacity(feature: torch.Tensor) -> torch.Tensor:
    centered = feature - feature.mean(dim=(2, 3), keepdim=True)
    return centered.abs().mean(dim=(2, 3)).sum(dim=1)


def _perturbation_capacity_on_support(feature: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if feature.shape != support.shape:
        raise ValueError("feature and support must have the same shape")

    active = support.to(feature.dtype)
    active_count = active.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    active_mean = (feature * active).sum(dim=(2, 3), keepdim=True) / active_count
    active_mad = ((feature - active_mean).abs() * active).sum(dim=(2, 3), keepdim=True) / active_count
    return active_mad.flatten(1).sum(dim=1)


def _perturbation_capacity_relu_active(
    clean: torch.Tensor,
    feature: torch.Tensor,
) -> torch.Tensor:
    """Capacity on the clean feature's positive (ReLU-active) support."""
    if clean.shape != feature.shape:
        raise ValueError("clean and feature must have the same shape")
    return _perturbation_capacity_on_support(feature, clean > 0)


def _perturbation_capacity_above_mean(
    clean: torch.Tensor,
    feature: torch.Tensor,
) -> torch.Tensor:
    """Capacity where clean activations exceed their per-channel spatial mean."""
    if clean.shape != feature.shape:
        raise ValueError("clean and feature must have the same shape")
    clean_mean = clean.mean(dim=(2, 3), keepdim=True)
    return _perturbation_capacity_on_support(feature, clean > clean_mean)


def _channel_gram(feature: torch.Tensor) -> torch.Tensor:
    # 展开spacial维度，得到 [B, C, H*W]，然后计算 Gram 矩阵
    flat = feature.flatten(2)
    flat = F.normalize(flat, p=2, dim=2, eps=1e-12)
    return torch.bmm(flat, flat.transpose(1, 2)) / flat.shape[2]


def _feature_measure_spatial_gram_stats(
    clean: torch.Tensor,
    adv: torch.Tensor,
    max_positions: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_size = clean.shape[2] * clean.shape[3]
    if spatial_size > max_positions:
        side = int(max_positions**0.5)
        clean = F.adaptive_avg_pool2d(clean, (side, side))
        adv = F.adaptive_avg_pool2d(adv, (side, side))

    clean_atoms = F.normalize(_spatial_atoms(clean), p=2, dim=2, eps=1e-12)
    adv_atoms = F.normalize(_spatial_atoms(adv), p=2, dim=2, eps=1e-12)

    clean_gram = torch.bmm(clean_atoms, clean_atoms.transpose(1, 2))
    adv_gram = torch.bmm(adv_atoms, adv_atoms.transpose(1, 2))
    diff = (adv_gram - clean_gram).flatten(1)
    gram_delta = (diff.norm(p=2, dim=1) / clean_atoms.shape[1]).mean()
    gram_rel_delta = (diff.norm(p=2, dim=1) / clean_gram.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)).mean()
    return gram_delta, gram_rel_delta


def _relu_active_spatial_weights(clean: torch.Tensor) -> torch.Tensor:
    # 
    clean_atoms = _spatial_atoms(clean)
    weights = (clean_atoms > 0).to(clean_atoms.dtype).mean(dim=2)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _feature_measure_relu_active_stats(
    clean: torch.Tensor,
    adv: torch.Tensor,
    max_positions: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial_size = clean.shape[2] * clean.shape[3]
    if spatial_size > max_positions:
        side = int(max_positions**0.5)
        clean = F.adaptive_avg_pool2d(clean, (side, side))
        adv = F.adaptive_avg_pool2d(adv, (side, side))

    weights = _relu_active_spatial_weights(clean)
    clean_atoms = F.normalize(_spatial_atoms(clean), p=2, dim=2, eps=1e-12)
    adv_atoms = F.normalize(_spatial_atoms(adv), p=2, dim=2, eps=1e-12)

    local_delta = (weights * (adv_atoms - clean_atoms).norm(p=2, dim=2)).sum(dim=1).mean()

    sqrt_weights = weights.clamp_min(0.0).sqrt().unsqueeze(2)
    clean_weighted = clean_atoms * sqrt_weights
    adv_weighted = adv_atoms * sqrt_weights
    clean_gram = torch.bmm(clean_weighted, clean_weighted.transpose(1, 2))
    adv_gram = torch.bmm(adv_weighted, adv_weighted.transpose(1, 2))
    
    diff = (adv_gram - clean_gram).flatten(1)
    gram_delta = diff.norm(p=2, dim=1).mean()
    gram_rel_delta = (diff.norm(p=2, dim=1) / clean_gram.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)).mean()
    return local_delta, gram_delta, gram_rel_delta


def _match_spatial(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(a.shape[2], b.shape[2])
    w = min(a.shape[3], b.shape[3])

    if a.shape[2:] != (h, w):
        a = F.adaptive_avg_pool2d(a, (h, w))
    if b.shape[2:] != (h, w):
        b = F.adaptive_avg_pool2d(b, (h, w))
    return a, b


def _fsp_matrix(input_feature: torch.Tensor, output_feature: torch.Tensor) -> torch.Tensor:
    # input output 对齐spacial维度
    input_feature, output_feature = _match_spatial(input_feature, output_feature)
    input_atoms = input_feature.flatten(2)
    output_atoms = output_feature.flatten(2)
    # 
    return torch.bmm(input_atoms, output_atoms.transpose(1, 2)) / input_atoms.shape[2]


def _fsp_corr_matrix(input_feature: torch.Tensor, output_feature: torch.Tensor) -> torch.Tensor:
    """Cross-layer channel correlation (cosine) matrix, channel-wise L2 normalized."""
    input_feature, output_feature = _match_spatial(input_feature, output_feature)
    input_atoms = F.normalize(input_feature.flatten(2), p=2, dim=2, eps=1e-12)
    output_atoms = F.normalize(output_feature.flatten(2), p=2, dim=2, eps=1e-12)
    return torch.bmm(input_atoms, output_atoms.transpose(1, 2)) / input_atoms.shape[2]


def _safe_cosine_distance(
    clean_value: torch.Tensor,
    adv_value: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-sample cosine distance with an exact zero for two zero vectors."""
    clean_flat = clean_value.flatten(1)
    adv_flat = adv_value.flatten(1)
    clean_norm = clean_flat.norm(p=2, dim=1)
    adv_norm = adv_flat.norm(p=2, dim=1)
    denom = clean_norm * adv_norm
    cosine = (clean_flat * adv_flat).sum(dim=1) / denom.clamp_min(eps)
    distance = (1.0 - cosine.clamp(-1.0, 1.0)).clamp_min(0.0)
    distance = torch.where(distance <= 1e-6, torch.zeros_like(distance), distance)
    both_zero = (clean_norm <= eps) & (adv_norm <= eps)
    return torch.where(both_zero, torch.zeros_like(distance), distance)


def _fsp_strict_stats(
    clean_input: torch.Tensor,
    clean_output: torch.Tensor,
    adv_input: torch.Tensor,
    adv_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_fsp = _fsp_matrix(clean_input, clean_output)
    adv_fsp = _fsp_matrix(adv_input, adv_output)
    
    diff = (adv_fsp - clean_fsp).flatten(1)
    strict_loss = diff.pow(2).sum(dim=1).mean()

    rms_delta = diff.pow(2).mean(dim=1).sqrt().mean()
    rel_delta = (diff.norm(p=2, dim=1) / clean_fsp.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)).mean()
    return strict_loss, rms_delta, rel_delta


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def compute_feature_metrics(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    labels: torch.Tensor,
    max_layers: int = 4,
    activation_mode: str = "all_leaf",
) -> MetricResult:
    model.eval()

    with torch.no_grad():
        with ActivationRecorder(model, activation_mode=activation_mode) as clean_recorder:
            clean_logits = model(x_clean)

        clean_features = clean_recorder.outputs

        with ActivationRecorder(model, activation_mode=activation_mode) as adv_recorder:
            adv_logits = model(x_adv)
        adv_features = adv_recorder.outputs

    n = min(len(clean_features), len(adv_features))
    # 
    indices = _anchor_indices(n, max_layers=max_layers)
    geometry_indices, _ = _geometry_anchor_indices(
        clean_features[:n], max_layers=max_layers
    )

    feature_deltas = []
    feature_measure_swds = []
    feature_measure_mmds = []
    feature_measure_cov_deltas = []
    feature_hausdorff_deltas = []
    feature_quantile_hausdorff_deltas = []
    feature_box_dimension_deltas = []
    feature_graph_fundamental_rank_deltas = []
    feature_measure_moment_deltas = []
    feature_measure_local_deltas = []
    feature_measure_weighted_deltas = []
    feature_measure_struct_deltas = []
    feature_measure_struct01_deltas = []
    feature_measure_spatial_gram_deltas = []
    feature_measure_spatial_gram_rel_deltas = []
    feature_measure_relu_active_deltas = []
    feature_measure_relu_active_spatial_deltas = []
    feature_measure_relu_active_spatial_rel_deltas = []
    feature_measure_robust_spatial_deltas = []
    feature_measure_robust_spatial_rel_deltas = []
    capacity_deltas = []
    capacity_relu_active_deltas = []
    capacity_above_mean_deltas = []
    fsp_deltas = []
    layer_weighted_by_idx = {}

    for idx in indices:
        clean = clean_features[idx]
        adv = adv_features[idx]

        layer_feature_delta = (_pooled_normalized(adv) - _pooled_normalized(clean)).norm(p=2, dim=1).mean()
        feature_deltas.append(layer_feature_delta)
        feature_measure_swds.append(_feature_measure_swd(clean, adv))
        feature_measure_mmds.append(_feature_measure_mmd(clean, adv))
        feature_measure_cov_deltas.append(_feature_measure_cov_delta(clean, adv))
        feature_measure_local_deltas.append(_feature_measure_local_delta(clean, adv))

        layer_weighted_delta = _feature_measure_weighted_delta(clean, adv)
        layer_weighted_by_idx[idx] = layer_weighted_delta
        feature_measure_weighted_deltas.append(layer_weighted_delta)

        # spatial gram matrix 256 位置值得商榷？
        layer_spatial_gram_delta, layer_spatial_gram_rel_delta = _feature_measure_spatial_gram_stats(clean, adv)
        feature_measure_spatial_gram_deltas.append(layer_spatial_gram_delta)
        feature_measure_spatial_gram_rel_deltas.append(layer_spatial_gram_rel_delta)

        # activate
        layer_relu_active_delta, layer_relu_active_spatial_delta, layer_relu_active_spatial_rel_delta = (
            _feature_measure_relu_active_stats(clean, adv)
        )
        feature_measure_relu_active_deltas.append(layer_relu_active_delta)
        feature_measure_relu_active_spatial_deltas.append(layer_relu_active_spatial_delta)
        feature_measure_relu_active_spatial_rel_deltas.append(layer_relu_active_spatial_rel_delta)
        feature_measure_robust_spatial_deltas.append(layer_spatial_gram_delta + layer_weighted_delta)
        feature_measure_robust_spatial_rel_deltas.append(layer_spatial_gram_rel_delta + layer_weighted_delta)
        clean_capacity = _perturbation_capacity(clean)
        adv_capacity = _perturbation_capacity(adv)
        capacity_deltas.append(((adv_capacity - clean_capacity).abs() / (clean_capacity.abs() + 1e-8)).mean())

        clean_relu_active_capacity = _perturbation_capacity_relu_active(clean, clean)
        adv_relu_active_capacity = _perturbation_capacity_relu_active(clean, adv)
        capacity_relu_active_deltas.append(
            (
                (adv_relu_active_capacity - clean_relu_active_capacity).abs()
                / (clean_relu_active_capacity.abs() + 1e-8)
            ).mean()
        )

        clean_above_mean_capacity = _perturbation_capacity_above_mean(clean, clean)
        adv_above_mean_capacity = _perturbation_capacity_above_mean(clean, adv)
        capacity_above_mean_deltas.append(
            (
                (adv_above_mean_capacity - clean_above_mean_capacity).abs()
                / (clean_above_mean_capacity.abs() + 1e-8)
            ).mean()
        )

        # 原始论文中的G_l(x, theta)
        clean_gram = _channel_gram(clean)
        adv_gram = _channel_gram(adv)
        fsp_scale = clean_gram.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)
        layer_fsp_delta = ((adv_gram - clean_gram).flatten(1).norm(p=2, dim=1) / fsp_scale).mean()
        fsp_deltas.append(layer_fsp_delta)

        # 
        feature_measure_moment_deltas.append((layer_feature_delta.pow(2) + layer_fsp_delta.pow(2)).sqrt())
        feature_measure_struct_deltas.append((layer_weighted_delta.pow(2) + layer_fsp_delta.pow(2)).sqrt())
        feature_measure_struct01_deltas.append((0.1 * layer_weighted_delta.pow(2) + layer_fsp_delta.pow(2)).sqrt())

    for idx in geometry_indices:
        layer_values = _feature_geometry_stats(clean_features[idx], adv_features[idx])
        feature_hausdorff_deltas.append(layer_values[0])
        feature_quantile_hausdorff_deltas.append(layer_values[1])
        feature_box_dimension_deltas.append(layer_values[2])
        feature_graph_fundamental_rank_deltas.append(layer_values[3])

    fsp_strict_deltas = []
    feature_measure_flow_deltas = []
    feature_measure_flow_rel_deltas = []
    feature_measure_robust_deltas = []
    feature_measure_robust_rel_deltas = []
    for left_idx, right_idx in zip(indices, indices[1:]):
        layer_fsp_strict_delta, layer_fsp_rms_delta, layer_fsp_rel_delta = _fsp_strict_stats(
            clean_features[left_idx],
            clean_features[right_idx],
            adv_features[left_idx],
            adv_features[right_idx],
        )
        fsp_strict_deltas.append(layer_fsp_strict_delta)
        feature_measure_flow_deltas.append(layer_fsp_rms_delta)
        feature_measure_flow_rel_deltas.append(layer_fsp_rel_delta)

        local_pair_delta = 0.5 * (layer_weighted_by_idx[left_idx] + layer_weighted_by_idx[right_idx])
        feature_measure_robust_deltas.append(layer_fsp_rms_delta + local_pair_delta)
        feature_measure_robust_rel_deltas.append(layer_fsp_rel_delta + local_pair_delta)

    clean_probs = F.softmax(clean_logits, dim=1)
    adv_log_probs = F.log_softmax(adv_logits, dim=1)
    logit_kl = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")
    natural_loss = F.cross_entropy(clean_logits, labels)
    adversarial_loss = F.cross_entropy(adv_logits, labels)
    input_loss_delta = adversarial_loss - natural_loss
    logit_step = (adv_logits - clean_logits).flatten(1).norm(p=2, dim=1)
    input_step = (x_adv - x_clean).flatten(1).norm(p=2, dim=1).clamp_min(1e-8)
    local_logit_lipschitz = (logit_step / input_step).mean()

    x_for_grad = x_clean.detach().clone().requires_grad_(True)
    grad_logits = model(x_for_grad)
    grad_loss = F.cross_entropy(grad_logits, labels)
    input_grad = torch.autograd.grad(grad_loss, x_for_grad)[0] * x_for_grad.shape[0]
    input_grad_l2_per_sample = input_grad.flatten(1).norm(p=2, dim=1)

    true_logits = grad_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    other_logits = grad_logits.masked_fill(F.one_hot(labels, grad_logits.shape[1]).bool(), float("-inf"))
    margins = true_logits - other_logits.max(dim=1).values
    margin_input_grad_ratio = (margins.clamp_min(0.0) / input_grad_l2_per_sample.clamp_min(1e-8)).mean()

    if feature_deltas:
        feature_delta = torch.stack(feature_deltas).mean().item()
        feature_measure_swd = torch.stack(feature_measure_swds).mean().item()
        feature_measure_mmd = torch.stack(feature_measure_mmds).mean().item()
        feature_measure_cov_delta = torch.stack(feature_measure_cov_deltas).mean().item()
        if feature_hausdorff_deltas:
            feature_hausdorff_delta = torch.stack(feature_hausdorff_deltas).mean().item()
            feature_quantile_hausdorff_delta = torch.stack(
                feature_quantile_hausdorff_deltas
            ).mean().item()
            feature_box_dimension_delta = torch.stack(
                feature_box_dimension_deltas
            ).mean().item()
            feature_graph_fundamental_rank_delta = torch.stack(
                feature_graph_fundamental_rank_deltas
            ).mean().item()
        else:
            feature_hausdorff_delta = float("nan")
            feature_quantile_hausdorff_delta = float("nan")
            feature_box_dimension_delta = float("nan")
            feature_graph_fundamental_rank_delta = float("nan")
        feature_measure_moment_delta = torch.stack(feature_measure_moment_deltas).mean().item()
        feature_measure_local_delta = torch.stack(feature_measure_local_deltas).mean().item()
        feature_measure_weighted_delta = torch.stack(feature_measure_weighted_deltas).mean().item()
        feature_measure_struct_delta = torch.stack(feature_measure_struct_deltas).mean().item()
        feature_measure_struct01_delta = torch.stack(feature_measure_struct01_deltas).mean().item()
        feature_measure_spatial_gram_delta = torch.stack(feature_measure_spatial_gram_deltas).mean().item()
        feature_measure_spatial_gram_rel_delta = torch.stack(feature_measure_spatial_gram_rel_deltas).mean().item()
        feature_measure_relu_active_delta = torch.stack(feature_measure_relu_active_deltas).mean().item()
        feature_measure_relu_active_spatial_delta = torch.stack(feature_measure_relu_active_spatial_deltas).mean().item()
        feature_measure_relu_active_spatial_rel_delta = torch.stack(feature_measure_relu_active_spatial_rel_deltas).mean().item()
        feature_measure_robust_spatial_delta = torch.stack(feature_measure_robust_spatial_deltas).mean().item()
        feature_measure_robust_spatial_rel_delta = torch.stack(feature_measure_robust_spatial_rel_deltas).mean().item()
        capacity_delta = torch.stack(capacity_deltas).mean().item()
        capacity_relu_active_delta = torch.stack(capacity_relu_active_deltas).mean().item()
        capacity_above_mean_delta = torch.stack(capacity_above_mean_deltas).mean().item()
        fsp_delta = torch.stack(fsp_deltas).mean().item()
        if fsp_strict_deltas:
            fsp_strict_delta = torch.stack(fsp_strict_deltas).mean().item()
            feature_measure_flow_delta = torch.stack(feature_measure_flow_deltas).mean().item()
            feature_measure_flow_rel_delta = torch.stack(feature_measure_flow_rel_deltas).mean().item()
            feature_measure_robust_delta = torch.stack(feature_measure_robust_deltas).mean().item()
            feature_measure_robust_rel_delta = torch.stack(feature_measure_robust_rel_deltas).mean().item()
            feature_measure_robust_hybrid_delta = (
                0.5 * feature_measure_flow_delta
                + 0.5 * feature_measure_spatial_gram_delta
                + feature_measure_weighted_delta
            )
            feature_measure_robust_hybrid_rel_delta = (
                0.5 * feature_measure_flow_rel_delta
                + 0.5 * feature_measure_spatial_gram_rel_delta
                + feature_measure_weighted_delta
            )
            feature_measure_relu_active_hybrid_delta = (
                0.5 * feature_measure_flow_delta
                + 0.5 * feature_measure_relu_active_spatial_delta
                + feature_measure_relu_active_delta
            )
            feature_measure_relu_active_hybrid_rel_delta = (
                0.5 * feature_measure_flow_rel_delta
                + 0.5 * feature_measure_relu_active_spatial_rel_delta
                + feature_measure_relu_active_delta
            )
        else:
            fsp_strict_delta = float("nan")
            feature_measure_flow_delta = float("nan")
            feature_measure_flow_rel_delta = float("nan")
            feature_measure_robust_delta = float("nan")
            feature_measure_robust_rel_delta = float("nan")
            feature_measure_robust_hybrid_delta = float("nan")
            feature_measure_robust_hybrid_rel_delta = float("nan")
            feature_measure_relu_active_hybrid_delta = float("nan")
            feature_measure_relu_active_hybrid_rel_delta = float("nan")
    else:
        feature_delta = float("nan")
        feature_measure_swd = float("nan")
        feature_measure_mmd = float("nan")
        feature_measure_cov_delta = float("nan")
        feature_hausdorff_delta = float("nan")
        feature_quantile_hausdorff_delta = float("nan")
        feature_box_dimension_delta = float("nan")
        feature_graph_fundamental_rank_delta = float("nan")
        feature_measure_moment_delta = float("nan")
        feature_measure_local_delta = float("nan")
        feature_measure_weighted_delta = float("nan")
        feature_measure_struct_delta = float("nan")
        feature_measure_struct01_delta = float("nan")
        feature_measure_spatial_gram_delta = float("nan")
        feature_measure_spatial_gram_rel_delta = float("nan")
        feature_measure_relu_active_delta = float("nan")
        feature_measure_relu_active_spatial_delta = float("nan")
        feature_measure_relu_active_spatial_rel_delta = float("nan")
        feature_measure_robust_spatial_delta = float("nan")
        feature_measure_robust_spatial_rel_delta = float("nan")
        capacity_delta = float("nan")
        capacity_relu_active_delta = float("nan")
        capacity_above_mean_delta = float("nan")
        fsp_delta = float("nan")
        fsp_strict_delta = float("nan")
        feature_measure_flow_delta = float("nan")
        feature_measure_flow_rel_delta = float("nan")
        feature_measure_robust_delta = float("nan")
        feature_measure_robust_rel_delta = float("nan")
        feature_measure_robust_hybrid_delta = float("nan")
        feature_measure_robust_hybrid_rel_delta = float("nan")
        feature_measure_relu_active_hybrid_delta = float("nan")
        feature_measure_relu_active_hybrid_rel_delta = float("nan")

    return MetricResult(
        feature_delta=feature_delta,
        feature_measure_swd=feature_measure_swd,
        feature_measure_mmd=feature_measure_mmd,
        feature_measure_cov_delta=feature_measure_cov_delta,
        feature_hausdorff_delta=feature_hausdorff_delta,
        feature_quantile_hausdorff_delta=feature_quantile_hausdorff_delta,
        feature_box_dimension_delta=feature_box_dimension_delta,
        feature_graph_fundamental_rank_delta=feature_graph_fundamental_rank_delta,
        feature_measure_moment_delta=feature_measure_moment_delta,
        feature_measure_local_delta=feature_measure_local_delta,
        feature_measure_weighted_delta=feature_measure_weighted_delta,
        feature_measure_struct_delta=feature_measure_struct_delta,
        feature_measure_struct01_delta=feature_measure_struct01_delta,
        perturbation_capacity_delta=capacity_delta,
        perturbation_capacity_relu_active_delta=capacity_relu_active_delta,
        perturbation_capacity_above_mean_delta=capacity_above_mean_delta,
        logit_kl=logit_kl.item(),
        natural_loss=natural_loss.item(),
        adversarial_loss=adversarial_loss.item(),
        input_loss_delta=input_loss_delta.item(),
        input_grad_l2=input_grad_l2_per_sample.mean().item(),
        local_logit_lipschitz=local_logit_lipschitz.item(),
        margin_input_grad_ratio=margin_input_grad_ratio.item(),
        feature_measure_flow_delta=feature_measure_flow_delta,
        feature_measure_flow_rel_delta=feature_measure_flow_rel_delta,
        feature_measure_spatial_gram_delta=feature_measure_spatial_gram_delta,
        feature_measure_spatial_gram_rel_delta=feature_measure_spatial_gram_rel_delta,
        feature_measure_relu_active_delta=feature_measure_relu_active_delta,
        feature_measure_relu_active_spatial_delta=feature_measure_relu_active_spatial_delta,
        feature_measure_relu_active_spatial_rel_delta=feature_measure_relu_active_spatial_rel_delta,
        feature_measure_relu_active_hybrid_delta=feature_measure_relu_active_hybrid_delta,
        feature_measure_relu_active_hybrid_rel_delta=feature_measure_relu_active_hybrid_rel_delta,
        feature_measure_robust_delta=feature_measure_robust_delta,
        feature_measure_robust_rel_delta=feature_measure_robust_rel_delta,
        feature_measure_robust_spatial_delta=feature_measure_robust_spatial_delta,
        feature_measure_robust_spatial_rel_delta=feature_measure_robust_spatial_rel_delta,
        feature_measure_robust_hybrid_delta=feature_measure_robust_hybrid_delta,
        feature_measure_robust_hybrid_rel_delta=feature_measure_robust_hybrid_rel_delta,
        fsp_strict_delta=fsp_strict_delta,
        fsp_delta=fsp_delta,
        clean_acc=_accuracy(clean_logits, labels),
        adv_acc=_accuracy(adv_logits, labels),
        num_layers=len(indices),
        num_candidate_layers=n,
        sampled_layer_indices=",".join(str(index) for index in indices),
    )
