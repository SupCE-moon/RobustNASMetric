from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
import torch.nn.functional as F

from .metrics import _fsp_corr_matrix, _safe_cosine_distance, _spatial_atoms


E37_METRIC_NAMES = (
    "feature_measure_flow_corr_worst_delta",
    "feature_measure_flow_corr_rms_delta",
    "feature_measure_flow_corr_tophalf_delta",
    "feature_measure_spectrum_fisher_rao_delta",
    "feature_measure_spectrum_triangular_delta",
    "feature_measure_spectrum_ks_delta",
    "feature_measure_multiview_geometric_delta",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
)


def _symmetric_relative(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    difference = (adversarial - clean).flatten(1).norm(p=2, dim=1)
    scale = 0.5 * (
        clean.flatten(1).norm(p=2, dim=1)
        + adversarial.flatten(1).norm(p=2, dim=1)
    )
    return difference / scale.clamp_min(1e-12)


def _flow_drift(
    clean_left: torch.Tensor,
    clean_right: torch.Tensor,
    adversarial_left: torch.Tensor,
    adversarial_right: torch.Tensor,
) -> torch.Tensor:
    return _symmetric_relative(
        _fsp_corr_matrix(clean_left.double(), clean_right.double()),
        _fsp_corr_matrix(adversarial_left.double(), adversarial_right.double()),
    )


def _spectrum_probabilities(feature: torch.Tensor, max_positions: int = 64) -> torch.Tensor:
    if feature.shape[2] * feature.shape[3] > max_positions:
        side = max(1, int(math.sqrt(max_positions)))
        feature = F.adaptive_avg_pool2d(feature, (side, side))
    # Spectral probabilities feed acos near one.  Perform this small
    # (at most 64-position) decomposition in float64 so identical spectra map
    # to zero and positive rescaling does not create a false angular drift.
    atoms = _spatial_atoms(feature).double()
    atoms = atoms - atoms.mean(dim=1, keepdim=True)
    energy = torch.linalg.svdvals(atoms).square()
    total = energy.sum(dim=1, keepdim=True)
    uniform = torch.full_like(energy, 1.0 / energy.shape[1])
    return torch.where(total > 1e-12, energy / total.clamp_min(1e-12), uniform)


def _centered_channel_drift(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    def gram(feature: torch.Tensor) -> torch.Tensor:
        centered = feature - feature.mean(dim=(2, 3), keepdim=True)
        flat = F.normalize(centered.flatten(2), p=2, dim=2, eps=1e-12)
        return torch.bmm(flat, flat.transpose(1, 2)) / flat.shape[2]

    return _symmetric_relative(gram(clean.double()), gram(adversarial.double()))


def _spatial_cosine_drift(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    def gram(feature: torch.Tensor) -> torch.Tensor:
        atoms = F.normalize(_spatial_atoms(feature), p=2, dim=2, eps=1e-12)
        return torch.bmm(atoms, atoms.transpose(1, 2)) / atoms.shape[2]

    return _safe_cosine_distance(gram(clean.double()), gram(adversarial.double()))


def compute_e37_per_sample(
    clean_features: Sequence[torch.Tensor],
    adversarial_features: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> Mapping[str, torch.Tensor]:
    selected = tuple(indices)
    if len(selected) < 2:
        raise ValueError("E37 metrics require at least two anchors")
    if len(set(selected)) != len(selected):
        raise ValueError("E37 anchor indices must be unique")
    if any(index < 0 or index >= len(clean_features) or index >= len(adversarial_features) for index in selected):
        raise ValueError("E37 anchor index is out of range")
    batches = {
        int(clean_features[index].shape[0]) for index in selected
    } | {
        int(adversarial_features[index].shape[0]) for index in selected
    }
    if len(batches) != 1:
        raise ValueError("clean and adversarial feature batch sizes must match")

    flows = torch.stack(
        [
            _flow_drift(
                clean_features[left],
                clean_features[right],
                adversarial_features[left],
                adversarial_features[right],
            )
            for left, right in zip(selected, selected[1:])
        ],
        dim=1,
    )
    top_count = math.ceil(flows.shape[1] / 2)
    flow_worst = flows.max(dim=1).values
    flow_rms = flows.square().mean(dim=1).sqrt()
    flow_tophalf = flows.topk(top_count, dim=1).values.mean(dim=1)

    fisher_layers: list[torch.Tensor] = []
    triangular_layers: list[torch.Tensor] = []
    ks_layers: list[torch.Tensor] = []
    channel_layers: list[torch.Tensor] = []
    spatial_layers: list[torch.Tensor] = []
    for index in selected:
        clean = clean_features[index]
        adversarial = adversarial_features[index]
        clean_spectrum = _spectrum_probabilities(clean)
        adversarial_spectrum = _spectrum_probabilities(adversarial)
        affinity = (clean_spectrum.sqrt() * adversarial_spectrum.sqrt()).sum(dim=1)
        fisher_layers.append(2.0 * torch.acos(affinity.clamp(0.0, 1.0)))
        triangular_layers.append(
            0.5
            * (
                (clean_spectrum - adversarial_spectrum).square()
                / (clean_spectrum + adversarial_spectrum).clamp_min(1e-12)
            ).sum(dim=1)
        )
        ks_layers.append(
            (clean_spectrum - adversarial_spectrum)
            .cumsum(dim=1)
            .abs()
            .max(dim=1)
            .values
        )
        channel_layers.append(_centered_channel_drift(clean, adversarial))
        spatial_layers.append(_spatial_cosine_drift(clean, adversarial))

    spectrum_fisher = torch.stack(fisher_layers, dim=1).mean(dim=1)
    spectrum_triangular = torch.stack(triangular_layers, dim=1).mean(dim=1)
    spectrum_ks = torch.stack(ks_layers, dim=1).mean(dim=1)
    channel = torch.stack(channel_layers, dim=1).mean(dim=1)
    spatial = torch.stack(spatial_layers, dim=1).mean(dim=1)
    flow = flows.mean(dim=1)
    views = torch.stack((channel, spatial, flow), dim=1)
    bounded = views / (1.0 + views)
    multiview_geometric = bounded.prod(dim=1).clamp_min(0.0).pow(1.0 / 3.0)
    multiview_max = bounded.max(dim=1).values
    multiview_rms = bounded.square().mean(dim=1).sqrt()

    values = (
        flow_worst,
        flow_rms,
        flow_tophalf,
        spectrum_fisher,
        spectrum_triangular,
        spectrum_ks,
        multiview_geometric,
        multiview_max,
        multiview_rms,
    )
    result = dict(zip(E37_METRIC_NAMES, values))
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("E37 metric produced nonfinite values")
    return result
