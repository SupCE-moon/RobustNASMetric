import torch

from robust_metric_lab.metrics import _feature_geometry_stats, _geometry_anchor_indices


def test_feature_geometry_metrics_are_zero_for_identical_maps():
    torch.manual_seed(7)
    feature = torch.randn(2, 12, 4, 4)

    values = _feature_geometry_stats(feature, feature.clone())

    assert len(values) == 4
    for value in values:
        assert torch.isfinite(value)
        assert value.item() == 0.0


def test_feature_geometry_metrics_are_finite_under_perturbation():
    torch.manual_seed(11)
    clean = torch.randn(2, 16, 4, 4)
    adv = clean + 0.15 * torch.randn_like(clean)

    hausdorff, quantile_hausdorff, box_delta, graph_delta = _feature_geometry_stats(
        clean, adv
    )

    for value in (hausdorff, quantile_hausdorff, box_delta, graph_delta):
        assert torch.isfinite(value)
        assert value.item() >= 0.0
    assert hausdorff.item() > 0.0
    assert quantile_hausdorff.item() > 0.0
    assert quantile_hausdorff.item() <= hausdorff.item()


def test_single_spatial_atom_has_trivial_graph_topology():
    clean = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    adv = clean + 0.1

    values = _feature_geometry_stats(clean, adv)

    assert all(torch.isfinite(value) for value in values)
    assert values[3].item() == 0.0


def test_geometry_layer_sampling_excludes_degenerate_spatial_maps():
    features = [
        torch.randn(2, 4, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 16, 2, 2),
        torch.randn(2, 32, 1, 1),
    ]

    indices, candidate_count = _geometry_anchor_indices(features, max_layers=4)

    assert indices == [0, 1]
    assert candidate_count == 2
