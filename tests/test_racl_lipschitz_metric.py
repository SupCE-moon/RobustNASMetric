from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytest

from robust_metric_lab.nasrobbench201 import racl_architecture_lipschitz_stats
from robust_metric_lab.nb201_space import Nb201TinyNetwork


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    is_zero = True

    def forward(self, x):
        return x * 0


class InferCell(nn.Module):
    def __init__(self, layers: list[nn.Module], node_ix: list[list[int]]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.node_IX = node_ix

    def forward(self, x):
        return x


def test_racl_discrete_identity_cell_matches_node_sum_product() -> None:
    # Node incoming sums are 1, 2, and 3, so Eq. (10)'s cell term is 6.
    cell = InferCell([Identity() for _ in range(6)], [[0], [1, 2], [3, 4, 5]])
    stats = racl_architecture_lipschitz_stats(nn.Sequential(cell))
    assert stats.log_upper == pytest.approx(math.log(6.0))
    assert stats.num_cells == 1
    assert stats.num_nodes == 3
    assert stats.zero_node_sums == 0


def test_racl_zero_node_is_finite_and_audited() -> None:
    cell = InferCell([Zero(), Identity(), Identity()], [[0], [1, 2]])
    stats = racl_architecture_lipschitz_stats(nn.Sequential(cell))
    assert math.isfinite(stats.log_upper)
    assert stats.zero_node_sums == 1


def test_racl_conv_uses_weight_spectral_norm() -> None:
    conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, 0] = 3.0
        conv.weight[1, 1, 0, 0] = 2.0
    cell = InferCell([conv], [[0]])
    stats = racl_architecture_lipschitz_stats(nn.Sequential(cell))
    assert stats.log_upper == pytest.approx(math.log(3.0), rel=1e-5)


def test_racl_architecture_helper_recognizes_repository_nb201_cells() -> None:
    model = Nb201TinyNetwork(("skip_connect",) * 6, c=2, n=1)
    stats = racl_architecture_lipschitz_stats(model)

    assert stats.mode == "nb201_cell"
    assert stats.num_cells == len(model.cells)
    assert stats.num_nodes == len(model.cells) * 3
