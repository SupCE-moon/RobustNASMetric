import torch

from robust_metric_lab.darts_space import (
    DartsNetworkCIFAR,
    genotype_signature,
    sample_darts_genotypes,
)
from robust_metric_lab.nasrobbench201 import racl_fixed_network_lipschitz_stats


def test_darts_sampling_is_deterministic_and_unique():
    first = sample_darts_genotypes(4, seed=17)
    second = sample_darts_genotypes(4, seed=17)
    assert first == second
    assert len({genotype_signature(item) for item in first}) == 4
    assert all(len(item.normal) == 8 and len(item.reduce) == 8 for item in first)


def test_small_darts_network_forward_and_racl_cell_proxy():
    genotype = sample_darts_genotypes(1, seed=3)[0]
    model = DartsNetworkCIFAR(genotype, channels=4, layers=2)
    logits = model(torch.randn(2, 3, 32, 32))
    stats = racl_fixed_network_lipschitz_stats(model, power_iterations=2)
    assert logits.shape == (2, 10)
    assert stats.mode == "darts_cell"
    assert stats.num_cells == 2
    assert stats.num_nodes == 8
