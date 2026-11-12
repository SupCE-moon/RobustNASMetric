import numpy as np

from scripts.analyze_trnas_ours_comparison import (
    matched_pairs,
    pair_concordance,
    spearman_utility,
)


def test_exact_operation_matching_excludes_different_structures():
    controls = [
        {"op_count_a": 1, "op_count_b": 2},
        {"op_count_a": 1, "op_count_b": 2},
        {"op_count_a": 2, "op_count_b": 1},
    ]
    assert matched_pairs(controls, "nb201_e0") == [(0, 1)]


def test_rb40_matching_requires_family_and_parameter_caliper():
    controls = [
        {"model_family": "a", "log_parameter_count": 10.0},
        {"model_family": "a", "log_parameter_count": 10.05},
        {"model_family": "a", "log_parameter_count": 10.2},
        {"model_family": "b", "log_parameter_count": 10.01},
    ]
    assert matched_pairs(controls, "rb40_mature") == [(0, 1)]


def test_concordance_uses_direction_and_omits_target_ties():
    pairs = [(0, 1), (0, 2), (1, 2)]
    target = np.array([1.0, 2.0, 2.0])
    lower_is_better = np.array([3.0, 2.0, 1.0])
    assert pair_concordance(pairs, target, lower_is_better, "lower") == (2, 2, 1.0)


def test_spearman_utility_honors_lower_direction():
    assert spearman_utility(np.array([1, 2, 3]), np.array([3, 2, 1]), "lower") == 1.0
