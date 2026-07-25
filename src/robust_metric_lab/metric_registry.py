from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    attribution: str
    reference: str

    @property
    def lower_is_better(self) -> bool:
        return self.direction == "lower"


def _specs(names: tuple[str, ...], attribution: str, reference: str) -> list[MetricSpec]:
    return [MetricSpec(name, "lower", attribution, reference) for name in names]


METRIC_SPECS = [
    *_specs(
        (
            "feature_delta",
            "feature_measure_moment_delta",
            "feature_measure_local_delta",
            "feature_measure_weighted_delta",
            "feature_measure_struct_delta",
            "feature_measure_struct01_delta",
            "feature_hausdorff_delta",
            "feature_quantile_hausdorff_delta",
            "feature_box_dimension_delta",
            "feature_graph_fundamental_rank_delta",
            "perturbation_capacity_delta",
            "perturbation_capacity_relu_active_delta",
            "perturbation_capacity_above_mean_delta",
            "feature_measure_flow_delta",
            "feature_measure_flow_rel_delta",
            "feature_measure_spatial_gram_delta",
            "feature_measure_spatial_gram_rel_delta",
            "feature_measure_relu_active_delta",
            "feature_measure_relu_active_spatial_delta",
            "feature_measure_relu_active_spatial_rel_delta",
            "feature_measure_relu_active_hybrid_delta",
            "feature_measure_relu_active_hybrid_rel_delta",
            "feature_measure_robust_delta",
            "feature_measure_robust_rel_delta",
            "feature_measure_robust_spatial_delta",
            "feature_measure_robust_spatial_rel_delta",
            "feature_measure_robust_hybrid_delta",
            "feature_measure_robust_hybrid_rel_delta",
        ),
        "Ours",
        "Ours",
    ),
    *_specs(("feature_measure_mmd",), "Ours", "Ours"),
    *_specs(
        ("feature_measure_swd", "feature_measure_cov_delta"),
        "Distribution discrepancy",
        "baseline",
    ),
    *_specs(("fsp_strict_delta", "fsp_delta"), "FSP", "[1,2]"),
    *_specs(("logit_kl",), "TRADES divergence", "[3]"),
    *_specs(("natural_loss", "adversarial_loss"), "Loss control", "[10]"),
    *_specs(("input_loss_delta",), "Input-loss change", "[8,9]"),
    *_specs(("input_grad_l2",), "Input gradient", "[4,7]"),
    *_specs(("local_logit_lipschitz",), "Local logit sensitivity", "[5,6]"),
    MetricSpec("margin_input_grad_ratio", "higher", "Margin/gradient radius", "[7]"),
    MetricSpec(
        "racl_arch_lipschitz_log_upper",
        "lower",
        "RACL fixed-architecture counterpart",
        "[5]",
    ),
    MetricSpec(
        "racl_fixed_network_log_upper_proxy",
        "lower",
        "RACL generic fixed-network proxy",
        "[5]",
    ),
    MetricSpec(
        "racl_input_gradient_lower",
        "lower",
        "RACL lower term (alias of input_grad_l2)",
        "[5]",
    ),
    MetricSpec("trnas_activation_psi", "higher", "TRNAS activation capability", "[11,12]"),
    MetricSpec("trnas_consistency_phi", "higher", "TRNAS consistency", "[11]"),
    MetricSpec("trnas_r_score", "higher", "TRNAS R-Score", "[11]"),
]


METRIC_BY_NAME = {spec.name: spec for spec in METRIC_SPECS}


def canonical_metric_name(name: str) -> str:
    if name == "racl_fixed_network_log_upper_proxy":
        return "racl_arch_lipschitz_log_upper"
    return name
