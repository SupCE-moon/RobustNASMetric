# ReLU-Active Robust Metric Quick Test

## Metric Definition Implemented

For layer `l`, clean feature atoms are `a_{l,s}(x)` and adversarial feature atoms are `a_{l,s}(x_adv)`.

The ReLU-active weight is computed from the clean feature only:

```latex
m_{l,s,c}(x)=\mathbf{1}\{a_{l,s,c}(x)>0\},
\quad
w_{l,s}(x)=
\frac{\frac{1}{C_l}\sum_c m_{l,s,c}(x)}
{\sum_{s'}\frac{1}{C_l}\sum_c m_{l,s',c}(x)+\epsilon}.
```

This follows the user's motivation: features that do not pass ReLU should not contribute equally to the empirical feature measure.

Implemented metrics:

- `feature_measure_relu_active_delta`: first-order weighted local atom displacement.
- `feature_measure_relu_active_spatial_delta`: second-order weighted spatial Gram displacement.
- `feature_measure_relu_active_spatial_rel_delta`: relative second-order weighted spatial Gram displacement.
- `feature_measure_relu_active_hybrid_delta`: `0.5 * flow_delta + 0.5 * relu_active_spatial_delta + relu_active_delta`.
- `feature_measure_relu_active_hybrid_rel_delta`: relative hybrid variant.

## Experiment Setup

- Server: Chengdu 516 RTX 5090 docker.
- Benchmark: NAS-RobBench-201.
- Dataset: CIFAR-10.
- Oracle target: `autoattack`.
- Architectures: 512 stratified samples from 6466 non-isomorphic NAS-Bench-201 architectures.
- Training for proxy model: 1 epoch, 2048 CIFAR-10 samples, FGSM adversarial training.
- Metric samples: 256 CIFAR-10 test samples, PGD-3.
- Search validation: random proxy search and NSGA-II, budget 64, 100 trials.

This is a quick early-weight test, not a full 50-epoch NAS-RobBench-201 reproduction.

## Correlation Results

| Metric | Pearson | Spearman | Kendall | Direction |
|---|---:|---:|---:|---|
| feature_measure_relu_active_delta (Ours) | 0.0033 | -0.0805 | -0.0500 | lower |
| feature_measure_relu_active_spatial_delta (Ours) | 0.2393 | 0.1813 | 0.1175 | higher |
| feature_measure_relu_active_spatial_rel_delta (Ours) | 0.2786 | 0.2378 | 0.1569 | higher |
| feature_measure_relu_active_hybrid_delta (Ours) | 0.0145 | 0.1731 | 0.1229 | higher |
| feature_measure_relu_active_hybrid_rel_delta (Ours) | 0.1114 | 0.0255 | 0.0202 | higher |
| feature_measure_robust_delta (Ours) | 0.0145 | 0.1903 | 0.1347 | higher |
| feature_measure_robust_hybrid_delta (Ours) | 0.0145 | 0.2008 | 0.1407 | higher |
| fsp_delta | -0.1774 | -0.2937 | -0.1960 | lower |
| input_grad_l2 | 0.0315 | 0.3603 | 0.2505 | higher |
| local_logit_lipschitz | 0.0334 | 0.0546 | 0.0428 | higher |

## Search Results

| Metric | Direction | Algorithm | Mean Selected AutoAttack | Mean Regret | Best Selected AutoAttack |
|---|---|---|---:|---:|---:|
| fsp_delta | lower | nsga2_proxy_clean | 0.426383 | 0.017417 | 0.4438 |
| input_grad_l2 | higher | nsga2_proxy_clean | 0.424954 | 0.018846 | 0.4438 |
| feature_measure_relu_active_hybrid_delta (Ours) | higher | nsga2_proxy_clean | 0.424513 | 0.019287 | 0.4438 |
| feature_measure_robust_delta (Ours) | higher | nsga2_proxy_clean | 0.424513 | 0.019287 | 0.4438 |
| feature_measure_robust_hybrid_delta (Ours) | higher | nsga2_proxy_clean | 0.424513 | 0.019287 | 0.4438 |
| local_logit_lipschitz | higher | nsga2_proxy_clean | 0.422032 | 0.021768 | 0.4438 |
| feature_measure_relu_active_delta (Ours) | lower | nsga2_proxy_clean | 0.420789 | 0.023011 | 0.4438 |
| feature_measure_relu_active_spatial_rel_delta (Ours) | higher | nsga2_proxy_clean | 0.403709 | 0.040091 | 0.4265 |
| feature_measure_relu_active_hybrid_rel_delta (Ours) | higher | nsga2_proxy_clean | 0.400577 | 0.043223 | 0.4297 |
| feature_measure_relu_active_spatial_delta (Ours) | higher | nsga2_proxy_clean | 0.396843 | 0.046957 | 0.4261 |

## Initial Interpretation

The ReLU-active idea is useful, but the pure spatial second-order form does not yet outperform the older hybrid/flow metric in search. The best ReLU-aware variant is `feature_measure_relu_active_hybrid_delta`, which nearly matches `input_grad_l2` and is close to FSP under NSGA-II.

The positive result is conceptual: ReLU activity masking improves the theoretical alignment with activated feature transport, and the hybrid form remains competitive. The negative result is also useful: using only ReLU-active spatial Gram is too narrow in this quick setting.

Next, the most promising path is to use the ReLU mask as a component inside the existing hybrid metric, then test residual value against FSP and input-gradient baselines under the 5-epoch medium setting.
