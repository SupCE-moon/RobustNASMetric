"""Tiny smoke test for the three new capacity-family metrics (M1/M2/M3)."""
import torch

from robust_metric_lab.attacks import make_adversarial
from robust_metric_lab.metrics import compute_feature_metrics
from robust_metric_lab.nb201_space import Nb201TinyNetwork

EPS = 8 / 255
torch.manual_seed(0)


def check(name, model, x, y):
    model.eval()
    x_adv = make_adversarial(model, x, y, "fgsm", EPS, pgd_steps=1)
    r = compute_feature_metrics(model, x, x_adv, y, max_layers=4)
    for f in ("perturbation_capacity_gini_delta",
              "perturbation_capacity_layerprofile_delta",
              "perturbation_capacity_channelwise_delta",
              "perturbation_capacity_flow_rel_delta"):
        v = getattr(r, f)
        print(f"  {name:18s} {f:45s} = {v}")
        assert v == v and abs(v) < 1e6, f"bad value {f}={v}"


x = torch.randn(4, 3, 32, 32)
y = torch.randint(0, 10, (4,))

# NB201 with a mix of ops including none/skip/conv/pool
ops = ("none", "skip_connect", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3", "nor_conv_3x3")
nb = Nb201TinyNetwork(ops)
check("nb201-mixed", nb, x, y)

# NB201 all-conv (no dead channels) to sanity check importance is nonzero
ops2 = ("nor_conv_3x3", "nor_conv_1x1", "nor_conv_3x3", "nor_conv_1x1", "nor_conv_3x3", "nor_conv_1x1")
nb2 = Nb201TinyNetwork(ops2)
check("nb201-allconv", nb2, x, y)

print("SMOKE OK")
