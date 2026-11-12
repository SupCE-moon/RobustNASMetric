from __future__ import annotations

import torch
import torch.nn.functional as F


def clamp_linf(x_adv: torch.Tensor, x: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x + epsilon), x - epsilon).clamp(0.0, 1.0)


def fgsm(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return clamp_linf(x_adv + epsilon * grad.sign(), x, epsilon).detach()


def pgd_linf(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = 2 / 255,
    steps: int = 3,
    random_start: bool = True,
) -> torch.Tensor:
    if random_start:
        x_adv = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    else:
        x_adv = x.detach().clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = clamp_linf(x_adv + alpha * grad.sign(), x, epsilon)
    return x_adv.detach()


def make_adversarial(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attack: str,
    epsilon: float,
    pgd_steps: int = 3,
) -> torch.Tensor:
    attack_key = attack.lower()
    if attack_key == "none":
        return x.detach()
    if attack_key == "fgsm":
        return fgsm(model, x, y, epsilon)
    if attack_key in {"pgd", "pgd-lite", "pgd_lite"}:
        return pgd_linf(model, x, y, epsilon, steps=pgd_steps)
    if attack_key in {"fixed", "fixed-noise", "fixednoise"}:
        return fixed_linf(x, epsilon)
    raise ValueError(f"Unsupported attack: {attack}")


_FIXED_SIGN = None


def fixed_linf(x: torch.Tensor, epsilon: float, seed: int = 20260912) -> torch.Tensor:
    """Fixed random sign perturbation, identical across models/architectures.

    Unlike white-box fgsm (whose direction is the model's own random gradient at
    random init), this is a single fixed input-space direction. It removes the
    'arbitrary direction' confound E27 identified for zero-cost delta metrics.
    """
    global _FIXED_SIGN
    if _FIXED_SIGN is None or _FIXED_SIGN.device != x.device:
        g = torch.Generator(device=x.device).manual_seed(seed)
        _FIXED_SIGN = (
            torch.randint(0, 2, x.shape[1:], generator=g, device=x.device).to(x.dtype) * 2 - 1
        )
    return clamp_linf(x + epsilon * _FIXED_SIGN, x, epsilon).detach()

