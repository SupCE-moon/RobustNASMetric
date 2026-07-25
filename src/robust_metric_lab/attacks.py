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
    raise ValueError(f"Unsupported attack: {attack}")

