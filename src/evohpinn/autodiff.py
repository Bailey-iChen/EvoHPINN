"""Pointwise derivatives for functions whose rows are independent.

Reference-set normalization in model.py preserves this independence. Do not
replace it with normalization over the query batch: a summed vector-Jacobian
product would then mix derivatives between query points.
"""

import torch


def coordinates(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def partial(y: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
    if not y.requires_grad:
        return x[:, axis:axis + 1] * 0.0
    gradient = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True,
        retain_graph=True, allow_unused=True,
    )[0]
    if gradient is None:
        return x[:, axis:axis + 1] * 0.0
    return gradient[:, axis:axis + 1]


def second(y: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
    return partial(partial(y, x, axis), x, axis)
