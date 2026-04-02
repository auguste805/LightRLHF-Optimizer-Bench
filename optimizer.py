"""
Optimizer factory Adam / AdamW / SGD / RMSprop (+ hooks for custom solvers e.g. RoPID).
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple, Union

import torch
import torch.nn as nn


def get_optimizer(model: nn.Module, optim_name: str, **kwargs) -> torch.optim.Optimizer:
    """
    Build an optimizer.

    ``optim_name`` (case-insensitive): ``adam``, ``adamw``, ``sgd``, ``rmsprop``.

    Optional ``params``: iterable of ``nn.Parameter`` (e.g. LoRA + value head only).
    If omitted, uses ``model.parameters()``.

    Common kwargs: ``lr``, ``weight_decay``.

    Adam / AdamW: ``betas``, ``eps``.
    SGD: ``momentum`` (default ``0.9``), ``nesterov`` (default ``False``).
    RMSprop: ``alpha``, ``eps``, ``momentum`` (default ``0``), ``centered``.
    """
    name = optim_name.strip().lower()
    params: Union[Iterable[torch.nn.Parameter], Any] = kwargs.pop("params", None)
    if params is None:
        params = model.parameters()

    lr = kwargs.get("lr", 5e-5)
    weight_decay = kwargs.get("weight_decay", 0.01)

    if name == "adam":
        betas: Tuple[float, float] = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-8)
        return torch.optim.Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    if name == "adamw":
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-8)
        return torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    if name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", False)
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if name == "rmsprop":
        alpha = kwargs.get("alpha", 0.99)
        eps = kwargs.get("eps", 1e-8)
        momentum = kwargs.get("momentum", 0.0)
        centered = kwargs.get("centered", False)
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )

    raise ValueError(
        f"Unknown optimizer {optim_name!r}. "
        f"Supported: 'adam', 'adamw', 'sgd', 'rmsprop'."
    )
