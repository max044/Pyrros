"""Optimization helpers using Composer (fallback to vanilla PyTorch)."""
from __future__ import annotations

from typing import Any, Dict

import torch

try:
    from composer import Trainer as ComposerTrainer
    from composer.optim import DecoupledAdamW
except ImportError:  # Composer optional
    ComposerTrainer = None  # type: ignore
    DecoupledAdamW = None  # type: ignore

__all__ = [
    "build_optimizer",
    "step_gradient",
]


def build_optimizer(model: torch.nn.Module, lr: float = 2e-5, weight_decay: float = 0.0):
    """Return a Composer-compatible optimizer or vanilla AdamW."""
    if DecoupledAdamW is not None:
        return DecoupledAdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def step_gradient(loss: torch.Tensor, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
