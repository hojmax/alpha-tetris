from __future__ import annotations

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Track an eval-only EMA shadow copy for export and evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1) (got {decay})")
        self.decay = decay
        self._ema_model = copy.deepcopy(model).eval()
        self._ema_model.requires_grad_(False)

    @property
    def model(self) -> nn.Module:
        return self._ema_model

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Parameter/buffer lists are re-read every call: `Module.to()` (used by
        # ONNX export) replaces buffer tensors, so cached lists would go stale.
        # `deepcopy` preserves registration order, so the lists pair up 1:1.
        ema_parameters: list[torch.Tensor] = list(self._ema_model.parameters())
        parameters: list[torch.Tensor] = list(model.parameters())
        if len(parameters) != len(ema_parameters):
            raise ValueError(
                "EMA update model has a different parameter count "
                f"({len(parameters)} vs {len(ema_parameters)})"
            )
        torch._foreach_lerp_(ema_parameters, parameters, 1.0 - self.decay)
        ema_buffers = list(self._ema_model.buffers())
        if ema_buffers:
            torch._foreach_copy_(ema_buffers, list(model.buffers()))
