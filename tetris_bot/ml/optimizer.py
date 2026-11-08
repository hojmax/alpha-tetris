"""Hybrid Muon + AdamW optimizer for TetrisNet.

`torch.optim.Muon` strictly requires 2D parameter matrices and is intended for
hidden Linear weights only. Convolutions (4D), norms/biases (1D), and the
policy/value output heads are therefore routed to AdamW.

`OptimizerBundle` and `SchedulerBundle` quack like a single optimizer/scheduler
for the rest of the training pipeline: they expose `param_groups`, `step()`,
`zero_grad()`, `state_dict()`, and `load_state_dict()` so existing trainer code
can treat them uniformly.
"""

from __future__ import annotations

from typing import Iterable

import structlog
import torch
from torch import nn

logger = structlog.get_logger()


# Output-head module names: their 2D weights stay on AdamW per Muon convention.
_ADAMW_HEAD_MODULE_NAMES: frozenset[str] = frozenset({"policy_head", "value_head"})


def split_muon_adamw_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable parameters between Muon and AdamW.

    Muon receives 2D Linear `weight` tensors that are not part of an output
    head; AdamW receives everything else (conv weights, norms, biases, output
    heads).
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        segments = name.split(".")
        on_excluded_head = any(seg in _ADAMW_HEAD_MODULE_NAMES for seg in segments[:-1])
        is_2d_weight = param.ndim == 2 and segments[-1] == "weight"
        if is_2d_weight and not on_excluded_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if not muon_params:
        raise ValueError(
            "split_muon_adamw_params produced no Muon parameters; "
            "model has no eligible 2D hidden Linear weights"
        )
    return muon_params, adamw_params


def use_fused_adamw(params: list[nn.Parameter]) -> bool:
    """Use PyTorch's fused AdamW kernel when every param lives on CUDA or MPS.

    The fused kernel updates all ~80 AdamW tensors in one launch; the for-loop
    and foreach paths cost hundreds of launches (and the for-loop path also a
    Python loop) per step. Elsewhere (CPU) keep PyTorch's default choice.
    """
    device_types = {param.device.type for param in params}
    return len(device_types) == 1 and device_types <= {"cuda", "mps"}


class OptimizerBundle:
    """Wraps Muon (2D hidden weights) and AdamW (everything else)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        muon_adjust_lr_fn: str | None = "match_rms_adamw",
    ):
        # `match_rms_adamw` rescales Muon's effective step to AdamW's RMS, so a
        # single shared LR works for both inner optimizers.
        muon_params, adamw_params = split_muon_adamw_params(model)
        # `None` keeps PyTorch's default implementation choice.
        self._adamw_fused: bool | None = True if use_fused_adamw(adamw_params) else None
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=self._adamw_fused,
        )
        self.muon = torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        logger.info(
            "Created hybrid Muon + AdamW optimizer",
            num_muon_params=sum(p.numel() for p in muon_params),
            num_adamw_params=sum(p.numel() for p in adamw_params),
            num_muon_tensors=len(muon_params),
            num_adamw_tensors=len(adamw_params),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            adamw_fused=bool(self._adamw_fused),
        )

    @property
    def inner_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # AdamW first so param_groups[0] / base_lrs[0] surface the AdamW LR
        # (matches the historical single-optimizer behavior in logs).
        return (self.adamw, self.muon)

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        for opt in self.inner_optimizers:
            groups.extend(opt.param_groups)
        return groups

    def step(self, closure=None) -> None:
        if closure is not None:
            raise NotImplementedError(
                "OptimizerBundle does not support closure-based step()"
            )
        for opt in self.inner_optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.inner_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "adamw": self.adamw.state_dict(),
            "muon": self.muon.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "adamw" not in state_dict or "muon" not in state_dict:
            raise ValueError(
                "OptimizerBundle.load_state_dict expected keys "
                f"{{'adamw', 'muon'}}; got {sorted(state_dict)}"
            )
        self.adamw.load_state_dict(state_dict["adamw"])
        # `Optimizer.load_state_dict` adopts the saved param-group flags, so a
        # checkpoint written by a different implementation (e.g. foreach=False)
        # would otherwise silently switch this run back to it. Step-counter
        # placement for the restored flags is fixed by
        # `sanitize_optimizer_state_steps` at every load/step boundary.
        for group in self.adamw.param_groups:
            group["fused"] = self._adamw_fused
            group["foreach"] = None
        self.muon.load_state_dict(state_dict["muon"])


class SchedulerBundle:
    """Wraps one LR scheduler per inner optimizer in an OptimizerBundle."""

    def __init__(self, schedulers: Iterable[torch.optim.lr_scheduler.LRScheduler]):
        schedulers = tuple(schedulers)
        if not schedulers:
            raise ValueError("SchedulerBundle requires at least one inner scheduler")
        self._schedulers = schedulers

    @property
    def inner_schedulers(self) -> tuple[torch.optim.lr_scheduler.LRScheduler, ...]:
        return self._schedulers

    @property
    def first(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._schedulers[0]

    @property
    def base_lrs(self) -> list[float]:
        return [lr for s in self._schedulers for lr in s.base_lrs]

    @property
    def last_epoch(self) -> int:
        return self._schedulers[0].last_epoch

    @last_epoch.setter
    def last_epoch(self, value: int) -> None:
        for s in self._schedulers:
            s.last_epoch = value

    @property
    def _last_lr(self) -> list[float]:
        return [lr for s in self._schedulers for lr in s._last_lr]

    @_last_lr.setter
    def _last_lr(self, value: list[float]) -> None:
        idx = 0
        for s in self._schedulers:
            count = len(s.optimizer.param_groups)
            s._last_lr = list(value[idx : idx + count])
            idx += count

    def step(self) -> None:
        for s in self._schedulers:
            s.step()

    def state_dict(self) -> list[dict]:
        return [s.state_dict() for s in self._schedulers]

    def load_state_dict(self, state_dict: list[dict]) -> None:
        if len(state_dict) != len(self._schedulers):
            raise ValueError(
                "SchedulerBundle.load_state_dict expected "
                f"{len(self._schedulers)} entries; got {len(state_dict)}"
            )
        for s, sd in zip(self._schedulers, state_dict):
            s.load_state_dict(sd)
