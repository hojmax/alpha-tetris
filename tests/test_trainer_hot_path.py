from __future__ import annotations

import copy

import torch

from tetris_bot.ml.ema import ExponentialMovingAverage
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.optimizer import OptimizerBundle, use_fused_adamw
from tetris_bot.ml.policy_mirroring import (
    FLAT_MIRROR_INDEX_BY_PIECE,
    _table_on_device,
)


def _small_net() -> TetrisNet:
    return TetrisNet(
        trunk_channels=4,
        num_conv_residual_blocks=1,
        reduction_channels=4,
        board_stats_hidden=8,
        board_proj_hidden=16,
        fc_hidden=16,
        aux_hidden=8,
        fusion_hidden=16,
        num_fusion_blocks=1,
    )


def test_fused_adamw_only_on_accelerators() -> None:
    assert not use_fused_adamw([torch.nn.Parameter(torch.zeros(2))])
    if torch.backends.mps.is_available():
        assert use_fused_adamw([torch.nn.Parameter(torch.zeros(2, device="mps"))])


def test_optimizer_bundle_load_keeps_own_adamw_implementation() -> None:
    source = OptimizerBundle(_small_net(), learning_rate=1e-3, weight_decay=0.0)
    for group in source.adamw.param_groups:
        group["foreach"] = False  # e.g. a checkpoint written by older code
    state = source.state_dict()

    restored = OptimizerBundle(_small_net(), learning_rate=1e-3, weight_decay=0.0)
    restored.load_state_dict(state)

    for group in restored.adamw.param_groups:
        assert group["foreach"] is None
        assert group["fused"] is None


def test_ema_update_matches_per_tensor_lerp_and_survives_device_moves() -> None:
    torch.manual_seed(0)
    model = _small_net()
    ema = ExponentialMovingAverage(model, decay=0.9)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
        model.row_coords.add_(1.0)  # type: ignore[operator]
    # Module.to() replaces buffer tensors; the update must still reach them.
    ema.model.to("cpu", dtype=torch.float64).to(dtype=torch.float32)

    ema.update(model)

    with torch.no_grad():
        for ref_param, param in zip(reference.parameters(), model.parameters()):
            ref_param.lerp_(param, 0.1)
    for (name, expected), actual in zip(
        reference.named_parameters(), ema.model.parameters()
    ):
        assert torch.equal(expected, actual), name
    assert torch.equal(ema.model.row_coords, model.row_coords)  # type: ignore[arg-type]


def test_mirror_tables_are_cached_per_device() -> None:
    assert _table_on_device(FLAT_MIRROR_INDEX_BY_PIECE, torch.device("cpu")) is (
        FLAT_MIRROR_INDEX_BY_PIECE
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        first = _table_on_device(FLAT_MIRROR_INDEX_BY_PIECE, device)
        assert first is _table_on_device(FLAT_MIRROR_INDEX_BY_PIECE, device)
        assert torch.equal(first.cpu(), FLAT_MIRROR_INDEX_BY_PIECE)
