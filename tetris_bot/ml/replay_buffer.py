"""Replay batches in the compact layout the trainer stores, and their dense decode.

The Rust replay buffer hands out compact arrays (see `tetris_core/src/replay/compact.rs`):
bit-packed boards and legal-action masks plus a 64-slot 16-bit policy over the legal
actions only. The device replay mirror keeps that layout (~370 bytes per example
instead of ~3.9 KB dense), and only sampled batches are widened to dense tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import torch

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS
from tetris_bot.ml.network import AUX_FEATURES

BOARD_CELLS = BOARD_HEIGHT * BOARD_WIDTH
BOARD_BYTES = (BOARD_CELLS + 7) // 8
MASK_BYTES = (NUM_ACTIONS + 7) // 8
POLICY_SLOTS = 64
# Aux layout: current one-hot (7), hold one-hot (8), hold available (1) and five queue
# one-hots (35) are all binary, so the first 51 features are stored as bits.
AUX_BINARY_PREFIX = 51
AUX_BINARY_BYTES = (AUX_BINARY_PREFIX + 7) // 8
AUX_FLOAT_FEATURES = AUX_FEATURES - AUX_BINARY_PREFIX
# `CompactBatch.bits` columns: board | aux binary prefix | legal-action mask.
_AUX_BITS_START = BOARD_BYTES
_MASK_BITS_START = BOARD_BYTES + AUX_BINARY_BYTES
BITS_BYTES = _MASK_BITS_START + MASK_BYTES
# `CompactBatch.floats` columns: aux float features | value target | overhang fields.
FLOATS_WIDTH = AUX_FLOAT_FEATURES + 2

# Row b holds the 8 bits of byte value b, little bit order. One gather unpacks a batch;
# it is several times faster than shift-and-mask on MPS.
_BYTE_TO_BITS = (torch.arange(256).unsqueeze(1) >> torch.arange(8)) & 1 == 1


@dataclass
class TrainingBatch:
    """Dense training batch, as the loss consumes it."""

    boards: torch.Tensor
    aux: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    overhang_fields: torch.Tensor
    masks: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.boards.shape[0])

    @property
    def device(self) -> torch.device:
        return self.boards.device

    def split(self, batch_size: int) -> list[TrainingBatch]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {batch_size})")
        batches = [
            TrainingBatch(
                **{
                    f.name: getattr(self, f.name)[start : start + batch_size]
                    for f in fields(self)
                }
            )
            for start in range(0, self.size, batch_size)
        ]
        if not batches:
            raise ValueError("Cannot split empty staged batch")
        return batches


@dataclass
class CompactBatch:
    """Replay examples in storage layout (368 bytes each). Bits use little bit order."""

    bits: torch.Tensor  # (N, BITS_BYTES) uint8: board, aux binary prefix, action mask
    floats: torch.Tensor  # (N, FLOATS_WIDTH) float32: aux floats, value, overhang
    policy_legal: torch.Tensor  # (N, POLICY_SLOTS) int16 holding uint16 bit patterns

    @classmethod
    def from_numpy(cls, arrays: tuple) -> CompactBatch:
        """Build from the `(board_bits, aux, policy_legal, values, overhangs, mask_bits)`
        tuple returned by `GameGenerator.sample_batch` / `replay_buffer_delta`."""
        board_bits, aux, policy_legal, values, overhangs, mask_bits = arrays
        aux_bits = np.packbits(
            aux[:, :AUX_BINARY_PREFIX] > 0.5, axis=1, bitorder="little"
        )
        bits = np.concatenate([board_bits, aux_bits, mask_bits], axis=1)
        floats = np.concatenate(
            [
                aux[:, AUX_BINARY_PREFIX:],
                np.asarray(values, dtype=np.float32)[:, None],
                np.asarray(overhangs, dtype=np.float32)[:, None],
            ],
            axis=1,
            dtype=np.float32,
        )
        return cls(
            bits=torch.from_numpy(bits),
            floats=torch.from_numpy(floats),
            policy_legal=torch.from_numpy(
                np.ascontiguousarray(policy_legal, dtype=np.uint16).view(np.int16)
            ),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [getattr(self, f.name) for f in fields(self)]

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> CompactBatch:
        return CompactBatch(*(fn(t) for t in self.tensors()))

    @property
    def size(self) -> int:
        return int(self.bits.shape[0])

    @property
    def device(self) -> torch.device:
        return self.bits.device

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def decode(self) -> TrainingBatch:
        """Dense batch on the same device. Policy is renormalised over stored slots."""
        n = self.size
        bits = unpack_bits(self.bits)
        boards = bits[:, :BOARD_CELLS].reshape(n, 1, BOARD_HEIGHT, BOARD_WIDTH)
        aux_start = _AUX_BITS_START * 8
        aux = torch.cat(
            [
                bits[:, aux_start : aux_start + AUX_BINARY_PREFIX].float(),
                self.floats[:, :AUX_FLOAT_FEATURES],
            ],
            dim=1,
        )
        mask_start = _MASK_BITS_START * 8
        masks = bits[:, mask_start : mask_start + NUM_ACTIONS]
        quantised = (self.policy_legal.to(torch.int32) & 0xFFFF).float()
        slot = (masks.float().cumsum(dim=1) - 1).clamp_(0, POLICY_SLOTS - 1).long()
        policy = torch.gather(quantised, 1, slot) * masks
        policy = policy / policy.sum(dim=1, keepdim=True).clamp_min(1.0)
        return TrainingBatch(
            boards=boards,
            aux=aux,
            policy_targets=policy,
            value_targets=self.floats[:, AUX_FLOAT_FEATURES],
            overhang_fields=self.floats[:, AUX_FLOAT_FEATURES + 1],
            masks=masks,
        )


_byte_to_bits_by_device: dict[torch.device, torch.Tensor] = {}


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """(N, B) uint8, little bit order -> (N, 8 * B) bool."""
    table = _byte_to_bits_by_device.get(packed.device)
    if table is None:
        table = _BYTE_TO_BITS.to(packed.device)
        _byte_to_bits_by_device[packed.device] = table
    return table[packed.int()].reshape(packed.shape[0], -1)


class CircularReplayMirror:
    """Pre-allocated circular buffer of compact examples on the training device.

    All tensors are allocated once at full capacity and updated in place with
    copy_(); sampled rows are decoded to a dense batch on the device.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(capacity, *shape, device=device, dtype=dtype)

        self.storage = CompactBatch(
            bits=zeros(BITS_BYTES, dtype=torch.uint8),
            floats=zeros(FLOATS_WIDTH, dtype=torch.float32),
            policy_legal=zeros(POLICY_SLOTS, dtype=torch.int16),
        )
        self.capacity = capacity
        self.count = 0
        self.write_pos = 0
        self.logical_end = 0

    @property
    def size(self) -> int:
        return self.count

    @property
    def device(self) -> torch.device:
        return self.storage.device

    def reset(self, logical_end: int = 0) -> None:
        self.count = 0
        self.write_pos = 0
        self.logical_end = logical_end

    def write(self, batch: CompactBatch) -> None:
        """Append `batch` (already on the mirror device), evicting the oldest rows."""
        n = batch.size
        if n == 0:
            return
        if n > self.capacity:
            batch = batch.map(lambda t: t[n - self.capacity :])
            n = self.capacity
        pairs = list(zip(self.storage.tensors(), batch.tensors()))
        end_pos = self.write_pos + n
        if end_pos <= self.capacity:
            for dst, src in pairs:
                dst[self.write_pos : end_pos].copy_(src)
        else:
            tail = self.capacity - self.write_pos
            for dst, src in pairs:
                dst[self.write_pos :].copy_(src[:tail])
                dst[: n - tail].copy_(src[tail:])
        self.write_pos = end_pos % self.capacity
        self.count = min(self.count + n, self.capacity)

    def sample(self, batch_size: int) -> TrainingBatch:
        if self.count <= 0:
            raise ValueError("Replay mirror is empty")
        indices = torch.randint(0, self.count, (batch_size,), device=self.device)
        return self.storage.map(lambda t: t.index_select(0, indices)).decode()
