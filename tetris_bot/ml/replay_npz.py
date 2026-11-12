"""Read replay NPZ archives (`training_data.npz`) with dense policy/mask columns.

The Rust writer stores the policy compactly (`policy_legal` (N, 64) uint16 over the legal
actions in ascending order, `action_mask_bits` (N, 84) uint8 little-bit-order; see
`tetris_core/src/replay/compact.rs`). Analysis scripts index `policy_targets` and
`action_masks` as dense (N, NUM_ACTIONS) arrays, so `ReplayArchive` exposes those two
keys as lazily decoded columns. Older archives that store them densely pass through.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

import numpy as np

from tetris_bot.constants import NUM_ACTIONS

POLICY_LEGAL_KEY = "policy_legal"
MASK_BITS_KEY = "action_mask_bits"
POLICY_KEY = "policy_targets"
MASK_KEY = "action_masks"

MmapMode = Literal["r+", "r", "w+", "c"]


def decode_masks(mask_bits: np.ndarray) -> np.ndarray:
    """(N, 84) uint8 -> (N, NUM_ACTIONS) bool."""
    return np.unpackbits(
        mask_bits, axis=-1, count=NUM_ACTIONS, bitorder="little"
    ).astype(bool)


def decode_policy(policy_legal: np.ndarray, mask_bits: np.ndarray) -> np.ndarray:
    """Dense, renormalised (N, NUM_ACTIONS) float32 policy from the compact columns."""
    masks = decode_masks(mask_bits)
    quantised = policy_legal.astype(np.float32)
    slots = np.clip(np.cumsum(masks, axis=-1) - 1, 0, policy_legal.shape[-1] - 1)
    policy = np.take_along_axis(quantised, slots, axis=-1) * masks
    totals = np.maximum(policy.sum(axis=-1, keepdims=True), 1.0)
    return (policy / totals).astype(np.float32)


class _DecodedColumn:
    """Indexable dense view over the compact policy/mask columns."""

    def __init__(self, archive: ReplayArchive, key: str) -> None:
        self._archive = archive
        self._key = key

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._archive.compact_columns()[1].shape[0]), NUM_ACTIONS)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32 if self._key == POLICY_KEY else bool)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: Any) -> np.ndarray:
        policy_legal, mask_bits = self._archive.compact_columns()
        rows_bits = mask_bits[index]
        if self._key == MASK_KEY:
            return decode_masks(rows_bits)
        return decode_policy(policy_legal[index], rows_bits)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        decoded = self[:]
        return decoded if dtype is None else decoded.astype(dtype)


class ReplayArchive:
    """`np.load` result plus dense `policy_targets` / `action_masks` views."""

    def __init__(self, path: Path | str, mmap_mode: MmapMode | None = None) -> None:
        self._npz = np.load(path, mmap_mode=mmap_mode)
        self._compact = POLICY_LEGAL_KEY in self._npz.files
        self._compact_cache: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def files(self) -> list[str]:
        if not self._compact:
            return list(self._npz.files)
        return [*self._npz.files, POLICY_KEY, MASK_KEY]

    def keys(self) -> list[str]:
        return self.files

    def __contains__(self, key: object) -> bool:
        return key in self.files

    def compact_columns(self) -> tuple[np.ndarray, np.ndarray]:
        # NpzFile re-reads a member on every access; these two are small, keep them.
        if self._compact_cache is None:
            self._compact_cache = (
                self._npz[POLICY_LEGAL_KEY],
                self._npz[MASK_BITS_KEY],
            )
        return self._compact_cache

    def __getitem__(self, key: str) -> np.ndarray:
        if self._compact and key in (POLICY_KEY, MASK_KEY):
            # Supports the shape/len/row-indexing subset of ndarray that callers use.
            return cast(np.ndarray, _DecodedColumn(self, key))
        return self._npz[key]

    def close(self) -> None:
        self._compact_cache = None
        self._npz.close()

    def __enter__(self) -> ReplayArchive:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_replay_npz(
    path: Path | str, mmap_mode: MmapMode | None = None
) -> ReplayArchive:
    return ReplayArchive(path, mmap_mode=mmap_mode)
