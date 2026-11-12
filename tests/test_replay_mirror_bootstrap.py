from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_CONFIG_PATH,
    NUM_ACTIONS,
)
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.network import AUX_FEATURES
from tetris_bot.ml.replay_buffer import (
    POLICY_SLOTS,
    CircularReplayMirror,
    CompactBatch,
)
from tetris_bot.ml.trainer import Trainer


def pack_dense(
    boards: np.ndarray,
    aux: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    overhang_fields: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Dense replay rows -> the compact tuple `GameGenerator` returns."""
    policy_legal = np.zeros((len(masks), POLICY_SLOTS), dtype=np.uint16)
    for row, mask_row in enumerate(masks.astype(bool)):
        legal = np.flatnonzero(mask_row)
        policy_legal[row, : len(legal)] = np.round(policy_targets[row, legal] * 65535)
    return (
        np.packbits(boards.reshape(len(boards), -1) != 0, axis=1, bitorder="little"),
        aux.astype(np.float32),
        policy_legal,
        value_targets.astype(np.float32),
        overhang_fields.astype(np.float32),
        np.packbits(masks.astype(bool), axis=1, bitorder="little"),
    )


class FakeReplayMirrorGenerator:
    def __init__(self, *, window_start: int, arrays: tuple[np.ndarray, ...]) -> None:
        self.window_start = window_start
        self.window_end = window_start + int(arrays[0].shape[0])
        self.arrays = arrays
        self.delta_calls: list[int] = []

    def replay_buffer_delta(
        self,
        from_index: int,
        max_examples: int,
    ) -> tuple[int, int, int, tuple[np.ndarray, ...]]:
        self.delta_calls.append(from_index)
        slice_start = max(from_index, self.window_start)
        slice_end = min(slice_start + max_examples, self.window_end)
        offset_start = slice_start - self.window_start
        offset_end = slice_end - self.window_start
        return (
            self.window_start,
            self.window_end,
            slice_start,
            tuple(a[offset_start:offset_end] for a in self.arrays),
        )


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    config.replay = config.replay.model_copy(
        update={
            "buffer_size": 8,
            "replay_mirror_delta_chunk_examples": 2,
        }
    )
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": checkpoint_dir,
            "data_dir": data_dir,
        }
    )
    return config


def test_load_replay_mirror_bootstraps_from_deltas_without_snapshot(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_make_config(tmp_path), device="cpu")
    num_examples = 5
    boards = (
        np.arange(num_examples * BOARD_HEIGHT * BOARD_WIDTH, dtype=np.float32).reshape(
            num_examples, BOARD_HEIGHT * BOARD_WIDTH
        )
        % 2
    )
    aux = np.arange(num_examples * AUX_FEATURES, dtype=np.float32).reshape(
        num_examples, AUX_FEATURES
    )
    aux[:, :51] %= 2  # one-hot/binary prefix
    policy_targets = np.zeros((num_examples, NUM_ACTIONS), dtype=np.float32)
    value_targets = np.arange(num_examples, dtype=np.float32)
    overhang_fields = np.arange(num_examples, dtype=np.float32) + 0.5
    masks = np.zeros((num_examples, NUM_ACTIONS), dtype=np.float32)
    for row in range(num_examples):
        policy_targets[row, row] = 1.0
        masks[row, row] = 1.0

    generator = FakeReplayMirrorGenerator(
        window_start=7,
        arrays=pack_dense(
            boards, aux, policy_targets, value_targets, overhang_fields, masks
        ),
    )

    mirror = trainer._load_replay_mirror(generator)

    assert mirror is not None
    assert generator.delta_calls == [0, 9, 11]
    assert mirror.count == num_examples
    assert mirror.write_pos == num_examples
    assert mirror.logical_end == 12
    stored = mirror.storage.map(lambda t: t[:num_examples]).decode()
    assert torch.equal(
        stored.boards,
        torch.from_numpy(boards)
        .reshape(num_examples, 1, BOARD_HEIGHT, BOARD_WIDTH)
        .to(torch.bool),
    )
    assert torch.equal(stored.aux, torch.from_numpy(aux))
    assert torch.equal(stored.policy_targets, torch.from_numpy(policy_targets))
    assert torch.equal(stored.value_targets, torch.from_numpy(value_targets))
    assert torch.equal(stored.overhang_fields, torch.from_numpy(overhang_fields))
    assert torch.equal(stored.masks, torch.from_numpy(masks).to(torch.bool))


def test_compact_decode_matches_dense_policy_and_wraps_in_mirror() -> None:
    rng = np.random.default_rng(0)
    n = 6
    masks = rng.random((n, NUM_ACTIONS)) < 0.05
    masks[:, 671] = True
    policy = rng.random((n, NUM_ACTIONS)) * masks
    policy /= policy.sum(axis=1, keepdims=True)
    boards = rng.random((n, BOARD_HEIGHT * BOARD_WIDTH)) < 0.3
    aux = np.zeros((n, AUX_FEATURES), dtype=np.float32)
    aux[:, :51] = rng.random((n, 51)) < 0.2
    aux[:, 51:] = rng.random((n, AUX_FEATURES - 51))
    arrays = pack_dense(boards, aux, policy, np.arange(n) * 1.0, np.zeros(n), masks)

    mirror = CircularReplayMirror(4, torch.device("cpu"))
    mirror.write(CompactBatch.from_numpy(tuple(a[:3] for a in arrays)))
    mirror.write(CompactBatch.from_numpy(tuple(a[3:] for a in arrays)))
    assert mirror.count == 4 and mirror.write_pos == 2
    # Rows 2..5 survive; ring order is [4, 5, 2, 3].
    decoded = mirror.storage.decode()
    order = [4, 5, 2, 3]
    assert torch.equal(decoded.masks, torch.from_numpy(masks[order]))
    assert torch.allclose(
        decoded.policy_targets, torch.from_numpy(policy[order]).float(), atol=2e-5
    )
    assert torch.equal(decoded.aux, torch.from_numpy(aux[order]))
    assert torch.equal(decoded.boards.reshape(4, -1), torch.from_numpy(boards[order]))
