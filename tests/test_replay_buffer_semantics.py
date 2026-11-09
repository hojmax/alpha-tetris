"""Semantic replay-buffer consistency checks.

This module intentionally focuses on content-level replay integrity rather than
basic NPZ schema sanity.

The validator checks:
- board-derived features match the stored board exactly:
  `column_heights`, `max_column_heights`, `row_fill_counts`, `total_blocks`,
  `bumpiness`, `holes`, and `overhang_fields`
- `action_masks` match the legal actions from the restored state
- aux-feature layout parity:
  the stored feature fields pack into the exact 80-feature aux layout expected
  by `tetris_bot.ml.network.build_aux_features(...)`
- per-game metadata is self-consistent:
  `move_numbers` are contiguous within a game, `game_total_attacks` are
  constant within a game, and `value_targets` are nonincreasing integer-like
- for consecutive states in the same game, some legal action reproduces the
  next state, with the expected placement-count delta and hidden-piece reveal
- matched transitions reproduce the next row's `combo` and `back_to_back`
  exactly
- the matched transition action has positive policy mass

For large compressed replay archives, the file also supports streaming a
contiguous row window directly from `training_data.npz` so the semantic checks
stay fast enough to use as a practical quicktest.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import numpy.typing as npt
from numpy.lib import format as npformat
import pytest
import tetris_core.tetris_core as tetris_core

from tetris_bot.ml.network import build_aux_features
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    PROJECT_ROOT,
    QUEUE_SIZE,
)


COLUMN_HEIGHT_NORMALIZATION_DIVISOR = 8.0
MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR = 20.0
TOTAL_BLOCKS_NORMALIZATION_DIVISOR = 60.0
BUMPINESS_NORMALIZATION_DIVISOR = 200.0
HOLES_NORMALIZATION_DIVISOR = 20.0
OVERHANG_NORMALIZATION_DIVISOR = 25.0
ROW_FILL_FEATURE_ROWS = 4
HOLD_ACTION_INDEX = NUM_ACTIONS - 1
COMBO_NORMALIZATION_MAX = 4.0

DEFAULT_REPLAY_BUFFER_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "replay_buffer_quicktest.npz"
)
REPLAY_BUFFER_FIXTURE_ENV_VAR = "TETRIS_REPLAY_BUFFER_QUICKTEST_PATH"
REPLAY_BUFFER_MAX_PLACEMENTS_ENV_VAR = "TETRIS_REPLAY_BUFFER_MAX_PLACEMENTS"
REPLAY_BUFFER_PAIR_SAMPLE_LIMIT_ENV_VAR = "TETRIS_REPLAY_BUFFER_PAIR_SAMPLE_LIMIT"
REPLAY_BUFFER_ROW_SAMPLE_LIMIT_ENV_VAR = "TETRIS_REPLAY_BUFFER_ROW_SAMPLE_LIMIT"
REPLAY_BUFFER_SKIP_ROW_CHECKS_ENV_VAR = "TETRIS_REPLAY_BUFFER_SKIP_ROW_CHECKS"
REPLAY_BUFFER_SKIP_TRANSITION_CHECKS_ENV_VAR = (
    "TETRIS_REPLAY_BUFFER_SKIP_TRANSITION_CHECKS"
)
REPLAY_BUFFER_WINDOW_START_ENV_VAR = "TETRIS_REPLAY_BUFFER_WINDOW_START"
REPLAY_BUFFER_WINDOW_SIZE_ENV_VAR = "TETRIS_REPLAY_BUFFER_WINDOW_SIZE"

REQUIRED_KEYS = (
    "boards",
    "current_pieces",
    "hold_pieces",
    "hold_available",
    "next_queue",
    "move_numbers",
    "placement_counts",
    "combos",
    "back_to_back",
    "next_hidden_piece_probs",
    "column_heights",
    "max_column_heights",
    "row_fill_counts",
    "total_blocks",
    "bumpiness",
    "holes",
    "policy_targets",
    "value_targets",
    "action_masks",
    "overhang_fields",
    "game_numbers",
    "game_total_attacks",
)

type StateRowValue = npt.NDArray[Any] | np.generic
type StateRow = dict[str, StateRowValue]


@dataclass
class ValidationErrors:
    max_errors: int = 40
    errors: list[str] = field(default_factory=list)
    total_errors: int = 0

    def add(self, message: str) -> None:
        self.total_errors += 1
        if len(self.errors) < self.max_errors:
            self.errors.append(message)

    def assert_empty(self) -> None:
        if not self.total_errors:
            return

        if self.total_errors > len(self.errors):
            omitted = self.total_errors - len(self.errors)
            details = "\n".join([*self.errors, f"... and {omitted} more"])
        else:
            details = "\n".join(self.errors)
        raise AssertionError(details)


def _window_start() -> int:
    raw_value = os.environ.get(REPLAY_BUFFER_WINDOW_START_ENV_VAR)
    if raw_value is None:
        return 0
    parsed = int(raw_value)
    if parsed < 0:
        raise ValueError(f"{REPLAY_BUFFER_WINDOW_START_ENV_VAR} must be >= 0")
    return parsed


def _window_size() -> int | None:
    raw_value = os.environ.get(REPLAY_BUFFER_WINDOW_SIZE_ENV_VAR)
    if raw_value is None:
        return None
    parsed = int(raw_value)
    if parsed <= 0:
        raise ValueError(f"{REPLAY_BUFFER_WINDOW_SIZE_ENV_VAR} must be > 0")
    return parsed


def _read_exact(stream: Any, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = num_bytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {num_bytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _discard_bytes(stream: Any, num_bytes: int) -> None:
    remaining = num_bytes
    chunk_size = 1 << 20
    while remaining > 0:
        chunk = stream.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF while discarding {num_bytes} bytes")
        remaining -= len(chunk)


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")

    if fortran_order:
        raise ValueError(
            "Fortran-ordered arrays are not supported in replay quicktests"
        )

    return tuple(int(dim) for dim in shape), np.dtype(dtype)


def _read_npy_window_from_npz(
    npz_path: Path,
    *,
    key: str,
    start: int,
    length: int,
) -> np.ndarray:
    entry_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        with archive.open(entry_name) as stream:
            shape, dtype = _read_npy_header(stream)
            if not shape:
                raise ValueError(
                    f"{entry_name} is scalar; expected at least 1 dimension"
                )
            total_rows = shape[0]
            if start >= total_rows:
                raise ValueError(
                    f"Window start {start} is out of range for {key} with {total_rows} rows"
                )
            window_rows = min(length, total_rows - start)
            trailing_shape = shape[1:]
            items_per_row = (
                int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
            )
            row_nbytes = items_per_row * dtype.itemsize
            _discard_bytes(stream, start * row_nbytes)
            payload = _read_exact(stream, window_rows * row_nbytes)
            flat = np.frombuffer(
                payload, dtype=dtype, count=window_rows * items_per_row
            )
            return flat.reshape((window_rows, *trailing_shape))


def _load_contiguous_replay_window(
    npz_path: Path,
    *,
    start: int,
    length: int,
    keys: tuple[str, ...] = REQUIRED_KEYS,
) -> dict[str, np.ndarray]:
    return {
        key: _read_npy_window_from_npz(npz_path, key=key, start=start, length=length)
        for key in keys
    }


def _piece_type_from_one_hot(one_hot: np.ndarray) -> int | None:
    idx = int(np.argmax(one_hot))
    if one_hot.shape[0] == NUM_PIECE_TYPES + 1 and idx == NUM_PIECE_TYPES:
        return None
    return idx


def _queue_from_one_hot(queue_row: np.ndarray) -> list[int]:
    return [int(np.argmax(slot)) for slot in queue_row]


def _restore_env(data: dict[str, np.ndarray] | np.lib.npyio.NpzFile, index: int) -> Any:
    env = tetris_core.TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 0)
    board = np.asarray(data["boards"][index], dtype=np.uint8)
    current_piece = _piece_type_from_one_hot(np.asarray(data["current_pieces"][index]))
    if current_piece is None:
        raise AssertionError(f"row {index}: current_pieces decodes to None")
    env.set_board(board.tolist())
    env.set_current_piece_type(current_piece)
    env.set_hold_piece_type(_piece_type_from_one_hot(data["hold_pieces"][index]))
    env.set_hold_used(not bool(data["hold_available"][index]))
    env.set_queue(_queue_from_one_hot(np.asarray(data["next_queue"][index])))
    return env


def _compute_board_diagnostics(
    board: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float, float]:
    filled = board.astype(bool, copy=False)

    raw_column_heights = np.zeros(BOARD_WIDTH, dtype=np.int32)
    for x in range(BOARD_WIDTH):
        filled_rows = np.flatnonzero(filled[:, x])
        if filled_rows.size > 0:
            raw_column_heights[x] = BOARD_HEIGHT - int(filled_rows[0])

    raw_row_fill_counts = filled.sum(axis=1, dtype=np.int32)
    raw_total_blocks = int(raw_row_fill_counts.sum())

    raw_bumpiness = 0
    if BOARD_WIDTH >= 2:
        deltas = np.diff(raw_column_heights)
        raw_bumpiness = int(np.sum(deltas * deltas))

    empty = ~filled
    reachable = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    frontier: deque[tuple[int, int]] = deque()
    for x in range(BOARD_WIDTH):
        if not empty[0, x]:
            continue
        reachable[0, x] = True
        frontier.append((0, x))

    while frontier:
        y, x = frontier.popleft()
        if y > 0 and empty[y - 1, x] and not reachable[y - 1, x]:
            reachable[y - 1, x] = True
            frontier.append((y - 1, x))
        if y + 1 < BOARD_HEIGHT and empty[y + 1, x] and not reachable[y + 1, x]:
            reachable[y + 1, x] = True
            frontier.append((y + 1, x))
        if x > 0 and empty[y, x - 1] and not reachable[y, x - 1]:
            reachable[y, x - 1] = True
            frontier.append((y, x - 1))
        if x + 1 < BOARD_WIDTH and empty[y, x + 1] and not reachable[y, x + 1]:
            reachable[y, x + 1] = True
            frontier.append((y, x + 1))

    raw_overhang_fields = 0
    raw_holes = 0
    for x in range(BOARD_WIDTH):
        seen_filled = False
        for y in range(BOARD_HEIGHT):
            if filled[y, x]:
                seen_filled = True
                continue
            if not seen_filled:
                continue
            raw_overhang_fields += 1
            if not reachable[y, x]:
                raw_holes += 1

    normalized_column_heights = (
        raw_column_heights.astype(np.float32) / COLUMN_HEIGHT_NORMALIZATION_DIVISOR
    )
    max_column_height = (
        float(np.max(raw_column_heights)) / MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR
    )
    normalized_row_fill_counts = raw_row_fill_counts[-ROW_FILL_FEATURE_ROWS:].astype(
        np.float32
    ) / float(BOARD_WIDTH)
    total_blocks = float(raw_total_blocks) / TOTAL_BLOCKS_NORMALIZATION_DIVISOR
    bumpiness = float(raw_bumpiness) / BUMPINESS_NORMALIZATION_DIVISOR
    holes = float(raw_holes) / HOLES_NORMALIZATION_DIVISOR
    overhang_fields = float(raw_overhang_fields) / OVERHANG_NORMALIZATION_DIVISOR
    return (
        normalized_column_heights,
        max_column_height,
        normalized_row_fill_counts,
        total_blocks,
        bumpiness,
        holes,
        overhang_fields,
    )


def _hidden_piece_distribution(env: Any) -> np.ndarray:
    visible_state = env.clone_state()
    visible_state.truncate_queue(QUEUE_SIZE)
    possible_next_pieces = visible_state.get_possible_next_pieces()
    if len(possible_next_pieces) == 0:
        raise AssertionError("Hidden-piece candidate set must not be empty")
    probabilities = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    probability = 1.0 / float(len(possible_next_pieces))
    for piece_type in possible_next_pieces:
        probabilities[int(piece_type)] = probability
    return probabilities


def _policy_row(chosen_action: int, action_mask: np.ndarray) -> np.ndarray:
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[chosen_action] = 1.0
    policy[~action_mask] = 0.0
    return policy


def _stored_aux_vector(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
) -> np.ndarray:
    hold_available = np.array([float(data["hold_available"][index])], dtype=np.float32)
    placement_count = np.array(
        [float(data["placement_counts"][index])], dtype=np.float32
    )
    combo = np.array([float(data["combos"][index])], dtype=np.float32)
    back_to_back = np.array([float(data["back_to_back"][index])], dtype=np.float32)
    max_column_height = np.array(
        [float(data["max_column_heights"][index])],
        dtype=np.float32,
    )
    total_blocks = np.array([float(data["total_blocks"][index])], dtype=np.float32)
    bumpiness = np.array([float(data["bumpiness"][index])], dtype=np.float32)
    holes = np.array([float(data["holes"][index])], dtype=np.float32)
    overhang_fields = np.array(
        [float(data["overhang_fields"][index])],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            np.asarray(data["current_pieces"][index], dtype=np.float32).reshape(-1),
            np.asarray(data["hold_pieces"][index], dtype=np.float32).reshape(-1),
            hold_available,
            np.asarray(data["next_queue"][index], dtype=np.float32).reshape(-1),
            placement_count,
            combo,
            back_to_back,
            np.asarray(
                data["next_hidden_piece_probs"][index], dtype=np.float32
            ).reshape(-1),
            np.asarray(data["column_heights"][index], dtype=np.float32).reshape(-1),
            max_column_height,
            np.asarray(data["row_fill_counts"][index], dtype=np.float32).reshape(-1),
            total_blocks,
            bumpiness,
            holes,
            overhang_fields,
        ]
    )


def _network_aux_vector(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
) -> np.ndarray:
    return build_aux_features(
        current_piece=np.asarray(data["current_pieces"][index], dtype=np.float32),
        hold_piece=np.asarray(data["hold_pieces"][index], dtype=np.float32),
        hold_available=float(data["hold_available"][index]),
        next_queue=np.asarray(data["next_queue"][index], dtype=np.float32),
        placement_count=float(data["placement_counts"][index]),
        combo_feature=float(data["combos"][index]),
        back_to_back=float(data["back_to_back"][index]),
        next_hidden_piece_probs=np.asarray(
            data["next_hidden_piece_probs"][index],
            dtype=np.float32,
        ),
        column_heights=np.asarray(data["column_heights"][index], dtype=np.float32),
        max_column_height=float(data["max_column_heights"][index]),
        row_fill_counts=np.asarray(data["row_fill_counts"][index], dtype=np.float32),
        total_blocks=float(data["total_blocks"][index]),
        bumpiness=float(data["bumpiness"][index]),
        holes=float(data["holes"][index]),
        overhang_fields=float(data["overhang_fields"][index]),
    )


def _placement_index(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    max_placements: int,
) -> int:
    placement_feature = float(data["placement_counts"][index])
    placement_index = placement_feature * float(max_placements)
    if not _is_close_to_integer(placement_index):
        raise AssertionError(
            f"row {index}: placement_counts is not integer-like for max_placements="
            f"{max_placements}: {placement_feature}"
        )
    return int(round(placement_index))


def _combo_attack(combo_before: int) -> int:
    combo_table = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    if combo_before < len(combo_table):
        return combo_table[combo_before]
    return 5


def _resolved_attack_for_transition(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    action_idx: int,
    env_after: Any,
) -> int:
    if action_idx == HOLD_ACTION_INDEX:
        return 0

    result = env_after.get_last_attack_result()
    if result is None:
        return 0

    if int(result.lines_cleared) == 0:
        combo_attack = 0
    else:
        combo_before = int(
            round(float(data["combos"][index]) * COMBO_NORMALIZATION_MAX)
        )
        combo_attack = _combo_attack(combo_before)

    if bool(result.is_perfect_clear):
        # Jstris: a perfect clear replaces the clear's attack and B2B bonus; combo still adds.
        return 10 + combo_attack

    is_mini_single = bool(result.is_mini_tspin) and int(result.lines_cleared) == 1
    receives_back_to_back_bonus = (bool(result.is_tspin) and not is_mini_single) or int(
        result.lines_cleared
    ) == 4
    back_to_back_attack = (
        1 if bool(data["back_to_back"][index]) and receives_back_to_back_bonus else 0
    )
    return int(result.base_attack) + combo_attack + back_to_back_attack


def _expected_next_combo_feature(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    action_idx: int,
    env_after: Any,
) -> float:
    if action_idx == HOLD_ACTION_INDEX:
        return float(data["combos"][index])

    result = env_after.get_last_attack_result()
    if result is None:
        return 0.0

    current_combo = int(round(float(data["combos"][index]) * COMBO_NORMALIZATION_MAX))
    return float(current_combo + 1) / COMBO_NORMALIZATION_MAX


def _expected_next_back_to_back(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    action_idx: int,
    env_after: Any,
) -> bool:
    if action_idx == HOLD_ACTION_INDEX:
        return bool(data["back_to_back"][index])

    result = env_after.get_last_attack_result()
    if result is None:
        return bool(data["back_to_back"][index])

    return (
        bool(result.is_perfect_clear)
        or bool(result.is_tspin)
        or int(result.lines_cleared) == 4
    )


def _collect_state_row(
    env: Any,
    *,
    move_number: int,
    placement_count: int,
    max_placements: int,
    chosen_action: int,
) -> StateRow:
    board = np.asarray(env.get_board(), dtype=np.uint8)
    current_piece = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    current_piece[env.get_current_piece().piece_type] = 1.0

    hold_piece = np.zeros(NUM_PIECE_TYPES + 1, dtype=np.float32)
    held = env.get_hold_piece()
    if held is None:
        hold_piece[NUM_PIECE_TYPES] = 1.0
    else:
        hold_piece[held.piece_type] = 1.0

    next_queue = np.zeros((QUEUE_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    queue = env.get_queue(QUEUE_SIZE)
    for slot, piece_type in enumerate(queue):
        next_queue[slot, piece_type] = 1.0

    (
        column_heights,
        max_column_height,
        row_fill_counts,
        total_blocks,
        bumpiness,
        holes,
        overhang_fields,
    ) = _compute_board_diagnostics(board)
    action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)

    return {
        "boards": board,
        "current_pieces": current_piece,
        "hold_pieces": hold_piece,
        "hold_available": np.bool_(not env.is_hold_used()),
        "next_queue": next_queue,
        "move_numbers": np.uint32(move_number),
        "placement_counts": np.float32(placement_count / max_placements),
        "combos": np.float32(env.combo / 4.0),
        "back_to_back": np.bool_(env.back_to_back),
        "next_hidden_piece_probs": _hidden_piece_distribution(env),
        "column_heights": column_heights,
        "max_column_heights": np.float32(max_column_height),
        "row_fill_counts": row_fill_counts,
        "total_blocks": np.float32(total_blocks),
        "bumpiness": np.float32(bumpiness),
        "holes": np.float32(holes),
        "policy_targets": _policy_row(chosen_action, action_mask),
        "action_masks": action_mask,
        "overhang_fields": np.float32(overhang_fields),
    }


def _stack_rows(rows: list[StateRow], attacks: list[int]) -> dict[str, np.ndarray]:
    values = np.zeros(len(attacks), dtype=np.float32)
    cumulative = 0.0
    for i in range(len(attacks) - 1, -1, -1):
        cumulative += float(attacks[i])
        values[i] = cumulative

    data: dict[str, np.ndarray] = {}
    for key in rows[0]:
        data[key] = np.stack([row[key] for row in rows])

    total_attack = np.uint32(sum(attacks))
    data["value_targets"] = values
    data["game_numbers"] = np.ones(len(rows), dtype=np.uint64)
    data["game_total_attacks"] = np.full(len(rows), total_attack, dtype=np.uint32)
    return data


def _build_synthetic_game_data(max_placements: int = 100) -> dict[str, np.ndarray]:
    env = tetris_core.TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 123)
    rows: list[StateRow] = []
    attacks: list[int] = []
    move_number = 0
    placement_count = 0

    for step_idx in range(4):
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)
        if step_idx == 0 and action_mask[HOLD_ACTION_INDEX]:
            chosen_action = HOLD_ACTION_INDEX
        else:
            placement_actions = np.flatnonzero(action_mask[:-1])
            if placement_actions.size == 0:
                raise AssertionError(
                    "Synthetic trajectory ran out of placement actions"
                )
            chosen_action = int(placement_actions[0])

        rows.append(
            _collect_state_row(
                env,
                move_number=move_number,
                placement_count=placement_count,
                max_placements=max_placements,
                chosen_action=chosen_action,
            )
        )
        attack = env.execute_action_index(chosen_action)
        assert attack is not None
        attacks.append(int(attack))
        move_number += 1
        if chosen_action != HOLD_ACTION_INDEX:
            placement_count += 1

    return _stack_rows(rows, attacks)


def _approx_equal(actual: float, expected: float, *, atol: float = 1e-6) -> bool:
    return abs(actual - expected) <= atol


def _is_close_to_integer(value: float, *, atol: float = 1e-4) -> bool:
    return _approx_equal(value, round(value), atol=atol)


def _iter_game_segments(
    game_numbers: np.ndarray,
) -> list[tuple[int, int]]:
    if game_numbers.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(game_numbers) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [game_numbers.size]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _infer_max_placements(data: dict[str, np.ndarray] | np.lib.npyio.NpzFile) -> int:
    override = os.environ.get(REPLAY_BUFFER_MAX_PLACEMENTS_ENV_VAR)
    if override is not None:
        return int(override)

    placement_counts = np.asarray(data["placement_counts"], dtype=np.float64)
    positive = placement_counts[placement_counts > 1e-6]
    if positive.size == 0:
        raise ValueError(
            "Could not infer max_placements from placement_counts; set "
            f"{REPLAY_BUFFER_MAX_PLACEMENTS_ENV_VAR}"
        )
    min_positive = float(np.min(positive))
    inferred = int(round(1.0 / min_positive))
    if inferred <= 0:
        raise ValueError(
            "Inferred non-positive max_placements; set "
            f"{REPLAY_BUFFER_MAX_PLACEMENTS_ENV_VAR}"
        )
    scaled = placement_counts * inferred
    if not np.allclose(scaled, np.round(scaled), atol=1e-4):
        raise ValueError(
            "placement_counts are not consistent with an inferred max_placements; set "
            f"{REPLAY_BUFFER_MAX_PLACEMENTS_ENV_VAR}"
        )
    return inferred


def _pair_sample_limit() -> int | None:
    raw_value = os.environ.get(REPLAY_BUFFER_PAIR_SAMPLE_LIMIT_ENV_VAR)
    if raw_value is None:
        return 512
    parsed = int(raw_value)
    if parsed <= 0:
        return None
    return parsed


def _row_sample_limit() -> int | None:
    raw_value = os.environ.get(REPLAY_BUFFER_ROW_SAMPLE_LIMIT_ENV_VAR)
    if raw_value is None:
        return 2048
    parsed = int(raw_value)
    if parsed <= 0:
        return None
    return parsed


def _select_row_indices(total_rows: int, sample_limit: int | None) -> list[int]:
    if sample_limit is None or total_rows <= sample_limit:
        return list(range(total_rows))

    positions = np.linspace(0, total_rows - 1, num=sample_limit, dtype=np.int64)
    return sorted({int(position) for position in positions})


def _select_transition_indices(
    game_numbers: np.ndarray, sample_limit: int | None
) -> list[int]:
    transition_indices = np.flatnonzero(game_numbers[:-1] == game_numbers[1:])
    if sample_limit is None or transition_indices.size <= sample_limit:
        return transition_indices.tolist()

    positions = np.linspace(
        0,
        transition_indices.size - 1,
        num=sample_limit,
        dtype=np.int64,
    )
    selected = sorted({int(transition_indices[pos]) for pos in positions})
    return selected


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _transition_matches(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    index: int,
    action_idx: int,
    attack: int,
    env_after: Any,
    board_after: np.ndarray,
    max_placements: int,
) -> bool:
    next_index = index + 1
    next_board = np.asarray(data["boards"][next_index], dtype=np.uint8)
    if not np.array_equal(board_after, next_board):
        return False

    value_drop = float(data["value_targets"][index] - data["value_targets"][next_index])
    if not _is_close_to_integer(value_drop) or int(round(value_drop)) != attack:
        return False

    current_piece = _piece_type_from_one_hot(np.asarray(data["current_pieces"][index]))
    hold_piece = _piece_type_from_one_hot(np.asarray(data["hold_pieces"][index]))
    next_current_piece = _piece_type_from_one_hot(
        np.asarray(data["current_pieces"][next_index])
    )
    next_hold_piece = _piece_type_from_one_hot(
        np.asarray(data["hold_pieces"][next_index])
    )
    queue = _queue_from_one_hot(np.asarray(data["next_queue"][index]))
    next_queue = _queue_from_one_hot(np.asarray(data["next_queue"][next_index]))
    next_hidden_piece_probs = np.asarray(
        data["next_hidden_piece_probs"][index], dtype=np.float32
    )

    placement_delta = float(
        data["placement_counts"][next_index] - data["placement_counts"][index]
    )
    expected_step = 1.0 / float(max_placements)
    next_combo_feature = float(data["combos"][next_index])
    next_back_to_back = bool(data["back_to_back"][next_index])

    if action_idx == HOLD_ACTION_INDEX:
        if not bool(data["hold_available"][index]):
            return False
        if not _approx_equal(placement_delta, 0.0, atol=1e-6):
            return False
        if bool(data["hold_available"][next_index]):
            return False
        if next_hold_piece != current_piece:
            return False
        if not _approx_equal(
            next_combo_feature,
            _expected_next_combo_feature(data, index, action_idx, env_after),
            atol=1e-6,
        ):
            return False
        if next_back_to_back != _expected_next_back_to_back(
            data,
            index,
            action_idx,
            env_after,
        ):
            return False

        if hold_piece is None:
            if next_current_piece != queue[0]:
                return False
            if next_queue[:-1] != queue[1:]:
                return False
            revealed_piece = next_queue[-1]
            return next_hidden_piece_probs[revealed_piece] > 0.0

        if next_current_piece != hold_piece:
            return False
        return next_queue == queue

    if not _approx_equal(placement_delta, expected_step, atol=1e-6):
        return False
    if not bool(data["hold_available"][next_index]):
        return False
    if next_hold_piece != hold_piece:
        return False
    if next_current_piece != queue[0]:
        return False
    if next_queue[:-1] != queue[1:]:
        return False

    if not _approx_equal(
        next_combo_feature,
        _expected_next_combo_feature(data, index, action_idx, env_after),
        atol=1e-6,
    ):
        return False
    if next_back_to_back != _expected_next_back_to_back(
        data,
        index,
        action_idx,
        env_after,
    ):
        return False

    revealed_piece = next_queue[-1]
    return next_hidden_piece_probs[revealed_piece] > 0.0


def validate_replay_buffer_semantics(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    max_placements: int | None = None,
    pair_sample_limit: int | None = None,
    row_sample_limit: int | None = None,
) -> None:
    missing_keys = [key for key in REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise AssertionError(f"Replay buffer is missing required keys: {missing_keys}")

    errors = ValidationErrors()
    n_examples = int(len(data["boards"]))
    if n_examples == 0:
        raise AssertionError("Replay buffer must contain at least one example")

    if max_placements is None:
        max_placements = _infer_max_placements(data)
    if pair_sample_limit is None:
        pair_sample_limit = _pair_sample_limit()
    if row_sample_limit is None:
        row_sample_limit = _row_sample_limit()

    game_numbers = np.asarray(data["game_numbers"], dtype=np.uint64)
    move_numbers = np.asarray(data["move_numbers"], dtype=np.uint32)
    value_targets = np.asarray(data["value_targets"], dtype=np.float32)
    game_total_attacks = np.asarray(data["game_total_attacks"], dtype=np.uint32)

    if not _env_flag_enabled(REPLAY_BUFFER_SKIP_ROW_CHECKS_ENV_VAR):
        for index in _select_row_indices(n_examples, row_sample_limit):
            board = np.asarray(data["boards"][index], dtype=np.uint8)
            (
                expected_column_heights,
                expected_max_column_height,
                expected_row_fill_counts,
                expected_total_blocks,
                expected_bumpiness,
                expected_holes,
                expected_overhang_fields,
            ) = _compute_board_diagnostics(board)

            if not np.allclose(
                np.asarray(data["column_heights"][index], dtype=np.float32),
                expected_column_heights,
                atol=1e-6,
            ):
                errors.add(f"row {index}: column_heights do not match board")
            if not _approx_equal(
                float(data["max_column_heights"][index]),
                expected_max_column_height,
                atol=1e-6,
            ):
                errors.add(f"row {index}: max_column_heights does not match board")
            if not np.allclose(
                np.asarray(data["row_fill_counts"][index], dtype=np.float32),
                expected_row_fill_counts,
                atol=1e-6,
            ):
                errors.add(f"row {index}: row_fill_counts do not match board")
            if not _approx_equal(
                float(data["total_blocks"][index]),
                expected_total_blocks,
                atol=1e-6,
            ):
                errors.add(f"row {index}: total_blocks does not match board")
            if not _approx_equal(
                float(data["bumpiness"][index]),
                expected_bumpiness,
                atol=1e-6,
            ):
                errors.add(f"row {index}: bumpiness does not match board")
            if not _approx_equal(
                float(data["holes"][index]),
                expected_holes,
                atol=1e-6,
            ):
                errors.add(f"row {index}: holes do not match board")
            if not _approx_equal(
                float(data["overhang_fields"][index]),
                expected_overhang_fields,
                atol=1e-6,
            ):
                errors.add(f"row {index}: overhang_fields do not match board")

            env = _restore_env(data, index)
            expected_mask = np.asarray(
                tetris_core.debug_get_action_mask(env), dtype=bool
            )
            actual_mask = np.asarray(data["action_masks"][index], dtype=bool)
            if not np.array_equal(expected_mask, actual_mask):
                errors.add(f"row {index}: action_masks do not match restored state")

            if not np.allclose(
                _network_aux_vector(data, index),
                _stored_aux_vector(data, index),
                atol=1e-6,
            ):
                errors.add(f"row {index}: build_aux_features aux vector mismatch")

    for start, end in _iter_game_segments(game_numbers):
        game_number = int(game_numbers[start])
        game_total_attack = int(game_total_attacks[start])
        if not np.all(game_total_attacks[start:end] == game_total_attacks[start]):
            errors.add(
                f"game {game_number}: game_total_attacks changes within the segment"
            )

        expected_moves = np.arange(
            move_numbers[start], move_numbers[start] + (end - start)
        )
        if not np.array_equal(move_numbers[start:end], expected_moves):
            errors.add(f"game {game_number}: move_numbers are not contiguous")

        segment_values = value_targets[start:end]
        if np.any(np.diff(segment_values) > 1e-6):
            errors.add(f"game {game_number}: value_targets increase within the game")

        for offset in range(start, end - 1):
            drop = float(value_targets[offset] - value_targets[offset + 1])
            if not _is_close_to_integer(drop):
                errors.add(
                    f"game {game_number}: non-integer value drop between rows "
                    f"{offset} and {offset + 1}"
                )

        if int(move_numbers[start]) == 0:
            if not _approx_equal(
                float(segment_values[0]), float(game_total_attack), atol=1e-4
            ):
                errors.add(
                    f"game {game_number}: first value_target does not equal game_total_attack"
                )

            drop_sum = 0
            for offset in range(start, end - 1):
                drop = float(value_targets[offset] - value_targets[offset + 1])
                drop_sum += int(round(drop))

            last_value = float(value_targets[end - 1])
            if not _is_close_to_integer(last_value):
                errors.add(
                    f"game {game_number}: final value_target is not integer-like"
                )
            elif drop_sum + int(round(last_value)) != game_total_attack:
                errors.add(
                    f"game {game_number}: value_targets do not sum back to game_total_attack"
                )

    if not _env_flag_enabled(REPLAY_BUFFER_SKIP_TRANSITION_CHECKS_ENV_VAR):
        for index in _select_transition_indices(game_numbers, pair_sample_limit):
            env = _restore_env(data, index)
            action_mask = np.asarray(data["action_masks"][index], dtype=bool)
            matched_action: int | None = None
            for action_idx in np.flatnonzero(action_mask):
                env_after = env.clone_state()
                if env_after.execute_action_index(int(action_idx)) is None:
                    errors.add(
                        f"row {index}: action {action_idx} is masked valid but not executable"
                    )
                    continue
                attack = _resolved_attack_for_transition(
                    data,
                    index,
                    int(action_idx),
                    env_after,
                )
                board_after = np.asarray(env_after.get_board(), dtype=np.uint8)
                if _transition_matches(
                    data,
                    index,
                    int(action_idx),
                    int(attack),
                    env_after,
                    board_after,
                    max_placements,
                ):
                    matched_action = int(action_idx)
                    break

            if matched_action is None:
                errors.add(
                    f"row {index}: no valid action reproduces the next replay-buffer state"
                )
                continue

            if float(data["policy_targets"][index][matched_action]) <= 0.0:
                errors.add(
                    f"row {index}: matched transition action {matched_action} has zero policy mass"
                )

    errors.assert_empty()


def _resolve_replay_buffer_fixture_path() -> Path:
    override = os.environ.get(REPLAY_BUFFER_FIXTURE_ENV_VAR)
    if override is not None:
        return Path(override)
    return DEFAULT_REPLAY_BUFFER_FIXTURE_PATH


def _find_newest_training_data_npz(search_root: Path) -> Path | None:
    candidates = sorted(
        search_root.rglob("training_data.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def test_replay_buffer_semantics_on_synthetic_game() -> None:
    data = _build_synthetic_game_data(max_placements=100)
    validate_replay_buffer_semantics(data, max_placements=100, pair_sample_limit=None)


def test_load_contiguous_replay_window_reads_expected_slice(tmp_path: Path) -> None:
    npz_path = tmp_path / "window_test.npz"
    boards = np.arange(10 * BOARD_HEIGHT * BOARD_WIDTH, dtype=np.uint8).reshape(
        10, BOARD_HEIGHT, BOARD_WIDTH
    )
    value_targets = np.arange(10, dtype=np.float32)
    game_numbers = np.arange(10, dtype=np.uint64)
    np.savez_compressed(
        npz_path,
        boards=boards,
        value_targets=value_targets,
        game_numbers=game_numbers,
    )

    window = _load_contiguous_replay_window(
        npz_path,
        start=3,
        length=4,
        keys=("boards", "value_targets", "game_numbers"),
    )

    assert np.array_equal(window["boards"], boards[3:7])
    assert np.array_equal(window["value_targets"], value_targets[3:7])
    assert np.array_equal(window["game_numbers"], game_numbers[3:7])


def test_replay_buffer_semantics_on_tracked_fixture() -> None:
    fixture_path = _resolve_replay_buffer_fixture_path()
    if not fixture_path.exists():
        pytest.skip(
            "Replay buffer fixture not present. Add the tracked NPZ at "
            f"{fixture_path} or set {REPLAY_BUFFER_FIXTURE_ENV_VAR}."
        )

    window_size = _window_size()
    if window_size is not None:
        data = _load_contiguous_replay_window(
            fixture_path,
            start=_window_start(),
            length=window_size,
        )
        validate_replay_buffer_semantics(data)
        return

    with np.load(fixture_path, mmap_mode="r") as data:
        validate_replay_buffer_semantics(data)


def test_replay_buffer_semantics_on_newest_training_run_window() -> None:
    fixture_path = _find_newest_training_data_npz(PROJECT_ROOT / "training_runs")
    if fixture_path is None:
        pytest.skip("No training_data.npz found under training_runs")

    data = _load_contiguous_replay_window(fixture_path, start=0, length=512)
    validate_replay_buffer_semantics(data)
