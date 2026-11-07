from __future__ import annotations

import torch

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE
from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT

PIECES_TWO_ROTATIONS: set[int] = {0, 3, 4}
PIECES_ONE_ROTATION: set[int] = {1}
MIRROR_PIECE_TYPE_ORDER: tuple[int, ...] = (0, 1, 2, 4, 3, 6, 5)

LEGACY_X_MIN = -3
LEGACY_X_MAX_EXCLUSIVE = 10
LEGACY_Y_MIN = -3
LEGACY_Y_MAX_EXCLUSIVE = 20

TETROMINO_CELLS: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = (
    (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    (
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
    ),
    (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
    (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
)

_EXTRA_FORCED_MASKS: dict[int, set[tuple[int, int]]] = {
    1: {(x, BOARD_HEIGHT - 2) for x in range(BOARD_WIDTH)},
    2: {
        *((BOARD_WIDTH - 2, y) for y in range(BOARD_HEIGHT)),
        *((x, BOARD_HEIGHT - 1) for x in range(BOARD_WIDTH)),
    },
    3: {
        *((BOARD_WIDTH - 1, y) for y in range(BOARD_HEIGHT)),
        *((x, BOARD_HEIGHT - 2) for x in range(BOARD_WIDTH)),
    },
}


def _is_redundant_rotation(piece_type: int, rotation: int) -> bool:
    if piece_type in PIECES_ONE_ROTATION:
        return rotation >= 1
    if piece_type in PIECES_TWO_ROTATIONS:
        return rotation >= 2
    return False


def _piece_bounds(piece_type: int, rotation: int) -> tuple[int, int]:
    cells = TETROMINO_CELLS[piece_type][rotation]
    min_dx = min(dx for dx, _ in cells)
    min_dy = min(dy for _, dy in cells)
    return min_dx, min_dy


def _occupied_cells_for_grid(
    piece_type: int,
    rotation: int,
    grid_x: int,
    grid_y: int,
) -> tuple[tuple[int, int], ...]:
    min_dx, min_dy = _piece_bounds(piece_type, rotation)
    anchor_x = grid_x - min_dx
    anchor_y = grid_y - min_dy
    return tuple(
        sorted(
            (anchor_x + dx, anchor_y + dy)
            for dx, dy in TETROMINO_CELLS[piece_type][rotation]
        )
    )


def _occupied_cells_for_legacy(
    piece_type: int,
    rotation: int,
    x: int,
    y: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((x + dx, y + dy) for dx, dy in TETROMINO_CELLS[piece_type][rotation])
    )


def _cells_are_in_bounds(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT for x, y in cells)


def _is_valid_grid_position(
    piece_type: int, rotation: int, grid_x: int, grid_y: int
) -> bool:
    return _cells_are_in_bounds(
        _occupied_cells_for_grid(piece_type, rotation, grid_x, grid_y)
    )


def _is_valid_legacy_position(piece_type: int, rotation: int, x: int, y: int) -> bool:
    return _cells_are_in_bounds(_occupied_cells_for_legacy(piece_type, rotation, x, y))


def _is_force_masked(rotation: int, x: int, y: int) -> bool:
    masks = _EXTRA_FORCED_MASKS.get(rotation)
    return masks is not None and (x, y) in masks


def _build_flat_action_index_to_cell() -> list[tuple[int, int, int]]:
    flat_actions: list[tuple[int, int, int]] = []
    for rotation in range(4):
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if _is_force_masked(rotation, x, y):
                    continue
                if any(
                    _is_valid_grid_position(piece_type, rotation, x, y)
                    for piece_type in range(NUM_PIECE_TYPES)
                ):
                    flat_actions.append((rotation, x, y))
    return flat_actions


def _build_flat_action_cells() -> tuple[tuple[tuple[int, int], ...], ...]:
    cells_by_rotation: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for rotation, x, y in FLAT_ACTION_INDEX_TO_CELL:
        cells_by_rotation[rotation].append((x, y))
    return tuple(tuple(cells) for cells in cells_by_rotation)


def _build_legacy_action_positions() -> list[tuple[int, int, int]]:
    positions: list[tuple[int, int, int]] = []
    for y in range(LEGACY_Y_MIN, LEGACY_Y_MAX_EXCLUSIVE):
        for x in range(LEGACY_X_MIN, LEGACY_X_MAX_EXCLUSIVE):
            for rotation in range(4):
                if any(
                    _is_valid_legacy_position(piece_type, rotation, x, y)
                    for piece_type in range(NUM_PIECE_TYPES)
                ):
                    positions.append((x, y, rotation))
    positions.sort(key=lambda position: (position[2], position[1], position[0]))
    return positions


FLAT_ACTION_INDEX_TO_CELL = _build_flat_action_index_to_cell()
FLAT_ACTION_CELLS = _build_flat_action_cells()
NEW_NUM_PLACEMENT_ACTIONS = len(FLAT_ACTION_INDEX_TO_CELL)
NEW_HOLD_ACTION_INDEX = NEW_NUM_PLACEMENT_ACTIONS
NEW_NUM_ACTIONS = NEW_HOLD_ACTION_INDEX + 1

LEGACY_ACTION_POSITIONS = _build_legacy_action_positions()
LEGACY_NUM_PLACEMENT_ACTIONS = len(LEGACY_ACTION_POSITIONS)
LEGACY_HOLD_ACTION_INDEX = LEGACY_NUM_PLACEMENT_ACTIONS
LEGACY_NUM_ACTIONS = LEGACY_HOLD_ACTION_INDEX + 1

if NEW_NUM_PLACEMENT_ACTIONS != 671:
    raise ValueError(
        "Flat action-space size drifted; expected 671 placements, "
        f"got {NEW_NUM_PLACEMENT_ACTIONS}"
    )
if LEGACY_NUM_PLACEMENT_ACTIONS != 734:
    raise ValueError(
        "Legacy action-space size drifted; expected 734 placements, "
        f"got {LEGACY_NUM_PLACEMENT_ACTIONS}"
    )


def _build_flat_lookup_by_piece() -> tuple[
    list[dict[tuple[tuple[int, int], ...], int]],
    list[list[int]],
]:
    index_by_cells: list[dict[tuple[tuple[int, int], ...], int]] = []
    valid_indices_by_piece: list[list[int]] = []
    for piece_type in range(NUM_PIECE_TYPES):
        piece_lookup: dict[tuple[tuple[int, int], ...], int] = {}
        valid_indices: list[int] = []
        for flat_index, (rotation, x, y) in enumerate(FLAT_ACTION_INDEX_TO_CELL):
            if _is_redundant_rotation(piece_type, rotation):
                continue
            occupied_cells = _occupied_cells_for_grid(piece_type, rotation, x, y)
            if not _cells_are_in_bounds(occupied_cells):
                continue
            if occupied_cells in piece_lookup:
                raise ValueError(
                    "Flat action-space compression collided for a piece placement: "
                    f"piece={piece_type}, flat_index={flat_index}"
                )
            piece_lookup[occupied_cells] = flat_index
            valid_indices.append(flat_index)
        index_by_cells.append(piece_lookup)
        valid_indices_by_piece.append(valid_indices)
    return index_by_cells, valid_indices_by_piece


FLAT_INDEX_BY_CELLS_BY_PIECE, FLAT_VALID_INDICES_BY_PIECE = (
    _build_flat_lookup_by_piece()
)


def _build_legacy_to_flat_maps() -> tuple[torch.Tensor, torch.Tensor]:
    mapping = torch.zeros((NUM_PIECE_TYPES, LEGACY_NUM_ACTIONS), dtype=torch.long)
    valid = torch.zeros((NUM_PIECE_TYPES, LEGACY_NUM_ACTIONS), dtype=torch.bool)
    mapping[:, LEGACY_HOLD_ACTION_INDEX] = NEW_HOLD_ACTION_INDEX
    valid[:, LEGACY_HOLD_ACTION_INDEX] = True

    for piece_type in range(NUM_PIECE_TYPES):
        lookup = FLAT_INDEX_BY_CELLS_BY_PIECE[piece_type]
        for legacy_index, (x, y, rotation) in enumerate(LEGACY_ACTION_POSITIONS):
            if not _is_valid_legacy_position(piece_type, rotation, x, y):
                continue
            occupied_cells = _occupied_cells_for_legacy(piece_type, rotation, x, y)
            flat_index = lookup.get(occupied_cells)
            if flat_index is None:
                raise ValueError(
                    "Legacy placement could not be mapped into flat action space: "
                    f"piece={piece_type}, legacy_index={legacy_index}, "
                    f"x={x}, y={y}, rotation={rotation}"
                )
            mapping[piece_type, legacy_index] = flat_index
            valid[piece_type, legacy_index] = True

    return mapping, valid


def _mirror_cells(
    cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((BOARD_WIDTH - 1 - x, y) for x, y in cells))


def _build_flat_mirror_maps() -> tuple[torch.Tensor, torch.Tensor]:
    mapping = torch.zeros((NUM_PIECE_TYPES, NEW_NUM_ACTIONS), dtype=torch.long)
    valid = torch.zeros((NUM_PIECE_TYPES, NEW_NUM_ACTIONS), dtype=torch.bool)
    mapping[:, NEW_HOLD_ACTION_INDEX] = NEW_HOLD_ACTION_INDEX
    valid[:, NEW_HOLD_ACTION_INDEX] = True

    for piece_type in range(NUM_PIECE_TYPES):
        target_piece = MIRROR_PIECE_TYPE_ORDER[piece_type]
        target_lookup = FLAT_INDEX_BY_CELLS_BY_PIECE[target_piece]
        for flat_index in FLAT_VALID_INDICES_BY_PIECE[piece_type]:
            rotation, x, y = FLAT_ACTION_INDEX_TO_CELL[flat_index]
            occupied_cells = _occupied_cells_for_grid(piece_type, rotation, x, y)
            target_index = target_lookup.get(_mirror_cells(occupied_cells))
            if target_index is None:
                raise ValueError(
                    "Mirrored flat action could not be mapped: "
                    f"piece={piece_type}, flat_index={flat_index}"
                )
            mapping[piece_type, flat_index] = target_index
            valid[piece_type, flat_index] = True

    return mapping, valid


LEGACY_TO_FLAT_INDEX_BY_PIECE, LEGACY_TO_FLAT_VALID_BY_PIECE = (
    _build_legacy_to_flat_maps()
)
FLAT_MIRROR_INDEX_BY_PIECE, FLAT_MIRROR_VALID_BY_PIECE = _build_flat_mirror_maps()


MIRROR_PIECE_PERMUTATION = torch.tensor(MIRROR_PIECE_TYPE_ORDER, dtype=torch.long)

# Per-device copies of the constant lookup tables above. A fresh `.to(device)`
# per call is a blocking host->device copy, which on CUDA/MPS synchronizes the
# stream (CPU waits for all queued GPU work) on every mirrored training step.
_DEVICE_TABLE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _table_on_device(table: torch.Tensor, device: torch.device) -> torch.Tensor:
    if table.device == device:
        return table
    key = (id(table), device)
    cached = _DEVICE_TABLE_CACHE.get(key)
    if cached is None:
        cached = table.to(device)
        _DEVICE_TABLE_CACHE[key] = cached
    return cached


def _piece_permutation_tensor(device: torch.device) -> torch.Tensor:
    return _table_on_device(MIRROR_PIECE_PERMUTATION, device)


def mirror_piece_indices(piece_indices: torch.Tensor) -> torch.Tensor:
    permutation = _piece_permutation_tensor(piece_indices.device)
    return permutation.index_select(0, piece_indices)


def current_piece_indices_from_aux(aux: torch.Tensor) -> torch.Tensor:
    return aux[:, AUX_FEATURE_LAYOUT.current_piece].argmax(dim=1)


def mirror_boards(boards: torch.Tensor) -> torch.Tensor:
    return boards.flip(dims=(-1,))


def mirror_aux_features(aux: torch.Tensor) -> torch.Tensor:
    layout = AUX_FEATURE_LAYOUT
    permutation = _piece_permutation_tensor(aux.device)
    mirrored = aux.clone()
    hold_piece_nonempty = slice(layout.hold_piece.start, layout.hold_piece.stop - 1)

    mirrored[:, layout.current_piece] = aux[:, layout.current_piece].index_select(
        1, permutation
    )

    mirrored_hold = aux[:, layout.hold_piece].clone()
    mirrored_hold[:, :NUM_PIECE_TYPES] = aux[:, hold_piece_nonempty].index_select(
        1, permutation
    )
    mirrored[:, layout.hold_piece] = mirrored_hold

    next_queue = aux[:, layout.next_queue].reshape(-1, QUEUE_SIZE, NUM_PIECE_TYPES)
    mirrored[:, layout.next_queue] = next_queue.index_select(2, permutation).reshape(
        -1, QUEUE_SIZE * NUM_PIECE_TYPES
    )
    mirrored[:, layout.next_hidden_piece_probs] = aux[
        :, layout.next_hidden_piece_probs
    ].index_select(1, permutation)
    mirrored[:, layout.column_heights] = aux[:, layout.column_heights].flip(dims=(1,))
    return mirrored


def _scatter_float(
    source: torch.Tensor,
    mapping: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    output_width: int,
) -> torch.Tensor:
    masked_source = torch.where(source_valid, source, torch.zeros_like(source))
    result = torch.zeros(
        (source.shape[0], output_width),
        dtype=source.dtype,
        device=source.device,
    )
    result.scatter_add_(1, mapping, masked_source)
    return result


def _scatter_bool(
    source: torch.Tensor,
    mapping: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    output_width: int,
) -> torch.Tensor:
    scattered = _scatter_float(
        source.to(dtype=torch.float32),
        mapping,
        source_valid,
        output_width=output_width,
    )
    return scattered > 0.0


def legacy_action_masks_to_flat(
    action_masks: torch.Tensor,
    current_pieces: torch.Tensor,
) -> torch.Tensor:
    if action_masks.shape[1] != LEGACY_NUM_ACTIONS:
        raise ValueError(
            "legacy_action_masks_to_flat expects legacy-width masks: "
            f"got {action_masks.shape[1]}, expected {LEGACY_NUM_ACTIONS}"
        )
    mapping = _table_on_device(
        LEGACY_TO_FLAT_INDEX_BY_PIECE, action_masks.device
    ).index_select(0, current_pieces)
    source_valid = _table_on_device(
        LEGACY_TO_FLAT_VALID_BY_PIECE, action_masks.device
    ).index_select(0, current_pieces)
    return _scatter_bool(
        action_masks,
        mapping,
        source_valid,
        output_width=NEW_NUM_ACTIONS,
    )


def legacy_policy_targets_to_flat(
    policy_targets: torch.Tensor,
    current_pieces: torch.Tensor,
) -> torch.Tensor:
    if policy_targets.shape[1] != LEGACY_NUM_ACTIONS:
        raise ValueError(
            "legacy_policy_targets_to_flat expects legacy-width policy targets: "
            f"got {policy_targets.shape[1]}, expected {LEGACY_NUM_ACTIONS}"
        )
    mapping = _table_on_device(
        LEGACY_TO_FLAT_INDEX_BY_PIECE, policy_targets.device
    ).index_select(0, current_pieces)
    source_valid = _table_on_device(
        LEGACY_TO_FLAT_VALID_BY_PIECE, policy_targets.device
    ).index_select(0, current_pieces)
    return _scatter_float(
        policy_targets,
        mapping,
        source_valid,
        output_width=NEW_NUM_ACTIONS,
    )


def mirror_flat_action_masks(
    action_masks: torch.Tensor,
    current_pieces: torch.Tensor,
) -> torch.Tensor:
    if action_masks.shape[1] != NEW_NUM_ACTIONS:
        raise ValueError(
            "mirror_flat_action_masks expects flat-width masks: "
            f"got {action_masks.shape[1]}, expected {NEW_NUM_ACTIONS}"
        )
    mapping = _table_on_device(
        FLAT_MIRROR_INDEX_BY_PIECE, action_masks.device
    ).index_select(0, current_pieces)
    source_valid = _table_on_device(
        FLAT_MIRROR_VALID_BY_PIECE, action_masks.device
    ).index_select(0, current_pieces)
    return _scatter_bool(
        action_masks,
        mapping,
        source_valid,
        output_width=NEW_NUM_ACTIONS,
    )


def mirror_flat_policy_targets(
    policy_targets: torch.Tensor,
    current_pieces: torch.Tensor,
) -> torch.Tensor:
    if policy_targets.shape[1] != NEW_NUM_ACTIONS:
        raise ValueError(
            "mirror_flat_policy_targets expects flat-width policy targets: "
            f"got {policy_targets.shape[1]}, expected {NEW_NUM_ACTIONS}"
        )
    mapping = _table_on_device(
        FLAT_MIRROR_INDEX_BY_PIECE, policy_targets.device
    ).index_select(0, current_pieces)
    source_valid = _table_on_device(
        FLAT_MIRROR_VALID_BY_PIECE, policy_targets.device
    ).index_select(0, current_pieces)
    return _scatter_float(
        policy_targets,
        mapping,
        source_valid,
        output_width=NEW_NUM_ACTIONS,
    )


def mirror_training_tensors(
    boards: torch.Tensor,
    aux: torch.Tensor,
    policy_targets: torch.Tensor,
    action_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_pieces = current_piece_indices_from_aux(aux)
    return (
        mirror_boards(boards),
        mirror_aux_features(aux),
        mirror_flat_policy_targets(policy_targets, current_pieces),
        mirror_flat_action_masks(action_masks, current_pieces),
    )


def maybe_mirror_training_tensors(
    boards: torch.Tensor,
    aux: torch.Tensor,
    policy_targets: torch.Tensor,
    action_masks: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"mirror augmentation probability must be in [0, 1] (got {probability})"
        )
    if probability == 0.0:
        return boards, aux, policy_targets, action_masks
    if policy_targets.shape[1] != NEW_NUM_ACTIONS:
        return boards, aux, policy_targets, action_masks
    if action_masks.shape[1] != NEW_NUM_ACTIONS:
        raise ValueError(
            "policy_targets/action_masks width mismatch while applying mirror "
            f"augmentation: policy={policy_targets.shape[1]}, "
            f"masks={action_masks.shape[1]}"
        )
    # Sample on CPU to avoid forcing a host/device sync per training step on
    # accelerator backends like CUDA/MPS.
    if torch.rand(()).item() >= probability:
        return boards, aux, policy_targets, action_masks
    return mirror_training_tensors(boards, aux, policy_targets, action_masks)
