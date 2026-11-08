"""
Board visualization for training monitoring.

Renders Tetris board states to PIL Images for logging to wandb.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_COLORS,
    PIECE_SPAWN_CELLS,
    QUEUE_SIZE,
)
from tetris_core.tetris_core import TetrisEnv


class ReplayMoveLike(Protocol):
    @property
    def action(self) -> int: ...
    @property
    def attack(self) -> int: ...


class ReplayLike(Protocol):
    """A Rust `GameReplay` or a picklable stand-in with the same fields."""

    @property
    def seed(self) -> int: ...
    @property
    def moves(self) -> Sequence[ReplayMoveLike]: ...


# Board cell size
CELL_SIZE = 20
PADDING = 10

# Sidebar layout (used when show_piece_info=True)
MINI_CELL = 12
LEFT_SIDEBAR = 76
RIGHT_SIDEBAR = 76
SIDEBAR_LABEL_Y = 2
SIDEBAR_PIECE_Y = 38
QUEUE_SLOT_SPACING = 44
STATS_Y = 86
STATS_LINE_H = 16

# Info bar heights
INFO_HEIGHT_SIMPLE = 36
INFO_HEIGHT_SIDEBAR = 34

# Colors
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
BORDER_COLOR = (80, 80, 80)
TEXT_COLOR = (200, 200, 200)
LABEL_COLOR = (140, 140, 140)
GRAY = (80, 80, 80)

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


@dataclass(frozen=True)
class PredictedMoveOverlay:
    probability: float
    piece_type: int
    cells: tuple[tuple[int, int], ...]
    rank: int
    is_hold: bool = False


@dataclass(frozen=True)
class _OverlayLabelPlacement:
    predicted_move: PredictedMoveOverlay
    text: str
    position: tuple[int, int]
    rect: tuple[int, int, int, int]


_TEXT_SHADOW_RADIUS = 1


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for font_path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


_SPAWN_X = (BOARD_WIDTH - 4) // 2


def compute_spawn_and_ghost(
    piece_type: int, board: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    spawn_y = -1 if piece_type in (0, 1) else 0  # I/O spawn one row higher
    spawn_cells = [
        (cx + _SPAWN_X, cy + spawn_y) for cx, cy in PIECE_SPAWN_CELLS[piece_type]
    ]

    # Drop until collision
    dy = 0
    while True:
        dy += 1
        for x, y in spawn_cells:
            ny = y + dy
            if ny >= BOARD_HEIGHT or (ny >= 0 and board[ny][x] != 0):
                ghost_cells = [(x, y + dy - 1) for x, y in spawn_cells]
                return spawn_cells, ghost_cells


def _draw_mini_piece(
    draw: ImageDraw.ImageDraw,
    piece_type: int,
    center_x: int,
    center_y: int,
):
    bounds = _mini_piece_bounds(piece_type, center_x, center_y, MINI_CELL)
    sx, sy = bounds[0], bounds[1]
    cells = PIECE_SPAWN_CELLS[piece_type]
    color = PIECE_COLORS[piece_type]

    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    min_x = min(xs)
    min_y = min(ys)

    for cx, cy in cells:
        px = sx + (cx - min_x) * MINI_CELL
        py = sy + (cy - min_y) * MINI_CELL
        draw.rectangle([px, py, px + MINI_CELL - 2, py + MINI_CELL - 2], fill=color)


def _mini_piece_bounds(
    piece_type: int,
    center_x: int,
    center_y: int,
    cell_size: int,
) -> tuple[int, int, int, int]:
    cells = PIECE_SPAWN_CELLS[piece_type]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = (max_x - min_x + 1) * cell_size
    height = (max_y - min_y + 1) * cell_size
    left = center_x - width // 2
    top = center_y - height // 2
    return (left, top, left + width, top + height)


def _draw_board_area(
    draw: ImageDraw.ImageDraw,
    board_x: int,
    board_y: int,
    board: np.ndarray,
    board_piece_types: list[list[int | None]] | None,
    current_piece_cells: list[tuple[int, int]] | None,
    current_piece_type: int | None,
    ghost_cells: list[tuple[int, int]] | None,
):
    # Grid
    for x in range(BOARD_WIDTH + 1):
        xp = board_x + x * CELL_SIZE
        draw.line(
            [(xp, board_y), (xp, board_y + BOARD_HEIGHT * CELL_SIZE)],
            fill=GRID_COLOR,
        )
    for y in range(BOARD_HEIGHT + 1):
        yp = board_y + y * CELL_SIZE
        draw.line(
            [(board_x, yp), (board_x + BOARD_WIDTH * CELL_SIZE, yp)],
            fill=GRID_COLOR,
        )

    # Locked pieces
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x] != 0:
                color = GRAY
                if board_piece_types is not None:
                    idx = board_piece_types[y][x]
                    if idx is not None:
                        color = PIECE_COLORS[idx]
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Ghost piece
    if ghost_cells and current_piece_type is not None:
        ghost_color = tuple(c // 2 for c in PIECE_COLORS[current_piece_type])
        for x, y in ghost_cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2],
                    outline=ghost_color,
                    width=2,
                )

    # Current piece
    if current_piece_cells and current_piece_type is not None:
        color = PIECE_COLORS[current_piece_type]
        for x, y in current_piece_cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Border
    draw.rectangle(
        [
            board_x,
            board_y,
            board_x + BOARD_WIDTH * CELL_SIZE,
            board_y + BOARD_HEIGHT * CELL_SIZE,
        ],
        outline=BORDER_COLOR,
        width=2,
    )


def _draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = position
    shadow = (0, 0, 0, 235)
    for dx, dy in (
        (-_TEXT_SHADOW_RADIUS, -_TEXT_SHADOW_RADIUS),
        (-_TEXT_SHADOW_RADIUS, 0),
        (-_TEXT_SHADOW_RADIUS, _TEXT_SHADOW_RADIUS),
        (0, -_TEXT_SHADOW_RADIUS),
        (0, _TEXT_SHADOW_RADIUS),
        (_TEXT_SHADOW_RADIUS, -_TEXT_SHADOW_RADIUS),
        (_TEXT_SHADOW_RADIUS, 0),
        (_TEXT_SHADOW_RADIUS, _TEXT_SHADOW_RADIUS),
    ):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)


def _rects_intersect(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> bool:
    left_a, top_a, right_a, bottom_a = rect_a
    left_b, top_b, right_b, bottom_b = rect_b
    return not (
        right_a <= left_b or right_b <= left_a or bottom_a <= top_b or bottom_b <= top_a
    )


def _inflate_rect(
    rect: tuple[int, int, int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    return (left - padding, top - padding, right + padding, bottom + padding)


def _clamp_label_rect(
    left: int,
    top: int,
    draw_bounds: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    min_y: int,
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    box_left, box_top, box_right, box_bottom = draw_bounds
    clamped_left = max(2 - box_left, min(img_w - box_right - 2, left))
    clamped_top = max(min_y - box_top, min(img_h - box_bottom - 2, top))
    position = (clamped_left, clamped_top)
    return (
        position,
        (
            clamped_left + box_left,
            clamped_top + box_top,
            clamped_left + box_right,
            clamped_top + box_bottom,
        ),
    )


def _measure_text_draw_bounds(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = font.getbbox(text)
    return (
        int(left) - _TEXT_SHADOW_RADIUS,
        int(top) - _TEXT_SHADOW_RADIUS,
        int(right) + _TEXT_SHADOW_RADIUS,
        int(bottom) + _TEXT_SHADOW_RADIUS,
    )


def _visible_overlay_bounds(
    board_x: int,
    board_y: int,
    cells: tuple[tuple[int, int], ...],
) -> tuple[int, int, int, int] | None:
    visible_cells = [
        (x, y) for x, y in cells if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT
    ]
    if not visible_cells:
        return None

    min_x = min(x for x, _ in visible_cells)
    max_x = max(x for x, _ in visible_cells)
    min_y = min(y for _, y in visible_cells)
    max_y = max(y for _, y in visible_cells)
    return (
        board_x + min_x * CELL_SIZE + 1,
        board_y + min_y * CELL_SIZE + 1,
        board_x + (max_x + 1) * CELL_SIZE - 1,
        board_y + (max_y + 1) * CELL_SIZE - 1,
    )


def _overlay_anchor_bounds(
    board_x: int,
    board_y: int,
    predicted_move: PredictedMoveOverlay,
) -> tuple[int, int, int, int] | None:
    if predicted_move.is_hold and board_x == LEFT_SIDEBAR:
        return _mini_piece_bounds(
            predicted_move.piece_type,
            LEFT_SIDEBAR // 2,
            board_y + SIDEBAR_PIECE_Y,
            MINI_CELL,
        )
    return _visible_overlay_bounds(board_x, board_y, predicted_move.cells)


def _overlay_fill_alpha(rank: int) -> int:
    rank = max(1, min(rank, 3))
    return 64 - (rank - 1) * 14


def _overlay_outline_alpha(rank: int) -> int:
    rank = max(1, min(rank, 3))
    return 245 - (rank - 1) * 20


def _overlay_outline_width(rank: int) -> int:
    return 1


def _count_overlay_cell_overlaps(
    predicted_move_overlays: list[PredictedMoveOverlay],
) -> Counter[tuple[int, int]]:
    overlap_counts: Counter[tuple[int, int]] = Counter()
    for predicted_move in predicted_move_overlays:
        if predicted_move.is_hold:
            continue
        for x, y in predicted_move.cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                overlap_counts[(x, y)] += 1
    return overlap_counts


def _candidate_label_rects(
    bounds: tuple[int, int, int, int],
    text_w: int,
    text_h: int,
    is_hold: bool,
) -> list[tuple[int, int]]:
    left, top, right, bottom = bounds
    center_x = (left + right - text_w) // 2
    center_y = (top + bottom - text_h) // 2
    mid_y = (top + bottom) // 2 - text_h // 2
    label_gap = 6

    raw_positions = (
        [
            (center_x, bottom + label_gap),
            (center_x, bottom + label_gap + text_h + 4),
            (center_x, top - text_h - label_gap),
            (center_x, top - text_h - label_gap),
            (right + label_gap, mid_y),
            (left - text_w - label_gap, mid_y),
        ]
        if is_hold
        else [
            (center_x, center_y),
            (center_x, top - text_h - label_gap),
            (center_x, bottom + label_gap),
            (left - text_w - label_gap, mid_y),
            (right + label_gap, mid_y),
            (left - text_w - label_gap, top - text_h - label_gap),
            (right + label_gap, top - text_h - label_gap),
            (left - text_w - label_gap, bottom + label_gap),
            (right + label_gap, bottom + label_gap),
        ]
    )

    candidate_positions: list[tuple[int, int]] = []
    seen_positions: set[tuple[int, int]] = set()
    for raw_left, raw_top in raw_positions:
        position = (raw_left, raw_top)
        if position not in seen_positions:
            candidate_positions.append(position)
            seen_positions.add(position)
    return candidate_positions


def _place_overlay_label_rects(
    board_x: int,
    board_y: int,
    img_size: tuple[int, int],
    predicted_move_overlays: list[PredictedMoveOverlay],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[_OverlayLabelPlacement]:
    img_w, img_h = img_size
    occupied_rects: list[tuple[int, int, int, int]] = []
    placements: list[_OverlayLabelPlacement] = []

    for predicted_move in sorted(
        predicted_move_overlays, key=lambda overlay: overlay.rank
    ):
        label = f"{predicted_move.probability * 100:.1f}%"
        bounds = _overlay_anchor_bounds(board_x, board_y, predicted_move)
        if bounds is None:
            continue

        draw_bounds = _measure_text_draw_bounds(label, font)
        text_w = draw_bounds[2] - draw_bounds[0]
        text_h = draw_bounds[3] - draw_bounds[1]
        candidate_positions = _candidate_label_rects(
            bounds,
            text_w,
            text_h,
            predicted_move.is_hold,
        )

        chosen_position, chosen_rect = _clamp_label_rect(
            candidate_positions[0][0],
            candidate_positions[0][1],
            draw_bounds,
            img_w,
            img_h,
            board_y + 2,
        )
        inflated_occupied = [_inflate_rect(rect, 3) for rect in occupied_rects]
        for raw_left, raw_top in candidate_positions:
            candidate_position, candidate = _clamp_label_rect(
                raw_left,
                raw_top,
                draw_bounds,
                img_w,
                img_h,
                board_y + 2,
            )
            if not any(
                _rects_intersect(candidate, occupied) for occupied in inflated_occupied
            ):
                chosen_position = candidate_position
                chosen_rect = candidate
                break
        else:
            base_left, base_top = candidate_positions[0]
            step_y = text_h + 6
            for offset_idx in range(1, 10):
                for signed_offset in (offset_idx, -offset_idx):
                    candidate_position, candidate = _clamp_label_rect(
                        base_left,
                        base_top + signed_offset * step_y,
                        draw_bounds,
                        img_w,
                        img_h,
                        board_y + 2,
                    )
                    if not any(
                        _rects_intersect(candidate, occupied)
                        for occupied in inflated_occupied
                    ):
                        chosen_position = candidate_position
                        chosen_rect = candidate
                        break
                else:
                    continue
                break

        placements.append(
            _OverlayLabelPlacement(
                predicted_move=predicted_move,
                text=label,
                position=chosen_position,
                rect=chosen_rect,
            )
        )
        occupied_rects.append(chosen_rect)

    return placements


def _apply_predicted_move_overlays(
    img: Image.Image,
    board_x: int,
    board_y: int,
    predicted_move_overlays: list[PredictedMoveOverlay],
) -> Image.Image:
    if not predicted_move_overlays:
        return img

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(12)

    # Draw piece fills first so overlaps remain readable.
    for predicted_move in reversed(predicted_move_overlays):
        if predicted_move.is_hold:
            continue
        color = PIECE_COLORS[predicted_move.piece_type]
        fill_color = (*color, _overlay_fill_alpha(predicted_move.rank))
        for x, y in predicted_move.cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2],
                    fill=fill_color,
                )

    # Draw outlines in a separate pass so stacked placements stay visible.
    overlap_counts = _count_overlay_cell_overlaps(predicted_move_overlays)
    for predicted_move in reversed(predicted_move_overlays):
        if predicted_move.is_hold:
            continue
        color = PIECE_COLORS[predicted_move.piece_type]
        outline_color = (*color, _overlay_outline_alpha(predicted_move.rank))
        for x, y in predicted_move.cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                overlap_width = 2 if overlap_counts[(x, y)] >= 2 else 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2],
                    outline=outline_color,
                    width=max(
                        _overlay_outline_width(predicted_move.rank), overlap_width
                    ),
                )

    label_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_overlay)
    label_placements = _place_overlay_label_rects(
        board_x,
        board_y,
        img.size,
        predicted_move_overlays,
        font,
    )
    for label_placement in label_placements:
        _draw_text_with_shadow(
            label_draw,
            (label_placement.position[0] + 2, label_placement.position[1]),
            label_placement.text,
            font,
            fill=(255, 255, 255, 255),
        )

    composited = Image.alpha_composite(img.convert("RGBA"), overlay)
    composited = Image.alpha_composite(composited, label_overlay)
    return composited.convert("RGB")


def render_board(
    board: np.ndarray,
    board_piece_types: list[list[int | None]] | None = None,
    current_piece_cells: list[tuple[int, int]] | None = None,
    current_piece_type: int | None = None,
    ghost_cells: list[tuple[int, int]] | None = None,
    placement_number: int = 0,
    placement_label: str = "Placement",
    attack: int = 0,
    total_attack: int | None = None,
    total_placements: int | None = None,
    info_text: str | None = None,
    can_hold: bool | None = None,
    combo: int | None = None,
    back_to_back: bool | None = None,
    is_terminal: bool = False,
    value_pred: float | None = None,
    # Sidebar piece display
    show_piece_info: bool = False,
    hold_piece_type: int | None = None,
    queue_piece_types: list[int] | None = None,
    predicted_move_overlays: list[PredictedMoveOverlay] | None = None,
    show_ghost_piece: bool = True,
) -> Image.Image:
    board_w_px = BOARD_WIDTH * CELL_SIZE
    board_h_px = BOARD_HEIGHT * CELL_SIZE

    if show_piece_info:
        info_h = INFO_HEIGHT_SIDEBAR
        img_w = LEFT_SIDEBAR + board_w_px + RIGHT_SIDEBAR
        img_h = info_h + board_h_px
        board_x = LEFT_SIDEBAR
        board_y = info_h
    else:
        info_h = INFO_HEIGHT_SIMPLE
        img_w = board_w_px + 2 * PADDING
        img_h = info_h + board_h_px + PADDING
        board_x = PADDING
        board_y = info_h

    img = Image.new("RGB", (img_w, img_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    _draw_board_area(
        draw,
        board_x,
        board_y,
        board,
        board_piece_types,
        current_piece_cells,
        current_piece_type,
        ghost_cells if show_ghost_piece else None,
    )

    if predicted_move_overlays:
        img = _apply_predicted_move_overlays(
            img,
            board_x,
            board_y,
            predicted_move_overlays,
        )
        draw = ImageDraw.Draw(img)

    font = _get_font(14)
    label_font = _get_font(11)

    placement_str = (
        f"{placement_label}: {placement_number}/{total_placements}"
        if total_placements is not None
        else f"{placement_label}: {placement_number}"
    )
    attack_str = (
        f"Attack: {attack}/{total_attack}"
        if total_attack is not None
        else f"Attack: {attack}"
    )

    if show_piece_info:
        # --- Top info bar ---
        parts = [placement_str, attack_str]
        if value_pred is not None:
            parts.append(f"NN: {value_pred:.1f}")
        info_str = "  ".join(parts)
        bbox_info = font.getbbox(info_str)
        info_tw = bbox_info[2] - bbox_info[0]
        info_x = LEFT_SIDEBAR + (board_w_px - info_tw) // 2
        draw.text((info_x, 6), info_str, fill=TEXT_COLOR, font=font)

        # --- Left sidebar: HOLD ---
        left_cx = LEFT_SIDEBAR // 2
        bbox = label_font.getbbox("HOLD")
        tw = bbox[2] - bbox[0]
        draw.text(
            (left_cx - tw // 2, board_y + SIDEBAR_LABEL_Y),
            "HOLD",
            fill=LABEL_COLOR,
            font=label_font,
        )

        if hold_piece_type is not None:
            _draw_mini_piece(draw, hold_piece_type, left_cx, board_y + SIDEBAR_PIECE_Y)

        # Stats below hold piece
        sy = board_y + STATS_Y
        sx = left_cx - 28
        stat_lines = []
        if combo is not None:
            stat_lines.append(f"Combo: {combo}")
        if back_to_back is not None:
            stat_lines.append(f"B2B: {'Y' if back_to_back else 'N'}")
        if can_hold is not None:
            stat_lines.append(f"Hold: {'Y' if can_hold else 'N'}")
        for line in stat_lines:
            draw.text((sx, sy), line, fill=LABEL_COLOR, font=label_font)
            sy += STATS_LINE_H

        # --- Right sidebar: NEXT ---
        right_cx = LEFT_SIDEBAR + board_w_px + RIGHT_SIDEBAR // 2
        bbox = label_font.getbbox("NEXT")
        tw = bbox[2] - bbox[0]
        draw.text(
            (right_cx - tw // 2, board_y + SIDEBAR_LABEL_Y),
            "NEXT",
            fill=LABEL_COLOR,
            font=label_font,
        )

        if queue_piece_types:
            for i, pt in enumerate(queue_piece_types):
                cy = board_y + SIDEBAR_PIECE_Y + i * QUEUE_SLOT_SPACING
                _draw_mini_piece(draw, pt, right_cx, cy)
    else:
        # Simple info line (no sidebars)
        text = f"{placement_str}  {attack_str}"
        if value_pred is not None:
            text += f"  NN: {value_pred:.1f}"
        if info_text:
            text += f"  {info_text}"
        draw.text((PADDING, 10), text, fill=TEXT_COLOR, font=font)

    return img


def _capture_frame(
    env: TetrisEnv,
    placement_number: int,
    attack: int,
    is_terminal: bool = False,
    value_pred: float | None = None,
    placement_label: str = "Placement",
    predicted_move_overlays: list[PredictedMoveOverlay] | None = None,
    show_ghost_piece: bool = True,
    total_attack: int | None = None,
    total_placements: int | None = None,
) -> Image.Image:
    board = np.array(env.get_board())
    board_piece_types = env.get_board_piece_types()
    piece = env.get_current_piece()
    hold_piece = env.get_hold_piece()
    queue_piece_types = env.get_queue(QUEUE_SIZE)
    can_hold = not env.is_hold_used()
    piece_cells = piece.get_cells() if piece else None
    piece_type = piece.piece_type if piece else None
    ghost = env.get_ghost_piece()
    ghost_cells = ghost.get_cells() if ghost else None

    return render_board(
        board=board,
        board_piece_types=board_piece_types,
        current_piece_cells=piece_cells,
        current_piece_type=piece_type,
        ghost_cells=ghost_cells,
        placement_number=placement_number,
        placement_label=placement_label,
        attack=attack,
        total_attack=total_attack,
        total_placements=total_placements,
        can_hold=can_hold,
        combo=env.combo,
        back_to_back=env.back_to_back,
        is_terminal=is_terminal,
        value_pred=value_pred,
        show_piece_info=True,
        hold_piece_type=hold_piece.piece_type if hold_piece else None,
        queue_piece_types=list(queue_piece_types),
        predicted_move_overlays=predicted_move_overlays,
        show_ghost_piece=show_ghost_piece,
    )


def render_replay(replay: ReplayLike) -> list[Image.Image]:
    final_total_attack, final_total_placements = _compute_replay_totals(replay)

    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, replay.seed)
    frames: list[Image.Image] = []
    total_attack = 0

    for move in replay.moves:
        if env.game_over:
            break

        frames.append(
            _capture_frame(
                env,
                env.placement_count,
                total_attack,
                total_attack=final_total_attack,
                total_placements=final_total_placements,
            )
        )

        attack = env.execute_action_index(move.action)
        if attack is None:
            raise ValueError(f"Invalid replay action index: {move.action}")
        total_attack += move.attack

    # Final frame (post-last-action)
    frames.append(
        _capture_frame(
            env,
            env.placement_count,
            total_attack,
            is_terminal=env.game_over,
            total_attack=final_total_attack,
            total_placements=final_total_placements,
        )
    )

    return frames


def _compute_replay_totals(replay: ReplayLike) -> tuple[int, int]:
    """Simulate the replay once to determine the final attack and placement counts."""
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, replay.seed)
    total_attack = 0
    for move in replay.moves:
        if env.game_over:
            break
        attack = env.execute_action_index(move.action)
        if attack is None:
            raise ValueError(f"Invalid replay action index: {move.action}")
        total_attack += move.attack
    return total_attack, env.placement_count


def create_trajectory_gif(
    frames: list[Image.Image],
    output_path: str,
    duration: int = 200,  # ms per frame
) -> None:
    """Save trajectory frames as an animated GIF."""
    if not frames:
        return

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
