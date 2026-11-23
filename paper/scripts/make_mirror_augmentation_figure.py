"""Render a game state next to its left-right mirror, as used for training augmentation.

Plays a greedy game with the final network, picks a mid-game state whose hold and
queue contain mirror-asymmetric pieces, and draws it and its mirror (board flipped,
S<->Z and J<->L swapped in the current piece, hold and queue) in the GIF style.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from PIL import Image, ImageDraw
from simple_parsing import parse

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, PROJECT_ROOT, QUEUE_SIZE
from tetris_bot.ml.policy_mirroring import MIRROR_PIECE_TYPE_ORDER
from tetris_bot.visualization import (
    BG_COLOR,
    INFO_HEIGHT_SIDEBAR,
    TEXT_COLOR,
    _get_font,
    render_board,
)
from tetris_core.tetris_core import MCTSAgent, MCTSConfig, TetrisEnv

logger = structlog.get_logger()

ASYMMETRIC_PIECES = {3, 4, 5, 6}  # S, Z, J, L


@dataclass
class Args:
    model_path: Path = (
        PROJECT_ROOT
        / "training_runs/new_run_plus_deterministic/final_weights/sync_step_3505140.onnx"
    )
    seed_start: int = 0  # First seed tried; the first suitable state wins
    num_seeds: int = 50
    num_simulations: int = 200
    output: Path = PROJECT_ROOT / "docs/mirror_augmentation.png"


def mirror_env(env: TetrisEnv) -> TetrisEnv:
    def m(piece_type: int) -> int:
        return MIRROR_PIECE_TYPE_ORDER[piece_type]

    mirrored = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 0)
    mirrored.set_board([row[::-1] for row in env.get_board()])
    mirrored.set_board_piece_types(
        [
            [None if p is None else m(p) for p in row[::-1]]
            for row in env.get_board_piece_types()
        ]
    )
    mirrored.set_queue([m(p) for p in env.get_queue(QUEUE_SIZE)])
    hold = env.get_hold_piece()
    if hold is not None:
        mirrored.set_hold_piece_type(m(hold.piece_type))
    mirrored.set_hold_used(env.is_hold_used())
    current = env.get_current_piece()
    assert current is not None
    mirrored.set_current_piece_type(m(current.piece_type))
    return mirrored


def render(env: TetrisEnv, title: str) -> Image.Image:
    piece = env.get_current_piece()
    ghost = env.get_ghost_piece()
    hold = env.get_hold_piece()
    img = render_board(
        board=np.array(env.get_board()),
        board_piece_types=env.get_board_piece_types(),
        current_piece_cells=piece.get_cells() if piece else None,
        current_piece_type=piece.piece_type if piece else None,
        ghost_cells=ghost.get_cells() if ghost else None,
        show_piece_info=True,
        hold_piece_type=hold.piece_type if hold else None,
        queue_piece_types=list(env.get_queue(QUEUE_SIZE)),
    )
    # Replace the placement/attack header with a plain title.
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, INFO_HEIGHT_SIDEBAR - 1)], fill=BG_COLOR)
    font = _get_font(16)
    bbox = font.getbbox(title)
    draw.text(
        ((img.width - (bbox[2] - bbox[0])) // 2, 6), title, fill=TEXT_COLOR, font=font
    )
    return img


def is_good_state(env: TetrisEnv) -> bool:
    # No lines cleared yet, so every piece on the board is whole and the state reads
    # as a plausible 7-bag history (including at least one T on the board).
    if env.lines_cleared != 0 or env.get_hold_piece() is None:
        return False
    on_board = {p for row in env.get_board_piece_types() for p in row if p is not None}
    visible = set(env.get_queue(QUEUE_SIZE))
    visible.add(env.get_hold_piece().piece_type)  # type: ignore[union-attr]
    visible.add(env.get_current_piece().piece_type)  # type: ignore[union-attr]
    filled = int(np.array(env.get_board()).sum())
    return 2 in on_board and len(visible & ASYMMETRIC_PIECES) >= 3 and filled >= 24


def find_state(args: Args) -> TetrisEnv:
    config = MCTSConfig()
    config.num_simulations = args.num_simulations
    config.temperature = 0.0
    config.seed = 0
    config.use_parent_value_for_unvisited_q = True
    agent = MCTSAgent(config)
    assert agent.load_model(str(args.model_path))
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
        while not env.game_over and env.lines_cleared == 0:
            if is_good_state(env):
                logger.info("Found state", seed=seed, placement=env.placement_count)
                return env
            result = agent.select_action(env, False)
            assert result is not None
            env.execute_action_index(result.action)
    raise RuntimeError("No suitable state found; widen the seed range")


def draw_centered(
    draw: ImageDraw.ImageDraw, text: str, cx: int, y: int, size: int
) -> None:
    font = _get_font(size)
    bbox = font.getbbox(text)
    draw.text((cx - (bbox[2] - bbox[0]) // 2, y), text, fill=TEXT_COLOR, font=font)


def main(args: Args) -> None:
    env = find_state(args)
    left = render(env, "Original")
    right = render(mirror_env(env), "Mirrored")
    gap = 120
    title_h = 44
    width = left.width + gap + right.width
    canvas = Image.new("RGB", (width, title_h + left.height), BG_COLOR)
    canvas.paste(left, (0, title_h))
    canvas.paste(right, (left.width + gap, title_h))
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, "Data augmentation: left-right mirroring", width // 2, 12, 20)

    mid = left.width + gap // 2
    arrow_y = title_h + left.height // 2
    half = gap // 2 - 18
    draw.line([(mid - half, arrow_y), (mid + half, arrow_y)], fill=TEXT_COLOR, width=3)
    for tip, sign in ((mid - half, 1), (mid + half, -1)):
        draw.polygon(
            [
                (tip, arrow_y),
                (tip + 10 * sign, arrow_y - 7),
                (tip + 10 * sign, arrow_y + 7),
            ],
            fill=TEXT_COLOR,
        )
    for i, line in enumerate(
        ("1. flip board", "", "2. swap non-", "symmetric", "pieces")
    ):
        draw_centered(draw, line, mid, arrow_y - 106 + 18 * i, 13)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    logger.info("Saved mirror figure", path=str(args.output))


if __name__ == "__main__":
    main(parse(Args))
