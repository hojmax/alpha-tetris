"""Render replay buffer entries as an animated GIF."""

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_bot.constants import (
    DEFAULT_GIF_FRAME_DURATION_MS,
    PROJECT_ROOT,
    QUEUE_SIZE,
)
from tetris_bot.ml.network import COMBO_NORMALIZATION_MAX
from tetris_bot.visualization import compute_spawn_and_ghost, render_board
from tetris_bot.scripts.inspection.inspect_training_data import (
    build_game_slices,
    get_piece_type,
)
from tetris_bot.ml.replay_npz import open_replay_npz

logger = structlog.get_logger()

SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"


@dataclass
class ScriptArgs:
    """Render replay buffer entries as an animated GIF."""

    data_path: Path = PROJECT_ROOT / "training_runs" / "v41" / "training_data.npz"
    num_frames: int = 200  # Number of frames to render
    start_index: int = 0  # First buffer index to render
    save_path: Path = OUTPUTS_DIR / "buffer_frames.gif"  # Output GIF path
    frame_duration: int = DEFAULT_GIF_FRAME_DURATION_MS  # Milliseconds per frame


def main(args: ScriptArgs) -> None:
    if not args.data_path.exists():
        logger.error("File not found", path=str(args.data_path))
        return

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    with open_replay_npz(args.data_path) as data:
        n_examples = len(data["boards"])
        games = build_game_slices(data)

        # Build index -> (game_total_attack, game_start) for cumulative attack calc
        example_to_game: dict[int, tuple[float, int]] = {}
        for game in games:
            for i in range(game.start, game.end):
                example_to_game[i] = (game.total_attack, game.start)

        end_index = min(args.start_index + args.num_frames, n_examples)
        actual_count = end_index - args.start_index
        if actual_count <= 0:
            logger.error(
                "No frames to render",
                start_index=args.start_index,
                n_examples=n_examples,
            )
            return

        logger.info(
            "Rendering frames",
            start=args.start_index,
            end=end_index,
            count=actual_count,
        )

        frames = []
        for i in range(args.start_index, end_index):
            board = data["boards"][i]
            current_piece = get_piece_type(data["current_pieces"][i])
            hold_piece = get_piece_type(data["hold_pieces"][i])
            next_queue = [
                get_piece_type(data["next_queue"][i][j]) for j in range(QUEUE_SIZE)
            ]
            can_hold = bool(data["hold_available"][i])
            combo = round(float(data["combos"][i]) * COMBO_NORMALIZATION_MAX)
            back_to_back = bool(data["back_to_back"][i])
            value_target = float(data["value_targets"][i])

            game_total_attack, _game_start = example_to_game[i]
            cumulative_attack = int(round(game_total_attack - value_target))
            placement_number = int(data["move_numbers"][i])

            piece_cells = None
            ghost_cells = None
            if current_piece is not None:
                piece_cells, ghost_cells = compute_spawn_and_ghost(current_piece, board)

            frame = render_board(
                board=board,
                current_piece_cells=piece_cells,
                current_piece_type=current_piece,
                ghost_cells=ghost_cells,
                placement_number=placement_number,
                attack=cumulative_attack,
                can_hold=can_hold,
                combo=combo,
                back_to_back=back_to_back,
                show_piece_info=True,
                hold_piece_type=hold_piece,
                queue_piece_types=[p for p in next_queue if p is not None],
                info_text=f"buf[{i}]",
            )
            frames.append(frame)

        frames[0].save(
            args.save_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.frame_duration,
            loop=0,
        )

        logger.info("Saved gif", num_frames=len(frames), path=str(args.save_path))


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
