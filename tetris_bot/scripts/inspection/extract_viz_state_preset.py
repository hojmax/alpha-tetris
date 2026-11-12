"""Extract one replay-buffer state into an MCTS visualizer preset file.

The output is a Python module containing ``VIZ_STATE_PRESET`` that can be loaded via:

    make viz VIZ_ARGS="--state_preset <path-to-generated-preset.py>"
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pprint

import numpy as np
import structlog
from simple_parsing import parse

from tetris_bot.constants import PIECE_NAMES, PROJECT_ROOT
from tetris_bot.ml.replay_npz import open_replay_npz

logger = structlog.get_logger()


def one_hot_to_piece_name(one_hot: np.ndarray) -> str:
    return PIECE_NAMES[int(np.argmax(one_hot))]


def one_hot_hold_to_piece_name_or_none(one_hot: np.ndarray) -> str | None:
    idx = int(np.argmax(one_hot))
    # Hold one-hot has 8 entries where index 7 means empty hold.
    if len(one_hot) == 8 and idx == 7:
        return None
    return PIECE_NAMES[idx]


def board_to_text_rows(board: np.ndarray) -> list[str]:
    rows: list[str] = []
    for row in board:
        rows.append("".join("1" if int(cell) != 0 else "." for cell in row))
    return rows


def hidden_probs_dict(
    probs: np.ndarray, *, keep_zero: bool = False
) -> dict[str, float]:
    result: dict[str, float] = {}
    for piece_idx, piece_name in enumerate(PIECE_NAMES):
        value = float(probs[piece_idx])
        if keep_zero or value > 0.0:
            result[piece_name] = value
    return result


@dataclass
class ScriptArgs:
    data_path: Path = PROJECT_ROOT / "training_data (1).npz"
    wandb_game_number: int = 721
    move_number: int = 32
    output_path: Path = (
        PROJECT_ROOT
        / "tetris_bot"
        / "scripts"
        / "inspection"
        / "viz_state_presets"
        / "training_data1_game721_move32.json"
    )
    seed_for_viz: int = 42


def main(args: ScriptArgs) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ not found: {args.data_path}")

    with open_replay_npz(args.data_path) as data:
        if "game_numbers" not in data:
            raise ValueError("NPZ missing required key: game_numbers")
        if "move_numbers" not in data:
            raise ValueError("NPZ missing required key: move_numbers")

        game_numbers = data["game_numbers"]
        game_indices = np.where(game_numbers == args.wandb_game_number)[0]
        if len(game_indices) == 0:
            raise ValueError(
                f"WandB game number not found in NPZ: {args.wandb_game_number}"
            )
        game_start = int(game_indices[0])
        game_end = int(game_indices[-1]) + 1
        game_length = game_end - game_start

        if args.move_number < 0 or args.move_number >= game_length:
            raise ValueError(
                "Requested move is out of range for game: "
                f"move_number={args.move_number}, valid=[0, {game_length})"
            )

        global_idx = game_start + args.move_number
        board_rows = board_to_text_rows(data["boards"][global_idx])
        current_piece = one_hot_to_piece_name(data["current_pieces"][global_idx])
        hold_piece = one_hot_hold_to_piece_name_or_none(data["hold_pieces"][global_idx])
        hold_available = bool(data["hold_available"][global_idx])
        queue_names = [
            one_hot_to_piece_name(data["next_queue"][global_idx][slot_idx])
            for slot_idx in range(data["next_queue"].shape[1])
        ]
        hidden_piece_probs = hidden_probs_dict(
            data["next_hidden_piece_probs"][global_idx], keep_zero=False
        )

        realized_hidden_tail_from_replay: list[str] = []
        for idx in range(global_idx, game_end - 1):
            next_tail_one_hot = data["next_queue"][idx + 1][-1]
            realized_hidden_tail_from_replay.append(
                one_hot_to_piece_name(next_tail_one_hot)
            )

        future_move_context: list[dict[str, object]] = []
        for idx in range(global_idx, game_end):
            move_num = int(data["move_numbers"][idx])
            current_name = one_hot_to_piece_name(data["current_pieces"][idx])
            queue = [
                one_hot_to_piece_name(data["next_queue"][idx][slot_idx])
                for slot_idx in range(data["next_queue"].shape[1])
            ]
            future_move_context.append(
                {
                    "move_number": move_num,
                    "current_piece": current_name,
                    "queue": queue,
                    "next_hidden_piece_probs": hidden_probs_dict(
                        data["next_hidden_piece_probs"][idx], keep_zero=False
                    ),
                }
            )

        preset: dict[str, object] = {
            "seed": int(args.seed_for_viz),
            "move_number": int(args.move_number),
            "current_piece": current_piece,
            "hold_piece": hold_piece,
            "hold_used": not hold_available,
            "queue": queue_names,
            "board": board_rows,
            "metadata": {
                "source_npz": str(args.data_path),
                "wandb_game_number": int(args.wandb_game_number),
                "game_start_index": game_start,
                "game_end_index_exclusive": game_end,
                "global_index": global_idx,
                "next_hidden_piece_probs_at_state": hidden_piece_probs,
                "realized_hidden_tail_from_replay": realized_hidden_tail_from_replay,
                "future_move_context": future_move_context,
            },
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = args.output_path.suffix.lower()
    if output_suffix == ".json":
        args.output_path.write_text(json.dumps(preset, indent=2))
    elif output_suffix == ".py":
        preset_repr = pprint.pformat(preset, width=100, sort_dicts=False)
        file_contents = (
            "from __future__ import annotations\n\n"
            "# Auto-generated by extract_viz_state_preset.py\n"
            '# Load with: make viz VIZ_ARGS="--state_preset <this-file>"\n\n'
            f"VIZ_STATE_PRESET = {preset_repr}\n"
        )
        args.output_path.write_text(file_contents)
    else:
        raise ValueError(
            "Unsupported output format for preset. "
            f"Use .json or .py, got: {args.output_path}"
        )

    logger.info(
        "Wrote visualizer state preset",
        output_path=str(args.output_path),
        source_npz=str(args.data_path),
        wandb_game_number=args.wandb_game_number,
        move_number=args.move_number,
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
