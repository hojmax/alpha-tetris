"""Inspect training data by rendering games as GIFs."""

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import structlog
import yaml
from rich.console import Console
from simple_parsing import parse

from tetris_core import TetrisEnv

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    DEFAULT_GIF_FRAME_DURATION_MS,
    LATEST_CHECKPOINT_FILENAME,
    PROJECT_ROOT,
    QUEUE_SIZE,
)
from tetris_bot.ml.network import COMBO_NORMALIZATION_MAX
from tetris_bot.scripts.inspection.value_predictor import try_load_value_predictor
from tetris_bot.visualization import compute_spawn_and_ghost, render_board
from tetris_bot.ml.replay_npz import ReplayArchive, open_replay_npz

SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
logger = structlog.get_logger()
console = Console()


@dataclass(frozen=True)
class GameSlice:
    local_index: int
    start: int
    end: int
    total_attack: float
    wandb_game_number: int | None


def get_piece_type(one_hot: np.ndarray) -> int | None:
    """Convert one-hot encoded piece to type index."""
    idx = np.argmax(one_hot)
    # For hold_pieces, index 7 means empty
    if len(one_hot) == 8 and idx == 7:
        return None
    return int(idx)


def find_game_boundaries(move_numbers: np.ndarray) -> list[tuple[int, int]]:
    game_starts = np.where(move_numbers == 0)[0]

    games = []
    for i, start in enumerate(game_starts):
        end = game_starts[i + 1] if i + 1 < len(game_starts) else len(move_numbers)
        games.append((start, end))
    return games


def build_game_slices(data: ReplayArchive) -> list[GameSlice]:
    """Build game slices with metadata aligned to WandB game numbers when available."""
    n_examples = len(data["boards"])
    if n_examples == 0:
        return []

    if "game_numbers" in data and "game_total_attacks" in data:
        game_numbers = data["game_numbers"].astype(np.int64)
        game_total_attacks = data["game_total_attacks"].astype(np.float32)

        boundaries = np.where(np.diff(game_numbers) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_examples]))

        games: list[GameSlice] = []
        for local_index, (start_raw, end_raw) in enumerate(zip(starts, ends)):
            start = int(start_raw)
            end = int(end_raw)
            game_number = int(game_numbers[start])
            total_attack = float(game_total_attacks[start])
            games.append(
                GameSlice(
                    local_index=local_index,
                    start=start,
                    end=end,
                    total_attack=total_attack,
                    wandb_game_number=game_number,
                )
            )
        return games

    logger.warning(
        "NPZ is missing game_numbers/game_total_attacks; WandB index alignment is unavailable",
        expected_keys=["game_numbers", "game_total_attacks"],
    )
    move_numbers = data["move_numbers"]
    value_targets = data["value_targets"]
    boundaries = find_game_boundaries(move_numbers)
    games = []
    for local_index, (start_raw, end_raw) in enumerate(boundaries):
        start = int(start_raw)
        end = int(end_raw)
        games.append(
            GameSlice(
                local_index=local_index,
                start=start,
                end=end,
                total_attack=float(value_targets[start]),
                wandb_game_number=None,
            )
        )
    return games


def find_game_by_wandb_number(
    games: list[GameSlice], wandb_game_number: int
) -> GameSlice:
    for game in games:
        if game.wandb_game_number == wandb_game_number:
            return game
    raise ValueError(f"WandB game number not found in snapshot: {wandb_game_number}")


def get_game_by_local_index(games: list[GameSlice], requested_index: int) -> GameSlice:
    n_games = len(games)
    game_idx = requested_index if requested_index >= 0 else n_games + requested_index
    if game_idx < 0 or game_idx >= n_games:
        raise ValueError(
            f"Game index out of range: requested={requested_index}, valid=[0, {n_games})"
        )
    return games[game_idx]


def estimate_total_attack_from_value_targets(game: GameSlice) -> int:
    return int(round(game.total_attack))


@dataclass
class ScriptArgs:
    """Inspect training data by rendering games as GIFs."""

    data_path: Path = (  # Path to training_data.npz file
        PROJECT_ROOT / "training_runs" / "v41" / "training_data.npz"
    )
    checkpoint_path: (  # Checkpoint path (default: <run_dir>/checkpoints/latest.pt)
        Path | None
    ) = None
    game_index: int = 0  # Which game to render (-1 for last)
    save_path: Path | None = (
        None  # Output path (default: script outputs/game_{index}.gif)
    )
    frame_duration: int = DEFAULT_GIF_FRAME_DURATION_MS  # Milliseconds per frame
    highest_attack_only: bool = (
        False  # If True, ignore game_index and select highest-attack game
    )
    wandb_game_number: int = (
        -1  # Select by WandB game_number (1-indexed); ignored when -1
    )
    show_value_pred: bool = (
        True  # Overlay model Vpred on each frame (requires checkpoint + config)
    )
    print_buffer_vectors: bool = True  # Print replay vectors/matrices for selected game (policy_targets is truncated)
    policy_targets_edge_items: int = (
        6  # Number of edge items per axis when truncating policy_targets display
    )
    config_path: Path | None = None  # Config path (default: <run_dir>/config.yaml)

    def __post_init__(self) -> None:
        run_dir = self.data_path.parent
        if self.checkpoint_path is None:
            self.checkpoint_path = (
                run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
            )
        if self.config_path is None:
            self.config_path = run_dir / CONFIG_FILENAME
        if self.highest_attack_only and self.wandb_game_number > 0:
            raise ValueError(
                "Cannot set both highest_attack_only=True and wandb_game_number > 0"
            )


def format_array(arr: np.ndarray) -> str:
    """Render full array contents without NumPy truncation."""
    return np.array2string(
        arr,
        separator=", ",
        threshold=sys.maxsize,
        max_line_width=160,
    )


def format_policy_targets_preview(arr: np.ndarray, edge_items: int) -> str:
    if edge_items <= 0:
        raise ValueError("policy_targets_edge_items must be > 0")
    threshold = edge_items * 4
    return np.array2string(
        arr,
        separator=", ",
        threshold=threshold,
        edgeitems=edge_items,
        max_line_width=160,
    )


def print_game_buffer_vectors(
    data: ReplayArchive, start: int, end: int, policy_targets_edge_items: int
) -> None:
    """Print full game slice vectors and matrices from the replay buffer."""
    game_slice = slice(start, end)
    dump_keys = [
        "boards",
        "current_pieces",
        "hold_pieces",
        "hold_available",
        "next_queue",
        "value_targets",
        "policy_targets",
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
        "overhang_fields",
    ]

    console.rule("[bold]Replay Buffer Slice[/bold]")
    for key in dump_keys:
        if key not in data:
            logger.warning("Buffer key missing, skipping dump", key=key)
            continue
        console.print(f"[bold]{key}[/bold]")
        key_data = data[key][game_slice]
        if key == "policy_targets":
            console.print(
                format_policy_targets_preview(key_data, policy_targets_edge_items)
            )
        else:
            console.print(format_array(key_data))
        console.print()


def main(args: ScriptArgs) -> None:
    if args.checkpoint_path is None:
        raise ValueError("checkpoint_path cannot be None")
    if args.config_path is None:
        raise ValueError("config_path cannot be None")
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path

    # Validate file
    if not args.data_path.exists():
        logger.error("File not found", path=str(args.data_path))
        return
    if args.data_path.suffix != ".npz":
        logger.error(
            "Expected .npz file", path=str(args.data_path), suffix=args.data_path.suffix
        )
        return

    with config_path.open() as f:
        config_yaml = yaml.safe_load(f)
    max_placements = int(config_yaml["self_play"]["max_placements"])

    if args.show_value_pred:
        value_predictor = try_load_value_predictor(checkpoint_path, config_path)
        if value_predictor is not None:
            logger.info(
                "Loaded model predictions",
                checkpoint_path=str(checkpoint_path),
                config_path=str(config_path),
            )
    else:
        value_predictor = None

    # Load data
    with open_replay_npz(args.data_path) as data:
        n_examples = len(data["boards"])
        games = build_game_slices(data)
        n_games = len(games)
        if n_games == 0:
            logger.error("No games found in dataset", path=str(args.data_path))
            return

        if args.highest_attack_only:
            game = max(
                games,
                key=lambda g: estimate_total_attack_from_value_targets(
                    g,
                ),
            )
        elif args.wandb_game_number > 0:
            has_wandb_metadata = any(
                game_slice.wandb_game_number is not None for game_slice in games
            )
            if not has_wandb_metadata:
                approximated_local_index = args.wandb_game_number - 1
                logger.warning(
                    "Approximating WandB game number via local index (wandb_game_number - 1)",
                    requested_wandb_game_number=args.wandb_game_number,
                    approximated_local_index=approximated_local_index,
                )
                try:
                    game = get_game_by_local_index(games, approximated_local_index)
                except ValueError:
                    logger.error(
                        "Approximated local index is out of range",
                        requested_wandb_game_number=args.wandb_game_number,
                        approximated_local_index=approximated_local_index,
                        min_index=0,
                        max_exclusive=n_games,
                    )
                    return
            else:
                try:
                    game = find_game_by_wandb_number(games, args.wandb_game_number)
                except ValueError:
                    logger.error(
                        "WandB game number not found",
                        requested_wandb_game_number=args.wandb_game_number,
                        min_available=min(
                            [
                                g.wandb_game_number
                                for g in games
                                if g.wandb_game_number is not None
                            ],
                            default=None,
                        ),
                        max_available=max(
                            [
                                g.wandb_game_number
                                for g in games
                                if g.wandb_game_number is not None
                            ],
                            default=None,
                        ),
                    )
                    return
        else:
            try:
                game = get_game_by_local_index(games, args.game_index)
            except ValueError:
                logger.error(
                    "Game index out of range",
                    requested_game_index=args.game_index,
                    min_index=0,
                    max_exclusive=n_games,
                )
                return

        start = game.start
        end = game.end
        game_length = end - start
        game_total_attack = estimate_total_attack_from_value_targets(
            game,
        )

        logger.info("Loaded dataset", num_games=n_games, total_examples=n_examples)
        logger.info(
            "Rendering game",
            game_index=game.local_index,
            wandb_game_number=game.wandb_game_number,
            game_total_attack=game_total_attack,
            example_start=start,
            example_end=end - 1,
            num_frames=game_length,
        )

        if args.print_buffer_vectors:
            print_game_buffer_vectors(data, start, end, args.policy_targets_edge_items)

        # placement_counts are stored normalized as placements / max_placements;
        # denormalize for display. The buffer stores pre-move placement counts,
        # so the true total includes the final move that produces the post-move state.
        game_placement_counts = (
            data["placement_counts"][start:end].astype(np.float32) * max_placements
        )
        final_placement_number = int(round(float(game_placement_counts[-1]))) + 1
        frames = []
        for i in range(start, end):
            board = data["boards"][i]
            current_piece = get_piece_type(data["current_pieces"][i])
            hold_piece = get_piece_type(data["hold_pieces"][i])
            next_queue = [
                get_piece_type(data["next_queue"][i][j]) for j in range(QUEUE_SIZE)
            ]
            can_hold = bool(data["hold_available"][i])
            combo = round(float(data["combos"][i]) * COMBO_NORMALIZATION_MAX)
            back_to_back = bool(data["back_to_back"][i])
            placement_number = int(round(float(game_placement_counts[i - start])))
            value_target = float(data["value_targets"][i])
            # Cumulative attack before this step (starts at 0, increases)
            cumulative_attack = int(round(game_total_attack - value_target))
            value_pred = None
            if value_predictor is not None:
                value_pred = value_predictor.predict_value(
                    index=i,
                    board=data["boards"][i],
                    current_piece=data["current_pieces"][i],
                    hold_piece=data["hold_pieces"][i],
                    hold_available=float(data["hold_available"][i]),
                    next_queue=data["next_queue"][i],
                    placement_count=float(data["placement_counts"][i]),
                    combo_feature=float(data["combos"][i]),
                    back_to_back=float(data["back_to_back"][i]),
                    next_hidden_piece_probs=data["next_hidden_piece_probs"][i],
                    column_heights=data["column_heights"][i],
                    max_column_height=float(data["max_column_heights"][i]),
                    row_fill_counts=data["row_fill_counts"][i],
                    total_blocks=float(data["total_blocks"][i]),
                    bumpiness=float(data["bumpiness"][i]),
                    holes=float(data["holes"][i]),
                    overhang_fields=float(data["overhang_fields"][i]),
                )

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
                total_attack=game_total_attack,
                total_placements=final_placement_number,
                value_pred=value_pred,
                can_hold=can_hold,
                combo=combo,
                back_to_back=back_to_back,
                show_piece_info=True,
                hold_piece_type=hold_piece,
                queue_piece_types=[p for p in next_queue if p is not None],
            )
            frames.append(frame)

        # Simulate the final move using the Rust engine so the GIF ends on the
        # post-move state (e.g. 50/50). The chosen action is the argmax of the
        # MCTS visit distribution stored in policy_targets.
        last_index = end - 1
        last_action = int(np.argmax(data["policy_targets"][last_index]))
        env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
        env.set_board(data["boards"][last_index].astype(np.uint8).tolist())
        last_hold_piece = get_piece_type(data["hold_pieces"][last_index])
        if last_hold_piece is not None:
            env.set_hold_piece_type(last_hold_piece)
        env.set_hold_used(not bool(data["hold_available"][last_index]))
        env.set_queue(
            [
                int(np.argmax(data["next_queue"][last_index][j]))
                for j in range(QUEUE_SIZE)
            ]
        )
        last_current_piece = get_piece_type(data["current_pieces"][last_index])
        if last_current_piece is not None:
            env.set_current_piece_type(last_current_piece)
        env.execute_action_index(last_action)

        post_board = np.array(env.get_board(), dtype=np.uint8)
        post_current = env.get_current_piece()
        post_current_type = (
            post_current.piece_type if post_current is not None else None
        )
        post_hold = env.get_hold_piece()
        post_hold_type = post_hold.piece_type if post_hold is not None else None
        post_queue = list(env.get_queue(QUEUE_SIZE))
        post_piece_cells = None
        post_ghost_cells = None
        if post_current_type is not None and not env.game_over:
            post_piece_cells, post_ghost_cells = compute_spawn_and_ghost(
                post_current_type, post_board
            )

        post_frame = render_board(
            board=post_board,
            current_piece_cells=post_piece_cells,
            current_piece_type=post_current_type,
            ghost_cells=post_ghost_cells,
            placement_number=final_placement_number,
            attack=game_total_attack,
            total_attack=game_total_attack,
            total_placements=final_placement_number,
            value_pred=None,
            can_hold=not env.is_hold_used(),
            combo=int(env.combo),
            back_to_back=bool(env.back_to_back),
            show_piece_info=True,
            hold_piece_type=post_hold_type,
            queue_piece_types=post_queue,
        )
        frames.append(post_frame)

        # Per-frame durations: linger 4x on the final post-move frame.
        durations = [args.frame_duration] * (len(frames) - 1) + [
            args.frame_duration * 4
        ]

        # Determine save path
        if args.save_path is None:
            suffix = "" if args.show_value_pred else "_no_vpred"
            if game.wandb_game_number is not None:
                save_path = (
                    OUTPUTS_DIR / f"game_wandb_{game.wandb_game_number}{suffix}.gif"
                )
            else:
                save_path = OUTPUTS_DIR / f"game_{game.local_index}{suffix}.gif"
        else:
            save_path = args.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as GIF
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )

        logger.info("Saved gif", num_frames=len(frames), path=str(save_path))


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
