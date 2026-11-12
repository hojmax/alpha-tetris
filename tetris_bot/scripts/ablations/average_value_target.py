from pathlib import Path

import numpy as np
from tetris_bot.ml.replay_npz import open_replay_npz
from pydantic.dataclasses import dataclass
from simple_parsing import field as sp_field
from simple_parsing import parse


@dataclass
class ScriptArgs:
    replay_buffer_path: Path = sp_field(
        positional=True
    )  # Path to replay buffer .npz file


def find_game_boundaries(move_numbers: np.ndarray) -> list[tuple[int, int]]:
    game_starts = np.where(move_numbers == 0)[0]
    if len(game_starts) == 0:
        raise ValueError("Replay buffer has no game starts (move_numbers never resets)")

    games: list[tuple[int, int]] = []
    for i, start in enumerate(game_starts):
        end = game_starts[i + 1] if i + 1 < len(game_starts) else len(move_numbers)
        games.append((int(start), int(end)))
    return games


def main(args: ScriptArgs) -> None:
    if not args.replay_buffer_path.exists():
        raise FileNotFoundError(
            f"Replay buffer file not found: {args.replay_buffer_path}"
        )
    if args.replay_buffer_path.suffix != ".npz":
        raise ValueError(
            f"Expected a .npz replay buffer file, got: {args.replay_buffer_path}"
        )

    with open_replay_npz(args.replay_buffer_path) as data:
        if "value_targets" not in data:
            raise KeyError("Replay buffer is missing required key 'value_targets'")
        if "move_numbers" not in data:
            raise KeyError("Replay buffer is missing required key 'move_numbers'")

        value_targets = data["value_targets"]
        move_numbers = data["move_numbers"]
        if len(value_targets) == 0:
            raise ValueError("Replay buffer has no examples (value_targets is empty)")
        if len(move_numbers) != len(value_targets):
            raise ValueError(
                "Replay buffer has mismatched lengths for value_targets and move_numbers"
            )

        average_value_target = float(np.mean(value_targets))
        std_value_target = float(np.std(value_targets))
        median_value_target = float(np.median(value_targets))
        min_value_target = float(np.min(value_targets))
        max_value_target = float(np.max(value_targets))
        p90_value_target = float(np.percentile(value_targets, 90))
        p99_value_target = float(np.percentile(value_targets, 99))
        nonzero_examples = int(np.count_nonzero(value_targets))

        games = find_game_boundaries(move_numbers)
        game_start_values = np.array([value_targets[start] for start, _ in games])
        nonzero_game_numbers = [
            i + 1 for i, value in enumerate(game_start_values) if value != 0
        ]

    print(f"file: {args.replay_buffer_path}")
    print(f"examples: {len(value_targets):,}")
    print(f"games: {len(games):,}")
    print(f"average_value_target: {average_value_target:.6f}")
    print(f"median_value_target: {median_value_target:.6f}")
    print(f"std_value_target: {std_value_target:.6f}")
    print(f"min_value_target: {min_value_target:.6f}")
    print(f"max_value_target: {max_value_target:.6f}")
    print(f"p90_value_target: {p90_value_target:.6f}")
    print(f"p99_value_target: {p99_value_target:.6f}")
    print(f"nonzero_examples: {nonzero_examples:,}")
    print(f"nonzero_games: {len(nonzero_game_numbers):,}")
    print(f"nonzero_game_numbers_1_indexed: {nonzero_game_numbers}")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
