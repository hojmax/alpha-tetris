from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tetris_bot.ml.replay_npz import ReplayArchive, open_replay_npz
from simple_parsing import parse

EXPECTED_NUM_COLUMNS = 10
EXPECTED_NUM_ROWS = 4
EXPECTED_NUM_PIECES = 7
PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
EPSILON = 1e-6
REQUIRED_KEYS = (
    "boards",
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
    "value_targets",
    "game_numbers",
    "game_total_attacks",
)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_histogram(values: np.ndarray, bins: int = 10) -> None:
    counts, edges = np.histogram(values, bins=bins)
    max_count = int(np.max(counts)) if counts.size else 0
    bar_width = 40
    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {left:10.4f} - {right:10.4f}: {bar} ({int(count)})")


def print_percentiles(values: np.ndarray) -> None:
    vals = np.percentile(values, PERCENTILES)
    for percentile, value in zip(PERCENTILES, vals):
        print(f"  p{percentile:3d}: {value:10.4f}")


def print_distribution(name: str, values: np.ndarray, bins: int = 10) -> None:
    flattened = values.astype(np.float64, copy=False).reshape(-1)
    print_section(name)
    print(f"  shape: {values.shape}")
    print(f"  mean:  {flattened.mean():10.4f}")
    print(f"  std:   {flattened.std():10.4f}")
    print(f"  min:   {flattened.min():10.4f}")
    print(f"  max:   {flattened.max():10.4f}")
    print_percentiles(flattened)
    print_histogram(flattened, bins=bins)


def print_normalized_range_check(name: str, values: np.ndarray) -> None:
    flattened = values.astype(np.float64, copy=False).reshape(-1)
    below_zero = int(np.count_nonzero(flattened < -EPSILON))
    above_one = int(np.count_nonzero(flattened > 1.0 + EPSILON))
    total = flattened.size
    print(
        f"  {name:<24} out_of_[0,1]: {below_zero + above_one:>8,} / {total:,} "
        f"(below_zero={below_zero:,}, above_one={above_one:,})"
    )


def validate_array_lengths(data: ReplayArchive, n_examples: int) -> None:
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"missing required key: {key}")
        if len(data[key]) != n_examples:
            raise ValueError(
                f"{key} length does not match boards length: {len(data[key])} != {n_examples}"
            )


def validate_shapes(data: ReplayArchive) -> None:
    if data["next_hidden_piece_probs"].shape[1] != EXPECTED_NUM_PIECES:
        raise ValueError(
            "next_hidden_piece_probs second dimension must be 7, got "
            f"{data['next_hidden_piece_probs'].shape[1]}"
        )
    if data["column_heights"].shape[1] != EXPECTED_NUM_COLUMNS:
        raise ValueError(
            "column_heights second dimension must be 10, got "
            f"{data['column_heights'].shape[1]}"
        )
    if data["row_fill_counts"].shape[1] != EXPECTED_NUM_ROWS:
        raise ValueError(
            "row_fill_counts second dimension must be 4, got "
            f"{data['row_fill_counts'].shape[1]}"
        )


def build_game_boundaries(game_numbers: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.where(np.diff(game_numbers) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(game_numbers)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def print_game_stats(
    game_numbers: np.ndarray, game_total_attacks: np.ndarray, value_targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    games = build_game_boundaries(game_numbers)
    if not games:
        raise ValueError("no games found in dataset")

    game_lengths = np.array([end - start for start, end in games], dtype=np.int32)
    game_attacks = np.array(
        [float(game_total_attacks[start]) for start, _ in games], dtype=np.float32
    )
    first_state_values = np.array(
        [float(value_targets[start]) for start, _ in games], dtype=np.float32
    )
    mismatch_count = int(
        np.count_nonzero(np.abs(game_attacks - first_state_values) > 1e-4)
    )
    unique_game_numbers = np.unique(game_numbers)
    game_number_diffs = np.diff(unique_game_numbers)
    has_gaps = bool(np.any(game_number_diffs != 1))

    print_section("Games")
    print(f"  total_games:             {len(games):,}")
    print(f"  first_game_number:       {int(unique_game_numbers[0]):,}")
    print(f"  last_game_number:        {int(unique_game_numbers[-1]):,}")
    print(f"  contiguous_game_numbers: {not has_gaps}")
    print(f"  attack/value_mismatches: {mismatch_count:,}")
    print_distribution("Game Length (examples)", game_lengths, bins=10)
    print_distribution("Game Total Attack", game_attacks, bins=10)
    return game_lengths, game_attacks


@dataclass
class ScriptArgs:
    data_path: Path  # Path to training_data.npz file


def main(args: ScriptArgs) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(f"file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"expected .npz file, got: {args.data_path}")

    with open_replay_npz(args.data_path) as data:
        n_examples = len(data["boards"])
        validate_array_lengths(data, n_examples)
        validate_shapes(data)

        move_numbers = data["move_numbers"]
        placement_counts = data["placement_counts"]
        combos = data["combos"]
        back_to_back = data["back_to_back"]
        next_hidden_piece_probs = data["next_hidden_piece_probs"]
        column_heights = data["column_heights"]
        max_column_heights = data["max_column_heights"]
        row_fill_counts = data["row_fill_counts"]
        total_blocks = data["total_blocks"]
        bumpiness = data["bumpiness"]
        holes = data["holes"]
        overhang_fields = data["overhang_fields"]
        value_targets = data["value_targets"]
        game_numbers = data["game_numbers"].astype(np.int64)
        game_total_attacks = data["game_total_attacks"].astype(np.float32)

        print("=" * 56)
        print("TRAINING DATA ANALYSIS")
        print("=" * 56)
        print(f"\nfile: {args.data_path}")
        print(f"examples: {n_examples:,}")
        print(f"fields_checked: {len(REQUIRED_KEYS)}")

        game_lengths, game_attacks = print_game_stats(
            game_numbers=game_numbers,
            game_total_attacks=game_total_attacks,
            value_targets=value_targets,
        )

        print_section("Scalar Features")
        print_distribution("Move Number (raw)", move_numbers, bins=12)
        print_distribution("Placement Count (normalized)", placement_counts, bins=12)
        print_distribution("Combo (normalized)", combos, bins=12)
        print_distribution("Back-to-Back Flag", back_to_back.astype(np.float32), bins=2)
        print_distribution("Value Targets", value_targets, bins=15)
        print_distribution("Total Blocks (normalized)", total_blocks, bins=12)
        print_distribution("Bumpiness (normalized)", bumpiness, bins=12)
        print_distribution("Holes (normalized)", holes, bins=12)
        print_distribution("Overhang Fields (normalized)", overhang_fields, bins=12)

        print_section("Vector Features")
        print_distribution(
            "Next Hidden Piece Probabilities", next_hidden_piece_probs, bins=12
        )
        print_distribution("Column Heights (normalized)", column_heights, bins=12)
        print_distribution("Row Fill Counts (normalized)", row_fill_counts, bins=12)
        print_distribution(
            "Max Column Height (normalized)", max_column_heights, bins=12
        )

        print_section("Normalized Range Checks")
        print_normalized_range_check("next_hidden_piece_probs", next_hidden_piece_probs)
        print_normalized_range_check("column_heights", column_heights)
        print_normalized_range_check("row_fill_counts", row_fill_counts)
        print_normalized_range_check("max_column_heights", max_column_heights)
        print_normalized_range_check("total_blocks", total_blocks)
        print_normalized_range_check("bumpiness", bumpiness)
        print_normalized_range_check("holes", holes)
        print_normalized_range_check("overhang_fields", overhang_fields)

        print("\n" + "=" * 56)
        print("SUMMARY")
        print("=" * 56)
        print(f"  games:                 {len(game_lengths):>10,}")
        print(f"  examples:              {n_examples:>10,}")
        print(f"  avg_game_len:          {float(np.mean(game_lengths)):>10.2f}")
        print(f"  median_game_attack:    {float(np.median(game_attacks)):>10.2f}")
        print(f"  mean_game_attack:      {float(np.mean(game_attacks)):>10.2f}")
        print(f"  max_game_attack:       {float(np.max(game_attacks)):>10.2f}")


if __name__ == "__main__":
    main(parse(ScriptArgs))
