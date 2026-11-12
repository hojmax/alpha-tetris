from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tetris_bot.ml.replay_npz import open_replay_npz
from simple_parsing import parse

EXPECTED_NUM_ROWS = 4


@dataclass
class ScriptArgs:
    data_path: Path  # Path to training_data.npz file
    epsilon: float = 1e-8  # Absolute tolerance used for zero checks


def main(args: ScriptArgs) -> None:
    if args.epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if not args.data_path.exists():
        raise FileNotFoundError(f"file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"expected .npz file, got: {args.data_path}")

    with open_replay_npz(args.data_path, mmap_mode="r") as data:
        if "row_fill_counts" not in data:
            raise KeyError("missing required key: row_fill_counts")
        row_fill_counts = np.asarray(data["row_fill_counts"], dtype=np.float32)

    if row_fill_counts.ndim != 2:
        raise ValueError(
            f"row_fill_counts must be 2D (N, rows), got shape: {row_fill_counts.shape}"
        )

    num_examples, num_rows = row_fill_counts.shape
    if num_rows != EXPECTED_NUM_ROWS:
        raise ValueError(
            f"row_fill_counts second dimension must be {EXPECTED_NUM_ROWS}, got {num_rows}"
        )
    if num_examples == 0:
        raise ValueError("row_fill_counts contains zero examples")

    zero_mask = np.isclose(row_fill_counts, 0.0, atol=args.epsilon)
    zero_counts = zero_mask.sum(axis=0, dtype=np.int64)
    zero_rates = zero_counts.astype(np.float64) / float(num_examples)

    print("ROW_FILL_COUNTS ZERO RATE ANALYSIS")
    print(f"file: {args.data_path}")
    print(f"examples: {num_examples:,}")
    print(f"rows: {num_rows}")
    print(f"epsilon: {args.epsilon:g}")
    print()
    print("index,zero_rate,zero_count,nonzero_count")
    for row_idx in range(num_rows):
        zero_count = int(zero_counts[row_idx])
        nonzero_count = int(num_examples - zero_count)
        zero_rate = float(zero_rates[row_idx])
        print(f"{row_idx},{zero_rate:.6f},{zero_count},{nonzero_count}")

    always_zero_rows = np.where(zero_counts == num_examples)[0]
    print()
    print(
        "overall_zero_rate,"
        f"{float(zero_mask.mean(dtype=np.float64)):.6f},"
        f"{int(zero_mask.sum(dtype=np.int64))},"
        f"{int(zero_mask.size - zero_mask.sum(dtype=np.int64))}"
    )
    print(
        "always_zero_rows,"
        + (
            ",".join(str(int(i)) for i in always_zero_rows)
            if always_zero_rows.size
            else "(none)"
        )
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
