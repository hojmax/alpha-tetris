"""Interactive terminal viewer for training replay buffer.

Controls:
    n / Right / Space  - Next frame
    p / Left           - Previous frame
    N / Shift+Right    - Next game
    P / Shift+Left     - Previous game
    a                  - Jump to next attack (value increase)
    A                  - Jump to previous attack
    g                  - Go to specific game
    f                  - Go to specific frame
    q / Escape         - Quit
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from simple_parsing import parse

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    PIECE_NAMES,
    QUEUE_SIZE,
)
from tetris_bot.scripts.inspection.value_predictor import ValuePredictor
from tetris_bot.ml.replay_npz import open_replay_npz

PIECE_CHARS = PIECE_NAMES
PIECE_COLORS = ["cyan", "yellow", "magenta", "green", "red", "blue", "bright_red"]

# Block characters for rendering
EMPTY = "  "
FILLED = "██"


def get_piece_type(one_hot: np.ndarray) -> int | None:
    idx = int(np.argmax(one_hot))
    if len(one_hot) == 8 and idx == 7:
        return None
    return idx


def find_game_boundaries(move_numbers: np.ndarray) -> list[tuple[int, int]]:
    game_starts = np.where(move_numbers == 0)[0]
    games = []
    for i, start in enumerate(game_starts):
        end = game_starts[i + 1] if i + 1 < len(game_starts) else len(move_numbers)
        games.append((int(start), int(end)))
    return games


class BufferViewer:
    def __init__(self, data_path: Path, value_predictor: ValuePredictor | None):
        self.data = open_replay_npz(data_path)
        self.n_examples = len(self.data["boards"])
        self.move_numbers = self.data["move_numbers"]
        self.games = find_game_boundaries(self.move_numbers)
        self.n_games = len(self.games)
        self.value_predictor = value_predictor

        # Current position
        self.game_idx = 0
        self.frame_idx = 0

        self.console = Console()

    @property
    def global_idx(self) -> int:
        start, _ = self.games[self.game_idx]
        return start + self.frame_idx

    @property
    def game_length(self) -> int:
        start, end = self.games[self.game_idx]
        return end - start

    def get_frame_data(self) -> dict:
        i = self.global_idx
        board = self.data["boards"][i].reshape(BOARD_HEIGHT, BOARD_WIDTH)
        board_flat = self.data["boards"][i]
        current_piece_raw = self.data["current_pieces"][i]
        hold_piece_raw = self.data["hold_pieces"][i]
        hold_available_raw = float(self.data["hold_available"][i])
        next_queue_raw = self.data["next_queue"][i]
        placement_count_raw = float(self.data["placement_counts"][i])
        combo_feature = float(self.data["combos"][i])
        back_to_back_raw = float(self.data["back_to_back"][i])
        next_hidden_piece_probs_raw = self.data["next_hidden_piece_probs"][i]
        column_heights_raw = self.data["column_heights"][i]
        max_column_height_raw = float(self.data["max_column_heights"][i])
        row_fill_counts_raw = self.data["row_fill_counts"][i]
        total_blocks_raw = float(self.data["total_blocks"][i])
        bumpiness_raw = float(self.data["bumpiness"][i])
        holes_raw = float(self.data["holes"][i])
        overhang_fields_raw = float(self.data["overhang_fields"][i])
        current_piece = get_piece_type(self.data["current_pieces"][i])
        hold_piece = get_piece_type(self.data["hold_pieces"][i])
        next_queue = [
            get_piece_type(self.data["next_queue"][i][j]) for j in range(QUEUE_SIZE)
        ]
        value_target = float(self.data["value_targets"][i])
        policy = self.data["policy_targets"][i]
        action_mask = self.data["action_masks"][i]
        value_prediction = None
        if self.value_predictor is not None:
            value_prediction = self.value_predictor.predict_value(
                index=i,
                board=board_flat,
                current_piece=current_piece_raw,
                hold_piece=hold_piece_raw,
                hold_available=hold_available_raw,
                next_queue=next_queue_raw,
                placement_count=placement_count_raw,
                combo_feature=combo_feature,
                back_to_back=back_to_back_raw,
                next_hidden_piece_probs=next_hidden_piece_probs_raw,
                column_heights=column_heights_raw,
                max_column_height=max_column_height_raw,
                row_fill_counts=row_fill_counts_raw,
                total_blocks=total_blocks_raw,
                bumpiness=bumpiness_raw,
                holes=holes_raw,
                overhang_fields=overhang_fields_raw,
            )

        return {
            "board": board,
            "current_piece": current_piece,
            "hold_piece": hold_piece,
            "next_queue": next_queue,
            "value_target": value_target,
            "value_prediction": value_prediction,
            "policy": policy,
            "action_mask": action_mask,
            "n_valid_actions": int(np.sum(action_mask)),
        }

    def render_board(self, board: np.ndarray) -> Text:
        text = Text()
        text.append("┌" + "──" * BOARD_WIDTH + "┐\n", style="white")
        for row in range(BOARD_HEIGHT):
            text.append("│", style="white")
            for col in range(BOARD_WIDTH):
                if board[row, col] > 0.5:
                    text.append(FILLED, style="bright_white")
                else:
                    text.append(EMPTY)
            text.append("│\n", style="white")
        text.append("└" + "──" * BOARD_WIDTH + "┘", style="white")
        return text

    def render_info(self, data: dict) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Game", f"{self.game_idx + 1} / {self.n_games}")
        table.add_row("Frame", f"{self.frame_idx + 1} / {self.game_length}")
        table.add_row("Global Index", str(self.global_idx))
        table.add_row("", "")

        # Piece info
        current = (
            PIECE_NAMES[data["current_piece"]]
            if data["current_piece"] is not None
            else "-"
        )
        hold = (
            PIECE_NAMES[data["hold_piece"]] if data["hold_piece"] is not None else "-"
        )
        queue = " ".join(
            PIECE_NAMES[p] if p is not None else "?" for p in data["next_queue"]
        )

        table.add_row("Current", current)
        table.add_row("Hold", hold)
        table.add_row("Queue", queue)
        table.add_row("", "")

        # Value and policy info
        table.add_row("Value Target", f"{data['value_target']:.1f}")
        value_prediction = data["value_prediction"]
        if value_prediction is None:
            table.add_row("Value Pred", "-")
            table.add_row("Value Error", "-")
        else:
            value_error = value_prediction - data["value_target"]
            table.add_row("Value Pred", f"{value_prediction:.2f}")
            table.add_row("Value Error", f"{value_error:+.2f}")
        table.add_row("Valid Actions", str(data["n_valid_actions"]))

        # Top policy actions
        top_actions = np.argsort(data["policy"])[-3:][::-1]
        top_probs = data["policy"][top_actions]
        top_str = ", ".join(f"{a}:{p:.2f}" for a, p in zip(top_actions, top_probs))
        table.add_row("Top Actions", top_str)

        return table

    def render_controls(self) -> Text:
        text = Text()
        text.append("Controls: ", style="bold")
        text.append("n", style="cyan")
        text.append("/")
        text.append("p", style="cyan")
        text.append(" frame  ")
        text.append("N", style="cyan")
        text.append("/")
        text.append("P", style="cyan")
        text.append(" game  ")
        text.append("a", style="cyan")
        text.append("/")
        text.append("A", style="cyan")
        text.append(" attack  ")
        text.append("g", style="cyan")
        text.append(" goto game  ")
        text.append("q", style="cyan")
        text.append(" quit")
        return text

    def render(self) -> Layout:
        data = self.get_frame_data()

        layout = Layout()
        layout.split_column(
            Layout(name="main", ratio=1),
            Layout(name="controls", size=3),
        )

        layout["main"].split_row(
            Layout(
                Panel(self.render_board(data["board"]), title="Board"),
                name="board",
                ratio=1,
            ),
            Layout(Panel(self.render_info(data), title="Info"), name="info", ratio=1),
        )

        layout["controls"].update(Panel(self.render_controls()))

        return layout

    def next_frame(self):
        if self.frame_idx < self.game_length - 1:
            self.frame_idx += 1
        elif self.game_idx < self.n_games - 1:
            self.game_idx += 1
            self.frame_idx = 0

    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
        elif self.game_idx > 0:
            self.game_idx -= 1
            self.frame_idx = self.game_length - 1

    def next_game(self):
        if self.game_idx < self.n_games - 1:
            self.game_idx += 1
            self.frame_idx = 0

    def prev_game(self):
        if self.game_idx > 0:
            self.game_idx -= 1
            self.frame_idx = 0

    def jump_to_next_attack(self):
        current_value = self.get_frame_data()["value_target"]
        start_global = self.global_idx

        # Search forward for value increase
        for _ in range(self.n_examples - start_global - 1):
            self.next_frame()
            new_value = self.get_frame_data()["value_target"]
            if new_value > current_value + 0.5:
                return
            current_value = new_value

    def jump_to_prev_attack(self):
        current_value = self.get_frame_data()["value_target"]
        start_global = self.global_idx

        # Search backward for value change
        for _ in range(start_global):
            self.prev_frame()
            new_value = self.get_frame_data()["value_target"]
            if new_value < current_value - 0.5:
                return
            current_value = new_value

    def goto_game(self, game_num: int):
        if 0 <= game_num < self.n_games:
            self.game_idx = game_num
            self.frame_idx = 0

    def run(self):
        import sys
        import termios
        import tty

        def get_key():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                # Handle escape sequences (arrow keys)
                if ch == "\x1b":
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "C":
                            return "right"
                        elif ch3 == "D":
                            return "left"
                        elif ch3 == "1":
                            # Shift+arrow
                            sys.stdin.read(1)
                            ch5 = sys.stdin.read(1)
                            if ch5 == "C":
                                return "shift_right"
                            elif ch5 == "D":
                                return "shift_left"
                    return "escape"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        with Live(
            self.render(), console=self.console, refresh_per_second=30, screen=True
        ) as live:
            while True:
                key = get_key()

                if key in ("q", "escape", "\x03"):  # q, Esc, Ctrl+C
                    break
                elif key in ("n", "right", " "):
                    self.next_frame()
                elif key in ("p", "left"):
                    self.prev_frame()
                elif key in ("N", "shift_right"):
                    self.next_game()
                elif key in ("P", "shift_left"):
                    self.prev_game()
                elif key == "a":
                    self.jump_to_next_attack()
                elif key == "A":
                    self.jump_to_prev_attack()
                elif key == "g":
                    live.stop()
                    self.console.clear()
                    try:
                        game_str = self.console.input(
                            f"[bold]Go to game (1-{self.n_games}): [/]"
                        )
                        game_num = int(game_str) - 1
                        self.goto_game(game_num)
                    except (ValueError, KeyboardInterrupt):
                        pass
                    live.start()

                live.update(self.render())


@dataclass
class ScriptArgs:
    """Interactive viewer for training replay buffer."""

    data_path: Path  # Path to training_data.npz file
    game: int = 0  # Starting game index (0-based)
    checkpoint_path: Path | None = (
        None  # Checkpoint path (default: <run_dir>/checkpoints/latest.pt)
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


def main(args: ScriptArgs) -> None:
    if not args.data_path.exists():
        print(f"Error: File not found: {args.data_path}")
        return

    if args.checkpoint_path is None or args.config_path is None:
        raise ValueError("checkpoint_path and config_path must be set")
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path

    value_predictor: ValuePredictor | None = None
    if checkpoint_path.exists() and config_path.exists():
        value_predictor = ValuePredictor(checkpoint_path, config_path)
        print(
            f"Loaded model predictions from {checkpoint_path} (config: {config_path})"
        )
    else:
        print(
            "Model predictions disabled: checkpoint/config not found "
            f"({checkpoint_path}, {config_path})"
        )

    viewer = BufferViewer(args.data_path, value_predictor)

    if args.game > 0:
        viewer.goto_game(args.game)

    print(f"Loaded {viewer.n_examples} examples across {viewer.n_games} games")
    print("Starting viewer...")

    viewer.run()


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
