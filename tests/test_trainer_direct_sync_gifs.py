from __future__ import annotations

from collections import deque
from pathlib import Path

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.trainer import CompletedGameLogEntry, Trainer


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": checkpoint_dir,
            "data_dir": data_dir,
            "model_sync_interval_seconds": 120.0,
        }
    )
    return config


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(_make_config(tmp_path), device="cpu")


def test_remember_recent_completed_replays_keeps_only_recent_replays(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    recent_completed_replays: deque[CompletedGameLogEntry] = deque()
    retained_replay = object()

    trainer._remember_recent_completed_replays(
        recent_completed_replays,
        [
            CompletedGameLogEntry(
                game_number=1,
                stats={"total_attack": 10.0},
                completed_time_s=10.0,
                replay=object(),
            ),
            CompletedGameLogEntry(
                game_number=2,
                stats={"total_attack": 20.0},
                completed_time_s=20.0,
                replay=None,
            ),
            CompletedGameLogEntry(
                game_number=3,
                stats={"total_attack": 30.0},
                completed_time_s=30.0,
                replay=retained_replay,
            ),
        ],
        min_completed_time_s=25.0,
    )

    assert [entry.game_number for entry in recent_completed_replays] == [3]
    assert recent_completed_replays[0].replay is retained_replay


def test_select_direct_sync_recent_game_picks_random_recent_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    stale_replay = object()
    selected_replay = object()
    recent_completed_replays: deque[CompletedGameLogEntry] = deque(
        [
            CompletedGameLogEntry(
                game_number=4,
                stats={"total_attack": 12.0},
                completed_time_s=50.0,
                replay=stale_replay,
            ),
            CompletedGameLogEntry(
                game_number=9,
                stats={"total_attack": 34.0},
                completed_time_s=95.0,
                replay=selected_replay,
            ),
        ]
    )

    monkeypatch.setattr("tetris_bot.ml.trainer.random.choice", lambda items: items[0])

    selection = trainer._select_direct_sync_recent_game(
        recent_completed_replays,
        now_s=100.0,
        window_s=10.0,
    )

    assert [entry.game_number for entry in recent_completed_replays] == [9]
    assert selection is not None
    data, gifs = selection
    assert data == {
        "trainer_step": 0,
        "model_sync/random_recent_game_number": 9.0,
        "model_sync/random_recent_game_attack": 34.0,
        "model_sync/random_recent_game_age_seconds": 5.0,
        "model_sync/random_recent_game_window_seconds": 10.0,
    }
    replay, gif_path = gifs["model_sync/random_recent_game"]
    assert replay is selected_replay
    assert gif_path.name == "direct_sync_step0_game9_attack34.gif"


def test_select_direct_sync_recent_game_skips_empty_window(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    recent_completed_replays: deque[CompletedGameLogEntry] = deque(
        [
            CompletedGameLogEntry(
                game_number=4,
                stats={"total_attack": 12.0},
                completed_time_s=50.0,
                replay=object(),
            )
        ]
    )

    assert (
        trainer._select_direct_sync_recent_game(
            recent_completed_replays, now_s=100.0, window_s=10.0
        )
        is None
    )
