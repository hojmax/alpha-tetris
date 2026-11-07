from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": tmp_path / "checkpoints",
            "data_dir": tmp_path / "data",
        }
    )
    return config


class _CountingGenerator:
    def __init__(self, *, games: int, examples: int) -> None:
        self.games = games
        self.examples = examples

    def games_generated(self) -> int:
        return self.games

    def examples_generated(self) -> int:
        return self.examples


def test_cumulative_counters_exclude_preloaded_generator_baseline(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_make_config(tmp_path), device="cpu")
    trainer._cumulative_games_offset = 1_000
    trainer._cumulative_examples_offset = 70_000
    # Generator preloaded a buffer: counters start at the loaded totals.
    generator = _CountingGenerator(games=900, examples=60_000)
    trainer._generator_games_baseline = generator.games_generated()
    trainer._generator_examples_baseline = generator.examples_generated()
    generator.games += 5
    generator.examples += 350

    assert trainer._cumulative_games_generated(generator) == 1_005  # type: ignore[arg-type]
    assert trainer._cumulative_examples_generated(generator) == 70_350  # type: ignore[arg-type]
    assert trainer._session_examples_generated(generator) == 350  # type: ignore[arg-type]


def test_replay_ratio_cap_disabled_by_default(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assert config.replay.max_train_samples_per_new_example == 0.0
    trainer = Trainer(config, device="cpu")
    generator = _CountingGenerator(games=0, examples=0)

    assert trainer._replay_ratio_allows_step(generator, session_steps=10**6)  # type: ignore[arg-type]


def test_replay_ratio_cap_limits_samples_per_new_example(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.optimizer.batch_size = 100
    config.replay.max_train_samples_per_new_example = 4.0
    trainer = Trainer(config, device="cpu")
    generator = _CountingGenerator(games=0, examples=10_000)
    trainer._generator_examples_baseline = 10_000  # preloaded buffer
    allows = trainer._replay_ratio_allows_step

    # Preloaded examples do not count: no new data yet, so no steps.
    assert not allows(generator, session_steps=0)  # type: ignore[arg-type]
    generator.examples += 50  # 50 new examples * 4 = 200 samples = 2 steps
    assert allows(generator, session_steps=0)  # type: ignore[arg-type]
    assert allows(generator, session_steps=1)  # type: ignore[arg-type]
    assert not allows(generator, session_steps=2)  # type: ignore[arg-type]


def test_replay_ratio_cap_rejects_negative_values(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with pytest.raises(ValueError):
        config.replay.max_train_samples_per_new_example = -1.0
