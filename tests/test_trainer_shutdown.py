from __future__ import annotations

import signal
from pathlib import Path

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": checkpoint_dir,
            "data_dir": data_dir,
        }
    )
    return config


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(_make_config(tmp_path), device="cpu")


def test_defer_sigint_during_shutdown_restores_previous_handler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    previous_handler = object()
    signal_calls: list[tuple[int, object]] = []

    monkeypatch.setattr(
        "tetris_bot.ml.trainer.signal.getsignal",
        lambda signum: previous_handler,
    )

    def fake_signal(signum: int, handler: object) -> None:
        signal_calls.append((signum, handler))

    monkeypatch.setattr("tetris_bot.ml.trainer.signal.signal", fake_signal)

    with trainer._defer_sigint_during_shutdown():
        pass

    assert signal_calls == [
        (signal.SIGINT, signal.SIG_IGN),
        (signal.SIGINT, previous_handler),
    ]


def test_shutdown_after_training_stops_generator_and_finishes_wandb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    calls: list[str] = []
    saved_state: dict[str, object] = {}

    class FakeGenerator:
        def stop(self) -> None:
            calls.append("stop")

        def games_generated(self) -> int:
            return 12

        def examples_generated(self) -> int:
            return 34

        def incumbent_uses_network(self) -> bool:
            return False

        def incumbent_model_step(self) -> int:
            return 56

        def incumbent_nn_value_weight(self) -> float:
            return 0.3

        def incumbent_death_penalty(self) -> float:
            return 5.0

        def incumbent_overhang_penalty_weight(self) -> float:
            return 4.0

        def incumbent_eval_avg_attack(self) -> float:
            return 9.5

    generator = FakeGenerator()

    monkeypatch.setattr(
        trainer,
        "_persist_incumbent_model_artifacts",
        lambda _generator: (None, "/tmp/bootstrap.onnx"),
    )
    monkeypatch.setattr(
        trainer,
        "_shutdown_async_checkpoint_saver",
        lambda: calls.append("flush_async"),
    )

    def fake_save(
        *,
        extra_checkpoint_state: dict[str, object] | None = None,
    ) -> dict[str, Path]:
        calls.append("save")
        saved_state["extra_checkpoint_state"] = extra_checkpoint_state
        return {
            "checkpoint": tmp_path / "latest.pt",
            "metadata": tmp_path / "metadata.json",
            "onnx": tmp_path / "latest.onnx",
        }

    monkeypatch.setattr(trainer, "save", fake_save)
    monkeypatch.setattr(
        trainer,
        "_log_final_wandb_model_artifact",
        lambda _saved_paths: calls.append("artifact"),
    )
    monkeypatch.setattr(
        trainer,
        "_cleanup_wandb_gif_files",
        lambda: calls.append("cleanup"),
    )
    monkeypatch.setattr(
        "tetris_bot.ml.trainer.wandb.finish",
        lambda: calls.append("finish"),
    )

    # Resumed run: offsets restored from the checkpoint, generator counters
    # seeded by the replay preload (baseline) must not be counted twice.
    trainer._cumulative_games_offset = 100
    trainer._cumulative_examples_offset = 1_000
    trainer._generator_games_baseline = 10
    trainer._generator_examples_baseline = 30

    stop_error = trainer._shutdown_after_training(
        generator=generator,
        export_model=trainer.model,
        log_to_wandb=True,
        interrupted=True,
    )

    assert stop_error is None
    assert calls == ["stop", "flush_async", "save", "artifact", "finish", "cleanup"]
    extra_checkpoint_state = saved_state["extra_checkpoint_state"]
    assert isinstance(extra_checkpoint_state, dict)
    assert {
        key: extra_checkpoint_state[key]
        for key in (
            "incumbent_uses_network",
            "incumbent_model_step",
            "incumbent_nn_value_weight",
            "incumbent_death_penalty",
            "incumbent_overhang_penalty_weight",
            "incumbent_eval_avg_attack",
            "incumbent_model_source_path",
            "incumbent_model_artifact",
            "cumulative_games_generated",
            "cumulative_examples_generated",
        )
    } == {
        "incumbent_uses_network": False,
        "incumbent_model_step": 56,
        "incumbent_nn_value_weight": 0.3,
        "incumbent_death_penalty": 5.0,
        "incumbent_overhang_penalty_weight": 4.0,
        "incumbent_eval_avg_attack": 9.5,
        "incumbent_model_source_path": "/tmp/bootstrap.onnx",
        "incumbent_model_artifact": None,
        "cumulative_games_generated": 102,
        "cumulative_examples_generated": 1_004,
    }
