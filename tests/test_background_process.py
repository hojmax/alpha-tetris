from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, DEFAULT_CONFIG_PATH
from tetris_bot.ml.artifacts import required_model_artifact_paths
from tetris_bot.ml.background_process import (
    CompletedExport,
    ReplayData,
    ReplayMoveData,
    TrainerBackgroundProcess,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import TetrisNet
from tetris_core.tetris_core import TetrisEnv

_MODEL_KWARGS = load_training_config(DEFAULT_CONFIG_PATH).network.model_dump()
_TIMEOUT_S = 120.0


def _wait_for_export(process: TrainerBackgroundProcess) -> CompletedExport:
    deadline = time.perf_counter() + _TIMEOUT_S
    while time.perf_counter() < deadline:
        completed = process.poll_export()
        if completed is not None:
            return completed
        time.sleep(0.05)
    raise TimeoutError("background export did not finish")


def _random_replay(seed: int, num_moves: int) -> ReplayData:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
    rng = random.Random(seed)
    moves: list[ReplayMoveData] = []
    while len(moves) < num_moves and not env.game_over:
        attack = None
        while attack is None:
            action = rng.randrange(671)
            attack = env.execute_action_index(action)
        moves.append(ReplayMoveData(action=action, attack=attack))
    return ReplayData(seed=seed, moves=tuple(moves))


@pytest.fixture
def process():
    background_process = TrainerBackgroundProcess(_MODEL_KWARGS)
    yield background_process
    background_process.shutdown()


def test_export_runs_out_of_process_from_a_weight_snapshot(
    process: TrainerBackgroundProcess, tmp_path: Path
) -> None:
    model = TetrisNet(**_MODEL_KWARGS)
    onnx_path = tmp_path / "sync_step_7.onnx"

    process.submit_export(model, step=7, onnx_path=onnx_path)
    assert process.export_in_flight
    with pytest.raises(RuntimeError, match="already in flight"):
        process.submit_export(model, step=8, onnx_path=tmp_path / "other.onnx")
    # Weights captured at submit time must not see later training updates.
    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    completed = _wait_for_export(process)

    assert completed.step == 7
    assert completed.onnx_path == onnx_path
    assert completed.export_ms > 0.0
    assert completed.latency_ms >= completed.export_ms
    assert not process.export_in_flight
    assert process.poll_export() is None
    assert all(path.exists() for path in required_model_artifact_paths(onnx_path))


def test_gif_log_returns_payload_after_all_gifs_are_written(
    process: TrainerBackgroundProcess, tmp_path: Path
) -> None:
    best_path = tmp_path / "best.gif"
    worst_path = tmp_path / "worst.gif"
    process.submit_gif_log(
        {"trainer_step": 3},
        {
            "eval/best_trajectory": (_random_replay(1, 8), best_path),
            "eval/worst_trajectory": (_random_replay(2, 5), worst_path),
        },
    )

    deadline = time.perf_counter() + _TIMEOUT_S
    completed = process.drain_gif_logs()
    while not completed and time.perf_counter() < deadline:
        time.sleep(0.05)
        completed = process.drain_gif_logs()

    assert len(completed) == 1
    assert completed[0].data == {"trainer_step": 3}
    assert completed[0].gif_paths == {
        "eval/best_trajectory": best_path,
        "eval/worst_trajectory": worst_path,
    }
    with Image.open(best_path) as gif:
        assert gif.n_frames == 9


def test_export_failure_is_raised_to_the_trainer(
    process: TrainerBackgroundProcess, tmp_path: Path
) -> None:
    wrong_shape = TetrisNet(**(_MODEL_KWARGS | {"fc_hidden": 64}))
    process.submit_export(wrong_shape, step=1, onnx_path=tmp_path / "bad.onnx")

    deadline = time.perf_counter() + _TIMEOUT_S
    with pytest.raises(RuntimeError, match="size mismatch"):
        while time.perf_counter() < deadline:
            if process.poll_export() is not None:
                break
            time.sleep(0.05)


def test_shutdown_reports_finished_but_unclaimed_export(tmp_path: Path) -> None:
    process = TrainerBackgroundProcess(_MODEL_KWARGS)
    onnx_path = tmp_path / "orphan.onnx"
    process.submit_export(TetrisNet(**_MODEL_KWARGS), step=1, onnx_path=onnx_path)

    assert process.shutdown() == onnx_path
    assert onnx_path.exists()


_ORPHAN_SCRIPT = """
import sys
from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.background_process import TrainerBackgroundProcess
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import TetrisNet

kwargs = load_training_config(DEFAULT_CONFIG_PATH).network.model_dump()
process = TrainerBackgroundProcess(kwargs)
process.submit_export(TetrisNet(**kwargs), step=1, onnx_path=__import__("pathlib").Path(sys.argv[1]))
while process.poll_export() is None:
    pass
print(process._executor._processes.popitem()[0], flush=True)
sys.stdin.read()
"""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_worker_exits_when_trainer_is_killed(tmp_path: Path) -> None:
    trainer = subprocess.Popen(
        [sys.executable, "-c", _ORPHAN_SCRIPT, str(tmp_path / "m.onnx")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert trainer.stdout is not None
    worker_pid = int(trainer.stdout.readline())
    assert _pid_alive(worker_pid)

    trainer.kill()
    trainer.wait()

    deadline = time.perf_counter() + 30.0
    while _pid_alive(worker_pid) and time.perf_counter() < deadline:
        time.sleep(0.2)
    assert not _pid_alive(worker_pid)
