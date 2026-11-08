"""Out-of-process model export and replay-GIF rendering for the training loop.

Why a separate process rather than a thread: ONNX export and GIF encoding are
Python-heavy and took seconds per model sync on a busy self-play host. Inline,
they stalled train steps every sync. On a thread they still hold the GIL
(continuous background export halved local train steps/s), and `export_onnx`'s
`redirect_stdout` swaps `sys.stdout` process-wide, swallowing the trainer's own
output while it runs.

At most one export is in flight; the caller decides what to do with the bundle
once `poll_export` returns it (direct sync, or queue it as a candidate).
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tetris_bot.constants import DEFAULT_GIF_FRAME_DURATION_MS
from tetris_bot.ml.artifacts import export_rust_inference_bundle
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import clone_state_value_to_cpu
from tetris_bot.visualization import (
    ReplayLike,
    create_trajectory_gif,
    render_replay,
)


@dataclass(frozen=True)
class ReplayMoveData:
    action: int
    attack: int


@dataclass(frozen=True)
class ReplayData:
    """Picklable copy of a Rust `GameReplay` (the pyclass cannot cross processes)."""

    seed: int
    moves: tuple[ReplayMoveData, ...]

    @classmethod
    def from_replay(cls, replay: ReplayLike) -> ReplayData:
        return cls(
            seed=replay.seed,
            moves=tuple(
                ReplayMoveData(action=move.action, attack=move.attack)
                for move in replay.moves
            ),
        )


@dataclass(frozen=True)
class CompletedExport:
    step: int
    onnx_path: Path
    # Time spent exporting inside the worker process.
    export_ms: float
    # Submit -> result observed by the trainer (includes queueing and pickling).
    latency_ms: float


@dataclass(frozen=True)
class CompletedGifLog:
    """A W&B payload whose GIFs have all been written; `gif_paths` maps key -> file."""

    data: dict[str, object]
    gif_paths: dict[str, Path]


@dataclass(frozen=True)
class _PendingExport:
    step: int
    onnx_path: Path
    submitted_s: float
    future: Future[float]


@dataclass(frozen=True)
class _PendingGifLog:
    data: dict[str, object]
    gif_paths: dict[str, Path]
    futures: tuple[Future[None], ...]


_PARENT_POLL_INTERVAL_SECONDS = 2.0


def _exit_when_orphaned(parent_pid: int) -> None:
    # An idle pool worker blocks on its call queue and never notices the
    # trainer dying from SIGKILL/OOM; without this it would linger forever.
    while os.getppid() == parent_pid:
        time.sleep(_PARENT_POLL_INTERVAL_SECONDS)
    os._exit(0)


def _init_worker(parent_pid: int) -> None:
    threading.Thread(
        target=_exit_when_orphaned,
        args=(parent_pid,),
        name="parent-watchdog",
        daemon=True,
    ).start()
    # Host cores belong to self-play workers; a single small-model export does
    # not benefit from intra-op parallelism.
    torch.set_num_threads(1)


def _export_in_worker(
    state_dict: dict[str, Any],
    model_kwargs: dict[str, Any],
    onnx_path: str,
) -> float:
    model = TetrisNet(**model_kwargs)
    model.load_state_dict(state_dict)
    return export_rust_inference_bundle(
        model, Path(onnx_path), export_name="Background"
    )


def _render_gif_in_worker(replay: ReplayData, gif_path: str) -> None:
    create_trajectory_gif(
        frames=render_replay(replay),
        output_path=gif_path,
        duration=DEFAULT_GIF_FRAME_DURATION_MS,
    )


class TrainerBackgroundProcess:
    def __init__(self, model_kwargs: dict[str, Any]):
        self._model_kwargs = dict(model_kwargs)
        # spawn: the child must never inherit the trainer's CUDA context.
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(os.getpid(),),
        )
        self._export: _PendingExport | None = None
        self._gif_logs: list[_PendingGifLog] = []

    @property
    def export_in_flight(self) -> bool:
        return self._export is not None

    def submit_export(self, model: nn.Module, *, step: int, onnx_path: Path) -> None:
        if self._export is not None:
            raise RuntimeError("A background model export is already in flight")
        state_dict = clone_state_value_to_cpu(model.state_dict())
        future = self._executor.submit(
            _export_in_worker, state_dict, self._model_kwargs, str(onnx_path)
        )
        self._export = _PendingExport(
            step=step,
            onnx_path=onnx_path,
            submitted_s=time.perf_counter(),
            future=future,
        )

    def poll_export(self) -> CompletedExport | None:
        """Return the finished export once; re-raises a failure from the worker."""
        pending = self._export
        if pending is None or not pending.future.done():
            return None
        self._export = None
        export_ms = pending.future.result()
        return CompletedExport(
            step=pending.step,
            onnx_path=pending.onnx_path,
            export_ms=export_ms,
            latency_ms=1000.0 * (time.perf_counter() - pending.submitted_s),
        )

    def submit_gif_log(
        self,
        data: dict[str, object],
        gifs: dict[str, tuple[ReplayLike, Path]],
    ) -> None:
        """Render `gifs` (W&B key -> (replay, output path)) in the worker.

        `drain_gif_logs` hands `data` back once every GIF in it is written,
        so the caller can attach the videos and log everything together.
        """
        futures = tuple(
            self._executor.submit(
                _render_gif_in_worker, ReplayData.from_replay(replay), str(gif_path)
            )
            for replay, gif_path in gifs.values()
        )
        self._gif_logs.append(
            _PendingGifLog(
                data=dict(data),
                gif_paths={key: gif_path for key, (_, gif_path) in gifs.items()},
                futures=futures,
            )
        )

    def drain_gif_logs(self) -> list[CompletedGifLog]:
        """Return finished payloads in submission order; re-raises worker failures."""
        completed: list[CompletedGifLog] = []
        while self._gif_logs and all(f.done() for f in self._gif_logs[0].futures):
            pending = self._gif_logs.pop(0)
            for future in pending.futures:
                future.result()
            completed.append(
                CompletedGifLog(data=pending.data, gif_paths=pending.gif_paths)
            )
        return completed

    def shutdown(self) -> Path | None:
        """Stop the worker, dropping queued GIF logs.

        Waits for an export already running and returns its ONNX path when it
        finished but was never handed to the caller, so the caller can delete
        the orphaned bundle.
        """
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._gif_logs.clear()
        pending = self._export
        self._export = None
        if pending is None or pending.future.cancelled():
            return None
        if pending.future.exception() is not None:
            return None
        return pending.onnx_path
