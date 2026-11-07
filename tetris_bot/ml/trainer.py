from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
from typing import cast

from pydantic import BaseModel, ConfigDict
import structlog
import torch
import wandb

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FRAME_DURATION_MS,
    INCUMBENT_ONNX_FILENAME,
    LATEST_ONNX_FILENAME,
    MODEL_CANDIDATES_DIRNAME,
    PARALLEL_ONNX_FILENAME,
    RUNTIME_OVERRIDES_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import (
    ResolvedRuntimeOptimizerOverrides,
    ResolvedRuntimeOverrides,
    ResolvedRuntimeRunOverrides,
    ResolvedRuntimeSelfPlayOverrides,
    RuntimeOverrides,
    SelfPlaySnapshot,
    TrainingConfig,
    load_runtime_overrides,
    save_runtime_overrides,
)
from tetris_bot.ml.ema import ExponentialMovingAverage
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.optimizer import OptimizerBundle, SchedulerBundle
from tetris_bot.ml.weights import (
    AsyncCheckpointSaver,
    WeightManager,
    capture_checkpoint_snapshot,
    export_onnx,
    export_split_models,
    sanitize_optimizer_state_steps,
    split_model_paths,
)
from tetris_bot.ml.loss import RunningLossBalancer, compute_loss, compute_metrics
from tetris_bot.visualization import create_trajectory_gif, render_replay
from tetris_bot.ml.artifacts import (
    assert_rust_inference_artifacts,
    copy_model_artifact_bundle,
    optional_model_artifact_paths,
)
from tetris_bot.ml.replay_buffer import TrainingBatch, CircularReplayMirror
from tetris_bot.ml.game_metrics import (
    average_completed_games,
    compute_batch_feature_metrics,
)
from tetris_bot.ml.nn_value_weight_schedule import compute_nn_value_weight
from tetris_bot.ml.penalty_schedule import scaled_penalties
from tetris_bot.ml.policy_mirroring import maybe_mirror_training_tensors
from tetris_bot.ml.r2_sync import (
    ChunkDownloader,
    GameStatsDownloader,
    MachineOffsetTable,
    R2Settings,
    upload_model_bundle,
    upload_self_play_snapshot,
)

from tetris_core.tetris_core import GameGenerator, GameReplay, MCTSConfig
from tetris_core.tetris_core import evaluate_model

logger = structlog.get_logger()
RUNTIME_OVERRIDES_POLL_INTERVAL_SECONDS = 15.0
REPLAY_RATIO_THROTTLE_SLEEP_SECONDS = 0.05


def roll_interval_deadline(deadline_s: float, interval_s: float, now_s: float) -> float:
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0 (got {interval_s})")
    if now_s < deadline_s:
        return deadline_s
    elapsed_intervals = int((now_s - deadline_s) // interval_s) + 1
    return deadline_s + elapsed_intervals * interval_s


@dataclass
class CandidateGateSchedule:
    current_interval_seconds: float
    failed_promotion_streak: int
    next_export_time_s: float


class CompletedGameLogEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    game_number: int
    stats: dict[str, float]
    completed_time_s: float
    replay: object | None = None

    @property
    def total_attack(self) -> int:
        return int(self.stats["total_attack"])


def _compute_warmup_cosine_factor(
    *,
    step: int,
    warmup_steps: int,
    decay_steps: int,
    warmup_start_factor: float,
    min_factor: float,
) -> float:
    """LR multiplier for `warmup_cosine`: linear warmup, cosine decay, then flat.

    `step=0` returns `warmup_start_factor`; the warmup ends at peak (factor 1)
    after `warmup_steps`, the cosine then decays to `min_factor` over
    `decay_steps`, and stays at `min_factor` afterward.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0 (got {step})")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0 (got {warmup_steps})")
    if decay_steps < 0:
        raise ValueError(f"decay_steps must be >= 0 (got {decay_steps})")
    if not 0.0 <= warmup_start_factor <= 1.0:
        raise ValueError(
            f"warmup_start_factor must be in [0, 1] (got {warmup_start_factor})"
        )
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f"min_factor must be in [0, 1] (got {min_factor})")

    if warmup_steps > 0 and step < warmup_steps:
        progress = step / warmup_steps
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress
    if decay_steps == 0:
        return min_factor
    decay_progress = min((step - warmup_steps) / decay_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_factor + (1.0 - min_factor) * cosine


def _candidate_gate_interval_seconds(
    *,
    base_interval_seconds: float,
    failure_backoff_seconds: float,
    failed_promotion_streak: int,
    max_interval_seconds: float | None,
) -> float:
    if failed_promotion_streak < 0:
        raise ValueError(
            f"failed_promotion_streak must be >= 0 (got {failed_promotion_streak})"
        )
    interval_seconds = (
        base_interval_seconds + failure_backoff_seconds * failed_promotion_streak
    )
    if max_interval_seconds is not None:
        interval_seconds = min(interval_seconds, max_interval_seconds)
    return interval_seconds


class Trainer:
    """Main training class."""

    def __init__(
        self,
        config: TrainingConfig,
        model: TetrisNet | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)

        # Validate paths are set (should be done by setup_run_directory)
        if (
            config.run.run_dir is None
            or config.run.checkpoint_dir is None
            or config.run.data_dir is None
        ):
            raise ValueError(
                "run_dir, checkpoint_dir, and data_dir must be set. "
                "Call setup_run_directory() before creating Trainer."
            )

        # Create model
        if model is None:
            model = TetrisNet(**config.network.model_dump())
        self.model = model.to(self.device)
        if not 0.0 <= self.config.optimizer.ema_decay < 1.0:
            raise ValueError(
                "config.optimizer.ema_decay must be in [0, 1) "
                f"(got {self.config.optimizer.ema_decay})"
            )
        if not 0.0 <= self.config.optimizer.mirror_augmentation_probability <= 1.0:
            raise ValueError(
                "config.optimizer.mirror_augmentation_probability must be in [0, 1] "
                "(got "
                f"{self.config.optimizer.mirror_augmentation_probability})"
            )
        self._export_model = self.model
        self.ema = (
            ExponentialMovingAverage(
                self._export_model, self.config.optimizer.ema_decay
            )
            if self.config.optimizer.ema_decay > 0.0
            else None
        )

        # torch.compile is slower than eager on Apple Silicon (MPS); auto-disable
        # there so the train loop uses the eager fast path.
        self._effective_use_torch_compile = (
            config.optimizer.use_torch_compile and self.device.type != "mps"
        )

        # Create hybrid Muon (2D hidden Linear weights) + AdamW (everything else).
        self.optimizer = OptimizerBundle(
            self.model,
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create weight manager
        self.weight_manager = WeightManager(config.run.checkpoint_dir)

        # Training state
        self.step = 0
        self.loss_balancer = RunningLossBalancer(
            config.optimizer.value_loss_weight_window
        )
        self._cached_value_loss_weight: float = 1.0
        self._pending_eval_gif_paths: list[Path] = []
        self._async_checkpoint_saver: AsyncCheckpointSaver | None = None
        self.initial_incumbent_model_path: Path | None = None
        self.initial_incumbent_eval_avg_attack: float = 0.0
        self.initial_candidate_gate_interval_seconds: float | None = None
        self.initial_candidate_gate_failed_promotion_streak: int = 0
        self.initial_candidate_gate_next_export_delay_seconds: float | None = None
        self.recompute_initial_incumbent_eval_avg_attack = False
        self._logged_live_optimizer_step_sanitization = False
        self._lr_multiplier = 1.0
        # One-shot latch for `self_play.force_promote_next_candidate`: set
        # when the runtime overrides file says so, consumed (and reset) when
        # the next candidate is queued.
        self._force_promote_next_candidate: bool = False
        # Tracks the un-multiplied LR each scheduler step would have produced.
        # `param_group["lr"]` is always `_scheduler_base_lrs[i] * _lr_multiplier`.
        # Keeping the base separate is what prevents the multiplier from
        # compounding when the scheduler reads `param_group["lr"]` (LinearLR).
        self._scheduler_base_lrs: list[float] = [
            float(pg["lr"]) for pg in self.optimizer.param_groups
        ]
        self._runtime_override_defaults = ResolvedRuntimeOverrides(
            optimizer=ResolvedRuntimeOptimizerOverrides(
                lr_multiplier=1.0,
                grad_clip_norm=self.config.optimizer.grad_clip_norm,
                weight_decay=self.config.optimizer.weight_decay,
                mirror_augmentation_probability=(
                    self.config.optimizer.mirror_augmentation_probability
                ),
            ),
            run=ResolvedRuntimeRunOverrides(
                log_interval_seconds=self.config.run.log_interval_seconds,
                checkpoint_interval_seconds=self.config.run.checkpoint_interval_seconds,
            ),
            self_play=ResolvedRuntimeSelfPlayOverrides(
                add_noise=self.config.self_play.add_noise,
                visit_sampling_epsilon=self.config.self_play.visit_sampling_epsilon,
            ),
        )
        self._runtime_overrides_path = config.run.run_dir / RUNTIME_OVERRIDES_FILENAME
        self._runtime_overrides_last_mtime_ns: int | None = None
        self._next_runtime_overrides_check_time_s: float | None = None
        self._r2_settings: R2Settings | None = None
        self._r2_chunk_downloader: ChunkDownloader | None = None
        self._r2_game_stats_downloader: GameStatsDownloader | None = None
        self._r2_last_uploaded_step: int = -1
        self._remote_completed_games_lock = threading.Lock()
        self._remote_completed_games: deque[CompletedGameLogEntry] = deque()
        # Per-machine completed-game counts feeding the
        # `throughput/games_per_second/<machine_id>` W&B series. The
        # trainer's own self-play workers use the local hostname as their
        # machine_id; remote generators report theirs through
        # `push_remote_completed_games`. Updated at the drain chokepoint
        # and from the GameStatsDownloader thread; protected by a
        # dedicated lock.
        import socket as _socket

        self._local_machine_id: str = (
            _socket.gethostname().replace(".", "_") or "trainer"
        )
        self._games_per_machine_lock = threading.Lock()
        self._games_per_machine: dict[str, int] = {}
        # Strictly-monotonic game number used as the W&B-facing identifier.
        # Buffer/replay game_numbers stay in their per-machine block ranges
        # (so per-game grouping in the buffer stays unique), but every game
        # passed to W&B logging gets renumbered into a single 1, 2, 3, …
        # stream at the drain chokepoint. Counter persists in checkpoints so
        # resumed runs continue without restarting at 1.
        self._next_display_game_number: int = 1

        # Cumulative offsets so resumed runs produce W&B curves that line up
        # with the prior run on wall-time, games, and examples axes — matching
        # what `step` and `_next_display_game_number` already do. The anchor
        # is set when the training loop starts; the offsets are populated
        # from the checkpoint by the resume path in train.py.
        self._wall_time_anchor_s: float = 0.0
        self._cumulative_wall_time_offset_s: float = 0.0
        self._cumulative_games_offset: int = 0
        self._cumulative_examples_offset: int = 0
        # Generator counters captured right after `generator.start()`. A replay
        # preload seeds `games_generated` / `examples_generated` with the loaded
        # buffer's max game number / example count, which the restored
        # cumulative offsets above already include; subtracting the baseline
        # keeps resumed runs from double counting the preloaded buffer.
        self._generator_games_baseline: int = 0
        self._generator_examples_baseline: int = 0

        # Create directories
        config.run.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.run.data_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_gating_enabled(self) -> bool:
        return self.config.self_play.use_candidate_gating

    def _cumulative_wall_time_seconds(self, now_s: float) -> float:
        return self._cumulative_wall_time_offset_s + (now_s - self._wall_time_anchor_s)

    def _cumulative_wall_time_hours(self, now_s: float) -> float:
        return self._cumulative_wall_time_seconds(now_s) / 3600.0

    def _session_examples_generated(self, generator: GameGenerator) -> int:
        return generator.examples_generated() - self._generator_examples_baseline

    def _cumulative_games_generated(self, generator: GameGenerator) -> int:
        return (
            self._cumulative_games_offset
            + generator.games_generated()
            - self._generator_games_baseline
        )

    def _cumulative_examples_generated(self, generator: GameGenerator) -> int:
        return self._cumulative_examples_offset + self._session_examples_generated(
            generator
        )

    def _replay_ratio_allows_step(
        self, generator: GameGenerator, *, session_steps: int
    ) -> bool:
        """Whether one more step keeps samples drawn per new example under the cap.

        Counts only this session's samples and newly generated (or ingested
        remote) examples, so a resumed run with a preloaded buffer waits for
        fresh data instead of re-training on the already-trained window.
        """
        max_ratio = self.config.replay.max_train_samples_per_new_example
        if max_ratio <= 0.0:
            return True
        next_session_samples = (session_steps + 1) * self.config.optimizer.batch_size
        return next_session_samples <= max_ratio * self._session_examples_generated(
            generator
        )

    def _cumulative_progress_checkpoint_state(
        self, generator: GameGenerator, *, now_s: float
    ) -> dict[str, float | int]:
        return {
            "cumulative_wall_time_seconds": self._cumulative_wall_time_seconds(now_s),
            "cumulative_games_generated": self._cumulative_games_generated(generator),
            "cumulative_examples_generated": self._cumulative_examples_generated(
                generator
            ),
        }

    def _model_sync_interval_seconds(self) -> float:
        model_sync_interval_seconds = self.config.run.model_sync_interval_seconds
        if model_sync_interval_seconds <= 0.0:
            raise ValueError(
                "config.run.model_sync_interval_seconds must be > 0 "
                f"(got {model_sync_interval_seconds})"
            )
        return model_sync_interval_seconds

    def _drain_completed_games(
        self, generator: GameGenerator
    ) -> list[CompletedGameLogEntry]:
        local = [
            CompletedGameLogEntry.model_validate(payload)
            for payload in generator.drain_completed_games()
        ]
        if local:
            self._increment_games_per_machine(self._local_machine_id, len(local))
        remote = self._drain_remote_completed_games()
        combined = local + remote
        if not combined:
            return combined
        # Sort by completion time so the strictly-sequential display numbers
        # we stamp next match wall-clock order; this keeps W&B per-game
        # charts using game_number as x-axis cleanly increasing in time.
        combined.sort(key=lambda entry: entry.completed_time_s)
        for entry in combined:
            entry.game_number = self._next_display_game_number
            self._next_display_game_number += 1
        return combined

    def _increment_games_per_machine(self, machine_id: str, count: int) -> None:
        if count <= 0:
            return
        with self._games_per_machine_lock:
            self._games_per_machine[machine_id] = (
                self._games_per_machine.get(machine_id, 0) + count
            )

    def _snapshot_games_per_machine(self) -> dict[str, int]:
        with self._games_per_machine_lock:
            return dict(self._games_per_machine)

    def _drain_remote_completed_games(self) -> list[CompletedGameLogEntry]:
        with self._remote_completed_games_lock:
            if not self._remote_completed_games:
                return []
            drained = list(self._remote_completed_games)
            self._remote_completed_games.clear()
        return drained

    def push_remote_completed_games(
        self, entries: list[dict], game_number_offset: int, machine_id: str
    ) -> None:
        """Sink for `GameStatsDownloader` — applies offset and queues entries.

        Called from the downloader background thread; the lock keeps reads
        from the trainer main loop consistent. `machine_id` is the source
        generator's id, used to attribute games to per-machine throughput
        series.
        """
        validated: list[CompletedGameLogEntry] = []
        for entry in entries:
            try:
                logged = CompletedGameLogEntry.model_validate(entry)
            except Exception:
                logger.exception(
                    "trainer.remote_completed_game_invalid",
                    entry_keys=list(entry.keys()) if isinstance(entry, dict) else None,
                )
                continue
            logged.game_number = logged.game_number + game_number_offset
            validated.append(logged)
        if not validated:
            return
        with self._remote_completed_games_lock:
            self._remote_completed_games.extend(validated)
        self._increment_games_per_machine(machine_id, len(validated))

    @staticmethod
    def _prune_recent_completed_replays(
        recent_completed_replays: deque[CompletedGameLogEntry],
        min_completed_time_s: float,
    ) -> None:
        retained_replays = [
            replay
            for replay in recent_completed_replays
            if replay.completed_time_s >= min_completed_time_s
        ]
        recent_completed_replays.clear()
        recent_completed_replays.extend(retained_replays)

    def _remember_recent_completed_replays(
        self,
        recent_completed_replays: deque[CompletedGameLogEntry],
        completed_games: list[CompletedGameLogEntry],
        *,
        min_completed_time_s: float,
    ) -> None:
        for completed_game in completed_games:
            if completed_game.replay is not None:
                recent_completed_replays.append(completed_game)
        self._prune_recent_completed_replays(
            recent_completed_replays,
            min_completed_time_s,
        )

    def _build_direct_sync_recent_game_wandb_data(
        self,
        recent_completed_replays: deque[CompletedGameLogEntry],
        *,
        now_s: float,
        window_s: float,
    ) -> dict[str, object]:
        self._prune_recent_completed_replays(
            recent_completed_replays,
            now_s - window_s,
        )
        if not recent_completed_replays:
            return {}

        selected_replay = random.choice(list(recent_completed_replays))
        replay_payload = selected_replay.replay
        if replay_payload is None:
            return {}
        replay = cast(GameReplay, replay_payload)

        frames = render_replay(replay)
        video, _ = self._create_wandb_gif_video(
            frames,
            attack=selected_replay.total_attack,
            gif_stem=(
                "direct_sync"
                f"_step{self.step}"
                f"_game{selected_replay.game_number}"
                f"_attack{selected_replay.total_attack}"
            ),
        )
        if video is None:
            return {}

        return {
            "model_sync/random_recent_game": video,
            "model_sync/random_recent_game_number": float(selected_replay.game_number),
            "model_sync/random_recent_game_attack": float(selected_replay.total_attack),
            "model_sync/random_recent_game_age_seconds": max(
                0.0, now_s - selected_replay.completed_time_s
            ),
            "model_sync/random_recent_game_window_seconds": window_s,
        }

    def _current_runtime_override_state(self) -> ResolvedRuntimeOverrides:
        return ResolvedRuntimeOverrides(
            optimizer=ResolvedRuntimeOptimizerOverrides(
                lr_multiplier=self._lr_multiplier,
                grad_clip_norm=self.config.optimizer.grad_clip_norm,
                weight_decay=self.config.optimizer.weight_decay,
                mirror_augmentation_probability=(
                    self.config.optimizer.mirror_augmentation_probability
                ),
            ),
            run=ResolvedRuntimeRunOverrides(
                log_interval_seconds=self.config.run.log_interval_seconds,
                checkpoint_interval_seconds=self.config.run.checkpoint_interval_seconds,
            ),
            self_play=ResolvedRuntimeSelfPlayOverrides(
                add_noise=self.config.self_play.add_noise,
                visit_sampling_epsilon=self.config.self_play.visit_sampling_epsilon,
            ),
        )

    def _resolved_runtime_overrides(
        self, override_file_values: ResolvedRuntimeOverrides | None = None
    ) -> ResolvedRuntimeOverrides:
        if override_file_values is not None:
            return override_file_values
        return self._runtime_override_defaults

    @staticmethod
    def _validate_runtime_override_value(
        name: str,
        value: float,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        strictly_positive: bool = False,
    ) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite (got {value})")
        if strictly_positive and value <= 0.0:
            raise ValueError(f"{name} must be > 0 (got {value})")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value} (got {value})")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value} (got {value})")

    def _read_runtime_overrides_file(
        self,
    ) -> tuple[int | None, ResolvedRuntimeOverrides, RuntimeOverrides]:
        if not self._runtime_overrides_path.exists():
            return None, self._runtime_override_defaults, RuntimeOverrides()

        file_mtime_ns = self._runtime_overrides_path.stat().st_mtime_ns
        overrides = load_runtime_overrides(self._runtime_overrides_path)
        resolved = ResolvedRuntimeOverrides(
            optimizer=ResolvedRuntimeOptimizerOverrides(
                lr_multiplier=(
                    self._runtime_override_defaults.optimizer.lr_multiplier
                    if overrides.optimizer.lr_multiplier is None
                    else overrides.optimizer.lr_multiplier
                ),
                grad_clip_norm=(
                    self._runtime_override_defaults.optimizer.grad_clip_norm
                    if overrides.optimizer.grad_clip_norm is None
                    else overrides.optimizer.grad_clip_norm
                ),
                weight_decay=(
                    self._runtime_override_defaults.optimizer.weight_decay
                    if overrides.optimizer.weight_decay is None
                    else overrides.optimizer.weight_decay
                ),
                mirror_augmentation_probability=(
                    self._runtime_override_defaults.optimizer.mirror_augmentation_probability
                    if overrides.optimizer.mirror_augmentation_probability is None
                    else overrides.optimizer.mirror_augmentation_probability
                ),
            ),
            run=ResolvedRuntimeRunOverrides(
                log_interval_seconds=(
                    self._runtime_override_defaults.run.log_interval_seconds
                    if overrides.run.log_interval_seconds is None
                    else overrides.run.log_interval_seconds
                ),
                checkpoint_interval_seconds=(
                    self._runtime_override_defaults.run.checkpoint_interval_seconds
                    if overrides.run.checkpoint_interval_seconds is None
                    else overrides.run.checkpoint_interval_seconds
                ),
            ),
            self_play=ResolvedRuntimeSelfPlayOverrides(
                add_noise=(
                    self._runtime_override_defaults.self_play.add_noise
                    if overrides.self_play.add_noise is None
                    else overrides.self_play.add_noise
                ),
                visit_sampling_epsilon=(
                    self._runtime_override_defaults.self_play.visit_sampling_epsilon
                    if overrides.self_play.visit_sampling_epsilon is None
                    else overrides.self_play.visit_sampling_epsilon
                ),
            ),
        )
        return file_mtime_ns, resolved, overrides

    @staticmethod
    def _reschedule_runtime_deadline(
        *,
        current_deadline_s: float | None,
        previous_interval_s: float,
        new_interval_s: float,
        now_s: float,
    ) -> float | None:
        if current_deadline_s is None or previous_interval_s == new_interval_s:
            return current_deadline_s
        previous_event_time_s = current_deadline_s - previous_interval_s
        next_deadline_s = previous_event_time_s + new_interval_s
        if next_deadline_s <= now_s:
            return now_s
        return next_deadline_s

    def _set_lr_multiplier(self, lr_multiplier: float) -> None:
        if lr_multiplier == self._lr_multiplier:
            return
        self._lr_multiplier = lr_multiplier
        for param_group, base_lr in zip(
            self.optimizer.param_groups, self._scheduler_base_lrs
        ):
            param_group["lr"] = base_lr * lr_multiplier

    def _step_scheduler(self) -> None:
        """Step the LR scheduler with the multiplier kept outside its state.

        Schedulers like `LinearLR` compute the next LR from the current
        `param_group["lr"]`, so we must restore the un-multiplied base before
        stepping. Otherwise, re-applying the multiplier after each step would
        compound it geometrically.
        """
        if self.scheduler is None:
            return
        for param_group, base_lr in zip(
            self.optimizer.param_groups, self._scheduler_base_lrs
        ):
            param_group["lr"] = base_lr
        self.scheduler.step()
        self._scheduler_base_lrs = [
            float(pg["lr"]) for pg in self.optimizer.param_groups
        ]
        if self._lr_multiplier != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self._lr_multiplier

    def restore_runtime_override_state(
        self,
        state: ResolvedRuntimeOverrides,
        *,
        lrs_already_scaled: bool,
    ) -> None:
        self._validate_runtime_override_value(
            "runtime override optimizer.lr_multiplier",
            state.optimizer.lr_multiplier,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.grad_clip_norm",
            state.optimizer.grad_clip_norm,
            min_value=0.0,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.weight_decay",
            state.optimizer.weight_decay,
            min_value=0.0,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.mirror_augmentation_probability",
            state.optimizer.mirror_augmentation_probability,
            min_value=0.0,
            max_value=1.0,
        )
        self._validate_runtime_override_value(
            "runtime override run.log_interval_seconds",
            state.run.log_interval_seconds,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override run.checkpoint_interval_seconds",
            state.run.checkpoint_interval_seconds,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override self_play.visit_sampling_epsilon",
            state.self_play.visit_sampling_epsilon,
            min_value=0.0,
            max_value=1.0,
        )

        self.config.optimizer.grad_clip_norm = state.optimizer.grad_clip_norm
        self.config.optimizer.weight_decay = state.optimizer.weight_decay
        self.config.optimizer.mirror_augmentation_probability = (
            state.optimizer.mirror_augmentation_probability
        )
        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = state.optimizer.weight_decay
        self.config.run.log_interval_seconds = state.run.log_interval_seconds
        self.config.run.checkpoint_interval_seconds = (
            state.run.checkpoint_interval_seconds
        )
        self.config.self_play.add_noise = state.self_play.add_noise
        self.config.self_play.visit_sampling_epsilon = (
            state.self_play.visit_sampling_epsilon
        )
        if lrs_already_scaled:
            # `param_group["lr"]` was saved as `scheduler_base * multiplier`.
            # Recover the un-multiplied scheduler base so future scheduler
            # steps operate on it instead of compounding the multiplier.
            self._lr_multiplier = state.optimizer.lr_multiplier
            self._scheduler_base_lrs = [
                float(pg["lr"]) / state.optimizer.lr_multiplier
                for pg in self.optimizer.param_groups
            ]
        else:
            self._scheduler_base_lrs = [
                float(pg["lr"]) for pg in self.optimizer.param_groups
            ]
            self._set_lr_multiplier(state.optimizer.lr_multiplier)

    def _maybe_reload_runtime_overrides(
        self,
        *,
        now_s: float,
        next_log_time_s: float | None,
        next_checkpoint_time_s: float | None,
        generator: GameGenerator | None = None,
        force: bool = False,
    ) -> tuple[float | None, float | None]:
        if (
            not force
            and self._next_runtime_overrides_check_time_s is not None
            and now_s < self._next_runtime_overrides_check_time_s
        ):
            return next_log_time_s, next_checkpoint_time_s
        self._next_runtime_overrides_check_time_s = (
            now_s + RUNTIME_OVERRIDES_POLL_INTERVAL_SECONDS
        )

        try:
            file_mtime_ns, resolved, raw_overrides = self._read_runtime_overrides_file()
        except Exception:
            logger.warning(
                "Failed to reload runtime overrides; keeping previous values",
                runtime_overrides_path=str(self._runtime_overrides_path),
                exc_info=True,
            )
            return next_log_time_s, next_checkpoint_time_s

        if not force and file_mtime_ns == self._runtime_overrides_last_mtime_ns:
            return next_log_time_s, next_checkpoint_time_s

        current_state = self._current_runtime_override_state()
        changes: dict[str, tuple[object, object]] = {}

        self._validate_runtime_override_value(
            "runtime override optimizer.lr_multiplier",
            resolved.optimizer.lr_multiplier,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.grad_clip_norm",
            resolved.optimizer.grad_clip_norm,
            min_value=0.0,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.weight_decay",
            resolved.optimizer.weight_decay,
            min_value=0.0,
        )
        self._validate_runtime_override_value(
            "runtime override optimizer.mirror_augmentation_probability",
            resolved.optimizer.mirror_augmentation_probability,
            min_value=0.0,
            max_value=1.0,
        )
        self._validate_runtime_override_value(
            "runtime override run.log_interval_seconds",
            resolved.run.log_interval_seconds,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override run.checkpoint_interval_seconds",
            resolved.run.checkpoint_interval_seconds,
            strictly_positive=True,
        )
        self._validate_runtime_override_value(
            "runtime override self_play.visit_sampling_epsilon",
            resolved.self_play.visit_sampling_epsilon,
            min_value=0.0,
            max_value=1.0,
        )

        if resolved.optimizer.lr_multiplier != current_state.optimizer.lr_multiplier:
            changes["optimizer.lr_multiplier"] = (
                current_state.optimizer.lr_multiplier,
                resolved.optimizer.lr_multiplier,
            )
            self._set_lr_multiplier(resolved.optimizer.lr_multiplier)

        if resolved.optimizer.grad_clip_norm != current_state.optimizer.grad_clip_norm:
            changes["optimizer.grad_clip_norm"] = (
                current_state.optimizer.grad_clip_norm,
                resolved.optimizer.grad_clip_norm,
            )
            self.config.optimizer.grad_clip_norm = resolved.optimizer.grad_clip_norm

        if resolved.optimizer.weight_decay != current_state.optimizer.weight_decay:
            changes["optimizer.weight_decay"] = (
                current_state.optimizer.weight_decay,
                resolved.optimizer.weight_decay,
            )
            self.config.optimizer.weight_decay = resolved.optimizer.weight_decay
            for param_group in self.optimizer.param_groups:
                param_group["weight_decay"] = resolved.optimizer.weight_decay

        if (
            resolved.optimizer.mirror_augmentation_probability
            != current_state.optimizer.mirror_augmentation_probability
        ):
            changes["optimizer.mirror_augmentation_probability"] = (
                current_state.optimizer.mirror_augmentation_probability,
                resolved.optimizer.mirror_augmentation_probability,
            )
            self.config.optimizer.mirror_augmentation_probability = (
                resolved.optimizer.mirror_augmentation_probability
            )

        previous_log_interval_s = current_state.run.log_interval_seconds
        if resolved.run.log_interval_seconds != previous_log_interval_s:
            changes["run.log_interval_seconds"] = (
                previous_log_interval_s,
                resolved.run.log_interval_seconds,
            )
            self.config.run.log_interval_seconds = resolved.run.log_interval_seconds
            next_log_time_s = self._reschedule_runtime_deadline(
                current_deadline_s=next_log_time_s,
                previous_interval_s=previous_log_interval_s,
                new_interval_s=resolved.run.log_interval_seconds,
                now_s=now_s,
            )

        previous_checkpoint_interval_s = current_state.run.checkpoint_interval_seconds
        if resolved.run.checkpoint_interval_seconds != previous_checkpoint_interval_s:
            changes["run.checkpoint_interval_seconds"] = (
                previous_checkpoint_interval_s,
                resolved.run.checkpoint_interval_seconds,
            )
            self.config.run.checkpoint_interval_seconds = (
                resolved.run.checkpoint_interval_seconds
            )
            next_checkpoint_time_s = self._reschedule_runtime_deadline(
                current_deadline_s=next_checkpoint_time_s,
                previous_interval_s=previous_checkpoint_interval_s,
                new_interval_s=resolved.run.checkpoint_interval_seconds,
                now_s=now_s,
            )

        self_play_changed = False
        if resolved.self_play.add_noise != current_state.self_play.add_noise:
            changes["self_play.add_noise"] = (
                current_state.self_play.add_noise,
                resolved.self_play.add_noise,
            )
            self.config.self_play.add_noise = resolved.self_play.add_noise
            self_play_changed = True

        if (
            resolved.self_play.visit_sampling_epsilon
            != current_state.self_play.visit_sampling_epsilon
        ):
            changes["self_play.visit_sampling_epsilon"] = (
                current_state.self_play.visit_sampling_epsilon,
                resolved.self_play.visit_sampling_epsilon,
            )
            self.config.self_play.visit_sampling_epsilon = (
                resolved.self_play.visit_sampling_epsilon
            )
            self_play_changed = True

        if self_play_changed:
            if generator is not None:
                try:
                    generator.update_search_overrides(
                        add_noise=resolved.self_play.add_noise,
                        visit_sampling_epsilon=(
                            resolved.self_play.visit_sampling_epsilon
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Failed to push self-play overrides to GameGenerator; "
                        "trainer config is updated but live workers may be stale",
                    )
            self._republish_self_play_snapshot_to_r2()

        if raw_overrides.self_play.force_promote_next_candidate:
            changes["self_play.force_promote_next_candidate"] = (False, True)
            if not self._candidate_gating_enabled():
                logger.warning(
                    "self_play.force_promote_next_candidate set while candidate "
                    "gating is disabled; clearing without effect",
                    runtime_overrides_path=str(self._runtime_overrides_path),
                )
                self._force_promote_next_candidate = False
            else:
                self._force_promote_next_candidate = True
            raw_overrides.self_play.force_promote_next_candidate = False
            try:
                save_runtime_overrides(raw_overrides, self._runtime_overrides_path)
                file_mtime_ns = self._runtime_overrides_path.stat().st_mtime_ns
            except Exception:
                logger.warning(
                    "Failed to clear force_promote_next_candidate trigger in "
                    "runtime_overrides.yaml; latch is set in memory but the "
                    "file still shows true",
                    runtime_overrides_path=str(self._runtime_overrides_path),
                    exc_info=True,
                )

        self._runtime_overrides_last_mtime_ns = file_mtime_ns
        if changes:
            flattened_changes = {
                key.replace(".", "_"): {"old": old, "new": new}
                for key, (old, new) in changes.items()
            }
            logger.info(
                "Applied runtime overrides",
                runtime_overrides_path=str(self._runtime_overrides_path),
                changes=flattened_changes,
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "trainer_step": self.step,
                        **{
                            f"runtime_override/{key.replace('.', '_')}": new
                            for key, (_, new) in changes.items()
                        },
                    }
                )

        return next_log_time_s, next_checkpoint_time_s

    def _export_rust_inference_artifacts(
        self,
        model: TetrisNet,
        onnx_path: Path,
        *,
        export_name: str,
    ) -> float:
        export_start = time.perf_counter()
        full_export_ok = export_onnx(model, onnx_path)
        if not full_export_ok:
            raise RuntimeError(
                f"{export_name} ONNX export failed due to missing dependencies"
            )
        split_export_ok = export_split_models(model, onnx_path)
        if not split_export_ok:
            raise RuntimeError(
                f"{export_name} split-model export failed due to missing dependencies"
            )
        assert_rust_inference_artifacts(onnx_path)
        return 1000.0 * (time.perf_counter() - export_start)

    def _candidate_gate_timing_config(self) -> tuple[float, float, float | None]:
        base_interval_seconds = self._model_sync_interval_seconds()
        failure_backoff_seconds = self.config.run.model_sync_failure_backoff_seconds
        max_interval_seconds = self.config.run.model_sync_max_interval_seconds
        if failure_backoff_seconds < 0.0:
            raise ValueError(
                "config.run.model_sync_failure_backoff_seconds must be >= 0 "
                f"(got {failure_backoff_seconds})"
            )
        if max_interval_seconds < 0.0:
            raise ValueError(
                "config.run.model_sync_max_interval_seconds must be >= 0 "
                f"(got {max_interval_seconds})"
            )
        if 0.0 < max_interval_seconds < base_interval_seconds:
            raise ValueError(
                "config.run.model_sync_max_interval_seconds must be 0 or >= "
                "config.run.model_sync_interval_seconds "
                f"(got max={max_interval_seconds}, base={base_interval_seconds})"
            )
        resolved_max_interval_seconds = (
            max_interval_seconds if max_interval_seconds > 0.0 else None
        )
        return (
            base_interval_seconds,
            failure_backoff_seconds,
            resolved_max_interval_seconds,
        )

    def _initialize_candidate_gate_schedule(
        self,
        *,
        now_s: float,
    ) -> CandidateGateSchedule:
        (
            base_interval_seconds,
            failure_backoff_seconds,
            max_interval_seconds,
        ) = self._candidate_gate_timing_config()
        failed_promotion_streak = self.initial_candidate_gate_failed_promotion_streak
        restored_interval_seconds = self.initial_candidate_gate_interval_seconds
        if restored_interval_seconds is None:
            current_interval_seconds = _candidate_gate_interval_seconds(
                base_interval_seconds=base_interval_seconds,
                failure_backoff_seconds=failure_backoff_seconds,
                failed_promotion_streak=failed_promotion_streak,
                max_interval_seconds=max_interval_seconds,
            )
        else:
            current_interval_seconds = restored_interval_seconds
        next_export_delay_seconds = (
            self.initial_candidate_gate_next_export_delay_seconds
        )
        next_export_time_s = (
            now_s + current_interval_seconds
            if next_export_delay_seconds is None
            else now_s + next_export_delay_seconds
        )
        return CandidateGateSchedule(
            current_interval_seconds=current_interval_seconds,
            failed_promotion_streak=failed_promotion_streak,
            next_export_time_s=next_export_time_s,
        )

    def _candidate_gate_checkpoint_state(
        self,
        candidate_gate_schedule: CandidateGateSchedule | None,
        *,
        now_s: float,
    ) -> dict[str, object]:
        if candidate_gate_schedule is None:
            return {}
        next_export_delay_seconds = max(
            0.0, candidate_gate_schedule.next_export_time_s - now_s
        )
        return {
            "candidate_gate_current_interval_seconds": (
                candidate_gate_schedule.current_interval_seconds
            ),
            "candidate_gate_failed_promotion_streak": (
                candidate_gate_schedule.failed_promotion_streak
            ),
            "candidate_gate_next_export_delay_seconds": next_export_delay_seconds,
        }

    def _runtime_override_checkpoint_state(self) -> dict[str, object]:
        runtime_overrides = self._current_runtime_override_state()
        return {
            "runtime_override_lr_multiplier": (
                runtime_overrides.optimizer.lr_multiplier
            ),
            "runtime_override_grad_clip_norm": (
                runtime_overrides.optimizer.grad_clip_norm
            ),
            "runtime_override_weight_decay": runtime_overrides.optimizer.weight_decay,
            "runtime_override_mirror_augmentation_probability": (
                runtime_overrides.optimizer.mirror_augmentation_probability
            ),
            "runtime_override_log_interval_seconds": (
                runtime_overrides.run.log_interval_seconds
            ),
            "runtime_override_checkpoint_interval_seconds": (
                runtime_overrides.run.checkpoint_interval_seconds
            ),
            "runtime_override_add_noise": runtime_overrides.self_play.add_noise,
            "runtime_override_visit_sampling_epsilon": (
                runtime_overrides.self_play.visit_sampling_epsilon
            ),
        }

    def _update_candidate_gate_schedule_from_eval(
        self,
        candidate_gate_schedule: CandidateGateSchedule,
        *,
        evaluation_seconds: float,
        promoted: bool,
        now_s: float,
    ) -> None:
        if evaluation_seconds < 0.0:
            raise ValueError(
                f"evaluation_seconds must be >= 0 (got {evaluation_seconds})"
            )
        (
            base_interval_seconds,
            failure_backoff_seconds,
            max_interval_seconds,
        ) = self._candidate_gate_timing_config()
        failed_promotion_streak = (
            0 if promoted else candidate_gate_schedule.failed_promotion_streak + 1
        )
        candidate_gate_schedule.failed_promotion_streak = failed_promotion_streak
        candidate_gate_schedule.current_interval_seconds = (
            _candidate_gate_interval_seconds(
                base_interval_seconds=base_interval_seconds,
                failure_backoff_seconds=failure_backoff_seconds,
                failed_promotion_streak=failed_promotion_streak,
                max_interval_seconds=max_interval_seconds,
            )
        )
        evaluation_start_time_s = now_s - evaluation_seconds
        candidate_gate_schedule.next_export_time_s = (
            evaluation_start_time_s + candidate_gate_schedule.current_interval_seconds
        )

    def _defer_candidate_gate_export(
        self,
        candidate_gate_schedule: CandidateGateSchedule,
        *,
        now_s: float,
    ) -> None:
        candidate_gate_schedule.next_export_time_s = (
            now_s + candidate_gate_schedule.current_interval_seconds
        )

    def _drain_model_eval_events(
        self,
        generator: GameGenerator,
        *,
        log_to_wandb: bool,
        now_s: float,
        candidate_gate_schedule: CandidateGateSchedule,
    ) -> None:
        for event in generator.drain_model_eval_events():
            promoted = bool(event["promoted"])
            evaluation_seconds = float(event["evaluation_seconds"])
            self._update_candidate_gate_schedule_from_eval(
                candidate_gate_schedule,
                evaluation_seconds=evaluation_seconds,
                promoted=promoted,
                now_s=now_s,
            )
            if promoted:
                self._publish_incumbent_to_r2_if_enabled(generator, reason="promotion")
            next_candidate_export_delay_seconds = max(
                0.0, candidate_gate_schedule.next_export_time_s - now_s
            )
            candidate_nn_value_weight = float(event["candidate_nn_value_weight"])
            incumbent_nn_value_weight = float(event["incumbent_nn_value_weight"])
            promoted_nn_value_weight = float(event["promoted_nn_value_weight"])
            promoted_death_penalty = float(event["promoted_death_penalty"])
            promoted_overhang_penalty_weight = float(
                event["promoted_overhang_penalty_weight"]
            )
            per_game_prediction_metrics = event["per_game_prediction_metrics"]
            prediction_metric_rows = [
                row for row in per_game_prediction_metrics if int(row[4]) > 0
            ]
            candidate_trajectory_predicted_total_attack_variance = (
                sum(float(row[1]) for row in prediction_metric_rows)
                / len(prediction_metric_rows)
                if prediction_metric_rows
                else 0.0
            )
            candidate_trajectory_predicted_total_attack_std = (
                sum(float(row[2]) for row in prediction_metric_rows)
                / len(prediction_metric_rows)
                if prediction_metric_rows
                else 0.0
            )
            candidate_trajectory_predicted_total_attack_rmse = (
                sum(float(row[3]) for row in prediction_metric_rows)
                / len(prediction_metric_rows)
                if prediction_metric_rows
                else 0.0
            )
            logger.info(
                "Model evaluation decision",
                trainer_step=self.step,
                candidate_step=int(event["candidate_step"]),
                candidate_games=int(event["candidate_games"]),
                candidate_avg_attack=event["candidate_avg_attack"],
                candidate_attack_variance=event["candidate_attack_variance"],
                candidate_nn_value_weight=candidate_nn_value_weight,
                incumbent_step=int(event["incumbent_step"]),
                incumbent_uses_network=bool(event["incumbent_uses_network"]),
                incumbent_avg_attack=event["incumbent_avg_attack"],
                incumbent_nn_value_weight=incumbent_nn_value_weight,
                candidate_trajectory_predicted_total_attack_variance=(
                    candidate_trajectory_predicted_total_attack_variance
                ),
                candidate_trajectory_predicted_total_attack_std=(
                    candidate_trajectory_predicted_total_attack_std
                ),
                candidate_trajectory_predicted_total_attack_rmse=(
                    candidate_trajectory_predicted_total_attack_rmse
                ),
                promoted_nn_value_weight=promoted_nn_value_weight,
                promoted_death_penalty=promoted_death_penalty,
                promoted_overhang_penalty_weight=promoted_overhang_penalty_weight,
                promoted=promoted,
                auto_promoted=bool(event["auto_promoted"]),
                force_promoted=bool(event["force_promoted"]),
                evaluation_seconds=evaluation_seconds,
                candidate_gate_failed_promotion_streak=(
                    candidate_gate_schedule.failed_promotion_streak
                ),
                candidate_gate_interval_seconds=(
                    candidate_gate_schedule.current_interval_seconds
                ),
                next_candidate_export_delay_seconds=(
                    next_candidate_export_delay_seconds
                ),
            )
            worst_tree_path = event.get("worst_game_tree_path")
            if worst_tree_path:
                logger.info(
                    "Saved worst candidate eval tree playback",
                    candidate_step=event["candidate_step"],
                    path=worst_tree_path,
                )
            if not log_to_wandb:
                continue

            wall_time_hours = self._cumulative_wall_time_hours(now_s)
            wandb_data: dict[str, object] = {
                "trainer_step": self.step,
                "wall_time_hours": wall_time_hours,
                "model_gate/candidate_step": event["candidate_step"],
                "model_gate/candidate_games": event["candidate_games"],
                "model_gate/candidate_avg_attack": event["candidate_avg_attack"],
                "model_gate_time/candidate_avg_attack": event["candidate_avg_attack"],
                "model_gate/candidate_attack_variance": event[
                    "candidate_attack_variance"
                ],
                "model_gate/candidate_nn_value_weight": candidate_nn_value_weight,
                "model_gate/incumbent_step": event["incumbent_step"],
                "model_gate/incumbent_uses_network": event["incumbent_uses_network"],
                "model_gate/incumbent_avg_attack": event["incumbent_avg_attack"],
                "model_gate/incumbent_nn_value_weight": incumbent_nn_value_weight,
                "model_gate/candidate_trajectory_predicted_total_attack_variance": (
                    candidate_trajectory_predicted_total_attack_variance
                ),
                "model_gate/candidate_trajectory_predicted_total_attack_std": (
                    candidate_trajectory_predicted_total_attack_std
                ),
                "model_gate/candidate_trajectory_predicted_total_attack_rmse": (
                    candidate_trajectory_predicted_total_attack_rmse
                ),
                "model_gate/promoted_nn_value_weight": promoted_nn_value_weight,
                "model_gate/promoted_death_penalty": promoted_death_penalty,
                "model_gate/promoted_overhang_penalty_weight": (
                    promoted_overhang_penalty_weight
                ),
                "model_gate/promoted": event["promoted"],
                "model_gate/auto_promoted": event["auto_promoted"],
                "model_gate/force_promoted": event["force_promoted"],
                "model_gate/evaluation_seconds": evaluation_seconds,
                "model_gate/failed_promotion_streak": (
                    candidate_gate_schedule.failed_promotion_streak
                ),
                "model_gate/current_export_interval_seconds": (
                    candidate_gate_schedule.current_interval_seconds
                ),
                "model_gate/next_export_delay_seconds": (
                    next_candidate_export_delay_seconds
                ),
            }

            per_game_results = event["per_game_results"]
            if per_game_results:
                num_games = len(per_game_results)
                total_attack = sum(r[1] for r in per_game_results)
                total_lines = sum(r[2] for r in per_game_results)
                total_moves = sum(r[3] for r in per_game_results)
                wandb_data["eval/num_games"] = num_games
                wandb_data["eval/avg_attack"] = total_attack / num_games
                wandb_data["eval/max_attack"] = max(r[1] for r in per_game_results)
                wandb_data["eval/avg_lines"] = total_lines / num_games
                wandb_data["eval/max_lines"] = max(r[2] for r in per_game_results)
                wandb_data["eval/avg_moves"] = total_moves / num_games
                wandb_data["eval/attack_per_piece"] = (
                    total_attack / total_moves if total_moves > 0 else 0.0
                )
                wandb_data["eval/lines_per_piece"] = (
                    total_lines / total_moves if total_moves > 0 else 0.0
                )
                if prediction_metric_rows:
                    wandb_data["eval/trajectory_predicted_total_attack_variance"] = (
                        candidate_trajectory_predicted_total_attack_variance
                    )
                    wandb_data["eval/trajectory_predicted_total_attack_std"] = (
                        candidate_trajectory_predicted_total_attack_std
                    )
                    wandb_data["eval/trajectory_predicted_total_attack_rmse"] = (
                        candidate_trajectory_predicted_total_attack_rmse
                    )
                wandb_data["eval/nn_value_weight"] = promoted_nn_value_weight

            best_replay = event.get("best_game_replay")
            worst_replay = event.get("worst_game_replay")
            if best_replay is not None:
                frames = render_replay(best_replay)
                video, _ = self._create_wandb_gif_video(
                    frames,
                    attack=best_replay.total_attack,
                    gif_stem=(
                        f"eval_best_step{self.step}_attack{best_replay.total_attack}"
                    ),
                )
                if video is not None:
                    wandb_data["eval/best_trajectory"] = video
            if worst_replay is not None:
                frames = render_replay(worst_replay)
                video, _ = self._create_wandb_gif_video(
                    frames,
                    attack=worst_replay.total_attack,
                    gif_stem=(
                        f"eval_worst_step{self.step}_attack{worst_replay.total_attack}"
                    ),
                )
                if video is not None:
                    wandb_data["eval/worst_trajectory"] = video

            wandb.log(wandb_data)

    def _build_generator_mcts_config(self) -> MCTSConfig:
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.config.self_play.num_simulations
        mcts_config.c_puct = self.config.self_play.c_puct
        mcts_config.temperature = self.config.self_play.temperature
        mcts_config.dirichlet_alpha = self.config.self_play.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.config.self_play.dirichlet_epsilon
        mcts_config.visit_sampling_epsilon = (
            self.config.self_play.visit_sampling_epsilon
        )
        mcts_config.seed = self.config.self_play.mcts_seed
        mcts_config.max_placements = self.config.self_play.max_placements
        mcts_config.death_penalty = self.config.self_play.penalty_schedule.death_penalty
        mcts_config.overhang_penalty_weight = (
            self.config.self_play.penalty_schedule.overhang_penalty_weight
        )
        mcts_config.nn_value_weight = (
            self.config.self_play.nn_value_weight_schedule.initial
        )
        mcts_config.use_parent_value_for_unvisited_q = (
            self.config.self_play.use_parent_value_for_unvisited_q
        )
        mcts_config.reuse_tree = self.config.self_play.reuse_tree
        return mcts_config

    def _effective_starting_incumbent_penalties(self) -> tuple[float, float]:
        return scaled_penalties(
            self.config.self_play.penalty_schedule,
            cumulative_games=self._cumulative_games_offset,
            nn_value_weight=self.config.self_play.nn_value_weight_schedule.initial,
            nn_value_weight_cap=self.config.self_play.nn_value_weight_schedule.cap,
        )

    def _scaled_penalties_for(
        self, generator: GameGenerator, *, nn_value_weight: float
    ) -> tuple[float, float]:
        return scaled_penalties(
            self.config.self_play.penalty_schedule,
            cumulative_games=self._cumulative_games_generated(generator),
            nn_value_weight=nn_value_weight,
            nn_value_weight_cap=self.config.self_play.nn_value_weight_schedule.cap,
        )

    def _evaluate_starting_incumbent_avg_attack(self, model_path: Path) -> float:
        eval_config = self._build_generator_mcts_config()
        eval_config.seed = 0
        eval_config.visit_sampling_epsilon = 0.0
        (
            eval_config.death_penalty,
            eval_config.overhang_penalty_weight,
        ) = self._effective_starting_incumbent_penalties()
        eval_seeds = list(range(self.config.self_play.model_promotion_eval_games))
        eval_workers = max(2, self.config.self_play.num_workers)
        logger.info(
            "Evaluating starting incumbent baseline for resumed run",
            model_path=str(model_path),
            num_games=len(eval_seeds),
            num_workers=eval_workers,
            nn_value_weight=eval_config.nn_value_weight,
            death_penalty=eval_config.death_penalty,
            overhang_penalty_weight=eval_config.overhang_penalty_weight,
        )
        eval_result = evaluate_model(
            str(model_path),
            eval_seeds,
            eval_config,
            self.config.self_play.max_placements,
            eval_workers,
            False,
        )
        logger.info(
            "Evaluated starting incumbent baseline for resumed run",
            model_path=str(model_path),
            avg_attack=eval_result.avg_attack,
            max_attack=eval_result.max_attack,
            num_games=eval_result.num_games,
        )
        return float(eval_result.avg_attack)

    def _create_scheduler(self):
        schedule = self.config.optimizer.lr_schedule
        if schedule not in {"linear", "cosine", "warmup_cosine", "step"}:
            return None

        def build_for(inner_optimizer: torch.optim.Optimizer):
            if schedule == "linear":
                return torch.optim.lr_scheduler.LinearLR(
                    inner_optimizer,
                    start_factor=1.0,
                    end_factor=self.config.optimizer.lr_min_factor,
                    total_iters=self.config.optimizer.lr_decay_steps,
                )
            if schedule == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    inner_optimizer,
                    T_max=self.config.optimizer.lr_decay_steps,
                    eta_min=self.config.optimizer.learning_rate
                    * self.config.optimizer.lr_min_factor,
                )
            if schedule == "warmup_cosine":
                return torch.optim.lr_scheduler.LambdaLR(
                    inner_optimizer,
                    lr_lambda=self._warmup_cosine_lr_lambda,
                )
            return torch.optim.lr_scheduler.StepLR(
                inner_optimizer,
                step_size=self.config.optimizer.lr_decay_steps
                // self.config.optimizer.lr_step_divisor,
                gamma=self.config.optimizer.lr_step_gamma,
            )

        return SchedulerBundle(
            tuple(build_for(opt) for opt in self.optimizer.inner_optimizers)
        )

    def _warmup_cosine_lr_lambda(self, step: int) -> float:
        return _compute_warmup_cosine_factor(
            step=step,
            warmup_steps=self.config.optimizer.lr_warmup_steps,
            decay_steps=self.config.optimizer.lr_decay_steps,
            warmup_start_factor=self.config.optimizer.lr_warmup_start_factor,
            min_factor=self.config.optimizer.lr_min_factor,
        )

    def align_scheduler_to_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0 (got {step})")
        if self.scheduler is None:
            return

        # Rebuild scheduler state from current config so resumed runs use the new
        # LR settings while keeping global step alignment.
        self.scheduler.last_epoch = step

        first_scheduler = self.scheduler.first
        if self.config.optimizer.lr_schedule == "linear":
            assert isinstance(first_scheduler, torch.optim.lr_scheduler.LinearLR)
            total_iters = self.config.optimizer.lr_decay_steps
            progress = min(step, total_iters) / total_iters
            factor = 1.0 + (self.config.optimizer.lr_min_factor - 1.0) * progress
            lrs = [base_lr * factor for base_lr in self.scheduler.base_lrs]
        elif self.config.optimizer.lr_schedule == "cosine":
            assert isinstance(
                first_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            )
            t_max = self.config.optimizer.lr_decay_steps
            eta_min = (
                self.config.optimizer.learning_rate
                * self.config.optimizer.lr_min_factor
            )
            cosine_factor = (
                1 + torch.cos(torch.tensor(torch.pi * step / t_max))
            ).item() / 2
            lrs = [
                eta_min + (base_lr - eta_min) * cosine_factor
                for base_lr in self.scheduler.base_lrs
            ]
        elif self.config.optimizer.lr_schedule == "warmup_cosine":
            assert isinstance(first_scheduler, torch.optim.lr_scheduler.LambdaLR)
            factor = _compute_warmup_cosine_factor(
                step=step,
                warmup_steps=self.config.optimizer.lr_warmup_steps,
                decay_steps=self.config.optimizer.lr_decay_steps,
                warmup_start_factor=self.config.optimizer.lr_warmup_start_factor,
                min_factor=self.config.optimizer.lr_min_factor,
            )
            lrs = [base_lr * factor for base_lr in self.scheduler.base_lrs]
        elif self.config.optimizer.lr_schedule == "step":
            assert isinstance(first_scheduler, torch.optim.lr_scheduler.StepLR)
            step_size = (
                self.config.optimizer.lr_decay_steps
                // self.config.optimizer.lr_step_divisor
            )
            decay = self.config.optimizer.lr_step_gamma ** (step // step_size)
            lrs = [base_lr * decay for base_lr in self.scheduler.base_lrs]
        else:
            raise ValueError(
                f"Unsupported lr_schedule: {self.config.optimizer.lr_schedule}"
            )

        if len(self.optimizer.param_groups) != len(lrs):
            raise ValueError(
                "Optimizer param group count does not match computed LR count"
            )
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self.scheduler._last_lr = lrs
        self._scheduler_base_lrs = list(lrs)
        if self._lr_multiplier != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self._lr_multiplier

    @staticmethod
    def _compute_candidate_nn_value_weight(
        current_weight: float,
        config: TrainingConfig,
        *,
        cumulative_games: int = 0,
    ) -> float:
        return compute_nn_value_weight(
            config.self_play.nn_value_weight_schedule,
            current_weight=current_weight,
            cumulative_games=cumulative_games,
        )

    def _build_training_batch(
        self,
        sample: tuple,
    ) -> TrainingBatch:
        (
            boards,
            aux,
            policy_targets,
            value_targets,
            overhang_fields,
            masks,
        ) = sample
        return TrainingBatch(
            boards=torch.from_numpy(boards).reshape(-1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            aux=torch.from_numpy(aux),
            policy_targets=torch.from_numpy(policy_targets),
            value_targets=torch.from_numpy(value_targets),
            overhang_fields=torch.from_numpy(overhang_fields),
            masks=torch.from_numpy(masks),
        )

    def _pin_batch_if_needed(self, batch: TrainingBatch) -> TrainingBatch:
        should_pin = (
            self.device.type == "cuda"
            and self.config.replay.pin_memory_batches
            and batch.device.type == "cpu"
        )
        if not should_pin:
            return batch
        return TrainingBatch(
            boards=batch.boards.pin_memory(),
            aux=batch.aux.pin_memory(),
            policy_targets=batch.policy_targets.pin_memory(),
            value_targets=batch.value_targets.pin_memory(),
            overhang_fields=batch.overhang_fields.pin_memory(),
            masks=batch.masks.pin_memory(),
        )

    def _is_batch_on_training_device(self, batch: TrainingBatch) -> bool:
        batch_device = batch.device
        if batch_device.type != self.device.type:
            return False
        if self.device.type != "cuda":
            return True
        if self.device.index is None:
            return True
        return batch_device.index == self.device.index

    def _to_training_device(self, batch: TrainingBatch) -> TrainingBatch:
        if self._is_batch_on_training_device(batch):
            return batch
        batch = self._pin_batch_if_needed(batch)
        non_blocking = self.device.type == "cuda"
        return TrainingBatch(
            boards=batch.boards.to(self.device, non_blocking=non_blocking),
            aux=batch.aux.to(self.device, non_blocking=non_blocking),
            policy_targets=batch.policy_targets.to(
                self.device, non_blocking=non_blocking
            ),
            value_targets=batch.value_targets.to(
                self.device, non_blocking=non_blocking
            ),
            overhang_fields=batch.overhang_fields.to(
                self.device, non_blocking=non_blocking
            ),
            masks=batch.masks.to(self.device, non_blocking=non_blocking),
        )

    def _training_batch_bytes(self, batch: TrainingBatch) -> int:
        tensors = (
            batch.boards,
            batch.aux,
            batch.policy_targets,
            batch.value_targets,
            batch.overhang_fields,
            batch.masks,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @staticmethod
    def _tensor_field_pairs(
        mirror: CircularReplayMirror,
        batch: TrainingBatch,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (mirror.boards, batch.boards),
            (mirror.aux, batch.aux),
            (mirror.policy_targets, batch.policy_targets),
            (mirror.value_targets, batch.value_targets),
            (mirror.overhang_fields, batch.overhang_fields),
            (mirror.masks, batch.masks),
        ]

    def _write_to_mirror(
        self,
        mirror: CircularReplayMirror,
        batch: TrainingBatch,
    ) -> None:
        n = batch.size
        if n == 0:
            return
        if n > mirror.capacity:
            offset = n - mirror.capacity
            batch = TrainingBatch(
                boards=batch.boards[offset:],
                aux=batch.aux[offset:],
                policy_targets=batch.policy_targets[offset:],
                value_targets=batch.value_targets[offset:],
                overhang_fields=batch.overhang_fields[offset:],
                masks=batch.masks[offset:],
            )
            n = mirror.capacity

        end_pos = mirror.write_pos + n
        pairs = self._tensor_field_pairs(mirror, batch)
        if end_pos <= mirror.capacity:
            s = slice(mirror.write_pos, end_pos)
            for dst, src in pairs:
                dst[s].copy_(src)
        else:
            tail = mirror.capacity - mirror.write_pos
            head = n - tail
            for dst, src in pairs:
                dst[mirror.write_pos :].copy_(src[:tail])
                dst[:head].copy_(src[tail:])

        mirror.write_pos = (mirror.write_pos + n) % mirror.capacity
        mirror.count = min(mirror.count + n, mirror.capacity)

    def _sample_prefetched_batches(
        self,
        generator: GameGenerator,
        staged_batch_size: int,
    ) -> list[TrainingBatch] | None:
        result = generator.sample_batch(staged_batch_size)
        if result is None:
            return None
        staged_batch = self._to_training_device(self._build_training_batch(result))
        return staged_batch.split(self.config.optimizer.batch_size)

    def _use_device_replay_mirror(self) -> bool:
        return (
            self.config.replay.mirror_replay_on_accelerator
            and self.device.type != "cpu"
        )

    def _load_replay_mirror(
        self,
        generator: GameGenerator,
        mirror: CircularReplayMirror | None = None,
    ) -> CircularReplayMirror | None:
        if mirror is None:
            mirror = CircularReplayMirror(self.config.replay.buffer_size, self.device)
        mirror.count = 0
        mirror.write_pos = 0
        mirror.logical_end = 0
        # Bootstrap from bounded replay deltas so a resumed full buffer does not
        # stage the entire replay window on the training device at once.
        return self._refresh_replay_mirror(generator, mirror)

    def _refresh_replay_mirror(
        self,
        generator: GameGenerator,
        mirror: CircularReplayMirror | None,
    ) -> CircularReplayMirror | None:
        if mirror is None:
            return self._load_replay_mirror(generator)

        delta_examples_total = 0
        delta_bytes_total = 0
        while True:
            result = generator.replay_buffer_delta(
                mirror.logical_end,
                self.config.replay.replay_mirror_delta_chunk_examples,
            )
            if result is None:
                return None
            (
                window_start,
                window_end,
                slice_start,
                boards,
                aux,
                policy_targets,
                value_targets,
                overhang_fields,
                masks,
            ) = result
            window_start = int(window_start)
            window_end = int(window_end)
            slice_start = int(slice_start)
            if slice_start < mirror.logical_end:
                logger.info(
                    "Replay mirror stale; loading full snapshot",
                    mirror_logical_end=mirror.logical_end,
                    window_start_index=window_start,
                    window_end_index=window_end,
                    slice_start_index=slice_start,
                )
                return self._load_replay_mirror(generator, mirror)
            if window_start > mirror.logical_end:
                if mirror.count > 0:
                    logger.info(
                        "Replay mirror fully evicted; rebasing",
                        mirror_logical_end=mirror.logical_end,
                        window_start_index=window_start,
                        window_end_index=window_end,
                    )
                mirror.count = 0
                mirror.write_pos = 0
                mirror.logical_end = window_start

            delta_batch = self._to_training_device(
                self._build_training_batch(
                    (boards, aux, policy_targets, value_targets, overhang_fields, masks)
                )
            )
            if delta_batch.size > 0:
                delta_examples_total += delta_batch.size
                delta_bytes_total += self._training_batch_bytes(delta_batch)
                self._write_to_mirror(mirror, delta_batch)
                mirror.logical_end += delta_batch.size

            if mirror.logical_end >= window_end:
                if delta_examples_total > 0:
                    logger.info(
                        "Synchronized replay mirror from deltas",
                        added_examples=delta_examples_total,
                        delta_gb=(delta_bytes_total / (1024.0 * 1024.0 * 1024.0)),
                        mirror_logical_end=mirror.logical_end,
                        mirror_examples=mirror.count,
                        window_start_index=window_start,
                        window_end_index=window_end,
                    )
                return mirror

    def _sample_from_replay_mirror(self, mirror: CircularReplayMirror) -> TrainingBatch:
        if mirror.count <= 0:
            raise ValueError("Replay mirror is empty")
        sample_indices = torch.randint(
            low=0,
            high=mirror.count,
            size=(self.config.optimizer.batch_size,),
            device=mirror.boards.device,
        )
        return TrainingBatch(
            boards=mirror.boards.index_select(0, sample_indices),
            aux=mirror.aux.index_select(0, sample_indices),
            policy_targets=mirror.policy_targets.index_select(0, sample_indices),
            value_targets=mirror.value_targets.index_select(0, sample_indices),
            overhang_fields=mirror.overhang_fields.index_select(0, sample_indices),
            masks=mirror.masks.index_select(0, sample_indices),
        )

    def train_step(
        self, batch: TrainingBatch, collect_metrics: bool
    ) -> dict[str, float]:
        """Execute one training step."""
        self.model.train()

        batch = self._to_training_device(batch)
        boards = batch.boards
        aux = batch.aux
        policy_targets = batch.policy_targets
        value_targets = batch.value_targets
        overhang_fields = batch.overhang_fields
        masks = batch.masks

        # Forward + backward
        self.optimizer.zero_grad(set_to_none=True)
        value_loss_weight = self._cached_value_loss_weight
        total_loss, policy_loss, value_loss = compute_loss(
            self.model,
            boards,
            aux,
            policy_targets,
            value_targets,
            masks,
            value_loss_weight,
        )
        total_loss.backward()

        # Gradient clipping. Treat <=0 as disabled; passing 0.0 to
        # clip_grad_norm_ would zero the gradients and freeze training.
        grad_clip_norm = self.config.optimizer.grad_clip_norm
        if grad_clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), grad_clip_norm
            )
        else:
            grad_norm = None

        normalized_steps = sanitize_optimizer_state_steps(self.optimizer)
        if normalized_steps > 0 and not self._logged_live_optimizer_step_sanitization:
            logger.warning(
                "Sanitized live optimizer step counters before optimizer step",
                normalized_steps=normalized_steps,
            )
            self._logged_live_optimizer_step_sanitization = True

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self._export_model)
        self._step_scheduler()

        if not collect_metrics:
            return {}

        # Only sync GPU and update loss balancer when collecting metrics; fetch
        # all scalars in one device->host transfer.
        loss_scalars = [total_loss.detach(), policy_loss.detach(), value_loss.detach()]
        if grad_norm is not None:
            loss_scalars.append(grad_norm)
        loss_values = torch.stack(loss_scalars).tolist()
        total_loss_scalar, policy_loss_scalar, value_loss_scalar = loss_values[:3]
        grad_norm_scalar = loss_values[3] if grad_norm is not None else float("nan")
        self.loss_balancer.append(policy_loss_scalar, value_loss_scalar)
        if self.loss_balancer.has_history():
            self._cached_value_loss_weight = (
                self.loss_balancer.value_loss_weight()
                / self.config.optimizer.policy_loss_scale
            )

        policy_loss_avg, value_loss_avg = self.loss_balancer.averages()
        metrics = {
            "train/loss": total_loss_scalar,
            "train/policy_loss": policy_loss_scalar,
            "train/value_loss": value_loss_scalar,
            "train/policy_loss_avg": policy_loss_avg,
            "train/value_loss_avg": value_loss_avg,
            "train/value_loss_weight": value_loss_weight,
            "train/grad_norm": grad_norm_scalar,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        metrics.update(
            compute_batch_feature_metrics(
                aux=aux,
                value_targets=value_targets,
                overhang_fields=overhang_fields,
                masks=masks,
            )
        )
        return metrics

    def _compute_extra_train_metrics(self, batch: TrainingBatch) -> dict:
        batch = self._to_training_device(batch)
        return compute_metrics(
            self.model,
            batch.boards,
            batch.aux,
            batch.policy_targets,
            batch.value_targets,
            batch.masks,
        )

    def _init_r2_sync(self, generator: GameGenerator) -> None:
        """Resolve R2 settings and start the trainer-side downloaders.

        Gated by `r2_sync.enabled`. The threads started here ingest replay
        chunks and game stats from remote generators; the trainer pushes
        its own incumbent bundle through
        `_publish_incumbent_to_r2_if_enabled` separately.

        `sync_run_id` defaults to the run dir's basename — i.e. the
        `<adjective>-<animal>-<timestamp>` id assigned when the run was
        created. That keeps the same id stable across resumes without
        requiring any config edits.
        """
        if not self.config.r2_sync.enabled:
            return
        if self.config.run.run_dir is None:
            return
        default_run_id = self.config.run.run_dir.name
        try:
            self._r2_settings = R2Settings.from_config(
                self.config.r2_sync, default_run_id=default_run_id
            )
        except ValueError as error:
            logger.warning("trainer.r2_sync_disabled", error=str(error))
            self._r2_settings = None
            return
        cursor_path = self.config.run.run_dir / "r2_ingest_cursor.json"
        game_stats_cursor_path = (
            self.config.run.run_dir / "r2_game_stats_ingest_cursor.json"
        )
        offset_table_path = self.config.run.run_dir / "r2_machine_id_offsets.json"
        offset_table = MachineOffsetTable(offset_table_path)
        self._r2_chunk_downloader = ChunkDownloader(
            generator=generator,
            settings=self._r2_settings,
            cursor_path=cursor_path,
            poll_interval_seconds=(
                self.config.r2_sync.chunk_download_poll_interval_seconds
            ),
            offset_table=offset_table,
        )
        self._r2_chunk_downloader.start()
        self._r2_game_stats_downloader = GameStatsDownloader(
            sink=self,
            settings=self._r2_settings,
            cursor_path=game_stats_cursor_path,
            poll_interval_seconds=(
                self.config.r2_sync.chunk_download_poll_interval_seconds
            ),
            offset_table=offset_table,
        )
        self._r2_game_stats_downloader.start()
        self._republish_self_play_snapshot_to_r2()
        logger.info(
            "trainer.r2_sync_initialized",
            sync_run_id=self._r2_settings.sync_run_id,
            bucket=self._r2_settings.bucket,
            endpoint_url=self._r2_settings.endpoint_url,
        )

    def _republish_self_play_snapshot_to_r2(self) -> None:
        """Push the current `self_play` snapshot to R2 if R2 is enabled.

        Called at startup and whenever runtime overrides change a field
        that's part of `SelfPlaySnapshot`. Generators poll this snapshot
        and apply any changes to their local `MCTSConfig` so cross-machine
        search settings stay in sync without restart.
        """
        if self._r2_settings is None:
            return
        snapshot = SelfPlaySnapshot.from_self_play(self.config.self_play)
        try:
            upload_self_play_snapshot(settings=self._r2_settings, snapshot=snapshot)
        except Exception:
            logger.exception("trainer.r2_self_play_snapshot_upload_failed")

    def _shutdown_r2_sync(self) -> None:
        chunk_downloader = self._r2_chunk_downloader
        game_stats_downloader = self._r2_game_stats_downloader
        if chunk_downloader is not None:
            try:
                chunk_downloader.stop()
            except Exception:
                logger.exception("trainer.r2_chunk_downloader_stop_failed")
        if game_stats_downloader is not None:
            try:
                game_stats_downloader.stop()
            except Exception:
                logger.exception("trainer.r2_game_stats_downloader_stop_failed")
        self._r2_chunk_downloader = None
        self._r2_game_stats_downloader = None

    def _upload_to_r2_if_enabled(
        self,
        onnx_path: Path,
        model_step: int,
        nn_value_weight: float,
        death_penalty: float,
        overhang_penalty_weight: float,
    ) -> None:
        if self._r2_settings is None:
            return
        if model_step <= self._r2_last_uploaded_step:
            return
        if not onnx_path.exists():
            logger.warning(
                "trainer.r2_upload_skipped_missing_bundle",
                path=str(onnx_path),
                step=model_step,
            )
            return
        try:
            upload_model_bundle(
                settings=self._r2_settings,
                onnx_path=onnx_path,
                step=model_step,
                nn_value_weight=nn_value_weight,
                death_penalty=death_penalty,
                overhang_penalty_weight=overhang_penalty_weight,
            )
            self._r2_last_uploaded_step = model_step
        except Exception:
            logger.exception(
                "trainer.r2_upload_failed",
                path=str(onnx_path),
                step=model_step,
            )

    def _publish_incumbent_to_r2_if_enabled(
        self, generator: GameGenerator, *, reason: str
    ) -> None:
        """Push the current incumbent bundle + pointer to R2 immediately.

        Why: remote generators block on `models/incumbent.json` to start
        playing. Without this hook the trainer would only publish at the
        checkpoint cadence (every few hours) or shutdown, leaving generators
        idle. Called at startup and after every promotion.
        """
        if self._r2_settings is None:
            return
        if not generator.incumbent_uses_network():
            return
        try:
            self._persist_incumbent_model_artifacts(generator)
        except Exception:
            logger.exception(
                "trainer.r2_incumbent_publish_failed",
                reason=reason,
                step=generator.incumbent_model_step(),
            )

    def _persist_incumbent_model_artifacts(
        self, generator: GameGenerator
    ) -> tuple[Path | None, str]:
        if not generator.incumbent_uses_network():
            source_path_string = generator.incumbent_model_path()
            return None, source_path_string
        if self.config.run.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is not set on training config")
        destination_path = self.config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
        source_path_string = ""
        for attempt in range(2):
            source_path = Path(generator.incumbent_model_path())
            source_path_string = str(source_path)
            try:
                copy_model_artifact_bundle(source_path, destination_path)
                self._upload_to_r2_if_enabled(
                    destination_path,
                    generator.incumbent_model_step(),
                    generator.incumbent_nn_value_weight(),
                    generator.incumbent_death_penalty(),
                    generator.incumbent_overhang_penalty_weight(),
                )
                return destination_path, source_path_string
            except (FileNotFoundError, RuntimeError) as error:
                latest_source_path = Path(generator.incumbent_model_path())
                if attempt == 0 and latest_source_path != source_path:
                    logger.warning(
                        "Incumbent artifact changed while checkpointing; retrying copy",
                        source_path=source_path_string,
                        latest_source_path=str(latest_source_path),
                    )
                    continue
                raise RuntimeError(
                    "Failed to persist incumbent model artifacts for checkpoint "
                    f"(source={source_path_string})"
                ) from error
        raise RuntimeError(
            "Failed to persist incumbent model artifacts after retry "
            f"(source={source_path_string})"
        )

    def _create_wandb_gif_video(
        self,
        frames: list,
        attack: int,
        *,
        gif_stem: str | None = None,
    ) -> tuple[wandb.Video | None, Path | None]:
        if not frames:
            return None, None

        if gif_stem is None:
            gif_stem = f"eval_step{self.step}_attack{attack}"
        gif_path = Path(tempfile.gettempdir()) / f"{gif_stem}.gif"

        try:
            create_trajectory_gif(
                frames=frames,
                output_path=str(gif_path),
                duration=DEFAULT_GIF_FRAME_DURATION_MS,
            )

            video = wandb.Video(
                str(gif_path),
                format="gif",
            )
            self._pending_eval_gif_paths.append(gif_path)
            return video, gif_path
        except BaseException:
            gif_path.unlink(missing_ok=True)
            raise

    def _cleanup_wandb_gif_files(self) -> None:
        for gif_path in self._pending_eval_gif_paths:
            try:
                gif_path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning(
                    "Failed to delete temporary eval GIF",
                    path=str(gif_path),
                    error=str(error),
                )
        self._pending_eval_gif_paths.clear()

    @contextmanager
    def _defer_sigint_during_shutdown(self):
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def save(self, extra_checkpoint_state: dict[str, object] | None = None):
        """Save model checkpoint."""
        paths = self.weight_manager.save(
            self._export_model,
            self.ema_model,
            self.optimizer,
            self.scheduler,
            self.step,
            export_for_rust=True,
            extra_checkpoint_state=extra_checkpoint_state,
        )
        logger.info(
            "Saved checkpoint",
            step=self.step,
            checkpoint=str(paths["checkpoint"]),
            onnx=str(paths.get("onnx")) if "onnx" in paths else None,
        )
        return paths

    @property
    def ema_model(self) -> TetrisNet | None:
        if self.ema is None:
            return None
        return cast(TetrisNet, self.ema.model)

    def _drain_async_checkpoint_saver(self) -> None:
        saver = self._async_checkpoint_saver
        if saver is None:
            return
        saver.raise_if_failed()
        for saved_step, paths in saver.drain_completed():
            logger.info(
                "Saved checkpoint asynchronously",
                step=saved_step,
                checkpoint=str(paths["checkpoint"]),
                onnx=str(paths.get("onnx")) if "onnx" in paths else None,
            )

    def _shutdown_async_checkpoint_saver(self) -> None:
        saver = self._async_checkpoint_saver
        self._async_checkpoint_saver = None
        if saver is None:
            return
        saver.shutdown()
        for saved_step, paths in saver.drain_completed():
            logger.info(
                "Saved checkpoint asynchronously",
                step=saved_step,
                checkpoint=str(paths["checkpoint"]),
                onnx=str(paths.get("onnx")) if "onnx" in paths else None,
            )

    def _log_final_wandb_model_artifact(self, saved_paths: dict[str, Path]) -> None:
        if wandb.run is None:
            logger.warning(
                "Skipping final WandB model artifact upload; no active WandB run",
                step=self.step,
            )
            return

        artifact_name = f"tetris-model-{wandb.run.id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description="Final model snapshot saved when training loop stops",
            metadata={
                "trainer_step": self.step,
                "run_name": self.config.run.run_name,
            },
        )

        files_to_upload = [saved_paths["checkpoint"], saved_paths["metadata"]]
        onnx_path = saved_paths.get("onnx")
        if onnx_path is not None:
            files_to_upload.append(onnx_path)
            conv_path, heads_path, fc_path = split_model_paths(onnx_path)
            files_to_upload.extend([conv_path, heads_path, fc_path])
            files_to_upload.extend(
                [
                    optional_path
                    for optional_path in optional_model_artifact_paths(onnx_path)
                    if optional_path.exists()
                ]
            )
        else:
            # WeightManager.save currently always exports ONNX for Rust; fail fast if this changes.
            checkpoint_dir = self.config.run.checkpoint_dir
            if checkpoint_dir is None:
                raise RuntimeError("checkpoint_dir is not set on training config")
            expected_onnx_path = checkpoint_dir / LATEST_ONNX_FILENAME
            raise RuntimeError(
                "Saved paths missing ONNX artifact during final WandB upload "
                f"(expected {expected_onnx_path})"
            )

        checkpoint_dir = self.config.run.checkpoint_dir
        if checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is not set on training config")
        incumbent_onnx_path = checkpoint_dir / INCUMBENT_ONNX_FILENAME
        if incumbent_onnx_path.exists():
            assert_rust_inference_artifacts(incumbent_onnx_path)
            files_to_upload.append(incumbent_onnx_path)
            incumbent_conv_path, incumbent_heads_path, incumbent_fc_path = (
                split_model_paths(incumbent_onnx_path)
            )
            files_to_upload.extend(
                [incumbent_conv_path, incumbent_heads_path, incumbent_fc_path]
            )
            files_to_upload.extend(
                [
                    optional_path
                    for optional_path in optional_model_artifact_paths(
                        incumbent_onnx_path
                    )
                    if optional_path.exists()
                ]
            )

        data_dir = self.config.run.data_dir
        if data_dir is None:
            raise RuntimeError("data_dir is not set on training config")
        training_data_path = data_dir / TRAINING_DATA_FILENAME
        if training_data_path.exists():
            files_to_upload.append(training_data_path)

        for file_path in files_to_upload:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Expected model artifact file is missing: {file_path}"
                )
            artifact.add_file(str(file_path), name=file_path.name)

        wandb.log_artifact(artifact, aliases=["latest", "final", f"step-{self.step}"])
        logger.info(
            "Uploaded final model artifact to WandB",
            artifact_name=artifact_name,
            step=self.step,
            files=[path.name for path in files_to_upload],
        )

    def _shutdown_after_training(
        self,
        generator: GameGenerator,
        export_model: TetrisNet,
        log_to_wandb: bool,
        interrupted: bool,
        candidate_gate_schedule: CandidateGateSchedule | None = None,
    ) -> BaseException | None:
        if interrupted:
            logger.info(
                "Completing graceful shutdown; deferring additional Ctrl+C until final checkpointing finishes",
                step=self.step,
            )

        stop_error: BaseException | None = None
        with self._defer_sigint_during_shutdown():
            self._shutdown_r2_sync()
            try:
                logger.info("Stopping game generator")
                generator.stop()
                logger.info(
                    "Game generator stopped",
                    games_generated=generator.games_generated(),
                    examples_generated=generator.examples_generated(),
                )
            except BaseException as error:
                stop_error = error
                logger.exception("Failed to stop game generator cleanly")

            try:
                self._shutdown_async_checkpoint_saver()
            except BaseException as error:
                if stop_error is None:
                    stop_error = error
                logger.exception("Failed to finalize async checkpoint saver")

            # Restore uncompiled model for final save + ONNX export
            self.model = export_model

            # Always save latest model state on shutdown/interruption.
            try:
                (
                    incumbent_model_artifact,
                    incumbent_model_source_path,
                ) = self._persist_incumbent_model_artifacts(generator)
                extra_checkpoint_state = {
                    "incumbent_uses_network": generator.incumbent_uses_network(),
                    "incumbent_model_step": generator.incumbent_model_step(),
                    "incumbent_nn_value_weight": generator.incumbent_nn_value_weight(),
                    "incumbent_death_penalty": generator.incumbent_death_penalty(),
                    "incumbent_overhang_penalty_weight": (
                        generator.incumbent_overhang_penalty_weight()
                    ),
                    "incumbent_eval_avg_attack": generator.incumbent_eval_avg_attack(),
                    "incumbent_model_source_path": incumbent_model_source_path,
                    "incumbent_model_artifact": (
                        incumbent_model_artifact.name
                        if incumbent_model_artifact is not None
                        else None
                    ),
                    "next_display_game_number": self._next_display_game_number,
                }
                extra_checkpoint_state.update(self._runtime_override_checkpoint_state())
                shutdown_now_s = time.perf_counter()
                extra_checkpoint_state.update(
                    self._candidate_gate_checkpoint_state(
                        candidate_gate_schedule,
                        now_s=shutdown_now_s,
                    )
                )
                extra_checkpoint_state.update(
                    self._cumulative_progress_checkpoint_state(
                        generator, now_s=shutdown_now_s
                    )
                )
                final_saved_paths = self.save(
                    extra_checkpoint_state=extra_checkpoint_state
                )
                if log_to_wandb:
                    self._log_final_wandb_model_artifact(final_saved_paths)
                    wandb.finish()
            finally:
                self._cleanup_wandb_gif_files()

        return stop_error

    def train(self, log_to_wandb: bool = True):
        """
        Run parallel training with Rust game generation in background.

        The Rust GameGenerator runs in a background thread, continuously
        generating games into a shared in-memory buffer. Python samples
        directly from the buffer via generator.sample_batch(). With candidate
        gating enabled, ONNX exports are gated by an adaptive wall-clock
        schedule so failed promotions back off future evaluations instead of
        exporting models continuously while the evaluator is busy. Without
        candidate gating, the trainer exports a fresh model artifact on the
        base sync interval and switches all workers to it immediately.

        Args:
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        num_steps = self.config.optimizer.total_steps
        if self.step >= num_steps:
            logger.info(
                "Target step already reached; skipping training loop",
                target_step=num_steps,
                current_step=self.step,
            )
            return
        # Paths for parallel training (validated in __init__)
        assert self.config.run.checkpoint_dir is not None
        assert self.config.run.data_dir is not None
        onnx_path = self.config.run.checkpoint_dir / PARALLEL_ONNX_FILENAME
        candidate_model_dir = self.config.run.checkpoint_dir / MODEL_CANDIDATES_DIRNAME
        candidate_model_dir.mkdir(parents=True, exist_ok=True)
        self._maybe_reload_runtime_overrides(
            now_s=time.perf_counter(),
            next_log_time_s=None,
            next_checkpoint_time_s=None,
            force=True,
        )
        candidate_gating_enabled = self._candidate_gating_enabled()

        # Export initial model (full ONNX + split models for cached Rust inference)
        initial_export_model = self.ema_model or self._export_model
        self._export_rust_inference_artifacts(
            initial_export_model,
            onnx_path,
            export_name="Initial",
        )

        # Optionally compile model for faster training forward/backward
        export_model = self._export_model
        if self._effective_use_torch_compile:
            logger.info("Compiling model with torch.compile")
            self.model = cast(TetrisNet, torch.compile(self.model))
        elif self.config.optimizer.use_torch_compile and self.device.type == "mps":
            logger.info(
                "Skipping torch.compile on MPS device (eager is faster on Apple Silicon)"
            )

        generator_model_path = onnx_path
        if self.initial_incumbent_model_path is not None:
            assert_rust_inference_artifacts(self.initial_incumbent_model_path)
            generator_model_path = self.initial_incumbent_model_path

        mcts_config = self._build_generator_mcts_config()
        if (
            self.recompute_initial_incumbent_eval_avg_attack
            and candidate_gating_enabled
        ):
            self.initial_incumbent_eval_avg_attack = (
                self._evaluate_starting_incumbent_avg_attack(generator_model_path)
            )
            self.recompute_initial_incumbent_eval_avg_attack = False
        elif self.recompute_initial_incumbent_eval_avg_attack:
            logger.info(
                "Skipping starting incumbent baseline recompute because candidate gating is disabled",
                model_path=str(generator_model_path),
            )
            self.initial_incumbent_eval_avg_attack = 0.0
            self.recompute_initial_incumbent_eval_avg_attack = False

        # Start background game generator
        training_data_path = self.config.run.data_dir / TRAINING_DATA_FILENAME
        generator = GameGenerator(
            model_path=str(generator_model_path),
            training_data_path=str(training_data_path),
            config=mcts_config,
            max_placements=self.config.self_play.max_placements,
            add_noise=self.config.self_play.add_noise,
            max_examples=self.config.replay.buffer_size,
            save_interval_seconds=self.config.run.save_interval_seconds,
            num_workers=self.config.self_play.num_workers,
            initial_model_step=self.step,
            candidate_eval_seeds=(
                list(range(self.config.self_play.model_promotion_eval_games))
                if candidate_gating_enabled
                else []
            ),
            start_with_network=not self.config.self_play.bootstrap_without_network,
            non_network_num_simulations=self.config.self_play.bootstrap_num_simulations,
            initial_incumbent_eval_avg_attack=self.initial_incumbent_eval_avg_attack,
            candidate_gating_enabled=candidate_gating_enabled,
            save_eval_trees=self.config.self_play.save_eval_trees,
        )
        logger.info(
            "Starting game generator replay preload if training data exists",
            training_data_path=str(training_data_path),
        )
        self._wall_time_anchor_s = time.perf_counter()
        generator.start()
        self._generator_games_baseline = generator.games_generated()
        self._generator_examples_baseline = generator.examples_generated()
        self._init_r2_sync(generator)
        self._publish_incumbent_to_r2_if_enabled(generator, reason="startup")
        generator_log_fields: dict[str, object] = {
            "model_path": str(generator_model_path),
            "trainer_parallel_model_path": str(onnx_path),
            "training_data_path": str(training_data_path),
            "num_workers": self.config.self_play.num_workers,
            "add_noise": self.config.self_play.add_noise,
            "candidate_gating_enabled": candidate_gating_enabled,
            "candidate_eval_seeds": (
                self.config.self_play.model_promotion_eval_games
                if candidate_gating_enabled
                else 0
            ),
            "bootstrap_without_network": self.config.self_play.bootstrap_without_network,
            "bootstrap_num_simulations": self.config.self_play.bootstrap_num_simulations,
            "incumbent_nn_value_weight": self.config.self_play.nn_value_weight_schedule.initial,
            "initial_incumbent_eval_avg_attack": self.initial_incumbent_eval_avg_attack,
            "nn_value_weight_promotion_multiplier": self.config.self_play.nn_value_weight_schedule.multiplier,
            "nn_value_weight_promotion_max_delta": self.config.self_play.nn_value_weight_schedule.max_delta,
            "nn_value_weight_cap": self.config.self_play.nn_value_weight_schedule.cap,
            "model_sync_interval_seconds": self.config.run.model_sync_interval_seconds,
        }
        if candidate_gating_enabled:
            generator_log_fields["candidate_gate_failure_backoff_seconds"] = (
                self.config.run.model_sync_failure_backoff_seconds
            )
            generator_log_fields["candidate_gate_max_interval_seconds"] = (
                self.config.run.model_sync_max_interval_seconds
            )
        logger.info("Started background game generator", **generator_log_fields)

        # Wait for minimum buffer size
        logger.info(
            "Waiting for minimum replay buffer size",
            min_examples=self.config.replay.min_buffer_size,
        )
        while generator.buffer_size() < self.config.replay.min_buffer_size:
            time.sleep(1.0)
            logger.info(
                "Buffer fill progress",
                buffer_size=generator.buffer_size(),
                games_generated=generator.games_generated(),
            )

        start_step = self.step
        logger.info(
            "Starting training loop",
            target_step=num_steps,
            start_step=start_step,
            config=str(self.config),
        )
        self._async_checkpoint_saver = AsyncCheckpointSaver(self.weight_manager)
        use_device_replay_mirror = self._use_device_replay_mirror()
        staged_batch_size = (
            self.config.optimizer.batch_size * self.config.replay.prefetch_batches
        )
        staged_queue_target_batches = self.config.replay.staged_batch_cache_batches
        if use_device_replay_mirror:
            logger.info(
                "Configured full replay device mirroring",
                device=str(self.device),
                train_batch_size=self.config.optimizer.batch_size,
                refresh_seconds=self.config.replay.replay_mirror_refresh_seconds,
                delta_chunk_examples=self.config.replay.replay_mirror_delta_chunk_examples,
                pin_memory_batches=(
                    self.config.replay.pin_memory_batches and self.device.type == "cuda"
                ),
            )
        else:
            logger.info(
                "Configured staged replay sampling",
                train_batch_size=self.config.optimizer.batch_size,
                prefetch_batches=self.config.replay.prefetch_batches,
                staged_batch_size=staged_batch_size,
                staged_queue_target_batches=staged_queue_target_batches,
                device=str(self.device),
                pin_memory_batches=(
                    self.config.replay.pin_memory_batches and self.device.type == "cuda"
                ),
            )

        sample_batch_time_s = 0.0
        sample_batch_count = 0
        replay_sync_time_s = 0.0
        replay_sync_count = 0
        train_step_time_s = 0.0
        train_step_count = 0
        replay_ratio_throttle_time_s = 0.0
        latest_train_metrics: dict[str, float] | None = None
        pending_batches: deque[TrainingBatch] = deque()
        replay_mirror: CircularReplayMirror | None = None
        interval_anchor_s = time.perf_counter()
        throughput_window_start_s = interval_anchor_s
        throughput_window_start_steps = 0
        throughput_window_start_per_machine: dict[str, int] = (
            self._snapshot_games_per_machine()
        )
        next_log_time_s = interval_anchor_s + self.config.run.log_interval_seconds
        next_replay_sync_time_s = interval_anchor_s
        candidate_gate_schedule: CandidateGateSchedule | None = None
        direct_sync_interval_seconds: float | None = None
        next_model_sync_time_s: float | None = None
        recent_completed_replays: deque[CompletedGameLogEntry] = deque()
        if candidate_gating_enabled:
            candidate_gate_schedule = self._initialize_candidate_gate_schedule(
                now_s=interval_anchor_s
            )
            logger.info(
                "Initialized candidate gate schedule",
                current_interval_seconds=(
                    candidate_gate_schedule.current_interval_seconds
                ),
                failed_promotion_streak=(
                    candidate_gate_schedule.failed_promotion_streak
                ),
                next_export_delay_seconds=max(
                    0.0,
                    candidate_gate_schedule.next_export_time_s - interval_anchor_s,
                ),
            )
        else:
            direct_sync_interval_seconds = self._model_sync_interval_seconds()
            next_model_sync_time_s = interval_anchor_s + direct_sync_interval_seconds
            logger.info(
                "Initialized direct model sync schedule",
                interval_seconds=direct_sync_interval_seconds,
                next_sync_delay_seconds=max(
                    0.0, next_model_sync_time_s - interval_anchor_s
                ),
            )
        next_checkpoint_time_s = (
            interval_anchor_s + self.config.run.checkpoint_interval_seconds
        )

        interrupted = False
        pending_error: BaseException | None = None
        stop_error: BaseException | None = None

        try:
            while self.step < num_steps:
                self._drain_async_checkpoint_saver()
                pre_step_time = time.perf_counter()
                next_log_time_s, next_checkpoint_time_s = (
                    self._maybe_reload_runtime_overrides(
                        now_s=pre_step_time,
                        next_log_time_s=next_log_time_s,
                        next_checkpoint_time_s=next_checkpoint_time_s,
                        generator=generator,
                    )
                )
                if next_log_time_s is None:
                    raise RuntimeError("Training log deadline is unavailable")
                if next_checkpoint_time_s is None:
                    raise RuntimeError("Training checkpoint deadline is unavailable")

                if not self._replay_ratio_allows_step(
                    generator, session_steps=self.step - start_step
                ):
                    time.sleep(REPLAY_RATIO_THROTTLE_SLEEP_SECONDS)
                    replay_ratio_throttle_time_s += time.perf_counter() - pre_step_time
                    continue

                if use_device_replay_mirror:
                    should_refresh_mirror = (
                        replay_mirror is None
                        or pre_step_time >= next_replay_sync_time_s
                    )
                    if should_refresh_mirror:
                        replay_sync_start = time.perf_counter()
                        replay_mirror = self._refresh_replay_mirror(
                            generator, replay_mirror
                        )
                        replay_sync_elapsed_s = time.perf_counter() - replay_sync_start
                        if replay_mirror is None:
                            time.sleep(0.1)
                            continue
                        replay_sync_time_s += replay_sync_elapsed_s
                        replay_sync_count += 1
                        next_replay_sync_time_s = (
                            pre_step_time
                            + self.config.replay.replay_mirror_refresh_seconds
                        )
                    if replay_mirror is None:
                        raise RuntimeError(
                            "Replay mirror mode active but replay mirror is unavailable"
                        )
                    batch = self._sample_from_replay_mirror(replay_mirror)
                else:
                    should_refill = len(pending_batches) < staged_queue_target_batches
                    if should_refill:
                        sample_batch_start = time.perf_counter()
                        prefetched_batches = self._sample_prefetched_batches(
                            generator=generator,
                            staged_batch_size=staged_batch_size,
                        )
                        sample_batch_elapsed_s = (
                            time.perf_counter() - sample_batch_start
                        )
                        if prefetched_batches is None:
                            if not pending_batches:
                                time.sleep(0.1)
                                continue
                        else:
                            pending_batches.extend(prefetched_batches)
                            sample_batch_time_s += sample_batch_elapsed_s
                            sample_batch_count += len(prefetched_batches)

                    batch = pending_batches.popleft()
                self.step += 1
                session_step = self.step - start_step
                (
                    batch.boards,
                    batch.aux,
                    batch.policy_targets,
                    batch.masks,
                ) = maybe_mirror_training_tensors(
                    batch.boards,
                    batch.aux,
                    batch.policy_targets,
                    batch.masks,
                    self.config.optimizer.mirror_augmentation_probability,
                )

                # Train step
                is_log_step = pre_step_time >= next_log_time_s
                collect_train_metrics = (
                    is_log_step
                    or latest_train_metrics is None
                    or session_step % self.config.optimizer.train_step_metrics_interval
                    == 0
                )
                step_metrics = self.train_step(
                    batch,
                    collect_metrics=collect_train_metrics,
                )
                post_step_time = time.perf_counter()
                train_step_time_s += post_step_time - pre_step_time
                train_step_count += 1
                if step_metrics:
                    latest_train_metrics = step_metrics

                if candidate_gate_schedule is not None:
                    self._drain_model_eval_events(
                        generator,
                        log_to_wandb=log_to_wandb,
                        now_s=post_step_time,
                        candidate_gate_schedule=candidate_gate_schedule,
                    )

                if post_step_time >= next_log_time_s:
                    if latest_train_metrics is None:
                        raise RuntimeError(
                            "No collected train metrics are available for logging"
                        )
                    metrics = dict(latest_train_metrics)
                    wall_time_hours = self._cumulative_wall_time_hours(post_step_time)
                    metrics["wall_time_hours"] = wall_time_hours
                    if self.config.optimizer.compute_extra_train_metrics_on_log:
                        extra_metrics_start = time.perf_counter()
                        metrics.update(self._compute_extra_train_metrics(batch))
                        metrics["timing/extra_metrics_ms"] = 1000.0 * (
                            time.perf_counter() - extra_metrics_start
                        )
                    window_elapsed_s = post_step_time - throughput_window_start_s
                    steps_delta = session_step - throughput_window_start_steps
                    metrics["replay/buffer_size"] = generator.buffer_size()
                    metrics["replay/games_generated"] = (
                        self._cumulative_games_generated(generator)
                    )
                    cumulative_examples_generated = self._cumulative_examples_generated(
                        generator
                    )
                    metrics["replay/examples_generated"] = cumulative_examples_generated
                    batch_size = self.config.optimizer.batch_size
                    metrics["replay/samples_per_generated_example"] = (
                        self.step * batch_size / cumulative_examples_generated
                        if cumulative_examples_generated > 0
                        else 0.0
                    )
                    session_examples_generated = self._session_examples_generated(
                        generator
                    )
                    metrics["replay/session_samples_per_new_example"] = (
                        session_step * batch_size / session_examples_generated
                        if session_examples_generated > 0
                        else 0.0
                    )
                    metrics["throughput/replay_ratio_throttle_fraction"] = (
                        replay_ratio_throttle_time_s / window_elapsed_s
                        if window_elapsed_s > 0
                        else 0.0
                    )
                    metrics["replay/source"] = 1.0 if use_device_replay_mirror else 0.0
                    if use_device_replay_mirror:
                        if replay_mirror is None:
                            raise RuntimeError(
                                "Replay mirror mode active but replay_mirror is missing"
                            )
                        metrics["replay/mirror_size"] = replay_mirror.size
                        metrics["replay/mirror_logical_end"] = replay_mirror.logical_end
                    metrics["incumbent/model_step"] = generator.incumbent_model_step()
                    metrics["incumbent/uses_network"] = (
                        generator.incumbent_uses_network()
                    )
                    metrics["incumbent/nn_value_weight"] = (
                        generator.incumbent_nn_value_weight()
                    )
                    # Per-machine games/sec breakdown: one W&B series per
                    # machine_id (trainer's hostname for local self-play,
                    # remote `machine_id`s for connected generators) plus a
                    # `/total` series that sums them. Replaces the older
                    # local-only `games_per_second` and combined
                    # `total_games_per_second` metrics.
                    games_per_machine_now = self._snapshot_games_per_machine()
                    total_per_machine_delta = 0
                    seen_machine_ids = set(games_per_machine_now.keys()) | set(
                        throughput_window_start_per_machine.keys()
                    )
                    for machine_id in seen_machine_ids:
                        machine_delta = games_per_machine_now.get(
                            machine_id, 0
                        ) - throughput_window_start_per_machine.get(machine_id, 0)
                        total_per_machine_delta += machine_delta
                        metrics[f"throughput/games_per_second/{machine_id}"] = (
                            machine_delta / window_elapsed_s
                            if window_elapsed_s > 0
                            else 0.0
                        )
                    metrics["throughput/games_per_second/total"] = (
                        total_per_machine_delta / window_elapsed_s
                        if window_elapsed_s > 0
                        else 0.0
                    )
                    metrics["throughput/steps_per_second"] = (
                        steps_delta / window_elapsed_s if window_elapsed_s > 0 else 0.0
                    )
                    throughput_window_start_s = post_step_time
                    throughput_window_start_per_machine = games_per_machine_now
                    throughput_window_start_steps = session_step
                    metrics["timing/sample_batch_ms"] = (
                        1000.0 * sample_batch_time_s / sample_batch_count
                        if sample_batch_count > 0 and not use_device_replay_mirror
                        else 0.0
                    )
                    metrics["timing/replay_sync_ms"] = (
                        1000.0 * replay_sync_time_s / replay_sync_count
                        if replay_sync_count > 0 and use_device_replay_mirror
                        else 0.0
                    )
                    metrics["timing/train_step_ms"] = (
                        1000.0 * train_step_time_s / train_step_count
                        if train_step_count > 0
                        else 0.0
                    )
                    # Always drain completed games so Rust-side queue doesn't
                    # grow unbounded when WandB logging is disabled.
                    completed_games = self._drain_completed_games(generator)
                    completed_game_stats = [
                        (completed_game.game_number, completed_game.stats)
                        for completed_game in completed_games
                    ]
                    if direct_sync_interval_seconds is not None:
                        self._remember_recent_completed_replays(
                            recent_completed_replays,
                            completed_games,
                            min_completed_time_s=(
                                post_step_time - direct_sync_interval_seconds
                            ),
                        )
                    if log_to_wandb:
                        metrics["trainer_step"] = self.step
                        wandb.log(metrics)
                        if self.config.optimizer.log_individual_games_to_wandb:
                            for (
                                game_number,
                                game_stats,
                            ) in completed_game_stats:
                                game_metrics = {
                                    "game_number": game_number,
                                    "game/number": game_number,
                                    "trainer_step": self.step,
                                    "wall_time_hours": wall_time_hours,
                                }
                                for key, value in game_stats.items():
                                    game_metrics[f"game/{key}"] = value
                                game_metrics["game_time/total_attack"] = game_stats[
                                    "total_attack"
                                ]
                                episode_length = game_stats["episode_length"]
                                if episode_length <= 0:
                                    raise ValueError(
                                        "Invalid episode_length for game "
                                        f"{game_number}: {episode_length}"
                                    )
                                game_metrics["game/attack_per_move"] = (
                                    game_stats["total_attack"] / episode_length
                                )
                                game_metrics["game/lines_per_move"] = (
                                    game_stats["total_lines"] / episode_length
                                )
                                game_metrics["game/hold_rate"] = (
                                    game_stats["holds"] / episode_length
                                )
                                # Don't pin per-game logs to the training step:
                                # multiple games can complete between train
                                # ticks, and reusing the same step causes only
                                # a subset to appear in history.
                                wandb.log(game_metrics)
                        else:
                            game_avg_metrics = average_completed_games(
                                completed_game_stats
                            )
                            if game_avg_metrics:
                                game_avg_metrics["trainer_step"] = self.step
                                game_avg_metrics["wall_time_hours"] = wall_time_hours
                                game_avg_metrics["game_time/total_attack"] = (
                                    game_avg_metrics["game/total_attack"]
                                )
                                wandb.log(game_avg_metrics)
                    sample_batch_time_s = 0.0
                    sample_batch_count = 0
                    replay_sync_time_s = 0.0
                    replay_sync_count = 0
                    train_step_time_s = 0.0
                    train_step_count = 0
                    replay_ratio_throttle_time_s = 0.0
                    logger.info(
                        "Training progress",
                        step=self.step,
                        loss=metrics["train/loss"],
                        learning_rate=metrics["train/learning_rate"],
                        buffer_size=generator.buffer_size(),
                        games_generated=self._cumulative_games_generated(generator),
                        local_games_per_second=metrics.get(
                            f"throughput/games_per_second/{self._local_machine_id}",
                            0.0,
                        ),
                        total_games_per_second=metrics[
                            "throughput/games_per_second/total"
                        ],
                        steps_per_second=metrics["throughput/steps_per_second"],
                        sample_batch_ms=metrics["timing/sample_batch_ms"],
                        replay_sync_ms=metrics["timing/replay_sync_ms"],
                        train_step_ms=metrics["timing/train_step_ms"],
                        samples_per_generated_example=metrics[
                            "replay/samples_per_generated_example"
                        ],
                        replay_ratio_throttle_fraction=metrics[
                            "throughput/replay_ratio_throttle_fraction"
                        ],
                    )
                    next_log_time_s = roll_interval_deadline(
                        next_log_time_s,
                        self.config.run.log_interval_seconds,
                        post_step_time,
                    )

                # Export updated model for generator
                if candidate_gate_schedule is not None:
                    gate_busy = generator.candidate_gate_busy()
                    if (
                        not gate_busy
                        and post_step_time >= candidate_gate_schedule.next_export_time_s
                    ):
                        candidate_onnx_path = (
                            candidate_model_dir / f"candidate_step_{self.step}.onnx"
                        )
                        candidate_export_model = self.ema_model or export_model
                        onnx_export_ms = self._export_rust_inference_artifacts(
                            candidate_export_model,
                            candidate_onnx_path,
                            export_name="Candidate",
                        )
                        incumbent_nn_value_weight = (
                            generator.incumbent_nn_value_weight()
                        )
                        candidate_nn_value_weight = (
                            self._compute_candidate_nn_value_weight(
                                current_weight=incumbent_nn_value_weight,
                                config=self.config,
                                cumulative_games=self._cumulative_games_generated(
                                    generator
                                ),
                            )
                        )
                        force_promote = self._force_promote_next_candidate
                        (
                            candidate_death_penalty,
                            candidate_overhang_penalty_weight,
                        ) = self._scaled_penalties_for(
                            generator, nn_value_weight=candidate_nn_value_weight
                        )
                        queued = generator.queue_candidate_model(
                            str(candidate_onnx_path),
                            self.step,
                            candidate_nn_value_weight,
                            candidate_death_penalty,
                            candidate_overhang_penalty_weight,
                            force_promote=force_promote,
                        )
                        if queued and force_promote:
                            self._force_promote_next_candidate = False
                        logger.info(
                            "Queued candidate model for evaluator",
                            step=self.step,
                            path=str(candidate_onnx_path),
                            queued=queued,
                            force_promote=force_promote,
                            onnx_export_ms=onnx_export_ms,
                            incumbent_nn_value_weight=incumbent_nn_value_weight,
                            candidate_nn_value_weight=candidate_nn_value_weight,
                            candidate_gate_failed_promotion_streak=(
                                candidate_gate_schedule.failed_promotion_streak
                            ),
                            candidate_gate_interval_seconds=(
                                candidate_gate_schedule.current_interval_seconds
                            ),
                        )
                        if not queued:
                            self._defer_candidate_gate_export(
                                candidate_gate_schedule,
                                now_s=time.perf_counter(),
                            )
                        if log_to_wandb:
                            next_candidate_export_delay_seconds = max(
                                0.0,
                                candidate_gate_schedule.next_export_time_s
                                - time.perf_counter(),
                            )
                            wandb.log(
                                {
                                    "trainer_step": self.step,
                                    "timing/onnx_export_ms": onnx_export_ms,
                                    "model_gate/queued_candidate_nn_value_weight": candidate_nn_value_weight,
                                    "model_gate/failed_promotion_streak": (
                                        candidate_gate_schedule.failed_promotion_streak
                                    ),
                                    "model_gate/current_export_interval_seconds": (
                                        candidate_gate_schedule.current_interval_seconds
                                    ),
                                    "model_gate/next_export_delay_seconds": (
                                        next_candidate_export_delay_seconds
                                    ),
                                }
                            )
                else:
                    if next_model_sync_time_s is None:
                        raise RuntimeError("Direct model sync schedule is unavailable")
                    if direct_sync_interval_seconds is None:
                        raise RuntimeError("Direct model sync interval is unavailable")
                    if post_step_time >= next_model_sync_time_s:
                        sync_onnx_path = (
                            candidate_model_dir / f"sync_step_{self.step}.onnx"
                        )
                        sync_export_model = self.ema_model or export_model
                        onnx_export_ms = self._export_rust_inference_artifacts(
                            sync_export_model,
                            sync_onnx_path,
                            export_name="Direct sync",
                        )
                        incumbent_nn_value_weight = (
                            generator.incumbent_nn_value_weight()
                        )
                        synced_nn_value_weight = (
                            self._compute_candidate_nn_value_weight(
                                current_weight=incumbent_nn_value_weight,
                                config=self.config,
                                cumulative_games=self._cumulative_games_generated(
                                    generator
                                ),
                            )
                        )
                        (
                            synced_death_penalty,
                            synced_overhang_penalty_weight,
                        ) = self._scaled_penalties_for(
                            generator, nn_value_weight=synced_nn_value_weight
                        )
                        synced = generator.sync_model_directly(
                            str(sync_onnx_path),
                            self.step,
                            synced_nn_value_weight,
                            synced_death_penalty,
                            synced_overhang_penalty_weight,
                        )
                        if synced:
                            self._upload_to_r2_if_enabled(
                                sync_onnx_path,
                                self.step,
                                synced_nn_value_weight,
                                synced_death_penalty,
                                synced_overhang_penalty_weight,
                            )
                        sync_now_s = time.perf_counter()
                        next_model_sync_time_s = roll_interval_deadline(
                            next_model_sync_time_s,
                            direct_sync_interval_seconds,
                            sync_now_s,
                        )
                        logger.info(
                            "Synced model directly for self-play",
                            step=self.step,
                            path=str(sync_onnx_path),
                            synced=synced,
                            onnx_export_ms=onnx_export_ms,
                            incumbent_nn_value_weight=incumbent_nn_value_weight,
                            synced_nn_value_weight=synced_nn_value_weight,
                            next_sync_delay_seconds=max(
                                0.0, next_model_sync_time_s - sync_now_s
                            ),
                        )
                        if log_to_wandb:
                            sync_wandb_data: dict[str, object] = {
                                "trainer_step": self.step,
                                "timing/onnx_export_ms": onnx_export_ms,
                                "model_sync/direct_sync_succeeded": (
                                    1.0 if synced else 0.0
                                ),
                                "model_sync/nn_value_weight": synced_nn_value_weight,
                                "model_sync/next_sync_delay_seconds": max(
                                    0.0,
                                    next_model_sync_time_s - time.perf_counter(),
                                ),
                            }
                            sync_wandb_data.update(
                                self._build_direct_sync_recent_game_wandb_data(
                                    recent_completed_replays,
                                    now_s=sync_now_s,
                                    window_s=direct_sync_interval_seconds,
                                )
                            )
                            wandb.log(sync_wandb_data)

                # Queue checkpoint/export work on the background saver.
                if post_step_time >= next_checkpoint_time_s:
                    if self._async_checkpoint_saver is None:
                        raise RuntimeError(
                            "Async checkpoint saver is unavailable during training"
                        )
                    (
                        incumbent_model_artifact,
                        incumbent_model_source_path,
                    ) = self._persist_incumbent_model_artifacts(generator)
                    checkpoint_snapshot = capture_checkpoint_snapshot(
                        model=export_model,
                        ema_model=self.ema_model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.step,
                        extra_checkpoint_state={
                            "incumbent_uses_network": generator.incumbent_uses_network(),
                            "incumbent_model_step": generator.incumbent_model_step(),
                            "incumbent_nn_value_weight": generator.incumbent_nn_value_weight(),
                            "incumbent_death_penalty": generator.incumbent_death_penalty(),
                            "incumbent_overhang_penalty_weight": (
                                generator.incumbent_overhang_penalty_weight()
                            ),
                            "incumbent_eval_avg_attack": generator.incumbent_eval_avg_attack(),
                            "incumbent_model_source_path": incumbent_model_source_path,
                            "incumbent_model_artifact": (
                                incumbent_model_artifact.name
                                if incumbent_model_artifact is not None
                                else None
                            ),
                            "next_display_game_number": self._next_display_game_number,
                        }
                        | self._runtime_override_checkpoint_state()
                        | self._candidate_gate_checkpoint_state(
                            candidate_gate_schedule,
                            now_s=time.perf_counter(),
                        )
                        | self._cumulative_progress_checkpoint_state(
                            generator, now_s=time.perf_counter()
                        ),
                    )
                    self._async_checkpoint_saver.submit(
                        snapshot=checkpoint_snapshot,
                        model_kwargs=self.config.network.model_dump(),
                    )
                    logger.info("Queued async checkpoint save", step=self.step)
                    next_checkpoint_time_s = roll_interval_deadline(
                        next_checkpoint_time_s,
                        self.config.run.checkpoint_interval_seconds,
                        time.perf_counter(),
                    )

        except KeyboardInterrupt:
            interrupted = True
            logger.info("Training interrupted by user", step=self.step)
        except BaseException as error:
            pending_error = error
            logger.exception("Training loop failed", step=self.step)
        finally:
            stop_error = self._shutdown_after_training(
                generator=generator,
                export_model=export_model,
                log_to_wandb=log_to_wandb,
                interrupted=interrupted,
                candidate_gate_schedule=candidate_gate_schedule,
            )

        if pending_error is not None:
            raise pending_error
        if stop_error is not None:
            raise stop_error
        if interrupted:
            logger.info("Training stopped cleanly after interrupt", step=self.step)
