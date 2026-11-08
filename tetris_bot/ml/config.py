"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class NetworkConfig(ConfigModel):
    """Neural network architecture hyperparameters."""

    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    board_stats_hidden: int
    board_proj_hidden: int
    fc_hidden: int
    aux_hidden: int
    num_aux_hidden_layers: int
    fusion_hidden: int
    num_fusion_blocks: int
    conv_kernel_size: int
    conv_padding: int


class OptimizerConfig(ConfigModel):
    """Training loop, optimizer, loss, and logging hyperparameters."""

    total_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    lr_schedule: str
    lr_decay_steps: int
    lr_min_factor: float
    lr_warmup_steps: int = 0
    lr_warmup_start_factor: float = 0.0
    lr_step_gamma: float
    lr_step_divisor: int
    value_loss_weight_window: int
    policy_loss_scale: float
    ema_decay: float
    use_torch_compile: bool
    train_step_metrics_interval: int
    compute_extra_train_metrics_on_log: bool
    mirror_augmentation_probability: float
    log_individual_games_to_wandb: bool


NNValueWeightScheduleStrategy = Literal["per_promotion", "per_games_interval"]


class NNValueWeightScheduleConfig(ConfigModel):
    """Schedule for the `nn_value_weight` ramp.

    Owns both the ramp shape (`initial`, `multiplier`, `max_delta`, `cap` —
    a multiplicative step capped per-tick by `max_delta`, clamped at `cap`)
    and the trigger that decides when a tick fires:
      - `per_promotion`: legacy. One tick per promotion (gating) or per
        direct sync (no-gating); stepped from the current incumbent weight,
        so missed syncs/non-promotions stall the ramp.
      - `per_games_interval`: one tick per `games_interval` cumulative
        completed games, deterministically re-derived from `initial`.
        Independent of sync/promotion cadence and resume-deterministic.
    """

    strategy: NNValueWeightScheduleStrategy = "per_promotion"
    games_interval: int = 1
    initial: float = 0.01
    multiplier: float = 1.4
    max_delta: float = 0.1
    cap: float = 1.0

    @model_validator(mode="after")
    def _validate(self) -> "NNValueWeightScheduleConfig":
        if self.games_interval < 1:
            raise ValueError("nn_value_weight_schedule.games_interval must be >= 1")
        if self.initial < 0.0:
            raise ValueError("nn_value_weight_schedule.initial must be >= 0")
        if self.multiplier < 1.0:
            raise ValueError("nn_value_weight_schedule.multiplier must be >= 1.0")
        if self.max_delta < 0.0:
            raise ValueError("nn_value_weight_schedule.max_delta must be >= 0")
        if self.cap < self.initial:
            raise ValueError("nn_value_weight_schedule.cap must be >= initial")
        return self


PenaltyScheduleStrategy = Literal["gated", "constant_then_linear"]


class PenaltyScheduleConfig(ConfigModel):
    """Schedule for `death_penalty` and `overhang_penalty_weight`.

    Owns the base penalty values and the ramp-down trigger. The two
    penalties share a single scale factor in [0, 1] applied uniformly:
      - `gated`: scale = 1.0 while `nn_value_weight <
        nn_value_weight_schedule.cap`, else 0.0. Legacy "drop at cap" rule.
      - `constant_then_linear`: scale = 1.0 for the first `hold_games`
        cumulative completed games, then linearly decays to 0.0 over the next
        `decay_games`, then stays at 0.0. Independent of `nn_value_weight`.
    """

    strategy: PenaltyScheduleStrategy = "gated"
    hold_games: int = 0
    decay_games: int = 1
    death_penalty: float = 10.0
    overhang_penalty_weight: float = 10.0

    @model_validator(mode="after")
    def _validate(self) -> "PenaltyScheduleConfig":
        if self.hold_games < 0:
            raise ValueError("penalty_schedule.hold_games must be >= 0")
        if self.decay_games < 1:
            raise ValueError("penalty_schedule.decay_games must be >= 1")
        if self.death_penalty < 0.0:
            raise ValueError("penalty_schedule.death_penalty must be >= 0")
        if self.overhang_penalty_weight < 0.0:
            raise ValueError("penalty_schedule.overhang_penalty_weight must be >= 0")
        return self


class SelfPlayConfig(ConfigModel):
    """MCTS and self-play generation hyperparameters."""

    num_simulations: int
    c_puct: float
    temperature: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    add_noise: bool
    use_parent_value_for_unvisited_q: bool
    visit_sampling_epsilon: float
    mcts_seed: int | None
    reuse_tree: bool
    # Visit budget instead of a playout budget: visits carried into a reused root count
    # toward num_simulations, so each search runs max(min_new_simulations,
    # num_simulations - carried) new simulations. With tree reuse ~37% of root visits are
    # carried over at 2000 sims, so the default (False) spends ~1.6x the nominal budget.
    count_reused_visits: bool = False
    min_new_simulations: int = Field(default=0, ge=0)
    num_workers: int
    max_placements: int
    penalty_schedule: PenaltyScheduleConfig = Field(
        default_factory=PenaltyScheduleConfig
    )
    nn_value_weight_schedule: NNValueWeightScheduleConfig = Field(
        default_factory=NNValueWeightScheduleConfig
    )
    use_candidate_gating: bool
    model_promotion_eval_games: int
    bootstrap_without_network: bool
    bootstrap_num_simulations: int
    save_eval_trees: bool


SELF_PLAY_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "num_simulations",
    "c_puct",
    "temperature",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "add_noise",
    "use_parent_value_for_unvisited_q",
    "visit_sampling_epsilon",
    "reuse_tree",
    "count_reused_visits",
    "min_new_simulations",
    "max_placements",
    "penalty_schedule",
    "nn_value_weight_schedule",
)


class SelfPlaySnapshot(ConfigModel):
    """Trainer-authoritative subset of `SelfPlayConfig` published to R2.

    Why: remote generators must run identical per-move MCTS logic to the
    trainer, but each machine reads its own `config.yaml`. The trainer
    publishes this snapshot at startup; generators fetch it before
    constructing `MCTSConfig` and override the listed fields. Fields not
    in the snapshot stay per-machine (`num_workers`, `mcts_seed`) or are
    trainer-only (gating, bootstrap).
    """

    num_simulations: int
    c_puct: float
    temperature: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    add_noise: bool
    use_parent_value_for_unvisited_q: bool
    visit_sampling_epsilon: float
    reuse_tree: bool
    # Defaults keep snapshots published by older trainers parseable.
    count_reused_visits: bool = False
    min_new_simulations: int = Field(default=0, ge=0)
    max_placements: int
    penalty_schedule: PenaltyScheduleConfig = Field(
        default_factory=PenaltyScheduleConfig
    )
    nn_value_weight_schedule: NNValueWeightScheduleConfig = Field(
        default_factory=NNValueWeightScheduleConfig
    )

    @classmethod
    def from_self_play(cls, self_play: "SelfPlayConfig") -> "SelfPlaySnapshot":
        return cls(**{f: getattr(self_play, f) for f in SELF_PLAY_SNAPSHOT_FIELDS})

    def apply_to(self, self_play: "SelfPlayConfig") -> None:
        for field in SELF_PLAY_SNAPSHOT_FIELDS:
            setattr(self_play, field, getattr(self, field))


class ReplayConfig(ConfigModel):
    """Replay buffer and batch sampling hyperparameters."""

    buffer_size: int
    min_buffer_size: int
    prefetch_batches: int
    staged_batch_cache_batches: int
    mirror_replay_on_accelerator: bool
    replay_mirror_refresh_seconds: float
    replay_mirror_delta_chunk_examples: int
    pin_memory_batches: bool
    # Cap on training samples drawn per newly generated (or ingested) replay
    # example this session; the trainer sleeps instead of stepping when it is
    # ahead. 0 disables the cap (free-running trainer). Defaulted so configs
    # saved before this knob existed still load.
    max_train_samples_per_new_example: float = Field(default=0.0, ge=0.0)


class RunConfig(ConfigModel):
    """Run management: WandB identity, timing intervals, and auto-populated paths."""

    project_name: str
    run_name: str | None
    model_sync_interval_seconds: float
    model_sync_failure_backoff_seconds: float
    model_sync_max_interval_seconds: float
    checkpoint_interval_seconds: float
    log_interval_seconds: float
    save_interval_seconds: float
    # Self-play total-attack distribution logged to W&B (histogram + plot) over
    # the last `attack_histogram_window_games` games; interval 0 disables it.
    # Defaulted so configs saved before these knobs existed still load.
    attack_histogram_window_games: int = Field(default=1000, gt=0)
    attack_histogram_interval_seconds: float = Field(default=600.0, ge=0.0)
    run_dir: Path | None
    checkpoint_dir: Path | None
    data_dir: Path | None


class R2SyncConfig(ConfigModel):
    """Cloudflare R2 (S3-compatible) sync configuration for multi-machine training.

    Account-specific values (`R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`,
    `R2_SECRET_ACCESS_KEY`) are sourced from environment variables — typically
    via a `.env` file loaded with `python-dotenv` at process startup. The
    config object only carries non-secret, run-shared fields.

    Which sync threads run is determined by the entry point: `train.py`
    starts the trainer-side downloaders (replay chunks, game stats) and
    publishes the incumbent bundle; `run_generator.py` starts the
    generator-side uploaders and the model-pointer poller. The two are
    not interchangeable, so there is no `role` field here — set
    `enabled: false` to disable R2 sync entirely.

    `sync_run_id` defaults to a fresh UTC timestamp generated at trainer
    startup and persisted into `<run_dir>/r2_sync_run_id.txt`, so the
    same run keeps the same id across resumes. Generators auto-discover
    active runs from R2 and only prompt when more than one is fresh
    (within `active_run_freshness_seconds`).
    """

    enabled: bool = False
    bucket: str | None = None
    prefix: str = "tetris-mcts"
    sync_run_id: str | None = None
    active_run_freshness_seconds: float = 300.0
    chunk_max_examples: int = 16384
    chunk_upload_interval_seconds: float = 60.0
    chunk_download_poll_interval_seconds: float = 60.0
    model_pointer_poll_interval_seconds: float = 60.0
    request_timeout_seconds: float = 30.0


class TrainingConfig(ConfigModel):
    """Training hyperparameters."""

    network: NetworkConfig
    optimizer: OptimizerConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    run: RunConfig
    r2_sync: R2SyncConfig = Field(default_factory=R2SyncConfig)


class RuntimeOptimizerOverrides(ConfigModel):
    """Live-tunable optimizer overrides loaded from runtime_overrides.yaml."""

    lr_multiplier: float | None = 1.0
    grad_clip_norm: float | None = None
    weight_decay: float | None = None
    mirror_augmentation_probability: float | None = None


class RuntimeRunOverrides(ConfigModel):
    """Live-tunable run interval overrides loaded from runtime_overrides.yaml."""

    log_interval_seconds: float | None = None
    checkpoint_interval_seconds: float | None = None


class RuntimeSelfPlayOverrides(ConfigModel):
    """Self-play overrides loaded from runtime_overrides.yaml.

    `force_promote_next_candidate` is one-shot — consumed and reset by
    the trainer. `add_noise` and `visit_sampling_epsilon` are continuous
    overrides applied to live MCTS workers and republished as part of
    the trainer's R2 self-play snapshot, so remote generators converge
    on the same setting without restart.
    """

    force_promote_next_candidate: bool = False
    add_noise: bool | None = None
    visit_sampling_epsilon: float | None = None


class RuntimeOverrides(ConfigModel):
    """Whitelisted runtime overrides that can change during training."""

    optimizer: RuntimeOptimizerOverrides = Field(
        default_factory=RuntimeOptimizerOverrides
    )
    run: RuntimeRunOverrides = Field(default_factory=RuntimeRunOverrides)
    self_play: RuntimeSelfPlayOverrides = Field(
        default_factory=RuntimeSelfPlayOverrides
    )


class ResolvedRuntimeOptimizerOverrides(ConfigModel):
    lr_multiplier: float
    grad_clip_norm: float
    weight_decay: float
    mirror_augmentation_probability: float


class ResolvedRuntimeRunOverrides(ConfigModel):
    log_interval_seconds: float
    checkpoint_interval_seconds: float


class ResolvedRuntimeSelfPlayOverrides(ConfigModel):
    add_noise: bool
    visit_sampling_epsilon: float


class ResolvedRuntimeOverrides(ConfigModel):
    optimizer: ResolvedRuntimeOptimizerOverrides
    run: ResolvedRuntimeRunOverrides
    self_play: ResolvedRuntimeSelfPlayOverrides


def save_training_config(config: TrainingConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text()))


def save_runtime_overrides(overrides: RuntimeOverrides, path: Path) -> None:
    path.write_text(yaml.safe_dump(overrides.model_dump(mode="json"), sort_keys=False))


def load_runtime_overrides(path: Path) -> RuntimeOverrides:
    raw_data = yaml.safe_load(path.read_text())
    if raw_data is None:
        return RuntimeOverrides()
    return RuntimeOverrides.model_validate(raw_data)
