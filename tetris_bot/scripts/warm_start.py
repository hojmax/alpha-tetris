from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch
import wandb
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model
from tetris_bot.constants import (
    BENCHMARKS_DIR,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    DEFAULT_CONFIG_PATH,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    LATEST_METADATA_FILENAME,
    LATEST_ONNX_FILENAME,
    PARALLEL_ONNX_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.artifacts import (
    assert_rust_inference_artifacts,
    copy_model_artifact_bundle,
)
from tetris_bot.ml.config import (
    TrainingConfig,
    load_training_config,
    save_training_config,
)
from tetris_bot.ml.ema import ExponentialMovingAverage
from tetris_bot.ml.loss import RunningLossBalancer, compute_loss
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.optimizer import OptimizerBundle, SchedulerBundle
from tetris_bot.ml.policy_mirroring import maybe_mirror_training_tensors
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import (
    OptimizerLike,
    SchedulerLike,
    export_metadata,
    load_optimizer_state_dict,
    save_checkpoint,
)
from tetris_bot.run_setup import setup_run_directory
from tetris_bot.scripts.inspection.optimize_machine import (
    machine_profile,
    machine_type_fingerprint,
)
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
    build_tensor_dataset,
    build_torch_batch,
    ensure_required_keys,
    get_preload_mode,
    pick_device,
    tensor_dataset_bytes,
    validate_shapes,
)

logger = structlog.get_logger()
_OPTIMIZE_CACHE_DIR = BENCHMARKS_DIR / "profiles" / "optimize_cache"
_OPTIMIZED_WORKERS_ENV_VAR = "TETRIS_OPT_NUM_WORKERS"
_OFFLINE_RESUME_CHECKPOINT_FILENAME = "warm_start_offline_latest.pt"


@dataclass
class ScriptArgs:
    source_run_dir: Path
    output_run_dir: Path | None = None
    resume_from_source_offline_state: bool = False
    device: str = "auto"
    seed: int = 123
    epochs_per_round: float = 4.0
    early_stopping_patience: int = 5
    max_rounds: int = 0
    max_examples: int = 0
    batch_size: int | None = None
    learning_rate: float | None = None
    warmup_epochs: float = 3.0
    lr_min_factor: float = 0.1
    weight_decay: float | None = None
    grad_clip_norm: float | None = None
    eval_examples: int = 32_768
    eval_batch_size: int = 2_048
    preload_to_gpu: bool = True
    preload_to_ram: bool = False
    num_eval_games: int = 20
    eval_seed_start: int = 0
    eval_num_workers: int = 0
    eval_num_simulations: int | None = None
    eval_max_placements: int | None = None
    mcts_seed: int | None = 0
    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["offline", "warm-start"])


@dataclass(frozen=True)
class EvalWorkerResolution:
    num_workers: int
    source: str
    cache_path: str | None = None


@dataclass(frozen=True)
class WarmStartDatasetSetup:
    source: OfflineDataSource
    train_local_indices: np.ndarray
    eval_local_indices: np.ndarray
    total_examples: int
    num_selected: int
    preload_sec: float


@dataclass(frozen=True)
class WarmStartTrainingResult:
    best_state_dict: dict[str, torch.Tensor]
    best_ema_state_dict: dict[str, torch.Tensor] | None
    best_record: dict[str, float | int | bool | str]
    history: list[dict[str, float | int | bool | str]]
    ema_state_dict: dict[str, torch.Tensor] | None
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object] | list[dict[str, object]]
    loss_balancer_state: dict[str, object]
    current_value_loss_weight: float
    rng_state: dict[str, object]
    total_steps: int
    steps_per_round: int
    lr_schedule_total_steps: int
    lr_schedule_total_rounds: int
    lr_warmup_steps: int
    rounds_completed: int
    stop_reason: str
    stopped_after_non_improving_rounds: int


@dataclass(frozen=True)
class WarmStartOfflineResumeState:
    checkpoint_path: Path
    best_state_dict: dict[str, torch.Tensor]
    best_ema_state_dict: dict[str, torch.Tensor] | None
    best_record: dict[str, float | int | bool | str]
    history: list[dict[str, float | int | bool | str]]
    total_steps: int
    rounds_completed: int
    non_improving_rounds: int
    current_value_loss_weight: float
    rng_state: dict[str, object]


@dataclass(frozen=True)
class WarmStartRunResult:
    output_run_dir: Path
    checkpoint_dir: Path
    latest_checkpoint_path: Path
    latest_onnx_path: Path
    incumbent_onnx_path: Path
    parallel_onnx_path: Path
    summary_path: Path
    summary: dict[str, object]


def validate_args(args: ScriptArgs) -> None:
    if args.epochs_per_round <= 0:
        raise ValueError(f"epochs_per_round must be > 0 (got {args.epochs_per_round})")
    if args.early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be > 0 (got {args.early_stopping_patience})"
        )
    if args.max_rounds < 0:
        raise ValueError(f"max_rounds must be >= 0 (got {args.max_rounds})")
    if args.max_examples < 0:
        raise ValueError(f"max_examples must be >= 0 (got {args.max_examples})")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {args.batch_size})")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be > 0 when set (got {args.learning_rate})"
        )
    if args.warmup_epochs < 0.0:
        raise ValueError(f"warmup_epochs must be >= 0 (got {args.warmup_epochs})")
    if not 0.0 <= args.lr_min_factor <= 1.0:
        raise ValueError(f"lr_min_factor must be in [0, 1] (got {args.lr_min_factor})")
    if args.weight_decay is not None and args.weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be >= 0 when set (got {args.weight_decay})"
        )
    if args.grad_clip_norm is not None and args.grad_clip_norm < 0.0:
        raise ValueError(
            f"grad_clip_norm must be >= 0 when set (got {args.grad_clip_norm})"
        )
    if args.eval_examples <= 0:
        raise ValueError(f"eval_examples must be > 0 (got {args.eval_examples})")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be > 0 (got {args.eval_batch_size})")
    if args.num_eval_games <= 0:
        raise ValueError(f"num_eval_games must be > 0 (got {args.num_eval_games})")
    if args.eval_num_workers < 0 or args.eval_num_workers == 1:
        raise ValueError(
            f"eval_num_workers must be 0 (auto) or >= 2 (got {args.eval_num_workers})"
        )
    if args.eval_num_simulations is not None and args.eval_num_simulations <= 0:
        raise ValueError(
            "eval_num_simulations must be > 0 when set "
            f"(got {args.eval_num_simulations})"
        )
    if args.eval_max_placements is not None and args.eval_max_placements <= 0:
        raise ValueError(
            f"eval_max_placements must be > 0 when set (got {args.eval_max_placements})"
        )

    source_run_dir = args.source_run_dir.resolve()
    if not source_run_dir.exists():
        raise FileNotFoundError(
            f"Source run directory does not exist: {source_run_dir}"
        )
    if not source_run_dir.is_dir():
        raise NotADirectoryError(
            f"Source run directory is not a directory: {source_run_dir}"
        )

    training_data_path = source_run_dir / TRAINING_DATA_FILENAME
    if not training_data_path.exists():
        raise FileNotFoundError(f"Source training data not found: {training_data_path}")
    if args.resume_from_source_offline_state:
        source_offline_resume_checkpoint = offline_resume_checkpoint_path(
            source_run_dir
        )
        if not source_offline_resume_checkpoint.exists():
            raise FileNotFoundError(
                "Source run has no offline warm-start resume checkpoint: "
                f"{source_offline_resume_checkpoint}"
            )

    if args.output_run_dir is not None:
        output_run_dir = args.output_run_dir.resolve()
        if output_run_dir == source_run_dir:
            raise ValueError("output_run_dir must differ from source_run_dir")
        if output_run_dir.exists():
            raise FileExistsError(
                f"Output run directory already exists: {output_run_dir}"
            )


def clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def compute_training_steps(
    train_examples: int,
    *,
    batch_size: int,
    epochs: float,
) -> int:
    if train_examples <= 0:
        raise ValueError(f"train_examples must be > 0 (got {train_examples})")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {batch_size})")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0 (got {epochs})")
    return max(1, math.ceil((train_examples * epochs) / batch_size))


def offline_resume_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_DIRNAME / _OFFLINE_RESUME_CHECKPOINT_FILENAME


def source_latest_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME


def optimized_worker_env_cache_path(
    cache_dir: Path = _OPTIMIZE_CACHE_DIR,
) -> Path:
    fingerprint = machine_type_fingerprint(machine_profile())
    return cache_dir / f"{fingerprint}.env"


def parse_positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer (got {value!r})") from error

    if parsed <= 0:
        raise ValueError(f"{label} must be > 0 (got {parsed})")
    return parsed


def load_optimized_worker_override_from_environment() -> int | None:
    raw_value = os.getenv(_OPTIMIZED_WORKERS_ENV_VAR)
    if raw_value is None or raw_value.strip() == "":
        return None
    return parse_positive_int(raw_value, label=_OPTIMIZED_WORKERS_ENV_VAR)


def load_optimized_worker_override_from_cache(
    cache_dir: Path = _OPTIMIZE_CACHE_DIR,
) -> tuple[int | None, Path]:
    env_cache_path = optimized_worker_env_cache_path(cache_dir)
    if not env_cache_path.exists():
        return None, env_cache_path

    for raw_line in env_cache_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == _OPTIMIZED_WORKERS_ENV_VAR:
            return (
                parse_positive_int(value.strip(), label=f"{env_cache_path}:{key}"),
                env_cache_path,
            )
    return None, env_cache_path


def resolve_eval_num_workers(
    requested_workers: int,
    *,
    default_workers: int,
    cache_dir: Path = _OPTIMIZE_CACHE_DIR,
) -> EvalWorkerResolution:
    if requested_workers > 0:
        return EvalWorkerResolution(num_workers=requested_workers, source="cli")

    environment_override = load_optimized_worker_override_from_environment()
    if environment_override is not None:
        return EvalWorkerResolution(
            num_workers=environment_override,
            source="environment",
        )

    cached_override, cache_path = load_optimized_worker_override_from_cache(cache_dir)
    if cached_override is not None:
        return EvalWorkerResolution(
            num_workers=cached_override,
            source="optimize_cache",
            cache_path=str(cache_path),
        )

    return EvalWorkerResolution(
        num_workers=max(2, default_workers),
        source="config",
    )


def save_offline_resume_checkpoint(
    path: Path,
    *,
    model: TetrisNet,
    ema_state_dict: dict[str, torch.Tensor] | None,
    optimizer_state_dict: dict[str, object],
    scheduler_state_dict: dict[str, object] | list[dict[str, object]],
    best_state_dict: dict[str, torch.Tensor],
    best_ema_state_dict: dict[str, torch.Tensor] | None,
    best_record: dict[str, float | int | bool | str],
    history: list[dict[str, float | int | bool | str]],
    total_steps: int,
    rounds_completed: int,
    non_improving_rounds: int,
    current_value_loss_weight: float,
    loss_balancer_state: dict[str, object],
    rng_state: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "best_state_dict": best_state_dict,
        "best_record": best_record,
        "history": history,
        "total_steps": total_steps,
        "rounds_completed": rounds_completed,
        "non_improving_rounds": non_improving_rounds,
        "current_value_loss_weight": current_value_loss_weight,
        "loss_balancer_state": loss_balancer_state,
        "rng_state": rng_state,
    }
    if ema_state_dict is not None:
        state["ema_state_dict"] = ema_state_dict
    if best_ema_state_dict is not None:
        state["best_ema_state_dict"] = best_ema_state_dict
    torch.save(state, path)


def load_offline_resume_checkpoint(
    path: Path,
    *,
    model: TetrisNet,
    ema_model: torch.nn.Module | None,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike | None,
    loss_balancer: RunningLossBalancer,
    lr_schedule_total_steps: int,
    lr_warmup_steps: int,
    lr_min_factor: float,
) -> WarmStartOfflineResumeState:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state_dict = state.get("model_state_dict")
    if model_state_dict is None:
        raise ValueError(f"Offline resume checkpoint is missing model state: {path}")
    optimizer_state_dict = state.get("optimizer_state_dict")
    if optimizer_state_dict is None:
        raise ValueError(
            f"Offline resume checkpoint is missing optimizer state: {path}"
        )
    best_state_dict = state.get("best_state_dict")
    if best_state_dict is None:
        raise ValueError(
            f"Offline resume checkpoint is missing best model state: {path}"
        )
    best_record = state.get("best_record")
    if best_record is None:
        raise ValueError(f"Offline resume checkpoint is missing best record: {path}")

    model.load_state_dict(model_state_dict)
    if ema_model is not None:
        ema_state_dict = state.get("ema_state_dict")
        if ema_state_dict is None:
            ema_model.load_state_dict(model_state_dict)
        else:
            ema_model.load_state_dict(ema_state_dict)
    load_optimizer_state_dict(optimizer, optimizer_state_dict, source=path)
    resumed_total_steps = int(state.get("total_steps", 0))
    if scheduler is not None:
        scheduler_state_dict = state.get("scheduler_state_dict")
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
            apply_scheduler_lrs(optimizer, scheduler._last_lr)
        else:
            assert isinstance(scheduler, SchedulerBundle)
            align_warm_start_lr_scheduler_to_step(
                optimizer=optimizer,
                scheduler=scheduler,
                step=resumed_total_steps,
                warmup_steps=lr_warmup_steps,
                total_steps=lr_schedule_total_steps,
                lr_min_factor=lr_min_factor,
            )
    loss_balancer.load_state_dict(
        state.get(
            "loss_balancer_state",
            {},
        )
    )
    best_ema_state_dict = state.get("best_ema_state_dict")
    if best_ema_state_dict is None and ema_model is not None:
        best_ema_state_dict = best_state_dict

    return WarmStartOfflineResumeState(
        checkpoint_path=path,
        best_state_dict={
            key: value.detach().cpu().clone() for key, value in best_state_dict.items()
        },
        best_ema_state_dict=(
            {
                key: value.detach().cpu().clone()
                for key, value in best_ema_state_dict.items()
            }
            if best_ema_state_dict is not None
            else None
        ),
        best_record=dict(best_record),
        history=[dict(record) for record in state.get("history", [])],
        total_steps=resumed_total_steps,
        rounds_completed=int(state.get("rounds_completed", 0)),
        non_improving_rounds=int(state.get("non_improving_rounds", 0)),
        current_value_loss_weight=float(state.get("current_value_loss_weight", 1.0)),
        rng_state=copy.deepcopy(state.get("rng_state", {})),
    )


def resolve_lr_schedule_round_budget(
    *,
    early_stopping_patience: int,
    max_rounds: int,
) -> int:
    if max_rounds > 0:
        return max_rounds
    return early_stopping_patience + 1


def compute_warmup_cosine_lr_factor(
    *,
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_min_factor: float,
) -> float:
    if step < 0:
        raise ValueError(f"step must be >= 0 (got {step})")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0 (got {warmup_steps})")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0 (got {total_steps})")
    if not 0.0 <= lr_min_factor <= 1.0:
        raise ValueError(f"lr_min_factor must be in [0, 1] (got {lr_min_factor})")

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps

    cosine_steps = max(total_steps - warmup_steps, 1)
    cosine_denominator = max(cosine_steps - 1, 1)
    progress = min(max(step - warmup_steps, 0) / cosine_denominator, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_factor + (1.0 - lr_min_factor) * cosine


def apply_scheduler_lrs(
    optimizer: OptimizerLike,
    lrs: list[float],
) -> None:
    if len(optimizer.param_groups) != len(lrs):
        raise ValueError("Optimizer param group count does not match computed LR count")
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group["lr"] = lr


def align_warm_start_lr_scheduler_to_step(
    *,
    optimizer: OptimizerLike,
    scheduler: SchedulerBundle,
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_min_factor: float,
) -> None:
    factor = compute_warmup_cosine_lr_factor(
        step=step,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_min_factor=lr_min_factor,
    )
    for inner in scheduler.inner_schedulers:
        assert isinstance(inner, torch.optim.lr_scheduler.LambdaLR)
        inner_lrs = [base_lr * factor for base_lr in inner.base_lrs]
        inner.last_epoch = step
        inner._step_count = max(step + 1, 1)
        inner._last_lr = inner_lrs
        for param_group, lr in zip(inner.optimizer.param_groups, inner_lrs):
            param_group["lr"] = lr


def build_warm_start_lr_scheduler(
    optimizer: OptimizerBundle,
    *,
    warmup_steps: int,
    total_steps: int,
    lr_min_factor: float,
) -> SchedulerBundle:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0 (got {total_steps})")

    def lr_lambda(step: int) -> float:
        return compute_warmup_cosine_lr_factor(
            step=step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_min_factor=lr_min_factor,
        )

    return SchedulerBundle(
        tuple(
            torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            for opt in optimizer.inner_optimizers
        )
    )


def warm_start_selection_metric(policy_loss: float, value_loss: float) -> float:
    return policy_loss + (value_loss / 4.0)


def has_better_eval_metric(
    candidate_metrics: dict[str, float],
    best_record: dict[str, float | int | bool | str] | None,
) -> bool:
    candidate_selection_metric = warm_start_selection_metric(
        candidate_metrics["policy_loss"],
        candidate_metrics["value_loss"],
    )
    if best_record is None:
        return True
    return candidate_selection_metric < float(best_record["eval_selection_metric"])


def setup_offline_dataset(
    npz: np.lib.npyio.NpzFile,
    *,
    seed: int,
    max_examples: int,
    eval_examples: int,
    preload_mode: str,
    device: torch.device,
) -> WarmStartDatasetSetup:
    ensure_required_keys(npz)
    total_examples = validate_shapes(npz)
    selected_global_indices = np.arange(total_examples, dtype=np.int64)

    split_rng = np.random.default_rng(seed)
    split_rng.shuffle(selected_global_indices)
    if max_examples > 0:
        selected_global_indices = selected_global_indices[:max_examples]

    num_selected = len(selected_global_indices)
    if eval_examples >= num_selected:
        raise ValueError(
            "eval_examples must be smaller than the selected dataset size "
            f"(got eval_examples={eval_examples}, used_examples={num_selected})"
        )
    eval_local_indices = np.arange(eval_examples, dtype=np.int64)
    train_local_indices = np.arange(eval_examples, num_selected, dtype=np.int64)

    preload_start = time.perf_counter()
    tensor_data = None
    if preload_mode != "none":
        tensor_data = build_tensor_dataset(
            data=npz,
            selected_global_indices=selected_global_indices,
            mode=preload_mode,
            train_device=device,
        )
    preload_sec = time.perf_counter() - preload_start

    source = OfflineDataSource(
        npz=npz,
        selected_global_indices=selected_global_indices,
        tensor_data=tensor_data,
    )

    logger.info(
        "Warm-start dataset split",
        total_examples=total_examples,
        used_examples=num_selected,
        train_examples=len(train_local_indices),
        eval_examples=len(eval_local_indices),
        preload_mode=preload_mode,
        preload_seconds=preload_sec,
        preload_bytes=(
            tensor_dataset_bytes(tensor_data) if tensor_data is not None else 0
        ),
    )

    return WarmStartDatasetSetup(
        source=source,
        train_local_indices=train_local_indices,
        eval_local_indices=eval_local_indices,
        total_examples=total_examples,
        num_selected=num_selected,
        preload_sec=preload_sec,
    )


def evaluate_offline_losses(
    model: torch.nn.Module,
    *,
    source: OfflineDataSource,
    local_indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    value_loss_weight: float,
) -> dict[str, float]:
    total_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_indices), eval_batch_size):
            batch_indices = local_indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(source, batch_indices, device)
            )
            total_loss, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=value_loss_weight,
            )
            batch_size = len(batch_indices)
            total_sum += total_loss.item() * batch_size
            policy_sum += policy_loss.item() * batch_size
            value_sum += value_loss.item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot evaluate offline losses on an empty split")

    return {
        "total_loss": total_sum / sample_count,
        "policy_loss": policy_sum / sample_count,
        "value_loss": value_sum / sample_count,
    }


def build_output_config(
    source_run_dir: Path,
    output_run_dir: Path | None,
    *,
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    config = (
        base_config.model_copy(deep=True)
        if base_config is not None
        else load_training_config(DEFAULT_CONFIG_PATH)
    )
    config.self_play.nn_value_weight_schedule.initial = 1.0
    config.self_play.penalty_schedule.death_penalty = 0.0
    config.self_play.penalty_schedule.overhang_penalty_weight = 0.0
    config.self_play.bootstrap_without_network = False
    config.run.run_name = None
    config.run.run_dir = None
    config.run.checkpoint_dir = None
    config.run.data_dir = None

    return setup_run_directory(
        config,
        base_dir=source_run_dir.parent,
        run_dir=output_run_dir,
    )


def train_warm_start_model(
    model: TetrisNet,
    *,
    dataset_setup: WarmStartDatasetSetup,
    source_offline_resume_checkpoint: Path | None,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    warmup_epochs: float,
    lr_min_factor: float,
    weight_decay: float,
    grad_clip_norm: float,
    epochs_per_round: float,
    early_stopping_patience: int,
    max_rounds: int,
    eval_batch_size: int,
    value_loss_window: int,
    seed: int,
    ema_decay: float,
    mirror_augmentation_probability: float,
) -> WarmStartTrainingResult:
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1) (got {ema_decay})")
    if not 0.0 <= mirror_augmentation_probability <= 1.0:
        raise ValueError(
            "mirror_augmentation_probability must be in [0, 1] "
            f"(got {mirror_augmentation_probability})"
        )
    optimizer = OptimizerBundle(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0.0 else None
    loss_balancer = RunningLossBalancer(value_loss_window)
    current_value_loss_weight = 1.0
    history: list[dict[str, float | int | bool | str]] = []
    best_record: dict[str, float | int | bool | str] | None = None
    best_state_dict = clone_model_state_dict(model)
    best_ema_state_dict = clone_model_state_dict(ema.model) if ema is not None else None
    rng = np.random.default_rng(seed)
    steps_per_round = compute_training_steps(
        len(dataset_setup.train_local_indices),
        batch_size=batch_size,
        epochs=epochs_per_round,
    )
    lr_schedule_total_rounds = resolve_lr_schedule_round_budget(
        early_stopping_patience=early_stopping_patience,
        max_rounds=max_rounds,
    )
    lr_warmup_steps = (
        compute_training_steps(
            len(dataset_setup.train_local_indices),
            batch_size=batch_size,
            epochs=warmup_epochs,
        )
        if warmup_epochs > 0.0
        else 0
    )
    lr_schedule_total_steps = max(steps_per_round * lr_schedule_total_rounds, 1)
    scheduler = build_warm_start_lr_scheduler(
        optimizer,
        warmup_steps=lr_warmup_steps,
        total_steps=lr_schedule_total_steps,
        lr_min_factor=lr_min_factor,
    )
    log_interval = max(1, steps_per_round // 10)
    start_time = time.perf_counter()
    total_steps = 0
    rounds_completed = 0
    non_improving_rounds = 0
    stop_reason = "patience_exhausted"
    resumed_from_checkpoint = False

    if source_offline_resume_checkpoint is not None:
        resume_state = load_offline_resume_checkpoint(
            source_offline_resume_checkpoint,
            model=model,
            ema_model=ema.model if ema is not None else None,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_balancer=loss_balancer,
            lr_schedule_total_steps=lr_schedule_total_steps,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_factor=lr_min_factor,
        )
        if resume_state.rng_state:
            rng.bit_generator.state = resume_state.rng_state
        current_value_loss_weight = resume_state.current_value_loss_weight
        history = [dict(record) for record in resume_state.history]
        best_record = dict(resume_state.best_record)
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in resume_state.best_state_dict.items()
        }
        best_ema_state_dict = (
            {
                key: value.detach().cpu().clone()
                for key, value in resume_state.best_ema_state_dict.items()
            }
            if resume_state.best_ema_state_dict is not None
            else None
        )
        total_steps = resume_state.total_steps
        rounds_completed = resume_state.rounds_completed
        non_improving_rounds = resume_state.non_improving_rounds
        resumed_from_checkpoint = True
        logger.info(
            "Resumed warm-start offline state from source run",
            checkpoint=str(source_offline_resume_checkpoint),
            total_steps=total_steps,
            rounds_completed=rounds_completed,
            non_improving_rounds=non_improving_rounds,
            best_round_index=best_record.get("round_index"),
            best_step=best_record.get("step"),
            best_eval_selection_metric=best_record.get("eval_selection_metric"),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    logger.info(
        "Warm-start offline LR schedule",
        lr_schedule="warmup_cosine",
        warmup_epochs=warmup_epochs,
        lr_warmup_steps=lr_warmup_steps,
        lr_schedule_total_rounds=lr_schedule_total_rounds,
        lr_schedule_total_steps=lr_schedule_total_steps,
        lr_min_factor=lr_min_factor,
        base_learning_rate=learning_rate,
        initial_learning_rate=optimizer.param_groups[0]["lr"],
    )
    log_wandb(
        {
            "offline_step": total_steps,
            "warm_start/lr_schedule_total_rounds": lr_schedule_total_rounds,
            "warm_start/lr_schedule_total_steps": lr_schedule_total_steps,
            "warm_start/lr_warmup_steps": lr_warmup_steps,
            "warm_start/lr_warmup_epochs": warmup_epochs,
            "warm_start/lr_min_factor": lr_min_factor,
            "warm_start/learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    def record_eval(round_index: int, step: int, epochs_seen: float) -> None:
        nonlocal best_record, best_state_dict, best_ema_state_dict
        nonlocal non_improving_rounds
        eval_model = ema.model if ema is not None else model
        eval_uses_ema = ema is not None
        eval_metrics = evaluate_offline_losses(
            eval_model,
            source=dataset_setup.source,
            local_indices=dataset_setup.eval_local_indices,
            device=device,
            eval_batch_size=eval_batch_size,
            value_loss_weight=current_value_loss_weight,
        )
        eval_selection_metric = warm_start_selection_metric(
            eval_metrics["policy_loss"],
            eval_metrics["value_loss"],
        )
        improved = has_better_eval_metric(eval_metrics, best_record)
        if improved:
            best_record = {
                "round_index": round_index,
                "step": step,
                "epochs_seen": epochs_seen,
                "eval_uses_ema": eval_uses_ema,
                "value_loss_weight": current_value_loss_weight,
                "eval_selection_metric": eval_selection_metric,
                "eval_total_loss": eval_metrics["total_loss"],
                "eval_policy_loss": eval_metrics["policy_loss"],
                "eval_value_loss": eval_metrics["value_loss"],
                "improved": True,
                "non_improving_rounds": 0,
            }
            best_state_dict = clone_model_state_dict(model)
            best_ema_state_dict = (
                clone_model_state_dict(eval_model) if ema is not None else None
            )
            non_improving_rounds = 0
        else:
            non_improving_rounds += 1
        record = {
            "round_index": round_index,
            "step": step,
            "epochs_seen": epochs_seen,
            "eval_uses_ema": eval_uses_ema,
            "value_loss_weight": current_value_loss_weight,
            "eval_selection_metric": eval_selection_metric,
            "eval_total_loss": eval_metrics["total_loss"],
            "eval_policy_loss": eval_metrics["policy_loss"],
            "eval_value_loss": eval_metrics["value_loss"],
            "improved": improved,
            "non_improving_rounds": non_improving_rounds,
        }
        history.append(record)
        logger.info(
            "Warm-start offline eval",
            round_index=round_index,
            step=step,
            epochs_seen=epochs_seen,
            eval_selection_metric=eval_selection_metric,
            eval_total_loss=record["eval_total_loss"],
            eval_policy_loss=record["eval_policy_loss"],
            eval_value_loss=record["eval_value_loss"],
            eval_uses_ema=eval_uses_ema,
            value_loss_weight=current_value_loss_weight,
            improved=improved,
            non_improving_rounds=non_improving_rounds,
            early_stopping_patience=early_stopping_patience,
        )
        log_wandb(
            {
                "offline_step": step,
                "warm_start/eval_round_index": round_index,
                "warm_start/eval_epochs_seen": epochs_seen,
                "warm_start/eval_selection_metric": eval_selection_metric,
                "warm_start/value_loss_weight": current_value_loss_weight,
                "warm_start/eval_total_loss": record["eval_total_loss"],
                "warm_start/eval_policy_loss": record["eval_policy_loss"],
                "warm_start/eval_value_loss": record["eval_value_loss"],
                "warm_start/eval_uses_ema": eval_uses_ema,
                "warm_start/eval_improved": improved,
                "warm_start/eval_non_improving_rounds": non_improving_rounds,
            }
        )

    while max_rounds == 0 or rounds_completed < max_rounds:
        round_index = rounds_completed + 1
        for round_step in range(1, steps_per_round + 1):
            positions = rng.integers(
                0, len(dataset_setup.train_local_indices), size=batch_size
            )
            batch_indices = dataset_setup.train_local_indices[positions]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(
                    dataset_setup.source,
                    batch_indices,
                    device,
                )
            )
            boards, aux, policy_targets, action_masks = maybe_mirror_training_tensors(
                boards,
                aux,
                policy_targets,
                action_masks,
                mirror_augmentation_probability,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=current_value_loss_weight,
            )
            total_loss.backward()
            if grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm
                )
            else:
                grad_norm = None
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            total_steps += 1
            policy_loss_scalar = policy_loss.item()
            value_loss_scalar = value_loss.item()
            loss_balancer.append(policy_loss_scalar, value_loss_scalar)
            if loss_balancer.has_history():
                current_value_loss_weight = loss_balancer.value_loss_weight()

            if round_step % log_interval == 0 or round_step == steps_per_round:
                elapsed_sec = time.perf_counter() - start_time
                epochs_seen = (total_steps * batch_size) / len(
                    dataset_setup.train_local_indices
                )
                logger.info(
                    "Warm-start offline train",
                    round_index=round_index,
                    round_step=round_step,
                    steps_per_round=steps_per_round,
                    step=total_steps,
                    epochs_seen=epochs_seen,
                    total_loss=total_loss.item(),
                    policy_loss=policy_loss_scalar,
                    value_loss=value_loss_scalar,
                    value_loss_weight=current_value_loss_weight,
                    grad_norm=(
                        float(grad_norm.item())
                        if grad_norm is not None
                        else float("nan")
                    ),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    elapsed_sec=elapsed_sec,
                )
                log_wandb(
                    {
                        "offline_step": total_steps,
                        "warm_start/train_round_index": round_index,
                        "warm_start/train_round_step": round_step,
                        "warm_start/train_epochs_seen": epochs_seen,
                        "warm_start/train_batch_total_loss": total_loss.item(),
                        "warm_start/train_batch_policy_loss": policy_loss_scalar,
                        "warm_start/train_batch_value_loss": value_loss_scalar,
                        "warm_start/value_loss_weight": current_value_loss_weight,
                        "warm_start/grad_norm": (
                            float(grad_norm.item())
                            if grad_norm is not None
                            else float("nan")
                        ),
                        "warm_start/learning_rate": optimizer.param_groups[0]["lr"],
                        "warm_start/elapsed_sec": elapsed_sec,
                        "warm_start/train_round_progress_fraction": (
                            round_step / steps_per_round
                        ),
                    }
                )

        rounds_completed = round_index
        epochs_seen = (total_steps * batch_size) / len(
            dataset_setup.train_local_indices
        )
        record_eval(round_index=round_index, step=total_steps, epochs_seen=epochs_seen)
        if non_improving_rounds >= early_stopping_patience:
            break
    else:
        stop_reason = "max_rounds_reached"

    if best_record is None:
        raise RuntimeError("Warm-start offline training never produced an evaluation")

    logger.info(
        "Warm-start offline training complete",
        resumed_from_checkpoint=resumed_from_checkpoint,
        rounds_completed=rounds_completed,
        total_steps=total_steps,
        steps_per_round=steps_per_round,
        stop_reason=stop_reason,
        stopped_after_non_improving_rounds=non_improving_rounds,
        best_round_index=best_record["round_index"],
        best_step=best_record["step"],
        best_eval_selection_metric=best_record["eval_selection_metric"],
    )

    return WarmStartTrainingResult(
        best_state_dict=best_state_dict,
        best_ema_state_dict=best_ema_state_dict,
        best_record=best_record,
        history=history,
        ema_state_dict=(clone_model_state_dict(ema.model) if ema is not None else None),
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        scheduler_state_dict=copy.deepcopy(scheduler.state_dict()),
        loss_balancer_state=loss_balancer.state_dict(),
        current_value_loss_weight=current_value_loss_weight,
        rng_state=dict(copy.deepcopy(rng.bit_generator.state)),
        total_steps=total_steps,
        steps_per_round=steps_per_round,
        lr_schedule_total_steps=lr_schedule_total_steps,
        lr_schedule_total_rounds=lr_schedule_total_rounds,
        lr_warmup_steps=lr_warmup_steps,
        rounds_completed=rounds_completed,
        stop_reason=stop_reason,
        stopped_after_non_improving_rounds=non_improving_rounds,
    )


def save_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def log_wandb(payload: dict[str, object]) -> None:
    if wandb.run is not None:
        wandb.log(payload)


def build_wandb_config(
    *,
    args: ScriptArgs,
    source_run_dir: Path,
    source_config_path: Path,
    source_training_data_path: Path,
    source_offline_resume_checkpoint: Path | None,
    resolved_output_config: TrainingConfig,
    device_str: str,
    preload_mode: str,
    batch_size: int,
    learning_rate: float,
    warmup_epochs: float,
    lr_min_factor: float,
    weight_decay: float,
    grad_clip_norm: float,
    eval_worker_resolution: EvalWorkerResolution,
) -> dict[str, object]:
    serialized_output_config = resolved_output_config.model_dump(mode="json")
    return {
        "source_run_dir": str(source_run_dir),
        "source_config_path": str(source_config_path),
        "source_training_data_path": str(source_training_data_path),
        "resume_from_source_offline_state": args.resume_from_source_offline_state,
        "source_offline_resume_checkpoint": (
            str(source_offline_resume_checkpoint)
            if source_offline_resume_checkpoint is not None
            else None
        ),
        "output_run_dir": str(resolved_output_config.run.run_dir),
        "device": device_str,
        "seed": args.seed,
        "epochs_per_round": args.epochs_per_round,
        "early_stopping_patience": args.early_stopping_patience,
        "max_rounds": args.max_rounds,
        "max_examples": args.max_examples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_epochs": warmup_epochs,
        "lr_min_factor": lr_min_factor,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "eval_examples": args.eval_examples,
        "eval_batch_size": args.eval_batch_size,
        "preload_to_gpu": args.preload_to_gpu,
        "preload_to_ram": args.preload_to_ram,
        "preload_mode": preload_mode,
        "num_eval_games": args.num_eval_games,
        "eval_seed_start": args.eval_seed_start,
        "eval_num_workers": eval_worker_resolution.num_workers,
        "eval_num_workers_source": eval_worker_resolution.source,
        "eval_num_simulations": args.eval_num_simulations,
        "eval_max_placements": args.eval_max_placements,
        "mcts_seed": args.mcts_seed,
        "wandb_tags": args.wandb_tags,
        "output_config": serialized_output_config,
        "training_config": serialized_output_config,
    }


def run_warm_start(
    args: ScriptArgs,
    *,
    output_config: TrainingConfig | None = None,
) -> WarmStartRunResult:
    validate_args(args)

    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = (
        args.output_run_dir.resolve() if args.output_run_dir is not None else None
    )
    source_offline_resume_checkpoint = (
        offline_resume_checkpoint_path(source_run_dir)
        if args.resume_from_source_offline_state
        else None
    )
    source_config_path = source_run_dir / CONFIG_FILENAME
    source_training_data_path = source_run_dir / TRAINING_DATA_FILENAME
    resolved_output_config = (
        build_output_config(
            source_run_dir=source_run_dir,
            output_run_dir=output_run_dir,
        )
        if output_config is None
        else output_config.model_copy(deep=True)
    )

    if (
        resolved_output_config.run.run_dir is None
        or resolved_output_config.run.checkpoint_dir is None
    ):
        raise RuntimeError("Output run directory was not set by setup_run_directory")
    if resolved_output_config.run.data_dir is None:
        raise RuntimeError("Output data directory was not set by setup_run_directory")
    save_training_config(
        resolved_output_config,
        resolved_output_config.run.run_dir / CONFIG_FILENAME,
    )

    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else resolved_output_config.optimizer.batch_size
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else resolved_output_config.optimizer.learning_rate
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else resolved_output_config.optimizer.weight_decay
    )
    grad_clip_norm = (
        args.grad_clip_norm
        if args.grad_clip_norm is not None
        else resolved_output_config.optimizer.grad_clip_norm
    )
    eval_worker_resolution = resolve_eval_num_workers(
        args.eval_num_workers,
        default_workers=resolved_output_config.self_play.num_workers,
    )
    resolved_wandb_run_name = (
        args.wandb_run_name
        if args.wandb_run_name is not None
        else (
            f"warm-start-{source_run_dir.name}-to-"
            f"{resolved_output_config.run.run_dir.name}"
        )
    )
    wandb_config = build_wandb_config(
        args=args,
        source_run_dir=source_run_dir,
        source_config_path=source_config_path,
        source_training_data_path=source_training_data_path,
        source_offline_resume_checkpoint=source_offline_resume_checkpoint,
        resolved_output_config=resolved_output_config,
        device_str=device_str,
        preload_mode=preload_mode,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        eval_worker_resolution=eval_worker_resolution,
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=resolved_wandb_run_name,
        tags=args.wandb_tags,
        config=wandb_config,
    )
    wandb.define_metric("offline_step")
    wandb.define_metric("warm_start/*", step_metric="offline_step")
    wandb.define_metric("dataset/*", step_metric="offline_step")
    wandb.define_metric("final_eval/*", step_metric="offline_step")

    logger.info(
        "Starting warm start",
        source_run_dir=str(source_run_dir),
        output_run_dir=str(resolved_output_config.run.run_dir),
        device=device_str,
        resume_from_source_offline_state=args.resume_from_source_offline_state,
        source_offline_resume_checkpoint=(
            str(source_offline_resume_checkpoint)
            if source_offline_resume_checkpoint is not None
            else None
        ),
        epochs_per_round=args.epochs_per_round,
        early_stopping_patience=args.early_stopping_patience,
        max_rounds=args.max_rounds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        preload_mode=preload_mode,
        eval_num_workers=eval_worker_resolution.num_workers,
        eval_num_workers_source=eval_worker_resolution.source,
        eval_num_workers_cache_path=eval_worker_resolution.cache_path,
        wandb_project=args.wandb_project,
        wandb_run_name=resolved_wandb_run_name,
    )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = TetrisNet(**resolved_output_config.network.model_dump()).to(device)

    try:
        npz = np.load(source_training_data_path, mmap_mode="r")
        try:
            dataset_setup = setup_offline_dataset(
                npz,
                seed=args.seed,
                max_examples=args.max_examples,
                eval_examples=args.eval_examples,
                preload_mode=preload_mode,
                device=device,
            )
            log_wandb(
                {
                    "offline_step": 0,
                    "dataset/total_examples": dataset_setup.total_examples,
                    "dataset/used_examples": dataset_setup.num_selected,
                    "dataset/train_examples": len(dataset_setup.train_local_indices),
                    "dataset/eval_examples": len(dataset_setup.eval_local_indices),
                    "dataset/preload_seconds": dataset_setup.preload_sec,
                    "dataset/preload_mode": preload_mode,
                }
            )
            training_result = train_warm_start_model(
                model,
                dataset_setup=dataset_setup,
                source_offline_resume_checkpoint=source_offline_resume_checkpoint,
                device=device,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_epochs=args.warmup_epochs,
                lr_min_factor=args.lr_min_factor,
                weight_decay=weight_decay,
                grad_clip_norm=grad_clip_norm,
                epochs_per_round=args.epochs_per_round,
                early_stopping_patience=args.early_stopping_patience,
                max_rounds=args.max_rounds,
                eval_batch_size=args.eval_batch_size,
                value_loss_window=(
                    resolved_output_config.optimizer.value_loss_weight_window
                ),
                seed=args.seed,
                ema_decay=resolved_output_config.optimizer.ema_decay,
                mirror_augmentation_probability=(
                    resolved_output_config.optimizer.mirror_augmentation_probability
                ),
            )
        finally:
            npz.close()

        offline_resume_checkpoint = offline_resume_checkpoint_path(
            resolved_output_config.run.run_dir
        )
        save_offline_resume_checkpoint(
            offline_resume_checkpoint,
            model=model,
            ema_state_dict=training_result.ema_state_dict,
            optimizer_state_dict=training_result.optimizer_state_dict,
            scheduler_state_dict=training_result.scheduler_state_dict,
            best_state_dict=training_result.best_state_dict,
            best_ema_state_dict=training_result.best_ema_state_dict,
            best_record=training_result.best_record,
            history=training_result.history,
            total_steps=training_result.total_steps,
            rounds_completed=training_result.rounds_completed,
            non_improving_rounds=(training_result.stopped_after_non_improving_rounds),
            current_value_loss_weight=training_result.current_value_loss_weight,
            loss_balancer_state=training_result.loss_balancer_state,
            rng_state=training_result.rng_state,
        )
        logger.info(
            "Saved warm-start offline resume checkpoint",
            checkpoint=str(offline_resume_checkpoint),
            total_steps=training_result.total_steps,
            rounds_completed=training_result.rounds_completed,
            non_improving_rounds=training_result.stopped_after_non_improving_rounds,
        )

        model.load_state_dict(training_result.best_state_dict)
        model.eval()

        output_training_data_path = (
            resolved_output_config.run.data_dir / TRAINING_DATA_FILENAME
        )
        shutil.copy2(source_training_data_path, output_training_data_path)

        trainer = Trainer(resolved_output_config, model=model, device=device_str)
        if (
            trainer.ema_model is not None
            and training_result.best_ema_state_dict is not None
        ):
            trainer.ema_model.load_state_dict(training_result.best_ema_state_dict)
        trainer.step = 0
        initial_checkpoint_state = {
            "incumbent_uses_network": True,
            "incumbent_model_step": 0,
            "incumbent_nn_value_weight": (
                resolved_output_config.self_play.nn_value_weight_schedule.initial
            ),
            "incumbent_death_penalty": (
                resolved_output_config.self_play.penalty_schedule.death_penalty
            ),
            "incumbent_overhang_penalty_weight": (
                resolved_output_config.self_play.penalty_schedule.overhang_penalty_weight
            ),
            "incumbent_eval_avg_attack": 0.0,
            "incumbent_model_source_path": str(
                resolved_output_config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
            ),
            "incumbent_model_artifact": INCUMBENT_ONNX_FILENAME,
        }
        saved_paths = trainer.save(extra_checkpoint_state=initial_checkpoint_state)

        latest_onnx_path = (
            resolved_output_config.run.checkpoint_dir / LATEST_ONNX_FILENAME
        )
        incumbent_onnx_path = (
            resolved_output_config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
        )
        parallel_onnx_path = (
            resolved_output_config.run.checkpoint_dir / PARALLEL_ONNX_FILENAME
        )
        copy_model_artifact_bundle(latest_onnx_path, incumbent_onnx_path)
        copy_model_artifact_bundle(latest_onnx_path, parallel_onnx_path)
        assert_rust_inference_artifacts(incumbent_onnx_path)
        assert_rust_inference_artifacts(parallel_onnx_path)

        eval_config = resolved_output_config.self_play.model_copy(deep=True)
        eval_num_workers = eval_worker_resolution.num_workers
        eval_num_simulations = (
            args.eval_num_simulations
            if args.eval_num_simulations is not None
            else eval_config.num_simulations
        )
        eval_max_placements = (
            args.eval_max_placements
            if args.eval_max_placements is not None
            else eval_config.max_placements
        )
        resolved_mcts_seed = (
            args.mcts_seed if args.mcts_seed is not None else eval_config.mcts_seed
        )
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = eval_num_simulations
        mcts_config.c_puct = eval_config.c_puct
        mcts_config.temperature = eval_config.temperature
        mcts_config.dirichlet_alpha = eval_config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = eval_config.dirichlet_epsilon
        mcts_config.visit_sampling_epsilon = 0.0
        mcts_config.max_placements = eval_max_placements
        mcts_config.reuse_tree = eval_config.reuse_tree
        mcts_config.use_parent_value_for_unvisited_q = (
            eval_config.use_parent_value_for_unvisited_q
        )
        mcts_config.nn_value_weight = (
            resolved_output_config.self_play.nn_value_weight_schedule.initial
        )
        mcts_config.death_penalty = (
            resolved_output_config.self_play.penalty_schedule.death_penalty
        )
        mcts_config.overhang_penalty_weight = (
            resolved_output_config.self_play.penalty_schedule.overhang_penalty_weight
        )
        mcts_config.seed = resolved_mcts_seed

        eval_seeds = list(
            range(args.eval_seed_start, args.eval_seed_start + args.num_eval_games)
        )
        logger.info(
            "Starting warm-start final MCTS eval",
            model_path=str(incumbent_onnx_path),
            num_games=len(eval_seeds),
            seed_start=eval_seeds[0],
            seed_end=eval_seeds[-1],
            num_workers=eval_num_workers,
            num_simulations=eval_num_simulations,
            max_placements=eval_max_placements,
            mcts_seed=resolved_mcts_seed,
            nn_value_weight=resolved_output_config.self_play.nn_value_weight_schedule.initial,
            death_penalty=resolved_output_config.self_play.penalty_schedule.death_penalty,
            overhang_penalty_weight=(
                resolved_output_config.self_play.penalty_schedule.overhang_penalty_weight
            ),
        )
        eval_result = evaluate_model(
            model_path=str(incumbent_onnx_path),
            seeds=eval_seeds,
            config=mcts_config,
            max_placements=eval_max_placements,
            num_workers=eval_num_workers,
            add_noise=False,
        )
        eval_metrics = {
            "num_games": int(eval_result.num_games),
            "avg_attack": float(eval_result.avg_attack),
            "avg_lines": float(eval_result.avg_lines),
            "avg_moves": float(eval_result.avg_moves),
            "max_attack": int(eval_result.max_attack),
            "attack_per_piece": float(eval_result.attack_per_piece),
            "avg_tree_nodes": float(eval_result.avg_tree_nodes),
            "avg_trajectory_predicted_total_attack_variance": float(
                eval_result.avg_trajectory_predicted_total_attack_variance
            ),
            "avg_trajectory_predicted_total_attack_std": float(
                eval_result.avg_trajectory_predicted_total_attack_std
            ),
            "avg_trajectory_predicted_total_attack_rmse": float(
                eval_result.avg_trajectory_predicted_total_attack_rmse
            ),
            "game_results": [
                {"attack": int(attack), "moves": int(moves)}
                for attack, moves in eval_result.game_results
            ],
        }
        log_wandb(
            {
                "offline_step": training_result.total_steps,
                "final_eval/num_games": eval_metrics["num_games"],
                "final_eval/avg_attack": eval_metrics["avg_attack"],
                "final_eval/avg_lines": eval_metrics["avg_lines"],
                "final_eval/avg_moves": eval_metrics["avg_moves"],
                "final_eval/max_attack": eval_metrics["max_attack"],
                "final_eval/attack_per_piece": eval_metrics["attack_per_piece"],
                "final_eval/avg_tree_nodes": eval_metrics["avg_tree_nodes"],
                "final_eval/avg_trajectory_predicted_total_attack_variance": (
                    eval_metrics["avg_trajectory_predicted_total_attack_variance"]
                ),
                "final_eval/avg_trajectory_predicted_total_attack_std": (
                    eval_metrics["avg_trajectory_predicted_total_attack_std"]
                ),
                "final_eval/avg_trajectory_predicted_total_attack_rmse": (
                    eval_metrics["avg_trajectory_predicted_total_attack_rmse"]
                ),
                "final_eval/seed_start": args.eval_seed_start,
                "final_eval/mcts_seed": resolved_mcts_seed,
                "final_eval/num_workers": eval_num_workers,
                "final_eval/num_workers_source": eval_worker_resolution.source,
                "final_eval/num_simulations": eval_num_simulations,
                "final_eval/max_placements": eval_max_placements,
            }
        )

        checkpoint_path = saved_paths["checkpoint"]
        save_checkpoint(
            trainer.model,
            trainer.ema_model,
            trainer.optimizer,
            trainer.scheduler,
            step=0,
            filepath=checkpoint_path,
            incumbent_uses_network=True,
            incumbent_model_step=0,
            incumbent_nn_value_weight=(
                resolved_output_config.self_play.nn_value_weight_schedule.initial
            ),
            incumbent_death_penalty=(
                resolved_output_config.self_play.penalty_schedule.death_penalty
            ),
            incumbent_overhang_penalty_weight=(
                resolved_output_config.self_play.penalty_schedule.overhang_penalty_weight
            ),
            incumbent_eval_avg_attack=eval_metrics["avg_attack"],
            incumbent_model_source_path=str(incumbent_onnx_path),
            incumbent_model_artifact=INCUMBENT_ONNX_FILENAME,
        )
        export_metadata(
            resolved_output_config.run.checkpoint_dir / LATEST_METADATA_FILENAME,
            step=0,
            eval_metrics={
                "warm_start_eval": eval_metrics,
                "offline_best": training_result.best_record,
            },
            config=resolved_output_config.model_dump(mode="json"),
        )

        summary = {
            "source_run_dir": str(source_run_dir),
            "resume_from_source_offline_state": args.resume_from_source_offline_state,
            "source_offline_resume_checkpoint": (
                str(source_offline_resume_checkpoint)
                if source_offline_resume_checkpoint is not None
                else None
            ),
            "output_run_dir": str(resolved_output_config.run.run_dir),
            "source_training_data_path": str(source_training_data_path),
            "copied_training_data_path": str(output_training_data_path),
            "device": device_str,
            "epochs_per_round": args.epochs_per_round,
            "early_stopping_patience": args.early_stopping_patience,
            "max_rounds": args.max_rounds,
            "steps_per_round": training_result.steps_per_round,
            "num_steps": training_result.total_steps,
            "rounds_completed": training_result.rounds_completed,
            "stop_reason": training_result.stop_reason,
            "stopped_after_non_improving_rounds": (
                training_result.stopped_after_non_improving_rounds
            ),
            "seed": args.seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_epochs": args.warmup_epochs,
            "lr_min_factor": args.lr_min_factor,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "lr_schedule_total_rounds": training_result.lr_schedule_total_rounds,
            "lr_schedule_total_steps": training_result.lr_schedule_total_steps,
            "lr_warmup_steps": training_result.lr_warmup_steps,
            "ema_decay": resolved_output_config.optimizer.ema_decay,
            "max_examples": args.max_examples,
            "preload_mode": preload_mode,
            "offline_dataset": {
                "total_examples": dataset_setup.total_examples,
                "used_examples": dataset_setup.num_selected,
                "train_examples": len(dataset_setup.train_local_indices),
                "eval_examples": len(dataset_setup.eval_local_indices),
                "preload_seconds": dataset_setup.preload_sec,
            },
            "offline_best": training_result.best_record,
            "offline_history": training_result.history,
            "final_eval": {
                **eval_metrics,
                "seed_start": args.eval_seed_start,
                "mcts_seed": resolved_mcts_seed,
                "num_workers": eval_num_workers,
                "num_workers_source": eval_worker_resolution.source,
                "num_simulations": eval_num_simulations,
                "max_placements": eval_max_placements,
            },
            "artifacts": {
                "offline_resume_checkpoint": str(offline_resume_checkpoint),
                "latest_checkpoint": str(checkpoint_path),
                "latest_onnx": str(latest_onnx_path),
                "incumbent_onnx": str(incumbent_onnx_path),
                "parallel_onnx": str(parallel_onnx_path),
            },
        }
        summary_path = (
            resolved_output_config.run.run_dir / "analysis" / "warm_start_summary.json"
        )
        save_summary(summary_path, summary)
        log_wandb(
            {
                "offline_step": training_result.total_steps,
                "warm_start/best_round_index": training_result.best_record[
                    "round_index"
                ],
                "warm_start/best_step": training_result.best_record["step"],
                "warm_start/best_epochs_seen": training_result.best_record[
                    "epochs_seen"
                ],
                "warm_start/best_eval_selection_metric": training_result.best_record[
                    "eval_selection_metric"
                ],
                "warm_start/best_eval_total_loss": training_result.best_record[
                    "eval_total_loss"
                ],
                "warm_start/rounds_completed": training_result.rounds_completed,
                "warm_start/stop_reason": training_result.stop_reason,
            }
        )

        logger.info(
            "Warm start complete",
            output_run_dir=str(resolved_output_config.run.run_dir),
            incumbent_onnx=str(incumbent_onnx_path),
            avg_attack=eval_metrics["avg_attack"],
            num_eval_games=eval_metrics["num_games"],
            rounds_completed=training_result.rounds_completed,
            stop_reason=training_result.stop_reason,
            best_round_index=training_result.best_record["round_index"],
        )
        return WarmStartRunResult(
            output_run_dir=resolved_output_config.run.run_dir,
            checkpoint_dir=resolved_output_config.run.checkpoint_dir,
            latest_checkpoint_path=checkpoint_path,
            latest_onnx_path=latest_onnx_path,
            incumbent_onnx_path=incumbent_onnx_path,
            parallel_onnx_path=parallel_onnx_path,
            summary_path=summary_path,
            summary=summary,
        )
    finally:
        wandb.finish()


def main(args: ScriptArgs) -> None:
    run_warm_start(args)


if __name__ == "__main__":
    main(parse(ScriptArgs))
