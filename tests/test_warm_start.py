from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml

import tetris_bot.scripts.warm_start as warm_start_module
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CONFIG_FILENAME,
    DEFAULT_CONFIG_PATH,
    NUM_ACTIONS,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.ema import ExponentialMovingAverage
from tetris_bot.ml.loss import RunningLossBalancer
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.optimizer import OptimizerBundle
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
)
from tetris_bot.scripts.warm_start import (
    align_warm_start_lr_scheduler_to_step,
    build_warm_start_lr_scheduler,
    build_output_config,
    build_wandb_config,
    compute_warmup_cosine_lr_factor,
    compute_training_steps,
    EvalWorkerResolution,
    has_better_eval_metric,
    load_offline_resume_checkpoint,
    offline_resume_checkpoint_path,
    optimized_worker_env_cache_path,
    resolve_eval_num_workers,
    save_offline_resume_checkpoint,
    ScriptArgs,
    train_warm_start_model,
    validate_args,
    WarmStartDatasetSetup,
    warm_start_selection_metric,
)

_DEFAULT_CONFIG = load_training_config(DEFAULT_CONFIG_PATH)
_DEFAULT_NETWORK = _DEFAULT_CONFIG.network


def test_load_training_config_round_trips_complete_yaml(
    tmp_path: Path,
) -> None:
    config = _DEFAULT_CONFIG.model_copy(deep=True)
    config.network = config.network.model_copy(
        update={
            "trunk_channels": 4,
            "num_conv_residual_blocks": 1,
            "reduction_channels": 8,
            "fc_hidden": 128,
        }
    )
    config.run = config.run.model_copy(
        update={
            "run_name": "v3",
            "run_dir": Path("/tmp/example/v3"),
            "checkpoint_dir": Path("/tmp/example/v3/checkpoints"),
            "data_dir": Path("/tmp/example/v3"),
        }
    )
    config_path = tmp_path / CONFIG_FILENAME
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )

    loaded = load_training_config(config_path)

    assert loaded.network.trunk_channels == 4
    assert loaded.network.board_stats_hidden == config.network.board_stats_hidden
    assert loaded.network.board_proj_hidden == config.network.board_proj_hidden
    assert loaded.network.aux_hidden == config.network.aux_hidden
    assert loaded.network.fusion_hidden == config.network.fusion_hidden
    assert loaded.network.num_fusion_blocks == config.network.num_fusion_blocks
    assert loaded.run.run_dir == Path("/tmp/example/v3")
    assert loaded.run.checkpoint_dir == Path("/tmp/example/v3/checkpoints")
    assert loaded.run.data_dir == Path("/tmp/example/v3")


def test_build_output_config_uses_current_repo_defaults_for_new_run(
    tmp_path: Path,
) -> None:
    default_config = _DEFAULT_CONFIG
    output_run_dir = tmp_path / "training_runs" / "v4"

    output_config = build_output_config(
        source_run_dir=tmp_path / "training_runs" / "v3",
        output_run_dir=output_run_dir,
    )

    assert output_config.self_play.nn_value_weight_schedule.initial == 1.0
    assert output_config.self_play.penalty_schedule.death_penalty == 0.0
    assert output_config.self_play.penalty_schedule.overhang_penalty_weight == 0.0
    assert output_config.self_play.bootstrap_without_network is False
    assert (
        output_config.self_play.num_simulations
        == default_config.self_play.num_simulations
    )
    assert output_config.self_play.num_workers == default_config.self_play.num_workers
    assert output_config.network.trunk_channels == default_config.network.trunk_channels
    assert (
        output_config.network.num_conv_residual_blocks
        == default_config.network.num_conv_residual_blocks
    )
    assert (
        output_config.network.reduction_channels
        == default_config.network.reduction_channels
    )
    assert (
        output_config.network.board_stats_hidden
        == default_config.network.board_stats_hidden
    )
    assert (
        output_config.network.board_proj_hidden
        == default_config.network.board_proj_hidden
    )
    assert output_config.network.fc_hidden == default_config.network.fc_hidden
    assert output_config.network.aux_hidden == default_config.network.aux_hidden
    assert output_config.network.fusion_hidden == default_config.network.fusion_hidden
    assert (
        output_config.network.num_fusion_blocks
        == default_config.network.num_fusion_blocks
    )
    assert output_config.optimizer.batch_size == default_config.optimizer.batch_size
    assert (
        output_config.optimizer.learning_rate == default_config.optimizer.learning_rate
    )
    assert output_config.optimizer.weight_decay == default_config.optimizer.weight_decay
    assert output_config.replay.buffer_size == default_config.replay.buffer_size
    assert output_config.run.project_name == default_config.run.project_name
    assert (
        output_config.run.model_sync_interval_seconds
        == default_config.run.model_sync_interval_seconds
    )
    assert (
        output_config.run.checkpoint_interval_seconds
        == default_config.run.checkpoint_interval_seconds
    )
    assert (
        output_config.run.log_interval_seconds
        == default_config.run.log_interval_seconds
    )
    assert (
        output_config.run.save_interval_seconds
        == default_config.run.save_interval_seconds
    )
    assert output_config.run.run_name == "v4"
    assert output_config.run.run_dir == output_run_dir
    assert output_config.run.checkpoint_dir == output_run_dir / "checkpoints"
    assert output_config.run.data_dir == output_run_dir

    saved_config = yaml.safe_load((output_run_dir / CONFIG_FILENAME).read_text())
    assert (
        saved_config["network"]["trunk_channels"]
        == default_config.network.trunk_channels
    )
    assert (
        saved_config["optimizer"]["batch_size"] == default_config.optimizer.batch_size
    )
    assert saved_config["self_play"]["nn_value_weight_schedule"]["initial"] == 1.0
    assert saved_config["self_play"]["penalty_schedule"]["death_penalty"] == 0.0
    assert (
        saved_config["self_play"]["penalty_schedule"]["overhang_penalty_weight"] == 0.0
    )


def test_validate_args_does_not_require_source_config_when_npz_exists(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "training_data.npz").write_bytes(b"placeholder")

    validate_args(
        ScriptArgs(
            source_run_dir=source_run_dir,
            output_run_dir=tmp_path / "training_runs" / "v6",
            preload_to_gpu=False,
        )
    )


def test_build_wandb_config_includes_full_resolved_training_config(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    output_run_dir = tmp_path / "training_runs" / "v6"
    resolved_output_config = build_output_config(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
    )
    args = ScriptArgs(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        preload_to_gpu=False,
    )
    eval_worker_resolution = EvalWorkerResolution(num_workers=8, source="config")

    wandb_config = build_wandb_config(
        args=args,
        source_run_dir=source_run_dir,
        source_config_path=source_run_dir / CONFIG_FILENAME,
        source_training_data_path=source_run_dir / "training_data.npz",
        source_offline_resume_checkpoint=None,
        resolved_output_config=resolved_output_config,
        device_str="cpu",
        preload_mode="none",
        batch_size=resolved_output_config.optimizer.batch_size,
        learning_rate=resolved_output_config.optimizer.learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        weight_decay=resolved_output_config.optimizer.weight_decay,
        grad_clip_norm=resolved_output_config.optimizer.grad_clip_norm,
        eval_worker_resolution=eval_worker_resolution,
    )

    serialized_output_config = resolved_output_config.model_dump(mode="json")
    assert wandb_config["output_config"] == serialized_output_config
    assert wandb_config["training_config"] == serialized_output_config
    assert wandb_config["warmup_epochs"] == args.warmup_epochs
    assert wandb_config["lr_min_factor"] == args.lr_min_factor


def test_compute_training_steps_rounds_up_epochs() -> None:
    assert compute_training_steps(900, batch_size=1024, epochs=20.0) == 18


def test_build_warm_start_lr_scheduler_matches_reference_curve() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerBundle(model, learning_rate=1.0, weight_decay=0.0)
    scheduler = build_warm_start_lr_scheduler(
        optimizer,
        warmup_steps=3,
        total_steps=6,
        lr_min_factor=0.1,
    )

    expected_lrs = [
        compute_warmup_cosine_lr_factor(
            step=step,
            warmup_steps=3,
            total_steps=6,
            lr_min_factor=0.1,
        )
        for step in range(7)
    ]
    actual_lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        actual_lrs.append(optimizer.param_groups[0]["lr"])

    assert actual_lrs == expected_lrs


def test_align_warm_start_lr_scheduler_to_step_restores_lr() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerBundle(model, learning_rate=2.0, weight_decay=0.0)
    scheduler = build_warm_start_lr_scheduler(
        optimizer,
        warmup_steps=3,
        total_steps=8,
        lr_min_factor=0.1,
    )

    align_warm_start_lr_scheduler_to_step(
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        warmup_steps=3,
        total_steps=8,
        lr_min_factor=0.1,
    )

    expected_lr = 2.0 * compute_warmup_cosine_lr_factor(
        step=5,
        warmup_steps=3,
        total_steps=8,
        lr_min_factor=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert scheduler.last_epoch == 5


def test_resolve_eval_num_workers_uses_machine_optimize_cache(
    tmp_path: Path,
) -> None:
    cache_path = optimized_worker_env_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("TETRIS_OPT_NUM_WORKERS=12\n")

    resolution = resolve_eval_num_workers(
        0,
        default_workers=6,
        cache_dir=tmp_path,
    )

    assert resolution.num_workers == 12
    assert resolution.source == "optimize_cache"
    assert resolution.cache_path == str(cache_path)


def test_resolve_eval_num_workers_prefers_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = optimized_worker_env_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("TETRIS_OPT_NUM_WORKERS=12\n")
    monkeypatch.setenv("TETRIS_OPT_NUM_WORKERS", "10")

    resolution = resolve_eval_num_workers(
        0,
        default_workers=6,
        cache_dir=tmp_path,
    )

    assert resolution.num_workers == 10
    assert resolution.source == "environment"
    assert resolution.cache_path is None


def test_warm_start_selection_metric_weights_value_loss_by_quarter() -> None:
    assert warm_start_selection_metric(1.5, 10.0) == 4.0


def test_has_better_eval_metric_uses_fixed_selection_metric() -> None:
    best_record: dict[str, float | int | bool | str] = {
        "eval_selection_metric": 4.0,
        "eval_policy_loss": 1.5,
        "eval_value_loss": 10.0,
    }

    assert (
        has_better_eval_metric(
            {"policy_loss": 1.4, "value_loss": 9.9},
            best_record,
        )
        is True
    )
    assert (
        has_better_eval_metric(
            {"policy_loss": 1.55, "value_loss": 9.0},
            best_record,
        )
        is True
    )
    assert (
        has_better_eval_metric(
            {"policy_loss": 1.4, "value_loss": 10.8},
            best_record,
        )
        is False
    )


def test_validate_args_requires_offline_resume_checkpoint_when_requested(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / CONFIG_FILENAME).write_text(yaml.safe_dump({}))
    (source_run_dir / "training_data.npz").write_bytes(b"placeholder")

    args = ScriptArgs(
        source_run_dir=source_run_dir,
        output_run_dir=tmp_path / "training_runs" / "v6",
        resume_from_source_offline_state=True,
        preload_to_gpu=False,
    )

    try:
        validate_args(args)
    except FileNotFoundError as error:
        assert "offline warm-start resume checkpoint" in str(error)
    else:
        raise AssertionError(
            "validate_args should require an offline resume checkpoint"
        )


def test_offline_resume_checkpoint_round_trip_restores_optimizer_and_history(
    tmp_path: Path,
) -> None:
    checkpoint_path = offline_resume_checkpoint_path(tmp_path / "training_runs" / "v5")
    model = TetrisNet(**_DEFAULT_NETWORK.to_model_kwargs())
    ema = ExponentialMovingAverage(model, decay=0.5)
    optimizer = OptimizerBundle(model, learning_rate=5e-4, weight_decay=0.0)
    scheduler = build_warm_start_lr_scheduler(
        optimizer,
        warmup_steps=4,
        total_steps=12,
        lr_min_factor=0.1,
    )
    loss_balancer = RunningLossBalancer(window_size=8)
    loss_balancer.append(2.0, 8.0)
    loss_balancer.append(1.8, 7.2)

    boards = torch.randn(2, 1, 20, 10)
    aux = torch.randn(2, 80)
    policy_logits, value = model(boards, aux)
    loss = policy_logits.sum() + value.sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    best_state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    with torch.no_grad():
        for parameter in ema.model.parameters():
            parameter.add_(0.25)
    ema_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in ema.model.state_dict().items()
    }
    best_ema_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in ema.model.state_dict().items()
    }
    best_record = {
        "round_index": 4,
        "step": 14064,
        "val_selection_metric": 4.25,
    }
    history = [{"round_index": 4, "step": 14064, "improved": True}]
    rng_state = {"seed": 123, "note": "test"}

    save_offline_resume_checkpoint(
        checkpoint_path,
        model=model,
        ema_state_dict=ema_state_dict,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        best_state_dict=best_state_dict,
        best_ema_state_dict=best_ema_state_dict,
        best_record=best_record,
        history=history,
        total_steps=14064,
        rounds_completed=4,
        non_improving_rounds=2,
        current_value_loss_weight=0.25,
        loss_balancer_state=loss_balancer.state_dict(),
        rng_state=rng_state,
    )

    restored_model = TetrisNet(**_DEFAULT_NETWORK.to_model_kwargs())
    restored_ema = ExponentialMovingAverage(restored_model, decay=0.5)
    restored_optimizer = OptimizerBundle(
        restored_model, learning_rate=5e-4, weight_decay=0.0
    )
    restored_scheduler = build_warm_start_lr_scheduler(
        restored_optimizer,
        warmup_steps=4,
        total_steps=12,
        lr_min_factor=0.1,
    )
    restored_loss_balancer = RunningLossBalancer(window_size=8)
    resumed = load_offline_resume_checkpoint(
        checkpoint_path,
        model=restored_model,
        ema_model=restored_ema.model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        loss_balancer=restored_loss_balancer,
        lr_schedule_total_steps=12,
        lr_warmup_steps=4,
        lr_min_factor=0.1,
    )

    assert resumed.checkpoint_path == checkpoint_path
    assert resumed.total_steps == 14064
    assert resumed.rounds_completed == 4
    assert resumed.non_improving_rounds == 2
    assert resumed.current_value_loss_weight == 0.25
    assert resumed.best_record == best_record
    assert resumed.history == history
    assert resumed.rng_state == rng_state
    assert resumed.best_state_dict.keys() == best_state_dict.keys()
    assert resumed.best_ema_state_dict is not None
    assert resumed.best_ema_state_dict.keys() == best_ema_state_dict.keys()
    assert restored_loss_balancer.state_dict() == loss_balancer.state_dict()
    restored_ema_state = restored_ema.model.state_dict()
    for name, expected_value in ema_state_dict.items():
        assert torch.equal(restored_ema_state[name], expected_value)
    for name, expected_value in best_ema_state_dict.items():
        assert torch.equal(resumed.best_ema_state_dict[name], expected_value)
    restored_optimizer_state = restored_optimizer.state_dict()["state"]
    expected_optimizer_state = optimizer.state_dict()["state"]
    assert restored_optimizer_state.keys() == expected_optimizer_state.keys()
    for parameter_id, expected_state in expected_optimizer_state.items():
        restored_state = restored_optimizer_state[parameter_id]
        assert restored_state.keys() == expected_state.keys()
        for state_key, expected_value in expected_state.items():
            restored_value = restored_state[state_key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(restored_value, expected_value)
            else:
                assert restored_value == expected_value
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_train_warm_start_model_evaluates_and_snapshots_ema_weights(
    monkeypatch,
) -> None:
    model = TetrisNet(**_DEFAULT_NETWORK.to_model_kwargs())
    dataset_setup = WarmStartDatasetSetup(
        source=OfflineDataSource(
            npz=cast(np.lib.npyio.NpzFile, object()),
            selected_global_indices=np.array([0, 1], dtype=np.int64),
            tensor_data=None,
        ),
        train_local_indices=np.array([0, 1], dtype=np.int64),
        eval_local_indices=np.array([0], dtype=np.int64),
        total_examples=2,
        num_selected=2,
        preload_sec=0.0,
    )

    boards = torch.randn(2, 1, BOARD_HEIGHT, BOARD_WIDTH)
    aux = torch.randn(2, 80)
    policy_targets = torch.zeros(2, NUM_ACTIONS)
    policy_targets[:, 0] = 1.0
    value_targets = torch.tensor([0.25, -0.25], dtype=torch.float32)
    action_masks = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    captured: dict[str, bool] = {}

    def fake_build_torch_batch(source, batch_indices, device):
        del source, batch_indices
        return (
            boards.to(device),
            aux.to(device),
            policy_targets.to(device),
            value_targets.to(device),
            action_masks.to(device),
        )

    def fake_evaluate_offline_losses(eval_model, **kwargs):
        del kwargs
        raw_state = model.state_dict()
        captured["eval_used_raw_model"] = all(
            torch.equal(eval_model.state_dict()[name], raw_state[name])
            for name in raw_state
        )
        return {
            "total_loss": 1.0,
            "policy_loss": 0.75,
            "value_loss": 0.5,
        }

    monkeypatch.setattr(warm_start_module, "build_torch_batch", fake_build_torch_batch)
    monkeypatch.setattr(
        warm_start_module,
        "evaluate_offline_losses",
        fake_evaluate_offline_losses,
    )

    training_result = train_warm_start_model(
        model,
        dataset_setup=dataset_setup,
        source_offline_resume_checkpoint=None,
        device=torch.device("cpu"),
        batch_size=2,
        learning_rate=1e-2,
        warmup_epochs=3.0,
        lr_min_factor=0.1,
        weight_decay=0.0,
        grad_clip_norm=1e6,
        epochs_per_round=1.0,
        early_stopping_patience=1,
        max_rounds=1,
        eval_batch_size=1,
        value_loss_window=4,
        seed=123,
        ema_decay=0.5,
        mirror_augmentation_probability=0.0,
    )

    assert captured["eval_used_raw_model"] is False
    assert training_result.best_record["eval_uses_ema"] is True
    assert training_result.best_ema_state_dict is not None
    assert training_result.ema_state_dict is not None
    assert any(
        not torch.equal(
            training_result.best_state_dict[name],
            training_result.best_ema_state_dict[name],
        )
        for name in training_result.best_state_dict
    )
