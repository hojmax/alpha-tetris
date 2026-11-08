from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import json
import logging
from queue import Empty, Queue
import struct
from threading import Lock, Thread
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import structlog
from torch import nn

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_FILENAME_PREFIX,
    LATEST_CHECKPOINT_FILENAME,
    LATEST_METADATA_FILENAME,
    LATEST_ONNX_FILENAME,
)
from tetris_bot.ml.network import (
    AUX_FEATURES,
    PIECE_AUX_FEATURES,
    ConvBackbone,
    HeadsModel,
    TetrisNet,
)
from tetris_bot.ml.optimizer import OptimizerBundle, SchedulerBundle

OptimizerLike = torch.optim.Optimizer | OptimizerBundle
SchedulerLike = torch.optim.lr_scheduler.LRScheduler | SchedulerBundle

logger = structlog.get_logger()
FC_BINARY_MAGIC = b"TCM2"


def _write_linear_layer(buffer: io.BufferedWriter, layer: nn.Linear) -> None:
    weight = layer.weight.detach().cpu().numpy().astype(np.float32)
    bias = layer.bias.detach().cpu().numpy().astype(np.float32)
    rows, cols = weight.shape
    buffer.write(struct.pack("<II", rows, cols))
    buffer.write(weight.tobytes())
    buffer.write(bias.tobytes())


def _write_layer_norm(buffer: io.BufferedWriter, layer: nn.LayerNorm) -> None:
    if len(layer.normalized_shape) != 1:
        raise ValueError(
            "Only 1D LayerNorm export is supported for cached board path "
            f"(got normalized_shape={layer.normalized_shape})"
        )
    hidden = int(layer.normalized_shape[0])
    weight = layer.weight.detach().cpu().numpy().astype(np.float32)
    bias = layer.bias.detach().cpu().numpy().astype(np.float32)
    buffer.write(struct.pack("<I", hidden))
    buffer.write(weight.tobytes())
    buffer.write(bias.tobytes())


def _export_cached_board_path_binary(model: TetrisNet, fc_path: Path) -> None:
    with open(fc_path, "wb") as f:
        f.write(FC_BINARY_MAGIC)
        _write_linear_layer(f, model.board_stats_fc)
        _write_layer_norm(f, model.board_stats_ln)
        _write_linear_layer(f, model.board_proj_fc1)
        _write_layer_norm(f, model.board_proj_ln1)
        _write_linear_layer(f, model.board_proj_fc2)


def _optimizer_step_scalar_dtype(*, fused: bool) -> torch.dtype:
    if fused:
        return torch.float32
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )


def sanitize_optimizer_state_steps(optimizer: object) -> int:
    """Normalize per-parameter optimizer step counters to PyTorch's tensor form."""
    inner_optimizers = getattr(optimizer, "inner_optimizers", None)
    if inner_optimizers is not None:
        return sum(sanitize_optimizer_state_steps(inner) for inner in inner_optimizers)
    normalized_steps = 0
    cpu_device = torch.device("cpu")
    for group in optimizer.param_groups:
        fused = bool(group.get("fused", False))
        capturable = bool(group.get("capturable", False))
        expected_dtype = _optimizer_step_scalar_dtype(fused=fused)
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "step" not in state:
                continue
            expected_device = parameter.device if capturable or fused else cpu_device
            step = state["step"]
            if torch.is_tensor(step):
                if step.dtype == expected_dtype and step.device == expected_device:
                    continue
                state["step"] = step.to(device=expected_device, dtype=expected_dtype)
            else:
                state["step"] = torch.tensor(
                    float(step),
                    dtype=expected_dtype,
                    device=expected_device,
                )
            normalized_steps += 1
    return normalized_steps


def load_optimizer_state_dict(
    optimizer: OptimizerLike,
    optimizer_state_dict: dict[str, Any],
    *,
    source: str | Path | None = None,
) -> None:
    optimizer.load_state_dict(optimizer_state_dict)
    normalized_steps = sanitize_optimizer_state_steps(optimizer)
    if normalized_steps > 0:
        logger.warning(
            "Sanitized optimizer step counters after optimizer restore",
            source=str(source) if source is not None else None,
            normalized_steps=normalized_steps,
        )


@dataclass(frozen=True)
class CheckpointSnapshot:
    step: int
    model_state_dict: dict[str, Any]
    ema_state_dict: dict[str, Any] | None
    optimizer_state_dict: dict[str, Any] | None
    # SchedulerBundle.state_dict returns a list (one entry per inner scheduler);
    # plain torch schedulers return a dict. Both are accepted.
    scheduler_state_dict: dict[str, Any] | list[dict[str, Any]] | None
    extra_state: dict[str, object]


@dataclass(frozen=True)
class CheckpointSaveRequest:
    snapshot: CheckpointSnapshot
    model_kwargs: dict[str, Any]
    eval_metrics: dict[str, Any] | None
    export_for_rust: bool


def clone_state_value_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return type(value)(
            (key, clone_state_value_to_cpu(item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [clone_state_value_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_value_to_cpu(item) for item in value)
    return value


def capture_checkpoint_snapshot(
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: OptimizerLike | None,
    scheduler: SchedulerLike | None,
    step: int,
    extra_checkpoint_state: dict[str, object] | None = None,
) -> CheckpointSnapshot:
    return CheckpointSnapshot(
        step=step,
        model_state_dict=clone_state_value_to_cpu(model.state_dict()),
        ema_state_dict=(
            clone_state_value_to_cpu(ema_model.state_dict())
            if ema_model is not None
            else None
        ),
        optimizer_state_dict=(
            clone_state_value_to_cpu(optimizer.state_dict())
            if optimizer is not None
            else None
        ),
        scheduler_state_dict=(
            clone_state_value_to_cpu(scheduler.state_dict())
            if scheduler is not None
            else None
        ),
        extra_state=dict(extra_checkpoint_state or {}),
    )


def save_checkpoint_snapshot(
    snapshot: CheckpointSnapshot, filepath: str | Path
) -> None:
    state = {
        "model_state_dict": snapshot.model_state_dict,
        "step": snapshot.step,
        **snapshot.extra_state,
    }
    if snapshot.ema_state_dict is not None:
        state["ema_state_dict"] = snapshot.ema_state_dict
    if snapshot.optimizer_state_dict is not None:
        state["optimizer_state_dict"] = snapshot.optimizer_state_dict
    if snapshot.scheduler_state_dict is not None:
        state["scheduler_state_dict"] = snapshot.scheduler_state_dict
    torch.save(state, filepath)


def save_checkpoint(
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: OptimizerLike | None,
    scheduler: SchedulerLike | None,
    step: int,
    filepath: str | Path,
    **extra_state,
) -> None:
    save_checkpoint_snapshot(
        CheckpointSnapshot(
            step=step,
            model_state_dict=model.state_dict(),
            ema_state_dict=ema_model.state_dict() if ema_model is not None else None,
            optimizer_state_dict=(
                optimizer.state_dict() if optimizer is not None else None
            ),
            scheduler_state_dict=(
                scheduler.state_dict() if scheduler is not None else None
            ),
            extra_state=extra_state,
        ),
        filepath,
    )


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module | None = None,
    ema_model: nn.Module | None = None,
    optimizer: OptimizerLike | None = None,
    scheduler: SchedulerLike | None = None,
) -> dict:
    state = torch.load(filepath, map_location="cpu", weights_only=True)

    if model is not None:
        model.load_state_dict(state["model_state_dict"])
    if ema_model is not None:
        ema_state_dict = state.get("ema_state_dict")
        if ema_state_dict is None:
            if model is not None:
                ema_model.load_state_dict(state["model_state_dict"])
        else:
            ema_model.load_state_dict(ema_state_dict)

    if optimizer is not None and "optimizer_state_dict" in state:
        load_optimizer_state_dict(
            optimizer,
            state["optimizer_state_dict"],
            source=filepath,
        )
    if scheduler is not None:
        if "scheduler_state_dict" not in state:
            logger.warning(
                "Checkpoint missing scheduler state; using fresh scheduler state",
                checkpoint=str(filepath),
            )
        else:
            scheduler.load_state_dict(state["scheduler_state_dict"])

    return state


@contextlib.contextmanager
def _onnx_export_context(model: TetrisNet):
    original_device = next(model.parameters()).device
    if original_device.type == "cuda":
        torch.cuda.empty_cache()
    onnx_logger = logging.getLogger("torch.onnx")
    old_level = onnx_logger.level
    try:
        model.cpu()
        model.eval()
        onnx_logger.setLevel(logging.ERROR)
        yield
    finally:
        onnx_logger.setLevel(old_level)
        model.to(original_device)


def export_onnx(
    model: TetrisNet,
    filepath: str | Path,
    opset_version: int = 18,
) -> bool:
    try:
        with _onnx_export_context(model):
            dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
            dummy_aux = torch.zeros(1, AUX_FEATURES)

            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.filterwarnings("ignore")
                torch.onnx.export(
                    model,
                    (dummy_board, dummy_aux),
                    filepath,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["board", "aux_features"],
                    output_names=["policy_logits", "value"],
                    verbose=False,
                    dynamo=False,
                )
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("ONNX export skipped; missing dependencies", error=str(e))
        return False


def split_model_paths(onnx_path: str | Path) -> tuple[Path, Path, Path]:
    base = Path(onnx_path).with_suffix("")
    return (
        base.with_suffix(".conv.onnx"),
        base.with_suffix(".heads.onnx"),
        base.with_suffix(".fc.bin"),
    )


def export_split_models(
    model: TetrisNet,
    onnx_path: str | Path,
    opset_version: int = 18,
) -> bool:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    try:
        with _onnx_export_context(model):
            conv_backbone = ConvBackbone(model)
            conv_backbone.eval()
            dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.filterwarnings("ignore")
                torch.onnx.export(
                    conv_backbone,
                    (dummy_board,),
                    str(conv_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["board"],
                    output_names=["conv_out"],
                    verbose=False,
                    dynamo=False,
                )

            heads_model = HeadsModel(model)
            heads_model.eval()
            board_hidden = model.board_proj.out_features
            dummy_board_h = torch.zeros(1, board_hidden)
            dummy_aux = torch.zeros(1, PIECE_AUX_FEATURES)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.filterwarnings("ignore")
                torch.onnx.export(
                    heads_model,
                    (dummy_board_h, dummy_aux),
                    str(heads_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["board_h", "piece_aux"],
                    output_names=["policy_logits", "value"],
                    verbose=False,
                    dynamo=False,
                )

            _export_cached_board_path_binary(model, fc_path)

        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("Split model export skipped; missing dependencies", error=str(e))
        return False


def export_metadata(
    filepath: str | Path,
    step: int,
    eval_metrics: dict | None = None,
    config: dict | None = None,
) -> None:
    metadata = {
        "step": step,
        "eval_metrics": eval_metrics or {},
        "config": config or {},
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


class WeightManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _export_rust_artifacts_from_model(
        self,
        model: TetrisNet,
        *,
        onnx_path: Path,
    ) -> None:
        onnx_export_ok = export_onnx(model, onnx_path)
        if not onnx_export_ok:
            raise RuntimeError(
                "ONNX export failed due to missing dependencies while saving checkpoint"
            )

        split_export_ok = export_split_models(model, onnx_path)
        if not split_export_ok:
            raise RuntimeError(
                "Split-model export failed due to missing dependencies while saving checkpoint"
            )

    def _finalize_latest_links(
        self,
        *,
        checkpoint_path: Path,
        metadata_path: Path,
        paths: dict[str, Path],
    ) -> dict[str, Path]:
        paths["metadata"] = metadata_path
        latest_path = self.checkpoint_dir / LATEST_CHECKPOINT_FILENAME
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)
        paths["latest"] = latest_path
        return paths

    def save(
        self,
        model: TetrisNet,
        ema_model: TetrisNet | None,
        optimizer: OptimizerLike | None,
        scheduler: SchedulerLike | None,
        step: int,
        eval_metrics: dict | None = None,
        export_for_rust: bool = True,
        extra_checkpoint_state: dict[str, object] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        ckpt_path = self.checkpoint_dir / f"{CHECKPOINT_FILENAME_PREFIX}_{step}.pt"
        save_checkpoint(
            model,
            ema_model,
            optimizer,
            scheduler,
            step,
            ckpt_path,
            **(extra_checkpoint_state or {}),
        )
        paths["checkpoint"] = ckpt_path

        if export_for_rust:
            export_source_model = ema_model if ema_model is not None else model
            onnx_path = self.checkpoint_dir / LATEST_ONNX_FILENAME
            self._export_rust_artifacts_from_model(
                export_source_model,
                onnx_path=onnx_path,
            )
            paths["onnx"] = onnx_path

        meta_path = self.checkpoint_dir / LATEST_METADATA_FILENAME
        export_metadata(meta_path, step, eval_metrics)
        return self._finalize_latest_links(
            checkpoint_path=ckpt_path,
            metadata_path=meta_path,
            paths=paths,
        )

    def save_snapshot(
        self,
        snapshot: CheckpointSnapshot,
        model_kwargs: dict[str, Any],
        eval_metrics: dict[str, Any] | None = None,
        export_for_rust: bool = True,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        ckpt_path = (
            self.checkpoint_dir / f"{CHECKPOINT_FILENAME_PREFIX}_{snapshot.step}.pt"
        )
        save_checkpoint_snapshot(snapshot, ckpt_path)
        paths["checkpoint"] = ckpt_path

        if export_for_rust:
            snapshot_model = TetrisNet(**model_kwargs)
            export_state_dict = (
                snapshot.ema_state_dict
                if snapshot.ema_state_dict is not None
                else snapshot.model_state_dict
            )
            snapshot_model.load_state_dict(export_state_dict)
            onnx_path = self.checkpoint_dir / LATEST_ONNX_FILENAME
            self._export_rust_artifacts_from_model(snapshot_model, onnx_path=onnx_path)
            paths["onnx"] = onnx_path

        meta_path = self.checkpoint_dir / LATEST_METADATA_FILENAME
        export_metadata(meta_path, snapshot.step, eval_metrics)
        return self._finalize_latest_links(
            checkpoint_path=ckpt_path,
            metadata_path=meta_path,
            paths=paths,
        )

    def load_latest(
        self,
        model: TetrisNet,
        optimizer: OptimizerLike | None = None,
        scheduler: SchedulerLike | None = None,
    ) -> int | None:
        latest_path = self.checkpoint_dir / LATEST_CHECKPOINT_FILENAME
        if not latest_path.exists():
            return None

        state = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return state.get("step", 0)

    def get_checkpoints(self) -> list[Path]:
        checkpoints = list(
            self.checkpoint_dir.glob(f"{CHECKPOINT_FILENAME_PREFIX}_*.pt")
        )
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints


class AsyncCheckpointSaver:
    def __init__(self, weight_manager: WeightManager):
        self.weight_manager = weight_manager
        self._requests: Queue[CheckpointSaveRequest | None] = Queue()
        self._completed: Queue[tuple[int, dict[str, Path]]] = Queue()
        self._error_lock = Lock()
        self._error: BaseException | None = None
        self._worker = Thread(
            target=self._worker_loop,
            name="async-checkpoint-saver",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        *,
        snapshot: CheckpointSnapshot,
        model_kwargs: dict[str, Any],
        eval_metrics: dict[str, Any] | None = None,
        export_for_rust: bool = True,
    ) -> None:
        self.raise_if_failed()
        self._requests.put(
            CheckpointSaveRequest(
                snapshot=snapshot,
                model_kwargs=model_kwargs,
                eval_metrics=eval_metrics,
                export_for_rust=export_for_rust,
            )
        )

    def drain_completed(self) -> list[tuple[int, dict[str, Path]]]:
        completed: list[tuple[int, dict[str, Path]]] = []
        while True:
            try:
                completed.append(self._completed.get_nowait())
            except Empty:
                return completed

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Asynchronous checkpoint save failed") from error

    def shutdown(self) -> None:
        self._requests.put(None)
        self._worker.join()
        self.raise_if_failed()

    def _worker_loop(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            try:
                paths = self.weight_manager.save_snapshot(
                    request.snapshot,
                    request.model_kwargs,
                    eval_metrics=request.eval_metrics,
                    export_for_rust=request.export_for_rust,
                )
                self._completed.put((request.snapshot.step, paths))
            except BaseException as error:
                with self._error_lock:
                    if self._error is None:
                        self._error = error
