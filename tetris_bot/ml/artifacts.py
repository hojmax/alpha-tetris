from __future__ import annotations

import shutil
import time
from pathlib import Path

import structlog

from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import export_onnx, export_split_models, split_model_paths

logger = structlog.get_logger()


def assert_rust_inference_artifacts(onnx_path: Path) -> None:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    required_paths = [onnx_path, conv_path, heads_path, fc_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(
            "Model export incomplete for Rust inference; missing artifacts: "
            + ", ".join(missing_paths)
        )


def export_rust_inference_bundle(
    model: TetrisNet,
    onnx_path: Path,
    *,
    export_name: str,
) -> float:
    """Export the full ONNX graph plus the split Rust-inference files; returns ms."""
    export_start = time.perf_counter()
    if not export_onnx(model, onnx_path):
        raise RuntimeError(
            f"{export_name} ONNX export failed due to missing dependencies"
        )
    if not export_split_models(model, onnx_path):
        raise RuntimeError(
            f"{export_name} split-model export failed due to missing dependencies"
        )
    assert_rust_inference_artifacts(onnx_path)
    return 1000.0 * (time.perf_counter() - export_start)


def required_model_artifact_paths(onnx_path: Path) -> list[Path]:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    return [onnx_path, conv_path, heads_path, fc_path]


def optional_model_artifact_paths(onnx_path: Path) -> list[Path]:
    conv_path, heads_path, _ = split_model_paths(onnx_path)
    return [
        onnx_path.with_suffix(".onnx.data"),
        conv_path.with_suffix(".onnx.data"),
        heads_path.with_suffix(".onnx.data"),
    ]


def _fix_onnx_external_data_references(onnx_path: Path) -> None:
    """Patch external data location references in an ONNX file to match its filename.

    When an ONNX file with external data (e.g. candidate_step_1000.conv.onnx) is
    copied to a new name (e.g. incumbent.conv.onnx), the protobuf still references
    the original data filename. This rewrites those references so tract/onnxruntime
    can find the co-located .data file.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    expected_data_filename = onnx_path.name + ".data"
    changed = False
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location" and entry.value != expected_data_filename:
                    entry.value = expected_data_filename
                    changed = True
    if changed:
        onnx.save_model(model, str(onnx_path))


def _paths_refer_to_same_file(source_path: Path, destination_path: Path) -> bool:
    if source_path == destination_path:
        return True
    if source_path.exists() and destination_path.exists():
        try:
            return source_path.samefile(destination_path)
        except OSError:
            return False
    return False


def copy_model_artifact_bundle(
    source_onnx_path: Path, destination_onnx_path: Path
) -> None:
    assert_rust_inference_artifacts(source_onnx_path)
    destination_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    source_required = required_model_artifact_paths(source_onnx_path)
    destination_required = required_model_artifact_paths(destination_onnx_path)
    for source_path, destination_path in zip(source_required, destination_required):
        if _paths_refer_to_same_file(source_path, destination_path):
            continue
        shutil.copy2(source_path, destination_path)

    source_optional = optional_model_artifact_paths(source_onnx_path)
    destination_optional = optional_model_artifact_paths(destination_onnx_path)
    for source_path, destination_path in zip(source_optional, destination_optional):
        if _paths_refer_to_same_file(source_path, destination_path):
            continue
        if source_path.exists():
            shutil.copy2(source_path, destination_path)
        else:
            destination_path.unlink(missing_ok=True)

    # Fix external data references in copied ONNX files — the protobuf embeds
    # the original source filename (e.g. "candidate_step_68937.conv.onnx.data")
    # which becomes stale after renaming to incumbent/latest.
    onnx_files = [destination_onnx_path]
    dest_conv, dest_heads, _ = split_model_paths(destination_onnx_path)
    onnx_files.extend([dest_conv, dest_heads])
    for onnx_file in onnx_files:
        if onnx_file.exists():
            _fix_onnx_external_data_references(onnx_file)
