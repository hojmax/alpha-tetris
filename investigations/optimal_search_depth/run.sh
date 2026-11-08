#!/usr/bin/env bash
# Reproduce the optimal-search-depth investigation.
#
# Prereqs:
#   1. `make build` (needs MCTSAgent.search_root_visit_snapshots in the Rust extension).
#   2. The final-run model from the README's Google Drive folder in a run directory, e.g.
#      training_runs/drive_final/{config.yaml,checkpoints/incumbent.onnx,
#      checkpoints/incumbent.{conv.onnx,heads.onnx,fc.bin},checkpoints/latest.pt}
#
# Usage: investigations/optimal_search_depth/run.sh [run_dir] [extra script args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN_DIR="${1:-training_runs/drive_final}"
shift || true
WORKERS="${TETRIS_OPT_NUM_WORKERS:-$(( $(getconf _NPROCESSORS_ONLN) - 2 ))}"
PYTHONPATH=. ./.venv/bin/python investigations/optimal_search_depth/search_depth_info_gain.py \
  --run_dir "$RUN_DIR" --num_workers "$WORKERS" "$@"
