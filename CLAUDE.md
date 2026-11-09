# Project Context for Agents

## Keep This File Current

Treat this document as a living guide.

- Fix outdated or incorrect info when you notice it.
- Add important constraints, workflow gotchas, or debugging tips as they are discovered.
- Prefer small, continuous updates over occasional big rewrites.

## Memory Workflow

- At the start of every session, read `memories/MEMORY.md` before doing substantive work. If it is missing, create it first.
- Treat `memories/MEMORY.md` as the short-term operational memory log for repo-specific learnings and user feedback.
- Write to `memories/MEMORY.md` liberally whenever you learn something useful: corrected command usage, debugging realizations, unintuitive behavior, hard-to-find facts, user preferences, workflow gotchas, or other details likely to save time later.
- Prefer concise entries with the final takeaway, not a long play-by-play.
- When `memories/MEMORY.md` reaches 200 lines, compact it. Group related stable notes into themed Markdown files under `memories/`.
- When compacting, append to an existing themed memory file if one already fits the topic; otherwise create a new file with a descriptive name.
- After compacting, keep `memories/MEMORY.md` as the active short-term log with a reduced set of current notes and pointers to the long-term themed files.

## Non-Negotiable Policies

### Feature Worktrees

- Small edits (tiny bug fixes, docs tweaks, narrowly scoped changes) can be done directly on `main`.
- Feature work or risky changes must use a dedicated worktree located outside of the primary checkout (sibling directory).
- Do not implement large feature work directly in the primary checkout of this repo.
- Create isolated feature work with:
  `git worktree add -b <branch-name> ../tetris-mcts-worktrees/<worktree-name> <base-ref>`
- After merge, immediately remove the feature worktree and delete the branch.

### Parallel Game Generation/Evaluation

- Always parallelize game generation/evaluation in normal runs.
- Do not use single-worker loops when a parallel path exists.
- Rust evaluation entry points (`evaluate_model`, `evaluate_model_without_nn`) must use `num_workers > 1`.
- Rust parallel evaluation uses dynamic seed claiming to reduce stragglers in fixed-seed sweeps; benchmark sweeps should prefer larger `num_games` than `num_workers` to reduce seed-length variance noise.
- GameGenerator workers commit completed games in small batches (`GAME_COMMIT_BATCH_SIZE` in `runtime/game_generator/runtime.rs`) to reduce lock contention at high worker counts.
- Python scripts that generate games should expose and use worker-count args, defaulting to multi-process execution.

## Project Overview

`tetris-mcts` is an AlphaZero-style Tetris system:

- Rust engine for game logic and MCTS
- PyO3 bindings for Python integration
- PyTorch model training in Python
- ONNX export for Rust inference
- Pygame + Dash tools for visualization

## Quick Commands

```bash
make install    # uv sync + Rust bootstrap + PATH persist + debug extension build
make build      # release Rust extension (slow, optimized; native CPU + thin LTO by default)
make build-ort  # release build with ONNX Runtime support (`nn-ort` feature)
make build-dev  # debug Rust extension (fast iteration)
make play       # interactive game
make viz        # MCTS tree visualizer
make viz-policy-grid # policy-grid visualizer for structured placement-head inspection
make test       # Rust + Python tests
make check      # ruff + pyright + rust fixes/formatting
make train      # training entry point (requires tmux; auto-loads machine optimize cache)
make warm-start # create a new warm-started run dir from another run's training_data.npz
make compare-offline-spatial-policy # compare current gated fusion vs 4x20x10 spatial policy decoder offline
make compare-warm-start-trunk-sizes # warm-start several trunk sizes from one replay buffer, then benchmark attack vs throughput
make search-depth-info-gain # Leela-style policy information gain per MCTS visit (see investigations/optimal_search_depth/)
make profile    # performance profile runner
make optimize   # auto-tune build/backend/workers for this machine (with cache)
```

`make install` performs a best-effort Linux system dependency bootstrap for ORT builds (`pkg-config` + OpenSSL headers) and maturin builds (`patchelf`); disable with `AUTO_INSTALL_SYSTEM_DEPS=0`.
`make viz` accepts pass-through args via `VIZ_ARGS`, e.g. `make viz VIZ_ARGS="--state_preset tetris_bot/scripts/inspection/viz_state_presets/training_data1_game721_move32.json"`; generate presets from NPZ with `python tetris_bot/scripts/inspection/extract_viz_state_preset.py`. By default it loads `checkpoints/incumbent.onnx` from the selected run.
`make viz-policy-grid` launches the browser visualizer for the normalized `20x10x4` placement scheme; pass flags such as `--port` via `POLICY_GRID_ARGS`.
Candidate-gate evaluation now also saves the worst full-game tree playback to `training_runs/vN/analysis/eval_trees/`; reopen it with `make viz VIZ_ARGS="--saved_playback training_runs/vN/analysis/eval_trees/latest_worst_candidate_eval_tree.json"`.
If a saved full-game playback is too large to load comfortably, extract one exact step into a smaller playback with `./.venv/bin/python tetris_bot/scripts/inspection/extract_saved_playback_step.py --saved_playback <full-playback.json> --step_index <n>` and open that output with `make viz VIZ_ARGS="--saved_playback <step-playback.json>"`.
For inspection/export defaults, prefer the effective incumbent search state from the saved checkpoint (`latest.pt`) over bare `config.yaml`; older runs can have promoted `nn_value_weight` / penalty settings that differ from the static config file.
The visualizer UI includes a `Play Full Game` control that rolls out the rest of the game from the current state using tree reuse, stitches the per-move search trees together with `reuse` edges, and highlights the chosen action/chance path.

Useful extras:

```bash
make sweep-lr-model
make eval-nn-value-weight
make sweep-mcts-config
make compare-offline-network-scaling
make compare-offline-spatial-policy
make compare-warm-start-trunk-sizes
```

## High-Level Repository Layout

```text
tetris_core/src/
├── game/       # deterministic Tetris rules and environment
├── search/     # MCTS implementation
├── inference/  # native conv trunk + board MLP, ONNX heads backends
├── runtime/    # self-play and evaluator runtime
├── replay/     # replay and NPZ persistence
└── lib.rs      # PyO3 module exports

tetris_bot/
├── ml/         # model, trainer, loss, replay buffer, checkpoints
├── scripts/    # training, ablations, inspection tools
├── visualization.py
└── constants.py
```

## Architecture and Ownership Boundaries

Data flow:

```text
Python training -> ONNX export -> Rust inference during self-play/eval
```

Ownership split:

- Rust (`tetris_core/`) owns game logic, move generation, scoring, MCTS, runtime state, inference.
- Python (`tetris_bot/`) owns training, analysis scripts, and visualization/UI.
- UI styling/color choices are Python-owned; Rust should expose state only.

## Core Concepts

### Action Space

- Fixed 672 actions: 671 canonical placement cells + hold.
- Canonical placement cells come from the normalized `4 x 20 x 10` policy grid after collapsing redundant piece rotations (`O -> 0`, `I/S/Z -> 0/1`) and removing permanently inactive cells.
- `ActionSpace` maps piece-specific `(x, y, rotation)` placements onto those canonical action indices.
- Placement caching is heavily used; hold availability is part of cache keys.

### MCTS with Chance Nodes

- `DecisionNode`: choose action.
- `ChanceNode`: random next piece from 7-bag process.
- Chance outcomes use visible queue-tail identity for consistent subtree reuse.
- Hold-swap deterministic transitions use `NO_CHANCE_OUTCOME` sentinel.

### Network Shape and Features

- Input: 280 features (`200 board + 80 aux`).
- Aux split:
  - 61 piece/game features for uncached heads path.
  - 19 board-derived stats folded into cached board embedding.
- Outputs: policy over 672 actions + scalar value.
- Architecture: conv trunk + board-stats encoder + deeper cached board MLP, then concat the board embedding and auxiliary embedding before the policy/value trunk.

### Scoring Reminder

- Attack baseline: single 0, double 1, triple 2, tetris 4.
- Rules follow Jstris's client code (`GameCore.checkTSpin` / `checkLineClears` / `rotateCurrentBlock`; decoded from a mirror of its `game.js`). Keep them in sync with Jstris, not TETR.IO or the guideline.
- T-spin attack: mini single 0, single 2, double 4, triple 6. A mini double scores as a full TSD (4) and a mini triple as a TST. Back-to-back +1 applies to tetrises and T-spins; a mini single keeps or starts the chain but never gets the +1.
- Perfect clear sends 10 in place of the clear's own attack and B2B bonus, plus the combo bonus. B2B afterwards follows the underlying clear.
- T-spin detection (`game/tspin.rs`, shared by the env and move generation): 3-corner rule, full = both front corners + at least one back corner, mini = one front + both back. No immobility check and no kick-based upgrade. Walls and floor count as filled; cells above the board count as empty.
- Spin flag (`spin_flag`, Jstris `spinPossible`): set by any rotation, cleared only by downward movement (soft drop, gravity, or a hard drop that moves the piece). Sideways moves keep it.
- 180° rotation exists (`rotate_180`, `step(8)`, `Action::Rotate180`) with Jstris's kicks: one fallback shift (up from spawn, right from R, down from 2, left from L).
- Move generation BFS runs over (position, spin flag) for T pieces and reports the spin path for a final position whenever its corners make it a T-spin, since a spin never scores worse under these rules.

## Important Defaults (TrainingConfig)

- Network: `trunk_channels=32`, `num_conv_residual_blocks=5`, `reduction_channels=32`, `board_stats_hidden=32`, `board_proj_hidden=512`, `fc_hidden=256`, `aux_hidden=128`, `fusion_hidden=256`, `num_fusion_blocks=1`.
- MCTS: `num_simulations=500` (policy info-gain plateau 240-542 sims; 500 keeps ~95% of 2000-sim attack at 3.1x throughput; see `investigations/optimal_search_depth/`), `c_puct=1.5`, `temperature=1.0`, `reuse_tree=true`, `max_placements=50`.
- Search budget: `config.yaml` sets `self_play.count_reused_visits=true` with `min_new_simulations=50`, so `num_simulations` is a per-search visit budget: visits carried in by tree reuse (~37% of a root's final visits, measured at 2000 sims) count toward it, and new sims = `max(min_new_simulations, num_simulations - carried)`. With it off (the pydantic field default, so older saved configs keep their behaviour), every move adds `num_simulations` new simulations on top of the carried visits. At a fixed network, budget 2000 (min 200) ran 1.54x faster than plain 2000 sims for -0.7 greedy attack (50.0 vs 50.8, 144 seeds), about the same trade as lowering `num_simulations` (plain 1000 sims: 49.3 at 0.55x cost); 144 games cannot tell the two apart. Not measured at the 500-sim default.
- Training: `batch_size=2048`, `learning_rate=5e-3` with linear decay (`lr_min_factor=0.02`, i.e. to `1e-4`), `weight_decay=5e-5`, `grad_clip_norm=10.0`.
- Workers/buffer: `num_workers=32`, replay ring buffer size `5_000_000` (sized so the GPU replay mirror fits a 24 GB card). These come only from the repo-root `config.yaml`; `ReplayConfig`/`SelfPlayConfig` in `tetris_bot/ml/config.py` have no code defaults for them.
- Replay ratio: the trainer is free-running by default (`replay.max_train_samples_per_new_example=0`). The user's 32-worker CUDA run drew ~320 training samples per generated example (11.6 steps/s x 2048 vs ~75 examples/s), so trainer speedups do not speed learning; self-play throughput does. Set the cap (> 0) to make the trainer sleep whenever this session's samples drawn exceed `cap x` newly generated/ingested examples (preloaded replay does not count). Logged as `replay/samples_per_generated_example`, `replay/session_samples_per_new_example`, `throughput/replay_ratio_throttle_fraction`.
- Training on Apple Silicon (MPS) next to local self-play costs self-play ~50% throughput through unified-memory contention (worker IPC drops; trainer CPU use is only ~0.2 cores); throttling the trainer (the cap above) recovers it roughly in proportion to the GPU duty cycle. Discrete-GPU (CUDA) hosts do not have this effect; there the trainer thread costs about one CPU core.
- Bootstrap: starts no-network, typically `bootstrap_num_simulations=4000` until first promotion.
- Candidate gate: deterministic fixed-seed eval window (default 20 games, no Dirichlet noise, `visit_sampling_epsilon=0`, fixed MCTS seed), promote when the candidate's evaluation average is greater than or equal to the stored incumbent evaluation average. Fixed-seed eval trajectories are not added to the replay buffer. Candidate export timing now uses `run.model_sync_interval_seconds` as the base interval, adds `run.model_sync_failure_backoff_seconds` after each failed promotion (optionally capped by `run.model_sync_max_interval_seconds`), measures the cooldown from eval start, and suppresses exports while a candidate is pending/evaluating so long evals do not churn throwaway ONNX bundles. Set `self_play.use_candidate_gating=false` to disable the evaluator worker entirely; in that mode all workers stay on generation and the trainer directly syncs a freshly exported incumbent to everyone every `run.model_sync_interval_seconds`.
- Every `run.attack_histogram_interval_seconds` (default 600; 0 disables) the trainer logs the total-attack distribution of the last `run.attack_histogram_window_games` (default 1000) self-play games under `game_dist/*`: a W&B histogram (`game_dist/total_attack`, shown as a heatmap over `trainer_step`), a paper-style plot image (`game_dist/total_attack_plot`), and mean/p10/p50/p90 scalars. Self-play games use noise and visit sampling, so they score below greedy eval. The plot code (`tetris_bot/attack_histogram.py`) is shared with `paper/scripts/make_attack_distribution_plot.py`; the ~70 ms render runs on the trainer thread.
- When `self_play.use_candidate_gating=false`, WandB direct-sync logs now also attach `model_sync/random_recent_game`: a GIF sampled from completed replays whose completion timestamps fall within the last `run.model_sync_interval_seconds`.
- NN value scaling: defaults start at cap (`nn_value_weight=1.0`, `nn_value_weight_cap=1.0`).
- Q normalization: search uses global min-max normalization for PUCT Q terms.

Memory note:

- CPU RAM usually becomes the bottleneck before GPU VRAM.
- Model weights are shared process-wide: `TetrisNN::load` goes through `inference/weight_registry.rs`, so all workers loading the same bundle (same canonical paths, sizes, mtimes) share one `Arc<ModelWeights>` and old models are freed when the last worker drops them. Private per-worker copies made multi-worker self-play DRAM-bandwidth-bound (M5 Pro, 18 workers: 27.1 -> 52.6 placements/s from sharing alone). Don't add load paths that bypass it.
- Biggest levers: `buffer_size`, `num_workers`, `bootstrap_num_simulations`, replay mirror/staging settings.
- Rust per-worker caches scale with worker count: the placement cache (`PLACEMENT_CACHE_MAX_ENTRIES` = 30K entries, ~1 KB each) and the NN board-embedding cache (`BOARD_CACHE_MAX_ENTRIES` = 64K entries, ~1 KB each, ~64 MB per worker when full; a game misses only ~2.4-4.1K boards and later games never hit earlier games' entries, so the old 1M cap only cost ~1.1 GB/worker). Board overhang/hole analysis is computed directly (bitboard), with no global cache.

Disk note:

- At the start of every fresh `wandb.init` call, training prunes `wandb/run-*` directories older than 24h and runs `wandb artifact cache cleanup 0` (without `--remove-temp`, so live artifact uploads are not disturbed). See `_prune_local_wandb_state` in `tetris_bot/run_setup.py`.

## Common Workflows

### Change Game Logic

1. Edit Rust code in `tetris_core/src/game/env/` (or related modules).
2. Update Python-facing bindings in `tetris_core/src/game/env/pymethods.rs` if needed.
3. Rebuild (`make build-dev` for iteration).
4. Validate with `make test` and optionally `make play`.

### Change Network/Features

1. Edit `tetris_bot/ml/network.py`.
2. Update Rust inference encoding in `tetris_core/src/inference/mod.rs` if feature schema changes.
3. The whole network runs natively in Rust (no tract/ORT at inference time). `tetris_core/src/inference/conv.rs` reads the conv trunk from `X.conv.onnx` and strictly pattern-matches the exported `ConvBackbone` graph; `tetris_core/src/inference/heads.rs` reads `X.heads.onnx` initializers by PyTorch parameter name. Any change to `ConvBackbone`/`ResidualConvBlock`/GroupNorm/SiLU structure or to the heads (aux MLP, board/aux concatenation + `fusion_fc`, residual fusion blocks, policy/value heads) must be mirrored there; loading fails loudly on any unexpected graph structure.
4. Retrain/export ONNX artifacts.

### Train / Resume

```bash
python tetris_bot/scripts/train.py --config config.yaml
python tetris_bot/scripts/train.py --config config.yaml --resume_dir training_runs/vN
python tetris_bot/scripts/train.py --config config.yaml --resume_wandb entity/project/run_id
python tetris_bot/scripts/inspection/download_wandb_training_data.py --reference entity/project/run_id --run_dir training_runs/vN --overwrite true
python tetris_bot/scripts/warm_start.py --source_run_dir training_runs/v3
```

Training runs live under `training_runs/vN/` with checkpoints and ONNX exports.
Each run directory now also includes a `runtime_overrides.yaml` file for live tuning during training. The trainer polls it every 15 seconds and only applies the whitelisted Python-side fields `optimizer.lr_multiplier`, `optimizer.grad_clip_norm`, `optimizer.weight_decay`, `optimizer.mirror_augmentation_probability`, `run.log_interval_seconds`, `run.checkpoint_interval_seconds`, `self_play.add_noise`, and `self_play.visit_sampling_epsilon`; `null` reverts a field to the run-start default. The two self-play fields drive a "final phase" lever: setting `add_noise: false` and `visit_sampling_epsilon: 0` (typically with a small `lr_multiplier`) makes self-play and training fully greedy without restarting the run. The trainer mutates `self.config.self_play`, calls `GameGenerator.update_search_overrides(...)` so workers pick up the change at the start of the next game, and re-publishes `SelfPlaySnapshot` to R2; remote generators poll the snapshot every `r2_sync.model_pointer_poll_interval_seconds` and converge on the same setting. Candidate-eval games still force `add_noise=false` and `visit_sampling_epsilon=0` deterministically. Resume runs copy the previous run's `runtime_overrides.yaml`, and the current applied override state is also stored in checkpoints so LR multiplier, interval, and self-play override settings survive resume.
`--resume_wandb` accepts either a run reference (`entity/project/run_id`, defaults to the run's `tetris-model-<run_id>:latest` artifact) or a direct artifact reference (`entity/project/tetris-model-<run_id>:alias`).
`warm_start.py` creates the next sibling `training_runs/vN` by default, builds the new run from the current defaults in the repo-root `config.yaml` (not the source run's saved `config.yaml`) unless a caller passes a custom `output_config`, trains offline on the source `training_data.npz`, copies that NPZ into the new run, saves `latest`/`incumbent`/`parallel` model bundles plus `checkpoint_0.pt`, forces the saved incumbent startup state to `nn_value_weight=1.0` with zero search penalties, writes `analysis/warm_start_summary.json` with the offline-loss history and fixed-seed eval result, and logs the offline train/eval run to the `tetris-mcts-offline` W&B project by default. Offline warm-start training now runs in `epochs_per_round` chunks (default `4.0`), evaluates after each chunk, stops after `early_stopping_patience=5` consecutive rounds without improving the fixed selection metric `val_policy_loss + val_value_loss / 4`, keeps the best checkpointed weights, saves `checkpoints/warm_start_offline_latest.pt` for true offline continuation, and defaults final eval workers from the machine optimize cache (`TETRIS_OPT_NUM_WORKERS`) when available. Use `--output_run_dir` to override the destination and `--eval_num_simulations` / `--eval_max_placements` for quicker smoke checks without changing the saved run config. Set `--resume_from_source_offline_state true` to continue from another warm-start run's saved offline weights and optimizer state; older warm-start runs created before this checkpoint existed cannot use that flag.
`compare_warm_start_trunk_sizes.py` lives at `tetris_bot/scripts/ablations/compare_warm_start_trunk_sizes.py`. It builds three warm-start variants by default (`0.5x`, `1.0x`, `2.0x` of the source run's trunk), scales reduction channels to match, and also steps residual-block depth with model size (`source_blocks - 1`, `source_blocks`, `source_blocks + 1` when the source size is included). It launches warm-start offline training for each into `benchmarks/warm_start_trunk_sizes/<run>_<timestamp>/runs/`, then benchmarks the resulting incumbent ONNX bundles with fixed seeds and writes `comparison_summary.json`, `comparison_summary.md`, and `benchmark_rows.jsonl`.
During shutdown, the first `Ctrl+C` requests graceful trainer/generator stop; additional `Ctrl+C` presses are deferred until the trainer finishes generator shutdown, the final replay snapshot/checkpoint save, and the WandB artifact upload.
Checkpoint resume now persists incumbent search penalties (`incumbent_death_penalty`, `incumbent_overhang_penalty_weight`) alongside `incumbent_nn_value_weight`; older checkpoints without those fields infer zero penalties when `nn_value_weight` is already at cap.

### Profile

```bash
make profile SIMS=200
make profile SIMS=1000 WORKERS_PROFILE=6
make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
make profile-samply SIMS=50
make optimize
```

Set `MODEL_PROFILE=<existing-onnx-path>` when the Makefile default model path is missing.
Inference no longer goes through tract/ORT: the conv trunk, board MLP and heads all run natively, so `TETRIS_NN_BACKEND` is not read and the `nn-ort` feature / `make build-ort` no longer change inference (cleanup pending).
`make optimize` is fast by default; for an exhaustive sweep: `make optimize OPT_BACKEND_STRATEGY=exhaustive OPT_WORKER_SEARCH=grid OPT_GAMES=40 OPT_SIMS=500`.
Results written to `benchmarks/profiles/optimize_cache/<machine_fingerprint>.env`; `make train` computes the current machine fingerprint and loads the matching env (runs optimize first if missing).
Do NOT use `maturin develop` — use `maturin build --out tetris_core/dist` then `$(PYTHON) -m pip install` (respects `.venv` `sys.prefix` correctly).

## Testing

```bash
make test
cd tetris_core && PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test
```

In feature worktrees, use the primary checkout virtualenv path for `PYO3_PYTHON`.
Native conv trunk check against a real bundle: `TETRIS_TEST_CONV_ONNX=<run>/checkpoints/incumbent.conv.onnx cargo test --release incumbent -- --nocapture` (skipped when unset).
SIMD kernels have NEON (aarch64), runtime-detected AVX2+FMA (x86_64) and portable paths; on Apple Silicon compile-check the x86 path with `PYO3_CROSS_PYTHON_VERSION=3.12 cargo check --release --tests --target x86_64-apple-darwin` (needs `rustup target add x86_64-apple-darwin`; it cannot run without Rosetta).
If you see `ModuleNotFoundError: No module named 'tetris_core.tetris_core'`, the native extension is missing for the active interpreter; rebuild with `make build-dev` (or `make build`) in that same `.venv`.

## Data/Artifacts Notes

- `training_data.npz` is for resume/inspection and includes policy/value targets plus per-state diagnostics.
- Large replay snapshots can exceed 4 GiB in `policy_targets.npy`; NPZ writing must keep ZIP64 enabled and snapshot writes should be atomic (`.tmp` then replace).
- Periodic replay snapshot saves are now asynchronous: worker `0` clones the current FIFO replay window and hands it to a background writer thread, so generation no longer blocks on the NPZ write itself. The tradeoff is temporary extra RAM roughly proportional to the retained replay window while a snapshot is in flight. Final shutdown save now happens after all workers join, so the saved `training_data.npz` includes every worker's last committed games.
- Periodic trainer checkpoints are now also asynchronous: the training loop snapshots model/optimizer/scheduler state to CPU and a background Python worker writes the `.pt` plus `latest.onnx`/split-model artifacts, so the old 3-hour checkpoint/export pause no longer blocks train steps. This still incurs a synchronous CPU snapshot copy at checkpoint time.
- Model-sync / candidate exports and all W&B replay GIFs run in a dedicated spawned worker process (`tetris_bot/ml/background_process.py`, `TrainerBackgroundProcess`). The training loop only copies the EMA weights to CPU at submit time and applies the finished bundle (`sync_model_directly` / `queue_candidate_model`) on a later step; at most one export is in flight. Inline, each 120 s sync cost 1.8-5.4 s of train steps on a 116-worker CUDA host (visible as periodic `throughput/steps_per_second` dips). Do not move these back onto a thread: they hold the GIL (continuous background export halved local train steps/s) and `export_onnx` uses `redirect_stdout`, which swaps `sys.stdout` process-wide. `timing/onnx_export_ms` is now worker-side export time; `timing/model_export_latency_ms` is submit-to-applied. Direct-sync R2 uploads run on a one-thread executor, and the next sync waits for the previous upload because syncing lets Rust delete that bundle's files.
- Final WandB model artifact upload includes `training_data.npz` whenever the file exists; no extra Python-side NPZ integrity gate is applied at upload time.
- Resumed runs create a new `vN` directory and initialize from the source run checkpoint.
- Promotion state (incumbent model + `nn_value_weight`) is treated as an atomic runtime state.

## Coding Rules (Condensed)

- Fix problems you touch; do not leave known errors behind.
- Prefer one clean implementation path; avoid backward-compatibility fallbacks for internal formats.
- Remove dead code when replacing functionality.
- Keep imports at top level (except explicit optional/cycle cases).
- Keep functions focused and short; extract reused logic.
- Keep the Rust PyO3 API minimal; do not add dict/string/repr convenience wrappers unless they have a concrete in-repo caller.
- Use modern typing syntax (`A | B`, `list[T]`) and `from __future__ import annotations` when useful.
- Use `pathlib.Path` and `PROJECT_ROOT` (`tetris_bot.constants`) for project-relative paths.
- Use `simple_parsing` dataclass args in scripts; put defaults directly in dataclass fields.
- Use `structlog` for logging; do not use `print()` as logging.
- Prefer early returns and readable positive conditions.
- Comment sparingly; explain why, not what.
- Fail fast on impossible internal states; only be defensive for truly external/user-provided data.

## When in Doubt

- Keep Rust focused on engine/search/runtime performance.
- Keep Python focused on training/analysis/visualization.
- Parallelize game workloads.
- Read `memories/MEMORY.md` first and record durable learnings there.
- Update this file whenever reality changes.
