.PHONY: run install ensure-rust ensure-system-deps build build-ort build-dev clean rebuild test check play viz viz-policy-grid train run-generator replay profile profile-samply optimize sweep-lr-model eval-nn-value-weight compare-offline-network-scaling compare-offline-spatial-policy compare-warm-start-trunk-sizes search-depth-info-gain sweep-mcts-config warm-start download-wandb-training-data

# Source cargo environment if available
SHELL := /bin/bash
CARGO_ENV := source $$HOME/.cargo/env 2>/dev/null || true
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PYTHON_ABS := $(abspath $(PYTHON))
PYO3_PYTHON := $(PYTHON_ABS)
UV_BIN := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
VENV_ABS := $(abspath $(VENV_DIR))
MATURIN_DIST := tetris_core/dist
MATURIN_BUILD := rm -rf $(MATURIN_DIST) && $(PYTHON) -m maturin build --manifest-path tetris_core/Cargo.toml --out $(MATURIN_DIST)
MATURIN_INSTALL := VIRTUAL_ENV=$(VENV_ABS) $(UV_BIN) pip install --no-deps --reinstall $(MATURIN_DIST)/tetris_core-*.whl
MATURIN_ENV := $(CARGO_ENV) && export PYO3_PYTHON=$(PYO3_PYTHON)
# `export ... &&` so the settings reach `maturin build`; a bare `VAR=x` prefix would only
# apply to the `rm -rf` at the start of $(MATURIN_BUILD).
MATURIN_RELEASE_ENV = export CARGO_PROFILE_RELEASE_LTO=$(RELEASE_LTO) CARGO_PROFILE_RELEASE_CODEGEN_UNITS=$(RELEASE_CODEGEN_UNITS) RUSTFLAGS="$(RELEASE_RUSTFLAGS)" &&
INSTALL_MARKER := $(VENV_DIR)/.install_marker

# Find all Rust source files (including subdirectories)
RUST_SRC := $(shell find tetris_core/src -name '*.rs')
RELEASE_MARKER := .build_marker_release
DEV_MARKER := .build_marker_dev
RELEASE_RUSTFLAGS ?= -C target-cpu=native
RELEASE_LTO ?= thin
RELEASE_CODEGEN_UNITS ?= 1
AUTO_INSTALL_SYSTEM_DEPS ?= 1

# Bootstrap project dependencies into local virtualenv with uv.
$(INSTALL_MARKER): pyproject.toml uv.lock
	@set -euo pipefail; \
	if command -v uv >/dev/null 2>&1; then \
		UV_BIN="$$(command -v uv)"; \
	elif [ -x "$$HOME/.local/bin/uv" ]; then \
		UV_BIN="$$HOME/.local/bin/uv"; \
	else \
		if ! command -v curl >/dev/null 2>&1; then \
			echo "Error: uv is missing and curl is required to install it." >&2; \
			exit 1; \
		fi; \
		echo "uv not found; installing to $$HOME/.local/bin..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		if [ -x "$$HOME/.local/bin/uv" ]; then \
			UV_BIN="$$HOME/.local/bin/uv"; \
		elif command -v uv >/dev/null 2>&1; then \
			UV_BIN="$$(command -v uv)"; \
		else \
			echo "Error: uv installation failed." >&2; \
			exit 1; \
		fi; \
	fi; \
	"$$UV_BIN" sync --frozen
	@touch $(INSTALL_MARKER)

ensure-rust:
	@set -euo pipefail; \
	need_install=0; \
	if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then \
		:; \
	else \
		if [ -f "$$HOME/.cargo/env" ]; then \
			source "$$HOME/.cargo/env"; \
		fi; \
		if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then \
			need_install=1; \
		fi; \
	fi; \
	if [ "$$need_install" -eq 1 ]; then \
		if ! command -v curl >/dev/null 2>&1; then \
			echo "Error: rustc/cargo are missing and curl is required to install rustup." >&2; \
			exit 1; \
		fi; \
		echo "Rust toolchain not found; installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable; \
		source "$$HOME/.cargo/env"; \
		rustc --version >/dev/null; \
		cargo --version >/dev/null; \
	fi; \
	PATH_LINE='export PATH="$$HOME/.cargo/bin:$$PATH"'; \
		for rc in "$$HOME/.bashrc" "$$HOME/.zshrc"; do \
			touch "$$rc"; \
			if ! grep -Fq "$$PATH_LINE" "$$rc"; then \
				echo "$$PATH_LINE" >> "$$rc"; \
			fi; \
		done

# Ensure Linux build prerequisites for ORT builds.
# Set AUTO_INSTALL_SYSTEM_DEPS=0 to skip automatic apt/dnf/yum/apk installation.
ensure-system-deps:
	@set -euo pipefail; \
	if [ "$$(uname -s)" != "Linux" ]; then \
		exit 0; \
	fi; \
	has_pkg_config=1; \
	has_openssl_pkg=1; \
	has_patchelf=1; \
	if ! command -v pkg-config >/dev/null 2>&1; then \
		has_pkg_config=0; \
		has_openssl_pkg=0; \
	else \
		if ! pkg-config --exists openssl >/dev/null 2>&1; then \
			has_openssl_pkg=0; \
		fi; \
	fi; \
	if ! command -v patchelf >/dev/null 2>&1; then \
		has_patchelf=0; \
	fi; \
	if [ "$$has_pkg_config" -eq 1 ] && [ "$$has_openssl_pkg" -eq 1 ] && [ "$$has_patchelf" -eq 1 ]; then \
		exit 0; \
	fi; \
	echo "[install] Missing Linux build prerequisites (pkg-config, OpenSSL dev headers, patchelf)."; \
	if [ "$(AUTO_INSTALL_SYSTEM_DEPS)" != "1" ]; then \
		echo "[install] Skipping auto-install (AUTO_INSTALL_SYSTEM_DEPS=$(AUTO_INSTALL_SYSTEM_DEPS))."; \
		echo "[install] Install manually on Ubuntu/Debian: apt-get update && apt-get install -y pkg-config libssl-dev patchelf"; \
		exit 0; \
	fi; \
	if command -v apt-get >/dev/null 2>&1; then \
		install_cmd='apt-get update && apt-get install -y pkg-config libssl-dev patchelf'; \
	elif command -v dnf >/dev/null 2>&1; then \
		install_cmd='dnf install -y pkgconf-pkg-config openssl-devel patchelf'; \
	elif command -v yum >/dev/null 2>&1; then \
		install_cmd='yum install -y pkgconfig openssl-devel patchelf'; \
	elif command -v apk >/dev/null 2>&1; then \
		install_cmd='apk add --no-cache pkgconf openssl-dev patchelf'; \
	else \
		echo "[install] Unsupported package manager; install pkg-config, OpenSSL dev headers, and patchelf manually."; \
		exit 0; \
	fi; \
	echo "[install] Attempting to install system dependencies..."; \
	if [ "$$(id -u)" -eq 0 ]; then \
		if ! bash -lc "$$install_cmd"; then \
			echo "[install] Auto-install failed; continuing. Install deps manually if ORT build is needed."; \
			exit 0; \
		fi; \
	elif command -v sudo >/dev/null 2>&1; then \
		if ! sudo bash -lc "$$install_cmd"; then \
			echo "[install] Auto-install failed; continuing. Install deps manually if ORT build is needed."; \
			exit 0; \
		fi; \
	else \
		echo "[install] No root/sudo available for auto-install; continuing."; \
		echo "[install] Install manually on Ubuntu/Debian: apt-get update && apt-get install -y pkg-config libssl-dev patchelf"; \
		exit 0; \
	fi; \
	if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists openssl >/dev/null 2>&1; then \
		echo "[install] Dependencies still not detected after auto-install; ORT builds may fail."; \
	fi; \
	if ! command -v patchelf >/dev/null 2>&1; then \
		echo "[install] patchelf still not detected after auto-install; maturin Linux builds may warn about rpath."; \
	fi

install: ensure-rust ensure-system-deps $(INSTALL_MARKER) $(DEV_MARKER)

# Build marker file to track if build is up to date (release mode)
$(RELEASE_MARKER): ensure-rust $(INSTALL_MARKER) $(RUST_SRC) tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(MATURIN_ENV) && $(MATURIN_RELEASE_ENV) $(MATURIN_BUILD) --release && $(MATURIN_INSTALL)
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Build marker file to track if debug build is up to date
$(DEV_MARKER): ensure-rust $(INSTALL_MARKER) $(RUST_SRC) tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(MATURIN_ENV) && $(MATURIN_BUILD) && $(MATURIN_INSTALL)
	@rm -f $(RELEASE_MARKER)
	@touch $(DEV_MARKER)

# Explicit build target
build: $(RELEASE_MARKER)

# Release build with ONNX Runtime backend support (includes nn-ort feature)
build-ort: ensure-rust ensure-system-deps $(INSTALL_MARKER) $(RUST_SRC) tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(MATURIN_ENV) && $(MATURIN_RELEASE_ENV) $(MATURIN_BUILD) --release --features extension-module,nn-ort && $(MATURIN_INSTALL)
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Fast debug build (much faster, for development only)
build-dev: $(DEV_MARKER)

# Run the game (builds first if needed)
play: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/tetris_game.py

# Run the MCTS visualizer (builds first if needed)
# Usage: make viz RUN_DIR=training_runs/v15
# Usage (dummy/no-network): make viz RUN_DIR=training_runs/v15 DUMMY_NETWORK=1
# Usage (state preset): make viz VIZ_ARGS="--state_preset tetris_bot/scripts/inspection/viz_state_presets/training_data1_game721_move32.json"
# Usage (saved eval playback): make viz VIZ_ARGS="--saved_playback training_runs/v15/analysis/eval_trees/latest_worst_candidate_eval_tree.json"
RUN_DIR ?= training_runs/v3
DUMMY_NETWORK ?= 0
VIZ_ARGS ?=
viz: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/mcts_visualizer.py $(if $(RUN_DIR),--run_dir $(RUN_DIR),) $(if $(filter 1 true TRUE yes YES,$(DUMMY_NETWORK)),--use_dummy_network true,) $(VIZ_ARGS)

# Run the policy-grid visualizer (pure Python; does not require Rust build)
# Usage: make viz-policy-grid
# Usage: make viz-policy-grid POLICY_GRID_ARGS="--port 8061"
POLICY_GRID_ARGS ?=
viz-policy-grid: $(INSTALL_MARKER)
	PYTHONPATH=$(CURDIR) $(PYTHON) tetris_bot/scripts/inspection/policy_grid_visualizer.py $(POLICY_GRID_ARGS)

# Force rebuild (clean first to avoid caching issues)
rebuild: ensure-rust $(INSTALL_MARKER)
	cd tetris_core && $(CARGO_ENV) && cargo clean
	$(MATURIN_ENV) && $(MATURIN_RELEASE_ENV) $(MATURIN_BUILD) --release && $(MATURIN_INSTALL)
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Run tests
test:
	$(MAKE) build-dev
	cd tetris_core && $(CARGO_ENV) && PYO3_PYTHON=$(PYO3_PYTHON) cargo test
	$(PYTHON) -m pytest

# Clean build artifacts
clean:
	rm -rf tetris_core/target $(RELEASE_MARKER) $(DEV_MARKER)

check:
	$(PYTHON) -m ruff check --fix
	$(PYTHON) -m ruff format
	$(CARGO_ENV) && cargo fix --manifest-path tetris_core/Cargo.toml --lib -p tetris_core --allow-dirty
	$(CARGO_ENV) && cargo fmt --manifest-path tetris_core/Cargo.toml
	$(PYTHON) -m pyright

# Run W&B sweep over learning rate and model size (builds first if needed)
# Usage: make sweep-lr-model ARGS="--count 20"
sweep-lr-model: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/wandb_sweep_lr_model_size.py $(ARGS)

# Evaluate one model across nn_value_weight values (no training)
# Usage: make eval-nn-value-weight ARGS="--run_dir training_runs/v17 --num_games 50"
eval-nn-value-weight: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/evaluate_nn_value_weight_sweep.py $(ARGS)

# Sweep an MCTS config parameter (e.g. c_puct, nn_value_weight, death_penalty) over multiple values
# Usage: make sweep-mcts-config ARGS="--run_dir training_runs/v32 --sweep_param c_puct --sweep_values '[0.75,1.0,1.5,2.0,2.5]'"
sweep-mcts-config: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/sweep_mcts_config.py $(ARGS)

# Compare offline network scaling variants (default, 2x board trunk, 2x post-fusion)
# Usage: make compare-offline-network-scaling ARGS="--data_path training_runs/v32/training_data.npz"
compare-offline-network-scaling: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/compare_offline_network_scaling.py $(ARGS)

# Compare the current gated-fusion network against a piece-conditioned 4x20x10 spatial policy decoder.
# Usage: make compare-offline-spatial-policy ARGS="--data_path training_runs/v32/training_data.npz"
compare-offline-spatial-policy: $(INSTALL_MARKER)
	PYTHONPATH=$(CURDIR) $(PYTHON) tetris_bot/scripts/ablations/compare_offline_spatial_policy.py $(ARGS)

# Compare several warm-started trunk sizes from one replay snapshot, then benchmark attack and throughput.
# Usage: make compare-warm-start-trunk-sizes ARGS="--source_run_dir training_runs/v32"
compare-warm-start-trunk-sizes: $(RELEASE_MARKER)
	PYTHONPATH=$(CURDIR) $(PYTHON) tetris_bot/scripts/ablations/compare_warm_start_trunk_sizes.py $(ARGS)

# Leela-Zero-style information gain per MCTS visit to pick the per-move search depth.
search-depth-info-gain: $(RELEASE_MARKER)
	PYTHONPATH=$(CURDIR) $(PYTHON) investigations/optimal_search_depth/search_depth_info_gain.py $(ARGS)

# Measure marginal utility of policy vs value loss by injecting runtime output noise and running games.
# Usage: make loss-sensitivity ARGS="training_runs/v32"
loss-sensitivity: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/loss_sensitivity_analysis.py $(ARGS)

# Build a new sibling run dir by offline-training on another run's replay snapshot.
# Usage: make warm-start ARGS="--source_run_dir training_runs/v3"
warm-start: $(RELEASE_MARKER)
	PYTHONPATH=$(CURDIR) $(PYTHON) tetris_bot/scripts/warm_start.py $(ARGS)

# Download training_data.npz directly from a WandB run/artifact.
# Usage:
#   make download-wandb-training-data ARGS="--reference entity/project/run_id --run_dir training_runs/v2 --overwrite true"
download-wandb-training-data: $(INSTALL_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/download_wandb_training_data.py $(ARGS)

# Profile game generation performance (builds first if needed)
# Usage: make profile MODEL=benchmarks/models/parallel.onnx SIMS=100 OUTPUT=benchmarks/profile.jsonl
# Usage (dummy/no-network): make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
MODEL_PROFILE ?= training_runs/v6/checkpoints/latest.onnx
SIMS ?= 1000
OUTPUT_PROFILE ?= benchmarks/profile_results.jsonl
WORKERS_PROFILE ?= 7
PROFILE_ARGS ?=
profile: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --num_workers $(WORKERS_PROFILE) --output $(OUTPUT_PROFILE) $(PROFILE_ARGS)

# Profile with samply (interactive flamegraph viewer)
# Usage: make profile-samply SIMS=50
# Requires: cargo install samply
profile-samply: $(RELEASE_MARKER)
	samply record $(PYTHON) tetris_bot/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --num_workers $(WORKERS_PROFILE) --num_games 3

# Auto-tune machine-specific build/backend/worker settings (cached by machine fingerprint)
# Defaults:
#   - adaptive worker search (not full grid)
#   - 20 games/config
#   - 2000 sims/move (matches training config)
MODEL_OPTIMIZE ?=
OPT_GAMES ?= 20
OPT_SIMS ?= 2000
OPT_REPEATS ?= 1
OPT_WORKER_SEARCH ?= adaptive
OPT_MAX_WORKER_EVALS ?= 6
OPT_BACKEND_STRATEGY ?= staged
OPT_PRIMARY_BACKEND ?= tract
OPTIMIZE_ARGS ?= --skip_build --backends tract
OPT_CACHE_DIR ?= benchmarks/profiles/optimize_cache
optimize: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/optimize_machine.py $(if $(MODEL_OPTIMIZE),--model_path $(MODEL_OPTIMIZE),) --num_games $(OPT_GAMES) --simulations $(OPT_SIMS) --num_repeats $(OPT_REPEATS) --worker_search $(OPT_WORKER_SEARCH) --max_worker_evals_per_combo $(OPT_MAX_WORKER_EVALS) --backend_strategy $(OPT_BACKEND_STRATEGY) --primary_backend $(OPT_PRIMARY_BACKEND) $(OPTIMIZE_ARGS)

# Build and train using machine-optimized settings when available.
# If no cached optimize result exists, this will attempt `make optimize` first.
# If optimization fails (e.g., missing model bundle on a fresh machine), training
# continues with default build/runtime settings.
train: ensure-rust $(INSTALL_MARKER)
	@if [ -z "$$TMUX" ]; then \
		echo "Error: tmux is not active. Training may stop if this terminal closes. Run inside tmux." >&2; \
		exit 1; \
	fi
	@set -euo pipefail; \
	FINGERPRINT="$$( $(PYTHON) -c "from tetris_bot.scripts.inspection.optimize_machine import machine_profile, machine_type_fingerprint; print(machine_type_fingerprint(machine_profile()))" 2>/dev/null || true )"; \
	OPT_ENV=""; \
	if [ -n "$$FINGERPRINT" ]; then \
		OPT_ENV="$(OPT_CACHE_DIR)/$$FINGERPRINT.env"; \
	fi; \
	if [ -z "$$OPT_ENV" ] || [ ! -f "$$OPT_ENV" ]; then \
		echo "[train] No optimization cache for this machine; running make optimize..."; \
		if ! $(MAKE) optimize MODEL_OPTIMIZE="$(MODEL_OPTIMIZE)" OPT_GAMES="$(OPT_GAMES)" OPT_SIMS="$(OPT_SIMS)" OPT_REPEATS="$(OPT_REPEATS)" OPT_WORKER_SEARCH="$(OPT_WORKER_SEARCH)" OPT_MAX_WORKER_EVALS="$(OPT_MAX_WORKER_EVALS)" OPTIMIZE_ARGS="$(OPTIMIZE_ARGS)"; then \
			echo "[train] Optimization failed; falling back to default build/runtime settings."; \
			OPT_ENV=""; \
		fi; \
	fi; \
	if [ -n "$$OPT_ENV" ] && [ -f "$$OPT_ENV" ]; then \
		echo "[train] Loading optimized settings from $$OPT_ENV"; \
		set -a; . "$$OPT_ENV"; set +a; \
		if [ "$${TETRIS_NN_BACKEND:-tract}" = "ort" ]; then \
			if ! $(MAKE) build-ort RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; then \
				echo "[train] ORT build failed; falling back to tract build/runtime settings."; \
				export TETRIS_NN_BACKEND=tract; \
				$(MAKE) build RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; \
			fi; \
		else \
			$(MAKE) build RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; \
		fi; \
	else \
		$(MAKE) build; \
	fi; \
	$(PYTHON) tetris_bot/scripts/train.py $(ARGS)

# Run the generator-only entrypoint for multi-machine training. Same
# build + optimize-cache logic as `train` so the laptop's self-play uses
# release-optimized Rust + machine-tuned worker counts. ARGS pass-through
# lets you set --machine_id, --num_workers, --sync_run_id, etc.
run-generator: ensure-rust $(INSTALL_MARKER)
	@if [ -z "$$TMUX" ]; then \
		echo "Error: tmux is not active. Generator may stop if this terminal closes. Run inside tmux." >&2; \
		exit 1; \
	fi
	@set -euo pipefail; \
	FINGERPRINT="$$( $(PYTHON) -c "from tetris_bot.scripts.inspection.optimize_machine import machine_profile, machine_type_fingerprint; print(machine_type_fingerprint(machine_profile()))" 2>/dev/null || true )"; \
	OPT_ENV=""; \
	if [ -n "$$FINGERPRINT" ]; then \
		OPT_ENV="$(OPT_CACHE_DIR)/$$FINGERPRINT.env"; \
	fi; \
	if [ -z "$$OPT_ENV" ] || [ ! -f "$$OPT_ENV" ]; then \
		echo "[run-generator] No optimization cache for this machine; running make optimize..."; \
		if ! $(MAKE) optimize MODEL_OPTIMIZE="$(MODEL_OPTIMIZE)" OPT_GAMES="$(OPT_GAMES)" OPT_SIMS="$(OPT_SIMS)" OPT_REPEATS="$(OPT_REPEATS)" OPT_WORKER_SEARCH="$(OPT_WORKER_SEARCH)" OPT_MAX_WORKER_EVALS="$(OPT_MAX_WORKER_EVALS)" OPTIMIZE_ARGS="$(OPTIMIZE_ARGS)"; then \
			echo "[run-generator] Optimization failed; falling back to default build/runtime settings."; \
			OPT_ENV=""; \
		fi; \
	fi; \
	if [ -n "$$OPT_ENV" ] && [ -f "$$OPT_ENV" ]; then \
		echo "[run-generator] Loading optimized settings from $$OPT_ENV"; \
		set -a; . "$$OPT_ENV"; set +a; \
		if [ "$${TETRIS_NN_BACKEND:-tract}" = "ort" ]; then \
			if ! $(MAKE) build-ort RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; then \
				echo "[run-generator] ORT build failed; falling back to tract build/runtime settings."; \
				export TETRIS_NN_BACKEND=tract; \
				$(MAKE) build RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; \
			fi; \
		else \
			$(MAKE) build RELEASE_RUSTFLAGS="$${RELEASE_RUSTFLAGS-$(RELEASE_RUSTFLAGS)}" RELEASE_LTO="$${RELEASE_LTO-$(RELEASE_LTO)}" RELEASE_CODEGEN_UNITS="$${RELEASE_CODEGEN_UNITS-$(RELEASE_CODEGEN_UNITS)}"; \
		fi; \
	else \
		$(MAKE) build; \
	fi; \
	$(PYTHON) tetris_bot/scripts/run_generator.py $(ARGS)
