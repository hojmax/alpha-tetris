"""Generator-only entrypoint for multi-machine training.

Run this on a machine that should produce self-play games for a remote trainer.
The script:

1. Auto-discovers active runs from R2 (or uses `--sync_run_id` if given) and
   downloads the trainer's current incumbent ONNX bundle.
2. Spawns a Rust GameGenerator with `candidate_gating_enabled=False` (the
   trainer is the source of truth for promotion decisions).
3. Starts a `ChunkUploader` that drains replay deltas to R2.
4. Starts a `ModelDownloader` that polls the incumbent pointer and applies new
   bundles via `sync_model_directly`.
5. Runs until SIGINT/SIGTERM, then shuts everything down cleanly.

Usage:
    R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    python tetris_bot/scripts/run_generator.py \
        --config config.yaml \
        --machine_id laptop \
        --num_workers 6
"""

from __future__ import annotations

import signal
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv
from simple_parsing import parse

# Load `.env` (R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)
# before any code that touches `os.environ` for those keys.
load_dotenv()

from tetris_core.tetris_core import GameGenerator, MCTSConfig  # noqa: E402

from tetris_bot.constants import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import (  # noqa: E402
    SELF_PLAY_SNAPSHOT_FIELDS,
    R2SyncConfig,
    SelfPlayConfig,
    SelfPlaySnapshot,
    load_training_config,
)
from tetris_bot.ml.r2_sync import (  # noqa: E402
    ChunkUploader,
    DiscoveredRun,
    GameStatsUploader,
    ModelDownloader,
    R2Settings,
    SelfPlaySnapshotMissing,
    discover_active_runs,
    download_model_bundle,
    fetch_model_pointer,
    fetch_self_play_snapshot,
    make_s3_client,
)
from tetris_bot.run_setup import apply_optimized_runtime_overrides  # noqa: E402

logger = structlog.get_logger()


@dataclass
class GeneratorArgs:
    """Generator script arguments."""

    config: Path = DEFAULT_CONFIG_PATH
    workspace: Path = Path("generator_workspace")
    machine_id: str | None = None  # defaults to hostname
    num_workers: int | None = None  # overrides config.self_play.num_workers
    sync_run_id: str | None = None  # overrides r2_sync.sync_run_id
    bootstrap_wait_seconds: float = 600.0


def _build_mcts_config(self_play: SelfPlayConfig) -> MCTSConfig:
    cfg = MCTSConfig()
    cfg.num_simulations = self_play.num_simulations
    cfg.c_puct = self_play.c_puct
    cfg.temperature = self_play.temperature
    cfg.dirichlet_alpha = self_play.dirichlet_alpha
    cfg.dirichlet_epsilon = self_play.dirichlet_epsilon
    cfg.visit_sampling_epsilon = self_play.visit_sampling_epsilon
    cfg.seed = self_play.mcts_seed
    cfg.max_placements = self_play.max_placements
    cfg.death_penalty = self_play.penalty_schedule.death_penalty
    cfg.overhang_penalty_weight = self_play.penalty_schedule.overhang_penalty_weight
    cfg.nn_value_weight = self_play.nn_value_weight_schedule.initial
    cfg.use_parent_value_for_unvisited_q = self_play.use_parent_value_for_unvisited_q
    cfg.reuse_tree = self_play.reuse_tree
    cfg.count_reused_visits = self_play.count_reused_visits
    cfg.min_new_simulations = self_play.min_new_simulations
    return cfg


def _format_discovered_run(run: DiscoveredRun, *, now: datetime) -> str:
    age_s = (now - run.incumbent_modified).total_seconds()
    if age_s < 60:
        age_str = f"{age_s:.0f}s ago"
    elif age_s < 3600:
        age_str = f"{age_s / 60:.1f}m ago"
    else:
        age_str = f"{age_s / 3600:.1f}h ago"
    return (
        f"{run.sync_run_id}  step={run.pointer.step}  "
        f"updated={age_str}  ({run.incumbent_modified.isoformat()})"
    )


def _select_sync_run_id(
    cfg: R2SyncConfig,
    *,
    explicit_override: str | None,
    interactive: bool,
    wait_seconds: float = 600.0,
    poll_interval_seconds: float = 5.0,
) -> str:
    """Determine which sync_run_id to join.

    Order of precedence:
    1. `--sync_run_id` CLI flag (explicit override).
    2. `r2_sync.sync_run_id` set in config (rare; we no longer pin it).
    3. Auto-discovery: list `<prefix>/`, pick the run with the most recent
       `models/incumbent.json` if one exists. Polls for up to
       `wait_seconds` so a generator can join during the trainer's
       bootstrap phase (no incumbent.json yet) and start as soon as
       the first promotion lands. If multiple are fresh
       (`active_run_freshness_seconds`), prompt the user (unless
       `interactive=False`, in which case raise).
    4. Otherwise raise — no run appeared in the wait window.
    """
    if explicit_override is not None:
        return explicit_override
    if cfg.sync_run_id is not None:
        return cfg.sync_run_id
    if cfg.bucket is None:
        raise ValueError("r2_sync.bucket must be set to auto-discover runs")
    # Build a temporary settings shim with a placeholder run_id so we can
    # construct an S3 client using the same config + env-var discovery
    # as the main path.
    discovery_cfg = cfg.model_copy(update={"sync_run_id": "_discovery"})
    discovery_settings = R2Settings.from_config(discovery_cfg)
    client = make_s3_client(discovery_settings)
    deadline = time.monotonic() + wait_seconds
    runs: list[DiscoveredRun] = []
    while True:
        runs = discover_active_runs(
            bucket=discovery_settings.bucket,
            prefix=discovery_settings.prefix,
            client=client,
        )
        if runs:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"No runs found under s3://{discovery_settings.bucket}"
                f"/{discovery_settings.prefix}/ after waiting "
                f"{wait_seconds:.0f}s. Start a trainer first, or pass "
                "--sync_run_id explicitly."
            )
        logger.info(
            "run_generator.waiting_for_active_run",
            bucket=discovery_settings.bucket,
            prefix=discovery_settings.prefix,
            poll_interval_seconds=poll_interval_seconds,
        )
        time.sleep(poll_interval_seconds)

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    fresh_window_s = cfg.active_run_freshness_seconds
    fresh = [
        r
        for r in runs
        if (now - r.incumbent_modified).total_seconds() <= fresh_window_s
    ]
    if not fresh:
        # Nothing recent — pick the freshest available and warn.
        chosen = runs[0]
        logger.warning(
            "run_generator.no_fresh_runs_using_most_recent",
            sync_run_id=chosen.sync_run_id,
            updated=chosen.incumbent_modified.isoformat(),
            available_count=len(runs),
        )
        return chosen.sync_run_id
    if len(fresh) == 1:
        chosen = fresh[0]
        logger.info(
            "run_generator.auto_selected_run",
            sync_run_id=chosen.sync_run_id,
            updated=chosen.incumbent_modified.isoformat(),
        )
        return chosen.sync_run_id
    # Multiple fresh runs — prompt.
    if not interactive:
        raise RuntimeError(
            f"Multiple fresh runs found ({len(fresh)}). Pass --sync_run_id "
            "explicitly or run interactively. Candidates:\n  "
            + "\n  ".join(_format_discovered_run(r, now=now) for r in fresh)
        )
    print("\nMultiple active runs found. Pick one to join:")
    for i, r in enumerate(fresh, start=1):
        print(f"  [{i}] {_format_discovered_run(r, now=now)}")
    while True:
        raw = input(f"\nEnter choice [1-{len(fresh)}]: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(fresh):
                return fresh[idx - 1].sync_run_id
        print("Invalid choice; try again.")


def _resolve_r2_settings(
    cfg: R2SyncConfig,
    *,
    explicit_run_id: str | None,
    interactive: bool,
    wait_seconds: float = 600.0,
) -> R2Settings:
    if not cfg.enabled:
        raise ValueError(
            "r2_sync.enabled must be true in the config used by run_generator.py"
        )
    sync_run_id = _select_sync_run_id(
        cfg,
        explicit_override=explicit_run_id,
        interactive=interactive,
        wait_seconds=wait_seconds,
    )
    cfg = cfg.model_copy(update={"sync_run_id": sync_run_id})
    return R2Settings.from_config(cfg)


def _apply_trainer_self_play_snapshot(
    settings: R2Settings, self_play: SelfPlayConfig
) -> SelfPlaySnapshot:
    """Fetch the trainer's self_play snapshot and overwrite local fields.

    Hard-fails if the snapshot is missing: per CLAUDE.md, generators must
    not silently fall back to local config (that is the drift surface this
    is designed to remove).
    """
    snapshot = fetch_self_play_snapshot(settings)
    if snapshot is None:
        raise SelfPlaySnapshotMissing(
            f"No self_play snapshot at s3://{settings.bucket}"
            f"/{settings.self_play_snapshot_key()}. The trainer publishes "
            "this on startup; if missing the trainer is too old or its R2 "
            "init failed."
        )
    overrides = {
        f: (getattr(self_play, f), getattr(snapshot, f))
        for f in SELF_PLAY_SNAPSHOT_FIELDS
        if getattr(self_play, f) != getattr(snapshot, f)
    }
    snapshot.apply_to(self_play)
    logger.info(
        "run_generator.self_play_snapshot_applied",
        key=settings.self_play_snapshot_key(),
        overridden_fields={
            k: {"local": v[0], "trainer": v[1]} for k, v in overrides.items()
        },
    )
    return snapshot


def _wait_for_initial_pointer(settings: R2Settings, timeout_seconds: float):
    """Block until the trainer publishes an incumbent pointer, or raise."""
    deadline = time.monotonic() + timeout_seconds
    client = make_s3_client(settings)
    poll_interval = 5.0
    while time.monotonic() < deadline:
        pointer = fetch_model_pointer(settings, client=client)
        if pointer is not None:
            return pointer
        logger.info(
            "run_generator.waiting_for_initial_model",
            sync_run_id=settings.sync_run_id,
            poll_interval_seconds=poll_interval,
        )
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting {timeout_seconds:.0f}s for initial model pointer at "
        f"{settings.bucket}/{settings.model_pointer_key()}"
    )


def main(args: GeneratorArgs) -> None:
    config = load_training_config(args.config.resolve())
    apply_optimized_runtime_overrides(config)
    import sys

    settings = _resolve_r2_settings(
        config.r2_sync,
        explicit_run_id=args.sync_run_id,
        interactive=sys.stdin.isatty(),
        wait_seconds=args.bootstrap_wait_seconds,
    )
    machine_id = args.machine_id or socket.gethostname().replace(".", "_")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    models_dir = workspace / "models"
    cursor_path = workspace / "r2_upload_cursor.json"
    game_stats_cursor_path = workspace / "r2_game_stats_cursor.json"
    training_data_path = workspace / TRAINING_DATA_FILENAME

    logger.info(
        "run_generator.starting",
        sync_run_id=settings.sync_run_id,
        machine_id=machine_id,
        workspace=str(workspace),
        endpoint_url=settings.endpoint_url,
        bucket=settings.bucket,
    )

    _apply_trainer_self_play_snapshot(settings, config.self_play)
    pointer = _wait_for_initial_pointer(settings, args.bootstrap_wait_seconds)
    bundle_dir = models_dir / f"step_{pointer.step:020d}"
    if bundle_dir.exists():
        logger.info(
            "run_generator.using_cached_initial_bundle",
            step=pointer.step,
            local_path=str(bundle_dir),
        )
        initial_main = bundle_dir / "bundle.onnx"
        if not initial_main.exists():
            initial_main = download_model_bundle(
                settings=settings, pointer=pointer, dest_dir=bundle_dir
            )
    else:
        initial_main = download_model_bundle(
            settings=settings, pointer=pointer, dest_dir=bundle_dir
        )
    logger.info(
        "run_generator.initial_bundle_ready",
        step=pointer.step,
        nn_value_weight=pointer.nn_value_weight,
        death_penalty=pointer.death_penalty,
        overhang_penalty_weight=pointer.overhang_penalty_weight,
        local_path=str(initial_main),
    )

    num_workers = args.num_workers or config.self_play.num_workers
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")

    mcts_config = _build_mcts_config(config.self_play)
    mcts_config.nn_value_weight = pointer.nn_value_weight
    mcts_config.death_penalty = pointer.death_penalty
    mcts_config.overhang_penalty_weight = pointer.overhang_penalty_weight
    generator = GameGenerator(
        model_path=str(initial_main),
        training_data_path=str(training_data_path),
        config=mcts_config,
        max_placements=config.self_play.max_placements,
        add_noise=config.self_play.add_noise,
        max_examples=config.replay.buffer_size,
        save_interval_seconds=0.0,
        num_workers=num_workers,
        initial_model_step=pointer.step,
        candidate_eval_seeds=[],
        start_with_network=True,
        non_network_num_simulations=config.self_play.bootstrap_num_simulations,
        initial_incumbent_eval_avg_attack=0.0,
        candidate_gating_enabled=False,
        save_eval_trees=False,
    )
    generator.start()
    logger.info("run_generator.generator_started", num_workers=num_workers)

    uploader = ChunkUploader(
        generator=generator,
        settings=settings,
        machine_id=machine_id,
        cursor_path=cursor_path,
        chunk_max_examples=config.r2_sync.chunk_max_examples,
        upload_interval_seconds=config.r2_sync.chunk_upload_interval_seconds,
    )
    game_stats_uploader = GameStatsUploader(
        generator=generator,
        settings=settings,
        machine_id=machine_id,
        cursor_path=game_stats_cursor_path,
        upload_interval_seconds=config.r2_sync.chunk_upload_interval_seconds,
    )
    downloader = ModelDownloader(
        generator=generator,
        settings=settings,
        local_models_dir=models_dir,
        poll_interval_seconds=config.r2_sync.model_pointer_poll_interval_seconds,
        initial_model_step=pointer.step,
    )
    uploader.start()
    game_stats_uploader.start()
    downloader.start()

    stop_event_handled = False

    def _shutdown(signum: int, _frame) -> None:
        nonlocal stop_event_handled
        if stop_event_handled:
            return
        stop_event_handled = True
        logger.info("run_generator.shutdown_signal", signum=signum)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log_interval_s = config.run.log_interval_seconds
    start_time_s = time.monotonic()
    start_games = generator.games_generated()
    start_examples = generator.examples_generated()
    last_log_time_s = start_time_s
    last_log_games = start_games
    last_log_examples = start_examples
    snapshot_poll_interval_s = config.r2_sync.model_pointer_poll_interval_seconds
    last_snapshot_poll_s = start_time_s
    last_applied_add_noise = config.self_play.add_noise
    last_applied_visit_sampling_epsilon = config.self_play.visit_sampling_epsilon
    try:
        while not stop_event_handled:
            time.sleep(0.5)
            now_s = time.monotonic()
            if now_s - last_snapshot_poll_s >= snapshot_poll_interval_s:
                last_snapshot_poll_s = now_s
                try:
                    refreshed_snapshot = fetch_self_play_snapshot(settings)
                except Exception:
                    logger.exception("run_generator.self_play_snapshot_refresh_failed")
                    refreshed_snapshot = None
                if refreshed_snapshot is not None:
                    new_add_noise = refreshed_snapshot.add_noise
                    new_visit_sampling_epsilon = (
                        refreshed_snapshot.visit_sampling_epsilon
                    )
                    if (
                        new_add_noise != last_applied_add_noise
                        or new_visit_sampling_epsilon
                        != last_applied_visit_sampling_epsilon
                    ):
                        try:
                            generator.update_search_overrides(
                                add_noise=new_add_noise,
                                visit_sampling_epsilon=new_visit_sampling_epsilon,
                            )
                        except Exception:
                            logger.exception(
                                "run_generator.update_search_overrides_failed",
                                add_noise=new_add_noise,
                                visit_sampling_epsilon=new_visit_sampling_epsilon,
                            )
                        else:
                            logger.info(
                                "run_generator.applied_self_play_snapshot_overrides",
                                add_noise=new_add_noise,
                                visit_sampling_epsilon=new_visit_sampling_epsilon,
                                previous_add_noise=last_applied_add_noise,
                                previous_visit_sampling_epsilon=(
                                    last_applied_visit_sampling_epsilon
                                ),
                            )
                            config.self_play.add_noise = new_add_noise
                            config.self_play.visit_sampling_epsilon = (
                                new_visit_sampling_epsilon
                            )
                            last_applied_add_noise = new_add_noise
                            last_applied_visit_sampling_epsilon = (
                                new_visit_sampling_epsilon
                            )
            if now_s - last_log_time_s >= log_interval_s:
                games_now = generator.games_generated()
                examples_now = generator.examples_generated()
                window_s = now_s - last_log_time_s
                total_s = now_s - start_time_s
                games_per_second = (games_now - last_log_games) / window_s
                examples_per_second = (examples_now - last_log_examples) / window_s
                avg_games_per_second = (games_now - start_games) / total_s
                avg_examples_per_second = (examples_now - start_examples) / total_s
                logger.info(
                    "run_generator.progress",
                    games_generated=games_now,
                    examples_generated=examples_now,
                    games_per_second=games_per_second,
                    examples_per_second=examples_per_second,
                    avg_games_per_second=avg_games_per_second,
                    avg_examples_per_second=avg_examples_per_second,
                    buffer_size=generator.buffer_size(),
                    window_seconds=window_s,
                    elapsed_seconds=total_s,
                )
                last_log_time_s = now_s
                last_log_games = games_now
                last_log_examples = examples_now
    finally:
        logger.info("run_generator.stopping_workers")
        try:
            generator.stop()
        except Exception:
            logger.exception("run_generator.generator_stop_failed")
        downloader.stop()
        game_stats_uploader.stop()
        uploader.stop()
        logger.info(
            "run_generator.shutdown_complete",
            games_generated=generator.games_generated(),
            examples_generated=generator.examples_generated(),
        )


if __name__ == "__main__":
    main(parse(GeneratorArgs))
