use super::*;
use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, ROW_FILL_FEATURE_ROWS};
use crate::runtime::test_utils;
use crate::search::NUM_ACTIONS;
use std::fs;

fn make_example(move_number: u32) -> TrainingExample {
    let mut policy = vec![0.0; NUM_ACTIONS];
    policy[0] = 1.0;
    let mut action_mask = vec![false; NUM_ACTIONS];
    action_mask[0] = true;

    TrainingExample {
        board: vec![0; BOARD_HEIGHT * BOARD_WIDTH],
        current_piece: 0,
        hold_piece: 7,
        hold_available: true,
        next_queue: vec![0, 1, 2, 3, 4],
        move_number,
        placement_count: move_number as f32 / 100.0,
        combo: 0.0,
        back_to_back: false,
        next_hidden_piece_probs: vec![1.0 / NUM_PIECE_TYPES as f32; NUM_PIECE_TYPES],
        column_heights: vec![0.0; BOARD_WIDTH],
        max_column_height: 0.0,
        row_fill_counts: vec![0.0; ROW_FILL_FEATURE_ROWS],
        total_blocks: 0.0,
        bumpiness: 0.0,
        holes: 0.0,
        policy,
        value: move_number as f32,
        action_mask,
        overhang_fields: crate::search::normalize_overhang_fields(move_number),
        game_number: 0,
        game_total_attack: 0,
    }
}

fn unique_temp_path(name: &str) -> PathBuf {
    test_utils::unique_temp_path("tetris_generator", name)
}

#[test]
fn test_shared_buffer_fifo_eviction() {
    let buffer = SharedBuffer::new(3);
    buffer.add_examples(vec![make_example(1), make_example(2)]);
    buffer.add_examples(vec![make_example(3), make_example(4)]);

    let (_, _, kept) = buffer.logical_window_snapshot().unwrap();
    assert_eq!(kept.len(), 3);
    let move_numbers: Vec<u32> = kept.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3, 4]);
}

#[test]
fn test_shared_buffer_logical_indices_follow_fifo_window() {
    let buffer = SharedBuffer::new(3);
    assert!(buffer.logical_window_snapshot().is_none());

    buffer.add_examples(vec![make_example(1), make_example(2)]);
    let (window_start, window_end, snapshot) = buffer
        .logical_window_snapshot()
        .expect("snapshot should exist after adding examples");
    assert_eq!(window_start, 0);
    assert_eq!(window_end, 2);
    let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![1, 2]);

    buffer.add_examples(vec![make_example(3), make_example(4)]);
    let (window_start, window_end, snapshot) = buffer
        .logical_window_snapshot()
        .expect("snapshot should exist after eviction");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3, 4]);

    let (window_start, window_end, slice_start, slice) = buffer
        .logical_delta_slice(2, 10)
        .expect("delta should exist");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    assert_eq!(slice_start, 2);
    let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![3, 4]);

    let (window_start, window_end, slice_start, slice) = buffer
        .logical_delta_slice(0, 2)
        .expect("delta should clamp to window start");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    assert_eq!(slice_start, 1);
    let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3]);
}

#[test]
fn test_snapshot_persister_writes_buffer_snapshot() {
    let buffer = SharedBuffer::new(4);
    buffer.add_examples(vec![make_example(1), make_example(2), make_example(3)]);
    let snapshot_path = unique_temp_path("snapshot").with_extension("npz");
    let persister = SnapshotPersister::new();

    assert!(persister.submit_buffer_snapshot(&buffer, &snapshot_path));
    persister.shutdown();

    let loaded = read_examples_from_npz(&snapshot_path).expect("snapshot should load");
    let move_numbers: Vec<u32> = loaded.iter().map(|example| example.move_number).collect();
    assert_eq!(move_numbers, vec![1, 2, 3]);
}

#[test]
fn test_snapshot_persister_shutdown_flushes_latest_pending_snapshot() {
    let snapshot_path = unique_temp_path("latest_snapshot").with_extension("npz");
    let persister = SnapshotPersister::new();

    let compact = |move_number| CompactExample::from_example(&make_example(move_number));
    persister.submit_snapshot(snapshot_path.clone(), vec![compact(1), compact(2)]);
    persister.submit_snapshot(snapshot_path.clone(), vec![compact(7)]);
    persister.shutdown();

    let loaded = read_examples_from_npz(&snapshot_path).expect("snapshot should load");
    let move_numbers: Vec<u32> = loaded.iter().map(|example| example.move_number).collect();
    assert_eq!(move_numbers, vec![7]);
}

fn make_game_result(
    total_attack: u32,
    total_lines: u32,
    max_combo: u32,
    move_numbers: &[u32],
) -> GameResult {
    GameResult {
        examples: move_numbers.iter().copied().map(make_example).collect(),
        total_attack,
        num_moves: move_numbers.len() as u32,
        avg_valid_actions: 5.0,
        max_valid_actions: 9,
        stats: GameStats {
            singles: total_lines,
            doubles: 0,
            triples: 0,
            tetrises: 0,
            tspin_minis: 0,
            tspin_singles: 0,
            tspin_doubles: 0,
            tspin_triples: 0,
            perfect_clears: 0,
            back_to_backs: 0,
            max_combo,
            total_lines,
            holds: 1,
        },
        tree_stats: GameTreeStats::default(),
        total_overhang_fields: 0,
        avg_overhang_fields: 0.0,
        cache_hits: 0,
        cache_misses: 0,
        cache_size: 0,
        tree_reuse_hits: 0,
        tree_reuse_misses: 0,
        tree_reuse_carry_fraction: 0.0,
        traversal_total: 0,
        traversal_expansions: 0,
        traversal_terminal_ends: 0,
        traversal_horizon_ends: 0,
        traversal_expansion_fraction: 0.0,
        traversal_terminal_fraction: 0.0,
        traversal_horizon_fraction: 0.0,
        trajectory_predicted_total_attack_count: 0,
        trajectory_predicted_total_attack_variance: 0.0,
        trajectory_predicted_total_attack_std: 0.0,
        trajectory_predicted_total_attack_rmse: 0.0,
    }
}

fn make_completed_game_result(
    seed: u64,
    completed_time_s: f64,
    total_attack: u32,
    total_lines: u32,
    max_combo: u32,
    move_numbers: &[u32],
) -> CompletedGameResult {
    CompletedGameResult {
        replay: Some(GameReplay {
            seed,
            moves: move_numbers
                .iter()
                .map(|move_number| ReplayMove {
                    action: 0,
                    attack: *move_number,
                })
                .collect(),
            total_attack,
            num_moves: move_numbers.len() as u32,
        }),
        completed_time_s,
        result: make_game_result(total_attack, total_lines, max_combo, move_numbers),
    }
}

#[test]
fn test_commit_game_results_batch_tags_examples_and_queues_completed_games() {
    let buffer = Arc::new(SharedBuffer::new(8));
    let games_generated = Arc::new(AtomicU64::new(0));
    let examples_generated = Arc::new(AtomicU64::new(0));
    let completed_games = Arc::new(RwLock::new(VecDeque::new()));
    let shared = WorkerSharedState {
        buffer: Arc::clone(&buffer),
        running: Arc::new(AtomicBool::new(true)),
        games_generated: Arc::clone(&games_generated),
        examples_generated: Arc::clone(&examples_generated),
        completed_games: Arc::clone(&completed_games),
        pending_candidate: Arc::new(RwLock::new(None)),
        evaluating_candidate: Arc::new(RwLock::new(None)),
        model_eval_events: Arc::new(RwLock::new(VecDeque::new())),
        incumbent: IncumbentState {
            model_path: Arc::new(RwLock::new(PathBuf::from("unused.onnx"))),
            uses_network: Arc::new(AtomicBool::new(false)),
            model_step: Arc::new(AtomicU64::new(0)),
            model_version: Arc::new(AtomicU64::new(0)),
            nn_value_weight: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            death_penalty: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            overhang_penalty_weight: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            eval_avg_attack: Arc::new(AtomicU32::new(0.0f32.to_bits())),
        },
        live_overrides: LiveSearchOverrides {
            add_noise: Arc::new(AtomicBool::new(false)),
            visit_sampling_epsilon: Arc::new(AtomicU32::new(0.0f32.to_bits())),
        },
    };

    let committed = GameGenerator::commit_game_results_batch(
        vec![
            make_completed_game_result(11, 10.5, 7, 4, 2, &[1, 2]),
            make_completed_game_result(22, 11.5, 3, 1, 1, &[3]),
        ],
        &shared,
    );

    assert_eq!(committed, 2);
    assert_eq!(games_generated.load(Ordering::SeqCst), 2);
    assert_eq!(examples_generated.load(Ordering::SeqCst), 3);

    let (_, _, buffered_examples) = buffer
        .logical_window_snapshot()
        .expect("buffer snapshot should exist");
    let game_numbers: Vec<u64> = buffered_examples
        .iter()
        .map(|example| example.game_number)
        .collect();
    let attacks: Vec<u32> = buffered_examples
        .iter()
        .map(|example| example.game_total_attack)
        .collect();
    assert_eq!(game_numbers, vec![1, 1, 2]);
    assert_eq!(attacks, vec![7, 7, 3]);

    let completed_games = completed_games.read().unwrap();
    assert_eq!(completed_games.len(), 2);
    assert_eq!(completed_games[0].game_number, 1);
    assert_eq!(completed_games[0].completed_time_s, 10.5);
    assert_eq!(completed_games[0].total_attack, 7);
    assert_eq!(completed_games[0].replay.as_ref().unwrap().seed, 11);
    assert_eq!(completed_games[0].stats.total_lines, 4);
    assert_eq!(completed_games[1].game_number, 2);
    assert_eq!(completed_games[1].total_attack, 3);
    assert_eq!(completed_games[1].stats.max_combo, 1);
}

#[test]
fn test_build_candidate_eval_config_forces_deterministic_gate_settings() {
    let mut config = MCTSConfig::default();
    config.seed = None;
    config.visit_sampling_epsilon = 0.37;
    config.num_simulations = 321;
    config.temperature = 0.9;
    config.dirichlet_alpha = 0.02;
    config.dirichlet_epsilon = 0.4;

    let eval_config = GameGenerator::build_candidate_eval_config(&config);
    assert_eq!(eval_config.seed, Some(0));
    assert_eq!(eval_config.visit_sampling_epsilon, 0.0);
    assert_eq!(eval_config.num_simulations, 321);
    assert_eq!(eval_config.temperature, 0.9);
    assert_eq!(eval_config.dirichlet_alpha, 0.02);
    assert_eq!(eval_config.dirichlet_epsilon, 0.4);
}

#[test]
fn test_incumbent_eval_rebaseline_only_when_penalties_change_for_network_incumbent() {
    assert!(!GameGenerator::needs_incumbent_eval_rebaseline(
        false,
        (5.0, 3.0),
        (0.0, 0.0),
    ));
    assert!(!GameGenerator::needs_incumbent_eval_rebaseline(
        true,
        (5.0, 3.0),
        (5.0, 3.0),
    ));
    assert!(GameGenerator::needs_incumbent_eval_rebaseline(
        true,
        (5.0, 3.0),
        (0.0, 0.0),
    ));
}

#[test]
fn test_build_rollout_config_keeps_sampling_settings() {
    let mut config = MCTSConfig::default();
    config.num_simulations = 123;
    config.visit_sampling_epsilon = 0.42;
    config.temperature = 1.5;
    config.dirichlet_alpha = 0.02;
    config.dirichlet_epsilon = 0.3;
    config.nn_value_weight = 0.123;
    config.death_penalty = 5.0;
    config.overhang_penalty_weight = 3.0;
    config.use_parent_value_for_unvisited_q = true;

    let network_config = GameGenerator::build_rollout_config(&config, true, 999, 0.123, 5.0, 3.0);
    assert_eq!(network_config.num_simulations, 123);
    assert_eq!(network_config.visit_sampling_epsilon, 0.42);
    assert_eq!(network_config.temperature, 1.5);
    assert_eq!(network_config.dirichlet_alpha, 0.02);
    assert_eq!(network_config.dirichlet_epsilon, 0.3);
    assert_eq!(network_config.nn_value_weight, 0.123);
    assert_eq!(network_config.death_penalty, 5.0);
    assert_eq!(network_config.overhang_penalty_weight, 3.0);
    assert!(network_config.use_parent_value_for_unvisited_q);

    let bootstrap_config =
        GameGenerator::build_rollout_config(&config, false, 999, 0.456, 5.0, 3.0);
    assert_eq!(bootstrap_config.num_simulations, 999);
    assert_eq!(bootstrap_config.visit_sampling_epsilon, 0.42);
    assert_eq!(bootstrap_config.temperature, 1.5);
    assert_eq!(bootstrap_config.dirichlet_alpha, 0.02);
    assert_eq!(bootstrap_config.dirichlet_epsilon, 0.3);
    assert_eq!(bootstrap_config.nn_value_weight, 0.456);
    assert_eq!(bootstrap_config.death_penalty, 5.0);
    assert_eq!(bootstrap_config.overhang_penalty_weight, 3.0);
    assert!(bootstrap_config.use_parent_value_for_unvisited_q);

    // Verify penalties are overridden when passed as 0
    let zeroed_config = GameGenerator::build_rollout_config(&config, true, 999, 1.0, 0.0, 0.0);
    assert_eq!(zeroed_config.death_penalty, 0.0);
    assert_eq!(zeroed_config.overhang_penalty_weight, 0.0);
}

#[test]
fn test_model_artifact_paths_cover_all_split_outputs() {
    let onnx_path = unique_temp_path("candidate").with_extension("onnx");
    let artifacts = GameGenerator::model_artifact_paths(&onnx_path);
    assert_eq!(artifacts[0], onnx_path);
    assert_eq!(artifacts[1], onnx_path.with_extension("onnx.data"));
    assert_eq!(artifacts[2], onnx_path.with_extension("conv.onnx"));
    assert_eq!(artifacts[3], onnx_path.with_extension("conv.onnx.data"));
    assert_eq!(artifacts[4], onnx_path.with_extension("heads.onnx"));
    assert_eq!(artifacts[5], onnx_path.with_extension("heads.onnx.data"));
    assert_eq!(artifacts[6], onnx_path.with_extension("fc.bin"));
}

#[test]
fn test_remove_model_artifacts_if_safe_preserves_bootstrap_and_incumbent() {
    let model_path = unique_temp_path("cleanup").with_extension("onnx");
    let artifacts = GameGenerator::model_artifact_paths(&model_path);
    for artifact in &artifacts {
        fs::write(artifact, b"model").expect("temp file write should succeed");
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &model_path,
        &PathBuf::from("different.onnx"),
        None,
    );
    for artifact in &artifacts {
        assert!(
            artifact.exists(),
            "bootstrap artifacts should not be removed"
        );
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &PathBuf::from("bootstrap.onnx"),
        &model_path,
        None,
    );
    for artifact in &artifacts {
        assert!(
            artifact.exists(),
            "incumbent artifacts should not be removed"
        );
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &PathBuf::from("bootstrap.onnx"),
        &PathBuf::from("incumbent.onnx"),
        None,
    );
    for artifact in &artifacts {
        assert!(!artifact.exists(), "candidate artifacts should be removed");
    }
}

#[test]
fn test_cleanup_queued_candidate_artifacts_removes_pending_and_evaluating() {
    let bootstrap_path = unique_temp_path("bootstrap").with_extension("onnx");
    let training_data_path = unique_temp_path("training_data");
    let generator = GameGenerator::new(
        bootstrap_path.to_string_lossy().to_string(),
        training_data_path.to_string_lossy().to_string(),
        None,
        100,
        true,
        16,
        1.0,
        1,
        0,
        Some(vec![0]),
        true,
        10,
        0.0,
        true,
        true,
    )
    .expect("generator should construct");

    let pending_path = unique_temp_path("pending").with_extension("onnx");
    for artifact in GameGenerator::model_artifact_paths(&pending_path) {
        fs::write(&artifact, b"pending").expect("pending artifact write should succeed");
    }
    let evaluating_path = unique_temp_path("evaluating").with_extension("onnx");
    for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
        fs::write(&artifact, b"evaluating").expect("evaluating artifact write should succeed");
    }

    {
        let mut pending = generator.pending_candidate.write().unwrap();
        *pending = Some(CandidateModelRequest {
            model_path: pending_path.clone(),
            model_step: 1,
            nn_value_weight: 0.1,
            death_penalty: 5.0,
            overhang_penalty_weight: 3.0,
            force_promote: false,
        });
    }
    {
        let mut evaluating = generator.evaluating_candidate.write().unwrap();
        *evaluating = Some(CandidateModelRequest {
            model_path: evaluating_path.clone(),
            model_step: 2,
            nn_value_weight: 0.2,
            death_penalty: 5.0,
            overhang_penalty_weight: 3.0,
            force_promote: false,
        });
    }

    generator.cleanup_queued_candidate_artifacts();

    for artifact in GameGenerator::model_artifact_paths(&pending_path) {
        assert!(!artifact.exists(), "pending artifacts should be removed");
    }
    for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
        assert!(!artifact.exists(), "evaluating artifacts should be removed");
    }
    assert!(generator.pending_candidate.read().unwrap().is_none());
    assert!(generator.evaluating_candidate.read().unwrap().is_none());
}

#[test]
fn test_sync_model_directly_updates_incumbent_when_candidate_gating_disabled() {
    let bootstrap_path = unique_temp_path("direct_sync_bootstrap").with_extension("onnx");
    fs::write(&bootstrap_path, b"bootstrap").expect("bootstrap write should succeed");
    let training_data_path = unique_temp_path("direct_sync_training_data");

    let mut config = MCTSConfig::default();
    config.death_penalty = 5.0;
    config.overhang_penalty_weight = 3.0;

    let generator = GameGenerator::new(
        bootstrap_path.to_string_lossy().to_string(),
        training_data_path.to_string_lossy().to_string(),
        Some(config),
        100,
        true,
        16,
        1.0,
        3,
        0,
        Some(vec![]),
        false,
        10,
        0.0,
        false,
        false,
    )
    .expect("generator should construct");

    assert!(!generator.candidate_gate_busy());
    assert!(!generator.incumbent_uses_network());

    let synced_path = unique_temp_path("direct_sync_model").with_extension("onnx");
    fs::write(&synced_path, b"synced").expect("synced model write should succeed");
    let synced = generator
        .sync_model_directly(synced_path.to_string_lossy().to_string(), 5, 0.75, 5.0, 3.0)
        .expect("direct sync should succeed");

    assert!(synced);
    assert_eq!(generator.incumbent_model_step(), 5);
    assert!(generator.incumbent_uses_network());
    assert_eq!(
        generator.incumbent_model_path(),
        synced_path.to_string_lossy().to_string()
    );
    assert_eq!(generator.incumbent_nn_value_weight(), 0.75);
    assert_eq!(generator.incumbent_death_penalty(), 5.0);
    assert_eq!(generator.incumbent_overhang_penalty_weight(), 3.0);
    assert_eq!(generator.incumbent_eval_avg_attack(), 0.0);
    assert!(!generator.candidate_gate_busy());

    // Direct sync now stores the penalties verbatim — Rust no longer zeros
    // them when nn_value_weight crosses any internal cap. Trainer is the
    // sole authority for the schedule.
    let above_cap_path = unique_temp_path("direct_sync_above_cap").with_extension("onnx");
    fs::write(&above_cap_path, b"above-cap").expect("above-cap model write should succeed");
    let above_cap_synced = generator
        .sync_model_directly(
            above_cap_path.to_string_lossy().to_string(),
            6,
            1.5,
            7.0,
            2.0,
        )
        .expect("above-cap direct sync should succeed");
    assert!(above_cap_synced);
    assert_eq!(generator.incumbent_nn_value_weight(), 1.5);
    assert_eq!(generator.incumbent_death_penalty(), 7.0);
    assert_eq!(generator.incumbent_overhang_penalty_weight(), 2.0);

    let stale_path = unique_temp_path("direct_sync_stale").with_extension("onnx");
    fs::write(&stale_path, b"stale").expect("stale model write should succeed");
    let stale_synced = generator
        .sync_model_directly(stale_path.to_string_lossy().to_string(), 4, 0.8, 5.0, 3.0)
        .expect("stale direct sync should return cleanly");

    assert!(!stale_synced);
    assert!(
        !stale_path.exists(),
        "stale direct sync artifacts should be removed"
    );
}

#[test]
fn test_update_search_overrides_flips_live_atomics() {
    let bootstrap_path = unique_temp_path("live_overrides_bootstrap").with_extension("onnx");
    let training_data_path = unique_temp_path("live_overrides_training_data");

    let mut config = MCTSConfig::default();
    config.visit_sampling_epsilon = 0.4;

    let generator = GameGenerator::new(
        bootstrap_path.to_string_lossy().to_string(),
        training_data_path.to_string_lossy().to_string(),
        Some(config),
        100,
        true, // initial add_noise = true (mirrors trainer default)
        16,
        1.0,
        1,
        0,
        Some(vec![0]),
        true,
        10,
        0.0,
        true,
        false,
    )
    .expect("generator should construct");

    assert!(generator.live_add_noise());
    assert_eq!(generator.live_visit_sampling_epsilon(), 0.4);

    generator
        .update_search_overrides(Some(false), Some(0.0))
        .expect("update should succeed");
    assert!(!generator.live_add_noise());
    assert_eq!(generator.live_visit_sampling_epsilon(), 0.0);

    // None args leave the corresponding setting unchanged.
    generator
        .update_search_overrides(None, Some(0.25))
        .expect("partial update should succeed");
    assert!(!generator.live_add_noise());
    assert_eq!(generator.live_visit_sampling_epsilon(), 0.25);

    // Out-of-range epsilon must be rejected and leave state untouched.
    // PyValueError construction needs the Python interpreter; prepare it
    // here so this lib-test can exercise the error path.
    pyo3::prepare_freethreaded_python();
    let err = generator
        .update_search_overrides(Some(true), Some(1.5))
        .expect_err("epsilon > 1 should error");
    assert!(err.to_string().contains("visit_sampling_epsilon"));
    assert!(!generator.live_add_noise());
    assert_eq!(generator.live_visit_sampling_epsilon(), 0.25);
}
