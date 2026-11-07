use super::*;
use proptest::prelude::*;

#[test]
fn test_board_encoding_is_binary() {
    let mut env = TetrisEnv::new(10, 20);

    // Place some pieces to create a non-empty board.
    env.hard_drop();
    env.hard_drop();

    let (board_tensor, _) = encode_state(&env, 100).expect("encoding failed");
    let board_array = board_tensor
        .to_array_view::<f32>()
        .expect("board tensor should contain f32");
    let board_values: Vec<f32> = board_array.iter().copied().collect();

    assert_eq!(board_values.len(), BOARD_HEIGHT * BOARD_WIDTH);

    for &val in &board_values {
        assert!(
            val == 0.0 || val == 1.0,
            "Board tensor should be binary, got value: {}",
            val
        );
    }

    let board = env.board_cells();
    for y in 0..BOARD_HEIGHT {
        for x in 0..BOARD_WIDTH {
            let idx = y * BOARD_WIDTH + x;
            let expected = if board[idx] != 0 { 1.0 } else { 0.0 };
            let actual = board_values[idx];
            assert_eq!(
                actual, expected,
                "Board[{},{}] with value {} should encode to {}, got {}",
                y, x, board[idx], expected, actual
            );
        }
    }
}

#[test]
fn test_auxiliary_features_format() {
    let mut env = TetrisEnv::new(10, 20);
    env.placement_count = 42;
    let (_, aux) = encode_state(&env, 100).expect("encoding failed");
    let aux_array = aux
        .to_array_view::<f32>()
        .expect("aux tensor should contain f32");
    let aux: Vec<f32> = aux_array.iter().copied().collect();

    assert_eq!(aux.len(), AUX_FEATURES);
    assert_eq!(aux.len(), 80);

    let mut idx = 0;

    let current_piece = require_current_piece_type(&env);
    let current_onehot = &aux[idx..idx + NUM_PIECE_TYPES];
    let sum: f32 = current_onehot.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Current piece should be one-hot, sum = {}",
        sum
    );
    assert_eq!(
        current_onehot[current_piece], 1.0,
        "Current piece not encoded correctly"
    );
    idx += NUM_PIECE_TYPES;

    let hold_piece = env.get_hold_piece();
    let hold_onehot = &aux[idx..idx + NUM_PIECE_TYPES + 1];
    let sum: f32 = hold_onehot.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Hold piece should be one-hot, sum = {}",
        sum
    );
    if let Some(piece) = hold_piece {
        assert_eq!(
            hold_onehot[piece.piece_type], 1.0,
            "Hold piece not encoded correctly"
        );
    } else {
        assert_eq!(
            hold_onehot[NUM_PIECE_TYPES], 1.0,
            "Empty hold should set index 7"
        );
    }
    idx += NUM_PIECE_TYPES + 1;

    let hold_avail = aux[idx];
    let expected_hold = if !env.is_hold_used() { 1.0 } else { 0.0 };
    assert_eq!(hold_avail, expected_hold, "Hold available incorrect");
    idx += 1;

    let queue = env.get_queue(QUEUE_SIZE);
    for i in 0..QUEUE_SIZE {
        let queue_slot = &aux[idx..idx + NUM_PIECE_TYPES];
        let sum: f32 = queue_slot.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Queue slot {} should be one-hot, sum = {}",
            i,
            sum
        );
        if i < queue.len() {
            assert_eq!(
                queue_slot[queue[i]], 1.0,
                "Queue slot {} not encoded correctly",
                i
            );
        }
        idx += NUM_PIECE_TYPES;
    }

    let move_norm = aux[idx];
    let expected_norm = 42.0 / 100.0;
    assert!(
        (move_norm - expected_norm).abs() < 1e-6,
        "Placement count feature should be {}, got {}",
        expected_norm,
        move_norm
    );
    idx += 1;

    let combo_norm = aux[idx];
    let expected_combo_norm = normalize_combo_for_feature(env.combo);
    assert!(
        (combo_norm - expected_combo_norm).abs() < 1e-6,
        "Combo feature should be {}, got {}",
        expected_combo_norm,
        combo_norm
    );
    idx += 1;

    let back_to_back = aux[idx];
    let expected_back_to_back = if env.back_to_back { 1.0 } else { 0.0 };
    assert_eq!(
        back_to_back, expected_back_to_back,
        "Back-to-back feature incorrect"
    );
    idx += 1;

    let hidden_distribution = &aux[idx..idx + NUM_PIECE_TYPES];
    let expected_hidden_distribution = next_hidden_piece_distribution(&env);
    let hidden_sum: f32 = hidden_distribution.iter().sum();
    assert!(
        (hidden_sum - 1.0).abs() < 1e-6,
        "Hidden-piece distribution should sum to 1.0, got {}",
        hidden_sum
    );
    for piece in 0..NUM_PIECE_TYPES {
        assert!(
            (hidden_distribution[piece] - expected_hidden_distribution[piece]).abs() < 1e-6,
            "Hidden-piece probability mismatch for piece {}: expected {}, got {}",
            piece,
            expected_hidden_distribution[piece],
            hidden_distribution[piece]
        );
    }
    idx += NUM_PIECE_TYPES;

    let expected_column_heights = normalize_column_heights(&env.column_heights);
    let encoded_column_heights = &aux[idx..idx + BOARD_WIDTH];
    for col in 0..BOARD_WIDTH {
        assert!(
            (encoded_column_heights[col] - expected_column_heights[col]).abs() < 1e-6,
            "Column height mismatch at col {}: expected {}, got {}",
            col,
            expected_column_heights[col],
            encoded_column_heights[col]
        );
    }
    idx += BOARD_WIDTH;

    let raw_max = require_max_u8(&env.column_heights, "column_heights");
    let expected_max_column_height = normalize_max_column_height(raw_max);
    assert!(
        (aux[idx] - expected_max_column_height).abs() < 1e-6,
        "Max column height should be {}, got {}",
        expected_max_column_height,
        aux[idx]
    );
    idx += 1;

    let expected_row_fill_counts = normalize_row_fill_counts(&env.row_fill_counts, env.width);
    let encoded_row_fill_counts = &aux[idx..idx + ROW_FILL_FEATURE_ROWS];
    for row in 0..ROW_FILL_FEATURE_ROWS {
        assert!(
            (encoded_row_fill_counts[row] - expected_row_fill_counts[row]).abs() < 1e-6,
            "Row fill count mismatch at row {}: expected {}, got {}",
            row,
            expected_row_fill_counts[row],
            encoded_row_fill_counts[row]
        );
    }
    idx += ROW_FILL_FEATURE_ROWS;

    let expected_total_blocks = normalize_total_blocks(env.total_blocks);
    assert!(
        (aux[idx] - expected_total_blocks).abs() < 1e-6,
        "Total blocks should be {}, got {}",
        expected_total_blocks,
        aux[idx]
    );
    idx += 1;

    let expected_bumpiness = normalize_bumpiness(compute_bumpiness(&env.column_heights));
    assert!(
        (aux[idx] - expected_bumpiness).abs() < 1e-6,
        "Bumpiness should be {}, got {}",
        expected_bumpiness,
        aux[idx]
    );
    idx += 1;

    let (expected_overhang_raw, expected_holes_raw) = count_overhang_fields_and_holes(&env);
    let expected_holes = normalize_holes(expected_holes_raw);
    assert!(
        (aux[idx] - expected_holes).abs() < 1e-6,
        "Holes should be {}, got {}",
        expected_holes,
        aux[idx]
    );
    idx += 1;

    let expected_overhang = normalize_overhang_fields(expected_overhang_raw);
    assert!(
        (aux[idx] - expected_overhang).abs() < 1e-6,
        "Overhang fields should be {}, got {}",
        expected_overhang,
        aux[idx]
    );
    idx += 1;

    assert_eq!(
        idx, AUX_FEATURES,
        "Should consume all {} aux features",
        AUX_FEATURES
    );
}

#[test]
fn test_encoding_specification() {
    let mut env = TetrisEnv::new(10, 20);
    env.placement_count = 50;
    let (board, aux) = encode_state(&env, 100).expect("encoding failed");
    let board_array = board
        .to_array_view::<f32>()
        .expect("board tensor should contain f32");
    let aux_array = aux
        .to_array_view::<f32>()
        .expect("aux tensor should contain f32");
    let board: Vec<f32> = board_array.iter().copied().collect();
    let aux: Vec<f32> = aux_array.iter().copied().collect();

    assert_eq!(board.len(), 20 * 10, "Board should be 20x10 = 200 values");
    assert_eq!(
        aux.len(),
        7 + 8 + 1 + 35 + 1 + 1 + 1 + 7 + 10 + 1 + 4 + 1 + 1 + 1 + 1,
        "Aux should be 80 values"
    );

    for &val in &board {
        assert!(val == 0.0 || val == 1.0, "Board must be binary");
    }

    let current_sum: f32 = aux[0..7].iter().sum();
    assert!(
        (current_sum - 1.0).abs() < 1e-6,
        "Current piece must be one-hot"
    );

    let hold_sum: f32 = aux[7..15].iter().sum();
    assert!(
        (hold_sum - 1.0).abs() < 1e-6,
        "Hold piece must be one-hot (8 values)"
    );

    let hold_avail = aux[15];
    assert!(
        hold_avail == 0.0 || hold_avail == 1.0,
        "Hold available must be binary"
    );

    for i in 0..5 {
        let start = 16 + i * 7;
        let queue_sum: f32 = aux[start..start + 7].iter().sum();
        assert!(
            (queue_sum - 1.0).abs() < 1e-6,
            "Queue slot {} must be one-hot",
            i
        );
    }

    let move_norm = aux[51];
    assert!(
        move_norm >= 0.0 && move_norm <= 1.0,
        "Placement count must be in [0, 1]"
    );
    assert!(
        (move_norm - 0.5).abs() < 1e-6,
        "Placement count 50 should normalize to 0.5"
    );

    let combo_norm = aux[52];
    assert!(combo_norm >= 0.0, "Combo feature must be non-negative");
    assert_eq!(combo_norm, 0.0, "Initial combo should be 0.0");

    let back_to_back = aux[53];
    assert!(
        back_to_back == 0.0 || back_to_back == 1.0,
        "Back-to-back must be binary"
    );
    assert_eq!(back_to_back, 0.0, "Initial back-to-back should be 0.0");

    let hidden_distribution = &aux[54..61];
    let hidden_sum: f32 = hidden_distribution.iter().sum();
    assert!(
        (hidden_sum - 1.0).abs() < 1e-6,
        "Hidden-piece distribution must sum to 1.0"
    );
    for &p in hidden_distribution {
        assert!(
            (0.0..=1.0).contains(&p),
            "Hidden-piece probabilities must be in [0, 1]"
        );
    }

    let column_heights = &aux[61..71];
    for &height in column_heights {
        assert!(
            height >= 0.0,
            "Normalized column heights must be non-negative"
        );
    }

    let max_column_height = aux[71];
    assert!(
        (0.0..=1.0).contains(&max_column_height),
        "Max column height must be in [0, 1]"
    );

    let row_fill_counts = &aux[72..72 + ROW_FILL_FEATURE_ROWS];
    for &row_fill in row_fill_counts {
        assert!(
            (0.0..=1.0).contains(&row_fill),
            "Normalized row fill counts must be in [0, 1]"
        );
    }

    let total_blocks = aux[72 + ROW_FILL_FEATURE_ROWS];
    let bumpiness = aux[73 + ROW_FILL_FEATURE_ROWS];
    let holes = aux[74 + ROW_FILL_FEATURE_ROWS];
    let overhang = aux[75 + ROW_FILL_FEATURE_ROWS];
    assert!(total_blocks >= 0.0, "Total blocks must be non-negative");
    assert!(bumpiness >= 0.0, "Bumpiness must be non-negative");
    assert!(holes >= 0.0, "Holes must be non-negative");
    assert!(overhang >= 0.0, "Overhang fields must be non-negative");
}

#[test]
fn test_hidden_distribution_changes_with_visible_queue_horizon() {
    let mut env = TetrisEnv::new(10, 20);

    let first = next_hidden_piece_distribution(&env);
    let first_non_zero = first.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(
        first_non_zero, 1,
        "At game start, hidden-piece distribution should be deterministic for the next hidden piece"
    );

    env.hard_drop();
    let second = next_hidden_piece_distribution(&env);
    let second_non_zero = second.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(
        second_non_zero, NUM_PIECE_TYPES,
        "After one placement, hidden-piece distribution should reset to full-bag uncertainty"
    );
}

proptest! {
    #[test]
    fn prop_encoded_diagnostics_match_env_state(
        seed in 0u64..10_000,
        actions in prop::collection::vec(0u8..8, 0..80),
        max_placements in 1usize..200,
    ) {
        let mut env = TetrisEnv::with_seed(10, 20, seed);

        for (step_idx, action) in actions.iter().copied().enumerate() {
            env.placement_count = (step_idx % max_placements) as u32;
            let (_, aux_tensor) =
                encode_state(&env, max_placements)
                    .expect("encoding should succeed for valid max_placements");
            let aux_array = aux_tensor
                .to_array_view::<f32>()
                .expect("aux tensor should contain f32");
            let aux: Vec<f32> = aux_array.iter().copied().collect();
            prop_assert_eq!(aux.len(), AUX_FEATURES);

            let diagnostics_start = NUM_PIECE_TYPES
                + (NUM_PIECE_TYPES + 1)
                + 1
                + (QUEUE_SIZE * NUM_PIECE_TYPES)
                + 1
                + 1
                + 1
                + NUM_PIECE_TYPES;

            let mut idx = diagnostics_start;

            let expected_column_heights = normalize_column_heights(&env.column_heights);
            for expected in expected_column_heights.iter().take(BOARD_WIDTH) {
                prop_assert!((aux[idx] - *expected).abs() < 1e-6);
                idx += 1;
            }

            let raw_max = require_max_u8(&env.column_heights, "column_heights");
            let expected_max_column_height = normalize_max_column_height(raw_max);
            prop_assert!((aux[idx] - expected_max_column_height).abs() < 1e-6);
            idx += 1;

            let expected_row_fill_counts = normalize_row_fill_counts(&env.row_fill_counts, env.width);
            for expected in expected_row_fill_counts.iter().take(ROW_FILL_FEATURE_ROWS) {
                prop_assert!((aux[idx] - *expected).abs() < 1e-6);
                prop_assert!((0.0..=1.0).contains(&aux[idx]));
                idx += 1;
            }

            let expected_total_blocks = normalize_total_blocks(env.total_blocks);
            prop_assert!((aux[idx] - expected_total_blocks).abs() < 1e-6);
            idx += 1;

            let expected_raw_bumpiness = compute_bumpiness(&env.column_heights);
            let expected_bumpiness = normalize_bumpiness(expected_raw_bumpiness);
            prop_assert!((aux[idx] - expected_bumpiness).abs() < 1e-6);
            idx += 1;

            let (expected_raw_overhang, expected_raw_holes) =
                count_overhang_fields_and_holes(&env);
            let expected_holes = normalize_holes(expected_raw_holes);
            prop_assert!((aux[idx] - expected_holes).abs() < 1e-6);
            idx += 1;

            let expected_overhang = normalize_overhang_fields(expected_raw_overhang);
            prop_assert!((aux[idx] - expected_overhang).abs() < 1e-6);
            idx += 1;

            prop_assert_eq!(idx, AUX_FEATURES);

            if env.game_over {
                break;
            }
            env.step(action);
        }
    }
}

#[test]
fn test_masked_softmax() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![true, false, true, false];
    let probs = masked_softmax(&logits, &mask);

    assert!((probs[0] + probs[2] - 1.0).abs() < 1e-5);
    assert_eq!(probs[1], 0.0);
    assert_eq!(probs[3], 0.0);
}

#[test]
fn test_combo_feature_round_trip() {
    assert_eq!(denormalize_combo_feature(normalize_combo_for_feature(3)), 3);
    assert_eq!(
        denormalize_combo_feature(normalize_combo_for_feature(99)),
        99
    );
}

#[test]
#[should_panic(expected = "combo_feature must be >= 0.0")]
fn test_combo_feature_denormalize_rejects_negative_values() {
    let _ = denormalize_combo_feature(-0.25);
}

#[test]
fn test_pack_board_empty() {
    let env = TetrisEnv::new(10, 20);
    let key = pack_board(&env);
    assert_eq!(key, [0u64; 4], "Empty board should pack to all zeros");
}

#[test]
fn test_pack_board_deterministic() {
    let mut env = TetrisEnv::new(10, 20);
    env.hard_drop();
    let key1 = pack_board(&env);
    let key2 = pack_board(&env);
    assert_eq!(key1, key2, "Same board should produce same key");
}

#[test]
fn test_pack_board_different_boards() {
    let mut env1 = TetrisEnv::new(10, 20);
    env1.hard_drop();
    let key1 = pack_board(&env1);

    let mut env2 = TetrisEnv::new(10, 20);
    env2.hard_drop();
    env2.hard_drop();
    let key2 = pack_board(&env2);

    assert_ne!(key1, key2, "Different boards should produce different keys");
}

#[test]
fn test_load_and_predict_split_model() {
    let model_path = "/tmp/tetris_split_test/test.onnx";
    if !std::path::Path::new("/tmp/tetris_split_test/test.conv.onnx").exists() {
        eprintln!("Skipping test - split model files not found (run Python export first)");
        return;
    }

    let nn = TetrisNN::load(model_path).expect("Failed to load split model");
    let env = TetrisEnv::new(10, 20);
    let mask = get_action_mask(&env);

    let (policy1, value1) = nn
        .predict_masked(&env, &mask, 100, PredictionNoiseSettings::default())
        .expect("First inference failed");
    assert_eq!(policy1.len(), mask.len());
    let policy_sum: f32 = policy1.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "Policy should sum to ~1.0, got {}",
        policy_sum
    );

    let (policy2, value2) = nn
        .predict_masked(&env, &mask, 100, PredictionNoiseSettings::default())
        .expect("Second inference (cache hit) failed");
    assert_eq!(policy1, policy2, "Cache hit should produce same policy");
    assert_eq!(value1, value2, "Cache hit should produce same value");

    let mut env_with_late_count = env.clone();
    env_with_late_count.placement_count = 50;
    let (_policy3, value3) = nn
        .predict_masked(
            &env_with_late_count,
            &mask,
            100,
            PredictionNoiseSettings::default(),
        )
        .expect("Third inference (different aux) failed");
    assert_ne!(
        value1, value3,
        "Different placement counts should produce different values"
    );

    assert_eq!(
        nn.board_cache.borrow().len(),
        1,
        "Cache should have 1 entry for the single board state"
    );
}

#[test]
fn test_predict_with_valid_actions_matches_masked_policy_subset() {
    let model_path = "/tmp/tetris_split_test/test.onnx";
    if !std::path::Path::new("/tmp/tetris_split_test/test.conv.onnx").exists() {
        eprintln!("Skipping test - split model files not found (run Python export first)");
        return;
    }

    let nn = TetrisNN::load(model_path).expect("Failed to load split model");
    let env = TetrisEnv::new(10, 20);
    let mask = get_action_mask(&env);
    let valid_actions: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(action_idx, &valid)| valid.then_some(action_idx))
        .collect();

    let (masked_policy, masked_value) = nn
        .predict_masked(&env, &mask, 100, PredictionNoiseSettings::default())
        .expect("Masked inference failed");
    let (valid_policy, valid_value) = nn
        .predict_with_valid_actions(
            &env,
            &valid_actions,
            100,
            PredictionNoiseSettings::default(),
        )
        .expect("Valid-actions inference failed");

    let expected_valid_policy: Vec<f32> = valid_actions
        .iter()
        .map(|&action_idx| masked_policy[action_idx])
        .collect();
    assert_eq!(
        valid_policy, expected_valid_policy,
        "Valid-action priors should match the masked policy restricted to valid actions"
    );
    assert_eq!(
        valid_value, masked_value,
        "Valid-actions and masked paths should return the same value prediction"
    );
}

#[test]
fn test_prediction_noise_is_deterministic_for_fixed_seed_and_state() {
    let mut policy_logits = vec![0.1, -0.3, 1.2, 0.0];
    let mut first_value = 2.5;
    let mut second_logits = policy_logits.clone();
    let mut second_value = first_value;
    let noise = PredictionNoiseSettings {
        policy_mean: 0.0,
        policy_std: 0.25,
        value_mean: 0.5,
        value_std: 0.1,
        seed: Some(1234),
    };

    apply_prediction_noise(&mut policy_logits, &mut first_value, noise, 77)
        .expect("noise application should succeed");
    apply_prediction_noise(&mut second_logits, &mut second_value, noise, 77)
        .expect("noise application should succeed");

    assert_eq!(policy_logits, second_logits);
    assert_eq!(first_value, second_value);
}

#[test]
fn test_prediction_noise_changes_with_state_hash() {
    let noise = PredictionNoiseSettings {
        policy_mean: 0.0,
        policy_std: 0.5,
        value_mean: 0.0,
        value_std: 0.25,
        seed: Some(7),
    };
    let mut first_logits = vec![0.0, 0.0, 0.0];
    let mut first_value = 0.0;
    let mut second_logits = vec![0.0, 0.0, 0.0];
    let mut second_value = 0.0;

    apply_prediction_noise(&mut first_logits, &mut first_value, noise, 1)
        .expect("noise application should succeed");
    apply_prediction_noise(&mut second_logits, &mut second_value, noise, 2)
        .expect("noise application should succeed");

    assert_ne!(first_logits, second_logits);
    assert_ne!(first_value, second_value);
}

#[test]
fn test_policy_mean_only_does_not_change_masked_probabilities() {
    let logits = vec![1.0, 3.0, -2.0, 0.5];
    let mask = vec![true, false, true, true];
    let baseline = masked_softmax(&logits, &mask);

    let mut shifted_logits = logits.clone();
    let mut value = 0.0;
    apply_prediction_noise(
        &mut shifted_logits,
        &mut value,
        PredictionNoiseSettings {
            policy_mean: 2.0,
            policy_std: 0.0,
            value_mean: 0.0,
            value_std: 0.0,
            seed: Some(5),
        },
        11,
    )
    .expect("noise application should succeed");

    let shifted = masked_softmax(&shifted_logits, &mask);
    assert_eq!(baseline, shifted);
}

/// Latency probe for the heads path on real states (`TETRIS_BENCH_MODEL=<bundle>.onnx`).
#[test]
#[ignore]
fn bench_heads_latency_probe() {
    let Ok(model_path) = std::env::var("TETRIS_BENCH_MODEL") else {
        eprintln!("TETRIS_BENCH_MODEL not set");
        return;
    };
    let nn = TetrisNN::load(&model_path).expect("load");
    let mut envs = Vec::new();
    for seed in 0..8u64 {
        let mut env = TetrisEnv::with_seed(10, 20, 1000 + seed);
        for step in 0..30usize {
            if env.game_over {
                break;
            }
            envs.push(env.clone());
            let actions = env.get_cached_valid_action_indices_arc();
            let action = actions[(step * 7 + seed as usize) % actions.len()];
            env.execute_action_index(action);
        }
    }
    let inputs: Vec<(Vec<f32>, Vec<usize>)> = envs
        .iter()
        .map(|env| {
            (
                encode_piece_aux_features(env, 50).unwrap(),
                env.get_cached_valid_action_indices_arc().to_vec(),
            )
        })
        .collect();
    let avg_valid: f64 = inputs.iter().map(|x| x.1.len() as f64).sum::<f64>() / inputs.len() as f64;
    for round in 0..3 {
        let iters = 50_000;
        let start = std::time::Instant::now();
        let mut acc = 0.0f32;
        let mut logits = vec![0.0f32; 64];
        let mut scratch = nn.heads_scratch.borrow_mut();
        for i in 0..iters {
            let (aux, valid) = &inputs[i % inputs.len()];
            let value = nn
                .run_heads_trunk(&envs[i % envs.len()], aux, &mut scratch)
                .unwrap();
            nn.heads
                .policy_logits(&scratch, valid, &mut logits[..valid.len()]);
            acc += value + logits[0];
        }
        drop(scratch);
        let heads_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let start = std::time::Instant::now();
        for i in 0..iters {
            let env = &envs[i % envs.len()];
            let (policy, value) = nn
                .predict_with_valid_actions(
                    env,
                    &inputs[i % inputs.len()].1,
                    50,
                    PredictionNoiseSettings::default(),
                )
                .unwrap();
            acc += value + policy[0];
        }
        let predict_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!(
            "round {round}: states={} avg_valid={avg_valid:.1} heads_us={heads_us:.2} predict_valid_hot_us={predict_us:.2} (acc {acc})",
            inputs.len()
        );
    }
}

/// Native heads vs an independent f64 forward pass on real states
/// (`TETRIS_BENCH_MODEL=<bundle>.onnx`).
#[test]
#[ignore]
fn native_heads_match_f64_reference_on_real_states() {
    let Ok(model_path) = std::env::var("TETRIS_BENCH_MODEL") else {
        eprintln!("TETRIS_BENCH_MODEL not set");
        return;
    };
    let nn = TetrisNN::load(&model_path).expect("load");
    let heads_path = Path::new(&model_path).with_extension("heads.onnx");
    let (weights, aux_layers, blocks) = heads::reference::load_weights(&heads_path);
    let mut scratch = nn.heads.new_scratch();
    let mut board_input = vec![0.0; nn.heads.board_fusion_width()];
    let (mut max_value_rel, mut max_logit_abs, mut max_prob_abs) = (0.0f64, 0.0f64, 0.0f64);
    let mut states = 0;
    for seed in 0..40u64 {
        let mut env = TetrisEnv::with_seed(10, 20, 3000 + seed);
        for step in 0..50usize {
            if env.game_over {
                break;
            }
            let valid = env.get_cached_valid_action_indices_arc().to_vec();
            if valid.is_empty() {
                break;
            }
            let board_h = nn
                .compute_board_embedding(&encode_board_features(&env), &encode_board_stats(&env))
                .unwrap();
            let board_h = board_h.as_slice().unwrap();
            let aux = encode_piece_aux_features(&env, 50).unwrap();
            nn.heads
                .board_fusion_input(board_h, &mut scratch, &mut board_input);
            let value = nn.heads.forward_trunk(&board_input, &aux, &mut scratch);
            let logits = nn.heads.all_policy_logits(&scratch);
            let (ref_logits, ref_value) =
                heads::reference::forward(&weights, aux_layers, blocks, board_h, &aux);

            max_value_rel =
                max_value_rel.max((f64::from(value) - ref_value).abs() / (1.0 + ref_value.abs()));
            for (&got, &want) in logits.iter().zip(&ref_logits) {
                max_logit_abs = max_logit_abs.max((f64::from(got) - want).abs());
            }
            let ref_max = valid
                .iter()
                .map(|&a| ref_logits[a])
                .fold(f64::MIN, f64::max);
            let ref_sum: f64 = valid.iter().map(|&a| (ref_logits[a] - ref_max).exp()).sum();
            let probs = softmax_over_valid_actions(&logits, &valid);
            for (&a, &p) in valid.iter().zip(&probs) {
                let want = (ref_logits[a] - ref_max).exp() / ref_sum;
                max_prob_abs = max_prob_abs.max((f64::from(p) - want).abs());
            }
            states += 1;
            env.execute_action_index(valid[(step * 7 + seed as usize) % valid.len()]);
        }
    }
    eprintln!(
        "native vs f64 reference over {states} states: value max rel {max_value_rel:.2e}, logit max abs {max_logit_abs:.2e}, prob max abs {max_prob_abs:.2e}"
    );
    assert!(max_value_rel < 1e-4 && max_logit_abs < 1e-3 && max_prob_abs < 1e-4);
}
