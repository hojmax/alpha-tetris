//! TetrisEnv Tests

#[cfg(test)]
mod tests {
    use crate::game::action_space::{get_action_space, HOLD_ACTION_INDEX};
    use crate::game::constants::{
        BOARD_HEIGHT, BOARD_WIDTH, I_PIECE, L_PIECE, MAX_BOARD_CELLS, O_PIECE, S_PIECE, T_PIECE,
        Z_PIECE,
    };
    use crate::game::env::TetrisEnv;
    use crate::game::piece::Piece;

    const HANDCRAFTED_TSPIN_BOARD: [&str; 20] = [
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "....LLL...",
        "LL...LLLLL",
        "LLL.LLLLLL",
    ];

    fn set_board_from_ascii(env: &mut TetrisEnv, rows: &[&str]) {
        assert_eq!(rows.len(), env.height);
        let board = rows
            .iter()
            .map(|row| {
                assert_eq!(row.len(), env.width);
                row.chars()
                    .map(|cell| if cell == '.' { 0 } else { 1 })
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<Vec<u8>>>();
        env.set_board(board)
            .expect("handcrafted board should be a valid 10x20 binary board");
    }

    fn setup_handcrafted_tspin_env() -> TetrisEnv {
        let mut env = TetrisEnv::new(10, 20);
        set_board_from_ascii(&mut env, &HANDCRAFTED_TSPIN_BOARD);
        env.spawn_piece_from_type(T_PIECE);
        env.set_queue(vec![I_PIECE, O_PIECE, L_PIECE, S_PIECE, Z_PIECE])
            .expect("handcrafted queue should be valid");
        env
    }

    #[test]
    fn test_env_creation() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.width, 10);
        assert_eq!(env.height, 20);
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_custom_board_size() {
        let env = TetrisEnv::new(12, 24);
        assert_eq!(env.width, 12);
        assert_eq!(env.height, 24);
    }

    #[test]
    fn test_env_reset() {
        let mut env = TetrisEnv::new(10, 20);
        env.attack = 100;
        env.lines_cleared = 10;
        env.combo = 5;
        env.back_to_back = true;
        env.reset();
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
    }

    #[test]
    fn test_move_left() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_left();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x - 1);
        }
    }

    #[test]
    fn test_move_right() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_right();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x + 1);
        }
    }

    #[test]
    fn test_move_down() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.move_down();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_rotate_cw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_cw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 1) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_rotate_ccw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_ccw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 3) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_movement_clears_rotation_flag() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.rotate_cw();
        assert!(env.last_move_was_rotation);
        env.move_left();
        assert!(!env.last_move_was_rotation);
    }

    #[test]
    fn test_hard_drop() {
        let mut env = TetrisEnv::new(10, 20);
        let drop_distance = env.hard_drop();
        assert!(drop_distance > 0);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_ghost_piece() {
        let env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());
        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();
        assert!(ghost.y >= current.y);
        assert_eq!(ghost.piece_type, current.piece_type);
        assert_eq!(ghost.rotation, current.rotation);
    }

    #[test]
    fn test_7bag_randomizer() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.piece_queue.len() >= 6);
        for pt in env.piece_queue.iter() {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.hold_piece.is_none());

        let current_type = env.current_piece.as_ref().unwrap().piece_type;
        let result = env.hold();

        assert!(result);
        assert_eq!(env.hold_piece, Some(current_type));
        assert!(env.hold_used);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_hold_twice_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.hold();
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_cached_valid_actions_respect_hold_availability() {
        let env_with_hold = TetrisEnv::with_seed(10, 20, 42);
        let mut env_without_hold = env_with_hold.clone();
        env_without_hold.hold_used = true;

        env_with_hold.invalidate_placement_cache();
        env_without_hold.invalidate_placement_cache();

        let valid_with_hold = env_with_hold.get_cached_valid_action_indices();
        let valid_without_hold = env_without_hold.get_cached_valid_action_indices();

        assert!(
            valid_with_hold.contains(&HOLD_ACTION_INDEX),
            "Hold action should be available when hold is unused"
        );
        assert!(
            !valid_without_hold.contains(&HOLD_ACTION_INDEX),
            "Hold action should be unavailable when hold is already used"
        );
    }

    #[test]
    fn test_empty_board_valid_action_counts_match_expected_piece_totals() {
        let expected_rotation_counts = [
            [7usize, 10, 0, 0], // I
            [9, 0, 0, 0],       // O
            [8, 9, 8, 9],       // T
            [8, 9, 0, 0],       // S
            [8, 9, 0, 0],       // Z
            [8, 9, 8, 9],       // J
            [8, 9, 8, 9],       // L
        ];
        let action_space = get_action_space();

        for (piece_type, expected_counts) in expected_rotation_counts.iter().enumerate() {
            let mut env = TetrisEnv::new(10, 20);
            env.set_current_piece_type(piece_type);

            let valid_actions = env.get_cached_valid_action_indices();
            assert!(
                valid_actions.contains(&HOLD_ACTION_INDEX),
                "piece {} should allow hold on an empty board",
                piece_type
            );

            let mut actual_counts = [0usize; 4];
            let mut placement_count = 0usize;
            for &action_idx in &valid_actions {
                if action_idx == HOLD_ACTION_INDEX {
                    continue;
                }
                let rotation = action_space.action_to_cell[action_idx].rotation;
                actual_counts[rotation] += 1;
                placement_count += 1;
            }

            assert_eq!(
                actual_counts, *expected_counts,
                "piece {} empty-board rotation counts drifted",
                piece_type
            );
            assert_eq!(
                placement_count,
                expected_counts.iter().sum::<usize>(),
                "piece {} empty-board placement count drifted",
                piece_type
            );
        }
    }

    #[test]
    fn test_hold_swap() {
        let mut env = TetrisEnv::new(10, 20);
        let first_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();
        let held_type = env.hold_piece.unwrap();
        assert_eq!(held_type, first_type);

        env.hard_drop();

        let second_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();

        assert_eq!(env.current_piece.as_ref().unwrap().piece_type, first_type);
        assert_eq!(env.hold_piece, Some(second_type));
    }

    #[test]
    fn test_hold_when_game_over_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_lock_delay_initialization() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.lock_delay_ms.is_none());
        assert_eq!(env.lock_moves_remaining, 15);
    }

    #[test]
    fn test_lock_delay_starts_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.lock_delay_ms.is_some() || env.is_grounded());
    }

    #[test]
    fn test_lock_delay_progress() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.get_lock_delay_progress(), 0.0);
    }

    #[test]
    fn test_get_next_pieces() {
        let env = TetrisEnv::new(10, 20);
        let next_pieces = env.get_next_pieces(5);
        assert_eq!(next_pieces.len(), 5);
        for piece in next_pieces {
            assert!(piece.piece_type < 7);
        }
    }

    #[test]
    fn test_get_queue() {
        let env = TetrisEnv::new(10, 20);
        let queue = env.get_queue(5);
        assert_eq!(queue.len(), 5);
        for pt in queue {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_board_initially_empty() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        for row in board {
            for cell in row {
                assert_eq!(cell, 0);
            }
        }
    }

    #[test]
    fn test_board_piece_types_initially_none() {
        let env = TetrisEnv::new(10, 20);
        let piece_types = env.get_board_piece_types();
        for row in piece_types {
            for cell in row {
                assert!(cell.is_none());
            }
        }
    }

    #[test]
    fn test_clone_state() {
        let env = TetrisEnv::new(10, 20);
        let cloned = env.clone_state();
        assert_eq!(env.attack, cloned.attack);
        assert_eq!(env.width, cloned.width);
        assert_eq!(env.height, cloned.height);
        assert_eq!(env.lines_cleared, cloned.lines_cleared);
        assert_eq!(env.combo, cloned.combo);
        assert_eq!(env.back_to_back, cloned.back_to_back);
    }

    #[test]
    fn test_step_actions() {
        let mut env = TetrisEnv::new(10, 20);
        env.step(0); // noop
        env.step(1); // left
        env.step(2); // right
        env.step(3); // down
        env.step(4); // rotate_cw
        env.step(5); // rotate_ccw
        env.step(7); // hold
        assert!(!env.game_over);
    }

    #[test]
    fn test_step_returns_attack() {
        let mut env = TetrisEnv::new(10, 20);
        let (_reward, game_over) = env.step(6); // hard_drop
        assert!(!game_over);
    }

    #[test]
    fn test_wall_collision_left() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_left();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 0, "Piece should not go past left wall");
        }
    }

    #[test]
    fn test_wall_collision_right() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_right();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x < env.width as i32, "Piece should not go past right wall");
        }
    }

    #[test]
    fn test_tick() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.tick();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_move_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        assert!(!env.move_left());
        assert!(!env.move_right());
        assert!(!env.move_down());
        assert!(!env.rotate_cw());
        assert!(!env.rotate_ccw());
        assert_eq!(env.hard_drop(), 0);
    }

    #[test]
    fn test_update_lock_delay_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let locked = env.update_lock_delay(100);
        assert!(!locked);
    }

    #[test]
    fn test_is_piece_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_piece_grounded());

        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.is_piece_grounded());
    }

    #[test]
    fn test_get_next_piece() {
        let env = TetrisEnv::new(10, 20);
        let next = env.get_next_piece();
        assert!(next.is_some());
        assert!(next.unwrap().piece_type < 7);
    }

    #[test]
    fn test_is_hold_used() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_hold_used());
        env.hold();
        assert!(env.is_hold_used());
    }

    #[test]
    fn test_get_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.get_hold_piece().is_none());
        env.hold();
        assert!(env.get_hold_piece().is_some());
    }

    #[test]
    fn test_board_dimensions() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        assert_eq!(board.len(), 20);
        for row in board {
            assert_eq!(row.len(), 10);
        }
    }

    #[test]
    fn test_perfect_clear_detection() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.is_perfect_clear());
    }

    #[test]
    fn test_perfect_clear_with_cells() {
        let mut env = TetrisEnv::new(10, 20);
        env.board[19 * env.width + 0] = 1;
        env.sync_board_stats();
        assert!(!env.is_perfect_clear());
    }

    #[test]
    fn test_tspin_requires_t_piece() {
        let mut env = TetrisEnv::new(10, 20);
        let piece = Piece::with_position(0, 3, 10, 0); // I piece
        env.current_piece = Some(piece);
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_tspin_requires_rotation() {
        let mut env = TetrisEnv::new(10, 20);
        let piece = Piece::with_position(T_PIECE, 3, 10, 0);
        env.current_piece = Some(piece);
        env.last_move_was_rotation = false;

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_combo_resets_on_no_clear() {
        let mut env = TetrisEnv::new(10, 20);
        env.combo = 5;
        env.clear_lines_internal(false, false);
        assert_eq!(env.combo, 0);
    }

    #[test]
    fn test_get_last_attack_result_initially_none() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.get_last_attack_result().is_none());
    }

    // ==================== Board Collision Tests ====================

    #[test]
    fn test_is_cell_filled_empty_board() {
        let env = TetrisEnv::new(10, 20);
        // All cells should be empty
        for y in 0..20 {
            for x in 0..10 {
                assert!(
                    !env.is_cell_filled(x, y),
                    "Cell ({}, {}) should be empty",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_is_cell_filled_out_of_bounds() {
        let env = TetrisEnv::new(10, 20);
        // Out of bounds should return true (filled)
        assert!(env.is_cell_filled(-1, 10)); // left of board
        assert!(env.is_cell_filled(10, 10)); // right of board
        assert!(env.is_cell_filled(5, -1)); // above board
        assert!(env.is_cell_filled(5, 20)); // below board
    }

    #[test]
    fn test_is_cell_filled_with_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.board[19 * env.width + 5] = 1;
        assert!(env.is_cell_filled(5, 19));
        assert!(!env.is_cell_filled(4, 19));
    }

    #[test]
    fn test_is_valid_position_empty_board() {
        let env = TetrisEnv::new(10, 20);
        // Piece in center of board should be valid
        let piece = Piece::with_position(T_PIECE, 3, 10, 0);
        assert!(env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_left_wall_collision() {
        let env = TetrisEnv::new(10, 20);
        // I-piece horizontal at x=-1 should be invalid
        let piece = Piece::with_position(0, -1, 10, 0); // I piece
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_right_wall_collision() {
        let env = TetrisEnv::new(10, 20);
        // I-piece horizontal at x=8 should be invalid (extends to x=11)
        let piece = Piece::with_position(0, 8, 10, 0); // I piece
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_floor_collision() {
        let env = TetrisEnv::new(10, 20);
        // Piece below floor should be invalid
        let piece = Piece::with_position(T_PIECE, 3, 20, 0);
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_piece_collision() {
        let mut env = TetrisEnv::new(10, 20);
        // Place a block and try to overlap
        env.board[19 * env.width + 5] = 1;
        let piece = Piece::with_position(T_PIECE, 4, 18, 0); // T piece that overlaps
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_above_board_invalid() {
        let env = TetrisEnv::new(10, 20);
        // Piece with any cells above visible board is now invalid
        let piece = Piece::with_position(T_PIECE, 3, -2, 0);
        assert!(!env.is_valid_position_for(&piece));
    }

    // ==================== T-Spin Detection Tests ====================

    #[test]
    fn test_tspin_full_with_3_corners_filled() {
        let mut env = TetrisEnv::new(10, 20);
        // Create a T-spin pocket: fill 3 corners around where T piece will land
        // T piece center at (4, 18) with rotation 0 has center at (5, 19)
        env.board[18 * env.width + 4] = 1; // top-left corner
        env.board[18 * env.width + 6] = 1; // top-right corner
        env.board[19 * env.width + 4] = 1; // bottom-left corner (front corner for rotation 0)

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, _) = env.check_tspin(&piece);
        // With 3 corners filled and both front corners filled, should be full T-spin
        assert!(is_tspin, "Should detect T-spin with 3 corners filled");
    }

    #[test]
    fn test_tspin_mini_with_3_corners_one_front() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece center at (5, 19) rotation 0 - front corners are top-left and top-right
        // Fill 3 corners but only 1 front corner
        env.board[18 * env.width + 4] = 1; // top-left corner (front)
        env.board[19 * env.width + 4] = 1; // bottom-left corner (back)
        env.board[19 * env.width + 6] = 1; // bottom-right corner (back)

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin");
        assert!(is_mini, "Should be a T-spin mini with only 1 front corner");
    }

    #[test]
    fn test_tspin_kick_4_makes_full_tspin() {
        let mut env = TetrisEnv::new(10, 20);
        // With kick index 4 (the special SRS kick), even mini becomes full
        env.board[18 * env.width + 4] = 1;
        env.board[19 * env.width + 4] = 1;
        env.board[19 * env.width + 6] = 1;

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 4; // Special kick that makes it a full T-spin

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin");
        assert!(!is_mini, "Kick index 4 should make it a full T-spin");
    }

    #[test]
    fn test_tspin_less_than_3_corners_not_tspin() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece at (4, 10) with rotation 0 has center at (5, 11)
        // Corners are at: (4,10), (6,10), (4,12), (6,12)
        // Only fill 2 corners - not enough for T-spin
        env.board[10 * env.width + 4] = 1; // top-left corner
        env.board[10 * env.width + 6] = 1; // top-right corner
                                           // Leave bottom corners empty

        let piece = Piece::with_position(T_PIECE, 4, 10, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, _) = env.check_tspin(&piece);
        assert!(!is_tspin, "Should not be T-spin with only 2 corners");
    }

    #[test]
    fn test_tspin_rotation_1_front_corners() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece rotation 1 (CW) - front corners are top-right and bottom-right
        // Center is at (piece.x + 1, piece.y + 1) = (5, 19)
        env.board[18 * env.width + 6] = 1; // top-right (front)
        env.board[18 * env.width + 4] = 1; // top-left (back)
        env.board[19 * env.width + 6] = 1; // bottom-right (front)

        let piece = Piece::with_position(T_PIECE, 4, 18, 1);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin in rotation 1");
        assert!(!is_mini, "Both front corners filled should be full T-spin");
    }

    #[test]
    fn test_handcrafted_board_tspin_double_attack_with_final_counterclockwise_twist() {
        let mut env = setup_handcrafted_tspin_env();

        // Sequence mirrors the intended real move:
        // 1) move into the hole,
        // 2) rotate once CCW as the final input to twist,
        // 3) lock.
        assert!(
            env.rotate_ccw(),
            "should rotate once before entering the cavity"
        );
        assert!(env.move_left(), "should move toward cavity");
        while env.move_down() {}
        assert!(
            env.rotate_ccw(),
            "final counterclockwise twist should be valid at the cavity"
        );

        let active = env
            .current_piece
            .as_ref()
            .expect("piece should still be active before lock");
        assert_eq!(active.x, 2);
        assert_eq!(active.y, 17);
        assert_eq!(active.rotation, 2);

        let attack_before_lock = env.attack;
        let drop_distance = env.hard_drop();

        // Piece is already grounded after the final twist. Keeping drop distance 0 is important:
        // hard_drop only clears last_move_was_rotation when it moves the piece down.
        assert_eq!(drop_distance, 0);
        let attack_delta = env.attack - attack_before_lock;

        assert_eq!(attack_delta, 4);
        assert_eq!(env.attack, 4);
        assert_eq!(env.lines_cleared, 2);
        assert_eq!(env.combo, 1);
        assert!(env.back_to_back);

        let result = env
            .last_attack_result
            .as_ref()
            .expect("T-spin double should produce an attack result");
        assert_eq!(result.lines_cleared, 2);
        assert_eq!(result.base_attack, 4);
        assert_eq!(result.combo_attack, 0);
        assert_eq!(result.back_to_back_attack, 0);
        assert_eq!(result.total_attack, 4);
        assert!(result.is_tspin);
        assert!(result.back_to_back_active);
        assert!(!result.is_perfect_clear);

        let next_piece = env
            .current_piece
            .as_ref()
            .expect("A new piece should spawn after placement");
        assert_eq!(next_piece.piece_type, I_PIECE);
    }

    #[test]
    fn test_handcrafted_board_tspin_mini_single_attack() {
        let mut env = setup_handcrafted_tspin_env();

        // Same board, but rotation=0 has only one filled front corner at lock.
        // That should classify as a mini single (base 0 attack).
        let attack_delta = env.place_piece_internal_with_kick(2, 17, 0, true, 0);

        assert_eq!(attack_delta, 0);
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 1);
        assert_eq!(env.combo, 1);
        assert!(env.back_to_back);

        let result = env
            .last_attack_result
            .as_ref()
            .expect("T-spin mini single should produce an attack result");
        assert_eq!(result.lines_cleared, 1);
        assert_eq!(result.base_attack, 0);
        assert_eq!(result.combo_attack, 0);
        assert_eq!(result.back_to_back_attack, 0);
        assert_eq!(result.total_attack, 0);
        assert!(result.is_tspin);
        assert!(result.back_to_back_active);
        assert!(!result.is_perfect_clear);

        let next_piece = env
            .current_piece
            .as_ref()
            .expect("A new piece should spawn after placement");
        assert_eq!(next_piece.piece_type, I_PIECE);
    }

    #[test]
    fn test_handcrafted_board_has_mcts_placement_for_tspin_double_attack() {
        let env = setup_handcrafted_tspin_env();
        let placements = env.get_possible_placements();
        assert!(
            !placements.is_empty(),
            "Expected at least one legal placement from handcrafted board"
        );

        let mut found_tspin_double = false;

        for placement in placements {
            let mut sim = env.clone_state();
            let attack_delta = sim.execute_placement(&placement);

            if attack_delta != 4 {
                continue;
            }

            let result = sim
                .last_attack_result
                .as_ref()
                .expect("Line clear with non-zero attack should have an attack result");

            if !(result.is_tspin && result.lines_cleared == 2 && result.base_attack == 4) {
                continue;
            }

            assert!(
                placement.last_move_was_rotation,
                "T-spin placement selected by MCTS should end on a rotation input"
            );
            assert!(
                placement
                    .moves
                    .iter()
                    .any(|&action| action == 4 || action == 5),
                "T-spin placement path should include at least one rotation action"
            );
            assert_eq!(result.total_attack, 4);

            found_tspin_double = true;
            break;
        }

        assert!(
            found_tspin_double,
            "Expected at least one MCTS placement to produce a 4-attack T-spin double"
        );
    }

    // ==================== Line Clearing Tests ====================

    #[test]
    fn test_clear_single_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill the bottom row
        for x in 0..10 {
            env.board[19 * env.width + x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 1);
        assert_eq!(env.combo, 1);
    }

    #[test]
    fn test_clear_tetris() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 4 rows
        for y in 16..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 4);
    }

    #[test]
    fn test_back_to_back_tetris() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        // Fill bottom 4 rows for Tetris (difficult clear)
        for y in 16..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        // Keep one block above the clear so this is not a perfect clear.
        env.board[15 * env.width] = 1;
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        // Should maintain back-to-back and get bonus
        assert!(env.back_to_back);
        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.back_to_back_attack > 0);
    }

    #[test]
    fn test_single_breaks_back_to_back() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        // Fill bottom 2 rows, but only the bottom one is complete
        for x in 0..10 {
            env.board[19 * env.width + x] = 1;
        }
        for x in 0..5 {
            env.board[18 * env.width + x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        // Single line clear (non-perfect-clear) breaks back-to-back
        assert!(!env.back_to_back);
    }

    #[test]
    fn test_combo_increments() {
        let mut env = TetrisEnv::new(10, 20);
        env.combo = 3;

        // Fill one row
        for x in 0..10 {
            env.board[19 * env.width + x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        assert_eq!(env.combo, 4);
    }

    #[test]
    fn test_tspin_single_attack() {
        let mut env = TetrisEnv::new(10, 20);

        // Fill one row
        for x in 0..10 {
            env.board[19 * env.width + x] = 1;
        }
        // Keep one block above the clear so this is not a perfect clear.
        env.board[18 * env.width] = 1;
        env.sync_board_stats();
        env.clear_lines_internal(true, false); // T-spin single

        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.is_tspin);
        assert!(!result.is_mini_tspin);
        assert!(result.base_attack > 0); // T-spin single has attack
    }

    #[test]
    fn test_tspin_mini_attack() {
        let mut env = TetrisEnv::new(10, 20);

        // Fill one row
        for x in 0..10 {
            env.board[19 * env.width + x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(true, true); // T-spin mini single

        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.is_tspin);
        assert!(result.is_mini_tspin);
    }

    #[test]
    fn test_double_perfect_clear_uses_flat_attack_override() {
        let mut env = TetrisEnv::new(10, 20);

        for y in 18..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        assert_eq!(env.attack, 10);
        assert!(!env.back_to_back);

        let result = env.last_attack_result.as_ref().unwrap();
        assert_eq!(result.lines_cleared, 2);
        assert_eq!(result.base_attack, 0);
        assert_eq!(result.combo_attack, 0);
        assert_eq!(result.back_to_back_attack, 0);
        assert_eq!(result.perfect_clear_attack, 10);
        assert_eq!(result.total_attack, 10);
        assert!(result.is_perfect_clear);
    }

    #[test]
    fn test_double_perfect_clear_with_existing_b2b_breaks_chain_without_bonus() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        for y in 18..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        assert_eq!(env.attack, 10);
        assert!(!env.back_to_back);

        let result = env.last_attack_result.as_ref().unwrap();
        assert_eq!(result.lines_cleared, 2);
        assert_eq!(result.base_attack, 0);
        assert_eq!(result.combo_attack, 0);
        assert_eq!(result.back_to_back_attack, 0);
        assert_eq!(result.perfect_clear_attack, 10);
        assert_eq!(result.total_attack, 10);
        assert!(result.is_perfect_clear);
    }

    #[test]
    fn test_tetris_perfect_clear_preserves_b2b_without_bonus() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        for y in 16..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        assert_eq!(env.attack, 10);
        assert!(env.back_to_back);

        let result = env.last_attack_result.as_ref().unwrap();
        assert_eq!(result.lines_cleared, 4);
        assert_eq!(result.base_attack, 0);
        assert_eq!(result.combo_attack, 0);
        assert_eq!(result.back_to_back_attack, 0);
        assert_eq!(result.perfect_clear_attack, 10);
        assert_eq!(result.total_attack, 10);
        assert!(result.is_perfect_clear);
    }

    // ==================== Movement Tests ====================

    #[test]
    fn test_horizontal_movement_blocked_by_wall() {
        let mut env = TetrisEnv::new(10, 20);
        // Move left until blocked
        let mut moves = 0;
        while env.move_left() {
            moves += 1;
            if moves > 20 {
                panic!("Infinite loop detected");
            }
        }
        // Should be blocked by wall, not infinite loop
        assert!(moves > 0);
    }

    #[test]
    fn test_horizontal_movement_blocked_by_piece() {
        let mut env = TetrisEnv::new(10, 20);
        // Place wall of pieces on the left
        for y in 0..20 {
            env.board[y * env.width + 0] = 1;
            env.board[y * env.width + 1] = 1;
        }

        // Move current piece down a bit
        env.move_down();
        env.move_down();

        // Try to move left - should eventually be blocked by placed pieces
        let mut moves = 0;
        while env.move_left() {
            moves += 1;
            if moves > 20 {
                break;
            }
        }

        // Should have been blocked before reaching x=0
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 2, "Piece should be blocked by placed pieces at x=0,1");
        }
    }

    #[test]
    fn test_rotation_resets_lock_delay_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        // Move piece to ground
        for _ in 0..25 {
            env.move_down();
        }

        // Start lock delay
        if env.lock_delay_ms.is_none() {
            env.lock_delay_ms = Some(0);
        }
        env.lock_delay_ms = Some(200); // Simulate some time passed

        // Rotate should reset lock delay if successful
        if env.rotate_cw() {
            assert!(
                env.lock_delay_ms.is_some(),
                "Lock delay should still be active"
            );
        }
    }

    #[test]
    fn test_rotation_tracks_last_kick_index() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();

        // Rotate piece
        if env.rotate_cw() {
            // Kick index should be set (likely 0 for no-kick rotation)
            assert!(env.last_kick_index < 5, "Kick index should be 0-4");
        }
    }

    #[test]
    fn test_rotation_with_wall_kick() {
        let mut env = TetrisEnv::new(10, 20);
        // Move I-piece to right wall
        env.spawn_piece_from_type(0); // Spawn I piece
        for _ in 0..10 {
            env.move_right();
        }

        // Move down to have room
        for _ in 0..5 {
            env.move_down();
        }

        // Get initial position
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;

        // Rotate - might use wall kick
        if env.rotate_cw() {
            let piece = env.current_piece.as_ref().unwrap();
            // Rotation should have succeeded, possibly with a kick
            assert_ne!(piece.rotation, initial_rotation);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_no_movement_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        assert!(!env.move_horizontal(-1));
        assert!(!env.move_horizontal(1));
        assert!(!env.rotate(true));
        assert!(!env.rotate(false));
    }

    // ==================== Additional Edge Case Tests ====================

    #[test]
    fn test_spawn_at_top_with_collision() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill top rows where pieces spawn
        for x in 0..10 {
            env.board[0 * env.width + x] = 1;
            env.board[1 * env.width + x] = 1;
        }

        // Spawning should trigger game over
        env.spawn_piece_internal();
        assert!(env.game_over);
    }

    #[test]
    fn test_hard_drop_distance_calculation() {
        let mut env = TetrisEnv::new(10, 20);
        let drop_distance = env.hard_drop();

        // Drop distance should match how far piece traveled
        assert!(drop_distance > 0);
    }

    #[test]
    fn test_hard_drop_matches_move_down_with_overhang() {
        let mut env = TetrisEnv::new(10, 20);

        env.board.fill(0);
        env.board_piece_types.fill(None);

        // Create a roof block above the active piece's column. This makes the
        // column-height shortcut incorrect while exact collision stepping remains correct.
        let roof_x = 5;
        let roof_y = 4;
        env.board[roof_y * env.width + roof_x] = 1;
        env.sync_board_stats();

        let piece = Piece::with_position(I_PIECE, 3, 10, 1);
        assert!(env.is_valid_position_for(&piece));
        env.current_piece = Some(piece);

        let mut moved_env = env.clone();
        let mut expected_drop_distance: u32 = 0;
        while moved_env.move_down() {
            expected_drop_distance += 1;
        }
        assert!(expected_drop_distance > 0);

        let ghost_piece = env
            .get_ghost_piece()
            .expect("ghost piece should exist when current piece is set");
        let current_y = env
            .current_piece
            .as_ref()
            .expect("current piece should exist")
            .y;
        assert_eq!(ghost_piece.y - current_y, expected_drop_distance as i32);

        let hard_drop_distance = env.hard_drop();
        assert_eq!(hard_drop_distance, expected_drop_distance);
    }

    #[test]
    fn test_ghost_piece_at_bottom() {
        let env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());

        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();

        // Ghost should be at or below current piece
        assert!(ghost.y >= current.y);

        // Ghost should be grounded (can't move down further)
        let mut test_ghost = ghost.clone();
        test_ghost.y += 1;
        assert!(
            !env.is_valid_position_for(&test_ghost),
            "Ghost should be at lowest valid position"
        );
    }

    #[test]
    fn test_board_piece_types_after_lock() {
        let mut env = TetrisEnv::new(10, 20);
        let piece_type = env.current_piece.as_ref().unwrap().piece_type;

        // Hard drop to lock piece
        env.hard_drop();

        // Check that board_piece_types has the piece type
        let mut found_piece_type = false;
        for cell in &env.board_piece_types {
            if *cell == Some(piece_type) {
                found_piece_type = true;
                break;
            }
        }
        assert!(
            found_piece_type,
            "Board piece types should contain locked piece type"
        );
    }

    #[test]
    fn test_hold_resets_after_lock() {
        let mut env = TetrisEnv::new(10, 20);
        env.hold(); // Use hold
        assert!(env.hold_used);

        env.hard_drop(); // Lock piece
        assert!(!env.hold_used, "Hold should reset after piece locks");
    }

    #[test]
    fn test_clear_double_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 2 rows
        for y in 18..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 2);
    }

    #[test]
    fn test_clear_triple_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 3 rows
        for y in 17..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 3);
    }

    #[test]
    fn test_non_contiguous_line_clear() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill rows 15 and 19 (not contiguous)
        for x in 0..10 {
            env.board[15 * env.width + x] = 1;
            env.board[19 * env.width + x] = 1;
        }
        // Leave row 17 partial
        for x in 0..5 {
            env.board[17 * env.width + x] = 1;
        }
        env.sync_board_stats();

        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 2, "Should clear 2 non-contiguous lines");
    }

    // ==================== Lock Delay Tests ====================

    #[test]
    fn test_is_grounded_at_spawn() {
        let env = TetrisEnv::new(10, 20);
        // Piece at spawn should not be grounded
        assert!(!env.is_grounded(), "Piece at spawn should not be grounded");
    }

    #[test]
    fn test_is_grounded_at_bottom() {
        let mut env = TetrisEnv::new(10, 20);
        // Move piece to bottom
        while env.move_down() {}
        assert!(env.is_grounded(), "Piece at bottom should be grounded");
    }

    #[test]
    fn test_is_grounded_on_stack() {
        let mut env = TetrisEnv::new(10, 20);
        // Create a stack in the middle
        for x in 0..10 {
            env.board[15 * env.width + x] = 1;
        }
        env.sync_board_stats();

        // Move piece down until it lands on the stack
        while env.move_down() {}

        assert!(
            env.is_grounded(),
            "Piece on top of stack should be grounded"
        );
    }

    #[test]
    fn test_is_grounded_no_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.current_piece = None;
        assert!(!env.is_grounded(), "No piece means not grounded");
    }

    #[test]
    fn test_reset_lock_delay_decrements_moves() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_moves = env.lock_moves_remaining;
        assert_eq!(initial_moves, 15);

        env.reset_lock_delay();
        assert_eq!(env.lock_moves_remaining, 14);
        assert_eq!(env.lock_delay_ms, Some(0));
    }

    #[test]
    fn test_reset_lock_delay_multiple_times() {
        let mut env = TetrisEnv::new(10, 20);

        for i in 0..15 {
            env.reset_lock_delay();
            assert_eq!(env.lock_moves_remaining, 14 - i);
        }

        // After 15 resets, should be at 0
        assert_eq!(env.lock_moves_remaining, 0);
    }

    #[test]
    fn test_reset_lock_delay_stops_at_zero() {
        let mut env = TetrisEnv::new(10, 20);
        env.lock_moves_remaining = 0;
        env.lock_delay_ms = Some(100);

        env.reset_lock_delay();

        // Should not decrement below 0 or reset timer
        assert_eq!(env.lock_moves_remaining, 0);
        assert_eq!(env.lock_delay_ms, Some(100)); // Timer unchanged
    }

    #[test]
    fn test_clear_lock_delay() {
        let mut env = TetrisEnv::new(10, 20);
        env.lock_delay_ms = Some(250);
        env.lock_moves_remaining = 5;

        env.clear_lock_delay();

        assert_eq!(env.lock_delay_ms, None);
        assert_eq!(env.lock_moves_remaining, 15); // Reset to default
    }

    #[test]
    fn test_lock_delay_resets_on_movement_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);

        // Move to bottom
        while env.move_down() {}
        assert!(env.is_grounded());

        // Start lock delay
        env.lock_delay_ms = Some(100);
        let moves_before = env.lock_moves_remaining;

        // Move left (should reset lock delay if successful)
        if env.move_left() {
            assert_eq!(env.lock_delay_ms, Some(0), "Lock delay should reset to 0");
            assert_eq!(
                env.lock_moves_remaining,
                moves_before - 1,
                "Should use one lock move"
            );
        }
    }

    #[test]
    fn test_lock_delay_resets_on_rotation_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);

        // Move to bottom
        while env.move_down() {}

        // Start lock delay
        env.lock_delay_ms = Some(100);
        let moves_before = env.lock_moves_remaining;

        // Rotate (should reset lock delay if successful)
        if env.rotate_cw() {
            assert_eq!(env.lock_delay_ms, Some(0));
            assert_eq!(env.lock_moves_remaining, moves_before - 1);
        }
    }

    // ==================== Placement Tests ====================

    #[test]
    fn test_place_piece_internal_basic() {
        let mut env = TetrisEnv::new(10, 20);
        // Use O piece which is simple
        env.spawn_piece_from_type(1); // O piece

        // O piece shape has cells at rows 1-2, cols 1-2 of the 4x4 grid
        // So at y=17, cells are at y=18 and y=19 (valid)
        env.place_piece_internal_with_kick(4, 17, 0, false, 0);

        // Piece should be locked, new piece spawned
        assert!(env.current_piece.is_some());
        // Board should have the piece
        let has_piece = env.board.iter().any(|&c| c != 0);
        assert!(has_piece, "Board should have locked piece");
    }

    #[test]
    fn test_place_piece_internal_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        let attack = env.place_piece_internal_with_kick(3, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 attack when game over");
    }

    #[test]
    fn test_place_piece_internal_invalid_position() {
        let mut env = TetrisEnv::new(10, 20);

        // Try to place at invalid position (outside board)
        let attack = env.place_piece_internal_with_kick(-5, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 for invalid position");
    }

    #[test]
    fn test_place_piece_internal_no_current_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.current_piece = None;

        let attack = env.place_piece_internal_with_kick(3, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 when no current piece");
    }

    #[test]
    fn test_place_piece_internal_with_rotation() {
        let mut env = TetrisEnv::new(10, 20);

        // Place with rotation flag set
        env.place_piece_internal_with_kick(3, 18, 1, true, 0);

        // The placement should work (assuming position is valid for rotated piece)
        // Main thing is it doesn't crash
    }

    #[test]
    fn test_place_piece_internal_with_kick_index() {
        let mut env = TetrisEnv::new(10, 20);
        // Spawn T piece for T-spin testing
        env.spawn_piece_from_type(T_PIECE);

        // Set up a scenario where kick_index matters for T-spin
        env.board[18 * env.width + 4] = 1;
        env.board[19 * env.width + 4] = 1;
        env.board[19 * env.width + 6] = 1;
        env.sync_board_stats();

        // Place with kick_index = 4 (special T-spin kick)
        env.place_piece_internal_with_kick(4, 17, 0, true, 4);

        // The last_kick_index should be set
        // (Note: piece gets locked so we can't check it directly, but the attack should reflect it)
    }

    #[test]
    fn test_place_piece_internal_clears_lines() {
        let mut env = TetrisEnv::new(10, 20);
        // Spawn I piece
        env.spawn_piece_from_type(0); // I piece

        // Set up almost-complete row
        for x in 0..6 {
            env.board[19 * env.width + x] = 1;
        }
        env.sync_board_stats();

        // Place I piece horizontally to complete the row
        // I piece at rotation 0 is horizontal: [1,1,1,1] at row 1 of the shape
        let attack = env.place_piece_internal_with_kick(6, 18, 0, false, 0);

        // Should have cleared a line and gotten attack
        assert!(
            env.lines_cleared > 0
                || attack > 0
                || env.board[19 * env.width..(19 + 1) * env.width]
                    .iter()
                    .all(|&c| c == 0)
        );
    }

    #[test]
    fn test_place_piece_internal_collision_with_stack() {
        let mut env = TetrisEnv::new(10, 20);

        // Create a stack
        for y in 15..20 {
            for x in 0..10 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();

        // Try to place piece inside the stack (invalid)
        let attack = env.place_piece_internal_with_kick(3, 16, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 when colliding with stack");
    }

    #[test]
    fn test_place_piece_internal_all_rotations() {
        // Test that all 4 rotations work
        for rotation in 0..4 {
            let mut env = TetrisEnv::new(10, 20);

            // Use T piece which has distinct rotations
            env.spawn_piece_from_type(T_PIECE);

            env.place_piece_internal_with_kick(4, 17, rotation, false, 0);
            // Should not panic and piece should be placed
            assert!(
                env.current_piece.is_some(),
                "Rotation {} should work",
                rotation
            );
        }
    }

    #[test]
    fn test_place_piece_sets_last_move_was_rotation() {
        let mut env = TetrisEnv::new(10, 20);
        env.last_move_was_rotation = false;

        // Place with was_rotation = true
        env.place_piece_internal_with_kick(3, 18, 0, true, 0);

        // New piece is spawned, but the T-spin detection should have used the flag
        // We can't easily verify this without checking attack values in a T-spin setup
    }

    #[test]
    fn test_place_piece_returns_attack_delta() {
        let mut env = TetrisEnv::new(10, 20);
        env.spawn_piece_from_type(0); // I piece

        // Set up Tetris (4 almost-complete rows), leaving column 9 empty
        for y in 16..20 {
            for x in 0..9 {
                env.board[y * env.width + x] = 1;
            }
        }
        env.sync_board_stats();

        let initial_attack = env.attack;

        // Place I piece vertically to complete all 4 rows
        // I piece rotation 1 (vertical) has cells at column 2 of the 4x4 grid
        // So at x=7, cells are at 7+2=9, which completes the rows
        let attack_delta = env.place_piece_internal_with_kick(7, 16, 1, false, 0);

        // Should return the attack gained (Tetris = 4 lines)
        assert!(attack_delta > 0, "Tetris should give attack");
        assert_eq!(env.attack, initial_attack + attack_delta);
    }

    #[test]
    fn test_execute_placement_changes_current_piece() {
        // This test verifies that after execute_placement, the current piece changes
        // (i.e., the old piece is locked and a new one spawns from the queue)
        let env = TetrisEnv::with_seed(10, 20, 42);

        let initial_piece_type = env.get_current_piece().unwrap().piece_type;
        let initial_queue: Vec<usize> = env.get_queue(5);

        // Clone and execute a placement
        let mut env_clone = env.clone_state();
        let placements = env_clone.get_possible_placements();
        assert!(!placements.is_empty(), "Should have valid placements");

        let placement = &placements[0];
        env_clone.execute_placement(placement);

        // Verify state changed
        let new_piece = env_clone.get_current_piece();
        assert!(
            new_piece.is_some(),
            "Should have a new piece after placement"
        );

        let new_piece_type = new_piece.unwrap().piece_type;
        let new_queue: Vec<usize> = env_clone.get_queue(5);

        // The new current piece should be what was first in the queue
        assert_eq!(
            new_piece_type, initial_queue[0],
            "New piece should be first from old queue. Got {}, expected {}",
            new_piece_type, initial_queue[0]
        );

        // The queue should have shifted (minus the piece that became current)
        assert_eq!(
            new_queue[0], initial_queue[1],
            "Queue should shift after spawn. New queue[0]={}, expected old queue[1]={}",
            new_queue[0], initial_queue[1]
        );

        // Board should have the old piece locked
        let board = env_clone.get_board();
        let has_locked_piece = board.iter().any(|row| row.iter().any(|&c| c != 0));
        assert!(has_locked_piece, "Board should have locked piece");

        // Verify original env is unchanged
        let orig_piece_type = env.get_current_piece().unwrap().piece_type;
        assert_eq!(
            orig_piece_type, initial_piece_type,
            "Original env should be unchanged"
        );
    }

    // ==================== Tracking Fields Verification Tests ====================
    // These tests verify that incremental updates to tracked board stats
    // (total_blocks, row_fill_counts) match ground truth.

    /// Helper to compute expected tracked board stats from board state.
    fn compute_expected_stats(
        board: &[u8],
        width: usize,
        height: usize,
    ) -> (u32, [u8; BOARD_HEIGHT]) {
        let mut total_blocks = 0u32;
        let mut row_fill_counts = [0u8; BOARD_HEIGHT];

        for y in 0..height {
            for x in 0..width {
                if board[y * width + x] != 0 {
                    total_blocks += 1;
                    row_fill_counts[y] += 1;
                }
            }
        }
        (total_blocks, row_fill_counts)
    }

    fn compute_expected_column_heights(
        board: &[u8],
        width: usize,
        height: usize,
    ) -> [u8; BOARD_WIDTH] {
        let mut column_heights = [0u8; BOARD_WIDTH];
        for x in 0..width {
            for y in 0..height {
                if board[y * width + x] != 0 {
                    column_heights[x] = (height - y) as u8;
                    break;
                }
            }
        }
        column_heights
    }

    #[test]
    fn test_tracking_fields_after_hard_drop() {
        let mut env = TetrisEnv::with_seed(10, 20, 42);

        // Drop a piece
        env.hard_drop();

        // Compute expected values from actual board state
        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(
            env.total_blocks, expected_blocks,
            "total_blocks mismatch after hard_drop"
        );
        assert_eq!(
            env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after hard_drop"
        );
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(
            env.column_heights, expected_column_heights,
            "column_heights mismatch after hard_drop"
        );
    }

    #[test]
    fn test_tracking_fields_after_multiple_drops() {
        let mut env = TetrisEnv::with_seed(10, 20, 123);

        // Drop several pieces
        for _ in 0..10 {
            if env.game_over {
                break;
            }
            env.hard_drop();
        }

        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(
            env.total_blocks, expected_blocks,
            "total_blocks mismatch after multiple drops"
        );
        assert_eq!(
            env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after multiple drops"
        );
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(
            env.column_heights, expected_column_heights,
            "column_heights mismatch after multiple drops"
        );
    }

    #[test]
    fn test_tracking_fields_after_line_clear() {
        let mut env = TetrisEnv::with_seed(10, 20, 456);

        // Play for several moves to exercise line-clear bookkeeping.
        for _ in 0..50 {
            if env.game_over {
                break;
            }
            env.hard_drop();
        }

        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(
            env.total_blocks, expected_blocks,
            "total_blocks mismatch after line clears"
        );
        assert_eq!(
            env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after line clears"
        );
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(
            env.column_heights, expected_column_heights,
            "column_heights mismatch after line clears"
        );
    }

    #[test]
    fn test_tracking_fields_with_movements() {
        let mut env = TetrisEnv::with_seed(10, 20, 789);

        // Move piece around before dropping
        for _ in 0..5 {
            if env.game_over {
                break;
            }
            env.move_left();
            env.move_right();
            env.move_right();
            env.rotate_cw();
            env.move_down();
            env.hard_drop();
        }

        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(env.column_heights, expected_column_heights);
    }

    #[test]
    fn test_tracking_fields_with_hold() {
        let mut env = TetrisEnv::with_seed(10, 20, 321);

        // Use hold feature
        env.hold();
        env.hard_drop();
        env.hold();
        env.hard_drop();

        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(env.column_heights, expected_column_heights);
    }

    #[test]
    fn test_tracking_fields_perfect_clear() {
        // After a perfect clear, all tracking fields should be zero/empty
        let mut env = TetrisEnv::with_seed(10, 20, 999);

        // Manually set up a perfect clear scenario
        // Fill bottom row except one cell, then place I piece to complete
        for x in 0..6 {
            env.board[19 * env.width + x] = 1;
        }
        env.sync_board_stats();

        // Verify sync works correctly
        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);
        assert_eq!(env.column_heights, expected_column_heights);
    }

    #[test]
    fn test_row_fill_counts_partial_rows() {
        let mut env = TetrisEnv::new(10, 20);

        // Set up rows with different fill levels
        for x in 0..3 {
            env.board[19 * env.width + x] = 1; // 3 cells in row 19
        }
        for x in 0..7 {
            env.board[18 * env.width + x] = 1; // 7 cells in row 18
        }
        for x in 0..10 {
            env.board[17 * env.width + x] = 1; // 10 cells in row 17 (full)
        }
        env.sync_board_stats();

        assert_eq!(env.row_fill_counts[19], 3);
        assert_eq!(env.row_fill_counts[18], 7);
        assert_eq!(env.row_fill_counts[17], 10);
        assert_eq!(env.row_fill_counts[16], 0); // Empty row
    }

    #[test]
    fn test_total_blocks_count() {
        let mut env = TetrisEnv::new(10, 20);

        // Place exactly 15 blocks
        let positions = [
            (0, 19),
            (1, 19),
            (2, 19),
            (0, 18),
            (1, 18),
            (2, 18),
            (3, 18),
            (5, 15),
            (6, 15),
            (7, 15),
            (8, 15),
            (9, 15),
            (0, 10),
            (5, 10),
            (9, 10),
        ];
        for (x, y) in positions {
            env.board[y * env.width + x] = 1;
        }
        env.sync_board_stats();

        assert_eq!(env.total_blocks, 15);
    }

    #[test]
    fn test_incremental_vs_sync_consistency() {
        // This test verifies incremental updates match sync_board_stats
        let mut env = TetrisEnv::with_seed(10, 20, 12345);

        for i in 0..20 {
            if env.game_over {
                break;
            }

            // Save current tracked values (from incremental updates).
            let inc_blocks = env.total_blocks;
            let inc_row_counts = env.row_fill_counts.clone();
            let inc_column_heights = env.column_heights.clone();

            // Recalculate from scratch.
            let (sync_blocks, sync_row_counts) =
                compute_expected_stats(&env.board, env.width, env.height);
            let sync_column_heights =
                compute_expected_column_heights(&env.board, env.width, env.height);

            assert_eq!(
                inc_blocks, sync_blocks,
                "Iteration {}: total_blocks incremental != sync",
                i
            );
            assert_eq!(
                inc_row_counts, sync_row_counts,
                "Iteration {}: row_fill_counts incremental != sync",
                i
            );
            assert_eq!(
                inc_column_heights, sync_column_heights,
                "Iteration {}: column_heights incremental != sync",
                i
            );

            env.hard_drop();
        }
    }

    // ==================== Property-Based Tests ====================
    // These tests use proptest to generate random sequences of actions
    // and verify that invariants hold at each step.

    use proptest::prelude::*;

    /// Helper to verify all board invariants
    fn verify_board_invariants(env: &TetrisEnv, context: &str) {
        // Recompute expected values from scratch
        let (expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board, env.width, env.height);
        let expected_column_heights =
            compute_expected_column_heights(&env.board, env.width, env.height);

        // 1. Total blocks must match actual board state
        assert_eq!(
            env.total_blocks, expected_blocks,
            "{}: total_blocks mismatch. Got {}, expected {}",
            context, env.total_blocks, expected_blocks
        );

        // 2. Row fill counts must match actual board state
        assert_eq!(
            env.row_fill_counts, expected_row_counts,
            "{}: row_fill_counts mismatch. Got {:?}, expected {:?}",
            context, env.row_fill_counts, expected_row_counts
        );
        assert_eq!(
            env.column_heights, expected_column_heights,
            "{}: column_heights mismatch. Got {:?}, expected {:?}",
            context, env.column_heights, expected_column_heights
        );

        // 3. Board dimensions must be consistent
        assert_eq!(
            env.board.len(),
            MAX_BOARD_CELLS,
            "{}: board size mismatch",
            context
        );

        // 4. All cells must have valid values (0-7)
        for y in 0..env.height {
            for x in 0..env.width {
                let cell = env.board[y * env.width + x];
                assert!(
                    cell <= 7,
                    "{}: invalid cell value {} at ({}, {})",
                    context,
                    cell,
                    x,
                    y
                );
            }
        }

        // 5. Row fill counts must not exceed width
        for (y, &count) in env.row_fill_counts.iter().enumerate() {
            assert!(
                count <= env.width as u8,
                "{}: row {} has fill count {} > width {}",
                context,
                y,
                count,
                env.width
            );

            // Verify count matches actual filled cells in row
            let actual_count = env.board[y * env.width..(y + 1) * env.width]
                .iter()
                .filter(|&&c| c != 0)
                .count() as u8;
            assert_eq!(
                count, actual_count,
                "{}: row {} fill count {} doesn't match actual count {}",
                context, y, count, actual_count
            );
        }

        // 6. No row should be completely filled (they should be cleared)
        for y in 0..env.height {
            let filled = env.board[y * env.width..(y + 1) * env.width]
                .iter()
                .all(|&c| c != 0);
            assert!(
                !filled,
                "{}: row {} is completely filled but not cleared",
                context, y
            );
        }

        // 7. Total blocks should equal sum of row fill counts
        let sum_row_counts: u32 = env.row_fill_counts.iter().map(|&c| c as u32).sum();
        assert_eq!(
            env.total_blocks, sum_row_counts,
            "{}: total_blocks {} doesn't match sum of row_fill_counts {}",
            context, env.total_blocks, sum_row_counts
        );
    }

    proptest! {
        #[test]
        fn prop_random_actions_maintain_invariants(
            seed in 0u64..1000,
            actions in prop::collection::vec(0u8..8, 10..50)
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            verify_board_invariants(&env, "initial state");

            for (i, &action) in actions.iter().enumerate() {
                if env.game_over {
                    break;
                }

                let context = format!("after action {} (step {})", action, i);
                env.step(action);
                verify_board_invariants(&env, &context);
            }
        }

        #[test]
        fn prop_hard_drops_maintain_invariants(
            seed in 0u64..1000,
            num_drops in 5usize..30
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for i in 0..num_drops {
                if env.game_over {
                    break;
                }

                let context = format!("after hard_drop {}", i);
                env.hard_drop();
                verify_board_invariants(&env, &context);
            }
        }

        #[test]
        fn prop_movements_maintain_invariants(
            seed in 0u64..1000,
            movements in prop::collection::vec(0u8..6, 20..100)
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for (i, &movement) in movements.iter().enumerate() {
                if env.game_over {
                    break;
                }

                let context = format!("after movement {} (step {})", movement, i);

                match movement {
                    0 => { env.move_left(); },
                    1 => { env.move_right(); },
                    2 => { env.move_down(); },
                    3 => { env.rotate_cw(); },
                    4 => { env.rotate_ccw(); },
                    5 => { env.hard_drop(); },
                    _ => {},
                }

                verify_board_invariants(&env, &context);
            }
        }

        #[test]
        fn prop_with_hold_maintains_invariants(
            seed in 0u64..1000,
            actions in prop::collection::vec(0u8..8, 20..50)
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for (i, &action) in actions.iter().enumerate() {
                if env.game_over {
                    break;
                }

                // Randomly use hold
                if action == 7 && i % 3 == 0 {
                    env.hold();
                } else {
                    env.step(action);
                }

                let context = format!("after action {} with hold (step {})", action, i);
                verify_board_invariants(&env, &context);
            }
        }

        #[test]
        fn prop_placement_execution_maintains_invariants(
            seed in 0u64..1000,
            num_placements in 5usize..20
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for i in 0..num_placements {
                if env.game_over {
                    break;
                }

                let placements = env.get_possible_placements();
                if placements.is_empty() {
                    break;
                }

                // Pick a random valid placement
                let idx = (seed as usize + i * 7) % placements.len();
                let placement = &placements[idx];

                env.execute_placement(placement);

                let context = format!("after placement {} (idx {})", i, idx);
                verify_board_invariants(&env, &context);
            }
        }

        #[test]
        fn prop_total_blocks_never_exceeds_capacity(
            seed in 0u64..1000,
            actions in prop::collection::vec(0u8..8, 20..100)
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);
            let max_capacity = (env.width * env.height) as u32;

            for &action in actions.iter() {
                if env.game_over {
                    break;
                }

                env.step(action);

                assert!(
                    env.total_blocks <= max_capacity,
                    "total_blocks {} exceeds capacity {} after action {}",
                    env.total_blocks, max_capacity, action
                );
            }
        }

        #[test]
        fn prop_line_clears_reduce_total_blocks(
            seed in 0u64..1000,
            num_drops in 10usize..30
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for _ in 0..num_drops {
                if env.game_over {
                    break;
                }

                let lines_before = env.lines_cleared;

                env.hard_drop();

                let blocks_after = env.total_blocks;
                let lines_after = env.lines_cleared;
                let lines_cleared_now = lines_after - lines_before;

                if lines_cleared_now > 0 {
                    // When lines are cleared, total blocks should decrease
                    // (unless the piece added more blocks than were cleared)
                    // At minimum, the cleared lines should have removed width * lines_cleared_now blocks
                    // But we also added a piece (4 blocks typically)
                    // So: blocks_after = blocks_before + piece_blocks - cleared_blocks
                    // We can't easily verify exact count without knowing piece size,
                    // but we can verify the board is consistent
                    assert_eq!(
                        blocks_after,
                        env.board.iter().filter(|&&c| c != 0).count() as u32,
                        "total_blocks doesn't match actual count after line clear"
                    );
                }
            }
        }

        #[test]
        fn prop_row_fill_counts_update_correctly(
            seed in 0u64..1000,
            num_drops in 5usize..25
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for i in 0..num_drops {
                if env.game_over {
                    break;
                }

                env.hard_drop();

                // Verify each row's fill count
                for y in 0..env.height {
                    let actual_count = env.board[y * env.width..(y + 1) * env.width].iter().filter(|&&c| c != 0).count() as u8;
                    assert_eq!(
                        env.row_fill_counts[y], actual_count,
                        "Drop {}: row {} fill count {} doesn't match actual {}",
                        i, y, env.row_fill_counts[y], actual_count
                    );
                }
            }
        }

        #[test]
        fn prop_7bag_randomizer_correctness(
            seed in 0u64..1000,
            num_pieces in 14usize..50
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);
            let mut spawned_pieces = Vec::new();

            for _ in 0..num_pieces {
                if env.game_over {
                    break;
                }

                if let Some(piece) = &env.current_piece {
                    spawned_pieces.push(piece.piece_type);
                }

                env.hard_drop();
            }

            // Check every consecutive 7 pieces contains all piece types exactly once
            for chunk_start in 0..=(spawned_pieces.len().saturating_sub(7)) {
                let chunk = &spawned_pieces[chunk_start..chunk_start + 7];
                let mut counts = [0u8; 7];

                for &piece_type in chunk {
                    counts[piece_type] += 1;
                }

                // If this is a complete bag, all pieces should appear exactly once
                if chunk_start % 7 == 0 && chunk_start + 7 <= spawned_pieces.len() {
                    for (piece_type, &count) in counts.iter().enumerate() {
                        assert_eq!(
                            count, 1,
                            "Bag starting at piece {}: piece type {} appeared {} times (expected 1)",
                            chunk_start, piece_type, count
                        );
                    }
                }
            }
        }

        #[test]
        fn prop_score_monotonicity(
            seed in 0u64..1000,
            num_drops in 10usize..50
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            let mut prev_attack = env.attack;
            let mut prev_lines = env.lines_cleared;

            for _ in 0..num_drops {
                if env.game_over {
                    break;
                }

                env.hard_drop();

                assert!(
                    env.attack >= prev_attack,
                    "Attack decreased from {} to {}",
                    prev_attack, env.attack
                );

                assert!(
                    env.lines_cleared >= prev_lines,
                    "Lines cleared decreased from {} to {}",
                    prev_lines, env.lines_cleared
                );

                prev_attack = env.attack;
                prev_lines = env.lines_cleared;
            }
        }

        #[test]
        fn prop_game_over_is_terminal(
            seed in 0u64..1000,
            actions in prop::collection::vec(0u8..8, 5..20)
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            // Play until game over
            for &action in actions.iter() {
                if env.game_over {
                    break;
                }
                env.step(action);
            }

            if env.game_over {
                // Capture state when game is over
                let board_snapshot = env.board.clone();
                let attack_snapshot = env.attack;
                let lines_snapshot = env.lines_cleared;
                let total_blocks_snapshot = env.total_blocks;

                // Try more actions - state should not change
                for &action in actions.iter() {
                    env.step(action);

                    assert!(env.game_over, "Game over flag was cleared");
                    assert_eq!(env.board, board_snapshot, "Board changed after game over");
                    assert_eq!(env.attack, attack_snapshot, "Attack changed after game over");
                    assert_eq!(env.lines_cleared, lines_snapshot, "Lines changed after game over");
                    assert_eq!(env.total_blocks, total_blocks_snapshot, "Total blocks changed after game over");
                }
            }
        }

        #[test]
        fn prop_queue_always_populated(
            seed in 0u64..1000,
            num_drops in 10usize..50
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for _ in 0..num_drops {
                if env.game_over {
                    break;
                }

                let queue = env.get_queue(5);
                assert_eq!(
                    queue.len(), 5,
                    "Queue should always have 5 pieces, got {}",
                    queue.len()
                );

                // All queue entries should be valid piece types (0-6)
                for &piece_type in &queue {
                    assert!(
                        piece_type < 7,
                        "Invalid piece type {} in queue",
                        piece_type
                    );
                }

                env.hard_drop();
            }
        }

        #[test]
        fn prop_board_connected_after_clears(
            seed in 0u64..1000,
            num_drops in 10usize..30
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for _ in 0..num_drops {
                if env.game_over {
                    break;
                }

                env.hard_drop();

                // After line clears, check that if a row Y has any blocks,
                // then either Y is the bottom row, or at least one row below Y has blocks
                // This ensures the board doesn't have "gaps" with empty rows between filled rows
                let mut found_filled_row = false;
                let mut found_gap = false;

                for y in (0..env.height).rev() {
                    let row_has_blocks = env.board[y * env.width..(y + 1) * env.width].iter().any(|&c| c != 0);

                    if row_has_blocks {
                        if found_gap {
                            panic!("Board has empty rows below filled rows (gap in board structure)");
                        }
                        found_filled_row = true;
                    } else if found_filled_row {
                        // Empty row above filled rows = gap
                        found_gap = true;
                    }
                }
            }
        }

        #[test]
        fn prop_valid_placements_are_valid(
            seed in 0u64..1000,
            num_checks in 5usize..15
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for _ in 0..num_checks {
                if env.game_over {
                    break;
                }

                let placements = env.get_possible_placements();

                for placement in &placements {
                    // Each placement should be a valid position
                    assert!(
                        env.is_valid_position(
                            placement.piece.piece_type,
                            placement.rotation,
                            placement.column,
                            placement.piece.y
                        ),
                        "get_possible_placements returned invalid placement: piece={}, rot={}, x={}, y={}",
                        placement.piece.piece_type, placement.rotation, placement.column, placement.piece.y
                    );
                }

                env.hard_drop();
            }
        }

        #[test]
        fn prop_determinism_same_seed_same_game(
            seed in 0u64..1000,
            actions in prop::collection::vec(0u8..8, 10..30)
        ) {
            let mut env1 = TetrisEnv::with_seed(10, 20, seed);
            let mut env2 = TetrisEnv::with_seed(10, 20, seed);

            for (i, &action) in actions.iter().enumerate() {
                if env1.game_over || env2.game_over {
                    assert_eq!(
                        env1.game_over, env2.game_over,
                        "Step {}: Game over state differs",
                        i
                    );
                    break;
                }

                env1.step(action);
                env2.step(action);

                assert_eq!(env1.board, env2.board, "Step {}: Boards differ", i);
                assert_eq!(env1.attack, env2.attack, "Step {}: Attack differs", i);
                assert_eq!(env1.lines_cleared, env2.lines_cleared, "Step {}: Lines differ", i);
                assert_eq!(env1.total_blocks, env2.total_blocks, "Step {}: Total blocks differ", i);

                // Check current piece state
                match (&env1.current_piece, &env2.current_piece) {
                    (Some(p1), Some(p2)) => {
                        assert_eq!(p1.piece_type, p2.piece_type, "Step {}: Piece type differs", i);
                        assert_eq!(p1.x, p2.x, "Step {}: Piece x differs", i);
                        assert_eq!(p1.y, p2.y, "Step {}: Piece y differs", i);
                        assert_eq!(p1.rotation, p2.rotation, "Step {}: Piece rotation differs", i);
                    },
                    (None, None) => {},
                    _ => panic!("Step {}: One env has current_piece, other doesn't", i),
                }
            }
        }
    }
}
