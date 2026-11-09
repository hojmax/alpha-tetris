//! Line Clearing and T-Spin Detection
//!
//! Locking pieces, clearing lines, and T-spin mechanics.

use crate::game::piece::{get_cells, Piece};
use crate::game::scoring::{calculate_attack_breakdown, determine_clear_type, AttackResult};
use crate::game::tspin::{classify_tspin, TSpin};

use super::TetrisEnv;

impl TetrisEnv {
    /// T-spin status of `piece` locking now, as `(is_tspin, is_mini)`.
    pub(crate) fn check_tspin(&self, piece: &Piece) -> (bool, bool) {
        let tspin = classify_tspin(
            self.board_cells(),
            self.width,
            self.height,
            piece.piece_type,
            piece.x,
            piece.y,
            piece.rotation,
            self.spin_flag,
        );
        (tspin.is_spin(), tspin == TSpin::Mini)
    }

    pub(crate) fn lock_piece_internal(&mut self) {
        if let Some(piece) = self.current_piece.take() {
            let (is_tspin, is_mini) = self.check_tspin(&piece);

            let has_piece_types = !self.board_piece_types.is_empty();
            for (x, y) in get_cells(piece.piece_type, piece.rotation, piece.x, piece.y) {
                self.board[y as usize * self.width + x as usize] = 1;
                if has_piece_types {
                    self.board_piece_types[y as usize * self.width + x as usize] =
                        Some(piece.piece_type);
                }
                // Update row fill count
                self.row_fill_counts[y as usize] += 1;
                let column_height = (self.height - y as usize) as u8;
                self.column_heights[x as usize] =
                    self.column_heights[x as usize].max(column_height);
            }
            // Tetrominos always have 4 cells (all within bounds due to is_valid_position check)
            self.total_blocks += 4;
            self.placement_count += 1;

            // Invalidate placements cache since board changed
            self.invalidate_placement_cache();
            self.invalidate_board_analysis_cache();

            self.clear_lines_internal(is_tspin, is_mini);
            self.hold_used = false;
            self.spawn_piece_internal();
        }
    }

    pub(crate) fn clear_lines_internal(&mut self, is_tspin: bool, is_mini: bool) {
        let width = self.width as u8;
        let num_lines = self
            .row_fill_counts
            .iter()
            .filter(|&&count| count == width)
            .count() as u32;

        if num_lines == 0 {
            self.combo = 0;
            self.last_attack_result = None;
            return;
        }

        self.invalidate_board_analysis_cache();

        // Compact non-cleared rows downward using in-place copy_within (zero allocations)
        let w = self.width;
        let mut write_row = self.height;
        for read_row in (0..self.height).rev() {
            if self.row_fill_counts[read_row] < width {
                write_row -= 1;
                if write_row != read_row {
                    let src = read_row * w;
                    let dst = write_row * w;
                    self.board.copy_within(src..src + w, dst);
                    if !self.board_piece_types.is_empty() {
                        self.board_piece_types.copy_within(src..src + w, dst);
                    }
                    self.row_fill_counts[write_row] = self.row_fill_counts[read_row];
                }
            }
        }
        // Zero out top rows
        let n = num_lines as usize;
        self.board[..n * w].fill(0);
        if !self.board_piece_types.is_empty() {
            self.board_piece_types[..n * w].fill(None);
        }
        self.row_fill_counts[..n].fill(0);

        // Each cleared line removes width blocks
        self.total_blocks -= self.width as u32 * num_lines;
        self.recompute_column_heights();

        let clear_type = determine_clear_type(num_lines, is_tspin, is_mini);
        let is_pc = self.is_perfect_clear();
        let attack_breakdown =
            calculate_attack_breakdown(clear_type, self.combo, self.back_to_back, is_pc);

        let mut result = AttackResult::new();
        result.lines_cleared = num_lines;
        result.base_attack = attack_breakdown.base_attack;
        result.combo_attack = attack_breakdown.combo_attack;
        result.back_to_back_attack = attack_breakdown.back_to_back_attack;
        result.perfect_clear_attack = attack_breakdown.perfect_clear_attack;
        result.total_attack = attack_breakdown.total_attack;
        result.combo = self.combo + 1;
        result.back_to_back_active = attack_breakdown.back_to_back_active;
        result.is_tspin = is_tspin;
        result.is_mini_tspin = is_mini;
        result.is_perfect_clear = is_pc;

        self.attack += attack_breakdown.total_attack;
        self.combo += 1;
        self.back_to_back = attack_breakdown.back_to_back_active;
        self.last_attack_result = Some(result);
        self.lines_cleared += num_lines;
    }
}
