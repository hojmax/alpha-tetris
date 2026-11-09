//! Board Operations
//!
//! Board state queries and validation.

use crate::game::piece::{get_cells, Piece};

use super::TetrisEnv;

impl TetrisEnv {
    #[inline]
    pub(crate) fn board_cells(&self) -> &[u8] {
        &self.board[..self.width * self.height]
    }

    /// Compute how far a piece can drop using exact collision checks.
    /// Returns the drop distance (0 if already at bottom).
    pub(crate) fn compute_drop_distance(&self, piece: &Piece) -> i32 {
        let mut drop_distance = 0;
        while self.is_valid_position(
            piece.piece_type,
            piece.rotation,
            piece.x,
            piece.y + drop_distance + 1,
        ) {
            drop_distance += 1;
        }
        drop_distance
    }
}

impl TetrisEnv {
    /// Check if a position is valid for a piece
    pub(crate) fn is_valid_position_for(&self, piece: &Piece) -> bool {
        self.is_valid_position(piece.piece_type, piece.rotation, piece.x, piece.y)
    }

    pub(crate) fn is_valid_position(
        &self,
        piece_type: usize,
        rotation: usize,
        x: i32,
        y: i32,
    ) -> bool {
        for (cx, cy) in get_cells(piece_type, rotation, x, y) {
            if cx < 0 || cx >= self.width as i32 || cy < 0 || cy >= self.height as i32 {
                return false;
            }
            if self.board[cy as usize * self.width + cx as usize] != 0 {
                return false;
            }
        }
        true
    }

    /// Check if the board is completely empty (perfect clear)
    /// O(1) using total_blocks counter
    pub(crate) fn is_perfect_clear(&self) -> bool {
        self.total_blocks == 0
    }

    pub(crate) fn recompute_column_heights(&mut self) {
        self.column_heights.fill(0);
        for x in 0..self.width {
            for y in 0..self.height {
                if self.board[y * self.width + x] == 0 {
                    continue;
                }
                self.column_heights[x] = (self.height - y) as u8;
                break;
            }
        }
    }

    pub(crate) fn recompute_total_blocks_and_row_fill_counts(&mut self) {
        self.total_blocks = 0;
        self.row_fill_counts.fill(0);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.board[y * self.width + x] == 0 {
                    continue;
                }
                self.total_blocks += 1;
                self.row_fill_counts[y] += 1;
            }
        }
    }

    /// Recalculate total_blocks, row_fill_counts, and column_heights from the board state.
    /// Call this after directly modifying the board.
    pub(crate) fn sync_board_stats(&mut self) {
        self.recompute_total_blocks_and_row_fill_counts();
        self.recompute_column_heights();
        self.invalidate_board_analysis_cache();
    }
}
