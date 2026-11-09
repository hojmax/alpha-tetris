//! Placement Execution
//!
//! Internal placement logic.

use super::TetrisEnv;

impl TetrisEnv {
    /// Lock the current piece at `(x, y, rotation)`. `spin_flag` says whether the path that
    /// got it there rotated after its last downward move (needed for T-spin detection).
    pub(crate) fn place_piece_internal(
        &mut self,
        x: i32,
        y: i32,
        rotation: usize,
        spin_flag: bool,
    ) -> u32 {
        if self.game_over {
            return 0;
        }

        if let Some(ref piece) = self.current_piece {
            debug_assert!(rotation < 4, "Invalid rotation: {}", rotation);

            let piece_type = piece.piece_type;

            if !self.is_valid_position(piece_type, rotation, x, y) {
                return 0;
            }

            let mut new_piece = piece.clone();
            new_piece.x = x;
            new_piece.y = y;
            new_piece.rotation = rotation;
            self.current_piece = Some(new_piece);

            self.spin_flag = spin_flag;

            let old_attack = self.attack;
            self.lock_piece_internal();
            self.attack - old_attack
        } else {
            0
        }
    }
}
