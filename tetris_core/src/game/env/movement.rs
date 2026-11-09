//! Piece Movement
//!
//! Horizontal movement and rotation (internal implementations).

use crate::game::kicks::{get_kicks, Rotation};

use super::TetrisEnv;

impl TetrisEnv {
    /// Internal horizontal movement logic. Like Jstris, a sideways move keeps the spin
    /// flag; only downward movement clears it.
    pub(crate) fn move_horizontal(&mut self, dx: i32) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x += dx;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                if was_grounded && self.lock_delay_ms.is_some() {
                    self.reset_lock_delay();
                }
                return true;
            }
        }
        false
    }

    /// Internal rotation logic using the Jstris SRS wall kicks
    pub(crate) fn rotate(&mut self, rotation: Rotation) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let to_state = rotation.target(piece.rotation);
            for &(dx, dy) in get_kicks(piece.piece_type, piece.rotation, rotation) {
                let new_x = piece.x + dx;
                let new_y = piece.y + dy;
                if self.is_valid_position(piece.piece_type, to_state, new_x, new_y) {
                    let mut new_piece = piece.clone();
                    new_piece.x = new_x;
                    new_piece.y = new_y;
                    new_piece.rotation = to_state;
                    self.current_piece = Some(new_piece);
                    if was_grounded && self.lock_delay_ms.is_some() {
                        self.reset_lock_delay();
                    }
                    self.spin_flag = true;
                    return true;
                }
            }
        }
        false
    }
}
