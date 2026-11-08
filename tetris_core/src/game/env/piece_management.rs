//! Piece Queue and Hold Management
//!
//! 7-bag randomizer, queue management, and hold functionality (internal implementations).

use rand::seq::SliceRandom;

use crate::game::constants::NUM_PIECE_TYPES;
use crate::game::piece::Piece;

use super::TetrisEnv;

impl TetrisEnv {
    /// Ensure the piece queue has at least `count` pieces
    pub(crate) fn fill_queue(&mut self, count: usize) {
        while self.piece_queue.len() < count {
            let mut bag: [usize; NUM_PIECE_TYPES] = std::array::from_fn(|piece_type| piece_type);
            bag.shuffle(&mut self.rng);
            self.piece_queue.extend(bag);
        }
    }

    /// Bitmask of the piece types that can follow the first `queue_len` queued pieces under
    /// the 7-bag randomizer (bit `i` set = piece type `i` possible).
    pub(crate) fn possible_next_piece_mask(&self, queue_len: usize) -> u8 {
        debug_assert!(queue_len <= self.piece_queue.len());
        const ALL_PIECES: u8 = (1 << NUM_PIECE_TYPES) - 1;

        let next_position = self.pieces_spawned as usize + queue_len;
        let bag_number = next_position / 7;
        let position_in_bag = next_position % 7;

        if position_in_bag == 0 {
            return ALL_PIECES;
        }

        let bag_start = bag_number * 7;
        let bag_end = bag_start + 7;
        let mut used_in_bag = 0u8;

        let current_bag_pos = self.current_piece_bag_position as usize;
        if current_bag_pos >= bag_start && current_bag_pos < bag_end {
            if let Some(ref piece) = self.current_piece {
                used_in_bag |= 1 << piece.piece_type;
            }
        }

        if let (Some(hold_type), Some(hold_bag_pos)) =
            (self.hold_piece, self.hold_piece_bag_position)
        {
            let hold_pos = hold_bag_pos as usize;
            if hold_pos >= bag_start && hold_pos < bag_end {
                used_in_bag |= 1 << hold_type;
            }
        }

        for (i, piece_type) in self.piece_queue.iter().take(queue_len).enumerate() {
            let piece_pos = self.pieces_spawned as usize + i;
            if piece_pos >= bag_start && piece_pos < bag_end {
                used_in_bag |= 1 << piece_type;
            }
        }

        ALL_PIECES & !used_in_bag
    }

    pub(crate) fn spawn_piece_internal(&mut self) {
        // Ensure we have enough pieces in the queue
        self.fill_queue(7);

        let piece_type = self
            .piece_queue
            .pop_front()
            .expect("Queue should not be empty after fill_queue");
        let piece = Piece::spawn(piece_type, self.width);

        self.current_piece_bag_position = self.pieces_spawned;
        self.pieces_spawned += 1;

        self.clear_lock_delay();
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
        // Invalidate placements cache when piece changes
        self.invalidate_placement_cache();
    }

    pub(crate) fn spawn_piece_from_type(&mut self, piece_type: usize) {
        let piece = Piece::spawn(piece_type, self.width);

        self.clear_lock_delay();
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
        // Invalidate placements cache when piece changes
        self.invalidate_placement_cache();
    }
}
