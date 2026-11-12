//! Fixed-size replay example storage.
//!
//! A [`TrainingExample`] carries a dense 672-wide `f32` policy and a 672-wide `bool` mask
//! plus several small heap vectors (~3.9 KB and seven allocations per example). The replay
//! buffer instead keeps [`CompactExample`]s (~390 bytes, no heap):
//!
//! - the board and the legal-action mask are bit-packed (little bit order, like
//!   `numpy.packbits(..., bitorder="little")`);
//! - the policy is stored only for legal actions, in ascending action order, as 16-bit
//!   fixed point. MCTS never puts mass on illegal actions, so the mask says which action
//!   each slot belongs to and no indices are needed.
//!
//! Positions have at most ~50 legal actions in practice, so [`POLICY_SLOTS`] = 64 makes the
//! encoding lossless up to the 16-bit quantisation (max error 7.6e-6 per probability).
//! A position with more legal actions keeps the [`POLICY_SLOTS`] highest-probability ones
//! and drops the rest from its stored mask.

use crate::game::constants::{
    BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE, ROW_FILL_FEATURE_ROWS,
};
use crate::search::{TrainingExample, NUM_ACTIONS};

pub const BOARD_CELLS: usize = BOARD_HEIGHT * BOARD_WIDTH;
pub const BOARD_BYTES: usize = BOARD_CELLS.div_ceil(8);
pub const MASK_BYTES: usize = NUM_ACTIONS.div_ceil(8);
pub const POLICY_SLOTS: usize = 64;
pub const POLICY_QUANT_SCALE: f32 = u16::MAX as f32;
/// `next_queue` slot marker for queues shorter than [`QUEUE_SIZE`].
pub const EMPTY_QUEUE_SLOT: u8 = u8::MAX;

#[derive(Clone, Debug, PartialEq)]
pub struct CompactExample {
    pub board_bits: [u8; BOARD_BYTES],
    pub current_piece: u8,
    /// Piece type, or `NUM_PIECE_TYPES` when the hold slot is empty.
    pub hold_piece: u8,
    pub hold_available: bool,
    pub back_to_back: bool,
    pub next_queue: [u8; QUEUE_SIZE],
    pub move_number: u32,
    pub placement_count: f32,
    pub combo: f32,
    pub next_hidden_piece_probs: [f32; NUM_PIECE_TYPES],
    pub column_heights: [f32; BOARD_WIDTH],
    pub max_column_height: f32,
    pub row_fill_counts: [f32; ROW_FILL_FEATURE_ROWS],
    pub total_blocks: f32,
    pub bumpiness: f32,
    pub holes: f32,
    pub overhang_fields: f32,
    pub value: f32,
    pub game_number: u64,
    pub game_total_attack: u32,
    pub mask_bits: [u8; MASK_BYTES],
    /// Policy of the i-th set bit of `mask_bits`, as `round(p * POLICY_QUANT_SCALE)`.
    pub policy_legal: [u16; POLICY_SLOTS],
}

#[inline]
pub fn get_bit(bits: &[u8], index: usize) -> bool {
    bits[index / 8] >> (index % 8) & 1 == 1
}

#[inline]
fn set_bit(bits: &mut [u8], index: usize) {
    bits[index / 8] |= 1 << (index % 8);
}

fn copy_array<const N: usize>(values: &[f32], name: &str) -> [f32; N] {
    values
        .try_into()
        .unwrap_or_else(|_| panic!("{name} must have {N} values, got {}", values.len()))
}

/// Pack a dense policy + mask into (mask bits, legal-order quantised policy).
pub fn pack_policy(policy: &[f32], mask: &[bool]) -> ([u8; MASK_BYTES], [u16; POLICY_SLOTS]) {
    assert_eq!(policy.len(), NUM_ACTIONS, "policy width");
    assert_eq!(mask.len(), NUM_ACTIONS, "mask width");
    let mut legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&i| mask[i]).collect();
    if legal.len() > POLICY_SLOTS {
        legal.sort_by(|&a, &b| policy[b].total_cmp(&policy[a]).then(a.cmp(&b)));
        legal.truncate(POLICY_SLOTS);
        legal.sort_unstable();
    }
    let mut mask_bits = [0u8; MASK_BYTES];
    let mut policy_legal = [0u16; POLICY_SLOTS];
    for (slot, &action) in legal.iter().enumerate() {
        set_bit(&mut mask_bits, action);
        policy_legal[slot] = (policy[action].clamp(0.0, 1.0) * POLICY_QUANT_SCALE).round() as u16;
    }
    (mask_bits, policy_legal)
}

/// Dense, renormalised policy and mask from the packed form.
pub fn unpack_policy(
    mask_bits: &[u8; MASK_BYTES],
    policy_legal: &[u16; POLICY_SLOTS],
) -> (Vec<f32>, Vec<bool>) {
    let mut policy = vec![0.0f32; NUM_ACTIONS];
    let mut mask = vec![false; NUM_ACTIONS];
    let total: u32 = policy_legal.iter().map(|&q| u32::from(q)).sum();
    let mut slot = 0;
    for (action, is_legal) in mask.iter_mut().enumerate() {
        if !get_bit(mask_bits, action) {
            continue;
        }
        *is_legal = true;
        if total > 0 {
            policy[action] = f32::from(policy_legal[slot]) / total as f32;
        }
        slot += 1;
    }
    (policy, mask)
}

pub fn pack_board(board: &[u8]) -> [u8; BOARD_BYTES] {
    assert_eq!(board.len(), BOARD_CELLS, "board size");
    let mut bits = [0u8; BOARD_BYTES];
    for (cell, &value) in board.iter().enumerate() {
        if value != 0 {
            set_bit(&mut bits, cell);
        }
    }
    bits
}

impl CompactExample {
    pub fn from_example(example: &TrainingExample) -> Self {
        assert!(
            example.next_queue.len() <= QUEUE_SIZE,
            "next_queue too long"
        );
        let mut next_queue = [EMPTY_QUEUE_SLOT; QUEUE_SIZE];
        for (slot, &piece) in example.next_queue.iter().enumerate() {
            next_queue[slot] = piece as u8;
        }
        let (mask_bits, policy_legal) = pack_policy(&example.policy, &example.action_mask);
        Self {
            board_bits: pack_board(&example.board),
            current_piece: example.current_piece as u8,
            hold_piece: example.hold_piece.min(NUM_PIECE_TYPES) as u8,
            hold_available: example.hold_available,
            back_to_back: example.back_to_back,
            next_queue,
            move_number: example.move_number,
            placement_count: example.placement_count,
            combo: example.combo,
            next_hidden_piece_probs: copy_array(
                &example.next_hidden_piece_probs,
                "next_hidden_piece_probs",
            ),
            column_heights: copy_array(&example.column_heights, "column_heights"),
            max_column_height: example.max_column_height,
            row_fill_counts: copy_array(&example.row_fill_counts, "row_fill_counts"),
            total_blocks: example.total_blocks,
            bumpiness: example.bumpiness,
            holes: example.holes,
            overhang_fields: example.overhang_fields,
            value: example.value,
            game_number: example.game_number,
            game_total_attack: example.game_total_attack,
            mask_bits,
            policy_legal,
        }
    }

    pub fn board(&self) -> Vec<u8> {
        (0..BOARD_CELLS)
            .map(|cell| u8::from(get_bit(&self.board_bits, cell)))
            .collect()
    }

    pub fn next_queue_pieces(&self) -> Vec<usize> {
        self.next_queue
            .iter()
            .take_while(|&&piece| piece != EMPTY_QUEUE_SLOT)
            .map(|&piece| usize::from(piece))
            .collect()
    }

    pub fn hold_piece_type(&self) -> Option<usize> {
        let hold = usize::from(self.hold_piece);
        (hold < NUM_PIECE_TYPES).then_some(hold)
    }

    pub fn to_example(&self) -> TrainingExample {
        let (policy, action_mask) = unpack_policy(&self.mask_bits, &self.policy_legal);
        TrainingExample {
            board: self.board(),
            current_piece: usize::from(self.current_piece),
            hold_piece: usize::from(self.hold_piece),
            hold_available: self.hold_available,
            next_queue: self.next_queue_pieces(),
            move_number: self.move_number,
            placement_count: self.placement_count,
            combo: self.combo,
            back_to_back: self.back_to_back,
            next_hidden_piece_probs: self.next_hidden_piece_probs.to_vec(),
            column_heights: self.column_heights.to_vec(),
            max_column_height: self.max_column_height,
            row_fill_counts: self.row_fill_counts.to_vec(),
            total_blocks: self.total_blocks,
            bumpiness: self.bumpiness,
            holes: self.holes,
            policy,
            value: self.value,
            action_mask,
            overhang_fields: self.overhang_fields,
            game_number: self.game_number,
            game_total_attack: self.game_total_attack,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_with(policy: Vec<f32>, action_mask: Vec<bool>) -> TrainingExample {
        let mut board = vec![0u8; BOARD_CELLS];
        board[0] = 1;
        board[37] = 1;
        board[BOARD_CELLS - 1] = 1;
        TrainingExample {
            board,
            current_piece: 3,
            hold_piece: NUM_PIECE_TYPES,
            hold_available: true,
            next_queue: vec![0, 6, 2],
            move_number: 12,
            placement_count: 0.24,
            combo: 0.5,
            back_to_back: true,
            next_hidden_piece_probs: vec![1.0 / 7.0; NUM_PIECE_TYPES],
            column_heights: (0..BOARD_WIDTH).map(|i| i as f32 / 20.0).collect(),
            max_column_height: 0.45,
            row_fill_counts: vec![0.1, 0.2, 0.3, 0.4],
            total_blocks: 0.3,
            bumpiness: 0.2,
            holes: 0.05,
            policy,
            value: 33.0,
            action_mask,
            overhang_fields: 0.1,
            game_number: 99,
            game_total_attack: 57,
        }
    }

    #[test]
    fn test_round_trip_keeps_every_field_and_policy_within_quantisation() {
        let mut policy = vec![0.0f32; NUM_ACTIONS];
        let mut mask = vec![false; NUM_ACTIONS];
        for (action, p) in [(3, 0.6), (100, 0.25), (671, 0.149), (500, 0.001)] {
            policy[action] = p;
            mask[action] = true;
        }
        mask[200] = true; // legal but never visited
        let example = example_with(policy.clone(), mask.clone());
        let restored = CompactExample::from_example(&example).to_example();

        assert_eq!(restored.action_mask, mask);
        for action in 0..NUM_ACTIONS {
            assert!((restored.policy[action] - policy[action]).abs() < 2e-5);
        }
        assert_eq!(restored.board, example.board);
        assert_eq!(restored.next_queue, example.next_queue);
        assert_eq!(restored.hold_piece, NUM_PIECE_TYPES);
        assert_eq!(restored.column_heights, example.column_heights);
        assert_eq!(restored.game_number, 99);
        assert!(restored.back_to_back && restored.hold_available);
    }

    #[test]
    fn test_overflow_keeps_highest_probability_legal_actions() {
        let mut policy = vec![0.0f32; NUM_ACTIONS];
        let mask = vec![true; NUM_ACTIONS];
        policy[650] = 0.5;
        policy[10] = 0.5;
        let compact = CompactExample::from_example(&example_with(policy, mask));
        let (restored_policy, restored_mask) =
            unpack_policy(&compact.mask_bits, &compact.policy_legal);

        assert_eq!(restored_mask.iter().filter(|&&m| m).count(), POLICY_SLOTS);
        assert!(restored_mask[650] && restored_mask[10]);
        assert!((restored_policy[650] - 0.5).abs() < 1e-6);
        assert!((restored_policy[10] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compact_example_is_small() {
        assert!(std::mem::size_of::<CompactExample>() <= 400);
    }
}
