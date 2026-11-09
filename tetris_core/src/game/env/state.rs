//! TetrisEnv State Definition
//!
//! Core struct definition and initialization.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::Arc;

use crate::game::constants::{
    BOARD_HEIGHT, BOARD_WIDTH, DEFAULT_LOCK_DELAY_MS, DEFAULT_LOCK_MOVES, MAX_BOARD_CELLS,
};
use crate::game::moves::PlacementParams;
use crate::game::scoring::AttackResult;

use super::piece_queue::PieceQueue;

/// Placements reachable by the current piece plus hold availability. Shared (behind one
/// `Arc`) by env clones and the per-worker placement cache.
pub(crate) struct PlacementSet {
    /// Reachable placements in ascending action-index order.
    pub placements: Vec<PlacementParams>,
    /// Valid action indices: the placement actions (ascending), then hold if available.
    pub valid_action_indices: Vec<usize>,
}

impl PlacementSet {
    pub(crate) fn placement_for_action(&self, action_idx: usize) -> Option<&PlacementParams> {
        let index = self
            .placements
            .binary_search_by_key(&action_idx, |placement| placement.action_index as usize)
            .ok()?;
        Some(&self.placements[index])
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TetrisEnv {
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
    #[pyo3(get, set)]
    pub placement_count: u32,
    #[pyo3(get)]
    pub lines_cleared: u32,
    #[pyo3(get)]
    pub game_over: bool,
    #[pyo3(get)]
    pub attack: u32,
    #[pyo3(get)]
    pub combo: u32,
    #[pyo3(get)]
    pub back_to_back: bool,
    pub(crate) board: [u8; MAX_BOARD_CELLS],
    pub(crate) board_piece_types: Vec<Option<usize>>,
    pub(crate) current_piece: Option<crate::game::piece::Piece>,
    pub(crate) piece_queue: PieceQueue,
    pub(crate) hold_piece: Option<usize>,
    pub(crate) hold_piece_bag_position: Option<u32>,
    pub(crate) hold_used: bool,
    pub(crate) lock_delay_ms: Option<u32>,
    pub(crate) lock_delay_max: u32,
    pub(crate) lock_moves_remaining: u32,
    /// Jstris `spinPossible`: a rotation happened since the piece last moved down.
    pub(crate) spin_flag: bool,
    pub(crate) last_attack_result: Option<AttackResult>,
    pub(crate) pieces_spawned: u32,
    pub(crate) current_piece_bag_position: u32,
    pub(crate) rng: StdRng,
    /// The seed used to initialize this environment's RNG (for determinism tracking)
    pub(crate) seed: u64,
    /// Total number of filled cells on the board. Used for O(1) perfect clear detection.
    pub(crate) total_blocks: u32,
    /// Number of filled cells per row. Used for O(1) line clear detection.
    /// A row is full when `row_fill_counts[y] == width`.
    pub(crate) row_fill_counts: [u8; BOARD_HEIGHT],
    /// Height per column measured from the bottom (0 for empty, up to board height).
    pub(crate) column_heights: [u8; BOARD_WIDTH],
    /// Cached placements for current piece (invalidated when piece or board changes)
    /// Using RefCell for interior mutability to cache with &self
    pub(crate) placements_cache: RefCell<Option<Arc<PlacementSet>>>,
    /// Cached board diagnostics (overhang_fields, holes), invalidated only when board changes.
    pub(crate) board_analysis_cache: RefCell<Option<(u32, u32)>>,
}

impl TetrisEnv {
    /// Create a new TetrisEnv with a specific random seed.
    pub fn new_with_seed(width: usize, height: usize, seed: u64) -> Self {
        let mut env = TetrisEnv {
            width,
            height,
            placement_count: 0,
            attack: 0,
            lines_cleared: 0,
            game_over: false,
            combo: 0,
            back_to_back: false,
            board: [0u8; MAX_BOARD_CELLS],
            board_piece_types: vec![None; width * height],
            current_piece: None,
            piece_queue: PieceQueue::default(),
            hold_piece: None,
            hold_piece_bag_position: None,
            hold_used: false,
            lock_delay_ms: None,
            lock_delay_max: DEFAULT_LOCK_DELAY_MS,
            lock_moves_remaining: DEFAULT_LOCK_MOVES,
            spin_flag: false,
            last_attack_result: None,
            pieces_spawned: 0,
            current_piece_bag_position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            total_blocks: 0,
            row_fill_counts: [0u8; BOARD_HEIGHT],
            column_heights: [0u8; BOARD_WIDTH],
            placements_cache: RefCell::new(None),
            board_analysis_cache: RefCell::new(None),
        };
        env.spawn_piece_internal();
        env
    }

    /// Reset the game with a specific random seed for reproducibility.
    pub fn reset_internal(&mut self, seed: u64) {
        self.board.fill(0);
        self.board_piece_types = vec![None; self.width * self.height];
        self.placement_count = 0;
        self.attack = 0;
        self.lines_cleared = 0;
        self.game_over = false;
        self.combo = 0;
        self.back_to_back = false;
        self.current_piece = None;
        self.piece_queue.clear();
        self.hold_piece = None;
        self.hold_piece_bag_position = None;
        self.hold_used = false;
        self.lock_delay_ms = None;
        self.lock_moves_remaining = DEFAULT_LOCK_MOVES;
        self.spin_flag = false;
        self.last_attack_result = None;
        self.pieces_spawned = 0;
        self.current_piece_bag_position = 0;
        self.rng = StdRng::seed_from_u64(seed);
        self.seed = seed;
        self.total_blocks = 0;
        self.row_fill_counts.fill(0);
        self.column_heights.fill(0);
        *self.placements_cache.borrow_mut() = None;
        *self.board_analysis_cache.borrow_mut() = None;
        self.spawn_piece_internal();
    }

    /// Lightweight clone that skips rendering-only fields (board_piece_types).
    /// Use this in MCTS search where board_piece_types is never read.
    pub(crate) fn mcts_clone(&self) -> Self {
        TetrisEnv {
            width: self.width,
            height: self.height,
            placement_count: self.placement_count,
            lines_cleared: self.lines_cleared,
            game_over: self.game_over,
            attack: self.attack,
            combo: self.combo,
            back_to_back: self.back_to_back,
            board: self.board,
            board_piece_types: Vec::new(),
            current_piece: self.current_piece.clone(),
            piece_queue: self.piece_queue.clone(),
            hold_piece: self.hold_piece,
            hold_piece_bag_position: self.hold_piece_bag_position,
            hold_used: self.hold_used,
            lock_delay_ms: self.lock_delay_ms,
            lock_delay_max: self.lock_delay_max,
            lock_moves_remaining: self.lock_moves_remaining,
            spin_flag: self.spin_flag,
            last_attack_result: self.last_attack_result.clone(),
            pieces_spawned: self.pieces_spawned,
            current_piece_bag_position: self.current_piece_bag_position,
            rng: self.rng.clone(),
            seed: self.seed,
            total_blocks: self.total_blocks,
            row_fill_counts: self.row_fill_counts,
            column_heights: self.column_heights,
            placements_cache: self.placements_cache.clone(),
            board_analysis_cache: self.board_analysis_cache.clone(),
        }
    }

    /// Invalidate the placements cache (call when board or piece changes)
    #[inline]
    pub(crate) fn invalidate_placement_cache(&self) {
        *self.placements_cache.borrow_mut() = None;
    }

    /// Invalidate cached board analysis metrics (overhang fields and holes).
    #[inline]
    pub(crate) fn invalidate_board_analysis_cache(&self) {
        *self.board_analysis_cache.borrow_mut() = None;
    }

    #[inline]
    pub(crate) fn get_cached_overhang_fields_and_holes(&self) -> Option<(u32, u32)> {
        *self.board_analysis_cache.borrow()
    }

    #[inline]
    pub(crate) fn set_cached_overhang_fields_and_holes(&self, value: (u32, u32)) {
        *self.board_analysis_cache.borrow_mut() = Some(value);
    }
}
