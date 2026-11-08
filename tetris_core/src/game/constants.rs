//! Game constants for Tetris
//!
//! Centralized constants to avoid hardcoding values throughout the codebase.

/// Standard Tetris board width
pub const BOARD_WIDTH: usize = 10;

/// Standard Tetris board height
pub const BOARD_HEIGHT: usize = 20;

/// Number of pieces visible in the preview queue
pub const QUEUE_SIZE: usize = 5;

/// Number of tetromino piece types
pub const NUM_PIECE_TYPES: usize = 7;

/// Number of occupied cells in any tetromino
pub const MAX_PIECE_CELLS: usize = 4;

/// Maximum board cells (width * height) for fixed-size board storage
pub const MAX_BOARD_CELLS: usize = BOARD_WIDTH * BOARD_HEIGHT;

/// Combo feature scaling divisor (normalized_combo = combo / COMBO_NORMALIZATION_MAX).
/// Uses linear scaling without capping.
pub const COMBO_NORMALIZATION_MAX: u32 = 4;

// ── Empirical normalization divisors ──
// Derived from v37 training data percentiles. Replace theoretical board-maximum
// divisors with tighter empirical ones to improve gradient signal and float precision.

/// Old: (W-1)*H² = 3600. New: 200 (p99=161, p100=906; covers normal play).
pub const BUMPINESS_NORMALIZATION_DIVISOR: f32 = 200.0;

/// Old: W*(H-1) = 190. New: 20 (p99=13, p100=55; good spread for 0-13).
pub const HOLES_NORMALIZATION_DIVISOR: f32 = 20.0;

/// Old: W*(H-1) = 190. New: 25 (p99=19, p100=55).
pub const OVERHANG_NORMALIZATION_DIVISOR: f32 = 25.0;

/// Old: W*H = 200. New: 60 (p90=40, p99=110; most play is 10-50).
pub const TOTAL_BLOCKS_NORMALIZATION_DIVISOR: f32 = 60.0;

/// Old: H = 20. New: 8 (p90=5, p99=15; most columns 0-6).
pub const COLUMN_HEIGHT_NORMALIZATION_DIVISOR: f32 = 8.0;

/// Unchanged: H = 20 (already good range for max column height).
pub const MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR: f32 = 20.0;

/// Keep only the bottom N row-fill diagnostics in aux features.
pub const ROW_FILL_FEATURE_ROWS: usize = 4;

/// Piece/game auxiliary features sent to the uncached heads model:
/// current piece (7) + hold piece (8) + hold available (1) + queue (35) + placement count (1)
/// + combo (1) + back-to-back (1) + hidden-piece distribution (7).
pub const PIECE_AUX_FEATURES: usize = NUM_PIECE_TYPES
    + (NUM_PIECE_TYPES + 1)
    + 1
    + (QUEUE_SIZE * NUM_PIECE_TYPES)
    + 1
    + 1
    + 1
    + NUM_PIECE_TYPES; // 61

/// Board-derived statistics folded into the cached board embedding:
/// column heights (10) + max column height (1)
/// + bottom row fill counts (4) + total blocks (1) + bumpiness (1)
/// + holes (1) + overhang fields (1).
pub const BOARD_STATS_FEATURES: usize = BOARD_WIDTH + 1 + ROW_FILL_FEATURE_ROWS + 1 + 1 + 1 + 1; // 19

/// Full auxiliary feature vector size (training data packing).
pub const AUX_FEATURES: usize = PIECE_AUX_FEATURES + BOARD_STATS_FEATURES; // 80

/// Default lock delay in milliseconds
pub const DEFAULT_LOCK_DELAY_MS: u32 = 500;

/// Default number of moves/rotations allowed during lock delay
pub const DEFAULT_LOCK_MOVES: u32 = 15;

/// Max number of placement-cache entries in each worker thread's global cache.
pub const PLACEMENT_CACHE_MAX_ENTRIES: usize = 30_000;

/// Max number of board-embedding entries in Rust NN inference cache (per worker).
/// A 50-placement self-play game misses ~2.4-4.1K distinct boards and later games never hit
/// earlier games' entries, so 64K (~64 MB/worker) keeps the 97-98% in-game hit rate; the old
/// 1M cap only grew to ~1.2 GB per worker over a few hundred games.
pub const BOARD_CACHE_MAX_ENTRIES: usize = 65_536;

/// Piece type indices
pub const I_PIECE: usize = 0;
pub const O_PIECE: usize = 1;
pub const T_PIECE: usize = 2;
pub const S_PIECE: usize = 3;
pub const Z_PIECE: usize = 4;
pub const J_PIECE: usize = 5;
pub const L_PIECE: usize = 6;
