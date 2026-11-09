//! Move Generation Module
//!
//! This module provides functionality to find all possible piece placements
//! and the move sequences required to reach them. This is useful for AI/MCTS
//! planning and move evaluation.

use pyo3::prelude::*;
use std::collections::{HashSet, VecDeque};

use crate::game::action_space::get_action_space;
use crate::game::constants::{I_PIECE, O_PIECE, S_PIECE, T_PIECE, Z_PIECE};
use crate::game::kicks::{get_kicks, Rotation};
use crate::game::piece::{get_cells, Piece, TETROMINO_CELLS};
use crate::game::tspin::classify_tspin;

/// Actions that can be taken during piece movement. Codes match `TetrisEnv::step`
/// (which also uses 7 for hold).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Left = 1,
    Right = 2,
    Down = 3,
    RotateCW = 4,
    RotateCCW = 5,
    HardDrop = 6,
    Rotate180 = 8,
}

impl Action {
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// A piece state during pathfinding (position + rotation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PieceState {
    x: i32,
    y: i32,
    rotation: usize,
}

const X_MIN: i32 = -3;
const X_STATES: usize = 16; // x in [-3, 12]
const Y_MIN: i32 = -3;
const Y_STATES: usize = 26; // y in [-3, 22]
const ROT_STATES: usize = 4;
const X_STRIDE: usize = Y_STATES * ROT_STATES;
const Y_STRIDE: usize = ROT_STATES;
const NUM_STATE_INDICES: usize = X_STATES * Y_STATES * ROT_STATES; // 1,664
const STATE_WORDS: usize = NUM_STATE_INDICES.div_ceil(64); // 26
/// BFS nodes pair a piece state with the spin flag (a rotation since the last downward
/// move), so a position reachable both with and without a pending spin keeps both paths.
const NUM_BFS_NODES: usize = NUM_STATE_INDICES * 2;
const NODE_WORDS: usize = NUM_BFS_NODES.div_ceil(64); // 52
/// Largest board the BFS state space covers (x up to 12, y up to 22).
const MAX_STATE_BOARD_WIDTH: usize = (X_MIN + X_STATES as i32) as usize;
const MAX_STATE_BOARD_HEIGHT: usize = (Y_MIN + Y_STATES as i32) as usize;
const ALL_Y_STATES: u32 = (1 << Y_STATES) - 1;
/// Widest board whose column occupancy masks are kept inline in `Board`.
const MAX_MASKED_COLUMNS: usize = 16;

/// A possible placement for a piece, including the move sequence to reach it
#[pyclass]
#[derive(Debug, Clone)]
pub struct Placement {
    /// The final piece position (after hard drop)
    #[pyo3(get)]
    pub piece: Piece,
    /// The sequence of moves to reach this placement (action codes)
    #[pyo3(get)]
    pub moves: Vec<u8>,
    /// The column where the piece lands (leftmost cell x coordinate)
    #[pyo3(get)]
    pub column: i32,
    /// The rotation state (0-3)
    #[pyo3(get)]
    pub rotation: usize,
    /// Whether the piece locks with the spin flag set: `moves` rotate after the last
    /// downward move and the final hard drop does not move the piece.
    #[pyo3(get)]
    pub spin_flag: bool,
    /// Pre-computed placement action index for fast lookup (0..NUM_PLACEMENT_ACTIONS-1)
    #[pyo3(get)]
    pub action_index: usize,
}

/// Compact placement metadata used by MCTS action execution/cache paths (6 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementParams {
    pub x: i8,
    pub y: i8,
    pub rotation: u8,
    /// See `Placement::spin_flag`.
    pub spin_flag: bool,
    pub action_index: u16,
}

/// Board representation for collision checking (borrows cells to avoid cloning)
pub struct Board<'a> {
    width: usize,
    width_u32: u32,
    height_u32: u32,
    cells: &'a [u8],
    /// Per-column occupancy bitmasks (bit `y` set = cell occupied), when the board fits.
    column_masks: Option<[u32; MAX_MASKED_COLUMNS]>,
}

impl<'a> Board<'a> {
    pub fn new(width: usize, height: usize, cells: &'a [u8]) -> Self {
        debug_assert!(width <= u32::MAX as usize);
        debug_assert!(height <= u32::MAX as usize);
        let column_masks = if height <= u32::BITS as usize && width <= MAX_MASKED_COLUMNS {
            let mut masks = [0u32; MAX_MASKED_COLUMNS];
            for y in 0..height {
                let row = &cells[y * width..(y + 1) * width];
                let row_bit = 1u32 << (y as u32);
                for (x, &cell) in row.iter().enumerate() {
                    if cell != 0 {
                        masks[x] |= row_bit;
                    }
                }
            }
            Some(masks)
        } else {
            None
        };
        Board {
            width,
            width_u32: width as u32,
            height_u32: height as u32,
            cells,
            column_masks,
        }
    }

    /// T-spin status of `piece_type` locking at `state` with the spin flag set.
    fn locks_as_tspin(&self, piece_type: usize, state: &PieceState) -> bool {
        classify_tspin(
            self.cells,
            self.width,
            self.height_u32 as usize,
            piece_type,
            state.x,
            state.y,
            state.rotation,
            true,
        )
        .is_spin()
    }

    /// Check if a piece at the given state is in a valid position
    fn is_valid_position(&self, piece_type: usize, state: &PieceState) -> bool {
        self.is_valid_position_at(piece_type, state.rotation, state.x, state.y)
    }

    /// Get the final y position after hard dropping from a state
    fn get_drop_y(&self, piece_type: usize, state: &PieceState) -> i32 {
        if let Some(column_masks) = &self.column_masks {
            let offsets = &TETROMINO_CELLS[piece_type][state.rotation];
            let mut min_drop = i32::MAX;
            for &(dx, dy) in offsets.iter() {
                let cx = state.x + dx as i32;
                let cy = state.y + dy as i32;
                debug_assert!((cx as u32) < self.width_u32);
                debug_assert!((cy as u32) < self.height_u32);

                let search_start = cy as u32 + 1;
                let next_occupied_row = if search_start >= self.height_u32 {
                    self.height_u32 as i32
                } else {
                    let shifted = column_masks[cx as usize] >> search_start;
                    if shifted == 0 {
                        self.height_u32 as i32
                    } else {
                        (search_start + shifted.trailing_zeros()) as i32
                    }
                };
                let cell_drop = next_occupied_row - 1 - cy;
                min_drop = min_drop.min(cell_drop);
            }
            debug_assert!(min_drop >= 0);
            return state.y + min_drop;
        }

        // Intentionally step downward with exact collision checks.
        // In move-generation BFS we evaluate arbitrary intermediate states
        // (including kick/slide positions under overhangs), where a simple
        // column-height profile shortcut is not always correct.
        let mut y = state.y;
        while self.is_valid_position_at(piece_type, state.rotation, state.x, y + 1) {
            y += 1;
        }
        y
    }

    #[inline]
    fn is_valid_position_at(&self, piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let [(dx0, dy0), (dx1, dy1), (dx2, dy2), (dx3, dy3)] =
            TETROMINO_CELLS[piece_type][rotation];
        if let Some(column_masks) = &self.column_masks {
            let cx0 = x + dx0 as i32;
            let cy0 = y + dy0 as i32;
            if (cx0 as u32) >= self.width_u32 || (cy0 as u32) >= self.height_u32 {
                return false;
            }
            if ((column_masks[cx0 as usize] >> cy0 as u32) & 1) != 0 {
                return false;
            }

            let cx1 = x + dx1 as i32;
            let cy1 = y + dy1 as i32;
            if (cx1 as u32) >= self.width_u32 || (cy1 as u32) >= self.height_u32 {
                return false;
            }
            if ((column_masks[cx1 as usize] >> cy1 as u32) & 1) != 0 {
                return false;
            }

            let cx2 = x + dx2 as i32;
            let cy2 = y + dy2 as i32;
            if (cx2 as u32) >= self.width_u32 || (cy2 as u32) >= self.height_u32 {
                return false;
            }
            if ((column_masks[cx2 as usize] >> cy2 as u32) & 1) != 0 {
                return false;
            }

            let cx3 = x + dx3 as i32;
            let cy3 = y + dy3 as i32;
            if (cx3 as u32) >= self.width_u32 || (cy3 as u32) >= self.height_u32 {
                return false;
            }
            if ((column_masks[cx3 as usize] >> cy3 as u32) & 1) != 0 {
                return false;
            }

            return true;
        }

        let cells_ptr = self.cells.as_ptr();

        let cx0 = x + dx0 as i32;
        let cy0 = y + dy0 as i32;
        if (cx0 as u32) >= self.width_u32 || (cy0 as u32) >= self.height_u32 {
            return false;
        }
        let idx0 = cy0 as usize * self.width + cx0 as usize;
        if unsafe { *cells_ptr.add(idx0) } != 0 {
            return false;
        }

        let cx1 = x + dx1 as i32;
        let cy1 = y + dy1 as i32;
        if (cx1 as u32) >= self.width_u32 || (cy1 as u32) >= self.height_u32 {
            return false;
        }
        let idx1 = cy1 as usize * self.width + cx1 as usize;
        if unsafe { *cells_ptr.add(idx1) } != 0 {
            return false;
        }

        let cx2 = x + dx2 as i32;
        let cy2 = y + dy2 as i32;
        if (cx2 as u32) >= self.width_u32 || (cy2 as u32) >= self.height_u32 {
            return false;
        }
        let idx2 = cy2 as usize * self.width + cx2 as usize;
        if unsafe { *cells_ptr.add(idx2) } != 0 {
            return false;
        }

        let cx3 = x + dx3 as i32;
        let cy3 = y + dy3 as i32;
        if (cx3 as u32) >= self.width_u32 || (cy3 as u32) >= self.height_u32 {
            return false;
        }
        let idx3 = cy3 as usize * self.width + cx3 as usize;
        if unsafe { *cells_ptr.add(idx3) } != 0 {
            return false;
        }

        true
    }
}

/// Try to rotate a piece and return the new state if successful
fn try_rotate(
    board: &Board<'_>,
    piece_type: usize,
    state: &PieceState,
    rotation: Rotation,
) -> Option<PieceState> {
    let to_state = rotation.target(state.rotation);
    get_kicks(piece_type, state.rotation, rotation)
        .iter()
        .map(|&(dx, dy)| PieceState {
            x: state.x + dx,
            y: state.y + dy,
            rotation: to_state,
        })
        .find(|candidate| board.is_valid_position(piece_type, candidate))
}

/// Convert piece state to bitset index
#[inline(always)]
fn state_to_index(state: &PieceState) -> usize {
    debug_assert!(state.x >= X_MIN && state.x < X_MIN + X_STATES as i32);
    debug_assert!(state.y >= Y_MIN && state.y < Y_MIN + Y_STATES as i32);
    debug_assert!(state.rotation < ROT_STATES);

    let x_offset = (state.x - X_MIN) as usize;
    let y_offset = (state.y - Y_MIN) as usize;
    x_offset * X_STRIDE + y_offset * Y_STRIDE + state.rotation
}

#[inline]
fn index_to_state(idx: usize) -> PieceState {
    let x_offset = idx / X_STRIDE;
    let rem = idx % X_STRIDE;
    let y_offset = rem / ROT_STATES;
    let rotation = rem % ROT_STATES;
    PieceState {
        x: X_MIN + x_offset as i32,
        y: Y_MIN + y_offset as i32,
        rotation,
    }
}

/// A movement input explored by the placement BFS.
#[derive(Debug, Clone, Copy)]
enum Move {
    Shift(i32),
    Down,
    Rotate(Rotation),
}

/// BFS transition order, shared by both placement searches.
const MOVES: [(Action, Move); 6] = [
    (Action::Left, Move::Shift(-1)),
    (Action::Right, Move::Shift(1)),
    (Action::Down, Move::Down),
    (Action::RotateCW, Move::Rotate(Rotation::Clockwise)),
    (Action::RotateCCW, Move::Rotate(Rotation::CounterClockwise)),
    (Action::Rotate180, Move::Rotate(Rotation::Half)),
];

#[inline(always)]
fn node_index(state_idx: usize, spin: bool) -> usize {
    state_idx * 2 + spin as usize
}

/// Spin flag after a move (Jstris): rotations set it, downward moves clear it, sideways
/// moves keep it. Only T pieces can spin, so other pieces never set it.
#[inline(always)]
fn spin_after(mv: Move, spin: bool, tracks_spin: bool) -> bool {
    match mv {
        Move::Shift(_) => spin,
        Move::Down => false,
        Move::Rotate(_) => tracks_spin,
    }
}

#[inline(always)]
fn test_and_set(bits: &mut [u64], idx: usize) -> bool {
    let mask = 1u64 << (idx % 64);
    let was_set = bits[idx / 64] & mask != 0;
    bits[idx / 64] |= mask;
    was_set
}

#[inline(always)]
fn is_set(bits: &[u64], idx: usize) -> bool {
    bits[idx / 64] & (1u64 << (idx % 64)) != 0
}

/// Which lock path to report for a final position, as its spin flag. A T-spin never
/// scores worse than the same clear without one under Jstris rules (more attack, or
/// keeps back-to-back), so the spin path wins whenever the corners make it a T-spin.
/// Otherwise prefer the path without a pending spin. `None` if the position is unreachable.
fn chosen_lock_spin(
    board: &Board<'_>,
    piece_type: usize,
    final_state: &PieceState,
    reachable_plain: bool,
    reachable_spin: bool,
) -> Option<bool> {
    match (reachable_plain, reachable_spin) {
        (false, false) => None,
        (false, true) => Some(true),
        (true, false) => Some(false),
        (true, true) => Some(board.locks_as_tspin(piece_type, final_state)),
    }
}

/// Packed occupied-cell key used to deduplicate placements that cover the same cells.
fn cell_key(board: &Board<'_>, piece_type: usize, state: &PieceState) -> u64 {
    let mut packed_cells = [0u16; 4];
    for (cell_idx, (cx, cy)) in get_cells(piece_type, state.rotation, state.x, state.y)
        .into_iter()
        .enumerate()
    {
        let packed = cy as usize * board.width + cx as usize;
        debug_assert!(packed <= u16::MAX as usize);
        packed_cells[cell_idx] = packed as u16;
    }
    packed_cells.sort_unstable();
    ((packed_cells[0] as u64) << 48)
        | ((packed_cells[1] as u64) << 32)
        | ((packed_cells[2] as u64) << 16)
        | packed_cells[3] as u64
}

/// Find all possible placements for a piece on the given board
///
/// Uses BFS over (position, spin flag) from the spawn position, then returns every
/// unique final position (after hard drop) with the shortest move sequence that reaches
/// it, using a T-spin path when one exists (see `chosen_lock_spin`).
pub fn find_all_placements(
    board: &Board<'_>,
    piece_type: usize,
    spawn_x: i32,
    spawn_y: i32,
) -> Vec<Placement> {
    let start_state = PieceState {
        x: spawn_x,
        y: spawn_y,
        rotation: 0,
    };

    // Check if spawn position is valid
    if !board.is_valid_position(piece_type, &start_state) {
        return Vec::new();
    }
    let tracks_spin = piece_type == T_PIECE;

    let mut visited = [0u64; NODE_WORDS];
    let mut queue: VecDeque<(PieceState, bool)> = VecDeque::with_capacity(128);

    // Parent-pointer BFS metadata per reachable node.
    // This avoids cloning move sequences at each edge expansion.
    let mut parents = [usize::MAX; NUM_BFS_NODES];
    let mut action_from_parent = [0u8; NUM_BFS_NODES];
    // First (so shortest-path) BFS node whose hard drop locks at each final node.
    let mut lock_source = [usize::MAX; NUM_BFS_NODES];

    let start_node = node_index(state_to_index(&start_state), false);
    test_and_set(&mut visited, start_node);
    parents[start_node] = start_node;
    queue.push_back((start_state, false));

    while let Some((state, spin)) = queue.pop_front() {
        let current = node_index(state_to_index(&state), spin);

        let final_state = PieceState {
            y: board.get_drop_y(piece_type, &state),
            ..state
        };
        let locked_spin = spin && final_state.y == state.y;
        let final_node = node_index(state_to_index(&final_state), locked_spin);
        if lock_source[final_node] == usize::MAX {
            lock_source[final_node] = current;
        }

        for &(action, mv) in &MOVES {
            let next = match mv {
                Move::Shift(dx) => Some(PieceState {
                    x: state.x + dx,
                    ..state
                })
                .filter(|next| board.is_valid_position(piece_type, next)),
                Move::Down => Some(PieceState {
                    y: state.y + 1,
                    ..state
                })
                .filter(|next| board.is_valid_position(piece_type, next)),
                Move::Rotate(rotation) => try_rotate(board, piece_type, &state, rotation),
            };
            let Some(next) = next else {
                continue;
            };
            let next_spin = spin_after(mv, spin, tracks_spin);
            let next_node = node_index(state_to_index(&next), next_spin);
            if test_and_set(&mut visited, next_node) {
                continue;
            }
            parents[next_node] = current;
            action_from_parent[next_node] = action.to_u8();
            queue.push_back((next, next_spin));
        }
    }

    // Deduplicate by actual cells occupied (not just x, y, rotation).
    // This handles cases like O piece where different rotations look identical.
    let mut seen_cells: HashSet<u64> = HashSet::new();
    let mut placements: Vec<Placement> = Vec::new();
    let action_space = get_action_space();

    for final_idx in 0..NUM_STATE_INDICES {
        let plain_source = lock_source[node_index(final_idx, false)];
        let spin_source = lock_source[node_index(final_idx, true)];
        let final_state = index_to_state(final_idx);
        let Some(spin_flag) = chosen_lock_spin(
            board,
            piece_type,
            &final_state,
            plain_source != usize::MAX,
            spin_source != usize::MAX,
        ) else {
            continue;
        };
        let source = if spin_flag { spin_source } else { plain_source };

        if !seen_cells.insert(cell_key(board, piece_type, &final_state)) {
            continue;
        }

        let mut moves = Vec::new();
        let mut cursor = source;
        while cursor != start_node {
            moves.push(action_from_parent[cursor]);
            cursor = parents[cursor];
        }
        moves.reverse();
        moves.push(Action::HardDrop.to_u8());

        let PieceState { x, y, rotation } = final_state;
        let Some(action_index) = action_space.placement_to_index(piece_type, x, y, rotation) else {
            debug_assert!(
                false,
                "BUG: valid placement ({}, {}, {}) missing from action space",
                x, y, rotation
            );
            continue;
        };

        placements.push(Placement {
            piece: Piece::with_position(piece_type, x, y, rotation),
            moves,
            column: x,
            rotation,
            spin_flag,
            action_index,
        });
    }

    // Sort by rotation, column, then y for fully deterministic ordering
    placements.sort_unstable_by(|a, b| {
        a.rotation
            .cmp(&b.rotation)
            .then_with(|| a.column.cmp(&b.column))
            .then_with(|| a.piece.y.cmp(&b.piece.y))
    });

    placements
}

/// Valid positions of one piece type on one board as bitmasks over y: bit `y - Y_MIN` of
/// `valid[rotation][x - X_MIN]` is set iff the piece fits at `(x, y)` (every cell inside the
/// board and empty). One mask build replaces per-cell collision checks during the BFS.
struct PieceFitMasks {
    valid: [[u32; X_STATES]; ROT_STATES],
}

impl PieceFitMasks {
    fn new(board: &Board<'_>, piece_type: usize) -> Self {
        let column_masks = board
            .column_masks
            .as_ref()
            .expect("BFS boards always have inline column masks");
        let height_mask = (1u32 << board.height_u32) - 1;
        let mut free_rows = [0u32; MAX_MASKED_COLUMNS];
        for (free, &occupied) in free_rows.iter_mut().zip(column_masks).take(board.width) {
            *free = !occupied & height_mask;
        }

        let mut valid = [[0u32; X_STATES]; ROT_STATES];
        for (rotation, row) in valid.iter_mut().enumerate() {
            let offsets = &TETROMINO_CELLS[piece_type][rotation];
            for (x_offset, fit_mask) in row.iter_mut().enumerate() {
                let x = X_MIN + x_offset as i32;
                let mut mask = ALL_Y_STATES;
                for &(dx, dy) in offsets {
                    let column = x + dx as i32;
                    if column < 0 || column >= board.width as i32 {
                        mask = 0;
                        break;
                    }
                    // Bit j of this cell's mask is board row `j + Y_MIN + dy`.
                    let shift = -(Y_MIN + dy as i32);
                    debug_assert!((0..32).contains(&shift));
                    mask &= free_rows[column as usize] << shift;
                }
                *fit_mask = mask;
            }
        }
        PieceFitMasks { valid }
    }

    #[inline(always)]
    fn fits(&self, rotation: usize, x: i32, y: i32) -> bool {
        let x_offset = (x - X_MIN) as u32;
        let y_offset = (y - Y_MIN) as u32;
        x_offset < X_STATES as u32
            && y_offset < Y_STATES as u32
            && (self.valid[rotation][x_offset as usize] >> y_offset) & 1 != 0
    }

    /// Final y after hard-dropping a piece that fits at `(x, y)`.
    #[inline(always)]
    fn drop_y(&self, rotation: usize, x: i32, y: i32) -> i32 {
        let fits_from_y = self.valid[rotation][(x - X_MIN) as usize] >> (y - Y_MIN);
        debug_assert!(fits_from_y & 1 != 0, "drop_y called on a blocked position");
        y + fits_from_y.trailing_ones() as i32 - 1
    }
}

/// Find all possible placements needed by MCTS.
///
/// This variant skips move-sequence reconstruction and returns only placement
/// parameters used by action execution and valid-action masking, sorted by
/// ascending action index. It explores the same (position, spin flag) graph and applies
/// the same lock-path choice as `find_all_placements`, with collision checks answered
/// from precomputed `PieceFitMasks`.
pub fn find_all_placement_params(
    board: &Board<'_>,
    piece_type: usize,
    spawn_x: i32,
    spawn_y: i32,
) -> Vec<PlacementParams> {
    assert!(
        board.width <= MAX_STATE_BOARD_WIDTH && board.height_u32 as usize <= MAX_STATE_BOARD_HEIGHT,
        "placement BFS supports boards up to {}x{}, got {}x{}",
        MAX_STATE_BOARD_WIDTH,
        MAX_STATE_BOARD_HEIGHT,
        board.width,
        board.height_u32
    );
    let fit = PieceFitMasks::new(board, piece_type);

    let start_state = PieceState {
        x: spawn_x,
        y: spawn_y,
        rotation: 0,
    };

    if !fit.fits(0, spawn_x, spawn_y) {
        return Vec::new();
    }
    let tracks_spin = piece_type == T_PIECE;

    // kicks[from_rotation][rotation as usize]
    let kicks: [[&[(i32, i32)]; 3]; ROT_STATES] = std::array::from_fn(|from| {
        Rotation::ALL.map(|rotation| get_kicks(piece_type, from, rotation))
    });

    let mut visited = [0u64; NODE_WORDS];
    let mut locks = [0u64; NODE_WORDS];
    let mut final_seen = [0u64; STATE_WORDS];
    let mut queue: Vec<(PieceState, usize, bool)> = Vec::with_capacity(256);
    let mut queue_head = 0usize;
    let mut final_candidate_indices: Vec<usize> = Vec::with_capacity(128);

    let start_idx = state_to_index(&start_state);
    test_and_set(&mut visited, node_index(start_idx, false));
    queue.push((start_state, start_idx, false));

    while queue_head < queue.len() {
        let (state, state_idx, spin) = queue[queue_head];
        queue_head += 1;

        let final_y = fit.drop_y(state.rotation, state.x, state.y);
        let final_idx = ((state.x - X_MIN) as usize) * X_STRIDE
            + ((final_y - Y_MIN) as usize) * Y_STRIDE
            + state.rotation;
        test_and_set(
            &mut locks,
            node_index(final_idx, spin && final_y == state.y),
        );
        if !test_and_set(&mut final_seen, final_idx) {
            final_candidate_indices.push(final_idx);
        }

        for &(_, mv) in &MOVES {
            let (candidate, idx) = match mv {
                Move::Shift(_) | Move::Down => {
                    let (dx, dy, idx) = match mv {
                        Move::Shift(-1) => (-1, 0, state_idx.wrapping_sub(X_STRIDE)),
                        Move::Shift(_) => (1, 0, state_idx + X_STRIDE),
                        _ => (0, 1, state_idx + Y_STRIDE),
                    };
                    let candidate = PieceState {
                        x: state.x + dx,
                        y: state.y + dy,
                        rotation: state.rotation,
                    };
                    if !fit.fits(candidate.rotation, candidate.x, candidate.y) {
                        continue;
                    }
                    (candidate, idx)
                }
                Move::Rotate(rotation) => {
                    let to_rotation = rotation.target(state.rotation);
                    let rotation_kicks = kicks[state.rotation][rotation as usize];
                    let Some(&(dx, dy)) = rotation_kicks
                        .iter()
                        .find(|&&(dx, dy)| fit.fits(to_rotation, state.x + dx, state.y + dy))
                    else {
                        continue;
                    };
                    let candidate = PieceState {
                        x: state.x + dx,
                        y: state.y + dy,
                        rotation: to_rotation,
                    };
                    (candidate, state_to_index(&candidate))
                }
            };

            let next_spin = spin_after(mv, spin, tracks_spin);
            if test_and_set(&mut visited, node_index(idx, next_spin)) {
                continue;
            }
            queue.push((candidate, idx, next_spin));
        }
    }

    final_candidate_indices.sort_unstable();

    let dedup_by_cells = matches!(piece_type, I_PIECE | O_PIECE | S_PIECE | Z_PIECE);
    let mut seen_cells = [0u64; crate::search::NUM_PLACEMENT_ACTIONS];
    let mut seen_cells_len = 0usize;
    let mut placements: Vec<PlacementParams> = Vec::with_capacity(final_candidate_indices.len());
    let action_space = get_action_space();

    for &final_idx in &final_candidate_indices {
        let final_state = index_to_state(final_idx);
        let spin_flag = chosen_lock_spin(
            board,
            piece_type,
            &final_state,
            is_set(&locks, node_index(final_idx, false)),
            is_set(&locks, node_index(final_idx, true)),
        )
        .expect("every final candidate was locked by at least one BFS node");

        if dedup_by_cells {
            let key = cell_key(board, piece_type, &final_state);
            if seen_cells[..seen_cells_len].contains(&key) {
                continue;
            }
            debug_assert!(
                seen_cells_len < seen_cells.len(),
                "Seen-cell cache overflow while deduplicating placements"
            );
            if seen_cells_len >= seen_cells.len() {
                continue;
            }
            seen_cells[seen_cells_len] = key;
            seen_cells_len += 1;
        }

        let PieceState { x, y, rotation } = final_state;
        let Some(action_index) = action_space.placement_to_index(piece_type, x, y, rotation) else {
            debug_assert!(
                false,
                "BUG: valid placement ({}, {}, {}) missing from action space",
                x, y, rotation
            );
            continue;
        };
        placements.push(PlacementParams {
            x: x as i8,
            y: y as i8,
            rotation: rotation as u8,
            spin_flag,
            action_index: action_index as u16,
        });
    }

    placements.sort_unstable_by_key(|placement| placement.action_index);
    debug_assert!(
        placements
            .windows(2)
            .all(|pair| pair[0].action_index < pair[1].action_index),
        "Duplicate action index in placement params"
    );
    placements
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn empty_cells(width: usize, height: usize) -> Vec<u8> {
        vec![0u8; width * height]
    }

    #[test]
    fn test_find_placements_empty_board() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        // I piece (type 0)
        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(!placements.is_empty());

        // Should have placements in different columns and rotations
        let columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(columns.len() > 1);
    }

    #[test]
    fn test_find_placements_all_pieces() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        for piece_type in 0..7 {
            let placements = find_all_placements(&board, piece_type, 3, 0);
            assert!(
                !placements.is_empty(),
                "Piece type {} should have placements",
                piece_type
            );
        }
    }

    #[test]
    fn test_find_placement_params_matches_full_placements() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);

        for piece_type in 0..7 {
            let full = find_all_placements(&board, piece_type, 3, 0);
            let params = find_all_placement_params(&board, piece_type, 3, 0);

            let full_by_action: HashMap<usize, &Placement> = full
                .iter()
                .map(|placement| (placement.action_index, placement))
                .collect();
            assert_eq!(
                full_by_action.len(),
                full.len(),
                "Full placements should not contain duplicate action indices"
            );
            assert_eq!(
                params.len(),
                full.len(),
                "Fast params count should match full placement count for piece {}",
                piece_type
            );

            for param in params {
                let full_match = full_by_action
                    .get(&(param.action_index as usize))
                    .copied()
                    .expect("Fast params produced an unknown action index");
                assert_eq!(param.x as i32, full_match.piece.x);
                assert_eq!(param.y as i32, full_match.piece.y);
                assert_eq!(param.rotation as usize, full_match.piece.rotation);
                assert_eq!(param.spin_flag, full_match.spin_flag);
            }
        }
    }

    /// Boards with ragged stacks, overhangs and sealed holes, from a fixed-seed generator.
    fn random_boards(count: usize, seed: u64) -> Vec<Vec<u8>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let mut cells = empty_cells(10, 20);
                let max_height = rng.gen_range(0..=17);
                for x in 0..10 {
                    let height = rng.gen_range(0..=max_height);
                    for y in (20 - height)..20 {
                        if rng.gen_bool(0.85) {
                            cells[y * 10 + x] = 1;
                        }
                    }
                }
                // Floating blocks create overhangs / tuck spots reachable only by sliding.
                for _ in 0..rng.gen_range(0..6) {
                    let x = rng.gen_range(0..10);
                    let y = rng.gen_range(3..20);
                    cells[y * 10 + x] = 1;
                }
                cells
            })
            .collect()
    }

    #[test]
    fn test_find_placement_params_matches_full_placements_on_random_boards() {
        for cells in random_boards(1500, 0xC0FFEE) {
            let board = Board::new(10, 20, &cells);
            for piece_type in 0..7 {
                for (spawn_x, spawn_y) in [(3, 0), (3, -1), (4, 2), (0, 1)] {
                    let full = find_all_placements(&board, piece_type, spawn_x, spawn_y);
                    let params = find_all_placement_params(&board, piece_type, spawn_x, spawn_y);
                    let mut full_by_action: Vec<(usize, i32, i32, usize, bool)> = full
                        .iter()
                        .map(|p| {
                            (
                                p.action_index,
                                p.piece.x,
                                p.piece.y,
                                p.piece.rotation,
                                p.spin_flag,
                            )
                        })
                        .collect();
                    full_by_action.sort_unstable();
                    let fast: Vec<(usize, i32, i32, usize, bool)> = params
                        .iter()
                        .map(|p| {
                            (
                                p.action_index as usize,
                                p.x as i32,
                                p.y as i32,
                                p.rotation as usize,
                                p.spin_flag,
                            )
                        })
                        .collect();
                    assert_eq!(
                        fast, full_by_action,
                        "piece {piece_type} from ({spawn_x}, {spawn_y}) on board {cells:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_placements_have_moves() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        for placement in &placements {
            // Every placement should end with hard drop
            assert!(!placement.moves.is_empty(), "Placement should have moves");
            assert_eq!(
                *placement.moves.last().unwrap(),
                Action::HardDrop.to_u8(),
                "Last move should be hard drop"
            );
        }
    }

    #[test]
    fn test_placements_reach_bottom() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 0, 3, 0); // I piece

        for placement in &placements {
            // On empty board, pieces should reach near bottom
            // I piece horizontal is at row 1 in its 4x4 grid, so y=18 means cells at y=19
            assert!(
                placement.piece.y >= 16,
                "Piece should be near bottom, got y={}",
                placement.piece.y
            );
        }
    }

    #[test]
    fn test_placements_unique() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        let mut positions: HashSet<(i32, i32, usize)> = HashSet::new();
        for placement in &placements {
            let key = (
                placement.piece.x,
                placement.piece.y,
                placement.piece.rotation,
            );
            assert!(
                positions.insert(key),
                "Duplicate placement found: {:?}",
                key
            );
        }
    }

    #[test]
    fn test_i_piece_horizontal_positions() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 0, 3, 0); // I piece

        // I piece horizontal (rotation 0 and 2) can be at columns -1 to 6 (piece x position)
        // because the I piece extends 4 cells from x
        let horizontal_placements: Vec<_> = placements
            .iter()
            .filter(|p| p.rotation == 0 || p.rotation == 2)
            .collect();

        assert!(!horizontal_placements.is_empty());
    }

    #[test]
    fn test_o_piece_no_rotation_change() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 1, 3, 0); // O piece

        // O piece looks the same in all rotations, so we should have
        // placements that effectively only differ by column
        let unique_columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(!unique_columns.is_empty());
    }

    #[test]
    fn test_blocked_board() {
        // Create a board with a wall blocking most positions
        let mut cells = vec![0u8; 10 * 20];
        // Fill rows 10-19 except column 0
        for y in 10..20 {
            for x in 1..10 {
                cells[y * 10 + x] = 1;
            }
        }
        let board = Board::new(10, 20, &cells);

        // I piece can only fit in column 0 when vertical
        let placements = find_all_placements(&board, 0, 3, 0);

        // Should still find some placements (vertical I in column 0)
        // or placements above the filled area
        assert!(!placements.is_empty());
    }

    #[test]
    fn test_spawn_blocked() {
        // Create a board where spawn is blocked
        let mut cells = vec![0u8; 10 * 20];
        for x in 0..10 {
            cells[0 * 10 + x] = 1;
            cells[1 * 10 + x] = 1;
        }
        let board = Board::new(10, 20, &cells);

        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(
            placements.is_empty(),
            "Should have no placements when spawn is blocked"
        );
    }

    #[test]
    fn test_move_sequence_valid() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        for placement in &placements {
            for &action in &placement.moves {
                assert!(
                    (1..=6).contains(&action) || action == Action::Rotate180.to_u8(),
                    "Invalid action code: {}",
                    action
                );
            }
        }
    }

    #[test]
    fn test_action_enum() {
        assert_eq!(Action::Left.to_u8(), 1);
        assert_eq!(Action::Right.to_u8(), 2);
        assert_eq!(Action::Down.to_u8(), 3);
        assert_eq!(Action::RotateCW.to_u8(), 4);
        assert_eq!(Action::RotateCCW.to_u8(), 5);
        assert_eq!(Action::HardDrop.to_u8(), 6);
        assert_eq!(Action::Rotate180.to_u8(), 8);
    }

    #[test]
    fn test_placements_sorted() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        // Check that placements are sorted by rotation first, then column
        for i in 1..placements.len() {
            let prev = &placements[i - 1];
            let curr = &placements[i];
            assert!(
                prev.rotation < curr.rotation
                    || (prev.rotation == curr.rotation && prev.column <= curr.column),
                "Placements not sorted: {:?} should come before {:?}",
                prev,
                curr
            );
        }
    }

    #[test]
    fn test_t_piece_rotations() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        // T piece should have placements in all 4 rotations
        let rotations: HashSet<usize> = placements.iter().map(|p| p.rotation).collect();
        assert_eq!(rotations.len(), 4, "T piece should have all 4 rotations");
    }

    #[test]
    fn test_empty_board_placements_have_no_pending_spin() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        for piece_type in 0..7 {
            for placement in find_all_placements(&board, piece_type, 3, 0) {
                assert!(
                    !placement.spin_flag,
                    "no corner-filled T-spin exists on an empty board: {placement:?}"
                );
            }
        }
    }

    #[test]
    fn test_half_turn_reaches_rotation_two_in_one_move() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placement = find_all_placements(&board, 2, 3, 0)
            .into_iter()
            .find(|p| p.rotation == 2 && p.column == 3)
            .expect("T pointing down at the spawn column");
        assert_eq!(
            placement.moves,
            vec![Action::Rotate180.to_u8(), Action::HardDrop.to_u8()]
        );
    }

    #[test]
    fn test_spin_path_chosen_when_slot_is_reachable_both_ways() {
        // A T pointing left against the right wall, resting on one block at (8, 17):
        // center (9, 16), both back corners are wall and one front corner is filled, so it
        // is a mini when it locks with the spin flag. Dropping it straight down in that
        // rotation reaches the same cells without a spin; kicking in from rotation 0
        // after the drop does spin. The spin path must win.
        let mut cells = empty_cells(10, 20);
        cells[17 * 10 + 8] = 1;
        let board = Board::new(10, 20, &cells);

        let placement = find_all_placements(&board, 2, 3, 0)
            .into_iter()
            .find(|p| (p.piece.x, p.piece.y, p.rotation) == (8, 15, 3))
            .expect("wall slot is reachable");
        assert!(placement.spin_flag, "{placement:?}");
        assert_eq!(
            placement.moves[placement.moves.len() - 2],
            Action::RotateCCW.to_u8(),
            "the path must rotate after its last downward move"
        );

        let params = find_all_placement_params(&board, 2, 3, 0);
        let param = params
            .iter()
            .find(|p| p.action_index as usize == placement.action_index)
            .unwrap();
        assert!(param.spin_flag);
    }
}
