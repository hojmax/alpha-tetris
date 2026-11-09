//! Deterministic Tetris engine primitives.

pub mod action_space;
pub mod constants;
pub mod env;
pub mod kicks;
pub mod moves;
pub mod piece;
pub mod scoring;
pub mod tspin;

pub use action_space::{
    get_action_space, ActionSpace, HOLD_ACTION_INDEX, NUM_ACTIONS, NUM_PLACEMENT_ACTIONS,
};
pub use constants::NUM_PIECE_TYPES;
pub use env::TetrisEnv;
pub use kicks::{get_kicks, Rotation};
pub use moves::{find_all_placements, Action, Board, Placement};
pub use piece::{get_cells, Piece, TETROMINOS, TETROMINO_CELLS};
pub use scoring::{
    calculate_attack, combo_attack, determine_clear_type, AttackResult, ClearType,
    BACK_TO_BACK_BONUS, PERFECT_CLEAR_ATTACK,
};
