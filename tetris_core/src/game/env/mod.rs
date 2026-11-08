//! Tetris game environment
//!
//! This module contains the main TetrisEnv struct which manages the game state,
//! including the board, current piece, piece queue, hold functionality, and scoring.

mod board;
mod clearing;
pub(crate) mod global_cache;
mod lock_delay;
mod movement;
mod piece_management;
mod piece_queue;
mod placement;
mod pymethods;
mod state;

#[cfg(test)]
mod tests;

// Re-export the main type
pub use state::TetrisEnv;
