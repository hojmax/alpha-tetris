//! Background Game Generator
//!
//! Spawns a worker thread that continuously generates self-play games
//! using MCTS. Training data is kept in a shared in-memory buffer that
//! Python can sample from directly, avoiding disk I/O during training.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::game::constants::{AUX_FEATURES, BOARD_HEIGHT, BOARD_WIDTH};
use crate::game::env::TetrisEnv;
use crate::search::GameStats;
use crate::search::{GameResult, GameTreeStats, MCTSAgent, MCTSConfig, TrainingExample};

use crate::replay::compact::{BOARD_BYTES, MASK_BYTES, POLICY_SLOTS};
use crate::replay::npz::{read_examples_from_npz, write_examples_to_npz};
use crate::replay::{CompactExample, GameReplay, ReplayMove};

/// `(board_bits, aux, policy_legal, values, overhang_fields, mask_bits)`; see
/// `GameGenerator::examples_to_numpy`.
type CompactNumpyBatch<'py> = (
    &'py PyArray2<u8>,
    &'py PyArray2<f32>,
    &'py PyArray2<u16>,
    &'py PyArray1<f32>,
    &'py PyArray1<f32>,
    &'py PyArray2<u8>,
);

mod py_api;
mod runtime;
mod shared;

#[cfg(test)]
mod tests;

use shared::{
    CandidateModelRequest, CompletedGameResult, IncumbentState, LastGameInfo, LiveSearchOverrides,
    ModelEvalEvent, SharedBuffer, SnapshotPersister, WorkerContext, WorkerSettings,
    WorkerSharedState,
};

#[pyclass]
pub struct GameGenerator {
    /// Initial model path used on startup and protected from cleanup.
    bootstrap_model_path: PathBuf,
    /// Full path to training_data.npz (for periodic saves and replay preload)
    training_data_path: PathBuf,
    /// MCTS configuration
    config: MCTSConfig,
    /// Maximum placements per game (hold actions do not count)
    max_placements: u32,
    /// Live-tunable Dirichlet noise toggle. Stored as `Arc<AtomicBool>`
    /// so trainer + remote-generator overrides take effect on the next
    /// game without restarting workers. Candidate-eval games still
    /// force `false` deterministically inside the worker loop.
    live_add_noise: Arc<AtomicBool>,
    /// Live-tunable `visit_sampling_epsilon` (epsilon-greedy at the
    /// posterior). Stored as `Arc<AtomicU32>` of `f32::to_bits()`; the
    /// worker stamps it onto the agent's `MCTSConfig` at the top of
    /// every game.
    live_visit_sampling_epsilon: Arc<AtomicU32>,
    /// Wall-clock interval between disk saves (for resume capability)
    save_interval_seconds: f64,
    /// Number of worker threads
    num_workers: usize,
    /// Whether one worker is reserved for candidate evaluation before promotion.
    candidate_gating_enabled: bool,
    /// Background snapshot writer for replay buffer persistence.
    snapshot_persister: Option<Arc<SnapshotPersister>>,
    /// Fixed seeds used by the evaluator worker for candidate games.
    candidate_eval_seeds: Arc<[u64]>,
    /// Number of simulations per move before the first promoted NN model.
    non_network_num_simulations: u32,
    /// Shared replay buffer (accessed by both worker threads and Python)
    buffer: Arc<SharedBuffer>,
    /// Whether the generator is running
    running: Arc<AtomicBool>,
    /// Number of games generated since start
    games_generated: Arc<AtomicU64>,
    /// Number of examples generated since start
    examples_generated: Arc<AtomicU64>,
    /// Completed game stats queue for per-game logging
    completed_games: Arc<RwLock<VecDeque<LastGameInfo>>>,
    /// Most recent pending candidate model (latest wins, older pending models are dropped).
    /// Used only when candidate gating is enabled.
    pending_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    /// Candidate currently under evaluation by the evaluator worker.
    /// Used only when candidate gating is enabled.
    evaluating_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    /// Queue of evaluator decisions for Python-side logging.
    /// Used only when candidate gating is enabled.
    model_eval_events: Arc<RwLock<VecDeque<ModelEvalEvent>>>,
    /// Shared incumbent model path used by all non-evaluator workers.
    incumbent_model_path: Arc<RwLock<PathBuf>>,
    /// Whether the incumbent currently uses NN guidance or no-network bootstrap mode.
    incumbent_uses_network: Arc<AtomicBool>,
    /// Training step associated with the incumbent model.
    incumbent_model_step: Arc<AtomicU64>,
    /// Incremented whenever a candidate is promoted and workers should reload.
    incumbent_model_version: Arc<AtomicU64>,
    /// Current nn_value_weight used by incumbent NN-guided rollouts.
    incumbent_nn_value_weight: Arc<AtomicU32>,
    /// Current death penalty for incumbent search. The trainer is the sole
    /// authority and pushes new values via `queue_candidate_model` (gating)
    /// or `sync_model_directly` (no-gating); Rust does not modify this.
    incumbent_death_penalty: Arc<AtomicU32>,
    /// Current overhang penalty weight for incumbent search. Trainer-authoritative,
    /// see `incumbent_death_penalty`.
    incumbent_overhang_penalty_weight: Arc<AtomicU32>,
    /// Average attack from the evaluation that promoted the current incumbent.
    incumbent_eval_avg_attack: Arc<AtomicU32>,
    /// Whether to persist worst-case eval game trees to disk when candidate gating runs.
    save_eval_trees: bool,
    /// Thread handles (for joining on stop)
    thread_handles: Vec<JoinHandle<()>>,
}
