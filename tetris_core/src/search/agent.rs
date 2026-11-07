//! MCTS Agent
//!
//! The main agent that runs MCTS search and self-play.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES};
use crate::game::env::TetrisEnv;
use crate::inference::PredictionNoiseSettings;
use crate::replay::ReplayMove;

use super::config::MCTSConfig;
use super::export::export_decision_node;
use super::nodes::DecisionNode;
use super::nodes::NO_CHANCE_OUTCOME;
use super::results::{
    GameResult, GameStats, GameTreePlayback, GameTreeStep, MCTSResult, MCTSTreeExport,
    TrainingExample, TraversalStats, TreeNodeExport, TreeStats, TreeStatsAccumulator,
};
use super::search::{extract_subtree, run_search, search_internal, search_internal_without_nn};
use crate::game::action_space::HOLD_ACTION_INDEX;
use crate::game::constants::QUEUE_SIZE;

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::inference::TetrisNN>,
}

struct StateSnapshot {
    state: TetrisEnv,
    frame_idx: u32,
    policy: Vec<f32>,
    mask: Vec<bool>,
    overhang_fields: u32,
    hole_count: u32,
    predicted_total_attack: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrajectoryPredictionMetrics {
    count: u32,
    variance: f32,
    std: f32,
    rmse: f32,
}

fn summarize_trajectory_predictions(
    predicted_totals: &[f32],
    final_total_attack: f32,
) -> TrajectoryPredictionMetrics {
    if predicted_totals.is_empty() {
        return TrajectoryPredictionMetrics::default();
    }

    let count = predicted_totals.len() as u32;
    let mean_prediction = predicted_totals.iter().sum::<f32>() / count as f32;
    let variance = predicted_totals
        .iter()
        .map(|prediction| {
            let centered = *prediction - mean_prediction;
            centered * centered
        })
        .sum::<f32>()
        / count as f32;
    let rmse = (predicted_totals
        .iter()
        .map(|prediction| {
            let error = *prediction - final_total_attack;
            error * error
        })
        .sum::<f32>()
        / count as f32)
        .sqrt();

    TrajectoryPredictionMetrics {
        count,
        variance,
        std: variance.sqrt(),
        rmse,
    }
}

fn export_tree(
    root: &DecisionNode,
    num_simulations: u32,
    mcts_result: &MCTSResult,
    q_bounds: (f32, f32),
) -> MCTSTreeExport {
    let mut nodes: Vec<TreeNodeExport> = Vec::new();
    export_decision_node(root, None, None, &mut nodes);

    MCTSTreeExport {
        nodes,
        root_id: 0,
        num_simulations,
        selected_action: mcts_result.action,
        policy: mcts_result.policy.clone(),
        q_min: q_bounds.0,
        q_max: q_bounds.1,
    }
}

fn action_reveals_new_visible_piece(env: &TetrisEnv, selected_action: usize) -> bool {
    selected_action != HOLD_ACTION_INDEX || env.get_hold_piece().is_none()
}

fn realized_chance_outcome(env: &TetrisEnv, action_reveals_new_visible_piece: bool) -> usize {
    if action_reveals_new_visible_piece {
        env.piece_queue
            .get(QUEUE_SIZE - 1)
            .expect("Visible queue tail should exist when an action reveals a new piece")
    } else {
        NO_CHANCE_OUTCOME
    }
}

fn compute_value_targets(attacks: &[u32]) -> Vec<f32> {
    let mut values = vec![0.0f32; attacks.len()];
    let mut cumulative = 0.0f32;
    for i in (0..attacks.len()).rev() {
        cumulative += attacks[i] as f32;
        values[i] = cumulative;
    }
    values
}

fn prediction_noise_from_config(config: &MCTSConfig) -> PredictionNoiseSettings {
    PredictionNoiseSettings {
        policy_mean: config.policy_noise_mean,
        policy_std: config.policy_noise_std,
        value_mean: config.value_noise_mean,
        value_std: config.value_noise_std,
        seed: config.prediction_noise_seed,
    }
}

#[pymethods]
impl MCTSAgent {
    #[new]
    pub fn new(config: MCTSConfig) -> Self {
        MCTSAgent { config, nn: None }
    }

    /// Load a neural network model
    ///
    /// Args:
    ///     path: Path to ONNX model file
    ///
    /// Returns:
    ///     True if loaded successfully, False otherwise
    pub fn load_model(&mut self, path: &str) -> bool {
        match crate::inference::TetrisNN::load(path) {
            Ok(nn) => {
                self.nn = Some(nn);
                true
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                false
            }
        }
    }

    /// Check if a neural network model is loaded
    pub fn has_model(&self) -> bool {
        self.nn.is_some()
    }

    /// Enable or disable board-embedding cache inside Rust NN inference.
    ///
    /// Returns False if no model is loaded.
    pub fn set_board_cache_enabled(&self, enabled: bool) -> bool {
        if let Some(nn) = self.nn.as_ref() {
            nn.set_board_cache_enabled(enabled);
            true
        } else {
            false
        }
    }

    /// Read and reset board-embedding cache stats (hits, misses, cache_size).
    ///
    /// Returns None if no model is loaded.
    pub fn get_and_reset_cache_stats(&self) -> Option<(u64, u64, usize)> {
        self.nn
            .as_ref()
            .map(crate::inference::TetrisNN::get_and_reset_cache_stats)
    }

    /// Run one direct NN evaluation on an environment using the same
    /// valid-action path the Rust search uses for leaf evaluation.
    ///
    /// Returns:
    ///     Tuple of (policy over valid actions, raw value)
    #[pyo3(signature = (env, max_placements=100))]
    pub fn predict_with_valid_actions(
        &self,
        env: &TetrisEnv,
        max_placements: usize,
    ) -> PyResult<(Vec<f32>, f32)> {
        let Some(nn) = self.nn.as_ref() else {
            return Err(PyValueError::new_err("No model loaded"));
        };

        let valid_actions = env.get_cached_valid_action_indices();
        nn.predict_with_valid_actions(
            env,
            &valid_actions,
            max_placements,
            prediction_noise_from_config(&self.config),
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to run inference: {e}")))
    }

    /// Play a full game using MCTS with the loaded model
    ///
    /// All neural network inference happens in Rust. Returns training data.
    ///
    /// Args:
    ///     max_placements: Maximum placements per game (hold actions don't count)
    ///     add_noise: Whether to add Dirichlet noise (for exploration)
    ///
    /// Returns:
    ///     GameResult with training examples
    #[pyo3(signature = (max_placements=100, add_noise=true))]
    pub fn play_game(&self, max_placements: u32, add_noise: bool) -> Option<GameResult> {
        let env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
        self.play_game_on_env(env, max_placements, add_noise)
            .map(|(result, _replay)| result)
    }

    /// Generate multiple games of training data
    ///
    /// Args:
    ///     num_games: Number of games to play
    ///     max_placements: Maximum placements per game (hold actions don't count)
    ///     add_noise: Whether to add Dirichlet noise
    ///
    /// Returns:
    ///     List of all training examples from all games
    #[pyo3(signature = (num_games, max_placements=100, add_noise=true))]
    pub fn generate_games(
        &self,
        num_games: u32,
        max_placements: u32,
        add_noise: bool,
    ) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for _ in 0..num_games {
            if let Some(result) = self.play_game(max_placements, add_noise) {
                all_examples.extend(result.examples);
            }
        }

        all_examples
    }

    /// Run MCTS search and return both the result and the tree structure.
    ///
    /// This is for visualization/debugging - it exports the entire MCTS tree.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///
    /// Returns:
    ///     Tuple of (MCTSResult, MCTSTreeExport), or None if no valid actions exist
    #[pyo3(signature = (env, add_noise=false))]
    pub fn search_with_tree(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
    ) -> Option<(MCTSResult, MCTSTreeExport)> {
        // Keep parity with select_action/play_game: no valid actions means terminal state.
        let mask = crate::inference::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (mcts_result, root, _tree_stats, _traversal_stats, q_bounds) =
            if let Some(nn) = self.nn.as_ref() {
                let (policy, nn_value) = nn
                    .predict_masked(
                        env,
                        &mask,
                        self.config.max_placements as usize,
                        prediction_noise_from_config(&self.config),
                    )
                    .expect("Neural network prediction failed");

                search_internal(&self.config, nn, env, policy, nn_value, add_noise)
            } else {
                search_internal_without_nn(&self.config, env, add_noise)
            };

        Some((
            mcts_result.clone(),
            export_tree(&root, self.config.num_simulations, &mcts_result, q_bounds),
        ))
    }

    /// Play out a full game from an existing environment and export the tree at every step.
    ///
    /// This follows the same subtree-reuse path as normal self-play/evaluation, but keeps
    /// each pre-action tree snapshot so the visualizer can render the full played trajectory.
    ///
    /// Args:
    ///     env: Starting game state (left unchanged; the playback clones it)
    ///     max_placements: Absolute placement horizon for the rollout
    ///     add_noise: Whether to add Dirichlet noise at each root
    ///
    /// Returns:
    ///     GameTreePlayback with per-move tree exports, or None if NN inference fails
    #[pyo3(signature = (env, max_placements=100, add_noise=true))]
    pub fn play_game_with_trees(
        &self,
        env: &TetrisEnv,
        max_placements: u32,
        add_noise: bool,
    ) -> Option<GameTreePlayback> {
        let mut env = env.clone();
        let starting_placement_count = env.placement_count;
        let mut frame_index: u32 = 0;
        let mut total_attack: u32 = 0;
        let mut replay_moves: Vec<ReplayMove> = Vec::new();
        let mut steps: Vec<GameTreeStep> = Vec::new();
        let mut reused_root: Option<DecisionNode> = None;
        let mut tree_reuse_hits: u32 = 0;
        let mut tree_reuse_misses: u32 = 0;

        while env.placement_count < max_placements {
            if env.game_over {
                break;
            }

            let mask = crate::inference::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                debug_assert!(
                    env.game_over,
                    "No valid actions but game not over - this is a bug"
                );
                break;
            }

            let current_placement_count = env.placement_count;
            let (result, root, _tree_stats, _traversal_stats, q_bounds) = self.search_maybe_reuse(
                &env,
                &mask,
                add_noise,
                max_placements,
                reused_root.take(),
            )?;
            let root = root.expect("search_maybe_reuse should always return a root");
            let tree_export = export_tree(&root, self.config.num_simulations, &result, q_bounds);

            let selected_action = result.action;
            let reveals_new_visible_piece = action_reveals_new_visible_piece(&env, selected_action);

            let attack = env
                .execute_action_index(selected_action)
                .expect("MCTS selected action is not executable");
            total_attack += attack;
            replay_moves.push(ReplayMove {
                action: selected_action,
                attack,
            });

            let selected_chance_outcome = realized_chance_outcome(&env, reveals_new_visible_piece);
            steps.push(GameTreeStep {
                frame_index,
                placement_count: current_placement_count,
                selected_action,
                selected_chance_outcome,
                attack,
                tree: tree_export,
            });
            frame_index += 1;

            let should_attempt_tree_reuse = !env.game_over && env.placement_count < max_placements;
            if self.config.reuse_tree && should_attempt_tree_reuse {
                if let Some(subtree) =
                    extract_subtree(root, selected_action, selected_chance_outcome)
                {
                    reused_root = Some(subtree);
                    tree_reuse_hits += 1;
                } else {
                    tree_reuse_misses += 1;
                }
            }
        }

        Some(GameTreePlayback {
            steps,
            replay_moves,
            total_attack,
            num_moves: env.placement_count.saturating_sub(starting_placement_count),
            num_frames: frame_index,
            tree_reuse_hits,
            tree_reuse_misses,
        })
    }

    /// Run MCTS search and return the result (simpler than search_with_tree).
    ///
    /// This is the main method for selecting actions during play/evaluation.
    /// Handles NN inference internally.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///
    /// Returns:
    ///     MCTSResult with selected action and policy, or None if no model loaded
    #[pyo3(signature = (env, add_noise=false))]
    pub fn select_action(&self, env: &TetrisEnv, add_noise: bool) -> Option<MCTSResult> {
        let nn = self.nn.as_ref()?;

        // Get action mask and initial policy
        let mask = crate::inference::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (policy, nn_value) = nn
            .predict_masked(
                env,
                &mask,
                self.config.max_placements as usize,
                prediction_noise_from_config(&self.config),
            )
            .expect("Neural network prediction failed");

        let (mcts_result, _root, _tree_stats, _traversal_stats, _q_bounds) =
            search_internal(&self.config, nn, env, policy, nn_value, add_noise);
        Some(mcts_result)
    }
}

impl MCTSAgent {
    /// Get a reference to the neural network (if loaded).
    pub fn get_nn(&self) -> Option<&crate::inference::TetrisNN> {
        self.nn.as_ref()
    }

    /// Live-update the agent's `visit_sampling_epsilon` without rebuilding
    /// the agent. The runtime worker calls this from the per-game read of
    /// the shared `LiveSearchOverrides` atomics.
    pub(crate) fn set_visit_sampling_epsilon(&mut self, value: f32) {
        self.config.visit_sampling_epsilon = value;
    }

    /// Play a full game on a pre-created environment, returning both the
    /// GameResult (with training examples) and a Vec of ReplayMove for replay.
    pub(crate) fn play_game_on_env(
        &self,
        mut env: TetrisEnv,
        max_placements: u32,
        add_noise: bool,
    ) -> Option<(GameResult, Vec<ReplayMove>)> {
        let mut states: Vec<StateSnapshot> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();
        let mut replay_moves: Vec<ReplayMove> = Vec::new();
        let mut stats = GameStats::default();
        let mut valid_moves_sum: u32 = 0;
        let mut max_valid_moves: u32 = 0;
        let mut tree_stats_acc = TreeStatsAccumulator::new();
        let mut frame_index: u32 = 0;
        let starting_placement_count = env.placement_count;

        // Tree reuse state: carry the subtree from previous move
        let mut reused_root: Option<DecisionNode> = None;
        let mut tree_reuse_hits: u32 = 0;
        let mut tree_reuse_misses: u32 = 0;
        let mut carry_fraction_sum: f32 = 0.0;
        let mut traversal_total: u64 = 0;
        let mut traversal_expansions: u64 = 0;
        let mut traversal_terminal_ends: u64 = 0;
        let mut traversal_horizon_ends: u64 = 0;

        while env.placement_count < max_placements {
            if env.game_over {
                break;
            }

            // Get action mask
            let mask = crate::inference::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                debug_assert!(
                    env.game_over,
                    "No valid actions but game not over - this is a bug"
                );
                break;
            }
            let valid_moves = mask.iter().filter(|&&is_valid| is_valid).count() as u32;

            // Run MCTS search (with tree reuse if available)
            let (result, root, move_tree_stats, move_traversal_stats, _q_bounds) = self
                .search_maybe_reuse(&env, &mask, add_noise, max_placements, reused_root.take())?;

            let tree_total_nodes = move_tree_stats.total_nodes;
            valid_moves_sum += valid_moves;
            max_valid_moves = max_valid_moves.max(valid_moves);
            tree_stats_acc.add(move_tree_stats);
            traversal_total += u64::from(move_traversal_stats.total());
            traversal_expansions += u64::from(move_traversal_stats.expansions);
            traversal_terminal_ends += u64::from(move_traversal_stats.terminal_ends);
            traversal_horizon_ends += u64::from(move_traversal_stats.horizon_ends);
            if result.action == HOLD_ACTION_INDEX {
                stats.holds += 1;
            }

            let (overhang_count, hole_count) = super::utils::count_overhang_fields_and_holes(&env);
            let predicted_total_attack = if self.nn.is_some() {
                root.as_ref()
                    .map(|search_root| env.attack as f32 + search_root.raw_nn_value)
            } else {
                None
            };
            states.push(StateSnapshot {
                state: env.clone(),
                frame_idx: frame_index,
                policy: result.policy.clone(),
                mask: mask.clone(),
                overhang_fields: overhang_count,
                hole_count,
                predicted_total_attack,
            });

            // Decide which chance outcome key will be realized after executing this action.
            // Queue-advancing actions reveal a new visible tail piece; hold-swap does not.
            let selected_action = result.action;
            let action_reveals_new_visible_piece =
                selected_action != HOLD_ACTION_INDEX || env.get_hold_piece().is_none();

            // Execute the selected action
            let attack = env
                .execute_action_index(selected_action)
                .expect("MCTS selected action is not executable");
            attacks.push(attack);
            replay_moves.push(ReplayMove {
                action: selected_action,
                attack,
            });
            frame_index += 1;

            // Try to extract subtree for reuse on the next move.
            // For queue-advancing actions, the chance key is the realized visible tail piece
            // (queue index QUEUE_SIZE - 1 after transition). For hold-swap, queue is unchanged
            // and the chance key is deterministic NO_CHANCE_OUTCOME.
            //
            // Do not count terminal or max-placement transitions as reuse misses:
            // there is no next search step, so reuse is not applicable.
            let should_attempt_tree_reuse = !env.game_over && env.placement_count < max_placements;
            if self.config.reuse_tree && should_attempt_tree_reuse {
                if let Some(root) = root {
                    let chance_outcome = if action_reveals_new_visible_piece {
                        env.piece_queue.get(QUEUE_SIZE - 1)
                    } else {
                        Some(NO_CHANCE_OUTCOME)
                    };

                    if let Some(chance_outcome) = chance_outcome {
                        if let Some(subtree) =
                            extract_subtree(root, selected_action, chance_outcome)
                        {
                            let subtree_nodes =
                                super::search::compute_tree_stats(&subtree).total_nodes;
                            if tree_total_nodes > 0 {
                                carry_fraction_sum +=
                                    subtree_nodes as f32 / tree_total_nodes as f32;
                            }
                            reused_root = Some(subtree);
                            tree_reuse_hits += 1;
                        } else {
                            tree_reuse_misses += 1;
                        }
                    } else {
                        tree_reuse_misses += 1;
                    }
                }
            }

            // Collect stats from the attack result
            if let Some(ref attack_result) = env.get_last_attack_result() {
                let lines = attack_result.lines_cleared;
                stats.total_lines += lines;

                if attack_result.combo > stats.max_combo {
                    stats.max_combo = attack_result.combo;
                }
                if attack_result.back_to_back_attack > 0 {
                    stats.back_to_backs += 1;
                }
                if attack_result.is_perfect_clear {
                    stats.perfect_clears += 1;
                }

                if attack_result.is_tspin {
                    match lines {
                        1 => {
                            if attack_result.is_mini_tspin {
                                stats.tspin_minis += 1;
                            } else {
                                stats.tspin_singles += 1;
                            }
                        }
                        2 => stats.tspin_doubles += 1,
                        3 => stats.tspin_triples += 1,
                        _ => {}
                    }
                } else {
                    match lines {
                        1 => stats.singles += 1,
                        2 => stats.doubles += 1,
                        3 => stats.triples += 1,
                        4 => stats.tetrises += 1,
                        _ => {}
                    }
                }
            }
        }

        let num_states = states.len();
        debug_assert_eq!(
            states.len(),
            attacks.len(),
            "States and attacks should have same length"
        );

        let values = compute_value_targets(&attacks);
        let mut examples = Vec::with_capacity(num_states);

        for (snapshot, value) in states.iter().zip(values.iter().copied()) {
            let state = &snapshot.state;
            let board: Vec<u8> = state
                .board_cells()
                .iter()
                .map(|&cell| if cell != 0 { 1 } else { 0 })
                .collect();

            let current_piece = state
                .get_current_piece()
                .map(|piece| piece.piece_type)
                .expect("Training examples require a current piece");
            let hold_piece = match state.get_hold_piece() {
                Some(piece) => piece.piece_type,
                None => NUM_PIECE_TYPES,
            };
            let hold_available = !state.is_hold_used();
            let next_queue = state.get_queue(5);
            let next_hidden_piece_probs = crate::inference::next_hidden_piece_distribution(state);
            let raw_bumpiness =
                super::utils::compute_bumpiness(&state.column_heights[..state.width]);
            let column_heights =
                super::utils::normalize_column_heights(&state.column_heights[..state.width]);
            let row_fill_counts = super::utils::normalize_row_fill_counts(
                &state.row_fill_counts[..state.height],
                state.width,
            );
            let total_blocks = super::utils::normalize_total_blocks(state.total_blocks);
            let bumpiness = super::utils::normalize_bumpiness(raw_bumpiness);
            let holes = super::utils::normalize_holes(snapshot.hole_count);
            let overhang_fields = super::utils::normalize_overhang_fields(snapshot.overhang_fields);
            let raw_max = state.column_heights[..state.width]
                .iter()
                .copied()
                .max()
                .expect("Training examples require at least one column");
            let max_column_height = super::utils::normalize_max_column_height(raw_max);

            examples.push(TrainingExample {
                board,
                current_piece,
                hold_piece,
                hold_available,
                next_queue,
                move_number: snapshot.frame_idx,
                placement_count: snapshot.state.placement_count as f32 / max_placements as f32,
                combo: crate::inference::normalize_combo_for_feature(state.combo),
                back_to_back: state.back_to_back,
                next_hidden_piece_probs,
                column_heights,
                max_column_height,
                row_fill_counts,
                total_blocks,
                bumpiness,
                holes,
                policy: snapshot.policy.clone(),
                value,
                action_mask: snapshot.mask.clone(),
                overhang_fields,
                game_number: 0,
                game_total_attack: 0,
            });
        }

        let total_attack: u32 = attacks.iter().sum();
        let predicted_total_attacks: Vec<f32> = states
            .iter()
            .filter_map(|snapshot| snapshot.predicted_total_attack)
            .collect();
        let prediction_metrics =
            summarize_trajectory_predictions(&predicted_total_attacks, total_attack as f32);
        let total_overhang_fields: u32 = states.iter().map(|s| s.overhang_fields).sum();
        let num_frames = states.len() as u32;
        let num_moves = env.placement_count.saturating_sub(starting_placement_count);
        let avg_valid_moves = if num_frames > 0 {
            valid_moves_sum as f32 / num_frames as f32
        } else {
            0.0
        };
        let avg_overhang_fields = if num_frames > 0 {
            total_overhang_fields as f32 / num_frames as f32
        } else {
            0.0
        };
        let traversal_expansion_fraction = if traversal_total > 0 {
            traversal_expansions as f32 / traversal_total as f32
        } else {
            0.0
        };
        let traversal_terminal_fraction = if traversal_total > 0 {
            traversal_terminal_ends as f32 / traversal_total as f32
        } else {
            0.0
        };
        let traversal_horizon_fraction = if traversal_total > 0 {
            traversal_horizon_ends as f32 / traversal_total as f32
        } else {
            0.0
        };

        let tree_stats = tree_stats_acc.finalize();
        let (cache_hits, cache_misses, cache_size) = if let Some(nn) = self.nn.as_ref() {
            nn.get_and_reset_cache_stats()
        } else {
            (0, 0, 0)
        };

        Some((
            GameResult {
                examples,
                total_attack,
                num_moves,
                avg_valid_actions: avg_valid_moves,
                max_valid_actions: max_valid_moves,
                stats,
                tree_stats,
                total_overhang_fields,
                avg_overhang_fields,
                cache_hits,
                cache_misses,
                cache_size,
                tree_reuse_hits,
                tree_reuse_misses,
                tree_reuse_carry_fraction: if tree_reuse_hits > 0 {
                    carry_fraction_sum / tree_reuse_hits as f32
                } else {
                    0.0
                },
                traversal_total,
                traversal_expansions,
                traversal_terminal_ends,
                traversal_horizon_ends,
                traversal_expansion_fraction,
                traversal_terminal_fraction,
                traversal_horizon_fraction,
                trajectory_predicted_total_attack_count: prediction_metrics.count,
                trajectory_predicted_total_attack_variance: prediction_metrics.variance,
                trajectory_predicted_total_attack_std: prediction_metrics.std,
                trajectory_predicted_total_attack_rmse: prediction_metrics.rmse,
            },
            replay_moves,
        ))
    }

    /// Run MCTS search, optionally reusing a subtree from a previous search.
    ///
    /// Returns (MCTSResult, Option<DecisionNode>, TreeStats, TraversalStats, q_bounds).
    /// The DecisionNode is returned for tree reuse extraction on the next move.
    /// Returns None (propagated from NN prediction failure) to discard the game.
    fn search_maybe_reuse(
        &self,
        env: &TetrisEnv,
        mask: &[bool],
        add_noise: bool,
        max_placements: u32,
        reused_root: Option<DecisionNode>,
    ) -> Option<(
        MCTSResult,
        Option<DecisionNode>,
        TreeStats,
        TraversalStats,
        (f32, f32),
    )> {
        if let Some(nn) = self.nn.as_ref() {
            let (result, root, tree_stats, traversal_stats, q_bounds) = if let Some(root) =
                reused_root
            {
                let evaluator = super::search::NeuralLeafEvaluator {
                    nn,
                    nn_value_weight: self.config.nn_value_weight,
                    prediction_noise: prediction_noise_from_config(&self.config),
                };
                run_search(&self.config, &evaluator, root, add_noise)
            } else {
                let (policy, nn_value) = match nn.predict_masked(
                    env,
                    mask,
                    max_placements as usize,
                    prediction_noise_from_config(&self.config),
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!(
                            "[MCTSAgent] NN prediction failed at placement {}: {}. Discarding rollout.",
                            env.placement_count, e
                        );
                        return None;
                    }
                };
                search_internal(&self.config, nn, env, policy, nn_value, add_noise)
            };
            Some((result, Some(root), tree_stats, traversal_stats, q_bounds))
        } else {
            let (result, root, tree_stats, traversal_stats, q_bounds) =
                if let Some(root) = reused_root {
                    run_search(
                        &self.config,
                        &super::search::BootstrapLeafEvaluator,
                        root,
                        add_noise,
                    )
                } else {
                    search_internal_without_nn(&self.config, env, add_noise)
                };
            Some((result, Some(root), tree_stats, traversal_stats, q_bounds))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::nodes::DecisionNode;
    use super::*;
    use crate::search::NUM_ACTIONS;

    #[test]
    fn test_summarize_trajectory_predictions_zero_variance_and_error() {
        let metrics = summarize_trajectory_predictions(&[10.0, 10.0, 10.0], 10.0);

        assert_eq!(metrics.count, 3);
        assert_eq!(metrics.variance, 0.0);
        assert_eq!(metrics.std, 0.0);
        assert_eq!(metrics.rmse, 0.0);
    }

    #[test]
    fn test_summarize_trajectory_predictions_separates_variance_and_rmse() {
        let metrics = summarize_trajectory_predictions(&[9.0, 9.0, 9.0], 10.0);

        assert_eq!(metrics.count, 3);
        assert_eq!(metrics.variance, 0.0);
        assert_eq!(metrics.std, 0.0);
        assert!((metrics.rmse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_expand_action_updates_state() {
        // Create an agent with a model
        let config = MCTSConfig::default();
        let mut agent = MCTSAgent::new(config);

        // We need a model for expand_action to work
        // Skip test if no model is available
        if !agent.load_model("outputs/checkpoints/selfplay.onnx") {
            eprintln!("Skipping test - no model available");
            return;
        }

        // Create initial env with a known seed
        let env = TetrisEnv::with_seed(10, 20, 42);
        let initial_piece = env.get_current_piece().unwrap().piece_type;
        let initial_queue: Vec<usize> = env.get_queue(5);

        // Create a decision node
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let mut root = DecisionNode::new(env.clone());
        root.set_nn_output(&policy, 0.0);

        // Get a valid action
        // Run MCTS search to expand the action
        let mask = crate::inference::get_action_mask(&env);
        let nn = agent.nn.as_ref().unwrap();
        let (nn_policy, nn_value) = nn
            .predict_masked(
                &env,
                &mask,
                100,
                prediction_noise_from_config(&agent.config),
            )
            .unwrap();

        let (_result, root_after, _tree_stats, _traversal_stats, _q_bounds) =
            search_internal(&agent.config, nn, &env, nn_policy, nn_value, false);

        // The root should have children after search
        assert!(
            root_after.has_children(),
            "Root should have children after search"
        );

        // Check that actions were expanded
        let first_child = root_after.children().next();
        if let Some((_, chance_node)) = first_child {
            let new_piece = chance_node.state.get_current_piece();

            // The current piece should have changed (old piece was placed, new one spawned)
            if let Some(ref new_p) = new_piece {
                assert_ne!(
                    new_p.piece_type, initial_piece,
                    "Current piece should have changed after placing"
                );
            }

            // The first piece from the old queue should now be current
            if let Some(ref new_p) = new_piece {
                assert_eq!(
                    new_p.piece_type, initial_queue[0],
                    "New current piece should be the first from old queue"
                );
            }
        }
    }

    #[test]
    fn test_compute_value_targets_without_penalties() {
        let attacks = vec![1, 2, 3];
        let values = compute_value_targets(&attacks);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 6.0).abs() < 1e-6);
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_play_game_examples_use_current_state_overhang_and_holes() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 5;
        let agent = MCTSAgent::new(config);
        let result = agent
            .play_game(12, false)
            .expect("play_game should return a result");
        assert!(
            !result.examples.is_empty(),
            "play_game should generate at least one training example"
        );

        for example in result.examples.iter() {
            let mut env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
            env.board.fill(0);
            env.board[..example.board.len()].copy_from_slice(&example.board);
            env.invalidate_board_analysis_cache();

            let (expected_overhang, expected_holes_raw) =
                super::super::utils::count_overhang_fields_and_holes(&env);
            let expected_normalized_overhang =
                super::super::utils::normalize_overhang_fields(expected_overhang);
            let expected_holes = super::super::utils::normalize_holes(expected_holes_raw);

            assert!(
                (example.overhang_fields - expected_normalized_overhang).abs() < 1e-6,
                "overhang_fields must match the saved board at move {}",
                example.move_number
            );
            assert!(
                (example.holes - expected_holes).abs() < 1e-6,
                "holes must match the saved board at move {}",
                example.move_number
            );
        }
    }

    #[test]
    fn test_play_game_with_trees_returns_step_snapshots() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 5;
        let agent = MCTSAgent::new(config);
        let env = TetrisEnv::with_seed(10, 20, 123);

        let playback = agent
            .play_game_with_trees(&env, 4, false)
            .expect("play_game_with_trees should return playback data");

        assert!(
            !playback.steps.is_empty(),
            "play_game_with_trees should capture at least one step"
        );
        assert_eq!(playback.steps.len(), playback.replay_moves.len());
        assert_eq!(playback.num_frames as usize, playback.steps.len());
        assert!(playback.num_moves <= 4);

        for step in playback.steps.iter() {
            assert_eq!(step.tree.root_id, 0);
            assert_eq!(step.tree.selected_action, step.selected_action);
            assert!(
                !step.tree.nodes.is_empty(),
                "each playback step should include an exported tree"
            );
        }
    }
}
