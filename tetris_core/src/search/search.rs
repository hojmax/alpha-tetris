//! MCTS Search Implementation
//!
//! Core search algorithm including simulation, expansion, and backpropagation.

use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};

use crate::game::constants::{NUM_PIECE_TYPES, QUEUE_SIZE};
use crate::game::env::TetrisEnv;
use crate::inference::{PredictionNoiseSettings, TetrisNN};

use super::config::MCTSConfig;
use super::nodes::{ChanceNode, ChanceOutcomes, DecisionNode, NodeRef, NO_CHANCE_OUTCOME};
use super::results::{MCTSResult, TraversalStats, TreeStats};
use crate::game::action_space::{HOLD_ACTION_INDEX, NUM_ACTIONS};

fn sample_action_from_policy(policy: &[f32], rng: &mut StdRng) -> usize {
    let total_mass: f32 = policy.iter().sum();
    assert!(
        total_mass.is_finite() && total_mass > 0.0,
        "Policy sampling requires positive finite mass, got {total_mass}"
    );

    let mut threshold = rng.gen::<f32>() * total_mass;
    for (idx, &probability) in policy.iter().enumerate() {
        if probability <= 0.0 {
            continue;
        }
        threshold -= probability;
        if threshold <= 0.0 {
            return idx;
        }
    }

    policy
        .iter()
        .enumerate()
        .rfind(|(_, probability)| **probability > 0.0)
        .map(|(idx, _)| idx)
        .expect("Policy sampling should find a positive-probability action when mass is positive")
}

#[derive(Debug)]
enum ExpandActionError {
    InvariantViolation { action_idx: usize },
    ChanceOutcomeInvariantViolation { outcome_idx: usize },
    EvaluatorFailed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SimulationOutcome {
    Expansion,
    TerminalEnd,
    HorizonEnd,
}

fn uniform_action_priors_for_valid_actions(valid_action_count: usize) -> Vec<f32> {
    if valid_action_count == 0 {
        return Vec::new();
    }

    vec![1.0 / valid_action_count as f32; valid_action_count]
}

fn scale_nn_value(value: f32, nn_value_weight: f32) -> f32 {
    value * nn_value_weight
}

fn action_reveals_new_visible_piece(state: &TetrisEnv, action_idx: usize) -> bool {
    if action_idx != HOLD_ACTION_INDEX {
        return true;
    }

    // Holding with an empty hold slot consumes queue front and reveals a new visible tail piece.
    // Holding with an occupied hold slot swaps current<->hold and leaves queue unchanged.
    state.hold_piece.is_none()
}

pub(super) trait LeafEvaluator {
    /// Evaluate a freshly expanded leaf whose valid actions are `valid_actions`.
    fn evaluate(
        &self,
        state: &TetrisEnv,
        valid_actions: &[usize],
        max_placements: u32,
    ) -> Result<LeafEvaluation, String>;
}

pub(super) struct LeafEvaluation {
    pub(super) action_priors: Vec<f32>,
    pub(super) raw_value: f32,
    pub(super) value: f32,
}

pub(super) struct NeuralLeafEvaluator<'a> {
    pub(super) nn: &'a TetrisNN,
    pub(super) nn_value_weight: f32,
    pub(super) prediction_noise: PredictionNoiseSettings,
}

impl LeafEvaluator for NeuralLeafEvaluator<'_> {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        valid_actions: &[usize],
        max_placements: u32,
    ) -> Result<LeafEvaluation, String> {
        match self.nn.predict_with_valid_actions(
            state,
            valid_actions,
            max_placements as usize,
            self.prediction_noise,
        ) {
            Ok((policy, value)) => Ok(LeafEvaluation {
                action_priors: policy,
                raw_value: value,
                value: scale_nn_value(value, self.nn_value_weight),
            }),
            Err(error) => Err(format!(
                "NN prediction failed during expansion at move {}: {}",
                state.placement_count, error
            )),
        }
    }
}

pub(super) struct BootstrapLeafEvaluator;

impl LeafEvaluator for BootstrapLeafEvaluator {
    fn evaluate(
        &self,
        _state: &TetrisEnv,
        valid_actions: &[usize],
        _max_placements: u32,
    ) -> Result<LeafEvaluation, String> {
        Ok(LeafEvaluation {
            action_priors: uniform_action_priors_for_valid_actions(valid_actions.len()),
            raw_value: 0.0,
            value: 0.0,
        })
    }
}

fn update_q_bounds(q_bounds: &mut (f32, f32), value: f32) {
    q_bounds.0 = q_bounds.0.min(value);
    q_bounds.1 = q_bounds.1.max(value);
}

/// Seed q_bounds from existing visited descendants of a search root.
///
/// run_search updates q_bounds online from each backed-up `total_value`, but a
/// reused subtree carries old visit/value stats whose mean values would be
/// normalized against degenerate `(0, 0)` bounds for the first few simulations,
/// collapsing every visited child's Q to 0.5. Walking the carried tree once
/// before the first simulation puts every existing mean value inside the
/// initial bounds. For a fresh root with no visits this returns `(0.0, 0.0)`,
/// matching the previous unconditional initialization.
pub(super) fn seed_q_bounds_from_tree(root: &DecisionNode) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    if root.visit_count > 0 {
        let v = root.mean_value();
        lo = lo.min(v);
        hi = hi.max(v);
    }

    let mut stack: Vec<NodeRef> = root
        .children()
        .map(|(_, chance)| NodeRef::Chance(chance))
        .collect();
    while let Some(node) = stack.pop() {
        let (visit_count, mean) = match node {
            NodeRef::Decision(d) => (d.visit_count, d.mean_value()),
            NodeRef::Chance(c) => (c.visit_count, c.mean_value()),
        };
        if visit_count > 0 {
            lo = lo.min(mean);
            hi = hi.max(mean);
        }
        match node {
            NodeRef::Decision(d) => {
                stack.extend(d.children().map(|(_, chance)| NodeRef::Chance(chance)))
            }
            NodeRef::Chance(c) => stack.extend(
                c.children()
                    .map(|(_, decision)| NodeRef::Decision(decision)),
            ),
        }
    }

    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

/// Decision nodes visited by one simulation, each with the `valid_actions` position taken.
type SimulationPath = Vec<(*mut DecisionNode, usize)>;

/// Run a single MCTS simulation using the provided leaf evaluator.
///
/// Uses raw pointers for tree traversal to track the path from root to leaf.
/// This is a common pattern in tree structures where we need mutable access
/// to nodes at multiple levels simultaneously. `path` is caller-provided scratch
/// space so repeated simulations reuse one allocation.
fn simulate<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    root: &mut DecisionNode,
    rng: &mut StdRng,
    q_bounds: &mut (f32, f32),
    path: &mut SimulationPath,
) -> SimulationOutcome {
    let root_cumulative_attack = root.state.attack as f32;
    path.clear();
    let mut path_attack_sum: f32 = 0.0;
    let mut current = root as *mut DecisionNode;

    loop {
        let node = unsafe { &mut *current };
        node.visit_count += 1;

        // No value tail beyond horizon.
        if node.state.placement_count >= config.max_placements {
            let total_value = root_cumulative_attack + path_attack_sum;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(path, total_value, config.track_value_history);
            return SimulationOutcome::HorizonEnd;
        }

        if node.is_terminal {
            let penalty = super::utils::compute_death_penalty(
                node.state.placement_count,
                config.max_placements,
                config.death_penalty,
            );
            let total_value = root_cumulative_attack + path_attack_sum - penalty;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(path, total_value, config.track_value_history);
            return SimulationOutcome::TerminalEnd;
        }

        if node.valid_actions.is_empty() {
            panic!("BUG: non-terminal DecisionNode has no valid actions");
        }

        let action_position = node.select_action_position(
            config.c_puct,
            *q_bounds,
            config.use_parent_value_for_unvisited_q,
        );
        let action_idx = node.valid_actions[action_position];
        let mut expanded_action = false;
        if node.child(action_position).is_none() {
            let child = match expand_action(node, action_idx) {
                Ok(child) => child,
                Err(ExpandActionError::InvariantViolation { action_idx }) => {
                    panic!(
                        "BUG: expand_action failed for action {} - action mask is inconsistent",
                        action_idx
                    );
                }
                Err(ExpandActionError::ChanceOutcomeInvariantViolation { outcome_idx }) => {
                    panic!(
                        "BUG: invalid chance outcome {} produced during action expansion",
                        outcome_idx
                    );
                }
                Err(ExpandActionError::EvaluatorFailed(error)) => {
                    panic!("MCTS leaf evaluation failed during expansion: {}", error);
                }
            };
            node.set_child(action_position, child);
            expanded_action = true;
        }

        let chance_node = node
            .child_mut(action_position)
            .expect("BUG: child should exist after expansion check");
        chance_node.visit_count += 1;
        path.push((current, action_position));
        path_attack_sum += chance_node.attack as f32;
        if chance_node.state.game_over {
            let penalty = super::utils::compute_death_penalty(
                chance_node.state.placement_count,
                config.max_placements,
                config.death_penalty,
            );
            let total_value = root_cumulative_attack + path_attack_sum - penalty;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(path, total_value, config.track_value_history);
            return if expanded_action {
                SimulationOutcome::Expansion
            } else {
                SimulationOutcome::TerminalEnd
            };
        }
        if chance_node.state.placement_count >= config.max_placements {
            let total_value = root_cumulative_attack + path_attack_sum;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(path, total_value, config.track_value_history);
            return if expanded_action {
                SimulationOutcome::Expansion
            } else {
                SimulationOutcome::HorizonEnd
            };
        }

        let chance_outcome = chance_node.select_piece_random(rng);
        if chance_node.child(chance_outcome).is_none() {
            let decision_child = match expand_chance(
                evaluator,
                chance_node,
                chance_outcome,
                config.max_placements,
            ) {
                Ok(child) => child,
                Err(ExpandActionError::InvariantViolation { action_idx }) => {
                    panic!(
                        "BUG: chance expansion used invalid action {} while evaluating leaf",
                        action_idx
                    );
                }
                Err(ExpandActionError::ChanceOutcomeInvariantViolation { outcome_idx }) => {
                    panic!(
                        "BUG: chance expansion received invalid outcome key {}",
                        outcome_idx
                    );
                }
                Err(ExpandActionError::EvaluatorFailed(error)) => {
                    panic!(
                        "MCTS leaf evaluation failed during chance expansion: {}",
                        error
                    );
                }
            };
            let overhang_fields = chance_node.overhang_fields;
            let decision_node = chance_node.set_child(chance_outcome, decision_child);

            let leaf_overhang = super::utils::compute_overhang_penalty(
                overhang_fields,
                config.overhang_penalty_weight,
            );
            let leaf_value = if decision_node.is_terminal {
                -super::utils::compute_death_penalty(
                    decision_node.state.placement_count,
                    config.max_placements,
                    config.death_penalty,
                )
            } else if decision_node.state.placement_count >= config.max_placements {
                0.0
            } else {
                decision_node.nn_value - leaf_overhang
            };
            let total_value = root_cumulative_attack + path_attack_sum + leaf_value;
            decision_node.set_initial_total_value_estimate(total_value);
            update_q_bounds(q_bounds, total_value);
            backup_with_value(path, total_value, config.track_value_history);
            return SimulationOutcome::Expansion;
        }

        current = chance_node
            .child_mut(chance_outcome)
            .expect("BUG: chance child should exist after expansion check")
            as *mut DecisionNode;
    }
}

/// Expand an action from a decision node (creates chance node)
///
/// Returns an error if action execution fails.
fn expand_action(
    parent: &DecisionNode,
    action_idx: usize,
) -> Result<ChanceNode, ExpandActionError> {
    let mut new_state = parent.state.mcts_clone();
    let attack = match new_state.execute_action_index(action_idx) {
        Some(attack) => attack,
        None => {
            return Err(ExpandActionError::InvariantViolation { action_idx });
        }
    };

    let overhang_fields = super::utils::count_overhang_fields(&new_state);
    let reveals_new_visible_piece = action_reveals_new_visible_piece(&parent.state, action_idx);
    let visible_queue_len = if reveals_new_visible_piece {
        QUEUE_SIZE.saturating_sub(1)
    } else {
        QUEUE_SIZE
    };
    new_state.truncate_queue(visible_queue_len);
    let outcomes = if reveals_new_visible_piece {
        let piece_mask = new_state.possible_next_piece_mask(new_state.piece_queue.len());
        if piece_mask == 0 {
            return Err(ExpandActionError::InvariantViolation { action_idx });
        }
        ChanceOutcomes::from_piece_mask(piece_mask)
    } else {
        ChanceOutcomes::from_keys(&[NO_CHANCE_OUTCOME])
    };

    Ok(ChanceNode::new(
        new_state,
        attack,
        overhang_fields,
        outcomes,
    ))
}

/// Expand a chance node for a specific piece (creates decision node)
///
/// The chance outcome either appends a newly revealed visible-tail piece (0..6) or is
/// a deterministic no-op (`NO_CHANCE_OUTCOME`) for hold-swap transitions where the
/// queue does not advance.
fn expand_chance<E: LeafEvaluator>(
    evaluator: &E,
    parent: &ChanceNode,
    chance_outcome: usize,
    max_placements: u32,
) -> Result<DecisionNode, ExpandActionError> {
    let mut new_state = parent.state.mcts_clone();

    if chance_outcome < NUM_PIECE_TYPES {
        new_state.push_queue_piece(chance_outcome);
    } else if chance_outcome != NO_CHANCE_OUTCOME {
        return Err(ExpandActionError::ChanceOutcomeInvariantViolation {
            outcome_idx: chance_outcome,
        });
    }

    let mut node = DecisionNode::new(new_state);
    let evaluation = evaluator
        .evaluate(&node.state, &node.valid_actions, max_placements)
        .map_err(ExpandActionError::EvaluatorFailed)?;
    node.set_nn_output_for_valid_actions_with_raw(
        evaluation.action_priors,
        evaluation.raw_value,
        evaluation.value,
    );

    Ok(node)
}

/// Backpropagate total episode value through the path.
///
/// Callers precompute:
///   total_value = root_cumulative_attack + path_rewards + leaf_value
/// and all nodes in the path receive that same total value.
fn backup_with_value(
    path: &[(*mut DecisionNode, usize)],
    total_value: f32,
    track_value_history: bool,
) {
    if path.is_empty() {
        return;
    }

    // All nodes on the path get the same total value
    for &(node_ptr, action_position) in path.iter() {
        // SAFETY: node_ptr was stored during the simulation traversal from valid
        // &mut DecisionNode references. The tree hasn't been modified, so pointers
        // remain valid. Each pointer in the path refers to a distinct node.
        let node = unsafe { &mut *node_ptr };

        // Update child stats (the ChanceNode we traversed through)
        if let Some(chance_child) = node.child_mut(action_position) {
            chance_child.value_sum += total_value;
            if track_value_history {
                chance_child
                    .value_history
                    .get_or_insert_with(Vec::new)
                    .push(total_value);
            }
        }

        // Update the DecisionNode itself
        node.value_sum += total_value;
        if track_value_history {
            node.value_history
                .get_or_insert_with(Vec::new)
                .push(total_value);
        }
    }
}

/// Compute statistics about the MCTS tree structure.
pub(super) fn compute_tree_stats(root: &DecisionNode) -> TreeStats {
    let mut total_nodes: u32 = 0;
    let mut num_leaves: u32 = 0;
    let mut children_sum: u32 = 0;
    let mut non_leaf_count: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut max_depth: u32 = 0;

    let mut stack: Vec<(NodeRef, u32)> = vec![(NodeRef::Decision(root), 0)];

    while let Some((node, depth)) = stack.pop() {
        total_nodes += 1;
        max_depth = max_depth.max(depth);

        match node {
            NodeRef::Decision(d) => {
                let stack_len_before = stack.len();
                stack.extend(
                    d.children()
                        .map(|(_, chance)| (NodeRef::Chance(chance), depth + 1)),
                );
                let num_children = (stack.len() - stack_len_before) as u32;
                if num_children == 0 {
                    num_leaves += 1;
                } else {
                    non_leaf_count += 1;
                    children_sum += num_children;
                }
            }
            NodeRef::Chance(c) => {
                max_attack = max_attack.max(c.attack);
                let stack_len_before = stack.len();
                stack.extend(
                    c.children()
                        .map(|(_, decision)| (NodeRef::Decision(decision), depth + 1)),
                );
                let num_children = (stack.len() - stack_len_before) as u32;
                if num_children == 0 {
                    num_leaves += 1;
                } else {
                    non_leaf_count += 1;
                    children_sum += num_children;
                }
            }
        }
    }

    let branching_factor = if non_leaf_count > 0 {
        children_sum as f32 / non_leaf_count as f32
    } else {
        0.0
    };

    TreeStats {
        branching_factor,
        num_leaves,
        total_nodes,
        max_depth,
        max_attack,
    }
}

fn create_search_rng(config: &MCTSConfig, env: &TetrisEnv) -> StdRng {
    if let Some(mcts_seed) = config.seed {
        let combined_seed = mcts_seed
            .wrapping_add(env.seed)
            .wrapping_add(env.placement_count as u64);
        StdRng::seed_from_u64(combined_seed)
    } else {
        StdRng::from_rng(thread_rng()).expect("Failed to create RNG from thread_rng")
    }
}

fn build_result_from_root(
    config: &MCTSConfig,
    root: &DecisionNode,
    rng: &mut StdRng,
    num_simulations: u32,
) -> MCTSResult {
    let mut result_policy = vec![0.0; NUM_ACTIONS];

    debug_assert!(
        root.children().map(|(_, c)| c.visit_count).sum::<u32>() > 0,
        "MCTS should have visits after simulations"
    );

    let greedy_action = root
        .children()
        .max_by(|(idx_a, child_a), (idx_b, child_b)| {
            child_a
                .visit_count
                .cmp(&child_b.visit_count)
                .then_with(|| idx_b.cmp(idx_a))
        })
        .map(|(idx, _)| idx)
        .expect("MCTS root should have children after simulations");

    if config.temperature == 0.0 {
        result_policy[greedy_action] = 1.0;
    } else {
        for (action_idx, child) in root.children() {
            result_policy[action_idx] = (child.visit_count as f32).powf(1.0 / config.temperature);
        }
        let sum: f32 = result_policy.iter().sum();
        if sum > 0.0 {
            for probability in &mut result_policy {
                *probability /= sum;
            }
        }
    }

    let should_sample_action =
        config.visit_sampling_epsilon > 0.0 && rng.gen::<f32>() < config.visit_sampling_epsilon;
    let action = if should_sample_action {
        sample_action_from_policy(&result_policy, rng)
    } else {
        greedy_action
    };

    let root_value = if root.visit_count > 0 {
        root.value_sum / root.visit_count as f32
    } else {
        0.0
    };

    MCTSResult {
        policy: result_policy,
        action,
        value: root_value,
        num_simulations,
    }
}

/// Core search loop: run simulations on a root and return results.
///
/// Works for both fresh roots and reused subtree roots. Takes ownership
/// of the root and returns it alongside the result for tree reuse.
pub(super) fn run_search<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    mut root: DecisionNode,
    add_noise: bool,
) -> (
    MCTSResult,
    DecisionNode,
    TreeStats,
    TraversalStats,
    (f32, f32),
) {
    let mut rng = create_search_rng(config, &root.state);
    if add_noise {
        root.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon, &mut rng);
    }

    let mut traversal_stats = TraversalStats::default();
    let mut q_bounds = seed_q_bounds_from_tree(&root);
    let num_new_simulations = config.new_simulations_for_root(root.visit_count);
    let mut path = SimulationPath::new();
    for _ in 0..num_new_simulations {
        match simulate(
            config,
            evaluator,
            &mut root,
            &mut rng,
            &mut q_bounds,
            &mut path,
        ) {
            SimulationOutcome::Expansion => traversal_stats.expansions += 1,
            SimulationOutcome::TerminalEnd => traversal_stats.terminal_ends += 1,
            SimulationOutcome::HorizonEnd => traversal_stats.horizon_ends += 1,
        }
    }
    debug_assert_eq!(traversal_stats.total(), num_new_simulations);

    let mcts_result = build_result_from_root(config, &root, &mut rng, num_new_simulations);
    let tree_stats = compute_tree_stats(&root);
    (mcts_result, root, tree_stats, traversal_stats, q_bounds)
}

/// Run MCTS search with NN guidance from a fresh root.
pub(super) fn search_internal(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &TetrisEnv,
    policy: Vec<f32>,
    nn_value: f32,
    add_noise: bool,
) -> (
    MCTSResult,
    DecisionNode,
    TreeStats,
    TraversalStats,
    (f32, f32),
) {
    let evaluator = neural_leaf_evaluator(config, nn);
    let root = neural_root(config, env, &policy, nn_value);
    run_search(config, &evaluator, root, add_noise)
}

fn neural_leaf_evaluator<'a>(config: &MCTSConfig, nn: &'a TetrisNN) -> NeuralLeafEvaluator<'a> {
    NeuralLeafEvaluator {
        nn,
        nn_value_weight: config.nn_value_weight,
        prediction_noise: PredictionNoiseSettings {
            policy_mean: config.policy_noise_mean,
            policy_std: config.policy_noise_std,
            value_mean: config.value_noise_mean,
            value_std: config.value_noise_std,
            seed: config.prediction_noise_seed,
        },
    }
}

fn neural_root(
    config: &MCTSConfig,
    env: &TetrisEnv,
    policy: &[f32],
    nn_value: f32,
) -> DecisionNode {
    let mut root = DecisionNode::new(env.clone());
    root.set_nn_output_with_raw(
        policy,
        nn_value,
        scale_nn_value(nn_value, config.nn_value_weight),
    );
    root.set_initial_total_value_estimate(root.state.attack as f32 + root.nn_value);
    root
}

/// Run one fresh NN-guided search (no root noise) and record the root action visit
/// counts after each requested simulation count.
///
/// Because simulations are sequential, the snapshot at `k` is exactly the tree a
/// standalone `k`-simulation search would produce with the same seed, so a single long
/// search yields the whole visit-count trajectory. `snapshot_counts` must be ascending.
pub(super) fn search_root_visit_snapshots(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &TetrisEnv,
    policy: &[f32],
    nn_value: f32,
    snapshot_counts: &[u32],
) -> Vec<Vec<u32>> {
    let evaluator = neural_leaf_evaluator(config, nn);
    let root = neural_root(config, env, policy, nn_value);
    root_visit_snapshots(config, &evaluator, root, snapshot_counts)
}

fn root_visit_snapshots<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    mut root: DecisionNode,
    snapshot_counts: &[u32],
) -> Vec<Vec<u32>> {
    assert!(
        snapshot_counts.windows(2).all(|w| w[0] <= w[1]),
        "snapshot_counts must be ascending"
    );
    let mut rng = create_search_rng(config, &root.state);
    let mut q_bounds = seed_q_bounds_from_tree(&root);
    let mut path = SimulationPath::new();

    let mut snapshots = Vec::with_capacity(snapshot_counts.len());
    let mut completed: u32 = 0;
    for &target in snapshot_counts {
        while completed < target {
            simulate(
                config,
                evaluator,
                &mut root,
                &mut rng,
                &mut q_bounds,
                &mut path,
            );
            completed += 1;
        }
        snapshots.push(root_action_visits(&root));
    }
    snapshots
}

/// Root visit counts indexed by action index (zero for unexpanded actions).
fn root_action_visits(root: &DecisionNode) -> Vec<u32> {
    let mut visits = vec![0u32; NUM_ACTIONS];
    for (action_idx, child) in root.children() {
        visits[action_idx] = child.visit_count;
    }
    visits
}

/// Run MCTS search without NN guidance (uniform priors and zero value).
pub(crate) fn search_internal_without_nn(
    config: &MCTSConfig,
    env: &TetrisEnv,
    add_noise: bool,
) -> (
    MCTSResult,
    DecisionNode,
    TreeStats,
    TraversalStats,
    (f32, f32),
) {
    let root_policy = env.uniform_policy();
    let mut root = DecisionNode::new(env.clone());
    root.set_nn_output(&root_policy, 0.0);
    root.set_initial_total_value_estimate(root.state.attack as f32 + root.nn_value);
    run_search(config, &BootstrapLeafEvaluator, root, add_noise)
}

/// Extract the subtree corresponding to a chosen action and realized chance outcome.
///
/// `chance_outcome` is either:
/// - piece type 0..6 for transitions that revealed a new visible-tail queue piece, or
/// - `NO_CHANCE_OUTCOME` for deterministic hold-swap transitions.
///
/// Returns the extracted DecisionNode, or None if the subtree path wasn't explored.
/// Value sums require no adjustment: each backed-up value equals
/// `root.state.attack + path_attack_sum + leaf_value`, and because
/// `old_root.state.attack + step_attack == new_root.state.attack`, old and new
/// simulations contribute identical magnitudes for the same leaf path.
pub(super) fn extract_subtree(
    mut root: DecisionNode,
    action_idx: usize,
    chance_outcome: usize,
) -> Option<DecisionNode> {
    // Take the ChanceNode child for the chosen action, then the DecisionNode child
    // for the realized chance outcome.
    root.take_child_for_action(action_idx)?
        .take_child(chance_outcome)
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::game::action_space::HOLD_ACTION_INDEX;

    use super::*;

    fn simulate_once<E: LeafEvaluator>(
        config: &MCTSConfig,
        evaluator: &E,
        root: &mut DecisionNode,
        rng: &mut StdRng,
        q_bounds: &mut (f32, f32),
    ) -> SimulationOutcome {
        simulate(
            config,
            evaluator,
            root,
            rng,
            q_bounds,
            &mut SimulationPath::new(),
        )
    }

    struct ConstantEvaluator {
        value: f32,
    }

    impl LeafEvaluator for ConstantEvaluator {
        fn evaluate(
            &self,
            _state: &TetrisEnv,
            valid_actions: &[usize],
            _max_placements: u32,
        ) -> Result<LeafEvaluation, String> {
            Ok(LeafEvaluation {
                action_priors: uniform_action_priors_for_valid_actions(valid_actions.len()),
                raw_value: self.value,
                value: self.value,
            })
        }
    }

    #[test]
    fn test_root_visit_snapshots_match_standalone_searches() {
        let env = TetrisEnv::with_seed(10, 20, 42);
        let mut config = MCTSConfig::default();
        config.seed = Some(7);
        let make_root = || {
            let root_policy = uniform_action_priors_for_valid_actions(
                env.get_cached_valid_action_indices().len(),
            );
            let mut root = DecisionNode::new(env.clone());
            root.set_nn_output_for_valid_actions(&root_policy, 0.0);
            root.set_initial_total_value_estimate(root.state.attack as f32);
            root
        };

        let evaluator = ConstantEvaluator { value: 0.5 };
        let snapshot_counts = [0, 1, 5, 40, 120];
        let snapshots = root_visit_snapshots(&config, &evaluator, make_root(), &snapshot_counts);

        assert_eq!(snapshots.len(), snapshot_counts.len());
        assert!(snapshots[0].iter().all(|&v| v == 0));
        for (&count, snapshot) in snapshot_counts.iter().zip(&snapshots).skip(1) {
            config.num_simulations = count;
            let (_, root, _, _, _) = run_search(&config, &evaluator, make_root(), false);
            assert_eq!(
                snapshot,
                &root_action_visits(&root),
                "snapshot at {count} sims diverged"
            );
            assert_eq!(snapshot.iter().sum::<u32>(), count);
        }
    }

    #[test]
    fn test_scale_nn_value() {
        assert_eq!(scale_nn_value(7.0, 0.0), 0.0);
        assert!((scale_nn_value(7.0, 0.01) - 0.07).abs() < 1e-6);
        assert_eq!(scale_nn_value(-3.0, 1.0), -3.0);
    }

    #[test]
    fn test_expand_action_ignores_hidden_sixth_piece_for_queue_advancing_actions() {
        let base_env = TetrisEnv::with_seed(10, 20, 123);
        assert!(
            base_env.piece_queue.len() > QUEUE_SIZE,
            "test requires at least 6 queued pieces"
        );

        let env_a = base_env.clone();
        let mut env_b = base_env;
        let hidden_idx = QUEUE_SIZE;
        let hidden_piece = env_a.piece_queue.get(hidden_idx).unwrap();
        let alt_hidden_piece = (hidden_piece + 1) % NUM_PIECE_TYPES;
        env_b.piece_queue.set(hidden_idx, alt_hidden_piece);

        let root_a = DecisionNode::new(env_a);
        let root_b = DecisionNode::new(env_b);
        let placement_action = root_a
            .valid_actions
            .iter()
            .copied()
            .find(|&action_idx| action_idx != HOLD_ACTION_INDEX)
            .expect("Expected at least one placement action");
        assert!(
            root_b.valid_actions.contains(&placement_action),
            "Placement action should remain valid after hidden-piece mutation"
        );

        let chance_a = expand_action(&root_a, placement_action)
            .expect("expand_action should succeed for placement");
        let chance_b = expand_action(&root_b, placement_action)
            .expect("expand_action should succeed for placement");

        assert_eq!(chance_a.state.get_queue_len(), QUEUE_SIZE - 1);
        assert_eq!(chance_b.state.get_queue_len(), QUEUE_SIZE - 1);
        assert_eq!(
            chance_a.state.get_queue(QUEUE_SIZE - 1),
            chance_b.state.get_queue(QUEUE_SIZE - 1),
            "Post-action observable queue should not depend on hidden piece #6"
        );
        assert_eq!(
            chance_a.outcomes, chance_b.outcomes,
            "Chance outcomes should not depend on hidden piece #6"
        );
    }

    #[test]
    fn test_expand_action_hold_swap_uses_deterministic_no_chance_outcome() {
        let mut env = TetrisEnv::with_seed(10, 20, 1234);
        let current_piece = env
            .get_current_piece()
            .expect("current piece should exist")
            .piece_type;
        let hold_piece = (current_piece + 1) % NUM_PIECE_TYPES;
        env.hold_piece = Some(hold_piece);
        env.hold_used = false;
        env.hold_piece_bag_position = Some(env.current_piece_bag_position);

        let root = DecisionNode::new(env.clone());
        assert!(
            root.valid_actions.contains(&HOLD_ACTION_INDEX),
            "Hold should be valid when hold slot is occupied and hold is unused"
        );

        let chance = expand_action(&root, HOLD_ACTION_INDEX)
            .expect("expand_action should succeed for hold-swap");

        assert_eq!(
            chance.state.get_queue_len(),
            QUEUE_SIZE,
            "Hold-swap chance state should keep full visible queue"
        );
        assert_eq!(
            chance.state.get_queue(QUEUE_SIZE),
            env.get_queue(QUEUE_SIZE),
            "Hold-swap should not advance the queue"
        );
        assert_eq!(chance.outcomes.as_slice(), &[NO_CHANCE_OUTCOME as u8]);
    }

    #[test]
    fn test_simulate_terminal_after_expansion_uses_death_penalty() {
        let mut env = TetrisEnv::new(10, 20);
        for x in 0..env.width {
            env.board[x] = 1;
            env.board[env.width + x] = 1;
        }

        let mut root = DecisionNode::new(env);
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[HOLD_ACTION_INDEX] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 100;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 123.0 };
        let mut rng = StdRng::seed_from_u64(1234);
        let outcome = simulate_once(
            &config,
            &evaluator,
            &mut root,
            &mut rng,
            &mut (f32::INFINITY, f32::NEG_INFINITY),
        );
        assert_eq!(outcome, SimulationOutcome::Expansion);

        let expected_value = -super::super::utils::compute_death_penalty(
            0,
            config.max_placements,
            config.death_penalty,
        );
        assert!(
            (root.value_sum - expected_value).abs() < 1e-6,
            "Expected root value_sum to use death penalty ({}), got {}",
            expected_value,
            root.value_sum
        );
    }

    #[test]
    fn test_simulate_stops_at_max_placements_horizon() {
        let mut env = TetrisEnv::new(10, 20);
        env.placement_count = 3;
        let mut root = DecisionNode::new(env);
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[HOLD_ACTION_INDEX] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 42.0 };
        let mut rng = StdRng::seed_from_u64(7);
        let outcome = simulate_once(
            &config,
            &evaluator,
            &mut root,
            &mut rng,
            &mut (f32::INFINITY, f32::NEG_INFINITY),
        );
        assert_eq!(outcome, SimulationOutcome::HorizonEnd);

        assert_eq!(root.value_sum, 0.0);
        assert!(!root.has_children());
    }

    #[test]
    #[should_panic(expected = "non-terminal DecisionNode has no valid actions")]
    fn test_simulate_panics_when_non_terminal_node_has_no_valid_actions() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env);
        root.valid_actions.clear();
        root.action_priors.clear();

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 0.0 };
        let mut rng = StdRng::seed_from_u64(123);
        simulate_once(
            &config,
            &evaluator,
            &mut root,
            &mut rng,
            &mut (f32::INFINITY, f32::NEG_INFINITY),
        );
    }

    #[test]
    fn test_simulate_horizon_after_last_placement_uses_immediate_reward_only() {
        let mut env = TetrisEnv::new(10, 20);
        env.placement_count = 2;
        let mut root = DecisionNode::new(env);
        let placement_action = root
            .valid_actions
            .iter()
            .copied()
            .find(|&action_idx| action_idx != HOLD_ACTION_INDEX)
            .expect("Expected at least one placement action");
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[placement_action] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 123.0 };
        let mut rng = StdRng::seed_from_u64(99);
        let outcome = simulate_once(
            &config,
            &evaluator,
            &mut root,
            &mut rng,
            &mut (f32::INFINITY, f32::NEG_INFINITY),
        );
        assert_eq!(outcome, SimulationOutcome::Expansion);

        let placement_position = root
            .action_position(placement_action)
            .expect("placement action should be valid");
        let chance_node = root
            .child(placement_position)
            .expect("Expected expanded child to be a ChanceNode");
        let expected_value = chance_node.attack as f32;
        assert!(
            (root.value_sum - expected_value).abs() < 1e-6,
            "Expected horizon backup to ignore NN tail value and use immediate reward only (expected {}, got {})",
            expected_value,
            root.value_sum
        );
    }

    #[test]
    fn test_root_greedy_tie_break_prefers_lower_action_index() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env.clone());
        let root_policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        root.set_nn_output(&root_policy, 0.0);

        let left_action = root.valid_actions[1];
        let right_action = root.valid_actions[3];
        let mut left_env = env.clone();
        left_env.placement_count = 1;
        let mut left_child = ChanceNode::new(left_env, 0, 0, ChanceOutcomes::default());
        left_child.visit_count = 5;
        let mut right_env = env;
        right_env.placement_count = 1;
        let mut right_child = ChanceNode::new(right_env, 0, 0, ChanceOutcomes::default());
        right_child.visit_count = 5;
        root.set_child_for_action(right_action, right_child);
        root.set_child_for_action(left_action, left_child);

        let mut config = MCTSConfig::default();
        config.temperature = 0.0;
        config.visit_sampling_epsilon = 0.0;

        let mut rng = StdRng::seed_from_u64(11);
        let result = build_result_from_root(&config, &root, &mut rng, config.num_simulations);
        assert!(left_action < right_action);
        assert_eq!(result.action, left_action);
    }

    #[test]
    fn test_extract_subtree_returns_matching_decision_node() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 200;
        config.reuse_tree = true;

        let evaluator = ConstantEvaluator { value: 1.0 };

        // Build a tree with a fresh search
        let root_policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env.clone());
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (0.0_f32, 0.0_f32);
        for _ in 0..config.num_simulations {
            simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }

        // Find the best action (most visits)
        let (best_action, chance_node) = root
            .children()
            .max_by_key(|(_, child)| child.visit_count)
            .expect("Root should have children");

        // Find an explored piece under the chance node for the best action
        let (explored_piece, explored_child) = chance_node
            .children()
            .next()
            .expect("ChanceNode should have at least one child");
        let old_visit_count = explored_child.visit_count;

        // Extract the subtree
        let subtree = extract_subtree(root, best_action, explored_piece);
        assert!(
            subtree.is_some(),
            "Should extract subtree for explored piece"
        );

        let reused_root = subtree.unwrap();
        assert_eq!(
            reused_root.visit_count, old_visit_count,
            "Reused root should preserve visit count"
        );
        assert!(
            !reused_root.valid_actions.is_empty(),
            "Reused root should have valid actions"
        );
    }

    #[test]
    fn test_extract_subtree_returns_none_for_unexplored_piece() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 5; // Very few sims to leave some pieces unexplored

        let evaluator = ConstantEvaluator { value: 1.0 };
        let root_policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env.clone());
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (0.0_f32, 0.0_f32);
        for _ in 0..config.num_simulations {
            simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }

        // Find an action that was explored
        let (action, chance_node) = root
            .children()
            .next()
            .expect("Should have at least one child");

        // Find a piece that was NOT explored in this chance node
        let unexplored_piece = (0..7).find(|&p| chance_node.child(p).is_none());
        if let Some(piece) = unexplored_piece {
            let result = extract_subtree(root, action, piece);
            assert!(result.is_none(), "Should return None for unexplored piece");
        }
        // If all pieces were explored (unlikely with 5 sims), test is vacuously true
    }

    #[test]
    fn test_run_search_on_reused_tree_adds_simulations() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 50;

        let evaluator = ConstantEvaluator { value: 1.0 };
        let root_policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env.clone());
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);

        // Run initial search
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (0.0_f32, 0.0_f32);
        for _ in 0..config.num_simulations {
            simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }
        let visits_after_first = root.visit_count;

        // Continue search via run_search (simulates tree reuse)
        let (result, root, _tree_stats, traversal_stats, _q_bounds) =
            run_search(&config, &evaluator, root, false);
        assert_eq!(result.num_simulations, config.num_simulations);
        assert_eq!(traversal_stats.total(), config.num_simulations);

        assert_eq!(
            root.visit_count,
            visits_after_first + config.num_simulations,
            "Visit count should increase by num_simulations"
        );
        assert!(
            result.policy.iter().any(|&p| p > 0.0),
            "Policy should have non-zero entries"
        );
    }

    #[test]
    fn test_run_search_counts_reused_visits_toward_budget() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 50;
        config.count_reused_visits = true;
        config.min_new_simulations = 5;

        let evaluator = ConstantEvaluator { value: 1.0 };
        let root_policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env);
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);

        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (0.0_f32, 0.0_f32);
        for _ in 0..30 {
            simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }
        assert_eq!(root.visit_count, 30);

        // 30 carried visits: only the remaining 20 of the 50-visit budget are run.
        let (result, root, _tree_stats, traversal_stats, _q_bounds) =
            run_search(&config, &evaluator, root, false);
        assert_eq!(traversal_stats.total(), 20);
        assert_eq!(result.num_simulations, 20);
        assert_eq!(root.visit_count, 50);

        // Budget already exceeded: the min_new_simulations floor still applies.
        let (result, root, _tree_stats, traversal_stats, _q_bounds) =
            run_search(&config, &evaluator, root, false);
        assert_eq!(traversal_stats.total(), 5);
        assert_eq!(result.num_simulations, 5);
        assert_eq!(root.visit_count, 55);
    }

    #[test]
    fn test_simulate_revisiting_terminal_child_reports_terminal_outcome() {
        let mut env = TetrisEnv::new(10, 20);
        for x in 0..env.width {
            env.board[x] = 1;
            env.board[env.width + x] = 1;
        }

        let mut root = DecisionNode::new(env);
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[HOLD_ACTION_INDEX] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 100;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 0.0 };
        let mut rng = StdRng::seed_from_u64(1234);
        let mut q_bounds = (0.0_f32, 0.0_f32);
        let first = simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        let second = simulate_once(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);

        assert_eq!(first, SimulationOutcome::Expansion);
        assert_eq!(second, SimulationOutcome::TerminalEnd);
    }

    #[test]
    fn test_seed_q_bounds_from_tree_returns_zero_for_fresh_root() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env);
        let policy = uniform_action_priors_for_valid_actions(root.valid_actions.len());
        root.set_nn_output_for_valid_actions(&policy, 0.0);

        let bounds = seed_q_bounds_from_tree(&root);
        assert_eq!(bounds, (0.0, 0.0));
    }

    #[test]
    fn test_seed_q_bounds_from_tree_covers_visited_node_means_after_search() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 80;
        config.max_placements = 8;

        let evaluator = ConstantEvaluator { value: 3.5 };
        let policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env);
        root.set_nn_output_for_valid_actions(&policy, evaluator.value);

        // Drive q_bounds online via direct simulate calls so we know the exact
        // online bounds the prior `(0.0, 0.0)` initialization would have produced.
        let mut rng = StdRng::seed_from_u64(7);
        let mut online_bounds = (0.0_f32, 0.0_f32);
        for _ in 0..config.num_simulations {
            simulate_once(&config, &evaluator, &mut root, &mut rng, &mut online_bounds);
        }

        let seeded = seed_q_bounds_from_tree(&root);
        assert!(
            seeded.0 <= seeded.1,
            "lo {} must not exceed hi {}",
            seeded.0,
            seeded.1
        );
        // Mean values are inside the per-sample online bounds, so seeding from
        // means must produce a tighter (or equal) range.
        assert!(
            seeded.0 >= online_bounds.0 - 1e-5,
            "seeded lo {} must be >= online lo {}",
            seeded.0,
            online_bounds.0
        );
        assert!(
            seeded.1 <= online_bounds.1 + 1e-5,
            "seeded hi {} must be <= online hi {}",
            seeded.1,
            online_bounds.1
        );

        // The visited root and its visited children all sit inside the seeded bounds.
        let root_mean = root.mean_value();
        assert!(
            root_mean >= seeded.0 - 1e-5 && root_mean <= seeded.1 + 1e-5,
            "root mean {} must lie inside seeded bounds {:?}",
            root_mean,
            seeded
        );
        for (_, child) in root.children() {
            if child.visit_count > 0 {
                let m = child.mean_value();
                assert!(
                    m >= seeded.0 - 1e-5 && m <= seeded.1 + 1e-5,
                    "child mean {} must lie inside seeded bounds {:?}",
                    m,
                    seeded
                );
            }
        }
    }

    #[test]
    fn test_run_search_seeds_q_bounds_from_reused_subtree() {
        let env = TetrisEnv::with_seed(10, 20, 5);
        let mut config = MCTSConfig::default();
        config.num_simulations = 60;
        config.max_placements = 8;
        // Non-zero overhang penalty causes leaf values to vary by board state, so
        // mean values differ across visited children rather than all collapsing to a
        // single constant.
        config.overhang_penalty_weight = 1.0;

        let evaluator = ConstantEvaluator { value: 2.0 };
        let policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env);
        root.set_nn_output_for_valid_actions(&policy, evaluator.value);

        // First search: builds up children with non-zero mean values.
        let (_result, root_after, _tree_stats, _traversal_stats, _q_bounds) =
            run_search(&config, &evaluator, root, false);
        let seeded_bounds = seed_q_bounds_from_tree(&root_after);
        assert!(
            seeded_bounds.0 < seeded_bounds.1,
            "Expected non-degenerate seeded bounds after first search, got {:?}",
            seeded_bounds
        );

        // Second search reuses the tree. Final q_bounds must enclose the seeded range
        // because run_search seeds from the carried tree before running new sims.
        let (_result, _root, _tree_stats, _traversal_stats, final_bounds) =
            run_search(&config, &evaluator, root_after, false);
        assert!(
            final_bounds.0 <= seeded_bounds.0 + 1e-5,
            "Reused-search final lo {} must not exceed seeded lo {}",
            final_bounds.0,
            seeded_bounds.0
        );
        assert!(
            final_bounds.1 >= seeded_bounds.1 - 1e-5,
            "Reused-search final hi {} must not be below seeded hi {}",
            final_bounds.1,
            seeded_bounds.1
        );
    }

    #[test]
    fn test_run_search_traversal_stats_sum_to_num_simulations() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 40;
        config.max_placements = 6;

        let evaluator = ConstantEvaluator { value: 0.0 };
        let root_policy =
            uniform_action_priors_for_valid_actions(env.get_cached_valid_action_indices().len());
        let mut root = DecisionNode::new(env.clone());
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);

        let (_result, _root, _tree_stats, traversal_stats, _q_bounds) =
            run_search(&config, &evaluator, root, false);

        assert_eq!(traversal_stats.total(), config.num_simulations);
        assert!(traversal_stats.expansions > 0);
    }
}
