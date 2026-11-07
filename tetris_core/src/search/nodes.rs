//! MCTS Node Types
//!
//! Decision nodes (player moves) and chance nodes (piece spawns).
//!
//! Children are stored densely: a decision node keeps its chance children in a vector aligned
//! with `valid_actions`, and a chance node keeps its decision children in a fixed array indexed
//! by chance outcome key. This keeps selection and backup free of hashing.

use rand::prelude::*;
use rand::rngs::StdRng;

use crate::game::constants::NUM_PIECE_TYPES;
use crate::game::env::TetrisEnv;

use super::utils::sample_dirichlet;
#[cfg(test)]
use crate::game::action_space::NUM_ACTIONS;

const Q_NORMALIZATION_EPSILON: f32 = 1e-6;

/// Sentinel chance outcome for transitions where no new visible queue piece is revealed.
/// This occurs on hold-swap actions (holding with an already occupied hold slot).
pub(crate) const NO_CHANCE_OUTCOME: usize = NUM_PIECE_TYPES;

/// Number of distinct chance outcome keys: one per piece type plus `NO_CHANCE_OUTCOME`.
const NUM_CHANCE_OUTCOMES: usize = NUM_PIECE_TYPES + 1;

fn mean_value(visit_count: u32, value_sum: f32) -> f32 {
    if visit_count > 0 {
        value_sum / visit_count as f32
    } else {
        0.0
    }
}

fn normalize_q_value(q: f32, q_min: f32, q_max: f32) -> f32 {
    let range = q_max - q_min;
    if !range.is_finite() || range.abs() < Q_NORMALIZATION_EPSILON {
        return 0.5;
    }
    (q - q_min) / range
}

/// Borrowed reference to either node type, for whole-tree traversals.
#[derive(Clone, Copy)]
pub(super) enum NodeRef<'a> {
    Decision(&'a DecisionNode),
    Chance(&'a ChanceNode),
}

/// Decision node where player chooses an action
#[derive(Clone)]
pub struct DecisionNode {
    /// Game state at this node
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Expanded chance children aligned with `valid_actions` (`children[i]` follows
    /// `valid_actions[i]`). Empty (no allocation) until the first child is expanded.
    children: Vec<Option<Box<ChanceNode>>>,
    /// Valid action indices for this state
    pub valid_actions: Vec<usize>,
    /// Cached priors for valid actions (from neural network)
    pub action_priors: Vec<f32>,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Search leaf value estimate used by MCTS backups.
    /// In NN mode this is after applying `nn_value_weight`.
    pub nn_value: f32,
    /// Raw unweighted neural-network value estimate for this state.
    pub raw_nn_value: f32,
    /// Total value estimate that was initially backed up when this node was expanded.
    /// Used as a first-play urgency baseline for unvisited action children when enabled.
    pub initial_total_value_estimate: f32,
    /// All backed-up total values that contributed to value_sum (only when track_value_history=true)
    pub value_history: Option<Vec<f32>>,
}

impl DecisionNode {
    fn normalize_action_priors(&mut self) {
        let sum: f32 = self.action_priors.iter().sum();
        if sum > 0.0 {
            for prior in &mut self.action_priors {
                *prior /= sum;
            }
        }
    }

    pub fn new(state: TetrisEnv) -> Self {
        let is_terminal = state.game_over;

        // Get valid actions
        let valid_actions = if is_terminal {
            Vec::new()
        } else {
            get_valid_action_indices(&state)
        };

        DecisionNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: Vec::new(),
            valid_actions,
            action_priors: Vec::new(),
            is_terminal,
            nn_value: 0.0,
            raw_nn_value: 0.0,
            initial_total_value_estimate: 0.0,
            value_history: None,
        }
    }

    /// Set priors and value from neural network output
    pub fn set_nn_output(&mut self, policy: &[f32], value: f32) {
        self.set_nn_output_with_raw(policy, value, value);
    }

    /// Set priors and both raw/scaled values from neural network output.
    pub fn set_nn_output_with_raw(&mut self, policy: &[f32], raw_value: f32, value: f32) {
        self.action_priors = self.valid_actions.iter().map(|&idx| policy[idx]).collect();
        self.normalize_action_priors();
        self.raw_nn_value = raw_value;
        self.nn_value = value;
    }

    /// Set priors and value from action priors already aligned with valid_actions.
    pub fn set_nn_output_for_valid_actions(&mut self, action_priors: &[f32], value: f32) {
        self.set_nn_output_for_valid_actions_with_raw(action_priors.to_vec(), value, value);
    }

    /// Set priors and both raw/scaled values from aligned action priors.
    pub fn set_nn_output_for_valid_actions_with_raw(
        &mut self,
        action_priors: Vec<f32>,
        raw_value: f32,
        value: f32,
    ) {
        assert_eq!(
            action_priors.len(),
            self.valid_actions.len(),
            "NN action priors len {} does not match valid actions len {}",
            action_priors.len(),
            self.valid_actions.len()
        );
        self.action_priors = action_priors;
        self.normalize_action_priors();
        self.raw_nn_value = raw_value;
        self.nn_value = value;
    }

    /// Add Dirichlet noise to priors (for root exploration)
    pub fn add_dirichlet_noise(&mut self, alpha: f32, epsilon: f32, rng: &mut StdRng) {
        let noise = sample_dirichlet(alpha, self.action_priors.len(), rng);
        for (prior, n) in self.action_priors.iter_mut().zip(noise.iter()) {
            *prior = (1.0 - epsilon) * *prior + epsilon * n;
        }
    }

    pub fn set_initial_total_value_estimate(&mut self, value: f32) {
        self.initial_total_value_estimate = value;
    }

    /// Select the best action using PUCT with global min-max Q normalization.
    /// `q_bounds` is the global (q_min, q_max) tracked across all simulations so far.
    ///
    /// Returns the position of the selected action in `valid_actions`. Ties prefer the
    /// lower action index.
    pub fn select_action_position(
        &self,
        c_puct: f32,
        q_bounds: (f32, f32),
        use_parent_value_for_unvisited_q: bool,
    ) -> usize {
        let sqrt_total = (self.visit_count as f32).sqrt();
        let mut best_position = 0;
        let mut best_action = self.valid_actions[0];
        let mut best_value = f32::NEG_INFINITY;
        let unvisited_q = if use_parent_value_for_unvisited_q {
            self.initial_total_value_estimate
        } else {
            0.0
        };

        for (position, &action_idx) in self.valid_actions.iter().enumerate() {
            let prior = self.action_priors[position];
            let (q, n) = match self.child(position) {
                Some(child) => (child.mean_value(), child.visit_count),
                None => (unvisited_q, 0),
            };
            let transformed_q = normalize_q_value(q, q_bounds.0, q_bounds.1);
            let u = c_puct * prior * sqrt_total / (1.0 + n as f32);
            let value = transformed_q + u;

            if value > best_value || (value == best_value && action_idx < best_action) {
                best_value = value;
                best_action = action_idx;
                best_position = position;
            }
        }

        best_position
    }

    /// Position of `action_idx` within `valid_actions`, if it is valid here.
    pub fn action_position(&self, action_idx: usize) -> Option<usize> {
        self.valid_actions
            .iter()
            .position(|&valid_action| valid_action == action_idx)
    }

    /// Expanded chance child reached through `valid_actions[position]`.
    #[inline]
    pub fn child(&self, position: usize) -> Option<&ChanceNode> {
        self.children
            .get(position)
            .and_then(|child| child.as_deref())
    }

    /// Mutable expanded chance child reached through `valid_actions[position]`.
    #[inline]
    pub fn child_mut(&mut self, position: usize) -> Option<&mut ChanceNode> {
        self.children
            .get_mut(position)
            .and_then(|child| child.as_deref_mut())
    }

    /// Store the expanded chance child for `valid_actions[position]`.
    pub fn set_child(&mut self, position: usize, child: ChanceNode) -> &mut ChanceNode {
        assert!(
            position < self.valid_actions.len(),
            "child position {} out of range for {} valid actions",
            position,
            self.valid_actions.len()
        );
        if self.children.is_empty() {
            self.children.resize_with(self.valid_actions.len(), || None);
        }
        let slot = &mut self.children[position];
        debug_assert!(slot.is_none(), "child {position} expanded twice");
        slot.insert(Box::new(child))
    }

    /// Store the expanded chance child for a valid `action_idx`.
    #[cfg(test)]
    pub fn set_child_for_action(&mut self, action_idx: usize, child: ChanceNode) {
        let position = self
            .action_position(action_idx)
            .unwrap_or_else(|| panic!("action {action_idx} is not valid at this node"));
        self.set_child(position, child);
    }

    /// Detach and return the chance child reached by `action_idx`, if expanded.
    pub fn take_child_for_action(&mut self, action_idx: usize) -> Option<ChanceNode> {
        let position = self.action_position(action_idx)?;
        self.children.get_mut(position)?.take().map(|child| *child)
    }

    /// Expanded children as `(action_idx, child)`, in `valid_actions` order.
    pub fn children(&self) -> impl Iterator<Item = (usize, &ChanceNode)> + '_ {
        self.children
            .iter()
            .zip(self.valid_actions.iter())
            .filter_map(|(child, &action_idx)| child.as_deref().map(|child| (action_idx, child)))
    }

    pub fn has_children(&self) -> bool {
        self.children.iter().any(Option::is_some)
    }
}

impl DecisionNode {
    pub fn mean_value(&self) -> f32 {
        mean_value(self.visit_count, self.value_sum)
    }
}

/// Possible chance outcome keys of a chance node, ascending, stored inline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChanceOutcomes {
    keys: [u8; NUM_CHANCE_OUTCOMES],
    len: u8,
}

impl ChanceOutcomes {
    /// Outcomes from explicit keys (piece types `0..7` or `NO_CHANCE_OUTCOME`).
    pub fn from_keys(keys: &[usize]) -> Self {
        assert!(
            keys.len() <= NUM_CHANCE_OUTCOMES,
            "too many chance outcomes: {}",
            keys.len()
        );
        let mut outcomes = ChanceOutcomes::default();
        for (slot, &key) in outcomes.keys.iter_mut().zip(keys) {
            assert!(key < NUM_CHANCE_OUTCOMES, "invalid chance outcome {key}");
            *slot = key as u8;
        }
        outcomes.len = keys.len() as u8;
        outcomes
    }

    /// Outcomes from a piece-type bitmask (bit `i` set = piece `i` possible), ascending.
    pub fn from_piece_mask(piece_mask: u8) -> Self {
        let mut outcomes = ChanceOutcomes::default();
        for piece_type in 0..NUM_PIECE_TYPES {
            if piece_mask & (1 << piece_type) != 0 {
                outcomes.keys[outcomes.len as usize] = piece_type as u8;
                outcomes.len += 1;
            }
        }
        outcomes
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.keys[..self.len as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Chance node for stochastic piece spawns
#[derive(Clone)]
pub struct ChanceNode {
    /// Game state (after piece placement, before new piece spawn)
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Expanded decision children indexed by chance outcome key
    children: [Option<Box<DecisionNode>>; NUM_CHANCE_OUTCOMES],
    /// Attack gained from the action that led to this node
    pub attack: u32,
    /// Overhang fields in the board state at this node
    pub overhang_fields: u32,
    /// Chance outcome keys that can be realized from this node
    pub outcomes: ChanceOutcomes,
    /// All backed-up total values that contributed to value_sum (only when track_value_history=true)
    pub value_history: Option<Vec<f32>>,
}

impl ChanceNode {
    pub fn new(
        state: TetrisEnv,
        attack: u32,
        overhang_fields: u32,
        outcomes: ChanceOutcomes,
    ) -> Self {
        ChanceNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: Default::default(),
            attack,
            overhang_fields,
            outcomes,
            value_history: None,
        }
    }

    /// Randomly select a piece from the possible outcomes
    pub fn select_piece_random(&self, rng: &mut StdRng) -> usize {
        let outcomes = self.outcomes.as_slice();
        if outcomes.is_empty() {
            // New bag - any piece is equally likely
            rng.gen_range(0..NUM_PIECE_TYPES)
        } else {
            // Select from remaining pieces in current bag
            *outcomes.choose(rng).unwrap() as usize
        }
    }

    /// Expanded decision child for chance outcome key `outcome`.
    #[inline]
    pub fn child(&self, outcome: usize) -> Option<&DecisionNode> {
        self.children
            .get(outcome)
            .and_then(|child| child.as_deref())
    }

    /// Mutable expanded decision child for chance outcome key `outcome`.
    #[inline]
    pub fn child_mut(&mut self, outcome: usize) -> Option<&mut DecisionNode> {
        self.children
            .get_mut(outcome)
            .and_then(|child| child.as_deref_mut())
    }

    /// Store the expanded decision child for chance outcome key `outcome`.
    pub fn set_child(&mut self, outcome: usize, child: DecisionNode) -> &mut DecisionNode {
        let slot = &mut self.children[outcome];
        debug_assert!(slot.is_none(), "chance outcome {outcome} expanded twice");
        slot.insert(Box::new(child))
    }

    /// Detach and return the decision child for chance outcome key `outcome`, if expanded.
    pub fn take_child(&mut self, outcome: usize) -> Option<DecisionNode> {
        self.children.get_mut(outcome)?.take().map(|child| *child)
    }

    /// Expanded children as `(outcome, child)`, in ascending outcome order.
    pub fn children(&self) -> impl Iterator<Item = (usize, &DecisionNode)> + '_ {
        self.children
            .iter()
            .enumerate()
            .filter_map(|(outcome, child)| child.as_deref().map(|child| (outcome, child)))
    }

    pub fn has_children(&self) -> bool {
        self.children.iter().any(Option::is_some)
    }
}

impl ChanceNode {
    pub fn mean_value(&self) -> f32 {
        mean_value(self.visit_count, self.value_sum)
    }
}

/// Get valid action indices for a state
pub fn get_valid_action_indices(env: &TetrisEnv) -> Vec<usize> {
    env.get_cached_valid_action_indices()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcomes(keys: &[usize]) -> ChanceOutcomes {
        ChanceOutcomes::from_keys(keys)
    }

    #[test]
    fn test_decision_node_creation() {
        let env = TetrisEnv::new(10, 20);
        let node = DecisionNode::new(env);

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert!(!node.has_children());
        assert!(!node.valid_actions.is_empty());
        assert!(!node.is_terminal);
        assert_eq!(node.state.placement_count, 0);
    }

    #[test]
    fn test_decision_node_terminal_state() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        env.placement_count = 5;
        let node = DecisionNode::new(env);

        assert!(node.is_terminal);
        assert!(node.valid_actions.is_empty());
        assert_eq!(node.state.placement_count, 5);
    }

    #[test]
    fn test_decision_node_set_nn_output() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);

        // Create mock policy over all actions
        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.5);

        // Priors should be normalized to sum to 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Priors should sum to 1, got {}",
            sum
        );

        // Each valid action should have a prior
        assert_eq!(node.action_priors.len(), node.valid_actions.len());

        // NN value should be stored
        assert!((node.nn_value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decision_node_set_nn_output_for_valid_actions() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);

        let action_priors = vec![1.0; node.valid_actions.len()];
        node.set_nn_output_for_valid_actions(&action_priors, 0.5);

        let sum: f32 = node.action_priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Priors should sum to 1, got {}",
            sum
        );
        assert_eq!(node.action_priors.len(), node.valid_actions.len());
        assert!((node.nn_value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_nn_output_ignores_invalid_action_mass() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);

        let invalid_action = (0..NUM_ACTIONS)
            .find(|idx| !node.valid_actions.contains(idx))
            .expect("Expected at least one invalid action");
        let preferred_valid_action = node.valid_actions[0];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[invalid_action] = 1000.0;
        policy[preferred_valid_action] = 1.0;

        node.set_nn_output(&policy, 0.0);

        let preferred_prior_index = node
            .valid_actions
            .iter()
            .position(|&idx| idx == preferred_valid_action)
            .expect("Preferred valid action should be present in valid_actions");

        for (i, prior) in node.action_priors.iter().enumerate() {
            if i == preferred_prior_index {
                assert!(
                    (*prior - 1.0).abs() < 1e-6,
                    "Preferred valid action should have all probability mass"
                );
            } else {
                assert_eq!(
                    *prior, 0.0,
                    "Other valid actions should have zero mass after renormalization"
                );
            }
        }
    }

    #[test]
    fn test_decision_node_add_dirichlet_noise() {
        use rand::SeedableRng;
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);
        let mut rng = StdRng::seed_from_u64(42);

        // Set uniform priors
        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);

        let priors_before: Vec<f32> = node.action_priors.clone();

        // Add Dirichlet noise
        node.add_dirichlet_noise(0.3, 0.25, &mut rng);

        // Priors should still sum to approximately 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Priors should still sum to 1 after noise"
        );

        // At least some priors should have changed
        let changed = node
            .action_priors
            .iter()
            .zip(priors_before.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed, "Dirichlet noise should modify priors");
    }

    #[test]
    fn test_decision_node_select_action_unvisited() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);

        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);

        // With no bounds yet, all Q values normalize to 0.5 and prior+U wins.
        let position = node.select_action_position(1.0, (f32::INFINITY, f32::NEG_INFINITY), false);
        assert!(position < node.valid_actions.len());
    }

    #[test]
    fn test_decision_node_select_action_with_children() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());

        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 10;

        // Add a child with high value
        let action_idx = node.valid_actions[0];
        let mut child = ChanceNode::new(env.clone(), 0, 0, outcomes(&[]));
        child.visit_count = 5;
        child.value_sum = 10.0; // Mean value = 2.0
        node.set_child_for_action(action_idx, child);

        // Selection should consider the child's value
        let position = node.select_action_position(1.0, (0.0, 2.0), false);
        assert!(position < node.valid_actions.len());
    }

    #[test]
    fn test_decision_node_select_action_tie_prefers_lower_action_index() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env);
        let low_action = node.valid_actions[0];
        let high_action = node.valid_actions[1];
        // Put the higher action index first so position order and index order disagree.
        node.valid_actions = vec![high_action, low_action];
        node.set_nn_output_for_valid_actions(&[0.5, 0.5], 0.0);
        node.visit_count = 4;

        let position = node.select_action_position(1.0, (0.0, 1.0), false);
        assert_eq!(node.valid_actions[position], low_action);
    }

    #[test]
    fn test_decision_node_children_follow_valid_action_order() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());
        let first = node.valid_actions[0];
        let third = node.valid_actions[2];
        node.set_child_for_action(third, ChanceNode::new(env.clone(), 3, 0, outcomes(&[])));
        node.set_child_for_action(first, ChanceNode::new(env, 1, 0, outcomes(&[])));

        let children: Vec<(usize, u32)> = node
            .children()
            .map(|(action_idx, child)| (action_idx, child.attack))
            .collect();
        assert_eq!(children, vec![(first, 1), (third, 3)]);
        assert!(node.has_children());

        let taken = node
            .take_child_for_action(first)
            .expect("expanded child should be returned");
        assert_eq!(taken.attack, 1);
        assert!(node.take_child_for_action(first).is_none());
        assert!(node.child(2).is_some());
    }

    #[test]
    fn test_chance_node_creation() {
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 5, 0, outcomes(&[0, 1, 2]));

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert_eq!(node.attack, 5);
        assert_eq!(node.overhang_fields, 0);
        assert!(!node.has_children());
        assert_eq!(node.outcomes.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_chance_outcomes_from_piece_mask_are_ascending() {
        let from_mask = ChanceOutcomes::from_piece_mask(0b101_0010);
        assert_eq!(from_mask.as_slice(), &[1, 4, 6]);
        assert_eq!(from_mask, outcomes(&[1, 4, 6]));
        assert!(ChanceOutcomes::from_piece_mask(0).is_empty());
        assert_eq!(
            outcomes(&[NO_CHANCE_OUTCOME]).as_slice(),
            &[NO_CHANCE_OUTCOME as u8]
        );
    }

    #[test]
    fn test_chance_node_empty_bag() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, outcomes(&[]));

        // Empty bag - any piece should be selectable
        let piece = node.select_piece_random(&mut rng);
        assert!(piece < NUM_PIECE_TYPES);
    }

    #[test]
    fn test_chance_node_select_piece_random() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, outcomes(&[1, 3, 5])); // O, S, J remaining

        // Select many pieces and verify they're from the bag
        for _ in 0..20 {
            let piece = node.select_piece_random(&mut rng);
            assert!(
                piece == 1 || piece == 3 || piece == 5,
                "Piece {} should be from bag [1, 3, 5]",
                piece
            );
        }
    }

    #[test]
    fn test_chance_node_select_piece_random_matches_slice_choose() {
        use rand::SeedableRng;
        // Selection must consume the RNG exactly like `choose` on the outcome list.
        let env = TetrisEnv::new(10, 20);
        let keys = [0usize, 2, 3, 6];
        let node = ChanceNode::new(env, 0, 0, outcomes(&keys));
        let mut rng_node = StdRng::seed_from_u64(9);
        let mut rng_ref = StdRng::seed_from_u64(9);
        for _ in 0..50 {
            let expected = *keys.choose(&mut rng_ref).unwrap();
            assert_eq!(node.select_piece_random(&mut rng_node), expected);
        }
    }

    #[test]
    fn test_chance_node_children_by_outcome() {
        let env = TetrisEnv::new(10, 20);
        let mut node = ChanceNode::new(env.clone(), 0, 0, outcomes(&[2, 5]));
        let mut later = DecisionNode::new(env.clone());
        later.visit_count = 5;
        let mut earlier = DecisionNode::new(env);
        earlier.visit_count = 2;
        node.set_child(5, later);
        node.set_child(2, earlier);

        let visits: Vec<(usize, u32)> = node
            .children()
            .map(|(outcome, child)| (outcome, child.visit_count))
            .collect();
        assert_eq!(visits, vec![(2, 2), (5, 5)]);
        assert_eq!(node.take_child(5).map(|child| child.visit_count), Some(5));
        assert!(node.take_child(5).is_none());
        assert!(node.take_child(NO_CHANCE_OUTCOME + 1).is_none());
    }

    #[test]
    fn test_node_mean_value_unvisited() {
        let env = TetrisEnv::new(10, 20);
        let decision = DecisionNode::new(env.clone());
        let chance = ChanceNode::new(env, 0, 0, outcomes(&[]));

        assert_eq!(decision.mean_value(), 0.0);
        assert_eq!(chance.mean_value(), 0.0);
    }

    #[test]
    fn test_node_mean_value_visited() {
        let env = TetrisEnv::new(10, 20);
        let mut decision = DecisionNode::new(env.clone());
        decision.visit_count = 10;
        decision.value_sum = 25.0;

        assert!((decision.mean_value() - 2.5).abs() < 0.01);

        let mut chance = ChanceNode::new(env, 0, 0, outcomes(&[]));
        chance.visit_count = 4;
        chance.value_sum = 10.0;

        assert!((chance.mean_value() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_get_valid_action_indices() {
        let env = TetrisEnv::new(10, 20);
        let indices = get_valid_action_indices(&env);

        // Should have at least some valid actions
        assert!(!indices.is_empty());

        // All indices should be within action space bounds
        for idx in &indices {
            assert!(*idx < NUM_ACTIONS, "Action index {} out of bounds", idx);
        }

        // No duplicate indices
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            indices.len(),
            "Should have no duplicate action indices"
        );
    }

    #[test]
    fn test_get_valid_action_indices_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        env.current_piece = None; // Clear the current piece as would happen in game over

        let indices = get_valid_action_indices(&env);

        // Game over state with no current piece should have no valid actions
        assert!(indices.is_empty());
    }

    #[test]
    fn test_decision_node_puct_exploration() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());

        // Create non-uniform priors
        let mut policy = vec![0.001; NUM_ACTIONS];
        if let Some(&first_action) = node.valid_actions.first() {
            policy[first_action] = 0.9; // High prior for first action
        }
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 1;

        // With high c_puct, should explore high-prior actions
        let position = node.select_action_position(10.0, (f32::INFINITY, f32::NEG_INFINITY), false);
        assert!(position < node.valid_actions.len());
    }

    #[test]
    fn test_normalize_q_value_tiny_range_is_neutral() {
        let q_norm = normalize_q_value(3.0, 2.0, 2.0 + (Q_NORMALIZATION_EPSILON * 0.5));
        assert!((q_norm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_decision_node_select_action_uses_min_max_q_normalization() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());

        let high_q_action = node.valid_actions[0];
        let high_prior_action = node.valid_actions[1];
        node.valid_actions = vec![high_q_action, high_prior_action];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[high_q_action] = 0.01;
        policy[high_prior_action] = 0.99;
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 100;

        let mut high_q_child = ChanceNode::new(env.clone(), 0, 0, outcomes(&[]));
        high_q_child.visit_count = 10;
        high_q_child.value_sum = 2000.0; // mean Q = 200
        node.set_child_for_action(high_q_action, high_q_child);

        let mut high_prior_child = ChanceNode::new(env, 0, 0, outcomes(&[]));
        high_prior_child.visit_count = 1;
        high_prior_child.value_sum = 100.0; // mean Q = 100
        node.set_child_for_action(high_prior_action, high_prior_child);

        // Raw-Q PUCT would choose high_q_action; global min-max normalized-Q PUCT
        // should let the strong prior/exploration term win here.
        let position = node.select_action_position(1.0, (100.0, 200.0), false);
        assert_eq!(node.valid_actions[position], high_prior_action);
    }

    #[test]
    fn test_decision_node_select_action_uses_min_max_q_normalization_with_large_q_gap() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());

        let high_q_action = node.valid_actions[0];
        let high_prior_action = node.valid_actions[1];
        node.valid_actions = vec![high_q_action, high_prior_action];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[high_q_action] = 0.01;
        policy[high_prior_action] = 0.99;
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 100;

        let mut high_q_child = ChanceNode::new(env.clone(), 0, 0, outcomes(&[]));
        high_q_child.visit_count = 10;
        high_q_child.value_sum = 2000.0; // mean Q = 200
        node.set_child_for_action(high_q_action, high_q_child);

        let mut high_prior_child = ChanceNode::new(env, 0, 0, outcomes(&[]));
        high_prior_child.visit_count = 1;
        high_prior_child.value_sum = 100.0; // mean Q = 100
        node.set_child_for_action(high_prior_action, high_prior_child);

        // Global min-max normalization also compresses this large Q gap into [0, 1],
        // so the strong prior/exploration term should dominate this comparison.
        let position = node.select_action_position(1.0, (100.0, 200.0), false);
        assert_eq!(node.valid_actions[position], high_prior_action);
    }

    #[test]
    fn test_decision_node_select_action_can_use_parent_value_for_unvisited_children() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone());

        let visited_action = node.valid_actions[0];
        let unvisited_action = node.valid_actions[1];
        node.valid_actions = vec![visited_action, unvisited_action];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[visited_action] = 0.99;
        policy[unvisited_action] = 0.01;
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 100;
        node.set_initial_total_value_estimate(16.0);

        let mut visited_child = ChanceNode::new(env, 0, 0, outcomes(&[]));
        visited_child.visit_count = 50;
        visited_child.value_sum = 0.0;
        node.set_child_for_action(visited_action, visited_child);

        let without_parent_value = node.select_action_position(1.0, (0.0, 16.0), false);
        let with_parent_value = node.select_action_position(1.0, (0.0, 16.0), true);

        assert_eq!(node.valid_actions[without_parent_value], visited_action);
        assert_eq!(node.valid_actions[with_parent_value], unvisited_action);
    }

    #[test]
    fn test_chance_node_random_selection_variety() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, outcomes(&[])); // Empty bag = all 7 pieces possible

        // Select many pieces and verify we get variety
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(node.select_piece_random(&mut rng));
        }

        // With 100 selections from 7 pieces, we should see most of them
        assert!(
            seen.len() >= 5,
            "Random selection should produce variety, got {} unique pieces",
            seen.len()
        );
    }
}
