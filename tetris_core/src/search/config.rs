//! MCTS Configuration

use pyo3::prelude::*;

/// Configuration for MCTS
#[pyclass]
#[derive(Clone, Debug)]
pub struct MCTSConfig {
    /// Number of simulations per move
    #[pyo3(get, set)]
    pub num_simulations: u32,
    /// PUCT exploration constant
    #[pyo3(get, set)]
    pub c_puct: f32,
    /// Temperature for action selection (0 = argmax, higher = more exploration)
    #[pyo3(get, set)]
    pub temperature: f32,
    /// Dirichlet noise alpha (for root exploration)
    #[pyo3(get, set)]
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight (epsilon)
    #[pyo3(get, set)]
    pub dirichlet_epsilon: f32,
    /// Epsilon for sampling root action from visit-policy instead of argmax
    #[pyo3(get, set)]
    pub visit_sampling_epsilon: f32,
    /// Optional RNG seed for deterministic behavior (None = non-deterministic)
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    /// Maximum placements per episode (hold actions do not count toward this limit).
    /// Used for placement-count normalization in NN features
    #[pyo3(get, set)]
    pub max_placements: u32,
    /// Whether to store per-visit backed-up values for visualization/debugging
    #[pyo3(get, set)]
    pub track_value_history: bool,
    /// Penalty applied when a simulation reaches game over (backs up as negative value)
    #[pyo3(get, set)]
    pub death_penalty: f32,
    /// Weight for normalized overhang penalty subtracted during MCTS search scoring
    #[pyo3(get, set)]
    pub overhang_penalty_weight: f32,
    /// Scale factor applied to neural value-head output during search (0 = ignore value head)
    #[pyo3(get, set)]
    pub nn_value_weight: f32,
    /// If true, unvisited action children start with the parent node's initial backed-up
    /// total-value estimate instead of zero Q (first-play urgency).
    #[pyo3(get, set)]
    pub use_parent_value_for_unvisited_q: bool,
    /// Reuse the MCTS subtree from the previous move instead of building a fresh tree.
    /// After selecting an action, the subtree corresponding to that action + actual piece
    /// spawn is extracted and used as the starting point for the next search.
    #[pyo3(get, set)]
    pub reuse_tree: bool,
    /// Count visits carried into a reused root toward `num_simulations`.
    /// When true, a reused root with `v` visits runs
    /// `max(min_new_simulations, num_simulations - v)` new simulations, so every search ends
    /// with roughly `num_simulations` root visits. When false, each search adds
    /// `num_simulations` new simulations on top of any carried visits. Fresh roots always run
    /// `num_simulations`.
    #[pyo3(get, set)]
    pub count_reused_visits: bool,
    /// Lower bound on new simulations per search when `count_reused_visits` is enabled
    /// (clamped to `num_simulations`), so root noise still influences every search.
    #[pyo3(get, set)]
    pub min_new_simulations: u32,
    /// Mean of Gaussian noise added independently to each policy logit before masking/softmax.
    #[pyo3(get, set)]
    pub policy_noise_mean: f32,
    /// Standard deviation of Gaussian noise added independently to each policy logit
    /// before masking/softmax.
    #[pyo3(get, set)]
    pub policy_noise_std: f32,
    /// Mean of Gaussian noise added to the raw value-head prediction.
    #[pyo3(get, set)]
    pub value_noise_mean: f32,
    /// Standard deviation of Gaussian noise added to the raw value-head prediction.
    #[pyo3(get, set)]
    pub value_noise_std: f32,
    /// Optional base seed for deterministic state-keyed prediction noise.
    /// Keep this separate from `seed` so search stochasticity and prediction noise can vary
    /// independently in evaluation sweeps.
    #[pyo3(get, set)]
    pub prediction_noise_seed: Option<u64>,
}

#[pymethods]
impl MCTSConfig {
    #[new]
    pub fn new() -> Self {
        MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
            visit_sampling_epsilon: 0.0,
            seed: None,
            max_placements: 100,
            track_value_history: false,
            death_penalty: 0.0,
            overhang_penalty_weight: 0.0,
            nn_value_weight: 1.0,
            use_parent_value_for_unvisited_q: false,
            reuse_tree: true,
            count_reused_visits: false,
            min_new_simulations: 0,
            policy_noise_mean: 0.0,
            policy_noise_std: 0.0,
            value_noise_mean: 0.0,
            value_noise_std: 0.0,
            prediction_noise_seed: None,
        }
    }
}

impl MCTSConfig {
    /// Number of new simulations to run from a root that already has `root_visits` visits.
    pub(crate) fn new_simulations_for_root(&self, root_visits: u32) -> u32 {
        if !self.count_reused_visits {
            return self.num_simulations;
        }
        let floor = self.min_new_simulations.min(self.num_simulations);
        self.num_simulations.saturating_sub(root_visits).max(floor)
    }
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = MCTSConfig::default();
        assert_eq!(config.num_simulations, 100);
        assert_eq!(config.c_puct, 1.5);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.dirichlet_alpha, 0.15);
        assert_eq!(config.dirichlet_epsilon, 0.25);
        assert_eq!(config.visit_sampling_epsilon, 0.0);
        assert_eq!(config.seed, None);
        assert_eq!(config.max_placements, 100);
        assert!(!config.track_value_history);
        assert_eq!(config.death_penalty, 0.0);
        assert_eq!(config.overhang_penalty_weight, 0.0);
        assert_eq!(config.nn_value_weight, 1.0);
        assert!(!config.use_parent_value_for_unvisited_q);
        assert!(config.reuse_tree);
        assert!(!config.count_reused_visits);
        assert_eq!(config.min_new_simulations, 0);
        assert_eq!(config.policy_noise_mean, 0.0);
        assert_eq!(config.policy_noise_std, 0.0);
        assert_eq!(config.value_noise_mean, 0.0);
        assert_eq!(config.value_noise_std, 0.0);
        assert_eq!(config.prediction_noise_seed, None);
    }

    #[test]
    fn test_new_simulations_for_root_ignores_carried_visits_by_default() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 2000;
        assert_eq!(config.new_simulations_for_root(0), 2000);
        assert_eq!(config.new_simulations_for_root(1500), 2000);
    }

    #[test]
    fn test_new_simulations_for_root_counts_carried_visits_with_floor() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 2000;
        config.count_reused_visits = true;
        config.min_new_simulations = 250;
        assert_eq!(config.new_simulations_for_root(0), 2000);
        assert_eq!(config.new_simulations_for_root(600), 1400);
        assert_eq!(config.new_simulations_for_root(1900), 250);
        assert_eq!(config.new_simulations_for_root(5000), 250);

        config.min_new_simulations = 10_000;
        assert_eq!(config.new_simulations_for_root(5000), 2000);
    }
}
