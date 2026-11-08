//! Neural Network Inference
//!
//! Loads split models and runs them natively: the conv backbone (`X.conv.onnx`) and the
//! cached-board-path MLP (`X.fc.bin`) via `board_embedding`, the heads (`X.heads.onnx`) via
//! `heads`. The board-only part of the network, up to and including the board columns of
//! `fusion_fc`, is cached per board.

use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tract_onnx::prelude::*;

#[cfg(test)]
use crate::game::constants::BOARD_HEIGHT;
use crate::game::constants::{
    AUX_FEATURES, BOARD_CACHE_MAX_ENTRIES, BOARD_STATS_FEATURES, BOARD_WIDTH,
    COMBO_NORMALIZATION_MAX, NUM_PIECE_TYPES, PIECE_AUX_FEATURES, QUEUE_SIZE,
    ROW_FILL_FEATURE_ROWS,
};
use crate::game::env::TetrisEnv;
use crate::search::{
    compute_bumpiness, count_overhang_fields_and_holes, normalize_bumpiness,
    normalize_column_heights, normalize_holes, normalize_max_column_height,
    normalize_overhang_fields, normalize_row_fill_counts, normalize_total_blocks,
};

mod board_cache;
mod board_embedding;
mod conv;
mod heads;
mod heads_kernels;
mod heads_onnx;
mod kernels;
mod onnx_proto;
#[cfg(test)]
mod tests;
mod weight_registry;

use board_cache::{BoardCache, BoardKey};
use board_embedding::{BoardEmbedder, BoardEmbeddingScratch};
use heads::{HeadsModel, HeadsScratch};
use weight_registry::{BundleKey, WeakRegistry};

/// Neural network model wrapper with board embedding cache
pub struct TetrisNN {
    weights: Arc<ModelWeights>,
    board_scratch: RefCell<BoardEmbeddingScratch>,
    /// Packed board -> board-only fusion input (`HeadsModel::board_fusion_input`).
    board_cache: RefCell<BoardCache>,
    heads_scratch: RefCell<HeadsScratch>,
    cache_capacity: usize,
    cache_hits: Cell<u64>,
    cache_misses: Cell<u64>,
    cache_enabled: Cell<bool>,
}

/// Immutable weights of one model bundle, shared by every `TetrisNN` loaded from it.
struct ModelWeights {
    board_embedder: BoardEmbedder,
    heads: HeadsModel,
}

/// One copy of each bundle's weights per process (see `weight_registry`).
static MODEL_WEIGHTS: std::sync::OnceLock<WeakRegistry<BundleKey, ModelWeights>> =
    std::sync::OnceLock::new();

#[derive(Clone)]
struct LinearLayer {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

#[derive(Clone)]
struct LayerNorm1D {
    weight: Array1<f32>,
    bias: Array1<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PredictionNoiseSettings {
    pub policy_mean: f32,
    pub policy_std: f32,
    pub value_mean: f32,
    pub value_std: f32,
    pub seed: Option<u64>,
}

impl PredictionNoiseSettings {
    pub fn is_enabled(&self) -> bool {
        self.policy_mean != 0.0
            || self.policy_std != 0.0
            || self.value_mean != 0.0
            || self.value_std != 0.0
    }
}

#[derive(Clone)]
struct CachedBoardPath {
    board_stats_fc: LinearLayer,
    board_stats_ln: LayerNorm1D,
    board_proj_fc1: LinearLayer,
    board_proj_ln1: LayerNorm1D,
    board_proj_fc2: LinearLayer,
}

impl TetrisNN {
    /// Load split models from file.
    /// Given a base path like "latest.onnx", loads:
    /// - "latest.conv.onnx" (conv backbone weights, run natively)
    /// - "latest.heads.onnx" (post-cache fusion + heads, run natively)
    /// - "latest.fc.bin" (cached-board-path weights)
    pub fn load<P: AsRef<Path>>(path: P) -> TractResult<Self> {
        let base = path.as_ref().with_extension("");
        let conv_path = base.with_extension("conv.onnx");
        let heads_path = base.with_extension("heads.onnx");
        let fc_path = base.with_extension("fc.bin");

        let key = BundleKey::for_files(&[&conv_path, &heads_path, &fc_path])?;
        let weights = MODEL_WEIGHTS
            .get_or_init(WeakRegistry::new)
            .get_or_try_insert_with(key, || -> TractResult<ModelWeights> {
                let board_embedder = BoardEmbedder::load(&conv_path, &load_fc_binary(&fc_path)?)?;
                let heads = HeadsModel::load(&heads_path, board_embedder.output_size())?;
                Ok(ModelWeights {
                    board_embedder,
                    heads,
                })
            })?;
        Ok(Self::from_weights(weights, BOARD_CACHE_MAX_ENTRIES, true))
    }

    /// Per-instance scratch buffers, board cache and counters around shared weights.
    fn from_weights(
        weights: Arc<ModelWeights>,
        cache_capacity: usize,
        cache_enabled: bool,
    ) -> Self {
        TetrisNN {
            board_scratch: RefCell::new(weights.board_embedder.new_scratch()),
            board_cache: RefCell::new(BoardCache::new(
                weights.heads.board_fusion_width(),
                cache_capacity,
            )),
            heads_scratch: RefCell::new(weights.heads.new_scratch()),
            weights,
            cache_capacity,
            cache_hits: Cell::new(0),
            cache_misses: Cell::new(0),
            cache_enabled: Cell::new(cache_enabled),
        }
    }

    fn compute_board_embedding(
        &self,
        board_f32: &[f32],
        board_stats: &[f32],
    ) -> TractResult<Array1<f32>> {
        let mut scratch = self.board_scratch.borrow_mut();
        let board_h = self
            .weights
            .board_embedder
            .embed(board_f32, board_stats, &mut scratch)?;
        Ok(Array1::from(board_h.to_vec()))
    }

    fn clear_board_embedding_cache(&self) {
        self.board_cache.borrow_mut().clear();
    }

    /// Board-only fusion input for one board: board embedding, then the board columns of
    /// `fusion_fc` plus its bias.
    fn compute_board_fusion_input(
        &self,
        board_f32: &[f32],
        board_stats: &[f32],
        scratch: &mut HeadsScratch,
        out: &mut [f32],
    ) -> TractResult<()> {
        let board_embed = self.compute_board_embedding(board_f32, board_stats)?;
        self.weights.heads.board_fusion_input(
            contiguous_board_embedding(&board_embed)?,
            scratch,
            out,
        );
        Ok(())
    }

    /// Run the heads up to the policy hidden layer for `env`, taking the board-only fusion
    /// input from the cache. Returns the value; the policy hidden activations stay in
    /// `scratch` for the policy logits.
    fn run_heads_trunk(
        &self,
        env: &TetrisEnv,
        piece_aux: &[f32],
        scratch: &mut HeadsScratch,
    ) -> TractResult<f32> {
        if !self.cache_enabled.get() {
            let mut board_input = vec![0.0; self.weights.heads.board_fusion_width()];
            self.compute_board_fusion_input(
                &encode_board_features(env),
                &encode_board_stats(env),
                scratch,
                &mut board_input,
            )?;
            return Ok(self
                .weights
                .heads
                .forward_trunk(&board_input, piece_aux, scratch));
        }

        let board_key = BoardKey(pack_board(env));
        let mut cache = self.board_cache.borrow_mut();
        let slot = match cache.slot_of(&board_key) {
            Some(slot) => {
                self.cache_hits.set(self.cache_hits.get() + 1);
                slot
            }
            None => {
                self.cache_misses.set(self.cache_misses.get() + 1);
                let board_embed = self.compute_board_embedding(
                    &encode_board_features(env),
                    &encode_board_stats(env),
                )?;
                let board_embed = contiguous_board_embedding(&board_embed)?;
                let (slot, row) = cache.insert(board_key);
                self.weights
                    .heads
                    .board_fusion_input(board_embed, scratch, row);
                slot
            }
        };
        Ok(self
            .weights
            .heads
            .forward_trunk(cache.row(slot), piece_aux, scratch))
    }

    /// Encode the piece/game aux features, run the heads trunk and return the value plus the
    /// prediction-noise state hash.
    fn predict_value_and_state_hash(
        &self,
        env: &TetrisEnv,
        max_placements: usize,
        scratch: &mut HeadsScratch,
    ) -> TractResult<(f32, u64)> {
        // Encode only piece/game aux features (61-dim) for the heads model.
        let piece_aux_vec = encode_piece_aux_features(env, max_placements)?;
        let state_hash = hash_prediction_inputs(&pack_board(env), &piece_aux_vec);
        let value = self.run_heads_trunk(env, &piece_aux_vec, scratch)?;
        Ok((value, state_hash))
    }

    fn validate_policy_output_len(
        &self,
        policy_len: usize,
        expected_len: usize,
    ) -> TractResult<()> {
        if policy_len == expected_len {
            return Ok(());
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "NN policy output size mismatch: model={}, expected={} (action space changed; re-export ONNX)",
                policy_len, expected_len
            ),
        )
        .into())
    }

    fn validate_valid_actions(
        &self,
        policy_len: usize,
        valid_actions: &[usize],
    ) -> TractResult<()> {
        if let Some(&invalid_action) = valid_actions
            .iter()
            .find(|&&action_idx| action_idx >= policy_len)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Valid action index {} out of range for policy logits len {}",
                    invalid_action, policy_len
                ),
            )
            .into());
        }

        Ok(())
    }

    fn split_aux_tensor<'a>(&self, aux_tensor: &'a [f32]) -> TractResult<(&'a [f32], &'a [f32])> {
        if aux_tensor.len() != AUX_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "aux tensor length mismatch: got {}, expected {}",
                    aux_tensor.len(),
                    AUX_FEATURES
                ),
            )
            .into());
        }

        Ok(aux_tensor.split_at(PIECE_AUX_FEATURES))
    }

    /// Run inference with action mask applied, using board embedding cache.
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        action_mask: &[bool],
        max_placements: usize,
        prediction_noise: PredictionNoiseSettings,
    ) -> TractResult<(Vec<f32>, f32)> {
        self.validate_policy_output_len(self.weights.heads.num_actions(), action_mask.len())?;
        let mut scratch = self.heads_scratch.borrow_mut();
        let (mut value, state_hash) =
            self.predict_value_and_state_hash(env, max_placements, &mut scratch)?;
        let mut policy_logits = self.weights.heads.all_policy_logits(&scratch);
        apply_prediction_noise(&mut policy_logits, &mut value, prediction_noise, state_hash)?;

        let policy = masked_softmax(&policy_logits, action_mask);
        Ok((policy, value))
    }

    /// Run inference with precomputed valid action indices.
    ///
    /// Without prediction noise only the policy logits of `valid_actions` are computed.
    pub fn predict_with_valid_actions(
        &self,
        env: &TetrisEnv,
        valid_actions: &[usize],
        max_placements: usize,
        prediction_noise: PredictionNoiseSettings,
    ) -> TractResult<(Vec<f32>, f32)> {
        self.validate_valid_actions(self.weights.heads.num_actions(), valid_actions)?;
        let mut scratch = self.heads_scratch.borrow_mut();
        let (mut value, state_hash) =
            self.predict_value_and_state_hash(env, max_placements, &mut scratch)?;
        if prediction_noise.is_enabled() {
            // Noise is drawn for every logit in action order, so it needs the full vector.
            let mut policy_logits = self.weights.heads.all_policy_logits(&scratch);
            apply_prediction_noise(&mut policy_logits, &mut value, prediction_noise, state_hash)?;
            return Ok((
                softmax_over_valid_actions(&policy_logits, valid_actions),
                value,
            ));
        }

        let mut priors = vec![0.0; valid_actions.len()];
        self.weights
            .heads
            .policy_logits(&scratch, valid_actions, &mut priors);
        softmax_in_place(&mut priors);
        Ok((priors, value))
    }

    pub fn predict_masked_from_tensors(
        &self,
        board_tensor: &[f32],
        aux_tensor: &[f32],
        action_mask: &[bool],
        prediction_noise: PredictionNoiseSettings,
    ) -> TractResult<(Vec<f32>, f32)> {
        // No caching for raw tensor inputs (used only by debug functions).
        // aux_tensor is the full 80-dim vector; split into piece_aux (61) + board_stats (19).
        let (piece_aux, board_stats) = self.split_aux_tensor(aux_tensor)?;
        self.validate_policy_output_len(self.weights.heads.num_actions(), action_mask.len())?;
        let mut scratch = self.heads_scratch.borrow_mut();
        let mut board_input = vec![0.0; self.weights.heads.board_fusion_width()];
        self.compute_board_fusion_input(board_tensor, board_stats, &mut scratch, &mut board_input)?;
        let mut value = self
            .weights
            .heads
            .forward_trunk(&board_input, piece_aux, &mut scratch);
        let mut policy_logits = self.weights.heads.all_policy_logits(&scratch);
        let state_hash = hash_prediction_inputs_for_tensors(board_tensor, piece_aux);
        apply_prediction_noise(&mut policy_logits, &mut value, prediction_noise, state_hash)?;

        let policy = masked_softmax(&policy_logits, action_mask);
        Ok((policy, value))
    }
}

impl TetrisNN {
    pub fn set_board_cache_enabled(&self, enabled: bool) {
        self.cache_enabled.set(enabled);
        if !enabled {
            self.clear_board_embedding_cache();
        }
    }

    /// Read and reset cache hit/miss counters. Returns (hits, misses, cache_size).
    pub fn get_and_reset_cache_stats(&self) -> (u64, u64, usize) {
        let hits = self.cache_hits.replace(0);
        let misses = self.cache_misses.replace(0);
        let size = self.board_cache.borrow().len();
        (hits, misses, size)
    }
}

impl Clone for TetrisNN {
    fn clone(&self) -> Self {
        // Share the immutable weights; create fresh scratch, cache and counters.
        Self::from_weights(
            Arc::clone(&self.weights),
            self.cache_capacity,
            self.cache_enabled.get(),
        )
    }
}

fn contiguous_board_embedding(board_embed: &Array1<f32>) -> TractResult<&[f32]> {
    Ok(board_embed.as_slice_memory_order().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Board embedding array is not contiguous",
        )
    })?)
}

/// Pack 200 binary board cells into [u64; 4] for use as a hash key.
/// Zero-collision: each unique board maps to a unique key.
fn pack_board(env: &TetrisEnv) -> [u64; 4] {
    let cells = env.board_cells();
    let mut key = [0u64; 4];
    for (i, &cell) in cells.iter().enumerate() {
        if cell != 0 {
            key[i / 64] |= 1u64 << (i % 64);
        }
    }
    key
}

fn hash_prediction_inputs(board_key: &[u64; 4], piece_aux: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    board_key.hash(&mut hasher);
    for &value in piece_aux {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_prediction_inputs_for_tensors(board_tensor: &[f32], piece_aux: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for &value in board_tensor {
        value.to_bits().hash(&mut hasher);
    }
    for &value in piece_aux {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn validate_prediction_noise(mean: f32, std: f32, label: &str) -> TractResult<()> {
    if !mean.is_finite() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{label} noise mean must be finite, got {mean}"),
        )
        .into());
    }
    if !std.is_finite() || std < 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{label} noise std must be finite and >= 0, got {std}"),
        )
        .into());
    }
    Ok(())
}

fn deterministic_noise_rng(base_seed: u64, state_hash: u64, stream_id: u64) -> StdRng {
    let derived_seed =
        base_seed ^ state_hash.rotate_left(17) ^ stream_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    StdRng::seed_from_u64(derived_seed)
}

fn add_gaussian_noise_with_rng<R: rand::Rng + ?Sized>(
    values: &mut [f32],
    mean: f32,
    std: f32,
    rng: &mut R,
) {
    if values.is_empty() {
        return;
    }
    if std == 0.0 {
        if mean != 0.0 {
            for value in values {
                *value += mean;
            }
        }
        return;
    }

    for value in values {
        let sample: f64 = StandardNormal.sample(rng);
        *value += mean + (std * sample as f32);
    }
}

fn apply_prediction_noise(
    policy_logits: &mut [f32],
    value: &mut f32,
    prediction_noise: PredictionNoiseSettings,
    state_hash: u64,
) -> TractResult<()> {
    validate_prediction_noise(
        prediction_noise.policy_mean,
        prediction_noise.policy_std,
        "policy",
    )?;
    validate_prediction_noise(
        prediction_noise.value_mean,
        prediction_noise.value_std,
        "value",
    )?;
    if !prediction_noise.is_enabled() {
        return Ok(());
    }

    if let Some(base_seed) = prediction_noise.seed {
        let mut policy_rng = deterministic_noise_rng(base_seed, state_hash, 0);
        add_gaussian_noise_with_rng(
            policy_logits,
            prediction_noise.policy_mean,
            prediction_noise.policy_std,
            &mut policy_rng,
        );

        let mut value_rng = deterministic_noise_rng(base_seed, state_hash, 1);
        let mut noisy_value = [*value];
        add_gaussian_noise_with_rng(
            &mut noisy_value,
            prediction_noise.value_mean,
            prediction_noise.value_std,
            &mut value_rng,
        );
        *value = noisy_value[0];
        return Ok(());
    }

    let mut rng = thread_rng();
    add_gaussian_noise_with_rng(
        policy_logits,
        prediction_noise.policy_mean,
        prediction_noise.policy_std,
        &mut rng,
    );
    let mut noisy_value = [*value];
    add_gaussian_noise_with_rng(
        &mut noisy_value,
        prediction_noise.value_mean,
        prediction_noise.value_std,
        &mut rng,
    );
    *value = noisy_value[0];
    Ok(())
}

/// Load cached-board-path weights from binary file.
/// Format (`TCM2`):
/// - board_stats_fc: [rows u32][cols u32][weight f32][bias f32]
/// - board_stats_ln: [hidden u32][weight f32][bias f32]
/// - board_proj_fc1: [rows u32][cols u32][weight f32][bias f32]
/// - board_proj_ln1: [hidden u32][weight f32][bias f32]
/// - board_proj_fc2: [rows u32][cols u32][weight f32][bias f32]
fn load_fc_binary(path: &Path) -> TractResult<CachedBoardPath> {
    let mut file = File::open(path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Cached board-path binary not found at {}: {}",
                path.display(),
                e
            ),
        )
    })?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if magic != *b"TCM2" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Unsupported cached board-path binary format in {} (expected magic TCM2)",
                path.display()
            ),
        )
        .into());
    }

    let board_stats_fc = read_linear_layer(&mut file, "board_stats_fc")?;
    if board_stats_fc.weight.ncols() != BOARD_STATS_FEATURES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "board_stats_fc input width mismatch: got {}, expected {}",
                board_stats_fc.weight.ncols(),
                BOARD_STATS_FEATURES
            ),
        )
        .into());
    }

    let board_stats_ln = read_layer_norm_1d(&mut file, "board_stats_ln")?;
    if board_stats_ln.weight.len() != board_stats_fc.weight.nrows() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "board_stats_ln width {} must match board_stats_fc output {}",
                board_stats_ln.weight.len(),
                board_stats_fc.weight.nrows()
            ),
        )
        .into());
    }

    let board_proj_fc1 = read_linear_layer(&mut file, "board_proj_fc1")?;
    if board_proj_fc1.weight.ncols() <= board_stats_fc.weight.nrows() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "board_proj_fc1 input width {} must exceed board_stats hidden {}",
                board_proj_fc1.weight.ncols(),
                board_stats_fc.weight.nrows()
            ),
        )
        .into());
    }

    let board_proj_ln1 = read_layer_norm_1d(&mut file, "board_proj_ln1")?;
    if board_proj_ln1.weight.len() != board_proj_fc1.weight.nrows() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "board_proj_ln1 width {} must match board_proj_fc1 output {}",
                board_proj_ln1.weight.len(),
                board_proj_fc1.weight.nrows()
            ),
        )
        .into());
    }

    let board_proj_fc2 = read_linear_layer(&mut file, "board_proj_fc2")?;
    if board_proj_fc2.weight.ncols() != board_proj_fc1.weight.nrows() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "board_proj_fc2 input width {} must match board_proj_fc1 output {}",
                board_proj_fc2.weight.ncols(),
                board_proj_fc1.weight.nrows()
            ),
        )
        .into());
    }

    Ok(CachedBoardPath {
        board_stats_fc,
        board_stats_ln,
        board_proj_fc1,
        board_proj_ln1,
        board_proj_fc2,
    })
}

fn read_u32(file: &mut File) -> TractResult<usize> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}

fn read_f32_vec(file: &mut File, len: usize) -> TractResult<Vec<f32>> {
    let mut data = vec![0u8; len * 4];
    file.read_exact(&mut data)?;
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_linear_layer(file: &mut File, label: &str) -> TractResult<LinearLayer> {
    let rows = read_u32(file)?;
    let cols = read_u32(file)?;
    let weight =
        Array2::from_shape_vec((rows, cols), read_f32_vec(file, rows * cols)?).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{label} weight shape mismatch: {e}"),
            )
        })?;
    let bias = Array1::from_vec(read_f32_vec(file, rows)?);
    Ok(LinearLayer { weight, bias })
}

fn read_layer_norm_1d(file: &mut File, label: &str) -> TractResult<LayerNorm1D> {
    let hidden = read_u32(file)?;
    let weight = Array1::from_vec(read_f32_vec(file, hidden)?);
    let bias = Array1::from_vec(read_f32_vec(file, hidden)?);
    if bias.len() != weight.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{label} bias width {} must match weight width {}",
                bias.len(),
                weight.len()
            ),
        )
        .into());
    }
    Ok(LayerNorm1D { weight, bias })
}

/// Encode a TetrisEnv state into neural network input tensors
#[cfg(test)]
fn encode_state(env: &TetrisEnv, max_placements: usize) -> TractResult<(Tensor, Tensor)> {
    let (board_tensor, aux_tensor) = encode_state_features(env, max_placements)?;
    let board =
        tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_tensor)?
            .into_tensor();
    let aux = tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor)?.into_tensor();
    Ok((board, aux))
}

pub fn encode_state_features(
    env: &TetrisEnv,
    max_placements: usize,
) -> TractResult<(Vec<f32>, Vec<f32>)> {
    let aux = encode_aux_state_features(env, max_placements)?;
    let board_tensor = encode_board_features(env);
    Ok((board_tensor, aux))
}

fn encode_board_features(env: &TetrisEnv) -> Vec<f32> {
    // Board tensor: binary (1 = filled, 0 = empty) - 200 values (will be reshaped to 1x20x10)
    env.board_cells()
        .iter()
        .map(|&cell| if cell != 0 { 1.0 } else { 0.0 })
        .collect()
}

#[track_caller]
fn require_current_piece_type(env: &TetrisEnv) -> usize {
    env.get_current_piece()
        .map(|piece| piece.piece_type)
        .expect("TetrisEnv should always have a current piece while encoding features")
}

#[track_caller]
fn require_max_u8(values: &[u8], label: &str) -> u8 {
    values
        .iter()
        .copied()
        .max()
        .unwrap_or_else(|| panic!("{label} should not be empty"))
}

struct PieceAuxInputs {
    current_piece: usize,
    hold_piece: Option<usize>,
    hold_available: bool,
    next_queue: Vec<usize>,
    placement_count_feature: f32,
    combo_feature: f32,
    back_to_back: bool,
    next_hidden_piece_probs: Vec<f32>,
}

impl PieceAuxInputs {
    fn from_env(env: &TetrisEnv, max_placements: usize) -> TractResult<Self> {
        if max_placements == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "max_placements must be > 0",
            )
            .into());
        }

        Ok(Self {
            current_piece: require_current_piece_type(env),
            hold_piece: env.get_hold_piece().map(|p| p.piece_type),
            hold_available: !env.is_hold_used(),
            next_queue: env.get_queue(QUEUE_SIZE),
            placement_count_feature: env.placement_count as f32 / max_placements as f32,
            combo_feature: normalize_combo_for_feature(env.combo),
            back_to_back: env.back_to_back,
            next_hidden_piece_probs: next_hidden_piece_distribution(env),
        })
    }

    fn encode_vec(&self) -> Vec<f32> {
        let mut aux = vec![0.0; PIECE_AUX_FEATURES];
        encode_piece_aux_into(&mut aux, self);
        aux
    }
}

struct BoardStatInputs {
    column_heights: Vec<f32>,
    max_column_height: f32,
    row_fill_counts: Vec<f32>,
    total_blocks: f32,
    bumpiness: f32,
    holes: f32,
    overhang_fields: f32,
}

impl BoardStatInputs {
    fn from_env(env: &TetrisEnv) -> Self {
        let column_heights = normalize_column_heights(&env.column_heights[..env.width]);
        let raw_max_column_height =
            require_max_u8(&env.column_heights[..env.width], "column_heights");
        let row_fill_counts =
            normalize_row_fill_counts(&env.row_fill_counts[..env.height], env.width);
        let total_blocks = normalize_total_blocks(env.total_blocks);
        let bumpiness = normalize_bumpiness(compute_bumpiness(&env.column_heights[..env.width]));
        let (raw_overhang_fields, raw_holes) = count_overhang_fields_and_holes(env);

        Self {
            column_heights,
            max_column_height: normalize_max_column_height(raw_max_column_height),
            row_fill_counts,
            total_blocks,
            bumpiness,
            holes: normalize_holes(raw_holes),
            overhang_fields: normalize_overhang_fields(raw_overhang_fields),
        }
    }

    fn encode_vec(&self) -> Vec<f32> {
        let mut stats = vec![0.0; BOARD_STATS_FEATURES];
        encode_board_stats_into(&mut stats, self);
        stats
    }
}

fn encode_piece_aux_into(aux_out: &mut [f32], inputs: &PieceAuxInputs) {
    debug_assert_eq!(aux_out.len(), PIECE_AUX_FEATURES);
    aux_out.fill(0.0);

    let mut aux_idx = 0;

    // Current piece: one-hot (7)
    aux_out[aux_idx + inputs.current_piece] = 1.0;
    aux_idx += NUM_PIECE_TYPES;

    // Hold piece: one-hot (8) - 7 pieces + empty
    if let Some(piece_type) = inputs.hold_piece {
        aux_out[aux_idx + piece_type] = 1.0;
    } else {
        aux_out[aux_idx + NUM_PIECE_TYPES] = 1.0;
    }
    aux_idx += NUM_PIECE_TYPES + 1;

    // Hold available: binary (1)
    aux_out[aux_idx] = if inputs.hold_available { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next queue: one-hot per slot (5 x 7 = 35)
    for (slot, &piece_type) in inputs.next_queue.iter().take(QUEUE_SIZE).enumerate() {
        aux_out[aux_idx + slot * NUM_PIECE_TYPES + piece_type] = 1.0;
    }
    aux_idx += QUEUE_SIZE * NUM_PIECE_TYPES;

    // Placement count: pre-normalized (1)
    aux_out[aux_idx] = inputs.placement_count_feature;
    aux_idx += 1;

    // Combo: pre-normalized (1)
    aux_out[aux_idx] = inputs.combo_feature;
    aux_idx += 1;

    // Back-to-back flag: binary (1)
    aux_out[aux_idx] = if inputs.back_to_back { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next hidden piece distribution from 7-bag (7)
    aux_out[aux_idx..aux_idx + NUM_PIECE_TYPES].copy_from_slice(&inputs.next_hidden_piece_probs);
    aux_idx += NUM_PIECE_TYPES;

    debug_assert_eq!(aux_idx, PIECE_AUX_FEATURES);
}

fn encode_board_stats_into(aux_out: &mut [f32], board_stats: &BoardStatInputs) {
    debug_assert_eq!(aux_out.len(), BOARD_STATS_FEATURES);

    let mut aux_idx = 0;
    aux_out[aux_idx..aux_idx + BOARD_WIDTH].copy_from_slice(&board_stats.column_heights);
    aux_idx += BOARD_WIDTH;

    aux_out[aux_idx] = board_stats.max_column_height;
    aux_idx += 1;

    aux_out[aux_idx..aux_idx + ROW_FILL_FEATURE_ROWS].copy_from_slice(&board_stats.row_fill_counts);
    aux_idx += ROW_FILL_FEATURE_ROWS;

    aux_out[aux_idx] = board_stats.total_blocks;
    aux_idx += 1;

    aux_out[aux_idx] = board_stats.bumpiness;
    aux_idx += 1;

    aux_out[aux_idx] = board_stats.holes;
    aux_idx += 1;

    aux_out[aux_idx] = board_stats.overhang_fields;
    aux_idx += 1;

    debug_assert_eq!(aux_idx, BOARD_STATS_FEATURES);
}

fn validate_aux_feature_inputs(
    aux_out: &[f32],
    current_piece: usize,
    hold_piece: Option<usize>,
    next_queue: &[usize],
    next_hidden_piece_probs: &[f32],
    column_heights: &[f32],
    row_fill_counts: &[f32],
) -> TractResult<()> {
    if aux_out.len() != AUX_FEATURES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "aux_out length must be {} (got {})",
                AUX_FEATURES,
                aux_out.len()
            ),
        )
        .into());
    }
    if current_piece >= NUM_PIECE_TYPES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "current_piece must be in 0..{} (got {})",
                NUM_PIECE_TYPES, current_piece
            ),
        )
        .into());
    }
    if let Some(piece_type) = hold_piece {
        if piece_type >= NUM_PIECE_TYPES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "hold_piece must be in 0..{} when present (got {})",
                    NUM_PIECE_TYPES, piece_type
                ),
            )
            .into());
        }
    }
    if next_hidden_piece_probs.len() != NUM_PIECE_TYPES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "next_hidden_piece_probs length must be {} (got {})",
                NUM_PIECE_TYPES,
                next_hidden_piece_probs.len()
            ),
        )
        .into());
    }
    if column_heights.len() != BOARD_WIDTH {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "column_heights length must be {} (got {})",
                BOARD_WIDTH,
                column_heights.len()
            ),
        )
        .into());
    }
    if row_fill_counts.len() != ROW_FILL_FEATURE_ROWS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "row_fill_counts length must be {} (got {})",
                ROW_FILL_FEATURE_ROWS,
                row_fill_counts.len()
            ),
        )
        .into());
    }
    if let Some(&invalid_piece) = next_queue
        .iter()
        .take(QUEUE_SIZE)
        .find(|&&piece_type| piece_type >= NUM_PIECE_TYPES)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "next_queue contains invalid piece type {} (expected < {})",
                invalid_piece, NUM_PIECE_TYPES
            ),
        )
        .into());
    }

    Ok(())
}

/// Encode the 19 board-derived statistics for the cached board embedding path.
fn encode_board_stats(env: &TetrisEnv) -> Vec<f32> {
    BoardStatInputs::from_env(env).encode_vec()
}

/// Encode only the 61 piece/game auxiliary features for the uncached heads model.
fn encode_piece_aux_features(env: &TetrisEnv, max_placements: usize) -> TractResult<Vec<f32>> {
    Ok(PieceAuxInputs::from_env(env, max_placements)?.encode_vec())
}

pub fn encode_aux_state_features(env: &TetrisEnv, max_placements: usize) -> TractResult<Vec<f32>> {
    let piece_aux = PieceAuxInputs::from_env(env, max_placements)?;
    let board_stats = BoardStatInputs::from_env(env);
    let mut aux = vec![0.0; AUX_FEATURES];
    let (piece_out, board_out) = aux.split_at_mut(PIECE_AUX_FEATURES);
    encode_piece_aux_into(piece_out, &piece_aux);
    encode_board_stats_into(board_out, &board_stats);

    Ok(aux)
}

pub fn normalize_combo_for_feature(combo: u32) -> f32 {
    combo as f32 / COMBO_NORMALIZATION_MAX as f32
}

pub fn denormalize_combo_feature(combo_feature: f32) -> u32 {
    assert!(
        combo_feature.is_finite(),
        "combo_feature must be finite, got {combo_feature}"
    );
    assert!(
        combo_feature >= 0.0,
        "combo_feature must be >= 0.0, got {combo_feature}"
    );

    let scaled_combo = combo_feature * COMBO_NORMALIZATION_MAX as f32;
    let rounded_combo = scaled_combo.round();
    assert!(
        rounded_combo <= u32::MAX as f32,
        "combo_feature {combo_feature} overflows u32 after scaling"
    );
    rounded_combo as u32
}

pub fn encode_aux_features(
    aux_out: &mut [f32],
    current_piece: usize,
    hold_piece: Option<usize>,
    hold_available: bool,
    next_queue: &[usize],
    placement_count_feature: f32,
    combo_feature: f32,
    back_to_back: bool,
    next_hidden_piece_probs: &[f32],
    column_heights: &[f32],
    max_column_height: f32,
    row_fill_counts: &[f32],
    total_blocks: f32,
    bumpiness: f32,
    holes: f32,
    overhang_fields: f32,
) -> TractResult<()> {
    validate_aux_feature_inputs(
        aux_out,
        current_piece,
        hold_piece,
        next_queue,
        next_hidden_piece_probs,
        column_heights,
        row_fill_counts,
    )?;

    let piece_aux = PieceAuxInputs {
        current_piece,
        hold_piece,
        hold_available,
        next_queue: next_queue.to_vec(),
        placement_count_feature,
        combo_feature,
        back_to_back,
        next_hidden_piece_probs: next_hidden_piece_probs.to_vec(),
    };
    let board_stats = BoardStatInputs {
        column_heights: column_heights.to_vec(),
        max_column_height,
        row_fill_counts: row_fill_counts.to_vec(),
        total_blocks,
        bumpiness,
        holes,
        overhang_fields,
    };

    aux_out.fill(0.0);
    let (piece_out, board_out) = aux_out.split_at_mut(PIECE_AUX_FEATURES);
    encode_piece_aux_into(piece_out, &piece_aux);
    encode_board_stats_into(board_out, &board_stats);
    Ok(())
}

/// Probability distribution over the next hidden queue piece implied by the 7-bag state.
pub fn next_hidden_piece_distribution(env: &TetrisEnv) -> Vec<f32> {
    let mut visible_state = env.clone();
    visible_state.truncate_queue(QUEUE_SIZE);
    let possible_pieces = visible_state.get_possible_next_pieces();
    if possible_pieces.is_empty() {
        panic!("Possible hidden-piece set must never be empty");
    }

    let probability = 1.0 / possible_pieces.len() as f32;
    let mut distribution = vec![0.0; NUM_PIECE_TYPES];
    for piece in possible_pieces {
        if piece >= NUM_PIECE_TYPES {
            panic!("Invalid piece type {} in possible next pieces", piece);
        }
        distribution[piece] = probability;
    }
    distribution
}

/// Softmax with mask (invalid actions get 0 probability)
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&x, _)| x)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut result = vec![0.0; logits.len()];
    let mut sum = 0.0;

    for (i, (&logit, &valid)) in logits.iter().zip(mask.iter()).enumerate() {
        if valid {
            let exp_val = (logit - max_logit).exp();
            result[i] = exp_val;
            sum += exp_val;
        }
    }

    if sum > 0.0 {
        for x in &mut result {
            *x /= sum;
        }
    }

    result
}

/// Softmax over precomputed valid action indices.
///
/// Returns priors aligned with `valid_actions` order.
pub fn softmax_over_valid_actions(logits: &[f32], valid_actions: &[usize]) -> Vec<f32> {
    let mut result: Vec<f32> = valid_actions
        .iter()
        .map(|&action_idx| logits[action_idx])
        .collect();
    softmax_in_place(&mut result);
    result
}

/// In-place softmax over a compact logit vector; matches `masked_softmax` restricted to the
/// same entries in the same order.
fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let max_logit = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        let exp_val = (*value - max_logit).exp();
        *value = exp_val;
        sum += exp_val;
    }

    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

/// Get action mask from environment
pub fn get_action_mask(env: &TetrisEnv) -> Vec<bool> {
    env.action_mask()
}
