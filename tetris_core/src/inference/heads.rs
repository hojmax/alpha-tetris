//! Native implementation of the uncached heads model (`X.heads.onnx`).
//!
//! The weights are read straight from the exported ONNX initializers (PyTorch parameter
//! names, `Gemm` with `transB=1`), so every existing model bundle works without new
//! artifacts. Loading validates the graph structure and fails loudly if the exported graph is
//! not the architecture implemented here:
//!
//! ```text
//! aux_h  = SiLU(LN(W . aux + b))   repeated for every aux_mlp layer
//! pre    = W_board . board_h + W_aux . aux_h + b      (fusion_fc on [board_h, aux_h])
//! fused  = SiLU(LN(pre))
//! fused += W2 . SiLU(LN2(W1 . SiLU(LN1(fused)) + b1)) + b2   per residual fusion block
//! logits = W_p2 . SiLU(W_p1 . fused + b_p1) + b_p2
//! value  = W_v2 . SiLU(W_v1 . fused + b_v1) + b_v2
//! ```
//!
//! `W_board . board_h + b` depends only on the board, so callers compute it once per board
//! (`board_fusion_input`) and cache it; each leaf then runs `forward_trunk` and asks for the
//! policy logits of just the actions it needs.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::heads_kernels::{
    gemv, gemv_rows, layer_norm, round_up, silu, Isa, PaddedMatrix, K_ALIGN,
};
use super::heads_onnx::{parse_model_graph, OnnxGraph, OnnxNode, OnnxTensor};
use crate::game::action_space::NUM_ACTIONS;
use crate::game::constants::PIECE_AUX_FEATURES;

type LoadResult<T> = Result<T, std::io::Error>;

fn invalid_model<T>(path: &Path, message: impl std::fmt::Display) -> LoadResult<T> {
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Heads model {}: {message}", path.display()),
    ))
}

struct Dense {
    weight: PaddedMatrix,
    /// Zero-padded to `weight.padded_rows()`.
    bias: Vec<f32>,
}

impl Dense {
    fn new(weight: PaddedMatrix, bias: &[f32]) -> Self {
        assert_eq!(bias.len(), weight.rows(), "bias width mismatch");
        let mut padded_bias = vec![0.0; weight.padded_rows()];
        padded_bias[..bias.len()].copy_from_slice(bias);
        Self {
            weight,
            bias: padded_bias,
        }
    }

    fn out_width(&self) -> usize {
        self.weight.rows()
    }
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct ResidualBlock {
    ln1: Norm,
    fc1: Dense,
    ln2: Norm,
    fc2: Dense,
}

pub(super) struct HeadsModel {
    isa: Isa,
    eps: f32,
    board_hidden: usize,
    fusion_hidden: usize,
    /// Each aux_mlp Linear + LayerNorm, followed by SiLU.
    aux_layers: Vec<(Dense, Norm)>,
    /// fusion_fc columns that read `board_h`, with the fusion bias.
    fusion_board: Dense,
    /// fusion_fc columns that read `aux_h` (bias already applied by `fusion_board`).
    fusion_aux: PaddedMatrix,
    fusion_aux_zero_bias: Vec<f32>,
    fusion_ln: Norm,
    blocks: Vec<ResidualBlock>,
    policy_fc: Dense,
    policy_head: Dense,
    value_fc: Dense,
    value_head: Dense,
}

/// Preallocated activations for one heads evaluation. Buffers are zero beyond their logical
/// width so they can be fed straight into the padded matrix kernels.
pub(super) struct HeadsScratch {
    board_h: Vec<f32>,
    aux_ping: Vec<f32>,
    aux_pong: Vec<f32>,
    fused: Vec<f32>,
    block_a: Vec<f32>,
    block_b: Vec<f32>,
    policy_hidden: Vec<f32>,
    value_hidden: Vec<f32>,
    value_out: Vec<f32>,
}

fn buffer(width: usize) -> Vec<f32> {
    vec![0.0; round_up(width.max(1), K_ALIGN)]
}

impl HeadsModel {
    pub(super) fn load(path: &Path, board_hidden: usize) -> LoadResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Heads model not found at {}: {e}", path.display()),
            )
        })?;
        let graph = parse_model_graph(&bytes).or_else(|e| invalid_model(path, e))?;
        Self::from_graph(graph, path, board_hidden, Isa::detect())
    }

    fn from_graph(
        graph: OnnxGraph,
        path: &Path,
        board_hidden: usize,
        isa: Isa,
    ) -> LoadResult<Self> {
        let OnnxGraph {
            nodes,
            initializers,
            input_names,
            output_names,
        } = graph;
        if input_names != ["board_h", "piece_aux"] || output_names != ["policy_logits", "value"] {
            return invalid_model(
                path,
                format!(
                    "unexpected graph signature inputs={input_names:?} outputs={output_names:?}"
                ),
            );
        }
        let mut reader = WeightReader::new(path, initializers, &nodes)?;

        let mut aux_layers = Vec::new();
        let mut aux_width = PIECE_AUX_FEATURES;
        let mut module_index = 0;
        while reader.has(&format!("aux_mlp.{module_index}.weight")) {
            let linear = reader.linear(&format!("aux_mlp.{module_index}"), None, aux_width)?;
            let norm = reader.norm(&format!("aux_mlp.{}", module_index + 1), linear.out_width())?;
            aux_width = linear.out_width();
            aux_layers.push((linear, norm));
            // Each hidden aux layer is Linear, LayerNorm, SiLU; the last one omits the SiLU
            // module (forward applies F.silu instead).
            module_index += 3;
        }
        if aux_layers.is_empty() {
            return invalid_model(path, "missing aux_mlp.0 layer");
        }

        let (fusion_weight, fusion_dims) = reader.take("fusion_fc.weight", 2)?;
        let fusion_hidden = fusion_dims[0];
        if fusion_dims[1] != board_hidden + aux_width {
            return invalid_model(
                path,
                format!(
                    "fusion_fc expects {} inputs but the cached board path gives {board_hidden} and aux_mlp gives {aux_width} (re-export split models together)",
                    fusion_dims[1]
                ),
            );
        }
        let (fusion_bias, _) = reader.take_shape("fusion_fc.bias", &[fusion_hidden])?;
        reader.check_gemm("fusion_fc")?;
        let fusion_cols = fusion_dims[1];
        let fusion_board = Dense::new(
            PaddedMatrix::from_row_major_columns(
                &fusion_weight,
                fusion_hidden,
                fusion_cols,
                0..board_hidden,
            ),
            &fusion_bias,
        );
        let fusion_aux = PaddedMatrix::from_row_major_columns(
            &fusion_weight,
            fusion_hidden,
            fusion_cols,
            board_hidden..fusion_cols,
        );
        let fusion_aux_zero_bias = vec![0.0; fusion_aux.padded_rows()];
        let fusion_ln = reader.norm("fusion_ln", fusion_hidden)?;

        let mut blocks = Vec::new();
        while reader.has(&format!("fusion_blocks.{}.ln1.weight", blocks.len())) {
            let prefix = format!("fusion_blocks.{}", blocks.len());
            blocks.push(ResidualBlock {
                ln1: reader.norm(&format!("{prefix}.ln1"), fusion_hidden)?,
                fc1: reader.linear(&format!("{prefix}.fc1"), Some(fusion_hidden), fusion_hidden)?,
                ln2: reader.norm(&format!("{prefix}.ln2"), fusion_hidden)?,
                fc2: reader.linear(&format!("{prefix}.fc2"), Some(fusion_hidden), fusion_hidden)?,
            });
        }

        let policy_fc = reader.linear("policy_fc", None, fusion_hidden)?;
        let policy_head = reader.linear("policy_head", None, policy_fc.out_width())?;
        if policy_head.out_width() != NUM_ACTIONS {
            return invalid_model(
                path,
                format!(
                    "NN policy output size mismatch: model={}, expected={NUM_ACTIONS} (action space changed; re-export ONNX)",
                    policy_head.out_width()
                ),
            );
        }
        let value_fc = reader.linear("value_fc", None, fusion_hidden)?;
        let value_head = reader.linear("value_head", Some(1), value_fc.out_width())?;

        let silu_count = aux_layers.len() + 1 + 2 * blocks.len() + 2;
        reader.finish(&[
            ("Gemm", reader.linear_layers),
            ("LayerNormalization", reader.norm_layers),
            ("Sigmoid", silu_count),
            ("Mul", silu_count),
            ("Concat", 1),
            ("Add", blocks.len()),
            ("Identity", reader.aliases.len()),
        ])?;
        let eps = reader
            .eps
            .ok_or_else(|| std::io::Error::other("heads graph has no LayerNormalization"))?;

        Ok(Self {
            isa,
            eps,
            board_hidden,
            fusion_hidden,
            aux_layers,
            fusion_board,
            fusion_aux,
            fusion_aux_zero_bias,
            fusion_ln,
            blocks,
            policy_fc,
            policy_head,
            value_fc,
            value_head,
        })
    }

    pub(super) fn new_scratch(&self) -> HeadsScratch {
        let aux_width = self
            .aux_layers
            .iter()
            .map(|(dense, _)| dense.weight.padded_rows())
            .max()
            .unwrap_or(0)
            .max(PIECE_AUX_FEATURES);
        HeadsScratch {
            board_h: buffer(self.board_hidden),
            aux_ping: buffer(aux_width),
            aux_pong: buffer(aux_width),
            fused: buffer(self.fusion_hidden),
            block_a: buffer(self.fusion_hidden),
            block_b: buffer(self.fusion_hidden),
            policy_hidden: buffer(self.policy_fc.weight.padded_rows()),
            value_hidden: buffer(self.value_fc.weight.padded_rows()),
            value_out: buffer(self.value_head.weight.padded_rows()),
        }
    }

    /// Width of the per-board fusion input (`fusion_hidden`).
    pub(super) fn board_fusion_width(&self) -> usize {
        self.fusion_hidden
    }

    pub(super) fn num_actions(&self) -> usize {
        self.policy_head.out_width()
    }

    /// `W_board . board_h + fusion bias`: the board-only part of fusion_fc, cacheable per board.
    pub(super) fn board_fusion_input(
        &self,
        board_h: &[f32],
        scratch: &mut HeadsScratch,
        out: &mut [f32],
    ) {
        assert_eq!(
            board_h.len(),
            self.board_hidden,
            "board embedding width mismatch"
        );
        assert_eq!(
            out.len(),
            self.fusion_hidden,
            "board fusion input width mismatch"
        );
        scratch.board_h[..self.board_hidden].copy_from_slice(board_h);
        gemv(
            self.isa,
            &self.fusion_board.weight,
            &self.fusion_board.bias,
            &scratch.board_h,
            &mut scratch.block_a,
        );
        out.copy_from_slice(&scratch.block_a[..self.fusion_hidden]);
    }

    /// Everything after the cached board input. Returns the value; the policy hidden
    /// activations stay in `scratch` for `policy_logits` / `all_policy_logits`.
    pub(super) fn forward_trunk(
        &self,
        board_fusion_input: &[f32],
        piece_aux: &[f32],
        scratch: &mut HeadsScratch,
    ) -> f32 {
        assert_eq!(
            piece_aux.len(),
            PIECE_AUX_FEATURES,
            "piece aux width mismatch"
        );
        assert_eq!(
            board_fusion_input.len(),
            self.fusion_hidden,
            "board fusion input width mismatch"
        );
        let isa = self.isa;
        let s = scratch;

        s.aux_ping[..PIECE_AUX_FEATURES].copy_from_slice(piece_aux);
        s.aux_ping[PIECE_AUX_FEATURES..].fill(0.0);
        for (dense, norm) in &self.aux_layers {
            gemv(
                isa,
                &dense.weight,
                &dense.bias,
                &s.aux_ping,
                &mut s.aux_pong,
            );
            let width = dense.out_width();
            self.norm_silu(&mut s.aux_pong[..width], norm);
            // Inputs beyond this layer's width must read as zero in the next product.
            s.aux_pong[dense.weight.padded_rows()..].fill(0.0);
            std::mem::swap(&mut s.aux_ping, &mut s.aux_pong);
        }

        let width = self.fusion_hidden;
        gemv(
            isa,
            &self.fusion_aux,
            &self.fusion_aux_zero_bias,
            &s.aux_ping,
            &mut s.fused,
        );
        for (pre, &board) in s.fused[..width].iter_mut().zip(board_fusion_input) {
            *pre += board;
        }
        self.norm_silu(&mut s.fused[..width], &self.fusion_ln);

        for block in &self.blocks {
            s.block_a[..width].copy_from_slice(&s.fused[..width]);
            self.norm_silu(&mut s.block_a[..width], &block.ln1);
            gemv(
                isa,
                &block.fc1.weight,
                &block.fc1.bias,
                &s.block_a,
                &mut s.block_b,
            );
            self.norm_silu(&mut s.block_b[..width], &block.ln2);
            gemv(
                isa,
                &block.fc2.weight,
                &block.fc2.bias,
                &s.block_b,
                &mut s.block_a,
            );
            for (x, &h) in s.fused[..width].iter_mut().zip(&s.block_a[..width]) {
                *x += h;
            }
        }

        gemv(
            isa,
            &self.policy_fc.weight,
            &self.policy_fc.bias,
            &s.fused,
            &mut s.policy_hidden,
        );
        silu(&mut s.policy_hidden[..self.policy_fc.out_width()]);
        gemv(
            isa,
            &self.value_fc.weight,
            &self.value_fc.bias,
            &s.fused,
            &mut s.value_hidden,
        );
        silu(&mut s.value_hidden[..self.value_fc.out_width()]);
        gemv(
            isa,
            &self.value_head.weight,
            &self.value_head.bias,
            &s.value_hidden,
            &mut s.value_out,
        );
        s.value_out[0]
    }

    /// Policy logits for `actions` (after `forward_trunk`), in the given order.
    pub(super) fn policy_logits(&self, scratch: &HeadsScratch, actions: &[usize], out: &mut [f32]) {
        gemv_rows(
            self.isa,
            &self.policy_head.weight,
            &self.policy_head.bias,
            &scratch.policy_hidden,
            actions,
            out,
        );
    }

    /// Policy logits for every action (after `forward_trunk`).
    pub(super) fn all_policy_logits(&self, scratch: &HeadsScratch) -> Vec<f32> {
        let mut logits = vec![0.0; self.policy_head.weight.padded_rows()];
        gemv(
            self.isa,
            &self.policy_head.weight,
            &self.policy_head.bias,
            &scratch.policy_hidden,
            &mut logits,
        );
        logits.truncate(self.num_actions());
        logits
    }

    fn norm_silu(&self, values: &mut [f32], norm: &Norm) {
        layer_norm(values, &norm.gamma, &norm.beta, self.eps);
        silu(values);
    }
}

/// Pulls weights out of the parsed graph by PyTorch parameter name and checks that the
/// consuming nodes are the ops the native implementation assumes.
///
/// The exporter keeps parameter names on the consuming nodes, but stores parameters with
/// identical values once and re-exposes the duplicates through `Identity` nodes
/// (e.g. freshly initialized LayerNorm weights), so names resolve through those aliases.
struct WeightReader<'a> {
    path: &'a Path,
    initializers: HashMap<String, OnnxTensor>,
    /// `Identity` output name -> initializer name.
    aliases: HashMap<String, String>,
    used_initializers: HashSet<String>,
    nodes: &'a [OnnxNode],
    eps: Option<f32>,
    linear_layers: usize,
    norm_layers: usize,
}

impl<'a> WeightReader<'a> {
    fn new(
        path: &'a Path,
        initializers: HashMap<String, OnnxTensor>,
        nodes: &'a [OnnxNode],
    ) -> LoadResult<Self> {
        let mut aliases = HashMap::new();
        for node in nodes.iter().filter(|node| node.op_type == "Identity") {
            match (node.inputs.as_slice(), node.outputs.as_slice()) {
                ([source], [alias]) if initializers.contains_key(source) => {
                    aliases.insert(alias.clone(), source.clone());
                }
                _ => {
                    return invalid_model(
                        path,
                        format!(
                            "Identity node {:?} does not alias an initializer",
                            node.outputs
                        ),
                    );
                }
            }
        }
        Ok(Self {
            path,
            initializers,
            aliases,
            used_initializers: HashSet::new(),
            nodes,
            eps: None,
            linear_layers: 0,
            norm_layers: 0,
        })
    }

    /// Name of the initializer holding `name`'s data.
    fn resolve(&self, name: &str) -> Option<String> {
        if self.initializers.contains_key(name) {
            return Some(name.to_string());
        }
        self.aliases.get(name).cloned()
    }

    fn has(&self, name: &str) -> bool {
        self.resolve(name).is_some()
    }

    fn take(&mut self, name: &str, rank: usize) -> LoadResult<(Vec<f32>, Vec<usize>)> {
        let Some(source) = self.resolve(name) else {
            return invalid_model(self.path, format!("missing initializer '{name}'"));
        };
        let tensor = &self.initializers[&source];
        if tensor.dims.len() != rank {
            return invalid_model(
                self.path,
                format!(
                    "initializer '{name}' has dims {:?}, expected rank {rank}",
                    tensor.dims
                ),
            );
        }
        let taken = (tensor.data.clone(), tensor.dims.clone());
        self.used_initializers.insert(source);
        Ok(taken)
    }

    fn take_shape(&mut self, name: &str, dims: &[usize]) -> LoadResult<(Vec<f32>, Vec<usize>)> {
        let (data, actual) = self.take(name, dims.len())?;
        if actual != dims {
            return invalid_model(
                self.path,
                format!("initializer '{name}' has dims {actual:?}, expected {dims:?}"),
            );
        }
        Ok((data, actual))
    }

    fn node_using(&self, op_type: &str, input_index: usize, name: &str) -> LoadResult<&OnnxNode> {
        let mut matches = self
            .nodes
            .iter()
            .filter(|node| node.inputs.get(input_index).map(String::as_str) == Some(name));
        match (matches.next(), matches.next()) {
            (Some(node), None) if node.op_type == op_type => Ok(node),
            (Some(node), None) => invalid_model(
                self.path,
                format!("'{name}' feeds a {} node, expected {op_type}", node.op_type),
            ),
            (None, _) => invalid_model(self.path, format!("no {op_type} node reads '{name}'")),
            (Some(_), Some(_)) => {
                invalid_model(self.path, format!("'{name}' is used by several nodes"))
            }
        }
    }

    fn check_gemm(&mut self, prefix: &str) -> LoadResult<()> {
        let weight_name = format!("{prefix}.weight");
        let node = self.node_using("Gemm", 1, &weight_name)?;
        let int_attr = |name: &str, default: i64| {
            node.attributes
                .get(name)
                .and_then(|a| a.int)
                .unwrap_or(default)
        };
        let float_attr = |name: &str| {
            node.attributes
                .get(name)
                .and_then(|a| a.float)
                .unwrap_or(1.0)
        };
        if node.inputs.get(2).map(String::as_str) != Some(format!("{prefix}.bias").as_str())
            || int_attr("transA", 0) != 0
            || int_attr("transB", 0) != 1
            || float_attr("alpha") != 1.0
            || float_attr("beta") != 1.0
        {
            return invalid_model(
                self.path,
                format!("Gemm for '{prefix}' must be Y = X . W^T + b (transB=1, alpha=beta=1)"),
            );
        }
        self.linear_layers += 1;
        Ok(())
    }

    fn linear(&mut self, prefix: &str, rows: Option<usize>, cols: usize) -> LoadResult<Dense> {
        let (weight, dims) = self.take(&format!("{prefix}.weight"), 2)?;
        let out_rows = rows.unwrap_or(dims[0]);
        if dims != [out_rows, cols] {
            return invalid_model(
                self.path,
                format!("'{prefix}.weight' has dims {dims:?}, expected [{out_rows}, {cols}]"),
            );
        }
        let (bias, _) = self.take_shape(&format!("{prefix}.bias"), &[out_rows])?;
        self.check_gemm(prefix)?;
        Ok(Dense::new(
            PaddedMatrix::from_row_major(&weight, out_rows, cols),
            &bias,
        ))
    }

    fn norm(&mut self, prefix: &str, width: usize) -> LoadResult<Norm> {
        let weight_name = format!("{prefix}.weight");
        let (gamma, _) = self.take_shape(&weight_name, &[width])?;
        let (beta, _) = self.take_shape(&format!("{prefix}.bias"), &[width])?;
        let node = self.node_using("LayerNormalization", 1, &weight_name)?;
        let axis = node
            .attributes
            .get("axis")
            .and_then(|a| a.int)
            .unwrap_or(-1);
        let eps = node
            .attributes
            .get("epsilon")
            .and_then(|a| a.float)
            .unwrap_or(1e-5);
        if node.inputs.get(2).map(String::as_str) != Some(format!("{prefix}.bias").as_str())
            || !(axis == -1 || axis == 1)
        {
            return invalid_model(
                self.path,
                format!("LayerNormalization for '{prefix}' must normalize the last axis"),
            );
        }
        match self.eps {
            Some(existing) if existing != eps => {
                return invalid_model(self.path, "LayerNormalization epsilons differ");
            }
            _ => self.eps = Some(eps),
        }
        self.norm_layers += 1;
        Ok(Norm { gamma, beta })
    }

    /// Every initializer must have been consumed and the op histogram must match exactly.
    fn finish(&self, expected_ops: &[(&str, usize)]) -> LoadResult<()> {
        let mut unused: Vec<&String> = self
            .initializers
            .keys()
            .filter(|name| !self.used_initializers.contains(*name))
            .collect();
        if !unused.is_empty() {
            unused.sort();
            return invalid_model(self.path, format!("unexpected initializers {unused:?}"));
        }
        let mut actual: HashMap<&str, usize> = HashMap::new();
        for node in self.nodes {
            *actual.entry(node.op_type.as_str()).or_default() += 1;
        }
        let expected: HashMap<&str, usize> = expected_ops
            .iter()
            .copied()
            .filter(|&(_, count)| count > 0)
            .collect();
        if actual != expected {
            return invalid_model(
                self.path,
                format!("unexpected op histogram {actual:?}, expected {expected:?}"),
            );
        }
        Ok(())
    }
}

/// Independent f64 implementation of the heads used to check the native kernels.
#[cfg(test)]
pub(super) mod reference {
    use std::collections::HashMap;
    use std::path::Path;

    use super::super::heads_onnx::parse_model_graph;

    /// PyTorch-named weights of a heads ONNX file (Identity aliases resolved), plus the
    /// number of aux_mlp layers and residual fusion blocks.
    pub(in crate::inference) fn load_weights(
        path: &Path,
    ) -> (HashMap<String, Vec<f32>>, usize, usize) {
        let graph =
            parse_model_graph(&std::fs::read(path).expect("read heads model")).expect("parse");
        let mut weights: HashMap<String, Vec<f32>> = graph
            .initializers
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.data.clone()))
            .collect();
        for node in graph.nodes.iter().filter(|node| node.op_type == "Identity") {
            let data = weights[&node.inputs[0]].clone();
            weights.insert(node.outputs[0].clone(), data);
        }
        let aux_layers = (0..)
            .take_while(|i| weights.contains_key(&format!("aux_mlp.{}.weight", 3 * i)))
            .count();
        let blocks = (0..)
            .take_while(|i| weights.contains_key(&format!("fusion_blocks.{i}.ln1.weight")))
            .count();
        (weights, aux_layers, blocks)
    }

    fn linear(w: &HashMap<String, Vec<f32>>, prefix: &str, x: &[f64]) -> Vec<f64> {
        let weight = &w[&format!("{prefix}.weight")];
        let bias = &w[&format!("{prefix}.bias")];
        let cols = x.len();
        bias.iter()
            .enumerate()
            .map(|(r, &b)| {
                let row = &weight[r * cols..(r + 1) * cols];
                row.iter()
                    .zip(x)
                    .map(|(&wv, &xv)| f64::from(wv) * xv)
                    .sum::<f64>()
                    + f64::from(b)
            })
            .collect()
    }

    fn norm(w: &HashMap<String, Vec<f32>>, prefix: &str, x: &[f64]) -> Vec<f64> {
        let gamma = &w[&format!("{prefix}.weight")];
        let beta = &w[&format!("{prefix}.bias")];
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| (v - mean) * inv_std * f64::from(gamma[i]) + f64::from(beta[i]))
            .collect()
    }

    fn silu(x: Vec<f64>) -> Vec<f64> {
        x.into_iter().map(|v| v / (1.0 + (-v).exp())).collect()
    }

    /// f64 forward pass straight from the PyTorch-named weights: (logits, value).
    pub(in crate::inference) fn forward(
        w: &HashMap<String, Vec<f32>>,
        aux_layers: usize,
        blocks: usize,
        board_h: &[f32],
        piece_aux: &[f32],
    ) -> (Vec<f64>, f64) {
        let mut aux: Vec<f64> = piece_aux.iter().map(|&v| f64::from(v)).collect();
        for layer in 0..aux_layers {
            let index = 3 * layer;
            let h = linear(w, &format!("aux_mlp.{index}"), &aux);
            aux = silu(norm(w, &format!("aux_mlp.{}", index + 1), &h));
        }
        let mut concat: Vec<f64> = board_h.iter().map(|&v| f64::from(v)).collect();
        concat.extend(aux);
        let pre = linear(w, "fusion_fc", &concat);
        let mut fused = silu(norm(w, "fusion_ln", &pre));
        for block in 0..blocks {
            let p = format!("fusion_blocks.{block}");
            let h = silu(norm(w, &format!("{p}.ln1"), &fused));
            let h = linear(w, &format!("{p}.fc1"), &h);
            let h = silu(norm(w, &format!("{p}.ln2"), &h));
            let h = linear(w, &format!("{p}.fc2"), &h);
            fused = fused.iter().zip(&h).map(|(a, b)| a + b).collect();
        }
        let p = silu(linear(w, "policy_fc", &fused));
        let logits = linear(w, "policy_head", &p);
        let v = silu(linear(w, "value_fc", &fused));
        let value = linear(w, "value_head", &v)[0];
        (logits, value)
    }
}

#[cfg(test)]
mod tests {
    use super::super::heads_onnx::proto_writer::{bytes_field, varint, varint_field};
    use super::*;

    const FLOAT: u64 = 1;

    /// Builds an ONNX heads graph shaped like the PyTorch exporter's output, keeping the raw
    /// weights for an independent f64 reference forward pass.
    struct HeadsGraphBuilder {
        nodes: Vec<Vec<u8>>,
        initializers: Vec<Vec<u8>>,
        weights: HashMap<String, Vec<f32>>,
        /// (dims, bits) -> first initializer name, for exporter-style de-duplication.
        stored: HashMap<(Vec<usize>, Vec<u32>), String>,
        fresh_norms: bool,
        rng_state: u64,
        next_value: usize,
    }

    #[derive(Clone, Copy)]
    struct Shape {
        board_hidden: usize,
        aux_hidden: usize,
        aux_layers: usize,
        fusion_hidden: usize,
        blocks: usize,
        /// LayerNorms at their initial values (ones/zeros), which the exporter de-duplicates.
        fresh_norms: bool,
    }

    fn attribute_int(name: &str, value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        bytes_field(1, name.as_bytes(), &mut out);
        varint_field(3, value as u64, &mut out);
        out
    }

    fn attribute_float(name: &str, value: f32) -> Vec<u8> {
        let mut out = Vec::new();
        bytes_field(1, name.as_bytes(), &mut out);
        varint(2 << 3 | 5, &mut out);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    impl HeadsGraphBuilder {
        fn new(seed: u64, fresh_norms: bool) -> Self {
            Self {
                nodes: Vec::new(),
                initializers: Vec::new(),
                weights: HashMap::new(),
                stored: HashMap::new(),
                fresh_norms,
                rng_state: seed,
                next_value: 0,
            }
        }

        fn random(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.rng_state = self
                        .rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.rng_state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
                })
                .collect()
        }

        fn initializer(&mut self, name: &str, dims: &[usize], data: Vec<f32>) {
            self.weights.insert(name.to_string(), data.clone());
            let key = (dims.to_vec(), data.iter().map(|v| v.to_bits()).collect());
            if let Some(source) = self.stored.get(&key).cloned() {
                self.node_with_output("Identity", &[&source], name, &[]);
                return;
            }
            self.stored.insert(key, name.to_string());
            let mut tensor = Vec::new();
            for &dim in dims {
                varint_field(1, dim as u64, &mut tensor);
            }
            varint_field(2, FLOAT, &mut tensor);
            bytes_field(8, name.as_bytes(), &mut tensor);
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            bytes_field(9, &raw, &mut tensor);
            self.initializers.push(tensor);
        }

        fn node(&mut self, op: &str, inputs: &[&str], attributes: &[Vec<u8>]) -> String {
            self.next_value += 1;
            let output = format!("/value_{}", self.next_value);
            self.node_with_output(op, inputs, &output, attributes);
            output
        }

        fn node_with_output(
            &mut self,
            op: &str,
            inputs: &[&str],
            output: &str,
            attributes: &[Vec<u8>],
        ) {
            let mut node = Vec::new();
            for input in inputs {
                bytes_field(1, input.as_bytes(), &mut node);
            }
            bytes_field(2, output.as_bytes(), &mut node);
            bytes_field(4, op.as_bytes(), &mut node);
            for attribute in attributes {
                bytes_field(5, attribute, &mut node);
            }
            self.nodes.push(node);
        }

        fn gemm(
            &mut self,
            x: &str,
            prefix: &str,
            rows: usize,
            cols: usize,
            output: Option<&str>,
        ) -> String {
            let scale = 1.0 / (cols as f32).sqrt();
            let weight = self.random(rows * cols, scale);
            let bias = self.random(rows, 0.3);
            self.initializer(&format!("{prefix}.weight"), &[rows, cols], weight);
            self.initializer(&format!("{prefix}.bias"), &[rows], bias);
            let attributes = [
                attribute_float("alpha", 1.0),
                attribute_float("beta", 1.0),
                attribute_int("transB", 1),
            ];
            let inputs = [x, &format!("{prefix}.weight"), &format!("{prefix}.bias")];
            match output {
                Some(output) => {
                    self.node_with_output("Gemm", &inputs, output, &attributes);
                    output.to_string()
                }
                None => self.node("Gemm", &inputs, &attributes),
            }
        }

        fn layer_norm(&mut self, x: &str, prefix: &str, width: usize) -> String {
            let (gamma, beta) = if self.fresh_norms {
                (vec![1.0; width], vec![0.0; width])
            } else {
                (
                    self.random(width, 0.5).iter().map(|v| v + 1.0).collect(),
                    self.random(width, 0.2),
                )
            };
            self.initializer(&format!("{prefix}.weight"), &[width], gamma);
            self.initializer(&format!("{prefix}.bias"), &[width], beta);
            self.node(
                "LayerNormalization",
                &[x, &format!("{prefix}.weight"), &format!("{prefix}.bias")],
                &[attribute_int("axis", -1), attribute_float("epsilon", 1e-5)],
            )
        }

        fn silu(&mut self, x: &str) -> String {
            let sigmoid = self.node("Sigmoid", &[x], &[]);
            self.node("Mul", &[x, &sigmoid], &[])
        }

        fn build(shape: Shape, seed: u64) -> (Vec<u8>, HashMap<String, Vec<f32>>) {
            let mut b = Self::new(seed, shape.fresh_norms);
            let mut aux = "piece_aux".to_string();
            let mut width = PIECE_AUX_FEATURES;
            for layer in 0..shape.aux_layers {
                let index = 3 * layer;
                let h = b.gemm(
                    &aux,
                    &format!("aux_mlp.{index}"),
                    shape.aux_hidden,
                    width,
                    None,
                );
                let h = b.layer_norm(&h, &format!("aux_mlp.{}", index + 1), shape.aux_hidden);
                aux = b.silu(&h);
                width = shape.aux_hidden;
            }
            let concat = b.node("Concat", &["board_h", &aux], &[attribute_int("axis", 1)]);
            let f = shape.fusion_hidden;
            let pre = b.gemm(
                &concat,
                "fusion_fc",
                f,
                shape.board_hidden + shape.aux_hidden,
                None,
            );
            let pre = b.layer_norm(&pre, "fusion_ln", f);
            let mut fused = b.silu(&pre);
            for block in 0..shape.blocks {
                let p = format!("fusion_blocks.{block}");
                let h = b.layer_norm(&fused, &format!("{p}.ln1"), f);
                let h = b.silu(&h);
                let h = b.gemm(&h, &format!("{p}.fc1"), f, f, None);
                let h = b.layer_norm(&h, &format!("{p}.ln2"), f);
                let h = b.silu(&h);
                let h = b.gemm(&h, &format!("{p}.fc2"), f, f, None);
                fused = b.node("Add", &[&fused, &h], &[]);
            }
            let p = b.gemm(&fused, "policy_fc", 2 * f, f, None);
            let p = b.silu(&p);
            b.gemm(&p, "policy_head", NUM_ACTIONS, 2 * f, Some("policy_logits"));
            let v = b.gemm(&fused, "value_fc", f / 2, f, None);
            let v = b.silu(&v);
            b.gemm(&v, "value_head", 1, f / 2, Some("value"));

            let mut graph = Vec::new();
            for node in &b.nodes {
                bytes_field(1, node, &mut graph);
            }
            for initializer in &b.initializers {
                bytes_field(5, initializer, &mut graph);
            }
            for (field, name) in [
                (11, "board_h"),
                (11, "piece_aux"),
                (12, "policy_logits"),
                (12, "value"),
            ] {
                let mut value_info = Vec::new();
                bytes_field(1, name.as_bytes(), &mut value_info);
                bytes_field(field, &value_info, &mut graph);
            }
            let mut model = Vec::new();
            bytes_field(7, &graph, &mut model);
            (model, b.weights)
        }
    }

    fn load_graph(bytes: &[u8], board_hidden: usize, isa: Isa) -> LoadResult<HeadsModel> {
        let graph = parse_model_graph(bytes).expect("synthetic graph parses");
        HeadsModel::from_graph(graph, Path::new("synthetic.heads.onnx"), board_hidden, isa)
    }

    fn sample_inputs(seed: u64, shape: Shape) -> (Vec<f32>, Vec<f32>) {
        let mut b = HeadsGraphBuilder::new(seed, false);
        let board_h = b
            .random(shape.board_hidden, 1.0)
            .iter()
            .map(|v| v.abs())
            .collect();
        let mut aux = vec![0.0; PIECE_AUX_FEATURES];
        for (i, v) in aux.iter_mut().enumerate() {
            if (i * 7 + seed as usize) % 5 == 0 {
                *v = 1.0;
            }
        }
        aux[51] = 0.37;
        (board_h, aux)
    }

    fn assert_forward_matches_reference(shape: Shape) {
        let (bytes, weights) = HeadsGraphBuilder::build(shape, 42);
        let mut isas = vec![Isa::Scalar];
        if Isa::detect() != Isa::Scalar {
            isas.push(Isa::detect());
        }
        for isa in isas {
            let model = load_graph(&bytes, shape.board_hidden, isa).expect("synthetic graph loads");
            let mut scratch = model.new_scratch();
            for seed in 0..4 {
                let (board_h, aux) = sample_inputs(seed, shape);
                let (expected_logits, expected_value) =
                    reference::forward(&weights, shape.aux_layers, shape.blocks, &board_h, &aux);

                let mut board_input = vec![0.0; model.board_fusion_width()];
                model.board_fusion_input(&board_h, &mut scratch, &mut board_input);
                let value = model.forward_trunk(&board_input, &aux, &mut scratch);
                assert!(
                    (f64::from(value) - expected_value).abs() < 2e-5 * (1.0 + expected_value.abs()),
                    "{isa:?} value {value} vs {expected_value}"
                );
                let logits = model.all_policy_logits(&scratch);
                assert_eq!(logits.len(), NUM_ACTIONS);
                for (a, (&got, &want)) in logits.iter().zip(&expected_logits).enumerate() {
                    assert!(
                        (f64::from(got) - want).abs() < 2e-5 * (1.0 + want.abs()),
                        "{isa:?} logit {a}: {got} vs {want}"
                    );
                }

                let actions = [671, 0, 13, 400, 13, 77, 5];
                let mut gathered = vec![0.0; actions.len()];
                model.policy_logits(&scratch, &actions, &mut gathered);
                for (&a, &g) in actions.iter().zip(&gathered) {
                    assert_eq!(
                        g.to_bits(),
                        logits[a].to_bits(),
                        "{isa:?} gathered logit {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_matches_reference_for_odd_widths_and_depths() {
        assert_forward_matches_reference(Shape {
            board_hidden: 40,
            aux_hidden: 24,
            aux_layers: 3,
            fusion_hidden: 36,
            blocks: 2,
            fresh_norms: false,
        });
    }

    #[test]
    fn forward_matches_reference_without_hidden_aux_layers_or_blocks() {
        assert_forward_matches_reference(Shape {
            board_hidden: 128,
            aux_hidden: 64,
            aux_layers: 1,
            fusion_hidden: 256,
            blocks: 0,
            fresh_norms: false,
        });
    }

    #[test]
    fn forward_resolves_deduplicated_initializers() {
        assert_forward_matches_reference(Shape {
            board_hidden: 32,
            aux_hidden: 32,
            aux_layers: 2,
            fusion_hidden: 32,
            blocks: 2,
            fresh_norms: true,
        });
    }

    #[test]
    fn rejects_board_width_mismatch() {
        let shape = Shape {
            board_hidden: 32,
            aux_hidden: 16,
            aux_layers: 2,
            fusion_hidden: 32,
            blocks: 1,
            fresh_norms: false,
        };
        let (bytes, _) = HeadsGraphBuilder::build(shape, 1);
        let error = load_graph(&bytes, 48, Isa::detect())
            .err()
            .expect("must fail");
        assert!(
            error
                .to_string()
                .contains("re-export split models together"),
            "{error}"
        );
    }

    #[test]
    fn rejects_unexpected_ops_and_initializers() {
        let shape = Shape {
            board_hidden: 32,
            aux_hidden: 16,
            aux_layers: 2,
            fusion_hidden: 32,
            blocks: 1,
            fresh_norms: false,
        };
        let (bytes, _) = HeadsGraphBuilder::build(shape, 1);

        let mut graph = parse_model_graph(&bytes).unwrap();
        graph.nodes.push(OnnxNode {
            op_type: "Relu".to_string(),
            inputs: vec!["/value_1".to_string()],
            outputs: vec!["/relu".to_string()],
            attributes: HashMap::new(),
        });
        let error = HeadsModel::from_graph(graph, Path::new("x"), 32, Isa::detect())
            .err()
            .expect("extra op must fail");
        assert!(error.to_string().contains("op histogram"), "{error}");

        let mut graph = parse_model_graph(&bytes).unwrap();
        graph.initializers.insert(
            "extra.weight".to_string(),
            OnnxTensor {
                dims: vec![1],
                data: vec![0.0],
            },
        );
        let error = HeadsModel::from_graph(graph, Path::new("x"), 32, Isa::detect())
            .err()
            .expect("extra initializer must fail");
        assert!(
            error.to_string().contains("unexpected initializers"),
            "{error}"
        );

        let mut graph = parse_model_graph(&bytes).unwrap();
        let gemm = graph
            .nodes
            .iter_mut()
            .find(|n| n.op_type == "Gemm")
            .unwrap();
        gemm.attributes.remove("transB");
        let error = HeadsModel::from_graph(graph, Path::new("x"), 32, Isa::detect())
            .err()
            .expect("non-transposed Gemm must fail");
        assert!(error.to_string().contains("transB=1"), "{error}");
    }
}
