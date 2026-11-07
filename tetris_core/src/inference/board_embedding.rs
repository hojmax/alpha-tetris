//! Native cached board path: conv trunk (`X.conv.onnx`) + board MLP (`X.fc.bin`).
//!
//! `board_h = SiLU(fc2(SiLU(LN(fc1([conv(board) ++ SiLU(LN(stats_fc(board_stats)))])))))`
//!
//! The fc1 columns are permuted at load time to consume the trunk's channels-last features
//! directly, so no transpose is needed per call.

use std::io;
use std::path::Path;

use super::conv::{ConvScratch, ConvTrunk};
#[cfg(target_arch = "x86_64")]
use super::kernels::Avx2;
#[cfg(target_arch = "aarch64")]
use super::kernels::Neon;
use super::kernels::{
    layer_norm_silu_in_place, linear, silu_in_place, ConvKernel, Isa, PackedLinear, Portable,
    CONV_NARROW_CHANNEL_TILE,
};
use super::onnx_proto::invalid_data;
use super::{CachedBoardPath, LayerNorm1D, LinearLayer};
use crate::game::constants::{BOARD_HEIGHT, BOARD_STATS_FEATURES, BOARD_WIDTH};

/// PyTorch `nn.LayerNorm` default; `fc.bin` (TCM2) does not store it.
const LAYER_NORM_EPSILON: f32 = 1e-5;

struct NormAffine {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

pub(crate) struct BoardEmbedder {
    trunk: ConvTrunk,
    stats_fc: PackedLinear,
    stats_ln: NormAffine,
    fc1: PackedLinear,
    ln1: NormAffine,
    fc2: PackedLinear,
}

pub(crate) struct BoardEmbeddingScratch {
    conv: ConvScratch,
    /// `[conv features ++ board-stats embedding]`, the fc1 input.
    fc1_input: Vec<f32>,
    stats_out: Vec<f32>,
    hidden: Vec<f32>,
    output: Vec<f32>,
}

fn pack_linear(layer: &LinearLayer, input_order: &[usize]) -> PackedLinear {
    let weight = layer
        .weight
        .as_slice()
        .expect("fc.bin weights are loaded as standard-layout arrays");
    PackedLinear::pack(
        weight,
        layer.bias.as_slice().expect("contiguous bias"),
        layer.weight.ncols(),
        layer.weight.nrows(),
        input_order,
    )
}

fn norm_affine(layer: &LayerNorm1D) -> NormAffine {
    NormAffine {
        gamma: layer.weight.to_vec(),
        beta: layer.bias.to_vec(),
    }
}

impl BoardEmbedder {
    /// Load the conv trunk from `conv_path` and combine it with the fc.bin board MLP.
    pub(crate) fn load(conv_path: &Path, board_path: &CachedBoardPath) -> io::Result<Self> {
        Self::new(ConvTrunk::load(conv_path)?, board_path)
    }

    pub(crate) fn new(trunk: ConvTrunk, board_path: &CachedBoardPath) -> io::Result<Self> {
        let stats_hidden = board_path.board_stats_fc.weight.nrows();
        let fc1_inputs = board_path.board_proj_fc1.weight.ncols();
        let conv_out = trunk.output_size();
        if fc1_inputs != conv_out + stats_hidden {
            return Err(invalid_data(format!(
                "Conv output size mismatch: conv.onnx={conv_out} fc.bin expects conv_out={} (re-export split models together)",
                fc1_inputs.saturating_sub(stats_hidden)
            )));
        }
        let hidden = board_path.board_proj_fc1.weight.nrows();
        let output = board_path.board_proj_fc2.weight.nrows();
        for (label, width) in [
            ("board_stats_hidden", stats_hidden),
            ("board_proj_hidden", hidden),
            ("board embedding", output),
        ] {
            if width % CONV_NARROW_CHANNEL_TILE != 0 {
                return Err(invalid_data(format!(
                    "{label} width {width} must be a multiple of {CONV_NARROW_CHANNEL_TILE}"
                )));
            }
        }

        let identity = |n: usize| (0..n).collect::<Vec<_>>();
        let fc1_order: Vec<usize> = (0..conv_out)
            .map(|k| trunk.feature_source_index(k))
            .chain(conv_out..fc1_inputs)
            .collect();
        Ok(Self {
            stats_fc: pack_linear(&board_path.board_stats_fc, &identity(BOARD_STATS_FEATURES)),
            stats_ln: norm_affine(&board_path.board_stats_ln),
            fc1: pack_linear(&board_path.board_proj_fc1, &fc1_order),
            ln1: norm_affine(&board_path.board_proj_ln1),
            fc2: pack_linear(&board_path.board_proj_fc2, &identity(hidden)),
            trunk,
        })
    }

    pub(crate) fn output_size(&self) -> usize {
        self.fc2.outputs
    }

    pub(crate) fn new_scratch(&self) -> BoardEmbeddingScratch {
        BoardEmbeddingScratch {
            conv: self.trunk.new_scratch(),
            fc1_input: vec![0.0; self.fc1.inputs],
            stats_out: vec![0.0; self.stats_fc.padded_outputs()],
            hidden: vec![0.0; self.fc1.padded_outputs()],
            output: vec![0.0; self.fc2.padded_outputs()],
        }
    }

    /// Board embedding `board_h` for a 200-cell board plane and the 19 board statistics.
    pub(crate) fn embed<'s>(
        &self,
        board: &[f32],
        board_stats: &[f32],
        scratch: &'s mut BoardEmbeddingScratch,
    ) -> io::Result<&'s [f32]> {
        if board.len() != BOARD_HEIGHT * BOARD_WIDTH || board_stats.len() != BOARD_STATS_FEATURES {
            return Err(invalid_data(format!(
                "board embedding inputs must be {} board cells and {BOARD_STATS_FEATURES} stats, got {} and {}",
                BOARD_HEIGHT * BOARD_WIDTH,
                board.len(),
                board_stats.len()
            )));
        }
        // SAFETY: each arm runs the kernels for an ISA that `Isa::detect` (or a test) verified.
        unsafe {
            match self.trunk.isa() {
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => self.embed_with::<Neon>(board, board_stats, scratch),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2Fma => embed_avx2(self, board, board_stats, scratch),
                Isa::Portable => self.embed_with::<Portable>(board, board_stats, scratch),
            }
        }
        Ok(&scratch.output[..self.output_size()])
    }

    #[inline(always)]
    unsafe fn embed_with<S: ConvKernel>(
        &self,
        board: &[f32],
        board_stats: &[f32],
        scratch: &mut BoardEmbeddingScratch,
    ) {
        let conv_out = self.trunk.output_size();
        let (conv_features, stats_features) = scratch.fc1_input.split_at_mut(conv_out);
        self.trunk
            .forward_with::<S>(board, &mut scratch.conv, conv_features);

        linear::<S>(&self.stats_fc, board_stats, &mut scratch.stats_out);
        let stats_h = &mut scratch.stats_out[..self.stats_fc.outputs];
        layer_norm_silu_in_place::<S>(
            stats_h,
            &self.stats_ln.gamma,
            &self.stats_ln.beta,
            LAYER_NORM_EPSILON,
        );
        stats_features.copy_from_slice(stats_h);

        linear::<S>(&self.fc1, &scratch.fc1_input, &mut scratch.hidden);
        let hidden = &mut scratch.hidden[..self.fc1.outputs];
        layer_norm_silu_in_place::<S>(hidden, &self.ln1.gamma, &self.ln1.beta, LAYER_NORM_EPSILON);

        linear::<S>(&self.fc2, hidden, &mut scratch.output);
        silu_in_place::<S>(&mut scratch.output[..self.fc2.outputs]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn embed_avx2(
    embedder: &BoardEmbedder,
    board: &[f32],
    board_stats: &[f32],
    scratch: &mut BoardEmbeddingScratch,
) {
    embedder.embed_with::<Avx2>(board, board_stats, scratch)
}
