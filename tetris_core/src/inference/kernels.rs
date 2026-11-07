//! SIMD kernels for the native board-embedding path (conv trunk + cached board MLP).
//!
//! Kernels are written once against the small [`Simd`] trait and instantiated per ISA:
//! - aarch64: NEON (always available).
//! - x86_64: AVX2 + FMA when detected at runtime, otherwise the portable fallback.
//! - everything else: [`Portable`], an 8-lane array type that LLVM auto-vectorizes.
//!
//! The x86 entry points carry `#[target_feature(enable = "avx2,fma")]` and every generic helper
//! is `#[inline(always)]`, so the whole forward pass is compiled with AVX2/FMA inside those entry
//! points without relying on `-C target-cpu=native`.
//!
//! Activations use channels-last layout (`[pixel][channel]`) so SIMD lanes run over channels.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Instruction-set flavour chosen once at model load time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Isa {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    /// x86 CPUs without AVX2/FMA and other architectures; tests also run it on aarch64.
    #[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
    Portable,
}

impl Isa {
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Isa::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Isa::Avx2Fma;
            }
            Isa::Portable
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Isa::Portable
        }
    }

    /// Every ISA usable on this machine (the detected one plus the portable fallback).
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Self> {
        let detected = Self::detect();
        if detected == Isa::Portable {
            vec![Isa::Portable]
        } else {
            vec![detected, Isa::Portable]
        }
    }

    /// Output pixels per register block of the conv kernel; pixel tables are padded to a
    /// multiple of this so the kernel never needs a remainder path.
    pub(crate) fn conv_block_pixels(self) -> usize {
        match self {
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => NEON_CONV_BLOCK_PIXELS,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => AVX2_CONV_BLOCK_PIXELS,
            Isa::Portable => PORTABLE_CONV_BLOCK_PIXELS,
        }
    }
}

/// Output-channel tile of the conv kernel. Channel counts must be multiples of this or of
/// [`CONV_NARROW_CHANNEL_TILE`] (the trunk loader validates this).
pub(crate) const CONV_CHANNEL_TILE: usize = 16;
pub(crate) const CONV_NARROW_CHANNEL_TILE: usize = 8;
/// Rows per GEMV block (16 NEON or 8 AVX2 accumulators).
pub(crate) const GEMV_ROW_BLOCK: usize = 64;

#[cfg(target_arch = "aarch64")]
const NEON_CONV_BLOCK_PIXELS: usize = 5;
#[cfg(target_arch = "x86_64")]
const AVX2_CONV_BLOCK_PIXELS: usize = 6;
const PORTABLE_CONV_BLOCK_PIXELS: usize = 6;

/// Minimal f32 SIMD vector abstraction used by all kernels.
///
/// # Safety
/// Implementations may only be used on CPUs that support their instruction set; all
/// pointer arguments must be valid for `LANES` consecutive f32 reads/writes.
pub(crate) trait Simd: Copy {
    const LANES: usize;
    unsafe fn splat(value: f32) -> Self;
    unsafe fn load(ptr: *const f32) -> Self;
    unsafe fn load_splat(ptr: *const f32) -> Self;
    unsafe fn store(self, ptr: *mut f32);
    unsafe fn add(self, other: Self) -> Self;
    unsafe fn sub(self, other: Self) -> Self;
    unsafe fn mul(self, other: Self) -> Self;
    unsafe fn div(self, other: Self) -> Self;
    /// `self * b + c`, fused where the ISA supports it.
    unsafe fn mul_add(self, b: Self, c: Self) -> Self;
    unsafe fn min(self, other: Self) -> Self;
    unsafe fn max(self, other: Self) -> Self;
    /// `2^self` for lanes holding integral values in `[-126, 127]`.
    unsafe fn pow2i(self) -> Self;
    unsafe fn reduce_add(self) -> f32;
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) struct Neon(float32x4_t);

#[cfg(target_arch = "aarch64")]
impl Simd for Neon {
    const LANES: usize = 4;
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Neon(vdupq_n_f32(value))
    }
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        Neon(vld1q_f32(ptr))
    }
    #[inline(always)]
    unsafe fn load_splat(ptr: *const f32) -> Self {
        Neon(vld1q_dup_f32(ptr))
    }
    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        vst1q_f32(ptr, self.0)
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        Neon(vaddq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn sub(self, other: Self) -> Self {
        Neon(vsubq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn mul(self, other: Self) -> Self {
        Neon(vmulq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn div(self, other: Self) -> Self {
        Neon(vdivq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn mul_add(self, b: Self, c: Self) -> Self {
        Neon(vfmaq_f32(c.0, self.0, b.0))
    }
    #[inline(always)]
    unsafe fn min(self, other: Self) -> Self {
        Neon(vminq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        Neon(vmaxq_f32(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn pow2i(self) -> Self {
        let biased = vaddq_s32(vcvtq_s32_f32(self.0), vdupq_n_s32(127));
        Neon(vreinterpretq_f32_s32(vshlq_n_s32::<23>(biased)))
    }
    #[inline(always)]
    unsafe fn reduce_add(self) -> f32 {
        vaddvq_f32(self.0)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx2(__m256);

#[cfg(target_arch = "x86_64")]
impl Simd for Avx2 {
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Avx2(_mm256_set1_ps(value))
    }
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        Avx2(_mm256_loadu_ps(ptr))
    }
    #[inline(always)]
    unsafe fn load_splat(ptr: *const f32) -> Self {
        Avx2(_mm256_broadcast_ss(&*ptr))
    }
    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        _mm256_storeu_ps(ptr, self.0)
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        Avx2(_mm256_add_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn sub(self, other: Self) -> Self {
        Avx2(_mm256_sub_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn mul(self, other: Self) -> Self {
        Avx2(_mm256_mul_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn div(self, other: Self) -> Self {
        Avx2(_mm256_div_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn mul_add(self, b: Self, c: Self) -> Self {
        Avx2(_mm256_fmadd_ps(self.0, b.0, c.0))
    }
    #[inline(always)]
    unsafe fn min(self, other: Self) -> Self {
        Avx2(_mm256_min_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        Avx2(_mm256_max_ps(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn pow2i(self) -> Self {
        let biased = _mm256_add_epi32(_mm256_cvttps_epi32(self.0), _mm256_set1_epi32(127));
        Avx2(_mm256_castsi256_ps(_mm256_slli_epi32::<23>(biased)))
    }
    #[inline(always)]
    unsafe fn reduce_add(self) -> f32 {
        let pairs = _mm_add_ps(
            _mm256_castps256_ps128(self.0),
            _mm256_extractf128_ps::<1>(self.0),
        );
        let quads = _mm_add_ps(pairs, _mm_movehl_ps(pairs, pairs));
        _mm_cvtss_f32(_mm_add_ss(quads, _mm_shuffle_ps::<0b01>(quads, quads)))
    }
}

/// Portable 8-lane fallback (also used by tests to exercise the 8-lane kernel shapes on any host).
#[derive(Clone, Copy)]
pub(crate) struct Portable([f32; 8]);

impl Portable {
    #[inline(always)]
    fn map2(self, other: Self, op: impl Fn(f32, f32) -> f32) -> Self {
        let mut out = [0.0; 8];
        for lane in 0..8 {
            out[lane] = op(self.0[lane], other.0[lane]);
        }
        Portable(out)
    }
}

impl Simd for Portable {
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Portable([value; 8])
    }
    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        Portable(std::ptr::read_unaligned(ptr as *const [f32; 8]))
    }
    #[inline(always)]
    unsafe fn load_splat(ptr: *const f32) -> Self {
        Portable([*ptr; 8])
    }
    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        std::ptr::write_unaligned(ptr as *mut [f32; 8], self.0)
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        self.map2(other, |a, b| a + b)
    }
    #[inline(always)]
    unsafe fn sub(self, other: Self) -> Self {
        self.map2(other, |a, b| a - b)
    }
    #[inline(always)]
    unsafe fn mul(self, other: Self) -> Self {
        self.map2(other, |a, b| a * b)
    }
    #[inline(always)]
    unsafe fn div(self, other: Self) -> Self {
        self.map2(other, |a, b| a / b)
    }
    #[inline(always)]
    unsafe fn mul_add(self, b: Self, c: Self) -> Self {
        // Unfused on purpose: `f32::mul_add` is a slow libm call on x86 CPUs without FMA,
        // which is exactly where this fallback runs.
        self.mul(b).add(c)
    }
    #[inline(always)]
    unsafe fn min(self, other: Self) -> Self {
        self.map2(other, f32::min)
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        self.map2(other, f32::max)
    }
    #[inline(always)]
    unsafe fn pow2i(self) -> Self {
        let mut out = [0.0; 8];
        for lane in 0..8 {
            out[lane] = f32::from_bits((((self.0[lane] as i32) + 127) << 23) as u32);
        }
        Portable(out)
    }
    #[inline(always)]
    unsafe fn reduce_add(self) -> f32 {
        let v = self.0;
        ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]))
    }
}

// ---------------------------------------------------------------------------------------------
// Elementwise math
// ---------------------------------------------------------------------------------------------

/// Vectorized `exp` (Cephes `expf` polynomial, ~1 ulp on the clamped range).
#[inline(always)]
pub(crate) unsafe fn exp<S: Simd>(x: S) -> S {
    const ROUND_MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: adding it rounds to the nearest integer
    let x = x.max(S::splat(-87.0)).min(S::splat(88.0));
    let n = x
        .mul_add(S::splat(std::f32::consts::LOG2_E), S::splat(ROUND_MAGIC))
        .sub(S::splat(ROUND_MAGIC));
    let r = n.mul_add(S::splat(-0.693_359_4), x);
    let r = n.mul_add(S::splat(2.121_944_4e-4), r);
    let z = r.mul(r);
    let mut y = S::splat(1.987_569_1e-4);
    y = y.mul_add(r, S::splat(1.398_199_9e-3));
    y = y.mul_add(r, S::splat(8.333_452e-3));
    y = y.mul_add(r, S::splat(4.166_579_6e-2));
    y = y.mul_add(r, S::splat(1.666_666_5e-1));
    y = y.mul_add(r, S::splat(5.000_000_1e-1));
    y = y.mul_add(z, r).add(S::splat(1.0));
    y.mul(n.pow2i())
}

/// `x * sigmoid(x)`.
#[inline(always)]
pub(crate) unsafe fn silu<S: Simd>(x: S) -> S {
    let neg_exp = exp(S::splat(0.0).sub(x));
    x.div(S::splat(1.0).add(neg_exp))
}

// ---------------------------------------------------------------------------------------------
// 3x3 convolution
// ---------------------------------------------------------------------------------------------

/// Per-output-pixel offsets for one conv geometry, padded to a multiple of the ISA's pixel block.
/// Padding entries read from offset 0 of the input and write to a scratch area past the real
/// output rows, so the kernel can always process full blocks.
pub(crate) struct ConvPixelTable {
    /// Offset (in floats) of each output pixel's top-left tap in the padded input.
    pub input_offsets: Vec<usize>,
    /// Output pixel row index (`offset = row * out_channels`).
    pub output_rows: Vec<usize>,
}

/// Packed weights of one 3x3 conv layer: `[channel tile][tap][in_channel][tile lanes]`.
pub(crate) struct PackedConv3x3 {
    pub in_channels: usize,
    pub out_channels: usize,
    pub channel_tile: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl PackedConv3x3 {
    /// Pack PyTorch/ONNX `[out][in][3][3]` weights.
    pub(crate) fn pack(
        weights_oihw: &[f32],
        bias: &[f32],
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        assert_eq!(weights_oihw.len(), out_channels * in_channels * 9);
        assert_eq!(bias.len(), out_channels);
        let channel_tile = if out_channels % CONV_CHANNEL_TILE == 0 {
            CONV_CHANNEL_TILE
        } else {
            assert_eq!(out_channels % CONV_NARROW_CHANNEL_TILE, 0);
            CONV_NARROW_CHANNEL_TILE
        };
        let mut packed = vec![0.0; weights_oihw.len()];
        let mut idx = 0;
        for tile_start in (0..out_channels).step_by(channel_tile) {
            for tap in 0..9 {
                for ci in 0..in_channels {
                    for lane in 0..channel_tile {
                        let co = tile_start + lane;
                        packed[idx] = weights_oihw[(co * in_channels + ci) * 9 + tap];
                        idx += 1;
                    }
                }
            }
        }
        Self {
            in_channels,
            out_channels,
            channel_tile,
            weights: packed,
            bias: bias.to_vec(),
        }
    }
}

/// One register block: `P` output pixels x `NV` vectors of output channels.
#[inline(always)]
unsafe fn conv3x3_block<S: Simd, const NV: usize, const P: usize>(
    input: *const f32,
    input_offsets: *const usize,
    output: *mut f32,
    output_rows: *const usize,
    taps: &[usize; 9],
    in_channels: usize,
    out_channels: usize,
    tile_start: usize,
    tile_weights: *const f32,
    bias: *const f32,
    accumulate: bool,
) {
    let tile = NV * S::LANES;
    let mut acc = [[S::splat(0.0); NV]; P];
    let mut bias_vec = [S::splat(0.0); NV];
    for v in 0..NV {
        bias_vec[v] = S::load(bias.add(tile_start + v * S::LANES));
    }
    let mut out_ptrs = [output; P];
    for p in 0..P {
        out_ptrs[p] = output.add(*output_rows.add(p) * out_channels + tile_start);
        for v in 0..NV {
            acc[p][v] = if accumulate {
                S::load(out_ptrs[p].add(v * S::LANES)).add(bias_vec[v])
            } else {
                bias_vec[v]
            };
        }
    }
    for (tap, &tap_offset) in taps.iter().enumerate() {
        let mut in_ptrs = [input; P];
        for p in 0..P {
            in_ptrs[p] = input.add(*input_offsets.add(p) + tap_offset);
        }
        let mut w = tile_weights.add(tap * in_channels * tile);
        for ci in 0..in_channels {
            let mut wv = [S::splat(0.0); NV];
            for v in 0..NV {
                wv[v] = S::load(w.add(v * S::LANES));
            }
            for p in 0..P {
                let a = S::load_splat(in_ptrs[p].add(ci));
                for v in 0..NV {
                    acc[p][v] = a.mul_add(wv[v], acc[p][v]);
                }
            }
            w = w.add(tile);
        }
    }
    for p in 0..P {
        for v in 0..NV {
            acc[p][v].store(out_ptrs[p].add(v * S::LANES));
        }
    }
}

/// 3x3 conv over a padded channels-last input; `accumulate` adds into the existing output
/// (fused residual connection).
#[inline(always)]
unsafe fn conv3x3_layer<S: Simd, const NV: usize, const P: usize>(
    input: &[f32],
    input_row_pitch: usize,
    table: &ConvPixelTable,
    layer: &PackedConv3x3,
    output: &mut [f32],
    accumulate: bool,
) {
    let cin = layer.in_channels;
    let cout = layer.out_channels;
    debug_assert_eq!(NV * S::LANES, layer.channel_tile);
    debug_assert_eq!(table.input_offsets.len() % P, 0);
    let max_input = table.input_offsets.iter().max().copied().unwrap_or(0)
        + (2 * input_row_pitch + 2) * cin
        + cin;
    assert!(max_input <= input.len(), "conv input buffer too small");
    let max_row = table.output_rows.iter().max().copied().unwrap_or(0);
    assert!(
        (max_row + 1) * cout <= output.len(),
        "conv output buffer too small"
    );

    let mut taps = [0usize; 9];
    for ky in 0..3 {
        for kx in 0..3 {
            taps[ky * 3 + kx] = (ky * input_row_pitch + kx) * cin;
        }
    }
    let tile = layer.channel_tile;
    for (tile_idx, tile_start) in (0..cout).step_by(tile).enumerate() {
        let tile_weights = layer.weights.as_ptr().add(tile_idx * 9 * cin * tile);
        for block in (0..table.input_offsets.len()).step_by(P) {
            conv3x3_block::<S, NV, P>(
                input.as_ptr(),
                table.input_offsets.as_ptr().add(block),
                output.as_mut_ptr(),
                table.output_rows.as_ptr().add(block),
                &taps,
                cin,
                cout,
                tile_start,
                tile_weights,
                layer.bias.as_ptr(),
                accumulate,
            );
        }
    }
}

/// ISA-specific register blocking choice for the conv kernel.
pub(crate) trait ConvKernel: Simd {
    unsafe fn conv3x3(
        input: &[f32],
        input_row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        output: &mut [f32],
        accumulate: bool,
    );
}

#[cfg(target_arch = "aarch64")]
impl ConvKernel for Neon {
    #[inline(always)]
    unsafe fn conv3x3(
        input: &[f32],
        input_row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        output: &mut [f32],
        accumulate: bool,
    ) {
        if layer.channel_tile == CONV_CHANNEL_TILE {
            conv3x3_layer::<Self, 4, NEON_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        } else {
            conv3x3_layer::<Self, 2, NEON_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ConvKernel for Avx2 {
    #[inline(always)]
    unsafe fn conv3x3(
        input: &[f32],
        input_row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        output: &mut [f32],
        accumulate: bool,
    ) {
        if layer.channel_tile == CONV_CHANNEL_TILE {
            conv3x3_layer::<Self, 2, AVX2_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        } else {
            conv3x3_layer::<Self, 1, AVX2_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        }
    }
}

impl ConvKernel for Portable {
    #[inline(always)]
    unsafe fn conv3x3(
        input: &[f32],
        input_row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        output: &mut [f32],
        accumulate: bool,
    ) {
        if layer.channel_tile == CONV_CHANNEL_TILE {
            conv3x3_layer::<Self, 2, PORTABLE_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        } else {
            conv3x3_layer::<Self, 1, PORTABLE_CONV_BLOCK_PIXELS>(
                input,
                input_row_pitch,
                table,
                layer,
                output,
                accumulate,
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GroupNorm + SiLU (channels-last)
// ---------------------------------------------------------------------------------------------

pub(crate) struct GroupNormParams {
    pub channels: usize,
    pub groups: usize,
    pub epsilon: f32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

/// Destination geometry for a normalized image: pixel `(y, x)` goes to
/// `offset + (y * row_pitch + x) * channels`.
#[derive(Clone, Copy)]
pub(crate) struct ImageLayout {
    pub height: usize,
    pub width: usize,
    pub offset: usize,
    pub row_pitch: usize,
}

impl ImageLayout {
    pub(crate) fn dense(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            offset: 0,
            row_pitch: width,
        }
    }
}

/// Max channels handled by the stack buffers below (validated when loading the trunk).
pub(crate) const MAX_NORM_CHANNELS: usize = 256;

/// Per-channel `(scale, shift)` so that `group_norm(x) = x * scale + shift`.
#[inline(always)]
unsafe fn group_norm_scale_shift<S: Simd>(
    src: &[f32],
    pixels: usize,
    gn: &GroupNormParams,
    scale: &mut [f32; MAX_NORM_CHANNELS],
    shift: &mut [f32; MAX_NORM_CHANNELS],
) {
    let c = gn.channels;
    assert!(src.len() >= pixels * c);
    let mut sums = [0.0f32; MAX_NORM_CHANNELS];
    channel_sums::<S>(src, pixels, c, None, &mut sums);
    let group_size = c / gn.groups;
    let count = (pixels * group_size) as f32;
    let mut means = [0.0f32; MAX_NORM_CHANNELS];
    for g in 0..gn.groups {
        let channels = g * group_size..(g + 1) * group_size;
        let mean = sums[channels.clone()].iter().sum::<f32>() / count;
        means[channels].fill(mean);
    }
    let mut sq_sums = [0.0f32; MAX_NORM_CHANNELS];
    channel_sums::<S>(src, pixels, c, Some(&means), &mut sq_sums);
    for g in 0..gn.groups {
        let channels = g * group_size..(g + 1) * group_size;
        let variance = sq_sums[channels.clone()].iter().sum::<f32>() / count;
        let inv_std = 1.0 / (variance + gn.epsilon).sqrt();
        for ch in channels {
            scale[ch] = gn.gamma[ch] * inv_std;
            shift[ch] = gn.beta[ch] - means[ch] * scale[ch];
        }
    }
}

/// Per-channel sums over pixels of `x` (or of `(x - mean)^2` when `means` is given).
#[inline(always)]
unsafe fn channel_sums<S: Simd>(
    src: &[f32],
    pixels: usize,
    channels: usize,
    means: Option<&[f32; MAX_NORM_CHANNELS]>,
    out: &mut [f32; MAX_NORM_CHANNELS],
) {
    for c0 in (0..channels).step_by(S::LANES) {
        let mean = means.map(|m| S::load(m.as_ptr().add(c0)));
        // Two interleaved accumulators break the add dependency chain.
        let mut acc = [S::splat(0.0); 2];
        let mut ptr = src.as_ptr().add(c0);
        for p in 0..pixels {
            let x = S::load(ptr);
            let term = match mean {
                Some(m) => {
                    let d = x.sub(m);
                    d.mul(d)
                }
                None => x,
            };
            acc[p & 1] = acc[p & 1].add(term);
            ptr = ptr.add(channels);
        }
        acc[0].add(acc[1]).store(out.as_mut_ptr().add(c0));
    }
}

/// `dst = silu(group_norm(src))`; `src` is dense `[pixel][channel]`, `dst` follows `layout`
/// (may alias `src` when the layouts match).
#[inline(always)]
pub(crate) unsafe fn group_norm_silu<S: Simd>(
    src: *const f32,
    src_len: usize,
    dst: *mut f32,
    dst_len: usize,
    layout: ImageLayout,
    gn: &GroupNormParams,
) {
    let c = gn.channels;
    let pixels = layout.height * layout.width;
    let src_slice = std::slice::from_raw_parts(src, src_len);
    let mut scale = [0.0f32; MAX_NORM_CHANNELS];
    let mut shift = [0.0f32; MAX_NORM_CHANNELS];
    group_norm_scale_shift::<S>(src_slice, pixels, gn, &mut scale, &mut shift);
    let last_dst = layout.offset + ((layout.height - 1) * layout.row_pitch + layout.width) * c;
    assert!(last_dst <= dst_len, "group norm destination too small");
    for c0 in (0..c).step_by(S::LANES) {
        let sc = S::load(scale.as_ptr().add(c0));
        let sh = S::load(shift.as_ptr().add(c0));
        for y in 0..layout.height {
            let src_row = src.add(y * layout.width * c + c0);
            let dst_row = dst.add(layout.offset + y * layout.row_pitch * c + c0);
            for x in 0..layout.width {
                let v = S::load(src_row.add(x * c)).mul_add(sc, sh);
                silu(v).store(dst_row.add(x * c));
            }
        }
    }
}

/// Copy a dense `[pixel][channel]` image into `layout` (no normalization).
#[inline(always)]
pub(crate) unsafe fn copy_image(
    src: &[f32],
    dst: &mut [f32],
    channels: usize,
    layout: ImageLayout,
) {
    let row = layout.width * channels;
    for y in 0..layout.height {
        let d = layout.offset + y * layout.row_pitch * channels;
        dst[d..d + row].copy_from_slice(&src[y * row..(y + 1) * row]);
    }
}

// ---------------------------------------------------------------------------------------------
// Dense layers
// ---------------------------------------------------------------------------------------------

/// Linear layer packed in column blocks: `[row block][input k][GEMV_ROW_BLOCK rows]`, rows padded
/// to a multiple of [`GEMV_ROW_BLOCK`] with zeros.
pub(crate) struct PackedLinear {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f32>,
    /// Bias padded to the block-rounded row count.
    pub bias: Vec<f32>,
}

impl PackedLinear {
    /// Pack a row-major `[outputs][inputs]` weight; `input_order[k]` names the source column
    /// feeding packed input position `k`.
    pub(crate) fn pack(
        weight_row_major: &[f32],
        bias: &[f32],
        inputs: usize,
        outputs: usize,
        input_order: &[usize],
    ) -> Self {
        assert_eq!(weight_row_major.len(), inputs * outputs);
        assert_eq!(bias.len(), outputs);
        assert_eq!(input_order.len(), inputs);
        let blocks = outputs.div_ceil(GEMV_ROW_BLOCK);
        let mut weights = vec![0.0; blocks * inputs * GEMV_ROW_BLOCK];
        for block in 0..blocks {
            for (k, &src_col) in input_order.iter().enumerate() {
                for lane in 0..GEMV_ROW_BLOCK {
                    let row = block * GEMV_ROW_BLOCK + lane;
                    if row < outputs {
                        weights[(block * inputs + k) * GEMV_ROW_BLOCK + lane] =
                            weight_row_major[row * inputs + src_col];
                    }
                }
            }
        }
        let mut padded_bias = vec![0.0; blocks * GEMV_ROW_BLOCK];
        padded_bias[..outputs].copy_from_slice(bias);
        Self {
            inputs,
            outputs,
            weights,
            bias: padded_bias,
        }
    }

    pub(crate) fn padded_outputs(&self) -> usize {
        self.bias.len()
    }
}

/// `out[..padded_outputs] = W x + b` (padding rows produce 0).
#[inline(always)]
pub(crate) unsafe fn linear<S: Simd>(layer: &PackedLinear, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), layer.inputs);
    assert!(out.len() >= layer.padded_outputs());
    match S::LANES {
        4 => linear_blocks::<S, { GEMV_ROW_BLOCK / 4 }>(layer, x, out),
        8 => linear_blocks::<S, { GEMV_ROW_BLOCK / 8 }>(layer, x, out),
        lanes => unreachable!("unsupported SIMD width {lanes}"),
    }
}

#[inline(always)]
unsafe fn linear_blocks<S: Simd, const NV: usize>(
    layer: &PackedLinear,
    x: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(NV * S::LANES, GEMV_ROW_BLOCK);
    let k_total = layer.inputs;
    for block in 0..layer.padded_outputs() / GEMV_ROW_BLOCK {
        let mut acc = [S::splat(0.0); NV];
        let bias = layer.bias.as_ptr().add(block * GEMV_ROW_BLOCK);
        for v in 0..NV {
            acc[v] = S::load(bias.add(v * S::LANES));
        }
        let mut w = layer.weights.as_ptr().add(block * k_total * GEMV_ROW_BLOCK);
        for k in 0..k_total {
            let a = S::load_splat(x.as_ptr().add(k));
            for v in 0..NV {
                acc[v] = a.mul_add(S::load(w.add(v * S::LANES)), acc[v]);
            }
            w = w.add(GEMV_ROW_BLOCK);
        }
        let dst = out.as_mut_ptr().add(block * GEMV_ROW_BLOCK);
        for v in 0..NV {
            acc[v].store(dst.add(v * S::LANES));
        }
    }
}

/// In-place `x = silu(x)`; `x.len()` must be a multiple of the SIMD width.
#[inline(always)]
pub(crate) unsafe fn silu_in_place<S: Simd>(x: &mut [f32]) {
    assert_eq!(x.len() % S::LANES, 0);
    for chunk in x.chunks_exact_mut(S::LANES) {
        silu(S::load(chunk.as_ptr())).store(chunk.as_mut_ptr());
    }
}

/// In-place `x = silu(layer_norm(x))`; `x.len()` must be a multiple of the SIMD width.
#[inline(always)]
pub(crate) unsafe fn layer_norm_silu_in_place<S: Simd>(
    x: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    let n = x.len();
    assert_eq!(n % S::LANES, 0);
    assert_eq!(gamma.len(), n);
    assert_eq!(beta.len(), n);
    let mut sum = S::splat(0.0);
    for chunk in x.chunks_exact(S::LANES) {
        sum = sum.add(S::load(chunk.as_ptr()));
    }
    let mean = S::splat(sum.reduce_add() / n as f32);
    let mut sq = S::splat(0.0);
    for chunk in x.chunks_exact(S::LANES) {
        let d = S::load(chunk.as_ptr()).sub(mean);
        sq = d.mul_add(d, sq);
    }
    let inv_std = S::splat(1.0 / (sq.reduce_add() / n as f32 + epsilon).sqrt());
    for i in (0..n).step_by(S::LANES) {
        let ptr = x.as_mut_ptr().add(i);
        let normalized = S::load(ptr).sub(mean).mul(inv_std);
        let v = normalized.mul_add(
            S::load(gamma.as_ptr().add(i)),
            S::load(beta.as_ptr().add(i)),
        );
        silu(v).store(ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_values(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f64) / ((1u64 << 24) as f64);
                ((unit * 2.0 - 1.0) as f32) * scale
            })
            .collect()
    }

    unsafe fn exp_vec<S: Simd>(xs: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; xs.len()];
        for (src, dst) in xs
            .chunks_exact(S::LANES)
            .zip(out.chunks_exact_mut(S::LANES))
        {
            exp(S::load(src.as_ptr())).store(dst.as_mut_ptr());
        }
        out
    }

    fn run_exp(isa: Isa, xs: &[f32]) -> Vec<f32> {
        unsafe {
            match isa {
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => exp_vec::<Neon>(xs),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2Fma => exp_avx2(xs),
                Isa::Portable => exp_vec::<Portable>(xs),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_avx2(xs: &[f32]) -> Vec<f32> {
        exp_vec::<Avx2>(xs)
    }

    #[test]
    fn exp_matches_libm_within_two_ulp() {
        let xs: Vec<f32> = (0..16_000).map(|i| -80.0 + i as f32 * 0.01).collect();
        for isa in Isa::available() {
            let got = run_exp(isa, &xs);
            for (&x, &y) in xs.iter().zip(&got) {
                let expected = (x as f64).exp();
                let rel = ((y as f64 - expected) / expected).abs();
                assert!(
                    rel < 2.5e-7,
                    "{isa:?}: exp({x}) = {y}, expected {expected}, rel {rel}"
                );
            }
        }
    }

    #[test]
    fn exp_saturates_without_overflow_or_nan() {
        let xs = [-1000.0f32, -100.0, -88.0, 89.0, 100.0, 1000.0, 0.0, -0.0];
        let padded: Vec<f32> = xs.iter().copied().cycle().take(16).collect();
        for isa in Isa::available() {
            for (&x, &y) in padded.iter().zip(&run_exp(isa, &padded)) {
                assert!(y.is_finite() && y >= 0.0, "{isa:?}: exp({x}) = {y}");
            }
        }
    }

    /// Naive reference: channels-last, zero padding 1, given stride.
    fn reference_conv(
        input: &[f32],
        height: usize,
        width: usize,
        cin: usize,
        weights_oihw: &[f32],
        bias: &[f32],
        cout: usize,
        stride: usize,
    ) -> Vec<f32> {
        let out_h = (height + 2 - 3) / stride + 1;
        let out_w = (width + 2 - 3) / stride + 1;
        let mut out = vec![0.0f32; out_h * out_w * cout];
        for oy in 0..out_h {
            for ox in 0..out_w {
                for co in 0..cout {
                    let mut acc = bias[co] as f64;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = (oy * stride + ky) as isize - 1;
                            let ix = (ox * stride + kx) as isize - 1;
                            if iy < 0 || ix < 0 || iy >= height as isize || ix >= width as isize {
                                continue;
                            }
                            for ci in 0..cin {
                                let x = input[(iy as usize * width + ix as usize) * cin + ci];
                                let w = weights_oihw[((co * cin + ci) * 3 + ky) * 3 + kx];
                                acc += x as f64 * w as f64;
                            }
                        }
                    }
                    out[(oy * out_w + ox) * cout + co] = acc as f32;
                }
            }
        }
        out
    }

    fn pixel_table(
        out_h: usize,
        out_w: usize,
        stride: usize,
        row_pitch: usize,
        cin: usize,
        block: usize,
    ) -> ConvPixelTable {
        let mut input_offsets = Vec::new();
        let mut output_rows = Vec::new();
        for oy in 0..out_h {
            for ox in 0..out_w {
                input_offsets.push((oy * stride * row_pitch + ox * stride) * cin);
                output_rows.push(oy * out_w + ox);
            }
        }
        let real = output_rows.len();
        while input_offsets.len() % block != 0 {
            input_offsets.push(0);
            output_rows.push(real);
        }
        ConvPixelTable {
            input_offsets,
            output_rows,
        }
    }

    unsafe fn conv_with<S: ConvKernel>(
        padded: &[f32],
        row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        out: &mut [f32],
        accumulate: bool,
    ) {
        S::conv3x3(padded, row_pitch, table, layer, out, accumulate)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn conv_avx2(
        padded: &[f32],
        row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        out: &mut [f32],
        accumulate: bool,
    ) {
        conv_with::<Avx2>(padded, row_pitch, table, layer, out, accumulate)
    }

    fn run_conv(
        isa: Isa,
        padded: &[f32],
        row_pitch: usize,
        table: &ConvPixelTable,
        layer: &PackedConv3x3,
        out: &mut [f32],
        accumulate: bool,
    ) {
        unsafe {
            match isa {
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => conv_with::<Neon>(padded, row_pitch, table, layer, out, accumulate),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2Fma => conv_avx2(padded, row_pitch, table, layer, out, accumulate),
                Isa::Portable => {
                    conv_with::<Portable>(padded, row_pitch, table, layer, out, accumulate)
                }
            }
        }
    }

    #[test]
    fn conv3x3_matches_naive_reference() {
        let (height, width) = (20usize, 10usize);
        for &(cin, cout, stride) in &[(32, 32, 1), (32, 32, 2), (16, 8, 1), (8, 16, 2), (5, 16, 1)]
        {
            let input = lcg_values(1 + cin as u64, height * width * cin, 1.5);
            let weights = lcg_values(7 + cout as u64, cout * cin * 9, 0.2);
            let bias = lcg_values(11, cout, 0.5);
            let residual = lcg_values(13, height * width * cout, 1.0);
            let row_pitch = width + 2;
            let mut padded = vec![0.0f32; (height + 2) * row_pitch * cin];
            for y in 0..height {
                for x in 0..width {
                    let dst = ((y + 1) * row_pitch + x + 1) * cin;
                    padded[dst..dst + cin]
                        .copy_from_slice(&input[(y * width + x) * cin..(y * width + x + 1) * cin]);
                }
            }
            let expected =
                reference_conv(&input, height, width, cin, &weights, &bias, cout, stride);
            let out_h = (height - 1) / stride + 1;
            let out_w = (width - 1) / stride + 1;
            let layer = PackedConv3x3::pack(&weights, &bias, cin, cout);
            for isa in Isa::available() {
                let table = pixel_table(
                    out_h,
                    out_w,
                    stride,
                    row_pitch,
                    cin,
                    isa.conv_block_pixels(),
                );
                let rows = table.output_rows.iter().max().unwrap() + 1;
                let mut out = vec![0.0f32; rows * cout];
                run_conv(isa, &padded, row_pitch, &table, &layer, &mut out, false);
                for (i, (&got, &want)) in out.iter().zip(&expected).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "{isa:?} cin={cin} cout={cout} stride={stride} idx={i}: {got} vs {want}"
                    );
                }
                if stride == 1 && cin == cout {
                    let mut acc_out = vec![0.0f32; rows * cout];
                    acc_out[..residual.len()].copy_from_slice(&residual);
                    run_conv(isa, &padded, row_pitch, &table, &layer, &mut acc_out, true);
                    for i in 0..expected.len() {
                        let want = expected[i] + residual[i];
                        assert!(
                            (acc_out[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                            "{isa:?} residual idx={i}: {} vs {want}",
                            acc_out[i]
                        );
                    }
                }
            }
        }
    }

    fn reference_group_norm_silu(x: &[f32], pixels: usize, gn: &GroupNormParams) -> Vec<f32> {
        let c = gn.channels;
        let gs = c / gn.groups;
        let mut out = vec![0.0f32; pixels * c];
        for g in 0..gn.groups {
            let vals: Vec<f64> = (0..pixels)
                .flat_map(|p| (g * gs..(g + 1) * gs).map(move |ch| (p, ch)))
                .map(|(p, ch)| x[p * c + ch] as f64)
                .collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
            for p in 0..pixels {
                for ch in g * gs..(g + 1) * gs {
                    let n = (x[p * c + ch] as f64 - mean) / (var + gn.epsilon as f64).sqrt();
                    let y = n * gn.gamma[ch] as f64 + gn.beta[ch] as f64;
                    out[p * c + ch] = (y / (1.0 + (-y).exp())) as f32;
                }
            }
        }
        out
    }

    unsafe fn gn_with<S: Simd>(
        src: &[f32],
        dst: &mut [f32],
        layout: ImageLayout,
        gn: &GroupNormParams,
    ) {
        group_norm_silu::<S>(
            src.as_ptr(),
            src.len(),
            dst.as_mut_ptr(),
            dst.len(),
            layout,
            gn,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gn_avx2(src: &[f32], dst: &mut [f32], layout: ImageLayout, gn: &GroupNormParams) {
        gn_with::<Avx2>(src, dst, layout, gn)
    }

    #[test]
    fn group_norm_silu_matches_reference_for_padded_destination() {
        let (height, width) = (20usize, 10usize);
        for &(channels, groups) in &[(32usize, 32usize), (32, 8), (16, 16), (8, 1)] {
            let pixels = height * width;
            let x: Vec<f32> = lcg_values(3, pixels * channels, 2.0)
                .iter()
                .enumerate()
                .map(|(i, v)| v + (i % channels) as f32 * 0.3)
                .collect();
            let gn = GroupNormParams {
                channels,
                groups,
                epsilon: 1e-5,
                gamma: lcg_values(5, channels, 1.0),
                beta: lcg_values(6, channels, 0.5),
            };
            let expected = reference_group_norm_silu(&x, pixels, &gn);
            let layout = ImageLayout {
                height,
                width,
                offset: (width + 3) * channels,
                row_pitch: width + 2,
            };
            for isa in Isa::available() {
                let mut dst = vec![0.0f32; (height + 2) * (width + 2) * channels];
                unsafe {
                    match isa {
                        #[cfg(target_arch = "aarch64")]
                        Isa::Neon => gn_with::<Neon>(&x, &mut dst, layout, &gn),
                        #[cfg(target_arch = "x86_64")]
                        Isa::Avx2Fma => gn_avx2(&x, &mut dst, layout, &gn),
                        Isa::Portable => gn_with::<Portable>(&x, &mut dst, layout, &gn),
                    }
                }
                for y in 0..height {
                    for xx in 0..width {
                        for ch in 0..channels {
                            let got = dst[((y + 1) * (width + 2) + xx + 1) * channels + ch];
                            let want = expected[(y * width + xx) * channels + ch];
                            assert!(
                                (got - want).abs() <= 2e-5 * (1.0 + want.abs()),
                                "{isa:?} c={channels} g={groups} ({y},{xx},{ch}): {got} vs {want}"
                            );
                        }
                    }
                }
                let border_sum: f32 = (0..width + 2)
                    .map(|xx| dst[xx * channels..(xx + 1) * channels].iter().sum::<f32>())
                    .sum();
                assert_eq!(border_sum, 0.0, "{isa:?}: padding row must stay zero");
            }
        }
    }

    unsafe fn dense_with<S: Simd>(
        layer: &PackedLinear,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; layer.padded_outputs()];
        linear::<S>(layer, x, &mut out);
        out.truncate(layer.outputs);
        layer_norm_silu_in_place::<S>(&mut out, gamma, beta, 1e-5);
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dense_avx2(layer: &PackedLinear, x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
        dense_with::<Avx2>(layer, x, gamma, beta)
    }

    #[test]
    fn linear_layer_norm_silu_matches_reference() {
        for &(inputs, outputs) in &[(1632usize, 512usize), (19, 32), (512, 256), (7, 40)] {
            let w = lcg_values(21, inputs * outputs, 0.1);
            let b = lcg_values(22, outputs, 0.2);
            let x = lcg_values(23, inputs, 1.0);
            let gamma = lcg_values(24, outputs, 1.0);
            let beta = lcg_values(25, outputs, 0.3);
            // Reverse the input order to exercise the column permutation.
            let order: Vec<usize> = (0..inputs).rev().collect();
            let x_permuted: Vec<f32> = order.iter().map(|&k| x[k]).collect();
            let layer = PackedLinear::pack(&w, &b, inputs, outputs, &order);

            let pre: Vec<f64> = (0..outputs)
                .map(|r| {
                    b[r] as f64
                        + (0..inputs)
                            .map(|k| w[r * inputs + k] as f64 * x[k] as f64)
                            .sum::<f64>()
                })
                .collect();
            let mean = pre.iter().sum::<f64>() / outputs as f64;
            let var = pre.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / outputs as f64;
            let expected: Vec<f64> = pre
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let y = (v - mean) / (var + 1e-5).sqrt() * gamma[i] as f64 + beta[i] as f64;
                    y / (1.0 + (-y).exp())
                })
                .collect();
            for isa in Isa::available() {
                if outputs % 8 != 0 {
                    continue;
                }
                let got = unsafe {
                    match isa {
                        #[cfg(target_arch = "aarch64")]
                        Isa::Neon => dense_with::<Neon>(&layer, &x_permuted, &gamma, &beta),
                        #[cfg(target_arch = "x86_64")]
                        Isa::Avx2Fma => dense_avx2(&layer, &x_permuted, &gamma, &beta),
                        Isa::Portable => dense_with::<Portable>(&layer, &x_permuted, &gamma, &beta),
                    }
                };
                for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        (g as f64 - e).abs() <= 2e-5 * (1.0 + e.abs()),
                        "{isa:?} {inputs}->{outputs} row {i}: {g} vs {e}"
                    );
                }
            }
            // Padding rows of the raw linear output must be exactly zero.
            let mut raw = vec![1.0f32; layer.padded_outputs()];
            unsafe { linear::<Portable>(&layer, &x_permuted, &mut raw) };
            assert!(raw[outputs..].iter().all(|&v| v == 0.0));
        }
    }
}
