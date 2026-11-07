//! Batch-1 kernels for the native heads MLP: matrix-vector products over row-major weights,
//! LayerNorm and SiLU.
//!
//! Matrix-vector products process rows in blocks of `ROW_BLOCK` with explicit SIMD:
//! NEON on aarch64 (always available there), AVX2+FMA on x86_64 when the CPU has it (runtime
//! detection, so the fast path does not depend on `-C target-cpu`), plus a scalar fallback.
//! Every row is reduced with the same instruction sequence wherever it sits in a block, so a
//! row's dot product is bit-identical whether it is computed by the dense or by the gathered
//! (selected-rows) kernel.
//!
//! LayerNorm and SiLU are plain Rust over fixed-width lane arrays; LLVM vectorizes them for
//! every target and, because they use no fused multiply-adds, they give the same bits on
//! every ISA.
//!
//! The kernels evaluate one input vector at a time. Batching several leaves would loop over
//! inputs inside each row block so every weight load is reused (turning GEMV into GEMM).

/// Row blocking of the matrix-vector kernels.
pub(super) const ROW_BLOCK: usize = 4;
/// Row strides and activation buffers are padded (with zeros) to a multiple of this many
/// floats, so the inner loops never need a tail.
pub(super) const K_ALIGN: usize = 16;

pub(super) fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

/// Instruction set used by the matrix-vector kernels, chosen once at model load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Isa {
    /// Portable fallback; on aarch64 only the tests use it.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
}

impl Isa {
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        let isa = Isa::Neon;
        #[cfg(target_arch = "x86_64")]
        let isa = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Isa::Avx2Fma
        } else {
            Isa::Scalar
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let isa = Isa::Scalar;
        isa
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct CacheLine([f32; K_ALIGN]);

/// Zero-initialized f32 buffer whose start is 64-byte aligned.
pub(super) struct AlignedBuf {
    lines: Vec<CacheLine>,
    len: usize,
}

impl AlignedBuf {
    pub(super) fn zeros(len: usize) -> Self {
        Self {
            lines: vec![CacheLine([0.0; K_ALIGN]); len.div_ceil(K_ALIGN)],
            len,
        }
    }

    pub(super) fn as_slice(&self) -> &[f32] {
        // SAFETY: `CacheLine` is `repr(C)` around `[f32; K_ALIGN]`, so the lines form one
        // contiguous, initialized f32 array of at least `len` elements.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast::<f32>(), self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as in `as_slice`, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast::<f32>(), self.len) }
    }
}

/// Row-major weight matrix (PyTorch `[out, in]` layout) with every row zero-padded to a
/// multiple of `K_ALIGN` floats and the row count zero-padded to a multiple of `ROW_BLOCK`.
pub(super) struct PaddedMatrix {
    rows: usize,
    stride: usize,
    data: AlignedBuf,
}

impl PaddedMatrix {
    /// Build from a row-major `[rows, cols]` slice, keeping only columns `col_range`.
    pub(super) fn from_row_major_columns(
        source: &[f32],
        rows: usize,
        source_cols: usize,
        col_range: std::ops::Range<usize>,
    ) -> Self {
        assert_eq!(source.len(), rows * source_cols, "weight shape mismatch");
        assert!(col_range.end <= source_cols, "column range out of bounds");
        let cols = col_range.len();
        let stride = round_up(cols.max(1), K_ALIGN);
        let mut data = AlignedBuf::zeros(round_up(rows, ROW_BLOCK) * stride);
        let dst = data.as_mut_slice();
        for row in 0..rows {
            let src_row = &source[row * source_cols..(row + 1) * source_cols];
            dst[row * stride..row * stride + cols].copy_from_slice(&src_row[col_range.clone()]);
        }
        Self { rows, stride, data }
    }

    pub(super) fn from_row_major(source: &[f32], rows: usize, cols: usize) -> Self {
        Self::from_row_major_columns(source, rows, cols, 0..cols)
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// Row count including the zero rows that pad it to a multiple of `ROW_BLOCK`.
    pub(super) fn padded_rows(&self) -> usize {
        round_up(self.rows, ROW_BLOCK)
    }

    /// Length an input vector must have (zero-padded beyond `cols`).
    pub(super) fn padded_cols(&self) -> usize {
        self.stride
    }

    fn row_ptr(&self, row: usize) -> *const f32 {
        debug_assert!(row < self.padded_rows());
        // SAFETY: `row < padded_rows`, and the buffer holds `padded_rows * stride` floats.
        unsafe { self.data.as_slice().as_ptr().add(row * self.stride) }
    }
}

/// `y[r] = W[r] . x + bias[r]` for every padded row `r`.
///
/// `x` must hold `w.padded_cols()` values (zero beyond the real columns), `bias` and `y` must hold
/// at least `w.padded_rows()` values; padded rows produce `bias[r]` (zero by construction).
pub(super) fn gemv(isa: Isa, w: &PaddedMatrix, bias: &[f32], x: &[f32], y: &mut [f32]) {
    assert!(x.len() >= w.padded_cols(), "gemv input too short");
    assert!(bias.len() >= w.padded_rows(), "gemv bias too short");
    assert!(y.len() >= w.padded_rows(), "gemv output too short");
    // SAFETY: shapes were checked above; `isa` comes from `Isa::detect`.
    unsafe {
        match isa {
            Isa::Scalar => gemv_impl::<ScalarDot>(w, bias, x, y),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => gemv_impl::<NeonDot>(w, bias, x, y),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => gemv_avx2(w, bias, x, y),
        }
    }
}

/// `out[i] = W[rows[i]] . x + bias[rows[i]]` for the selected rows only.
pub(super) fn gemv_rows(
    isa: Isa,
    w: &PaddedMatrix,
    bias: &[f32],
    x: &[f32],
    rows: &[usize],
    out: &mut [f32],
) {
    assert!(x.len() >= w.padded_cols(), "gemv_rows input too short");
    assert!(bias.len() >= w.padded_rows(), "gemv_rows bias too short");
    assert_eq!(out.len(), rows.len(), "gemv_rows output length mismatch");
    if let Some(&bad_row) = rows.iter().find(|&&row| row >= w.rows()) {
        panic!("gemv_rows row {bad_row} out of range for {} rows", w.rows());
    }
    // SAFETY: shapes and row indices were checked above; `isa` comes from `Isa::detect`.
    unsafe {
        match isa {
            Isa::Scalar => gemv_rows_impl::<ScalarDot>(w, bias, x, rows, out),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => gemv_rows_impl::<NeonDot>(w, bias, x, rows, out),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => gemv_rows_avx2(w, bias, x, rows, out),
        }
    }
}

/// Dot products of four rows with `x`, each `k` floats long (`k % K_ALIGN == 0`).
trait Dot4 {
    /// # Safety
    /// Every row pointer and `x` must be valid for reads of `k` floats, and the running CPU
    /// must support the kernel's instruction set.
    unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4];
}

#[inline(always)]
unsafe fn gemv_impl<D: Dot4>(w: &PaddedMatrix, bias: &[f32], x: &[f32], y: &mut [f32]) {
    for block_start in (0..w.padded_rows()).step_by(ROW_BLOCK) {
        let rows = std::array::from_fn(|i| w.row_ptr(block_start + i));
        let dots = D::dot4(rows, x.as_ptr(), w.padded_cols());
        for (i, dot) in dots.into_iter().enumerate() {
            y[block_start + i] = dot + bias[block_start + i];
        }
    }
}

#[inline(always)]
unsafe fn gemv_rows_impl<D: Dot4>(
    w: &PaddedMatrix,
    bias: &[f32],
    x: &[f32],
    rows: &[usize],
    out: &mut [f32],
) {
    for (block_rows, block_out) in rows.chunks(ROW_BLOCK).zip(out.chunks_mut(ROW_BLOCK)) {
        // A short final block repeats its last row; the extra results are discarded.
        let last_row = block_rows[block_rows.len() - 1];
        let ptrs = std::array::from_fn(|i| w.row_ptr(*block_rows.get(i).unwrap_or(&last_row)));
        let dots = D::dot4(ptrs, x.as_ptr(), w.padded_cols());
        for ((slot, &row), dot) in block_out.iter_mut().zip(block_rows).zip(dots) {
            *slot = dot + bias[row];
        }
    }
}

struct ScalarDot;

impl Dot4 for ScalarDot {
    #[inline(always)]
    unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4] {
        let x = std::slice::from_raw_parts(x, k);
        rows.map(|row| {
            let row = std::slice::from_raw_parts(row, k);
            let mut acc = [0.0f32; 8];
            for (w_chunk, x_chunk) in row.chunks_exact(8).zip(x.chunks_exact(8)) {
                for lane in 0..8 {
                    acc[lane] += w_chunk[lane] * x_chunk[lane];
                }
            }
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
        })
    }
}

#[cfg(target_arch = "aarch64")]
struct NeonDot;

#[cfg(target_arch = "aarch64")]
impl Dot4 for NeonDot {
    #[inline(always)]
    unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4] {
        neon::dot4(rows, x, k)
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2Dot;

#[cfg(target_arch = "x86_64")]
impl Dot4 for Avx2Dot {
    #[inline(always)]
    unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4] {
        avx2::dot4(rows, x, k)
    }
}

// The AVX2 drivers carry the target features themselves so the dot kernel inlines into them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_avx2(w: &PaddedMatrix, bias: &[f32], x: &[f32], y: &mut [f32]) {
    gemv_impl::<Avx2Dot>(w, bias, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_rows_avx2(
    w: &PaddedMatrix,
    bias: &[f32],
    x: &[f32],
    rows: &[usize],
    out: &mut [f32],
) {
    gemv_rows_impl::<Avx2Dot>(w, bias, x, rows, out)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::ROW_BLOCK;
    use std::arch::aarch64::*;

    /// 4 rows x 4 accumulators of 4 lanes: 16 independent FMA chains per 16-float step.
    #[inline(always)]
    pub(super) unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4] {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[zero; 4]; ROW_BLOCK];
        let mut i = 0;
        while i < k {
            let xs = [
                vld1q_f32(x.add(i)),
                vld1q_f32(x.add(i + 4)),
                vld1q_f32(x.add(i + 8)),
                vld1q_f32(x.add(i + 12)),
            ];
            for (row_acc, &row) in acc.iter_mut().zip(rows.iter()) {
                for (part, (lane_acc, &xv)) in row_acc.iter_mut().zip(xs.iter()).enumerate() {
                    *lane_acc = vfmaq_f32(*lane_acc, vld1q_f32(row.add(i + 4 * part)), xv);
                }
            }
            i += 16;
        }
        let sums = acc.map(|a| vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
        let reduced = vpaddq_f32(vpaddq_f32(sums[0], sums[1]), vpaddq_f32(sums[2], sums[3]));
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), reduced);
        out
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::ROW_BLOCK;
    use std::arch::x86_64::*;

    /// 4 rows x 2 accumulators of 8 lanes: 8 independent FMA chains per 16-float step.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot4(rows: [*const f32; ROW_BLOCK], x: *const f32, k: usize) -> [f32; 4] {
        let mut acc_lo = [_mm256_setzero_ps(); ROW_BLOCK];
        let mut acc_hi = [_mm256_setzero_ps(); ROW_BLOCK];
        let mut i = 0;
        while i < k {
            let x_lo = _mm256_loadu_ps(x.add(i));
            let x_hi = _mm256_loadu_ps(x.add(i + 8));
            for r in 0..ROW_BLOCK {
                acc_lo[r] = _mm256_fmadd_ps(_mm256_loadu_ps(rows[r].add(i)), x_lo, acc_lo[r]);
                acc_hi[r] = _mm256_fmadd_ps(_mm256_loadu_ps(rows[r].add(i + 8)), x_hi, acc_hi[r]);
            }
            i += 16;
        }
        let s0 = _mm256_add_ps(acc_lo[0], acc_hi[0]);
        let s1 = _mm256_add_ps(acc_lo[1], acc_hi[1]);
        let s2 = _mm256_add_ps(acc_lo[2], acc_hi[2]);
        let s3 = _mm256_add_ps(acc_lo[3], acc_hi[3]);
        // hadd(hadd(s0, s1), hadd(s2, s3)) = [s0, s1, s2, s3] partial sums of the low and
        // high 128-bit halves; adding the halves finishes each row.
        let h = _mm256_hadd_ps(_mm256_hadd_ps(s0, s1), _mm256_hadd_ps(s2, s3));
        let reduced = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h));
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), reduced);
        out
    }
}

const LANES: usize = 8;

#[inline(always)]
fn lane_sum(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

fn sum(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            lanes[lane] += chunk[lane];
        }
    }
    tail.iter().fold(lane_sum(lanes), |acc, &v| acc + v)
}

fn sum_squared_deviation(values: &[f32], mean: f32) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            let d = chunk[lane] - mean;
            lanes[lane] += d * d;
        }
    }
    tail.iter().fold(lane_sum(lanes), |acc, &v| {
        let d = v - mean;
        acc + d * d
    })
}

/// In-place LayerNorm over the whole slice (biased variance, like PyTorch / ONNX).
pub(super) fn layer_norm(values: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    assert!(!values.is_empty(), "layer_norm over an empty vector");
    assert_eq!(values.len(), gamma.len(), "layer_norm gamma width mismatch");
    assert_eq!(values.len(), beta.len(), "layer_norm beta width mismatch");
    let n = values.len() as f32;
    let mean = sum(values) / n;
    let inv_std = 1.0 / (sum_squared_deviation(values, mean) / n + eps).sqrt();
    for ((v, &g), &b) in values.iter_mut().zip(gamma).zip(beta) {
        *v = (*v - mean) * inv_std * g + b;
    }
}

const EXP_CLAMP: f32 = 87.0;
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// Adding 1.5 * 2^23 rounds to the nearest integer and leaves it in the low mantissa bits.
const ROUND_MAGIC: f32 = 12_582_912.0;
const ROUND_MAGIC_BITS: i32 = 0x4B40_0000;
/// ln(2) split into a short exact high part and a correction (Cephes).
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const EXP_P0: f32 = 1.987_569_1e-4;
const EXP_P1: f32 = 1.398_199_9e-3;
const EXP_P2: f32 = 8.333_452e-3;
const EXP_P3: f32 = 4.166_579_6e-2;
const EXP_P4: f32 = 1.666_666_5e-1;
const EXP_P5: f32 = 5.000_000_1e-1;

/// Cephes-style `exp` (max relative error ~2e-7 on the clamped range [-87, 87]).
#[inline(always)]
fn exp_approx(x: f32) -> f32 {
    let x = x.clamp(-EXP_CLAMP, EXP_CLAMP);
    let shifted = x * LOG2_E + ROUND_MAGIC;
    let n = shifted - ROUND_MAGIC;
    let r = x - n * LN2_HI - n * LN2_LO;
    let poly = ((((EXP_P0 * r + EXP_P1) * r + EXP_P2) * r + EXP_P3) * r + EXP_P4) * r + EXP_P5;
    let exp_r = poly * (r * r) + r + 1.0;
    let exponent = shifted.to_bits() as i32 - ROUND_MAGIC_BITS + 127;
    exp_r * f32::from_bits((exponent << 23) as u32)
}

#[inline(always)]
fn silu_one(x: f32) -> f32 {
    x / (1.0 + exp_approx(-x))
}

/// In-place SiLU, `x * sigmoid(x)`.
pub(super) fn silu(values: &mut [f32]) {
    let mut chunks = values.chunks_exact_mut(LANES);
    for chunk in &mut chunks {
        for v in chunk.iter_mut() {
            *v = silu_one(*v);
        }
    }
    for v in chunks.into_remainder() {
        *v = silu_one(*v);
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
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn available_isas() -> Vec<Isa> {
        let mut isas = vec![Isa::Scalar];
        if Isa::detect() != Isa::Scalar {
            isas.push(Isa::detect());
        }
        isas
    }

    fn naive_gemv(w: &[f32], rows: usize, cols: usize, bias: &[f32], x: &[f32]) -> Vec<f64> {
        (0..rows)
            .map(|r| {
                let dot: f64 = (0..cols)
                    .map(|c| f64::from(w[r * cols + c]) * f64::from(x[c]))
                    .sum();
                dot + f64::from(bias[r])
            })
            .collect()
    }

    #[test]
    fn gemv_matches_naive_reference_for_odd_shapes() {
        for &(rows, cols) in &[
            (1, 1),
            (3, 61),
            (4, 16),
            (7, 33),
            (128, 61),
            (130, 256),
            (672, 512),
        ] {
            let weights = lcg_values(rows as u64 * 31 + cols as u64, rows * cols, 0.2);
            let bias = lcg_values(7 + rows as u64, rows, 0.5);
            let x = lcg_values(11 + cols as u64, cols, 1.0);
            let matrix = PaddedMatrix::from_row_major(&weights, rows, cols);
            let mut x_padded = vec![0.0; matrix.padded_cols()];
            x_padded[..cols].copy_from_slice(&x);
            let mut bias_padded = vec![0.0; matrix.padded_rows()];
            bias_padded[..rows].copy_from_slice(&bias);
            let expected = naive_gemv(&weights, rows, cols, &bias, &x);
            for isa in available_isas() {
                let mut y = vec![f32::NAN; matrix.padded_rows()];
                gemv(isa, &matrix, &bias_padded, &x_padded, &mut y);
                for r in 0..rows {
                    let tolerance = 1e-5 * (1.0 + expected[r].abs());
                    assert!(
                        (f64::from(y[r]) - expected[r]).abs() < tolerance,
                        "{isa:?} {rows}x{cols} row {r}: {} vs {}",
                        y[r],
                        expected[r]
                    );
                }
                assert!(
                    y[rows..].iter().all(|&v| v == 0.0),
                    "padded rows must be zero"
                );
            }
        }
    }

    #[test]
    fn gemv_rows_is_bit_identical_to_dense_gemv() {
        let (rows, cols) = (37, 72);
        let weights = lcg_values(3, rows * cols, 0.3);
        let matrix = PaddedMatrix::from_row_major(&weights, rows, cols);
        let mut bias = lcg_values(5, matrix.padded_rows(), 0.1);
        bias[rows..].fill(0.0);
        let mut x = vec![0.0; matrix.padded_cols()];
        x[..cols].copy_from_slice(&lcg_values(9, cols, 1.0));
        for isa in available_isas() {
            let mut dense = vec![0.0; matrix.padded_rows()];
            gemv(isa, &matrix, &bias, &x, &mut dense);
            for selection in [
                vec![5],
                vec![36, 0, 17],
                vec![1, 2, 3, 4, 5, 30, 31],
                vec![9; 6],
            ] {
                let mut gathered = vec![0.0; selection.len()];
                gemv_rows(isa, &matrix, &bias, &x, &selection, &mut gathered);
                for (&row, &value) in selection.iter().zip(&gathered) {
                    assert_eq!(value.to_bits(), dense[row].to_bits(), "{isa:?} row {row}");
                }
            }
        }
    }

    #[test]
    fn column_slicing_keeps_requested_columns() {
        let weights: Vec<f32> = (0..12).map(|v| v as f32).collect(); // 3 x 4
        let right = PaddedMatrix::from_row_major_columns(&weights, 3, 4, 2..4);
        assert_eq!(right.padded_cols(), K_ALIGN);
        let mut x = vec![0.0; right.padded_cols()];
        x[0] = 1.0;
        x[1] = 10.0;
        let mut y = vec![0.0; right.padded_rows()];
        gemv(
            Isa::detect(),
            &right,
            &vec![0.0; right.padded_rows()],
            &x,
            &mut y,
        );
        assert_eq!(&y[..3], &[2.0 + 30.0, 6.0 + 70.0, 10.0 + 110.0]);
    }

    #[test]
    fn exp_approx_is_accurate() {
        let mut max_rel = 0.0f64;
        for i in -8700..=8700 {
            let x = i as f32 * 0.01;
            let expected = f64::from(x).exp();
            let rel = ((f64::from(exp_approx(x)) - expected) / expected).abs();
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 5e-7, "max relative error {max_rel}");
    }

    #[test]
    fn silu_matches_reference_including_extremes() {
        let inputs = [
            -1000.0f32, -100.0, -87.5, -20.0, -3.0, -1e-3, 0.0, 1e-3, 0.5, 4.0, 30.0, 100.0, 1e4,
        ];
        let mut values = inputs.to_vec();
        silu(&mut values);
        for (&x, &y) in inputs.iter().zip(&values) {
            let expected = f64::from(x) / (1.0 + (-f64::from(x)).exp());
            assert!(y.is_finite(), "silu({x}) = {y}");
            assert!(
                (f64::from(y) - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
                "silu({x}) = {y}, expected {expected}"
            );
        }
    }

    #[test]
    fn layer_norm_matches_reference() {
        for &n in &[1usize, 5, 64, 129, 256] {
            let values = lcg_values(n as u64, n, 3.0)
                .iter()
                .map(|v| v + 2.0)
                .collect::<Vec<_>>();
            let gamma = lcg_values(100 + n as u64, n, 1.0);
            let beta = lcg_values(200 + n as u64, n, 1.0);
            let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
            let var = values
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let mut normalized = values.clone();
            layer_norm(&mut normalized, &gamma, &beta, 1e-5);
            for i in 0..n {
                let expected = (f64::from(values[i]) - mean) / (var + 1e-5).sqrt()
                    * f64::from(gamma[i])
                    + f64::from(beta[i]);
                assert!(
                    (f64::from(normalized[i]) - expected).abs() < 1e-5,
                    "n={n} i={i}: {} vs {expected}",
                    normalized[i]
                );
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    fn time_gemv(rows: usize, cols: usize, copies: usize) -> (f64, f64) {
        // `copies` distinct matrices cycled to control the working-set size.
        let mats: Vec<PaddedMatrix> = (0..copies)
            .map(|c| {
                let data: Vec<f32> = (0..rows * cols)
                    .map(|i| ((i + c) % 13) as f32 * 0.01)
                    .collect();
                PaddedMatrix::from_row_major(&data, rows, cols)
            })
            .collect();
        let bias = vec![0.0; round_up(rows, ROW_BLOCK)];
        let x: Vec<f32> = (0..round_up(cols, K_ALIGN))
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let mut y = vec![0.0; round_up(rows, ROW_BLOCK)];
        let isa = Isa::detect();
        let iters = (200_000_000 / (rows * cols)).max(10);
        let start = Instant::now();
        let mut acc = 0.0;
        for i in 0..iters {
            gemv(isa, &mats[i % copies], &bias, &x, &mut y);
            acc += y[0];
        }
        let secs = start.elapsed().as_secs_f64();
        assert!(acc.is_finite());
        let per_call_us = secs * 1e6 / iters as f64;
        let gmacs = (rows * cols * iters) as f64 / secs / 1e9;
        (per_call_us, gmacs)
    }

    #[test]
    #[ignore]
    fn bench_kernels() {
        for &(rows, cols, copies) in &[
            (64, 256, 1),
            (128, 128, 1),
            (256, 256, 1),
            (512, 256, 1),
            (512, 256, 6),
            (256, 256, 24),
        ] {
            let (us, gmacs) = time_gemv(rows, cols, copies);
            let mb = (rows * cols * copies * 4) as f64 / 1e6;
            eprintln!(
                "gemv {rows}x{cols} working set {mb:.2} MB: {us:.3} us/call, {gmacs:.1} GMAC/s"
            );
        }
        let mut values: Vec<f32> = (0..1664).map(|i| (i as f32 - 800.0) * 0.01).collect();
        let iters = 200_000;
        let start = Instant::now();
        for _ in 0..iters {
            silu(&mut values);
            values[0] += 1e-3;
        }
        eprintln!(
            "silu x1664: {:.3} us/call",
            start.elapsed().as_secs_f64() * 1e6 / iters as f64
        );
        let gamma = vec![1.0; 256];
        let beta = vec![0.0; 256];
        let mut values: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let start = Instant::now();
        for _ in 0..iters {
            layer_norm(&mut values, &gamma, &beta, 1e-5);
            values[0] += 1e-3;
        }
        eprintln!(
            "layer_norm x256: {:.3} us/call",
            start.elapsed().as_secs_f64() * 1e6 / iters as f64
        );
    }
}
