//! MCTS Utility Functions

use rand_distr::{Distribution, Gamma};

use crate::game::constants::{
    BUMPINESS_NORMALIZATION_DIVISOR, COLUMN_HEIGHT_NORMALIZATION_DIVISOR,
    HOLES_NORMALIZATION_DIVISOR, MAX_BOARD_CELLS, MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR,
    OVERHANG_NORMALIZATION_DIVISOR, ROW_FILL_FEATURE_ROWS, TOTAL_BLOCKS_NORMALIZATION_DIVISOR,
};
use crate::game::env::global_cache::{
    build_board_key, get_cached_board_analysis, insert_cached_board_analysis,
};
use crate::game::env::TetrisEnv;

/// Sample from Dirichlet distribution.
///
/// # Arguments
/// * `alpha` - Concentration parameter (must be > 0)
/// * `n` - Number of samples
///
/// # Panics
/// Panics if alpha <= 0
pub fn sample_dirichlet<R: rand::Rng + ?Sized>(alpha: f32, n: usize, rng: &mut R) -> Vec<f32> {
    debug_assert!(
        alpha > 0.0,
        "Dirichlet alpha must be positive, got {}",
        alpha
    );
    debug_assert!(n > 0, "Dirichlet n must be positive");

    // Gamma::new only fails if alpha <= 0 or scale <= 0
    let gamma =
        Gamma::new(alpha as f64, 1.0).expect("Invalid Dirichlet alpha parameter (must be > 0)");

    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f64 = samples.iter().sum();

    if sum == 0.0 {
        // Fallback to uniform if all samples are zero (shouldn't happen with valid alpha)
        debug_assert!(
            false,
            "Dirichlet sampling produced all zeros with alpha={} n={} - falling back to uniform",
            alpha, n
        );
        return vec![1.0 / n as f32; n];
    }

    samples.into_iter().map(|x| (x / sum) as f32).collect()
}

/// Compute proportional death penalty based on how early the game ended.
/// Dying early (low move_number) incurs a larger penalty than dying late.
/// Returns 0.0 if death_penalty is 0.0 or move_number >= max_placements.
pub fn compute_death_penalty(move_number: u32, max_placements: u32, death_penalty: f32) -> f32 {
    if death_penalty == 0.0 {
        return 0.0;
    }
    let remaining = (max_placements as f32 - move_number as f32) / max_placements as f32;
    death_penalty * remaining.max(0.0)
}

/// Count overhang fields and holes in one pass.
///
/// Overhang fields are empty cells with at least one filled cell above in the same column.
/// Holes are overhang fields not reachable from top-row air via 4-neighbor flood-fill.
pub fn count_overhang_fields_and_holes(env: &TetrisEnv) -> (u32, u32) {
    if let Some(cached) = env.get_cached_overhang_fields_and_holes() {
        return cached;
    }

    let board_key = build_board_key(env);
    if let Some(key) = board_key {
        if let Some(cached) = get_cached_board_analysis(key) {
            env.set_cached_overhang_fields_and_holes(cached);
            return cached;
        }
    }

    let (overhang_fields, holes) =
        analyze_overhangs_and_holes(env.board_cells(), env.width, env.height);

    let result = (overhang_fields, holes);
    env.set_cached_overhang_fields_and_holes(result);
    if let Some(key) = board_key {
        insert_cached_board_analysis(key, result);
    }
    result
}

/// Bit-parallel overhang/hole analysis over per-row empty-cell masks.
///
/// Overhang fields are empty cells below the first filled cell of their column; holes are
/// the overhang fields outside the 4-connected empty region seeded by the top row.
fn analyze_overhangs_and_holes(cells: &[u8], width: usize, height: usize) -> (u32, u32) {
    assert!(
        width <= u32::BITS as usize && height <= MAX_BOARD_CELLS,
        "board analysis supports up to {} columns and {} rows, got {}x{}",
        u32::BITS,
        MAX_BOARD_CELLS,
        width,
        height
    );
    let full_row = u32::MAX >> (u32::BITS as usize - width.max(1));

    // empty[y]: bit x set iff cell (x, y) is empty; row 0 is the top row.
    let mut empty = [0u32; MAX_BOARD_CELLS];
    for (row_mask, row) in empty.iter_mut().zip(cells.chunks_exact(width.max(1))) {
        *row_mask = row
            .iter()
            .enumerate()
            .filter(|(_, &cell)| cell == 0)
            .fold(0u32, |mask, (x, _)| mask | (1 << x));
    }
    let empty = &empty[..height];

    // Flood-fill the air reachable from the top row, alternating downward and upward
    // sweeps (each row is filled horizontally to a fixpoint) until nothing changes.
    let mut reach = [0u32; MAX_BOARD_CELLS];
    let reach = &mut reach[..height];
    if height > 0 {
        reach[0] = empty[0];
    }
    let fill_row = |reach: &mut [u32], y: usize| -> bool {
        let above = if y > 0 { reach[y - 1] } else { 0 };
        let below = if y + 1 < height { reach[y + 1] } else { 0 };
        let mut row = (reach[y] | above | below) & empty[y];
        loop {
            let spread = (row | (row << 1) | (row >> 1)) & empty[y];
            if spread == row {
                break;
            }
            row = spread;
        }
        let changed = row != reach[y];
        reach[y] = row;
        changed
    };
    loop {
        let mut changed = false;
        for y in 0..height {
            changed |= fill_row(reach, y);
        }
        for y in (0..height).rev() {
            changed |= fill_row(reach, y);
        }
        if !changed {
            break;
        }
    }

    let mut overhang_fields = 0u32;
    let mut holes = 0u32;
    let mut covered = 0u32;
    for (&empty_row, &reach_row) in empty.iter().zip(reach.iter()) {
        let overhang_row = empty_row & covered;
        overhang_fields += overhang_row.count_ones();
        holes += (overhang_row & !reach_row).count_ones();
        covered |= !empty_row & full_row;
    }
    (overhang_fields, holes)
}

/// Count overhang fields (empty cells with at least one filled cell above in the same column).
pub fn count_overhang_fields(env: &TetrisEnv) -> u32 {
    count_overhang_fields_and_holes(env).0
}

pub fn normalize_overhang_fields(overhang_fields: u32) -> f32 {
    overhang_fields as f32 / OVERHANG_NORMALIZATION_DIVISOR
}

/// Compute normalized overhang penalty magnitude from raw overhang count.
pub fn compute_overhang_penalty(overhang_fields: u32, overhang_penalty_weight: f32) -> f32 {
    if overhang_penalty_weight == 0.0 {
        return 0.0;
    }
    normalize_overhang_fields(overhang_fields) * overhang_penalty_weight
}

/// Compute terrain bumpiness as the sum of squared adjacent column-height deltas.
/// For heights h[0..W-1], bumpiness = Σ_{i=0..W-2} (h[i] - h[i+1])^2.
pub fn compute_bumpiness(column_heights: &[u8]) -> u32 {
    if column_heights.len() < 2 {
        return 0;
    }

    let mut bumpiness: u32 = 0;
    for i in 0..(column_heights.len() - 1) {
        let delta = column_heights[i] as i32 - column_heights[i + 1] as i32;
        bumpiness += (delta * delta) as u32;
    }
    bumpiness
}

pub fn normalize_column_heights(column_heights: &[u8]) -> Vec<f32> {
    column_heights
        .iter()
        .map(|&height| height as f32 / COLUMN_HEIGHT_NORMALIZATION_DIVISOR)
        .collect()
}

pub fn normalize_max_column_height(raw: u8) -> f32 {
    raw as f32 / MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR
}

pub fn normalize_row_fill_counts(row_fill_counts: &[u8], board_width: usize) -> Vec<f32> {
    debug_assert!(board_width > 0, "board_width must be > 0");
    let denominator = board_width as f32;
    let start = row_fill_counts
        .len()
        .checked_sub(ROW_FILL_FEATURE_ROWS)
        .expect("row_fill_counts must include at least ROW_FILL_FEATURE_ROWS rows");
    let selected = &row_fill_counts[start..];

    let mut normalized = vec![0.0; ROW_FILL_FEATURE_ROWS];
    for (i, &count) in selected.iter().enumerate() {
        normalized[i] = count as f32 / denominator;
    }
    normalized
}

pub fn normalize_total_blocks(total_blocks: u32) -> f32 {
    total_blocks as f32 / TOTAL_BLOCKS_NORMALIZATION_DIVISOR
}

pub fn normalize_bumpiness(raw_bumpiness: u32) -> f32 {
    raw_bumpiness as f32 / BUMPINESS_NORMALIZATION_DIVISOR
}

pub fn normalize_holes(holes: u32) -> f32 {
    holes as f32 / HOLES_NORMALIZATION_DIVISOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    /// Straightforward queue flood fill, kept as the oracle for the bit-parallel version.
    fn reference_overhangs_and_holes(cells: &[u8], width: usize, height: usize) -> (u32, u32) {
        let mut reachable = vec![false; width * height];
        let mut queue: Vec<usize> = (0..width).filter(|&x| cells[x] == 0).collect();
        for &idx in &queue {
            reachable[idx] = true;
        }
        let mut head = 0;
        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            let (x, y) = (idx % width, idx / width);
            let mut neighbors = Vec::with_capacity(4);
            if y > 0 {
                neighbors.push(idx - width);
            }
            if y + 1 < height {
                neighbors.push(idx + width);
            }
            if x > 0 {
                neighbors.push(idx - 1);
            }
            if x + 1 < width {
                neighbors.push(idx + 1);
            }
            for next in neighbors {
                if cells[next] == 0 && !reachable[next] {
                    reachable[next] = true;
                    queue.push(next);
                }
            }
        }
        let (mut overhang_fields, mut holes) = (0, 0);
        for x in 0..width {
            let mut seen_filled = false;
            for y in 0..height {
                let idx = y * width + x;
                if cells[idx] != 0 {
                    seen_filled = true;
                } else if seen_filled {
                    overhang_fields += 1;
                    if !reachable[idx] {
                        holes += 1;
                    }
                }
            }
        }
        (overhang_fields, holes)
    }

    #[test]
    fn test_analyze_overhangs_and_holes_matches_flood_fill_reference() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0BAD_5EED);
        for (width, height) in [(10, 20), (3, 4), (7, 9), (1, 12), (10, 1)] {
            for _ in 0..4000 {
                let fill_probability = rng.gen_range(0.05..0.95);
                let cells: Vec<u8> = (0..width * height)
                    .map(|_| u8::from(rng.gen_bool(fill_probability)))
                    .collect();
                assert_eq!(
                    analyze_overhangs_and_holes(&cells, width, height),
                    reference_overhangs_and_holes(&cells, width, height),
                    "{width}x{height} board {cells:?}"
                );
            }
        }
    }

    #[test]
    fn test_dirichlet() {
        let mut rng = thread_rng();
        let noise = sample_dirichlet(0.15, 10, &mut rng);
        assert_eq!(noise.len(), 10);

        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_count_overhang_fields_empty_board() {
        let mut env = TetrisEnv::with_seed(10, 20, 1);
        env.set_board(vec![vec![0; 10]; 20])
            .expect("set_board should succeed");
        assert_eq!(count_overhang_fields(&env), 0);
    }

    #[test]
    fn test_count_overhang_fields_counts_holes_below_filled_cells() {
        let mut env = TetrisEnv::with_seed(10, 20, 2);
        let mut board = vec![vec![0; 10]; 20];
        board[0][0] = 1;
        board[2][0] = 1;
        board[5][1] = 1;
        env.set_board(board).expect("set_board should succeed");

        // Column 0: y=1,3..19 are overhangs => 18
        // Column 1: y=6..19 are overhangs => 14
        assert_eq!(count_overhang_fields(&env), 32);
    }

    #[test]
    fn test_count_holes_ignores_reachable_overhang_air() {
        let mut env = TetrisEnv::with_seed(3, 4, 3);
        let board = vec![vec![0, 1, 0], vec![0, 0, 0], vec![1, 0, 1], vec![1, 1, 1]];
        env.set_board(board).expect("set_board should succeed");

        let (overhang, holes) = count_overhang_fields_and_holes(&env);
        assert_eq!(overhang, 2);
        assert_eq!(holes, 0);
    }

    #[test]
    fn test_count_holes_counts_sealed_overhang_cells() {
        let mut env = TetrisEnv::with_seed(3, 4, 4);
        let board = vec![vec![0, 1, 0], vec![1, 0, 1], vec![1, 0, 1], vec![1, 1, 1]];
        env.set_board(board).expect("set_board should succeed");

        let (overhang, holes) = count_overhang_fields_and_holes(&env);
        assert_eq!(overhang, 2);
        assert_eq!(holes, 2);
    }

    #[test]
    fn test_compute_overhang_penalty_uses_fixed_normalization() {
        let penalty = compute_overhang_penalty(25, 2.0);
        // 25 / 25.0 = 1.0, * 2.0 = 2.0
        assert!((penalty - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_bumpiness_squared_adjacent_deltas() {
        let column_heights = vec![0, 2, 5, 1];
        // (0-2)^2 + (2-5)^2 + (5-1)^2 = 4 + 9 + 16 = 29
        assert_eq!(compute_bumpiness(&column_heights), 29);
    }

    #[test]
    fn test_compute_bumpiness_short_inputs_are_zero() {
        assert_eq!(compute_bumpiness(&[]), 0);
        assert_eq!(compute_bumpiness(&[3]), 0);
    }

    #[test]
    fn test_normalize_total_blocks_uses_empirical_divisor() {
        // divisor = 60.0
        let normalized = normalize_total_blocks(30);
        assert!((normalized - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_column_heights_uses_empirical_divisor() {
        // divisor = 8.0
        let normalized = normalize_column_heights(&[0, 4, 8]);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_normalize_max_column_height() {
        // divisor = 20.0
        assert!((normalize_max_column_height(10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_row_fill_counts_keeps_bottom_rows_only() {
        let mut row_fills = [0u8; 20];
        row_fills[16] = 2;
        row_fills[17] = 4;
        row_fills[18] = 6;
        row_fills[19] = 8;
        let normalized = normalize_row_fill_counts(&row_fills, 10);
        assert_eq!(normalized, vec![0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_normalize_row_fill_counts_keeps_bottom_rows_for_taller_input() {
        let mut row_fills = [0u8; 24];
        row_fills[20] = 1;
        row_fills[21] = 3;
        row_fills[22] = 5;
        row_fills[23] = 7;
        let normalized = normalize_row_fill_counts(&row_fills, 10);
        assert_eq!(normalized, vec![0.1, 0.3, 0.5, 0.7]);
    }

    #[test]
    #[should_panic(expected = "row_fill_counts must include at least ROW_FILL_FEATURE_ROWS rows")]
    fn test_normalize_row_fill_counts_panics_when_input_too_short() {
        let _ = normalize_row_fill_counts(&[1, 2, 3], 10);
    }

    #[test]
    fn test_normalize_bumpiness_uses_empirical_divisor() {
        // divisor = 200.0
        let normalized = normalize_bumpiness(100);
        assert!((normalized - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_holes_uses_empirical_divisor() {
        // divisor = 20.0
        let normalized = normalize_holes(10);
        assert!((normalized - 0.5).abs() < 1e-6);
    }
}
