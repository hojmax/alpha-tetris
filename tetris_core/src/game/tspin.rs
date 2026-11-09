//! T-spin classification, following Jstris (`GameCore.checkTSpin`).
//!
//! A T-spin needs the spin flag (a rotation since the piece last moved down) and the
//! 3-corner rule on the T's center. Jstris has no immobility test and no kick-based
//! upgrade, so the corners alone decide full vs mini:
//! - full: both front corners (the side the T points to) plus at least one back corner
//! - mini: one front corner plus both back corners
//!
//! Walls and the floor count as filled. Cells above the board count as empty, like the
//! empty buffer row above Jstris's visible field.

use crate::game::constants::T_PIECE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TSpin {
    None,
    Mini,
    Full,
}

impl TSpin {
    #[inline]
    pub fn is_spin(self) -> bool {
        self != TSpin::None
    }
}

/// Corner offsets from the T center, as `(front, back)` pairs per rotation state.
const CORNERS: [[[(i32, i32); 2]; 2]; 4] = [
    [[(-1, -1), (1, -1)], [(-1, 1), (1, 1)]], // 0: points up
    [[(1, -1), (1, 1)], [(-1, -1), (-1, 1)]], // R: points right
    [[(-1, 1), (1, 1)], [(-1, -1), (1, -1)]], // 2: points down
    [[(-1, -1), (-1, 1)], [(1, -1), (1, 1)]], // L: points left
];

#[inline]
fn corner_filled(cells: &[u8], width: usize, height: usize, x: i32, y: i32) -> bool {
    if x < 0 || x >= width as i32 || y >= height as i32 {
        return true;
    }
    y >= 0 && cells[y as usize * width + x as usize] != 0
}

/// Classify a piece locking at `(x, y, rotation)` on `cells` (row-major, `width x height`).
/// `spin_flag` is whether a rotation happened since the piece last moved down.
pub fn classify_tspin(
    cells: &[u8],
    width: usize,
    height: usize,
    piece_type: usize,
    x: i32,
    y: i32,
    rotation: usize,
    spin_flag: bool,
) -> TSpin {
    if piece_type != T_PIECE || !spin_flag {
        return TSpin::None;
    }
    // The T occupies a 3x3 box whose center is at (x + 1, y + 1).
    let (center_x, center_y) = (x + 1, y + 1);
    let [front, back] = &CORNERS[rotation];
    let count = |corners: &[(i32, i32); 2]| {
        corners
            .iter()
            .filter(|&&(dx, dy)| corner_filled(cells, width, height, center_x + dx, center_y + dy))
            .count()
    };
    match (count(front), count(back)) {
        (2, 1..) => TSpin::Full,
        (1, 2) => TSpin::Mini,
        _ => TSpin::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 10;
    const H: usize = 20;

    fn board_with(filled: &[(usize, usize)]) -> Vec<u8> {
        let mut cells = vec![0u8; W * H];
        for &(x, y) in filled {
            cells[y * W + x] = 1;
        }
        cells
    }

    fn classify(cells: &[u8], x: i32, y: i32, rotation: usize) -> TSpin {
        classify_tspin(cells, W, H, T_PIECE, x, y, rotation, true)
    }

    #[test]
    fn test_requires_t_piece_and_spin_flag() {
        // T pointing down into a slot, center (4, 18): front = (3, 19), (5, 19).
        let cells = board_with(&[(3, 19), (5, 19), (3, 17)]);
        assert_eq!(classify(&cells, 3, 17, 2), TSpin::Full);
        assert_eq!(
            classify_tspin(&cells, W, H, T_PIECE, 3, 17, 2, false),
            TSpin::None
        );
        assert_eq!(classify_tspin(&cells, W, H, 5, 3, 17, 2, true), TSpin::None);
    }

    #[test]
    fn test_full_needs_both_front_corners() {
        // Pointing up on the floor, center (4, 19): back corners are the floor.
        let both_front = board_with(&[(3, 18), (5, 18)]);
        assert_eq!(classify(&both_front, 3, 18, 0), TSpin::Full);
        let one_front = board_with(&[(3, 18)]);
        assert_eq!(classify(&one_front, 3, 18, 0), TSpin::Mini);
        assert_eq!(classify(&board_with(&[]), 3, 18, 0), TSpin::None);
    }

    #[test]
    fn test_mini_needs_both_back_corners() {
        // Pointing right, center (1, 10): front = right corners, back = left corners.
        let cells = board_with(&[(2, 9), (0, 9)]);
        assert_eq!(classify(&cells, 0, 9, 1), TSpin::None);
        let cells = board_with(&[(2, 9), (0, 9), (0, 11)]);
        assert_eq!(classify(&cells, 0, 9, 1), TSpin::Mini);
    }

    #[test]
    fn test_walls_fill_corners() {
        // Pointing right against the left wall, center (0, 10): both back corners are wall.
        assert_eq!(classify(&board_with(&[(1, 11)]), -1, 9, 1), TSpin::Mini);
        assert_eq!(
            classify(&board_with(&[(1, 9), (1, 11)]), -1, 9, 1),
            TSpin::Full
        );
        // Pointing left against the right wall, center (9, 10): back corners are wall.
        assert_eq!(classify(&board_with(&[(8, 9)]), 8, 9, 3), TSpin::Mini);
    }

    #[test]
    fn test_cells_above_board_are_empty() {
        // Pointing down with the center on row 0: the back corners sit above the board,
        // so two filled front corners alone are not a T-spin.
        let cells = board_with(&[(3, 1), (5, 1)]);
        assert_eq!(classify(&cells, 3, -1, 2), TSpin::None);
    }
}
