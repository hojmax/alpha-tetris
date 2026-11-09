//! Wall kick data, matching Jstris's SRS block set.
//!
//! The 90° kicks are standard SRS. Jstris also has a 180° rotation whose kick list is the
//! in-place rotation plus a single one-cell shift.
//!
//! Rotation states:
//! - 0: Spawn state
//! - 1: R (clockwise from spawn)
//! - 2: 180° (two rotations from spawn)
//! - 3: L (counter-clockwise from spawn)
//!
//! Offsets are `(dx, dy)` with y pointing down (positive = down), tried in order.

use crate::game::constants::{I_PIECE, O_PIECE};

/// A rotation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    Clockwise,
    CounterClockwise,
    Half,
}

impl Rotation {
    pub const ALL: [Rotation; 3] = [
        Rotation::Clockwise,
        Rotation::CounterClockwise,
        Rotation::Half,
    ];

    /// Rotation state reached by applying this rotation to `from_state`.
    #[inline]
    pub fn target(self, from_state: usize) -> usize {
        match self {
            Rotation::Clockwise => (from_state + 1) % 4,
            Rotation::CounterClockwise => (from_state + 3) % 4,
            Rotation::Half => (from_state + 2) % 4,
        }
    }
}

type KickList5 = [(i32, i32); 5];

/// J, L, S, T, Z clockwise kicks, indexed by the starting rotation state.
const JLSTZ_CLOCKWISE: [KickList5; 4] = [
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],  // 0 -> R
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],    // R -> 2
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],     // 2 -> L
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)], // L -> 0
];

/// J, L, S, T, Z counter-clockwise kicks, indexed by the starting rotation state.
const JLSTZ_COUNTER_CLOCKWISE: [KickList5; 4] = [
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],     // 0 -> L
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],    // R -> 0
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],  // 2 -> R
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)], // L -> 2
];

/// I clockwise kicks, indexed by the starting rotation state.
const I_CLOCKWISE: [KickList5; 4] = [
    [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)], // 0 -> R
    [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)], // R -> 2
    [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)], // 2 -> L
    [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)], // L -> 0
];

/// I counter-clockwise kicks, indexed by the starting rotation state.
const I_COUNTER_CLOCKWISE: [KickList5; 4] = [
    [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)], // 0 -> L
    [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)], // R -> 0
    [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)], // 2 -> R
    [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)], // L -> 2
];

/// Jstris 180° kicks for every rotating piece (I included), indexed by the starting state.
const HALF_TURN: [[(i32, i32); 2]; 4] = [
    [(0, 0), (0, -1)], // 0 -> 2: up one
    [(0, 0), (1, 0)],  // R -> L: right one
    [(0, 0), (0, 1)],  // 2 -> 0: down one
    [(0, 0), (-1, 0)], // L -> R: left one
];

const NO_KICK: [(i32, i32); 1] = [(0, 0)];

/// Kick offsets to try, in order, when rotating `piece_type` from `from_state`.
/// piece_type: 0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L
pub fn get_kicks(
    piece_type: usize,
    from_state: usize,
    rotation: Rotation,
) -> &'static [(i32, i32)] {
    assert!(
        from_state < 4,
        "Invalid rotation state for kicks: {from_state}. Expected 0..=3."
    );
    match (piece_type, rotation) {
        (O_PIECE, _) => &NO_KICK,
        (0..=6, Rotation::Half) => &HALF_TURN[from_state],
        (I_PIECE, Rotation::Clockwise) => &I_CLOCKWISE[from_state],
        (I_PIECE, Rotation::CounterClockwise) => &I_COUNTER_CLOCKWISE[from_state],
        (2..=6, Rotation::Clockwise) => &JLSTZ_CLOCKWISE[from_state],
        (2..=6, Rotation::CounterClockwise) => &JLSTZ_COUNTER_CLOCKWISE[from_state],
        _ => panic!("Invalid piece_type for kicks: {piece_type}. Expected 0..=6."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type JstrisKicks = [[&'static [(i32, i32)]; 3]; 4];

    /// Jstris's own SRS kick tables (y pointing up), indexed `[from_state][cw, ccw, 180]`,
    /// copied from its client's `getBlockSets`.
    const JSTRIS_JLSTZ: JstrisKicks = [
        [
            &[(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            &[(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            &[(0, 0), (0, 1)],
        ],
        [
            &[(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            &[(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            &[(0, 0), (1, 0)],
        ],
        [
            &[(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            &[(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            &[(0, 0), (0, -1)],
        ],
        [
            &[(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            &[(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            &[(0, 0), (-1, 0)],
        ],
    ];

    const JSTRIS_I: JstrisKicks = [
        [
            &[(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            &[(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
            &[(0, 0), (0, 1)],
        ],
        [
            &[(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
            &[(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
            &[(0, 0), (1, 0)],
        ],
        [
            &[(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
            &[(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
            &[(0, 0), (0, -1)],
        ],
        [
            &[(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
            &[(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            &[(0, 0), (-1, 0)],
        ],
    ];

    fn assert_matches_jstris(piece_type: usize, jstris: &JstrisKicks) {
        for (from_state, by_rotation) in jstris.iter().enumerate() {
            for (rotation, expected_y_up) in Rotation::ALL.iter().zip(by_rotation) {
                let expected: Vec<(i32, i32)> =
                    expected_y_up.iter().map(|&(dx, dy)| (dx, -dy)).collect();
                assert_eq!(
                    get_kicks(piece_type, from_state, *rotation),
                    expected.as_slice(),
                    "piece {piece_type} from {from_state} {rotation:?}"
                );
            }
        }
    }

    #[test]
    fn test_kicks_match_jstris_tables() {
        assert_matches_jstris(I_PIECE, &JSTRIS_I);
        for piece_type in 2..=6 {
            assert_matches_jstris(piece_type, &JSTRIS_JLSTZ);
        }
    }

    #[test]
    fn test_o_piece_never_kicks() {
        for from_state in 0..4 {
            for rotation in Rotation::ALL {
                assert_eq!(get_kicks(O_PIECE, from_state, rotation), &[(0, 0)]);
            }
        }
    }

    #[test]
    fn test_rotation_targets() {
        assert_eq!(Rotation::Clockwise.target(3), 0);
        assert_eq!(Rotation::CounterClockwise.target(0), 3);
        assert_eq!(Rotation::Half.target(1), 3);
        assert_eq!(Rotation::Half.target(3), 1);
    }

    #[test]
    #[should_panic(expected = "Invalid piece_type for kicks")]
    fn test_invalid_piece_type_panics() {
        let _ = get_kicks(7, 0, Rotation::Clockwise);
    }

    #[test]
    #[should_panic(expected = "Invalid rotation state for kicks")]
    fn test_invalid_rotation_state_panics() {
        let _ = get_kicks(2, 4, Rotation::Clockwise);
    }
}
