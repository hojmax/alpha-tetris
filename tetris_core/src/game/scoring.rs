//! Attack Scoring System
//!
//! Follows Jstris's default attack rules (`GameCore.checkLineClears`):
//! - Line clear attacks (single 0, double 1, triple 2, tetris 4)
//! - T-spins: mini single 0, single 2, double 4, triple 6. A mini double is scored as a
//!   full T-spin double, and a mini triple as a T-spin triple.
//! - Combo table indexed by consecutive clears (first clear = combo 0)
//! - Back-to-back +1 for consecutive difficult clears. A T-spin mini single keeps or starts
//!   the chain but never receives the bonus.
//! - Perfect clear: the clear's own attack and back-to-back bonus are replaced by 10, the
//!   combo bonus is still added, and the B2B state follows the underlying clear

use pyo3::prelude::*;

/// Type of line clear performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearType {
    /// No lines cleared
    None,
    /// Single line clear (no T-spin)
    Single,
    /// Double line clear (no T-spin)
    Double,
    /// Triple line clear (no T-spin)
    Triple,
    /// Tetris (4 lines)
    Tetris,
    /// T-spin mini single (1 line with mini T-spin)
    TSpinMiniSingle,
    /// T-spin single (1 line with T-spin)
    TSpinSingle,
    /// T-spin double (2 lines with T-spin)
    TSpinDouble,
    /// T-spin triple (3 lines with T-spin)
    TSpinTriple,
}

impl ClearType {
    /// Check if this clear type qualifies for back-to-back bonus
    pub fn is_difficult(&self) -> bool {
        matches!(
            self,
            ClearType::Tetris
                | ClearType::TSpinMiniSingle
                | ClearType::TSpinSingle
                | ClearType::TSpinDouble
                | ClearType::TSpinTriple
        )
    }

    /// Whether this clear adds the back-to-back bonus when a chain is active.
    /// Jstris lets a T-spin mini single extend the chain without paying the bonus.
    pub fn receives_back_to_back_bonus(&self) -> bool {
        self.is_difficult() && *self != ClearType::TSpinMiniSingle
    }

    /// Get the base attack value for this clear type
    pub fn base_attack(&self) -> u32 {
        match self {
            ClearType::None => 0,
            ClearType::Single => 0,
            ClearType::Double => 1,
            ClearType::Triple => 2,
            ClearType::Tetris => 4,
            ClearType::TSpinMiniSingle => 0,
            ClearType::TSpinSingle => 2,
            ClearType::TSpinDouble => 4,
            ClearType::TSpinTriple => 6,
        }
    }
}

/// Combo attack lookup table (indices 0-11, 12+ returns 5)
const COMBO_TABLE: [u32; 12] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4];

/// Get combo attack bonus for the given combo count
/// Combo count starts at 0 for the first consecutive clear
pub fn combo_attack(combo: u32) -> u32 {
    match combo {
        0..=11 => COMBO_TABLE[combo as usize],
        _ => 5,
    }
}

/// Perfect clear attack value (replaces the clear's own attack and B2B bonus; combo still adds)
pub const PERFECT_CLEAR_ATTACK: u32 = 10;

/// Back-to-back bonus attack value
pub const BACK_TO_BACK_BONUS: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttackBreakdown {
    pub base_attack: u32,
    pub combo_attack: u32,
    pub back_to_back_attack: u32,
    pub perfect_clear_attack: u32,
    pub total_attack: u32,
    pub back_to_back_active: bool,
}

/// Result of a line clear operation with full attack calculation
#[pyclass]
#[derive(Debug, Clone)]
pub struct AttackResult {
    /// Number of lines cleared
    #[pyo3(get)]
    pub lines_cleared: u32,
    /// Attack contributed by the clear type (0 on a perfect clear, which replaces it)
    #[pyo3(get)]
    pub base_attack: u32,
    /// Combo bonus attack
    #[pyo3(get)]
    pub combo_attack: u32,
    /// Back-to-back bonus (if applicable)
    #[pyo3(get)]
    pub back_to_back_attack: u32,
    /// Perfect-clear attack contribution (if applicable)
    #[pyo3(get)]
    pub perfect_clear_attack: u32,
    /// Total attack (lines sent)
    #[pyo3(get)]
    pub total_attack: u32,
    /// Current combo count after this clear
    #[pyo3(get)]
    pub combo: u32,
    /// Whether back-to-back is active after this clear
    #[pyo3(get)]
    pub back_to_back_active: bool,
    /// Whether this was a T-spin
    #[pyo3(get)]
    pub is_tspin: bool,
    /// Whether this was a mini T-spin
    #[pyo3(get)]
    pub is_mini_tspin: bool,
    /// Whether this was a perfect clear
    #[pyo3(get)]
    pub is_perfect_clear: bool,
}

impl AttackResult {
    pub fn new() -> Self {
        AttackResult {
            lines_cleared: 0,
            base_attack: 0,
            combo_attack: 0,
            back_to_back_attack: 0,
            perfect_clear_attack: 0,
            total_attack: 0,
            combo: 0,
            back_to_back_active: false,
            is_tspin: false,
            is_mini_tspin: false,
            is_perfect_clear: false,
        }
    }
}

impl Default for AttackResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the total attack for a line clear
pub fn calculate_attack_breakdown(
    clear_type: ClearType,
    combo: u32,
    back_to_back_active: bool,
    is_perfect_clear: bool,
) -> AttackBreakdown {
    let combo_attack = combo_attack(combo);
    if is_perfect_clear {
        return AttackBreakdown {
            base_attack: 0,
            combo_attack,
            back_to_back_attack: 0,
            perfect_clear_attack: PERFECT_CLEAR_ATTACK,
            total_attack: PERFECT_CLEAR_ATTACK + combo_attack,
            back_to_back_active: clear_type.is_difficult(),
        };
    }

    let base_attack = clear_type.base_attack();
    let back_to_back_attack = if back_to_back_active && clear_type.receives_back_to_back_bonus() {
        BACK_TO_BACK_BONUS
    } else {
        0
    };

    let total_attack = base_attack + combo_attack + back_to_back_attack;
    let next_back_to_back = if clear_type == ClearType::None {
        back_to_back_active
    } else {
        clear_type.is_difficult()
    };

    AttackBreakdown {
        base_attack,
        combo_attack,
        back_to_back_attack,
        perfect_clear_attack: 0,
        total_attack,
        back_to_back_active: next_back_to_back,
    }
}

pub fn calculate_attack(
    clear_type: ClearType,
    combo: u32,
    back_to_back_active: bool,
    is_perfect_clear: bool,
) -> (u32, bool) {
    let breakdown =
        calculate_attack_breakdown(clear_type, combo, back_to_back_active, is_perfect_clear);
    (breakdown.total_attack, breakdown.back_to_back_active)
}

/// Determine the clear type based on lines cleared and T-spin status.
/// Only single-line minis score differently from full T-spins (Jstris).
pub fn determine_clear_type(lines_cleared: u32, is_tspin: bool, is_mini_tspin: bool) -> ClearType {
    match (lines_cleared, is_tspin, is_mini_tspin) {
        (0, _, _) => ClearType::None,
        (1, false, _) => ClearType::Single,
        (1, true, true) => ClearType::TSpinMiniSingle,
        (1, true, false) => ClearType::TSpinSingle,
        (2, false, _) => ClearType::Double,
        (2, true, _) => ClearType::TSpinDouble,
        (3, false, _) => ClearType::Triple,
        (3, true, _) => ClearType::TSpinTriple,
        (4, _, _) => ClearType::Tetris,
        _ => panic!("Invalid lines_cleared for determine_clear_type: {lines_cleared}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_attack_values() {
        assert_eq!(ClearType::None.base_attack(), 0);
        assert_eq!(ClearType::Single.base_attack(), 0);
        assert_eq!(ClearType::Double.base_attack(), 1);
        assert_eq!(ClearType::Triple.base_attack(), 2);
        assert_eq!(ClearType::Tetris.base_attack(), 4);
        assert_eq!(ClearType::TSpinMiniSingle.base_attack(), 0);
        assert_eq!(ClearType::TSpinSingle.base_attack(), 2);
        assert_eq!(ClearType::TSpinDouble.base_attack(), 4);
        assert_eq!(ClearType::TSpinTriple.base_attack(), 6);
    }

    #[test]
    fn test_difficult_clears() {
        assert!(!ClearType::None.is_difficult());
        assert!(!ClearType::Single.is_difficult());
        assert!(!ClearType::Double.is_difficult());
        assert!(!ClearType::Triple.is_difficult());
        assert!(ClearType::Tetris.is_difficult());
        assert!(ClearType::TSpinMiniSingle.is_difficult());
        assert!(ClearType::TSpinSingle.is_difficult());
        assert!(ClearType::TSpinDouble.is_difficult());
        assert!(ClearType::TSpinTriple.is_difficult());
    }

    #[test]
    fn test_combo_attack_values() {
        assert_eq!(combo_attack(0), 0);
        assert_eq!(combo_attack(1), 0);
        assert_eq!(combo_attack(2), 1);
        assert_eq!(combo_attack(3), 1);
        assert_eq!(combo_attack(4), 1);
        assert_eq!(combo_attack(5), 2);
        assert_eq!(combo_attack(6), 2);
        assert_eq!(combo_attack(7), 3);
        assert_eq!(combo_attack(8), 3);
        assert_eq!(combo_attack(9), 4);
        assert_eq!(combo_attack(10), 4);
        assert_eq!(combo_attack(11), 4);
        assert_eq!(combo_attack(12), 5);
        assert_eq!(combo_attack(20), 5);
    }

    #[test]
    fn test_determine_clear_type() {
        // Normal clears
        assert_eq!(determine_clear_type(0, false, false), ClearType::None);
        assert_eq!(determine_clear_type(1, false, false), ClearType::Single);
        assert_eq!(determine_clear_type(2, false, false), ClearType::Double);
        assert_eq!(determine_clear_type(3, false, false), ClearType::Triple);
        assert_eq!(determine_clear_type(4, false, false), ClearType::Tetris);

        // T-spins
        assert_eq!(
            determine_clear_type(1, true, true),
            ClearType::TSpinMiniSingle
        );
        assert_eq!(determine_clear_type(1, true, false), ClearType::TSpinSingle);
        assert_eq!(determine_clear_type(2, true, false), ClearType::TSpinDouble);
        // Jstris scores mini doubles and mini triples as full T-spins.
        assert_eq!(determine_clear_type(2, true, true), ClearType::TSpinDouble);
        assert_eq!(determine_clear_type(3, true, true), ClearType::TSpinTriple);
        assert_eq!(determine_clear_type(3, true, false), ClearType::TSpinTriple);
    }

    #[test]
    fn test_calculate_attack_basic() {
        // Single - 0 attack
        let (attack, b2b) = calculate_attack(ClearType::Single, 0, false, false);
        assert_eq!(attack, 0);
        assert!(!b2b);

        // Double - 1 attack
        let (attack, b2b) = calculate_attack(ClearType::Double, 0, false, false);
        assert_eq!(attack, 1);
        assert!(!b2b);

        // Triple - 2 attack
        let (attack, b2b) = calculate_attack(ClearType::Triple, 0, false, false);
        assert_eq!(attack, 2);
        assert!(!b2b);

        // Tetris - 4 attack
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 0, false, false);
        assert_eq!(attack, 4);
        assert!(b2b); // Tetris activates B2B
    }

    #[test]
    fn test_calculate_attack_tspin() {
        // T-spin double - 4 attack
        let (attack, b2b) = calculate_attack(ClearType::TSpinDouble, 0, false, false);
        assert_eq!(attack, 4);
        assert!(b2b);

        // T-spin triple - 6 attack
        let (attack, b2b) = calculate_attack(ClearType::TSpinTriple, 0, false, false);
        assert_eq!(attack, 6);
        assert!(b2b);

        // T-spin single - 2 attack
        let (attack, b2b) = calculate_attack(ClearType::TSpinSingle, 0, false, false);
        assert_eq!(attack, 2);
        assert!(b2b);
    }

    #[test]
    fn test_calculate_attack_back_to_back() {
        // Tetris with B2B active - 4 + 1 = 5 attack
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 0, true, false);
        assert_eq!(attack, 5);
        assert!(b2b);

        // T-spin double with B2B - 4 + 1 = 5 attack
        let (attack, b2b) = calculate_attack(ClearType::TSpinDouble, 0, true, false);
        assert_eq!(attack, 5);
        assert!(b2b);

        // Single breaks B2B (no bonus)
        let (attack, b2b) = calculate_attack(ClearType::Single, 0, true, false);
        assert_eq!(attack, 0);
        assert!(!b2b);
    }

    #[test]
    fn test_tspin_mini_single_keeps_back_to_back_without_bonus() {
        let (attack, b2b) = calculate_attack(ClearType::TSpinMiniSingle, 0, true, false);
        assert_eq!(attack, 0);
        assert!(b2b);

        let (attack, b2b) = calculate_attack(ClearType::TSpinMiniSingle, 0, false, false);
        assert_eq!(attack, 0);
        assert!(b2b, "a mini single starts a back-to-back chain");

        // The next difficult clear still gets the bonus.
        let (attack, _) = calculate_attack(ClearType::TSpinSingle, 0, true, false);
        assert_eq!(attack, 3);
    }

    #[test]
    fn test_calculate_attack_combo() {
        // Tetris at combo 5 - 4 + 2 = 6 attack
        let (attack, _) = calculate_attack(ClearType::Tetris, 5, false, false);
        assert_eq!(attack, 6);

        // Single at combo 10 - 0 + 4 = 4 attack
        let (attack, _) = calculate_attack(ClearType::Single, 10, false, false);
        assert_eq!(attack, 4);
    }

    #[test]
    fn test_calculate_attack_perfect_clear() {
        // Perfect clears send 10 (plus combo) and B2B follows the underlying clear type.
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 0, false, true);
        assert_eq!(attack, 10);
        assert!(b2b);

        // Easy perfect clears break B2B even though they still send 10.
        let (attack, b2b) = calculate_attack(ClearType::Single, 0, false, true);
        assert_eq!(attack, 10);
        assert!(!b2b);

        // Existing B2B does not add extra attack on a perfect clear.
        let (attack, b2b) = calculate_attack(ClearType::Double, 0, true, true);
        assert_eq!(attack, 10);
        assert!(!b2b);
    }

    #[test]
    fn test_calculate_attack_combined() {
        // A perfect clear drops the B2B bonus but keeps the combo bonus (combo 5 -> +2).
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 5, true, true);
        assert_eq!(attack, 12);
        assert!(b2b);
    }

    #[test]
    fn test_calculate_attack_breakdown_perfect_clear_override() {
        let breakdown = calculate_attack_breakdown(ClearType::TSpinDouble, 7, true, true);

        assert_eq!(breakdown.base_attack, 0);
        assert_eq!(breakdown.combo_attack, 3);
        assert_eq!(breakdown.back_to_back_attack, 0);
        assert_eq!(breakdown.perfect_clear_attack, 10);
        assert_eq!(breakdown.total_attack, 13);
        assert!(breakdown.back_to_back_active);
    }

    #[test]
    fn test_back_to_back_chain() {
        // First tetris - activates B2B
        let (_, b2b) = calculate_attack(ClearType::Tetris, 0, false, false);
        assert!(b2b);

        // Second tetris with B2B - keeps B2B
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 0, true, false);
        assert_eq!(attack, 5); // 4 + 1
        assert!(b2b);

        // Single breaks B2B
        let (_, b2b) = calculate_attack(ClearType::Single, 0, true, false);
        assert!(!b2b);

        // Another tetris after break - no B2B bonus
        let (attack, b2b) = calculate_attack(ClearType::Tetris, 0, false, false);
        assert_eq!(attack, 4);
        assert!(b2b); // But it re-activates B2B
    }

    #[test]
    fn test_attack_result_default() {
        let result = AttackResult::new();
        assert_eq!(result.lines_cleared, 0);
        assert_eq!(result.total_attack, 0);
        assert_eq!(result.combo, 0);
        assert!(!result.back_to_back_active);
        assert!(!result.is_mini_tspin);
    }
}
