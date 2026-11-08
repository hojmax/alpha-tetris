use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::game::constants::PLACEMENT_CACHE_MAX_ENTRIES;
use crate::game::piece::Piece;

use std::sync::Arc;

use super::state::PlacementSet;
use super::TetrisEnv;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PackedBoardKey {
    bits: [u64; 4],
    width: u8,
    height: u8,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PlacementLookupKey {
    board: PackedBoardKey,
    piece_type: u8,
    x: i8,
    y: i8,
    rotation: u8,
    hold_available: bool,
}

#[derive(Default)]
struct WorkerGlobalCache {
    placement_cache: HashMap<PlacementLookupKey, Arc<PlacementSet>>,
    placement_order: VecDeque<PlacementLookupKey>,
}

thread_local! {
    static WORKER_GLOBAL_CACHE: RefCell<WorkerGlobalCache> =
        RefCell::new(WorkerGlobalCache::default());
}

pub(crate) fn build_board_key(env: &TetrisEnv) -> Option<PackedBoardKey> {
    if env.width > u8::MAX as usize || env.height > u8::MAX as usize {
        return None;
    }

    let cells = env.board_cells();
    if cells.len() > 256 {
        return None;
    }

    let mut bits = [0u64; 4];
    for (idx, &cell) in cells.iter().enumerate() {
        if cell != 0 {
            bits[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    Some(PackedBoardKey {
        bits,
        width: env.width as u8,
        height: env.height as u8,
    })
}

pub(crate) fn build_placement_lookup_key(
    env: &TetrisEnv,
    piece: &Piece,
    hold_available: bool,
) -> Option<PlacementLookupKey> {
    let board = build_board_key(env)?;
    let piece_type = u8::try_from(piece.piece_type).ok()?;
    let x = i8::try_from(piece.x).ok()?;
    let y = i8::try_from(piece.y).ok()?;
    let rotation = u8::try_from(piece.rotation).ok()?;

    Some(PlacementLookupKey {
        board,
        piece_type,
        x,
        y,
        rotation,
        hold_available,
    })
}

pub(crate) fn get_cached_placements(key: PlacementLookupKey) -> Option<Arc<PlacementSet>> {
    WORKER_GLOBAL_CACHE.with(|cache| cache.borrow().placement_cache.get(&key).cloned())
}

pub(crate) fn insert_cached_placements(key: PlacementLookupKey, value: Arc<PlacementSet>) {
    WORKER_GLOBAL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.placement_cache.insert(key, value).is_none() {
            cache.placement_order.push_back(key);
        }

        while cache.placement_cache.len() > PLACEMENT_CACHE_MAX_ENTRIES {
            let Some(oldest_key) = cache.placement_order.pop_front() else {
                break;
            };
            cache.placement_cache.remove(&oldest_key);
        }
    });
}
