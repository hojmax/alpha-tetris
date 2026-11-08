//! FIFO cache from packed boards to fixed-width f32 rows (the board-only fusion input).
//!
//! Values live in large chunks rather than one heap allocation per entry, lookups return a
//! borrowed slice instead of a clone, and keys hash with a cheap multiply-xorshift mix
//! instead of SipHash (the keys are exact board bitsets, not attacker-controlled input).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// Entries per value chunk; chunks are allocated as the cache grows.
const CHUNK_ENTRIES: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct BoardKey(pub(super) [u64; 4]);

impl Hash for BoardKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;
        for word in self.0 {
            h = (h ^ word).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 31;
        }
        state.write_u64(h);
    }
}

/// Passes through the single pre-mixed u64 written by `BoardKey::hash`.
#[derive(Default)]
pub(super) struct BoardKeyHasher(u64);

impl Hasher for BoardKeyHasher {
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("BoardKeyHasher only hashes BoardKey");
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

pub(super) struct BoardCache {
    width: usize,
    capacity: usize,
    slots: HashMap<BoardKey, u32, BuildHasherDefault<BoardKeyHasher>>,
    /// Slot -> key, in insertion order until the cache is full.
    keys: Vec<BoardKey>,
    chunks: Vec<Box<[f32]>>,
    /// Oldest slot, overwritten by the next insert once the cache is full.
    next_evict: usize,
}

impl BoardCache {
    pub(super) fn new(width: usize, capacity: usize) -> Self {
        assert!(width > 0, "board cache rows must be non-empty");
        assert!(
            capacity > 0 && capacity <= u32::MAX as usize,
            "bad board cache capacity"
        );
        Self {
            width,
            capacity,
            slots: HashMap::default(),
            keys: Vec::new(),
            chunks: Vec::new(),
            next_evict: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn clear(&mut self) {
        self.slots = HashMap::default();
        self.keys = Vec::new();
        self.chunks = Vec::new();
        self.next_evict = 0;
    }

    pub(super) fn slot_of(&self, key: &BoardKey) -> Option<usize> {
        self.slots.get(key).map(|&slot| slot as usize)
    }

    pub(super) fn row(&self, slot: usize) -> &[f32] {
        let offset = (slot % CHUNK_ENTRIES) * self.width;
        &self.chunks[slot / CHUNK_ENTRIES][offset..offset + self.width]
    }

    /// Insert `key` (which must be absent), evicting the oldest entry when full, and return
    /// its slot together with the row to fill. Rows of evicted entries are reused as is.
    pub(super) fn insert(&mut self, key: BoardKey) -> (usize, &mut [f32]) {
        debug_assert!(!self.slots.contains_key(&key), "board already cached");
        let slot = if self.keys.len() < self.capacity {
            let slot = self.keys.len();
            if slot % CHUNK_ENTRIES == 0 {
                let entries = CHUNK_ENTRIES.min(self.capacity - slot);
                self.chunks
                    .push(vec![0.0; entries * self.width].into_boxed_slice());
            }
            self.keys.push(key);
            slot
        } else {
            let slot = self.next_evict;
            self.next_evict = (slot + 1) % self.capacity;
            let evicted = std::mem::replace(&mut self.keys[slot], key);
            self.slots.remove(&evicted);
            slot
        };
        self.slots.insert(key, slot as u32);
        let offset = (slot % CHUNK_ENTRIES) * self.width;
        let row = &mut self.chunks[slot / CHUNK_ENTRIES][offset..offset + self.width];
        (slot, row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u64) -> BoardKey {
        BoardKey([i, i.rotate_left(13), 0, 1 << (i % 64)])
    }

    #[test]
    fn stores_and_evicts_in_fifo_order() {
        let mut cache = BoardCache::new(3, 5);
        for i in 0..5u64 {
            let (_, row) = cache.insert(key(i));
            row.copy_from_slice(&[i as f32, 0.5, -1.0]);
        }
        assert_eq!(cache.len(), 5);
        for i in 0..5u64 {
            let slot = cache.slot_of(&key(i)).expect("cached");
            assert_eq!(cache.row(slot), &[i as f32, 0.5, -1.0]);
        }

        for i in 5..8u64 {
            let (_, row) = cache.insert(key(i));
            row.fill(i as f32);
        }
        assert_eq!(cache.len(), 5);
        for i in 0..3u64 {
            assert!(
                cache.slot_of(&key(i)).is_none(),
                "key {i} should be evicted"
            );
        }
        for i in 3..5u64 {
            assert_eq!(cache.row(cache.slot_of(&key(i)).unwrap())[0], i as f32);
        }
        for i in 5..8u64 {
            assert_eq!(cache.row(cache.slot_of(&key(i)).unwrap()), &[i as f32; 3]);
        }

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.slot_of(&key(6)).is_none());
        let (slot, row) = cache.insert(key(6));
        row.fill(2.0);
        assert_eq!(slot, 0);
        assert_eq!(cache.row(cache.slot_of(&key(6)).unwrap()), &[2.0; 3]);
    }

    #[test]
    fn spans_multiple_chunks() {
        let entries = CHUNK_ENTRIES * 2 + 3;
        let mut cache = BoardCache::new(2, entries);
        for i in 0..entries as u64 {
            let (_, row) = cache.insert(key(i));
            row.copy_from_slice(&[i as f32, -(i as f32)]);
        }
        for i in (0..entries as u64).step_by(997) {
            let slot = cache.slot_of(&key(i)).unwrap();
            assert_eq!(cache.row(slot), &[i as f32, -(i as f32)]);
        }
    }
}
