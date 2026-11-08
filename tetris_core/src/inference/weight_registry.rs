//! Process-wide sharing of immutable model weights.
//!
//! Every self-play/eval worker thread loads the incumbent bundle itself. Without sharing,
//! each worker holds a private copy of the weights, and with many workers per shared cache
//! (6 cores per L2 cluster on an M5 Pro, 8 per L3 on a Zen 3 CCX) those copies no longer fit
//! and every leaf evaluation streams weights from DRAM. On an M5 Pro, 18 workers sharing one
//! copy ran ~1.9x faster than 18 private copies (IPC 0.70 -> 1.49) with identical results.
//!
//! The registry keeps only `Weak` references, so a bundle's weights are freed once the last
//! worker drops it (e.g. after a model sync).

use std::collections::HashMap;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::SystemTime;

/// Identity of a model bundle on disk: canonical path, length and mtime of each file, so a
/// bundle rewritten in place is loaded again instead of being served from the registry.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct BundleKey(Vec<(PathBuf, u64, SystemTime)>);

impl BundleKey {
    pub(super) fn for_files(paths: &[&Path]) -> std::io::Result<Self> {
        paths
            .iter()
            .map(|path| {
                let meta = std::fs::metadata(path)?;
                Ok((std::fs::canonicalize(path)?, meta.len(), meta.modified()?))
            })
            .collect::<std::io::Result<Vec<_>>>()
            .map(BundleKey)
    }
}

pub(super) struct WeakRegistry<K, V> {
    entries: Mutex<HashMap<K, Weak<V>>>,
}

impl<K: Eq + Hash, V> WeakRegistry<K, V> {
    pub(super) fn new() -> Self {
        WeakRegistry {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Return the live value for `key`, or build it with `load` and remember it.
    ///
    /// The lock is held while loading so concurrent loads of the same bundle (every worker
    /// at startup or after a model sync) wait for the first one and then share its result.
    pub(super) fn get_or_try_insert_with<E>(
        &self,
        key: K,
        load: impl FnOnce() -> Result<V, E>,
    ) -> Result<Arc<V>, E> {
        let mut entries = self.entries.lock().expect("weight registry lock poisoned");
        entries.retain(|_, weak| weak.strong_count() > 0);
        if let Some(value) = entries.get(&key).and_then(Weak::upgrade) {
            return Ok(value);
        }
        let value = Arc::new(load()?);
        entries.insert(key, Arc::downgrade(&value));
        Ok(value)
    }

    #[cfg(test)]
    fn live_entries(&self) -> usize {
        let entries = self.entries.lock().expect("weight registry lock poisoned");
        entries
            .values()
            .filter(|weak| weak.strong_count() > 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_shares_one_value_and_loads_once() {
        let registry = WeakRegistry::<u32, Vec<f32>>::new();
        let mut loads = 0;
        let a = registry
            .get_or_try_insert_with(1, || {
                loads += 1;
                Ok::<_, ()>(vec![1.0; 4])
            })
            .unwrap();
        let b = registry
            .get_or_try_insert_with(1, || {
                loads += 1;
                Ok::<_, ()>(vec![2.0; 4])
            })
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(loads, 1);
        assert_eq!(*b, vec![1.0; 4]);
    }

    #[test]
    fn dropped_values_are_released_and_reloaded() {
        let registry = WeakRegistry::<u32, Vec<f32>>::new();
        let first = registry
            .get_or_try_insert_with(7, || Ok::<_, ()>(vec![1.0]))
            .unwrap();
        assert_eq!(registry.live_entries(), 1);
        drop(first);
        assert_eq!(registry.live_entries(), 0);
        let second = registry
            .get_or_try_insert_with(7, || Ok::<_, ()>(vec![2.0]))
            .unwrap();
        assert_eq!(*second, vec![2.0]);
    }

    #[test]
    fn load_errors_are_returned_and_not_cached() {
        let registry = WeakRegistry::<u32, Vec<f32>>::new();
        let err = registry.get_or_try_insert_with(3, || Err::<Vec<f32>, _>("boom"));
        assert_eq!(err.unwrap_err(), "boom");
        let ok = registry
            .get_or_try_insert_with(3, || Ok::<_, &str>(vec![5.0]))
            .unwrap();
        assert_eq!(*ok, vec![5.0]);
    }

    #[test]
    fn bundle_key_changes_when_a_file_is_rewritten() {
        let dir = std::env::temp_dir().join(format!("tetris_bundle_key_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.fc.bin");
        std::fs::write(&file, b"abc").unwrap();
        let before = BundleKey::for_files(&[&file]).unwrap();
        assert_eq!(before, BundleKey::for_files(&[&file]).unwrap());
        std::fs::write(&file, b"abcdef").unwrap();
        let after = BundleKey::for_files(&[&file]).unwrap();
        assert_ne!(before, after);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
