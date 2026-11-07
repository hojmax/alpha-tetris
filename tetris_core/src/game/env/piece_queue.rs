//! Upcoming-piece queue with inline storage.

use smallvec::SmallVec;

/// Inline capacity: the game and the search never hold more than 13 queued pieces
/// (a 6-piece queue refilled with a 7-bag); longer queues set from Python spill to the heap.
const INLINE_QUEUE_CAPACITY: usize = 16;

/// Upcoming piece types, front first.
///
/// Stored inline as bytes so cloning an environment (done twice per MCTS expansion)
/// copies the queue instead of allocating.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PieceQueue(SmallVec<[u8; INLINE_QUEUE_CAPACITY]>);

impl PieceQueue {
    pub(crate) fn from_piece_types(piece_types: &[usize]) -> Self {
        let mut queue = PieceQueue::default();
        queue.extend(piece_types.iter().copied());
        queue
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<usize> {
        self.0.get(index).map(|&piece_type| piece_type as usize)
    }

    #[inline]
    pub(crate) fn front(&self) -> Option<usize> {
        self.get(0)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|&piece_type| piece_type as usize)
    }

    #[inline]
    pub(crate) fn push_back(&mut self, piece_type: usize) {
        self.0.push(Self::encode(piece_type));
    }

    pub(crate) fn extend(&mut self, piece_types: impl IntoIterator<Item = usize>) {
        self.0.extend(piece_types.into_iter().map(Self::encode));
    }

    pub(crate) fn pop_front(&mut self) -> Option<usize> {
        if self.0.is_empty() {
            return None;
        }
        Some(self.0.remove(0) as usize)
    }

    /// Keep at most the first `max_len` pieces.
    #[inline]
    pub(crate) fn truncate(&mut self, max_len: usize) {
        self.0.truncate(max_len);
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    #[cfg(test)]
    pub(crate) fn set(&mut self, index: usize, piece_type: usize) {
        self.0[index] = Self::encode(piece_type);
    }

    #[inline]
    fn encode(piece_type: usize) -> u8 {
        u8::try_from(piece_type).expect("piece type must fit in u8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_queue_fifo_order_and_truncate() {
        let mut queue = PieceQueue::from_piece_types(&[3, 1, 4]);
        queue.push_back(6);
        queue.extend([0, 2]);
        assert_eq!(queue.iter().collect::<Vec<_>>(), vec![3, 1, 4, 6, 0, 2]);
        assert_eq!(queue.front(), Some(3));
        assert_eq!(queue.pop_front(), Some(3));
        assert_eq!(queue.get(4), Some(2));
        assert_eq!(queue.get(5), None);
        queue.truncate(2);
        assert_eq!(queue.iter().collect::<Vec<_>>(), vec![1, 4]);
        queue.clear();
        assert_eq!(queue.pop_front(), None);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_piece_queue_spills_past_inline_capacity() {
        let pieces: Vec<usize> = (0..40).map(|i| i % 7).collect();
        let mut queue = PieceQueue::from_piece_types(&pieces);
        assert_eq!(queue.len(), 40);
        assert_eq!(queue.iter().collect::<Vec<_>>(), pieces);
        assert_eq!(queue.pop_front(), Some(0));
        assert_eq!(queue.len(), 39);
    }
}
