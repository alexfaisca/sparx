//! A crate-internal view of a graph's compressed sparse rows, as plain slices.

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

/// A graph's arcs, compressed by row (a graph's own, or its transpose's).
#[derive(Clone, Copy)]
pub(crate) struct Csr<'g, Ix, E> {
    pub(crate) offsets: &'g [Ix],
    pub(crate) targets: &'g [Ix],
    pub(crate) weights: &'g [E],
}

impl<'g, Ix: graph::IndexType, E: graph::E> Csr<'g, Ix, E> {
    pub(crate) fn of<N: graph::N, Dir: Directedness>(g: &'g GraphMemoryMap<N, E, Ix, Dir>) -> Self {
        let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
        let targets = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());
        let weights = SharedSlice::<E>::new(g.edge_labels_ptr(), g.width());
        // the slices live as long as `g`'s maps
        unsafe {
            Self {
                offsets: std::slice::from_raw_parts(
                    offsets.as_slice().as_ptr(),
                    offsets.as_slice().len(),
                ),
                targets: std::slice::from_raw_parts(
                    targets.as_slice().as_ptr(),
                    targets.as_slice().len(),
                ),
                weights: std::slice::from_raw_parts(
                    weights.as_slice().as_ptr(),
                    weights.as_slice().len(),
                ),
            }
        }
    }

    /// `u`'s arcs' range.
    #[inline(always)]
    pub(crate) fn range(&self, u: usize) -> std::ops::Range<usize> {
        self.offsets[u].index()..self.offsets[u + 1].index()
    }
}
