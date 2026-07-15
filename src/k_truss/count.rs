//! Integer types for triangle counts and trussness (`Tx`).
//!
//! Like the graphs' index type (`Ix`), the width is chosen by the user: `u8` keeps per-edge
//! counts as small as possible for large, sparse graphs (an edge in at most 255 triangles), while
//! `u16`, `u32` and `u64` allow for dense graphs or multigraphs with many parallel edges. Counts
//! that don't fit the chosen width fail with an error, never wrap around.
use std::fmt::Debug;

/// Integer type of triangle counts and trussness values.
pub trait TriangleCount: Copy + Default + Ord + Debug + Send + Sync + 'static {
    /// Atomic counterpart, used while counting (and as working memory by the k-truss
    /// decompositions).
    type Atomic: TriangleCounter;
    /// Largest representable count.
    const MAX: usize;

    /// Converts `x`, which must be at most [`MAX`](Self::MAX).
    fn from_usize(x: usize) -> Self;
    fn to_usize(self) -> usize;
}

/// Atomic triangle counter (all operations are `Relaxed`). Method names differ from the atomic
/// types' own so as to never be shadowed by them.
pub trait TriangleCounter: Default + Debug + Send + Sync + 'static {
    fn count(&self) -> usize;
    /// Stores `value`, which must fit the counter.
    fn set_count(&self, value: usize);
    /// Adds `n` (wrapping), returning the previous count.
    fn add_count(&self, n: usize) -> usize;
    /// Subtracts `n` (wrapping), returning the previous count.
    fn sub_count(&self, n: usize) -> usize;
}

macro_rules! impl_triangle_count {
    ($($plain:ty => $atomic:ty),* $(,)?) => {$(
        impl TriangleCount for $plain {
            type Atomic = $atomic;
            const MAX: usize = <$plain>::MAX as usize;

            #[inline(always)]
            fn from_usize(x: usize) -> Self {
                x as $plain
            }

            #[inline(always)]
            fn to_usize(self) -> usize {
                self as usize
            }
        }

        impl TriangleCounter for $atomic {
            #[inline(always)]
            fn count(&self) -> usize {
                self.load(portable_atomic::Ordering::Relaxed) as usize
            }

            #[inline(always)]
            fn set_count(&self, value: usize) {
                self.store(value as $plain, portable_atomic::Ordering::Relaxed)
            }

            #[inline(always)]
            fn add_count(&self, n: usize) -> usize {
                self.fetch_add(n as $plain, portable_atomic::Ordering::Relaxed) as usize
            }

            #[inline(always)]
            fn sub_count(&self, n: usize) -> usize {
                self.fetch_sub(n as $plain, portable_atomic::Ordering::Relaxed) as usize
            }
        }
    )*};
}

impl_triangle_count!(
    u8 => portable_atomic::AtomicU8,
    u16 => portable_atomic::AtomicU16,
    u32 => portable_atomic::AtomicU32,
    u64 => portable_atomic::AtomicU64,
);
