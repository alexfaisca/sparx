//! Integer types for k-core decompositions' coreness and degrees (`Cx`).
//!
//! Like the triangle counts and trussness of the k-truss decompositions
//! ([`TriangleCount`](crate::k_truss::count::TriangleCount)), the width is chosen by the user:
//! `u8` keeps per-node values as small as possible for large, sparse graphs (nodes of degree at
//! most 255), while `u16`, `u32` and `u64` allow for denser graphs. Degrees that don't fit the
//! chosen width fail with an error, never wrap around.
use std::fmt::Debug;

/// Integer type of coreness values (and of the degrees the decompositions peel).
pub trait CoreCount: Copy + Default + Ord + Debug + Send + Sync + 'static {
    /// Atomic counterpart, used as working memory by the parallel decomposition.
    type Atomic: CoreCounter;
    /// Largest representable value.
    const MAX: usize;

    /// Converts `x`, which must be at most [`MAX`](Self::MAX).
    fn from_usize(x: usize) -> Self;
    fn to_usize(self) -> usize;
}

/// Atomic degree counter (all operations are `Relaxed`). Method names differ from the atomic
/// types' own so as to never be shadowed by them.
pub trait CoreCounter: Default + Debug + Send + Sync + 'static {
    fn count(&self) -> usize;
    /// Stores `value`, which must fit the counter.
    fn set_count(&self, value: usize);
    /// Decrements the count by one if it's above `k`, returning the count before if it was.
    fn decrement_above(&self, k: usize) -> Option<usize>;
}

/// Errors unless a node's degree fits the coreness type `Cx`.
pub(crate) fn check_degree<Cx: CoreCount>(
    node: usize,
    degree: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if degree > Cx::MAX {
        return Err(format!(
            "error node {node} has degree {degree}, more than the coreness type {} holds ({}): use a \
             wider one (e.g. `new_cx::<u32>`)",
            std::any::type_name::<Cx>(),
            Cx::MAX
        )
        .into());
    }
    Ok(())
}

macro_rules! impl_core_count {
    ($($plain:ty => $atomic:ty),* $(,)?) => {$(
        impl CoreCount for $plain {
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

        impl CoreCounter for $atomic {
            #[inline(always)]
            fn count(&self) -> usize {
                self.load(portable_atomic::Ordering::Relaxed) as usize
            }

            #[inline(always)]
            fn set_count(&self, value: usize) {
                self.store(value as $plain, portable_atomic::Ordering::Relaxed)
            }

            #[inline(always)]
            fn decrement_above(&self, k: usize) -> Option<usize> {
                self.fetch_update(
                    portable_atomic::Ordering::Relaxed,
                    portable_atomic::Ordering::Relaxed,
                    |x| ((x as usize) > k).then(|| x - 1),
                )
                .ok()
                .map(|x| x as usize)
            }
        }
    )*};
}

impl_core_count!(
    u8 => portable_atomic::AtomicU8,
    u16 => portable_atomic::AtomicU16,
    u32 => portable_atomic::AtomicU32,
    u64 => portable_atomic::AtomicU64,
);
