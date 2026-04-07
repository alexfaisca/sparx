use portable_atomic::{AtomicF32, AtomicF64, Ordering};

/// Floating-point weight type usable for edge/node weights in algorithms such as GVE-Louvain
/// community detection, abstracting over `f32`/`f64` so the (hot, streamed-through-repeatedly)
/// weight arrays can be narrowed for reduced memory traffic --- same motivation, and same
/// `Plain`/`Atomic` pairing convention, as [`crate::graph::cache::AccumulatorAtomic`].
///
/// # Safety
///
/// Implementors must guarantee `Self` and `Self::Atomic` have identical size and alignment, so a
/// buffer of `Self` may be soundly reinterpreted in place as a buffer of `Self::Atomic` (used e.g.
/// for GVE-Louvain's `k`/`sigma` lock-free weight accumulators), and vice-versa.
pub unsafe trait Weight:
    Copy
    + Default
    + PartialOrd
    + std::fmt::Debug
    + std::fmt::Display
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + Send
    + Sync
    + 'static
{
    /// Width-matched atomic counterpart.
    type Atomic: WeightAtomic<Plain = Self>;
    fn zero() -> Self;
    fn one() -> Self;
    /// Widens a node/edge count (always non-negative, always representable) into a weight value.
    fn from_usize(x: usize) -> Self;
    /// Converts an `f32` value (e.g. a `HyperLogLog` cardinality estimate, which is always
    /// computed in `f32` regardless of `Self`) into a weight value.
    fn from_f32(x: f32) -> Self;
    /// Converts an `f64` value (e.g. `BFSDists::total_distances`, always computed in `f64`
    /// regardless of `Self`) into a weight value.
    fn from_f64(x: f64) -> Self;
    fn is_finite(self) -> bool;
    fn is_normal(self) -> bool;
    /// Machine epsilon at `Self`'s precision. Not currently used internally (the one place that
    /// used to need a width-aware significance threshold, GVE-Louvain's `louvain_move` gain
    /// check, now compares in `f64` unconditionally --- see `delta_q`'s docs), but kept as part
    /// of the trait's general-purpose numeric surface.
    fn epsilon() -> Self;
    /// Widens to `f64`, used only for final, single, summary scalars (always kept `f64`-precision
    /// regardless of `Self`, since it's one value, not a hot per-edge/per-node array).
    fn to_f64(self) -> f64;
}

/// Atomic counterpart of a [`Weight`] type, abstracting over [`portable_atomic`]'s `AtomicF32`/
/// `AtomicF64`.
///
/// # Safety
///
/// Implementors must guarantee `Self` and `Self::Plain` have identical size and alignment (same
/// requirement as, and mirrored by, [`Weight`]'s own safety section).
pub unsafe trait WeightAtomic: Default + Send + Sync + 'static {
    /// Non-atomic counterpart with identical size/layout as `Self`.
    type Plain: Weight<Atomic = Self>;
    fn new(v: Self::Plain) -> Self;
    fn load(&self, order: Ordering) -> Self::Plain;
    fn store(&self, v: Self::Plain, order: Ordering);
    fn fetch_add(&self, v: Self::Plain, order: Ordering) -> Self::Plain;
    fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        f: F,
    ) -> Result<Self::Plain, Self::Plain>
    where
        F: FnMut(Self::Plain) -> Option<Self::Plain>;
}

macro_rules! impl_weight {
    ($plain:ty, $atomic:ty) => {
        unsafe impl Weight for $plain {
            type Atomic = $atomic;
            #[inline(always)]
            fn zero() -> Self {
                0.
            }
            #[inline(always)]
            fn one() -> Self {
                1.
            }
            #[inline(always)]
            fn from_usize(x: usize) -> Self {
                x as $plain
            }
            #[inline(always)]
            fn from_f32(x: f32) -> Self {
                x as $plain
            }
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $plain
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <$plain>::is_finite(self)
            }
            #[inline(always)]
            fn is_normal(self) -> bool {
                <$plain>::is_normal(self)
            }
            #[inline(always)]
            fn epsilon() -> Self {
                <$plain>::EPSILON
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }

        unsafe impl WeightAtomic for $atomic {
            type Plain = $plain;
            #[inline(always)]
            fn new(v: $plain) -> Self {
                <$atomic>::new(v)
            }
            #[inline(always)]
            fn load(&self, order: Ordering) -> $plain {
                <$atomic>::load(self, order)
            }
            #[inline(always)]
            fn store(&self, v: $plain, order: Ordering) {
                <$atomic>::store(self, v, order)
            }
            #[inline(always)]
            fn fetch_add(&self, v: $plain, order: Ordering) -> $plain {
                <$atomic>::fetch_add(self, v, order)
            }
            #[inline(always)]
            fn fetch_update<F>(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: F,
            ) -> Result<$plain, $plain>
            where
                F: FnMut($plain) -> Option<$plain>,
            {
                <$atomic>::fetch_update(self, set_order, fetch_order, f)
            }
        }
    };
}

impl_weight!(f64, AtomicF64);
impl_weight!(f32, AtomicF32);
