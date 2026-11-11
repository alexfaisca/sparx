//! Node and edge labels: the ready-made ones, and [`label!`](crate::label!) to declare others.

use super::{E, N};
use bytemuck::{Pod, Zeroable};

/// No label: what unlabeled nodes and edges carry (as `()` does).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct VoidLabel {}

impl N for VoidLabel {
    fn new(_v: usize) -> Self {
        Self {}
    }
}
impl E for VoidLabel {
    fn new(_v: usize) -> Self {
        Self {}
    }
}

impl N for () {
    fn new(_v: usize) {}
}

impl E for () {
    fn new(_v: usize) {}
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct NodeLabel {
    label: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct EdgeLabel {
    label: usize,
}

impl N for NodeLabel {
    fn new(v: usize) -> Self {
        NodeLabel { label: v }
    }
}

impl E for EdgeLabel {
    fn new(v: usize) -> Self {
        EdgeLabel { label: v }
    }
}

/// Declares label types: each a newtype over an unsigned integer (`u8` to `u64`, `usize`), with
/// everything [`N`](crate::graph::N) or [`E`](crate::graph::E) labels need, so that graphs can
/// carry them (in their memory-mapped label files).
///
/// ```
/// sparx::label! {
///     /// A k-mer's color.
///     pub struct Color(u32): N;
///     /// A link's orientation, which also colors nodes.
///     pub struct Orientation(pub u8): N, E;
/// }
///
/// use sparx::graph::{E, N};
/// let c = <Color as N>::new(7);
/// assert!(<Color as N>::is_labeled());
/// assert_eq!(c.label(), 7);
/// assert_eq!(<Orientation as E>::new(3).0, 3);
/// ```
///
/// Each type gets:
///
/// * `#[repr(transparent)]`, and `Clone`, `Copy`, `Debug`, `Default`, `PartialEq`, `Eq`,
///   `PartialOrd`, `Ord` and `Hash` derived (attributes and doc comments written before `struct`
///   are kept);
/// * `bytemuck`'s `Pod` and `Zeroable`, sound as the type is a transparent integer (checked at
///   compile time), without the declaring crate depending on `bytemuck`;
/// * for each of `N` and `E` listed after the colon, the trait: `new(v)` wraps `v` (which must
///   fit the integer: it's cast, and checked in debug builds), `is_labeled()` is `true` and
///   `label()` is the integer.
///
/// Other traits, and other field types, are compile errors:
///
/// ```compile_fail
/// sparx::label! { struct Weight(u32): W; }
/// ```
///
/// ```compile_fail
/// sparx::label! { struct Fraction(f32): E; }
/// ```
#[macro_export]
macro_rules! label {
    ($(
        $(#[$meta:meta])*
        $vis:vis struct $name:ident($field_vis:vis $int:ty): $($kind:ident),+ $(,)?;
    )*) => {$(
        $(#[$meta])*
        #[derive(
            ::core::clone::Clone,
            ::core::marker::Copy,
            ::core::fmt::Debug,
            ::core::default::Default,
            ::core::cmp::PartialEq,
            ::core::cmp::Eq,
            ::core::cmp::PartialOrd,
            ::core::cmp::Ord,
            ::core::hash::Hash,
        )]
        #[repr(transparent)]
        $vis struct $name($field_vis $int);

        // SAFETY: a transparent wrapper of an integer, which is `Pod` (asserted below): any bit
        // pattern is valid, zeroes included, and there's no padding
        unsafe impl $crate::graph::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::graph::__bytemuck::Pod for $name {}
        const _: () = {
            const fn integer<T: $crate::graph::__bytemuck::Pod + ::core::convert::TryFrom<usize>>() {}
            integer::<$int>();
        };

        $($crate::label!(@impl $kind, $name, $int);)+
    )*};

    (@impl N, $name:ident, $int:ty) => {
        impl $crate::graph::N for $name {
            #[inline]
            fn new(v: usize) -> Self {
                debug_assert!(
                    <$int as ::core::convert::TryFrom<usize>>::try_from(v).is_ok(),
                    "label {v} doesn't fit a {}",
                    ::core::stringify!($int),
                );
                Self(v as $int)
            }
            #[inline]
            fn is_labeled() -> bool {
                true
            }
            #[inline]
            fn label(&self) -> usize {
                self.0 as usize
            }
        }
    };

    (@impl E, $name:ident, $int:ty) => {
        impl $crate::graph::E for $name {
            #[inline]
            fn new(v: usize) -> Self {
                debug_assert!(
                    <$int as ::core::convert::TryFrom<usize>>::try_from(v).is_ok(),
                    "label {v} doesn't fit a {}",
                    ::core::stringify!($int),
                );
                Self(v as $int)
            }
            #[inline]
            fn is_labeled() -> bool {
                true
            }
            #[inline]
            fn label(&self) -> usize {
                self.0 as usize
            }
        }
    };

    (@impl $other:ident, $name:ident, $int:ty) => {
        ::core::compile_error!(::core::concat!(
            "label! implements `N` and `E`, not `",
            ::core::stringify!($other),
            "`",
        ));
    };
}

#[cfg(test)]
mod tests {
    use crate::graph::{E, N};

    crate::label! {
        /// Both kinds.
        struct Both(u16): N, E;
        pub(crate) struct Wide(pub usize): E;
    }

    #[test]
    fn labels_carry_their_integer() {
        let b = <Both as N>::new(513);
        assert_eq!((N::label(&b), E::label(&<Both as E>::new(9))), (513, 9));
        assert!(<Both as N>::is_labeled() && <Both as E>::is_labeled());
        assert!(!<Both as E>::is_weighted());
        assert_eq!(<Wide as E>::new(usize::MAX).0, usize::MAX);
        assert_eq!(Both::default(), Both(0));
        assert!(Both(1) < Both(2));
        // the memory-mapped files' view of them
        assert_eq!(std::mem::size_of::<Both>(), 2);
        let bytes: &[u8] = bytemuck::cast_slice(&[Both(0x0102)]);
        assert_eq!(bytes, 0x0102u16.to_ne_bytes());
    }

    #[test]
    #[should_panic(expected = "doesn't fit")]
    #[cfg(debug_assertions)]
    fn labels_that_dont_fit_panic_in_debug_builds() {
        let _ = <Both as N>::new(1 << 16);
    }
}
