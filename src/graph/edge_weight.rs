//! Weighted edges: the [`EdgeWeight`] values an edge may store, and [`Weighted`], the edge label
//! holding one.
//!
//! An edge's *stored* weight type (any of `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`,
//! `i64`, `isize`, `f32`, `f64`) is independent of the [`Weight`] type an algorithm *computes*
//! with: algorithms read weights through [`E::weight`], which converts the stored value into
//! their own `W`.
//!
//! Weights are read from input files in one of two [`WeightParsing`] modes, chosen by the caller:
//! [`Fast`](WeightParsing::Fast) (the default) does no per-edge validation --- a token that isn't
//! a valid value of the chosen type gives an unspecified (but never undefined-behaviour) value ---
//! while [`Checked`](WeightParsing::Checked) rejects malformed, out-of-range or non-finite values.
//!
//! [`E::weight`]: crate::graph::E::weight

use std::cmp::Ordering;
use std::fmt::Debug;

use super::{E, Weight, WeightMerge};

/// How edge weights are read from input files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeightParsing {
    /// No per-edge validation: a token that isn't a valid value of the chosen weight type (say,
    /// `70000` for `u16`, `-3` for `u32` or `abc` for anything) gives an unspecified value.
    /// Choosing a type that fits the data is the caller's responsibility.
    #[default]
    Fast,
    /// Every weight is validated: malformed, out-of-range and (for floats) non-finite values fail
    /// the parse, naming the offending token.
    Checked,
}

/// What values a weight type can hold, checked once per input file against what the file
/// declares (e.g. a `real` MatrixMarket file can't be loaded into integer weights).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightDomain {
    /// Non-negative integers.
    Unsigned,
    /// Integers.
    Signed,
    /// Real numbers.
    Real,
}

/// A weight that failed [`WeightParsing::Checked`] validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseWeightError {
    token: String,
    ty: &'static str,
}

impl ParseWeightError {
    pub fn new(token: &[u8], ty: &'static str) -> Self {
        Self {
            token: String::from_utf8_lossy(token).into_owned(),
            ty,
        }
    }
}

impl std::fmt::Display for ParseWeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error {:?} isn't a valid `{}` edge weight",
            self.token, self.ty
        )
    }
}

impl std::error::Error for ParseWeightError {}

/// A value an edge can store as its weight.
pub trait EdgeWeight: Copy + Debug + Default + bytemuck::Pod + Send + Sync + 'static {
    const DOMAIN: WeightDomain;

    /// Fast path: parses `tok` without validation (see [`WeightParsing::Fast`]).
    fn from_token(tok: &[u8]) -> Self;
    /// Checked path: parses `tok`, failing on anything that isn't a valid (and, for floats,
    /// finite) value of `Self` (see [`WeightParsing::Checked`]).
    fn try_from_token(tok: &[u8]) -> Result<Self, ParseWeightError>;
    fn from_index(v: usize) -> Self;
    /// The weight of the reverse of an arc in a skew-symmetric input (wrapping for integers).
    fn negate(self) -> Self;
    fn as_f64(self) -> f64;
    /// The weight as an exact non-negative integer, if it is one (`3.0` is; `0.5`, `-1` and
    /// `NaN` aren't): for algorithms computing exactly in integers.
    fn as_exact_u64(self) -> Option<u64>;
    /// The sum of two weights, `None` if it overflows (integers).
    fn checked_sum(self, other: Self) -> Option<Self>;
    /// A total order (floats by [`f64::total_cmp`]), so weighted labels can be [`Ord`].
    fn weight_cmp(&self, other: &Self) -> Ordering;
}

/// Parses an unsigned decimal integer without any validation: each byte is taken as a digit.
#[inline(always)]
fn fast_unsigned<T>(tok: &[u8], zero: T, step: impl Fn(T, u8) -> T) -> T {
    tok.iter()
        .fold(zero, |acc, &b| step(acc, b.wrapping_sub(b'0')))
}

macro_rules! impl_edge_weight_unsigned {
    ($($t:ty),*) => {$(
        impl EdgeWeight for $t {
            const DOMAIN: WeightDomain = WeightDomain::Unsigned;
            #[inline(always)]
            fn checked_sum(self, other: Self) -> Option<Self> {
                self.checked_add(other)
            }
            #[inline(always)]
            fn as_exact_u64(self) -> Option<u64> {
                Some(self as u64)
            }
            #[inline(always)]
            fn from_token(tok: &[u8]) -> Self {
                fast_unsigned(tok, 0 as $t, |acc, d| acc.wrapping_mul(10).wrapping_add(d as $t))
            }
            #[inline]
            fn try_from_token(tok: &[u8]) -> Result<Self, ParseWeightError> {
                std::str::from_utf8(tok)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| ParseWeightError::new(tok, stringify!($t)))
            }
            #[inline(always)]
            fn from_index(v: usize) -> Self {
                v as $t
            }
            #[inline(always)]
            fn negate(self) -> Self {
                self.wrapping_neg()
            }
            #[inline(always)]
            fn as_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn weight_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )*};
}

macro_rules! impl_edge_weight_signed {
    ($($t:ty),*) => {$(
        impl EdgeWeight for $t {
            const DOMAIN: WeightDomain = WeightDomain::Signed;
            #[inline(always)]
            fn checked_sum(self, other: Self) -> Option<Self> {
                self.checked_add(other)
            }
            #[inline(always)]
            fn as_exact_u64(self) -> Option<u64> {
                u64::try_from(self).ok()
            }
            #[inline(always)]
            fn from_token(tok: &[u8]) -> Self {
                let step = |acc: $t, d: u8| acc.wrapping_mul(10).wrapping_add(d as $t);
                match tok.split_first() {
                    Some((b'-', digits)) => fast_unsigned(digits, 0, step).wrapping_neg(),
                    _ => fast_unsigned(tok, 0, step),
                }
            }
            #[inline]
            fn try_from_token(tok: &[u8]) -> Result<Self, ParseWeightError> {
                std::str::from_utf8(tok)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| ParseWeightError::new(tok, stringify!($t)))
            }
            #[inline(always)]
            fn from_index(v: usize) -> Self {
                v as $t
            }
            #[inline(always)]
            fn negate(self) -> Self {
                self.wrapping_neg()
            }
            #[inline(always)]
            fn as_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn weight_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )*};
}

macro_rules! impl_edge_weight_float {
    ($($t:ty),*) => {$(
        impl EdgeWeight for $t {
            const DOMAIN: WeightDomain = WeightDomain::Real;
            #[inline(always)]
            fn checked_sum(self, other: Self) -> Option<Self> {
                Some(self + other)
            }
            #[inline(always)]
            fn as_exact_u64(self) -> Option<u64> {
                // (`NaN` is in no range; 2^64 is exact in both widths)
                ((0.0..18446744073709551616.0).contains(&self) && self.fract() == 0.0)
                    .then_some(self as u64)
            }
            /// Malformed tokens give `NaN`.
            #[inline(always)]
            fn from_token(tok: &[u8]) -> Self {
                // `from_utf8` on a short ASCII token is a cheap scan, and keeps this safe
                std::str::from_utf8(tok)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(<$t>::NAN)
            }
            #[inline]
            fn try_from_token(tok: &[u8]) -> Result<Self, ParseWeightError> {
                std::str::from_utf8(tok)
                    .ok()
                    .and_then(|s| s.parse::<$t>().ok())
                    .filter(|w| w.is_finite())
                    .ok_or_else(|| ParseWeightError::new(tok, stringify!($t)))
            }
            #[inline(always)]
            fn from_index(v: usize) -> Self {
                v as $t
            }
            #[inline(always)]
            fn negate(self) -> Self {
                -self
            }
            #[inline(always)]
            fn as_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn weight_cmp(&self, other: &Self) -> Ordering {
                <$t>::total_cmp(self, other)
            }
        }
    )*};
}

impl_edge_weight_unsigned!(u8, u16, u32, u64, usize);
impl_edge_weight_signed!(i8, i16, i32, i64, isize);
impl_edge_weight_float!(f32, f64);

/// An edge label holding a weight of type `T` (see [`EdgeWeight`] for the types it may hold, and
/// [`WeightParsing`] for how they are read).
///
/// ```
/// use sparx::graph::{DiGraph, Weighted, label::VoidLabel};
/// # fn f() -> Result<(), Box<dyn std::error::Error>> {
/// // edges weighted by `u32`s, read with the fast (unvalidated) parser
/// let g = DiGraph::<VoidLabel, Weighted<u32>, usize>::from_file("graph.mtx", None, None)?;
/// # Ok(()) }
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Weighted<T: EdgeWeight>(pub T);

// SAFETY: `#[repr(transparent)]` over a `Pod` field
unsafe impl<T: EdgeWeight> bytemuck::Zeroable for Weighted<T> {}
unsafe impl<T: EdgeWeight> bytemuck::Pod for Weighted<T> {}

impl<T: EdgeWeight> PartialEq for Weighted<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.weight_cmp(&other.0) == Ordering::Equal
    }
}
impl<T: EdgeWeight> Eq for Weighted<T> {}
impl<T: EdgeWeight> PartialOrd for Weighted<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: EdgeWeight> Ord for Weighted<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.weight_cmp(&other.0)
    }
}

impl<T: EdgeWeight> E for Weighted<T> {
    #[inline(always)]
    fn new(v: usize) -> Self {
        Weighted(T::from_index(v))
    }
    #[inline(always)]
    fn is_labeled() -> bool {
        true
    }
    #[inline(always)]
    fn is_weighted() -> bool {
        true
    }
    #[inline(always)]
    fn weight<W: Weight>(&self) -> W {
        W::from_f64(self.0.as_f64())
    }
    #[inline(always)]
    fn integer_weight(&self) -> Option<u64> {
        self.0.as_exact_u64()
    }
    #[inline(always)]
    fn weight_domain() -> WeightDomain {
        T::DOMAIN
    }
    #[inline(always)]
    fn parse_weight<const CHECKED: bool>(tok: &[u8]) -> Result<Self, ParseWeightError> {
        if CHECKED {
            T::try_from_token(tok).map(Weighted)
        } else {
            Ok(Weighted(T::from_token(tok)))
        }
    }
    #[inline(always)]
    fn negate_weight(self) -> Self {
        Weighted(self.0.negate())
    }
    #[inline(always)]
    fn merge_weight(self, other: Self, merge: WeightMerge) -> Option<Self> {
        match merge {
            WeightMerge::Sum => self.0.checked_sum(other.0).map(Weighted),
            WeightMerge::Min => Some(self.min(other)),
            WeightMerge::Max => Some(self.max(other)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn fast<T: EdgeWeight>(s: &str) -> T {
        T::from_token(s.as_bytes())
    }
    fn checked<T: EdgeWeight>(s: &str) -> Result<T, ParseWeightError> {
        T::try_from_token(s.as_bytes())
    }

    #[test]
    fn fast_parsing_reads_valid_tokens_like_std() {
        for s in ["0", "7", "42", "65535", "18446744073709551615"] {
            assert_eq!(fast::<u64>(s), s.parse::<u64>().unwrap(), "{s}");
        }
        assert_eq!(fast::<u8>("255"), 255);
        assert_eq!(fast::<u16>("65535"), u16::MAX);
        assert_eq!(fast::<u32>("4000000000"), 4_000_000_000);
        assert_eq!(fast::<usize>("123456"), 123_456);
        for s in ["0", "-1", "17", "-128", "127"] {
            assert_eq!(fast::<i8>(s), s.parse::<i8>().unwrap(), "{s}");
        }
        assert_eq!(fast::<i64>("-9223372036854775808"), i64::MIN);
        assert_eq!(fast::<i32>("-123456"), -123_456);
        for s in ["0", "1.5", "-2.0e-3", "7", "1e10", ".5"] {
            assert_eq!(fast::<f64>(s), s.parse::<f64>().unwrap(), "{s}");
            assert_eq!(fast::<f32>(s), s.parse::<f32>().unwrap(), "{s}");
        }
    }

    /// Fast parsing never fails or panics, whatever it's given: it just gives some value.
    #[test]
    fn fast_parsing_never_fails_on_bad_tokens() {
        for s in [
            "",
            "-",
            "abc",
            "70000",
            "-3",
            "1.5",
            "99999999999999999999999",
            "é",
        ] {
            let _ = fast::<u8>(s);
            let _ = fast::<u16>(s);
            let _ = fast::<u64>(s);
            let _ = fast::<i8>(s);
            let _ = fast::<i64>(s);
        }
        assert!(fast::<f64>("abc").is_nan());
        assert!(fast::<f32>("").is_nan());
        // out of range wraps
        assert_eq!(fast::<u16>("65536"), 0);
    }

    #[test]
    fn checked_parsing_rejects_what_the_type_cant_hold() {
        assert_eq!(checked::<u16>("65535"), Ok(u16::MAX));
        assert_eq!(checked::<i32>("-5"), Ok(-5));
        assert_eq!(checked::<f64>("2.5e-1"), Ok(0.25));
        for (ok, s) in [
            (checked::<u16>("70000").is_err(), "u16 overflow"),
            (checked::<u32>("-3").is_err(), "negative unsigned"),
            (checked::<u64>("1.5").is_err(), "real into integer"),
            (checked::<i8>("128").is_err(), "i8 overflow"),
            (checked::<u8>("").is_err(), "empty"),
            (checked::<usize>("abc").is_err(), "garbage"),
            (checked::<f32>("abc").is_err(), "garbage float"),
            (checked::<f64>("inf").is_err(), "infinite"),
            (checked::<f64>("NaN").is_err(), "nan"),
        ] {
            assert!(ok, "{s}");
        }
        let err = checked::<u16>("70000").unwrap_err().to_string();
        assert!(err.contains("70000") && err.contains("u16"), "{err}");
    }

    #[test]
    fn weighted_labels_are_ordered_and_convert_to_any_weight() {
        let w = Weighted(3u16);
        assert_eq!(w.weight::<f64>(), 3.0);
        assert_eq!(w.weight::<f32>(), 3.0);
        assert_eq!(Weighted(0.5f32).weight::<f64>(), 0.5);
        assert_eq!(Weighted(-2i32).negate_weight(), Weighted(2));
        assert!(Weighted(1.0f64) < Weighted(2.0));
        assert!(Weighted(f64::NAN) == Weighted(f64::NAN));
        assert_eq!(<Weighted<u8> as E>::new(1), Weighted(1));
        assert!(<Weighted<u8> as E>::is_weighted());
        assert_eq!(
            Weighted::<u32>::parse_weight::<false>(b"12"),
            Ok(Weighted(12))
        );
        assert!(Weighted::<u32>::parse_weight::<true>(b"-1").is_err());
        assert_eq!(Weighted(7u16).integer_weight(), Some(7));
        assert_eq!(Weighted(u64::MAX).integer_weight(), Some(u64::MAX));
        assert_eq!(Weighted(3i32).integer_weight(), Some(3));
        assert_eq!(Weighted(-3i32).integer_weight(), None);
        assert_eq!(Weighted(4.0f32).integer_weight(), Some(4));
        assert_eq!(Weighted(0.0f64).integer_weight(), Some(0));
        for w in [0.5, -1.0, f64::NAN, f64::INFINITY, 1e20] {
            assert_eq!(Weighted(w).integer_weight(), None, "{w}");
        }
        assert_eq!(
            std::mem::size_of::<Weighted<u16>>(),
            std::mem::size_of::<u16>()
        );
    }

    /// Unweighted labels weigh one, whatever they hold.
    #[test]
    fn unweighted_labels_weigh_one() {
        use crate::graph::label::{EdgeLabel, VoidLabel};
        assert_eq!(VoidLabel {}.weight::<f64>(), 1.0);
        assert_eq!(().weight::<f32>(), 1.0);
        assert_eq!(<EdgeLabel as E>::new(7).weight::<f64>(), 1.0);
        assert_eq!(<EdgeLabel as E>::new(7).integer_weight(), Some(1));
        assert!(!VoidLabel::is_weighted() && !EdgeLabel::is_weighted());
    }
}
