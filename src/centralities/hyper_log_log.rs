//! *HyperLogLog* counters tailored to *HyperBall*.
//!
//! Each register is stored in its own byte (rather than bit-packed), so that the union of two
//! counters --- the *HyperBall* hot loop --- is a plain element-wise `max` over byte arrays, which
//! the compiler vectorizes. The same flat layout is directly usable by GPU kernels.
//!
//! Cardinalities are estimated with the *improved estimator* described in ["New cardinality
//! estimation algorithms for HyperLogLog sketches"](https://doi.org/10.48550/arXiv.1702.01284)
//! by Ertl O., which is accurate across the whole cardinality range without empirical bias
//! correction tables (with the usual finite register count correction of its constant, which
//! matters for few registers).
use std::fmt::Debug;

/// Number of registers (`2^EXPONENT`) of a [`HyperLogLog`] counter.
pub trait Precision: Copy + Debug + Send + Sync + 'static {
    /// Base-2 logarithm of the number of registers.
    const EXPONENT: usize;
    /// Register array, one byte per register.
    type Registers: Copy + Debug + Send + Sync + AsRef<[u8]> + AsMut<[u8]>;

    fn zeroed() -> Self::Registers;
}

macro_rules! precisions {
    ($($name:ident => $exp:literal),* $(,)?) => {
        $(
            #[doc = concat!("For *HyperLogLog* counters with 2^", stringify!($exp), " registers.")]
            #[derive(Clone, Copy, Debug)]
            pub struct $name;

            impl Precision for $name {
                const EXPONENT: usize = $exp;
                type Registers = [u8; 1 << $exp];

                #[inline(always)]
                fn zeroed() -> Self::Registers {
                    [0; 1 << $exp]
                }
            }
        )*
    };
}

precisions!(
    Precision4 => 4,
    Precision5 => 5,
    Precision6 => 6,
    Precision7 => 7,
    Precision8 => 8,
    Precision9 => 9,
    Precision10 => 10,
    Precision11 => 11,
    Precision12 => 12,
    Precision13 => 13,
    Precision14 => 14,
    Precision15 => 15,
    Precision16 => 16,
);

/// *HyperLogLog* counter with `2^P::EXPONENT` byte-sized registers.
#[repr(transparent)]
pub struct HyperLogLog<P: Precision> {
    registers: P::Registers,
}

impl<P: Precision> Clone for HyperLogLog<P> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Precision> Copy for HyperLogLog<P> {}

impl<P: Precision> Debug for HyperLogLog<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HyperLogLog")
            .field("precision", &P::EXPONENT)
            .field("estimate", &self.estimate_cardinality())
            .finish()
    }
}

impl<P: Precision> Default for HyperLogLog<P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            registers: P::zeroed(),
        }
    }
}

impl<P: Precision> PartialEq for HyperLogLog<P> {
    fn eq(&self, other: &Self) -> bool {
        self.registers.as_ref() == other.registers.as_ref()
    }
}

impl<P: Precision> Eq for HyperLogLog<P> {}

impl<P: Precision> HyperLogLog<P> {
    /// Number of registers.
    pub const REGISTERS: usize = 1 << P::EXPONENT;
    /// Number of hash bits left for the rank after the register index is taken (`q` in Ertl's
    /// notation) --- register values range over `0..=Q + 1`.
    const Q: usize = 64 - P::EXPONENT;

    /// Counter holding a single element.
    #[inline(always)]
    pub fn with_element(element: u64) -> Self {
        let mut c = Self::default();
        c.insert(element);
        c
    }

    /// Adds `element` to the counter.
    #[inline(always)]
    pub fn insert(&mut self, element: u64) {
        let hash = Self::hash(element);
        let index = (hash >> Self::Q) as usize;
        // the remaining `Q` bits, left aligned, padded with ones so that the rank never exceeds
        // `Q + 1`
        let rank = ((hash << P::EXPONENT) | (1 << (P::EXPONENT - 1))).leading_zeros() as u8 + 1;
        let register = &mut self.registers.as_mut()[index];
        *register = (*register).max(rank);
    }

    /// Merges `other` into `self` (set union), returning whether any register changed.
    #[inline(always)]
    pub fn union(&mut self, other: &Self) -> bool {
        let mut changed = 0u8;
        for (a, &b) in self
            .registers
            .as_mut()
            .iter_mut()
            .zip(other.registers.as_ref())
        {
            changed |= (b > *a) as u8;
            *a = (*a).max(b);
        }
        changed != 0
    }

    /// The counter's registers.
    #[inline(always)]
    pub fn registers(&self) -> &[u8] {
        self.registers.as_ref()
    }

    /// Views `counters` as their registers, counter after counter (`REGISTERS` bytes each) ---
    /// the layout GPU backends work on.
    pub fn registers_of_mut(counters: &mut [Self]) -> &mut [u8] {
        // SAFETY: `HyperLogLog` is `repr(transparent)` over a `[u8; REGISTERS]`, so a slice of
        // counters is a contiguous, initialized, `u8` aligned run of `len * REGISTERS` bytes
        unsafe {
            std::slice::from_raw_parts_mut(
                counters.as_mut_ptr() as *mut u8,
                counters.len() * Self::REGISTERS,
            )
        }
    }

    /// Estimates the number of distinct elements added to the counter.
    pub fn estimate_cardinality(&self) -> f64 {
        const LANES: usize = 8;
        let max = (Self::Q + 1) as u8;

        // a single pass computing sum(2^-r), and the number of zero (`C_0`) and saturated
        // (`C_{Q+1}`) registers, in independent lanes so that it vectorizes
        let mut sum = [0f64; LANES];
        let mut zeros = [0u32; LANES];
        let mut saturated = [0u32; LANES];
        for chunk in self.registers.as_ref().chunks_exact(LANES) {
            for l in 0..LANES {
                let r = chunk[l];
                sum[l] += f64::from_bits((1023 - r as u64) << 52);
                zeros[l] += (r == 0) as u32;
                saturated[l] += (r == max) as u32;
            }
        }
        let sum: f64 = sum.iter().sum();
        let zeros = zeros.iter().sum::<u32>() as f64;
        let saturated = saturated.iter().sum::<u32>() as f64;

        let m = Self::REGISTERS as f64;
        if zeros == m {
            return 0.;
        }

        // unrolled `z = m * tau(1 - C_{q+1} / m); for k = q..=1 { z = (z + C_k) / 2 }` plus
        // `m * sigma(C_0 / m)`, where the loop amounts to sum(C_k * 2^-k) for k in 1..=q
        let max_weight = f64::from_bits((1023 - max as u64) << 52);
        let middle = sum - zeros - saturated * max_weight;
        let z = m * Self::tau(1. - saturated / m) * (2. * max_weight)
            + middle
            + m * Self::sigma(zeros / m);

        // Ertl's estimator uses the asymptotic `alpha_inf = 1 / (2 ln 2)`, which overestimates by
        // ~7% for m = 16 (~2% for m = 64), so the usual finite-`m` correction
        // `alpha_m = alpha_inf / (1 + 1.079 / m)` is applied
        m * m / (2. * std::f64::consts::LN_2 * (1. + 1.079 / m) * z)
    }

    #[inline(always)]
    fn hash(element: u64) -> u64 {
        // SplitMix64 finalizer
        let mut z = element.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sigma(mut x: f64) -> f64 {
        if x == 1. {
            return f64::INFINITY;
        }
        let mut y = 1.;
        let mut z = x;
        loop {
            x *= x;
            let prev = z;
            z += x * y;
            y += y;
            if z == prev {
                return z;
            }
        }
    }

    fn tau(mut x: f64) -> f64 {
        if x == 0. || x == 1. {
            return 0.;
        }
        let mut y = 1.;
        let mut z = 1. - x;
        loop {
            x = x.sqrt();
            let prev = z;
            y *= 0.5;
            z -= (1. - x) * (1. - x) * y;
            if z == prev {
                return z / 3.;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Asserts estimates over the whole cardinality range (small cardinalities, where most
    /// registers are zero, through large ones) stay within a few standard errors
    /// (`1.04 / sqrt(m)`).
    fn estimates_within_error<P: Precision>() {
        let m = HyperLogLog::<P>::REGISTERS as f64;
        let tolerance = 4. * 1.04 / m.sqrt();
        for n in [1u64, 2, 3, 10, 100, 1_000, 10_000, 100_000, 1_000_000] {
            let mut c = HyperLogLog::<P>::default();
            for e in 0..n {
                c.insert(e);
            }
            let estimate = c.estimate_cardinality();
            let error = (estimate - n as f64).abs() / n as f64;
            assert!(
                error <= tolerance || (estimate - n as f64).abs() < 0.5,
                "precision {}: estimated {estimate} for {n} elements (relative error {error})",
                P::EXPONENT
            );
        }
    }

    #[test]
    fn hyperloglog_estimates_within_error() {
        estimates_within_error::<Precision4>();
        estimates_within_error::<Precision6>();
        estimates_within_error::<Precision8>();
        estimates_within_error::<Precision12>();
        estimates_within_error::<Precision16>();
    }

    #[test]
    fn hyperloglog_empty_is_zero() {
        assert_eq!(
            HyperLogLog::<Precision8>::default().estimate_cardinality(),
            0.
        );
    }

    #[test]
    fn hyperloglog_union_matches_inserting_both() {
        let mut a = HyperLogLog::<Precision8>::default();
        let mut b = HyperLogLog::<Precision8>::default();
        let mut both = HyperLogLog::<Precision8>::default();
        for e in 0..500 {
            a.insert(e);
            both.insert(e);
        }
        for e in 300..1_000 {
            b.insert(e);
            both.insert(e);
        }
        assert!(
            a.union(&b),
            "union with new elements should change registers"
        );
        assert_eq!(a, both);
        assert!(!a.union(&b), "repeated union should not change registers");
        assert!(!a.clone().union(&HyperLogLog::default()));
    }

    #[test]
    fn hyperloglog_rank_never_exceeds_max() {
        let max = (HyperLogLog::<Precision16>::Q + 1) as u8;
        let mut c = HyperLogLog::<Precision16>::default();
        for e in 0..1_000_000 {
            c.insert(e);
        }
        assert!(c.registers().iter().all(|&r| r <= max));
    }

    /// Estimates `trials` disjoint sets of `n` elements each, returning (mean relative bias,
    /// relative root mean square error).
    fn bias_and_rmse<P: Precision>(n: u64, trials: u64) -> (f64, f64) {
        let (mut bias, mut square) = (0., 0.);
        for trial in 0..trials {
            let mut c = HyperLogLog::<P>::default();
            for e in 0..n {
                c.insert(trial * n + e);
            }
            let error = (c.estimate_cardinality() - n as f64) / n as f64;
            bias += error;
            square += error * error;
        }
        (bias / trials as f64, (square / trials as f64).sqrt())
    }

    /// Asserts the estimator is unbiased (mean error within 4 standard errors of the mean) and
    /// its root mean square error is at most *HyperLogLog*'s theoretical `1.04 / sqrt(m)` (with
    /// some slack), from the small range (where most registers are zero) to the large one.
    fn unbiased_with_expected_rmse<P: Precision>() {
        const TRIALS: u64 = 400;
        let std_error = 1.04 / (HyperLogLog::<P>::REGISTERS as f64).sqrt();
        let m = HyperLogLog::<P>::REGISTERS as u64;
        for n in [m / 4, m, 3 * m, 10 * m, 100 * m] {
            let (bias, rmse) = bias_and_rmse::<P>(n, TRIALS);
            assert!(
                bias.abs() <= 4. * std_error / (TRIALS as f64).sqrt(),
                "precision {}, n = {n}: biased estimates (mean relative error {bias})",
                P::EXPONENT
            );
            assert!(
                rmse <= 1.2 * std_error,
                "precision {}, n = {n}: relative rmse {rmse} above 1.2 * {std_error}",
                P::EXPONENT
            );
        }
    }

    #[test]
    fn hyperloglog_unbiased_with_expected_rmse() {
        unbiased_with_expected_rmse::<Precision4>();
        unbiased_with_expected_rmse::<Precision6>();
        unbiased_with_expected_rmse::<Precision8>();
        unbiased_with_expected_rmse::<Precision10>();
    }
}
