//! Extensive tests of the [`HyperLogLog`] counters, at every precision: their layout, how
//! elements set registers (against a reference, and with crafted hashes reaching every rank,
//! saturation included), the union's algebra, the estimator (against a straightforward reference
//! implementation of Ertl's improved estimator, on arbitrary register states) and its accuracy,
//! from single elements to `2^62` of them.
//!
//! Accuracy is checked statistically: with real insertions up to millions of elements, and beyond
//! that with registers sampled from the Poisson model the estimator assumes (each register of a
//! counter of `n` elements is at most `k` with probability `exp(-n / m 2^-k)`), which reaches any
//! cardinality at the cost of `m` samples. Bounds account for the number of trials, and every
//! random choice is seeded, so failures reproduce. The `HLL_TEST_SEED` environment variable
//! changes the seeds (and elements), to check the bounds hold for others.
use super::*;

/// Number of trials: as given in release builds, an eighth (at least 16) in debug builds, whose
/// statistical bounds account for it.
fn trials(release: usize) -> usize {
    if cfg!(debug_assertions) {
        (release / 8).max(16)
    } else {
        release
    }
}

/// Seeded pseudo-random numbers, from a mixer other than the counters' hash.
struct Rng(u64);

/// The `HLL_TEST_SEED` environment variable's value (0 if unset).
fn seed() -> u64 {
    std::env::var("HLL_TEST_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self(
            (seed ^ 0x5851_F42D_4C95_7F2D)
                .wrapping_add(self::seed().wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        )
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0xD1B5_4A32_D192_ED03);
        let mut z = self.0;
        z = (z ^ (z >> 27)).wrapping_mul(0x3C79_AC49_2BA7_B653);
        z = (z ^ (z >> 33)).wrapping_mul(0x1C69_B3F7_4AC4_AE35);
        z ^ (z >> 27)
    }

    /// Uniform in `(0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn m<P: Precision>() -> usize {
    HyperLogLog::<P>::REGISTERS
}

fn max_rank<P: Precision>() -> u8 {
    (HyperLogLog::<P>::Q + 1) as u8
}

/// *HyperLogLog*'s relative standard error, `1.04 / sqrt(m)`.
fn std_error<P: Precision>() -> f64 {
    1.04 / (m::<P>() as f64).sqrt()
}

/// The register an element of hash `hash` goes to, and its rank there: the first `EXPONENT` bits
/// select the register, and the rank is one plus the number of leading zeros of the other `Q`
/// bits (`Q + 1` if they're all zero).
fn reference_rank<P: Precision>(hash: u64) -> (usize, u8) {
    let q = HyperLogLog::<P>::Q;
    let index = (hash >> q) as usize;
    let rest = hash & ((1u64 << q) - 1);
    let mut rank = 1;
    while rank <= q && rest & (1u64 << (q - rank)) == 0 {
        rank += 1;
    }
    (index, rank as u8)
}

fn reference_sigma(x: f64) -> f64 {
    if x == 1. {
        return f64::INFINITY;
    }
    // x + sum_{k >= 1} x^(2^k) 2^(k - 1)
    let (mut sum, mut power, mut weight) = (x, x, 0.5);
    loop {
        power *= power;
        weight *= 2.;
        let next = sum + power * weight;
        if next == sum {
            return sum;
        }
        sum = next;
    }
}

fn reference_tau(x: f64) -> f64 {
    if x == 0. || x == 1. {
        return 0.;
    }
    // (1 - x - sum_{k >= 1} (1 - x^(2^-k))^2 2^-k) / 3
    let (mut sum, mut root, mut weight) = (0., x, 1.);
    loop {
        root = root.sqrt();
        weight *= 0.5;
        let next = sum + (1. - root) * (1. - root) * weight;
        if next == sum {
            return (1. - x - sum) / 3.;
        }
        sum = next;
    }
}

/// Ertl's improved estimator (Algorithm 6 of "New cardinality estimation algorithms for
/// HyperLogLog sketches"), from the registers' histogram, with the finite `m` correction of its
/// constant.
fn reference_estimate<P: Precision>(registers: &[u8]) -> f64 {
    let q = HyperLogLog::<P>::Q;
    let m = registers.len() as f64;
    let mut histogram = vec![0usize; q + 2];
    for &r in registers {
        histogram[r as usize] += 1;
    }
    if histogram[0] == registers.len() {
        return 0.;
    }
    let mut z = m * reference_tau(1. - histogram[q + 1] as f64 / m);
    for k in (1..=q).rev() {
        z = 0.5 * (z + histogram[k] as f64);
    }
    z += m * reference_sigma(histogram[0] as f64 / m);
    m * m / (2. * std::f64::consts::LN_2 * (1. + 1.079 / m) * z)
}

/// Whether `a` and `b` are equal up to rounding (both infinite counts as equal).
fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

/// A counter of a Poisson distributed number of elements, of mean `n`: each register is at most
/// `k` (for `k <= Q`) with probability `exp(-n / m 2^-k)`.
fn poisson_counter<P: Precision>(n: f64, rng: &mut Rng) -> HyperLogLog<P> {
    let lambda = n / m::<P>() as f64;
    let mut c = HyperLogLog::<P>::default();
    for r in c.registers.as_mut() {
        // the smallest k with exp(-lambda 2^-k) >= u, from u's exponential -ln(u)
        let e = -rng.unit().ln();
        let k = if e >= lambda {
            0
        } else {
            (lambda / e).log2().ceil() as usize
        };
        *r = k.min(HyperLogLog::<P>::Q + 1) as u8;
    }
    c
}

/// Register states the estimator can meet (and some no insertion makes): uniform values
/// (saturated registers included), mostly zeros, values in a narrow band, the extremes...
fn register_states<P: Precision>(rng: &mut Rng) -> Vec<HyperLogLog<P>> {
    let max = max_rank::<P>();
    let mut states = vec![];
    let mut state = |f: &mut dyn FnMut(usize) -> u8| {
        let mut c = HyperLogLog::<P>::default();
        for (i, r) in c.registers.as_mut().iter_mut().enumerate() {
            *r = f(i);
        }
        states.push(c);
    };
    for v in 0..=max {
        state(&mut |_| v);
    }
    for _ in 0..16 {
        state(&mut |_| (rng.next() % (max as u64 + 1)) as u8);
        state(&mut |_| {
            if rng.below(8) == 0 {
                1 + rng.below(max as usize) as u8
            } else {
                0
            }
        });
        let base = 1 + rng.below(max as usize - 4) as u8;
        state(&mut |_| base + rng.below(4) as u8);
        state(&mut |_| {
            if rng.below(4) == 0 {
                max
            } else {
                rng.below(max as usize) as u8
            }
        });
        state(&mut |i| [0, 1, max][i % 3]);
    }
    // one register set
    state(&mut |i| (i == 0) as u8);
    state(&mut |i| if i + 1 == m::<P>() { max } else { 0 });
    // one register unsaturated
    state(&mut |i| if i == 0 { 1 } else { max });
    for e in [1., 10., 1e3, 1e6, 1e9, 1e12, 1e15, 1e18] {
        states.push(poisson_counter::<P>(e, rng));
    }
    states
}

/// Relative errors' mean (the bias) and root mean square.
fn bias_and_rmse(errors: &[f64]) -> (f64, f64) {
    let n = errors.len() as f64;
    let bias = errors.iter().sum::<f64>() / n;
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    (bias, rmse)
}

/// Asserts `trials` estimates' relative `errors` (each `estimate / n - 1`) are unbiased, but for
/// a bias of at most `bias / m` (the estimator's, at small cardinalities, is `-0.52 / m` to
/// `-1.0 / m`, see `single_elements_estimates_are_exact`),
/// with a root mean square of at most `limit` standard errors --- up to the trials' sampling
/// error, at more than 5 standard deviations of it.
fn assert_accurate<P: Precision>(what: &str, errors: &[f64], bias_limit: f64, limit: f64) {
    let t = errors.len() as f64;
    let (bias, rmse) = bias_and_rmse(errors);
    let mean_error = rmse / t.sqrt();
    assert!(
        bias.abs() <= bias_limit / m::<P>() as f64 + 5.5 * mean_error,
        "precision {}, {what}: bias {bias:+.3e} ({:+.2} / m) over {t} trials, rmse {rmse:.3e}",
        P::EXPONENT,
        bias * m::<P>() as f64
    );
    // the mean square of t errors of variance s^2 is s^2 chi^2_t / t: above
    // s^2 (1 + 6 sqrt(2 / t)) with a negligible probability
    let slack = (1. + 6. * (2. / t).sqrt()).sqrt();
    assert!(
        rmse <= limit * std_error::<P>() * slack,
        "precision {}, {what}: rmse {rmse:.3e} = {:.3} standard errors over {t} trials (limit {limit} * {slack:.3})",
        P::EXPONENT,
        rmse / std_error::<P>()
    );
}

/// Asserts a single `estimate` of `n` elements is within 6 standard errors --- on a log scale:
/// with few registers, estimates' errors are skewed (more likely far above than far below), but
/// their logarithms' much less so.
fn assert_within<P: Precision>(estimate: f64, n: f64, what: &str) {
    let error = (estimate / n).ln();
    assert!(
        error.abs() <= 6. * std_error::<P>(),
        "precision {}, {what}: estimated {estimate} for {n} (log ratio {error:+.3}, standard error {:.3})",
        P::EXPONENT,
        std_error::<P>()
    );
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

// --- tests at every precision ---

fn layout<P: Precision>() {
    assert_eq!(HyperLogLog::<P>::REGISTERS, 1 << P::EXPONENT);
    assert_eq!(std::mem::size_of::<HyperLogLog<P>>(), m::<P>());
    assert_eq!(std::mem::align_of::<HyperLogLog<P>>(), 1);
    assert_eq!(HyperLogLog::<P>::Q, 64 - P::EXPONENT);
    // ranks fit the registers, and the `2^-r` weights an `f64`
    assert!((max_rank::<P>() as u32) < 1023);
    assert_eq!(HyperLogLog::<P>::default().registers().len(), m::<P>());
    // the counters' slice view is the counters' registers, one after another
    let mut counters = vec![HyperLogLog::<P>::default(); 3];
    counters[1].insert(7);
    counters[2].insert(8);
    let expected: Vec<u8> = counters
        .iter()
        .flat_map(|c| c.registers().to_vec())
        .collect();
    let bytes = HyperLogLog::<P>::registers_of_mut(&mut counters);
    assert_eq!(bytes, expected.as_slice());
    bytes[m::<P>() + 1] = 5;
    assert_eq!(counters[1].registers()[1], 5);
    assert_eq!(HyperLogLog::<P>::registers_of_mut(&mut []).len(), 0);
}

fn empty<P: Precision>() {
    let c = HyperLogLog::<P>::default();
    assert!(c.registers().iter().all(|&r| r == 0));
    assert_eq!(c.estimate_cardinality(), 0.);
    assert_eq!(c, HyperLogLog::<P>::default());
    let mut a = c;
    assert!(!a.union(&c), "empty with empty");
    let one = HyperLogLog::<P>::with_element(1);
    assert!(a.union(&one), "empty with an element");
    assert_eq!(a, one);
    let mut b = one;
    assert!(!b.union(&c), "an element with empty");
    assert_eq!(b, one);
    assert!(format!("{c:?}").contains(&format!("precision: {}", P::EXPONENT)));
}

fn crafted_hashes<P: Precision>() {
    let q = HyperLogLog::<P>::Q;
    let max = max_rank::<P>();
    let last = m::<P>() - 1;
    for index in [0, 1, m::<P>() / 2, last] {
        let top = (index as u64) << q;
        // every rank, from a single one bit at each position
        for k in 0..q {
            let mut c = HyperLogLog::<P>::default();
            c.insert_hash(top | 1 << (q - 1 - k));
            assert_eq!(
                c.registers()[index],
                k as u8 + 1,
                "index {index}, rank {}",
                k + 1
            );
            assert_eq!(c.registers().iter().filter(|&&r| r != 0).count(), 1);
            // the lower bits don't matter
            let mut d = HyperLogLog::<P>::default();
            d.insert_hash(top | 1 << (q - 1 - k) | ((1 << (q - 1 - k)) - 1));
            assert_eq!(c, d);
        }
        // all `Q` bits zero: saturated
        let mut c = HyperLogLog::<P>::default();
        c.insert_hash(top);
        assert_eq!(c.registers()[index], max, "index {index}, saturated");
        // a lower rank never lowers a register
        c.insert_hash(top | 1 << (q - 1));
        assert_eq!(c.registers()[index], max);
    }
    let mut c = HyperLogLog::<P>::default();
    c.insert_hash(u64::MAX);
    assert_eq!(c.registers()[last], 1);
    let mut c = HyperLogLog::<P>::default();
    c.insert_hash(0);
    assert_eq!(c.registers()[0], max);
    assert!(c.registers()[1..].iter().all(|&r| r == 0));
}

fn insert_matches_reference<P: Precision>() {
    let mut rng = Rng::new(P::EXPONENT as u64);
    let mut counter = HyperLogLog::<P>::default();
    let mut expected = vec![0u8; m::<P>()];
    let extremes = [
        0,
        1,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) - 1,
        u32::MAX as u64,
    ];
    let elements = extremes
        .into_iter()
        .chain((0..trials(200_000)).map(|_| rng.next()));
    for (i, e) in elements.enumerate() {
        counter.insert(e);
        let (index, rank) = reference_rank::<P>(HyperLogLog::<P>::hash(e));
        expected[index] = expected[index].max(rank);
        if i % 997 == 0 {
            assert_eq!(
                counter.registers(),
                expected.as_slice(),
                "after {i} elements"
            );
        }
    }
    assert_eq!(counter.registers(), expected.as_slice());
    assert!(counter.registers().iter().all(|&r| r <= max_rank::<P>()));
}

fn duplicates_and_order<P: Precision>() {
    let mut rng = Rng::new(10 + P::EXPONENT as u64);
    let elements: Vec<u64> = (0..5 * m::<P>().min(4096)).map(|_| rng.next()).collect();
    let mut forwards = HyperLogLog::<P>::default();
    for &e in &elements {
        forwards.insert(e);
    }
    let once = forwards;
    // again, backwards, shuffled: the same counter
    for &e in elements.iter().rev() {
        forwards.insert(e);
    }
    assert_eq!(forwards, once, "inserting elements again");
    let mut shuffled = elements.clone();
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, rng.below(i + 1));
    }
    let mut c = HyperLogLog::<P>::default();
    for &e in &shuffled {
        c.insert(e);
    }
    assert_eq!(c, once, "inserting elements in another order");
    for &e in elements.iter().take(100) {
        let mut d = HyperLogLog::<P>::default();
        d.insert(e);
        assert_eq!(HyperLogLog::<P>::with_element(e), d);
        // and copies are independent
        let mut copy = d;
        copy.insert(e ^ 1);
        copy.insert(e ^ 2);
        assert_eq!(d, HyperLogLog::<P>::with_element(e));
    }
}

fn union_algebra<P: Precision>() {
    let mut rng = Rng::new(20 + P::EXPONENT as u64);
    let mut counter = |n: usize| {
        let mut c = HyperLogLog::<P>::default();
        let elements: Vec<u64> = (0..n).map(|_| rng.next() % 1_000_000).collect();
        for &e in &elements {
            c.insert(e);
        }
        (c, elements)
    };
    let sizes = [0, 1, 5, m::<P>() / 2, 3 * m::<P>()];
    for &x in &sizes {
        for &y in &sizes {
            let ((a, ea), (b, eb), (c, _)) = (counter(x), counter(y), counter(x + y));
            // the union is the counter of both sets, register-wise maximums
            let mut ab = a;
            let changed = ab.union(&b);
            let mut both = HyperLogLog::<P>::default();
            for &e in ea.iter().chain(&eb) {
                both.insert(e);
            }
            assert_eq!(ab, both, "{x} and {y} elements");
            for i in 0..m::<P>() {
                assert_eq!(ab.registers()[i], a.registers()[i].max(b.registers()[i]));
            }
            assert_eq!(
                changed,
                ab != a,
                "whether registers changed, {x} and {y} elements"
            );
            // commutative, idempotent, associative
            let mut ba = b;
            ba.union(&a);
            assert_eq!(ab, ba);
            let mut again = ab;
            assert!(!again.union(&a) && !again.union(&b) && !again.union(&ab));
            assert_eq!(again, ab);
            let (mut left, mut right, mut bc) = (ab, a, b);
            left.union(&c);
            bc.union(&c);
            right.union(&bc);
            assert_eq!(left, right, "(a b) c and a (b c)");
            // never lower than either, so neither are estimates
            assert!(
                ab.estimate_cardinality() >= a.estimate_cardinality().max(b.estimate_cardinality())
            );
        }
    }
}

fn union_of_many<P: Precision>() {
    let (counters, per_counter) = (trials(512), 64);
    let mut all = HyperLogLog::<P>::default();
    let mut union = HyperLogLog::<P>::default();
    for c in 0..counters {
        let mut counter = HyperLogLog::<P>::default();
        for e in 0..per_counter {
            // overlapping ranges: every other element is also in the next counter
            let element = (c * per_counter / 2 + e) as u64;
            counter.insert(element);
            all.insert(element);
        }
        union.union(&counter);
    }
    assert_eq!(union, all);
    let n = ((counters + 1) * per_counter / 2) as f64;
    assert_within::<P>(
        union.estimate_cardinality(),
        n,
        &format!("union of {counters} counters"),
    );
}

/// The SSE2 union (on x86_64) is the portable one: the same registers, and whether they changed.
fn union_paths_agree<P: Precision>() {
    let mut rng = Rng::new(35 + P::EXPONENT as u64);
    let states = register_states::<P>(&mut rng);
    for a in &states {
        for b in states.iter().step_by(7) {
            let (mut fast, mut portable) = (*a, *a);
            let changed = fast.union(b);
            let portable_changed = union_portable(portable.registers.as_mut(), b.registers());
            assert_eq!(fast, portable);
            assert_eq!(changed, portable_changed);
            // and once more, now unchanged
            assert!(!fast.union(b));
        }
    }
}

fn stats_paths_agree<P: Precision>() {
    let mut rng = Rng::new(30 + P::EXPONENT as u64);
    for c in register_states::<P>(&mut rng) {
        // the same sum (in the same order) and counts, bit for bit
        let (fused, split) = (c.register_stats_fused(), c.register_stats_split());
        assert_eq!(fused.0.to_bits(), split.0.to_bits(), "sums");
        assert_eq!(
            (fused.1, fused.2),
            (split.1, split.2),
            "zero and saturated counts"
        );
        let max = max_rank::<P>();
        let zeros = c.registers().iter().filter(|&&r| r == 0).count() as u32;
        let saturated = c.registers().iter().filter(|&&r| r == max).count() as u32;
        assert_eq!((fused.1, fused.2), (zeros, saturated));
        let sum: f64 = c.registers().iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        assert!(close(fused.0, sum) || (fused.0 - sum).abs() <= 1e-15 * m::<P>() as f64);
    }
}

fn estimate_matches_reference<P: Precision>() {
    let mut rng = Rng::new(40 + P::EXPONENT as u64);
    for c in register_states::<P>(&mut rng) {
        let (estimate, reference) = (
            c.estimate_cardinality(),
            reference_estimate::<P>(c.registers()),
        );
        assert!(
            close(estimate, reference),
            "estimate {estimate}, reference {reference} of {c:?}"
        );
    }
    // from actual insertions too
    let mut c = HyperLogLog::<P>::default();
    for e in 0..trials(1 << 18) as u64 {
        c.insert(e);
        if e.is_power_of_two() {
            assert!(
                close(
                    c.estimate_cardinality(),
                    reference_estimate::<P>(c.registers())
                ),
                "{} elements",
                e + 1
            );
        }
    }
}

fn estimates_well_defined<P: Precision>() {
    let mut rng = Rng::new(50 + P::EXPONENT as u64);
    let max = max_rank::<P>();
    for c in register_states::<P>(&mut rng) {
        let estimate = c.estimate_cardinality();
        let saturated = c.registers().iter().all(|&r| r == max);
        let empty = c.registers().iter().all(|&r| r == 0);
        assert!(!estimate.is_nan());
        if saturated {
            // every register saturated: no finite cardinality explains them
            assert_eq!(estimate, f64::INFINITY);
        } else if empty {
            assert_eq!(estimate, 0.);
        } else {
            assert!(
                estimate.is_finite() && estimate > 0.,
                "estimate {estimate} of {c:?}"
            );
        }
    }
    // a single saturated register (and no other): like any single element's
    let mut c = HyperLogLog::<P>::default();
    c.insert_hash(0);
    let estimate = c.estimate_cardinality();
    assert!((estimate - 1.).abs() <= 0.1, "{estimate}");
}

fn estimate_monotone<P: Precision>() {
    let mut rng = Rng::new(60 + P::EXPONENT as u64);
    let max = max_rank::<P>();
    for c in register_states::<P>(&mut rng) {
        let estimate = c.estimate_cardinality();
        for _ in 0..8 {
            let i = rng.below(m::<P>());
            if c.registers()[i] == max {
                continue;
            }
            // raising any register (an element's insertion does) never lowers the estimate ---
            // but for rounding: raising a register from `r` lowers `sum(2^-r)` by `2^-(r + 1)`,
            // which can be below the sum's precision
            let mut raised = c;
            raised.registers.as_mut()[i] += 1;
            let after = raised.estimate_cardinality();
            assert!(
                after >= estimate * (1. - 8. * f64::EPSILON),
                "raising register {i} from {}: {estimate} to {after}",
                c.registers()[i]
            );
        }
    }
}

/// Real insertions, from single elements to a few times as many as registers.
fn accuracy_small<P: Precision>() {
    let m = m::<P>();
    let mut next = seed() << 40;
    for n in [1, 2, 3, 5, 10, 30, 100, m / 4, m / 2, m, 2 * m, 4 * m] {
        if n == 0 {
            continue;
        }
        // about 2^20 insertions and 2^30 register reads per size, at least 400 trials
        let t = trials(((1 << 20) / n).min((1 << 30) / (m * n)).clamp(400, 20_000));
        let counters = (0..t).map(|_| {
            let mut c = HyperLogLog::<P>::default();
            for _ in 0..n {
                next += 1;
                c.insert(next);
            }
            c
        });
        if n * 16 > m {
            let errors: Vec<f64> = counters
                .map(|c| c.estimate_cardinality() / n as f64 - 1.)
                .collect();
            assert_accurate::<P>(&format!("{n} inserted elements"), &errors, 1.05, 1.1);
            continue;
        }
        // few elements in many registers: elements rarely share a register, so errors are mostly
        // the estimator's (tiny) one of `k` elements in as many registers, and rarely that of
        // (rare) collisions, too rare to be sampled well. So each estimate is checked to be that
        // of the registers set, and collisions to be as frequent as expected
        let mut collisions = 0;
        for c in counters {
            let k = c.registers().iter().filter(|&&r| r != 0).count();
            collisions += (k < n) as usize;
            let estimate = c.estimate_cardinality();
            // like linear counting, over by about k / 2m without collisions
            assert!(
                (estimate / k as f64 - 1.).abs() <= (1.05 + k as f64) / m as f64,
                "precision {}, {n} inserted elements in {k} registers: estimated {estimate}",
                P::EXPONENT
            );
        }
        // n elements share a register with probability 1 - prod_{i < n} (1 - i / m)
        let p = 1. - (0..n).map(|i| 1. - i as f64 / m as f64).product::<f64>();
        let expected = t as f64 * p;
        assert!(
            (collisions as f64) <= expected + 6. * expected.sqrt() + 4.,
            "precision {}, {n} inserted elements: {collisions} of {t} trials had collisions, {expected:.2} expected",
            P::EXPONENT
        );
        if expected >= 25. {
            assert!(
                (collisions as f64) >= expected - 6. * expected.sqrt(),
                "precision {}, {n} elements: {collisions} collisions, {expected:.2} expected",
                P::EXPONENT
            );
        }
    }
}

/// Real insertions, up to millions of elements.
fn accuracy_large<P: Precision>() {
    let mut next = seed() << 40;
    for (n, t) in [(100_000, 40), (1_000_000, 16), (4_000_000, 4)] {
        let t = if cfg!(debug_assertions) {
            (t / 4).max(2)
        } else {
            t
        };
        // few trials: each within bounds
        for _ in 0..t {
            let mut c = HyperLogLog::<P>::default();
            for _ in 0..n {
                next += 1;
                c.insert(next);
            }
            assert_within::<P>(c.estimate_cardinality(), n as f64, &format!("{n} elements"));
        }
    }
}

/// Registers sampled from the Poisson model, for cardinalities up to `2^62`.
fn accuracy_huge<P: Precision>() {
    let m = m::<P>();
    let mut rng = Rng::new(70 + P::EXPONENT as u64);
    let t = trials(((1 << 21) / m).clamp(32, 4096));
    // from where the Poisson count's own noise (1 / sqrt(n)) is negligible
    let first = (P::EXPONENT + 10).max(14);
    for k in (first..=62).step_by(2) {
        let n = (k as f64).exp2();
        let errors: Vec<f64> = (0..t)
            .map(|_| poisson_counter::<P>(n, &mut rng).estimate_cardinality() / n - 1.)
            .collect();
        assert_accurate::<P>(
            &format!("2^{k} elements (Poisson model)"),
            &errors,
            1.05,
            1.1,
        );
    }
}

/// The counters hold up to `2^64` distinct hashes: past that, every register saturates.
fn saturation<P: Precision>() {
    let mut c = HyperLogLog::<P>::default();
    for index in 0..m::<P>() as u64 {
        c.insert_hash(index << HyperLogLog::<P>::Q);
    }
    assert!(c.registers().iter().all(|&r| r == max_rank::<P>()));
    assert_eq!(c.estimate_cardinality(), f64::INFINITY);
    // registers past the largest rank never occur, but a saturated one plus finite ones still
    // estimates at most 2^64 per register
    let mut rng = Rng::new(80 + P::EXPONENT as u64);
    let near = poisson_counter::<P>(2f64.powi(62), &mut rng);
    let estimate = near.estimate_cardinality();
    assert!(estimate.is_finite());
    assert_within::<P>(estimate, 2f64.powi(62), "2^62 elements (Poisson model)");
}

/// Register indices of consecutive elements (as node ids are) are uniform, and ranks geometric.
fn hash_distribution<P: Precision>() {
    let m = m::<P>();
    let n = 32 * m.min(1 << 14);
    let mut counts = vec![0usize; m];
    let mut ranks = [0usize; 12];
    for e in 0..n as u64 {
        let (index, rank) = reference_rank::<P>(HyperLogLog::<P>::hash(e));
        counts[index] += 1;
        if (rank as usize) < ranks.len() {
            ranks[rank as usize] += 1;
        }
    }
    let expected = n as f64 / m as f64;
    let chi2: f64 = counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum();
    let dof = (m - 1) as f64;
    assert!(
        (chi2 - dof).abs() <= 6. * (2. * dof).sqrt(),
        "register indices: chi^2 {chi2} for {dof} degrees of freedom"
    );
    for (k, &count) in ranks.iter().enumerate().skip(1) {
        let p = 0.5f64.powi(k as i32);
        let sd = (n as f64 * p * (1. - p)).sqrt();
        assert!(
            (count as f64 - n as f64 * p).abs() <= 6. * sd,
            "rank {k}: {count} of {n}"
        );
    }
}

/// The registers of `0..10_000` and `u64::MAX - 10_000..` never change unnoticed: HyperBall
/// results, and counters cached by earlier runs, depend on them.
fn golden<P: Precision>(expected: u64) {
    let mut c = HyperLogLog::<P>::default();
    for e in (0..10_000).chain(u64::MAX - 10_000..=u64::MAX) {
        c.insert(e);
    }
    assert_eq!(
        checksum(c.registers()),
        expected,
        "precision {}: registers changed",
        P::EXPONENT
    );
}

macro_rules! at_every_precision {
    ($($precision:ident => $module:ident, $golden:literal),* $(,)?) => {$(
        mod $module {
            type P = super::super::$precision;

            #[test]
            fn layout() { super::layout::<P>() }
            #[test]
            fn empty() { super::empty::<P>() }
            #[test]
            fn crafted_hashes() { super::crafted_hashes::<P>() }
            #[test]
            fn insert_matches_reference() { super::insert_matches_reference::<P>() }
            #[test]
            fn duplicates_and_order() { super::duplicates_and_order::<P>() }
            #[test]
            fn union_algebra() { super::union_algebra::<P>() }
            #[test]
            fn union_of_many() { super::union_of_many::<P>() }
            #[test]
            fn union_paths_agree() { super::union_paths_agree::<P>() }
            #[test]
            fn stats_paths_agree() { super::stats_paths_agree::<P>() }
            #[test]
            fn estimate_matches_reference() { super::estimate_matches_reference::<P>() }
            #[test]
            fn estimates_well_defined() { super::estimates_well_defined::<P>() }
            #[test]
            fn estimate_monotone() { super::estimate_monotone::<P>() }
            #[test]
            fn accuracy_small() { super::accuracy_small::<P>() }
            #[test]
            fn accuracy_large() { super::accuracy_large::<P>() }
            #[test]
            fn accuracy_huge() { super::accuracy_huge::<P>() }
            #[test]
            fn saturation() { super::saturation::<P>() }
            #[test]
            fn hash_distribution() { super::hash_distribution::<P>() }
            #[test]
            fn golden() { super::golden::<P>($golden) }
        }
    )*};
}

at_every_precision!(
    Precision4 => p4, 13071996068041054661,
    Precision5 => p5, 12034139835738894441,
    Precision6 => p6, 16075990590333073412,
    Precision7 => p7, 3038473368787456779,
    Precision8 => p8, 17910542254325991087,
    Precision9 => p9, 10669160143947383258,
    Precision10 => p10, 15803265356437874507,
    Precision11 => p11, 4349093447625865938,
    Precision12 => p12, 11600529316297176744,
    Precision13 => p13, 17877404252604326891,
    Precision14 => p14, 12732170570802395933,
    Precision15 => p15, 1101110506967738717,
    Precision16 => p16, 3155549312282747321,
);

// --- tests independent of the precision ---

#[test]
fn sigma_and_tau_match_their_series() {
    type H = HyperLogLog<Precision8>;
    // computed to 60 digits
    for (x, sigma, tau) in [
        (0., 0., 0.),
        (0.0625, 6.643_676_850_944_757e-2, 1.937_323_739_660_220_3e-1),
        (0.25, 3.203_735_370_188_951_5e-1, 1.999_647_479_320_440_5e-1),
        (0.5, 8.907_470_740_377_903e-1, 1.499_294_958_640_881e-1),
        (0.75, 2.427_552_487_930_809_5, 7.988_094_143_733_783e-2),
        (
            0.9375,
            1.115_641_017_017_750_9e1,
            2.064_085_584_977_939_6e-2,
        ),
        (0.999, 7.209_897_350_836_828e2, 3.332_856_888_722_87e-4),
    ] {
        assert!(
            (H::sigma(x) - sigma).abs() <= 1e-14 * sigma.max(1.),
            "sigma({x}) = {}",
            H::sigma(x)
        );
        assert!((H::tau(x) - tau).abs() <= 1e-14, "tau({x}) = {}", H::tau(x));
        assert!(
            close(H::sigma(x), reference_sigma(x)) && (H::tau(x) - reference_tau(x)).abs() <= 1e-15
        );
    }
    assert_eq!(H::sigma(1.), f64::INFINITY);
    assert_eq!(H::tau(1.), 0.);
}

#[test]
fn hash_is_a_mixing_bijection() {
    type H = HyperLogLog<Precision8>;
    // SplitMix64's finalizer: the first outputs of its reference generator (seed 0)
    assert_eq!(H::hash(0), 0xE220_A839_7B1D_CDAF);
    assert_eq!(H::hash(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    let mut seen = std::collections::HashSet::new();
    for e in (0..1u64 << 16)
        .chain((1..1u64 << 16).map(|e| e << 48))
        .chain(u64::MAX - 1000..=u64::MAX)
    {
        assert!(
            seen.insert(H::hash(e)),
            "two elements hash to {}",
            H::hash(e)
        );
    }
    // flipping any input bit flips half of the output bits on average
    let mut rng = Rng::new(90);
    let samples = 2048;
    for bit in 0..64 {
        let flipped: u32 = (0..samples)
            .map(|_| {
                let e = rng.next();
                (H::hash(e) ^ H::hash(e ^ 1 << bit)).count_ones()
            })
            .sum();
        let mean = flipped as f64 / samples as f64;
        // 32 +- 6 standard deviations (4 bits each, over the samples)
        assert!(
            (mean - 32.).abs() <= 6. * 4. / (samples as f64).sqrt(),
            "input bit {bit}: {mean} output bits flip"
        );
    }
}

#[test]
fn single_elements_estimates_are_exact() {
    // a single element's estimate depends only on its rank r (2^-r likely): their mean, computed to
    // 80 digits (from sigma's series), is the estimator's bias at the smallest cardinality
    fn check<P: Precision>(exact: f64) {
        let mean: f64 = (1..=HyperLogLog::<P>::Q as i32)
            .map(|r| {
                let mut c = HyperLogLog::<P>::default();
                c.insert_hash(1 << (HyperLogLog::<P>::Q as i32 - r));
                0.5f64.powi(r) * c.estimate_cardinality()
            })
            .sum();
        assert!(
            (mean - exact).abs() <= 1e-12,
            "precision {}: {mean}, exactly {exact}",
            P::EXPONENT
        );
    }
    check::<Precision4>(0.9673592502470956);
    check::<Precision8>(0.9977466459500857);
    check::<Precision12>(0.9998523557540214);
    check::<Precision14>(0.9999583133353622);
    check::<Precision15>(0.9999759791766756);
    check::<Precision16>(0.9999848127665121);
}
