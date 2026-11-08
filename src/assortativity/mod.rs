//! Assortativity: how much the nodes linked by edges are alike, as the correlation, over every
//! edge, of a value of the nodes at either end.
//!
//! * [`degree_assortativity`] and [`directed_degree_assortativity`]: of their degrees, Newman's
//!   assortativity coefficient (["Assortative mixing in networks"](https://doi.org/10.1103/PhysRevLett.89.208701),
//!   and ["Mixing patterns in networks"](https://doi.org/10.1103/PhysRevE.67.026126) for
//!   directed graphs, by any combination of in- and out-degree of either end). Either takes a
//!   [`Correlation`]: Pearson's (Newman's coefficient, networkx's
//!   `degree_assortativity_coefficient`), or Spearman's, the Pearson correlation of the degrees'
//!   ranks. On graphs with heavy-tailed degrees, Pearson's coefficient depends on the graph's size
//!   (in disassortative graphs, it goes to zero as graphs grow) and is dominated by the largest
//!   degrees, while Spearman's doesn't: see Litvak and van der Hofstad's ["Uncovering disassortativity
//!   in large scale-free networks"](https://doi.org/10.1103/PhysRevE.87.022801) and van der Hoorn
//!   and Litvak's ["Degree-degree dependencies in directed networks with heavy-tailed degrees"](https://doi.org/10.1080/15427951.2014.927038).
//! * [`numeric_assortativity`]: of a numeric value of each node, networkx's
//!   `numeric_assortativity_coefficient`.
//! * [`attribute_assortativity`]: of a category of each node, networkx's
//!   `attribute_assortativity_coefficient`: `(Σ_i e_ii − Σ_i a_i b_i) / (1 − Σ_i a_i b_i)`,
//!   where `e_ij` is the fraction of arcs from category `i` to category `j`, and `a_i` and `b_i`
//!   the fractions of arcs from and to category `i`.
//!
//! Every coefficient is over the graph's arcs: an undirected edge counts once each way, a
//! self-loop once, and parallel edges each (networkx's `MultiGraph`s; on graphs without parallel
//! edges, its `Graph`s). An undirected node's degree counts its self-loops twice, as networkx's.
//!
//! All stream through the edges once (twice, for [`numeric_assortativity`] on directed graphs),
//! in parallel, reading the value of each arc's head: their cost is that of the reads, scattered
//! over the nodes. Degrees are stored in the narrowest unsigned integer type that holds the
//! largest (one byte per node, for most de Bruijn graphs), for these reads to hit the CPU's
//! caches as often as possible. Degree and attribute coefficients are computed exactly, in
//! integers, and so are the same whatever the threads; [`numeric_assortativity`] sums in a fixed
//! order, and so is the same whatever the threads too.
//!
//! A coefficient is `NaN` where undefined: for graphs without arcs, and when either end's values
//! are all the same (a regular graph's degrees, for instance), as networkx's.
use crate::graph::{self, CacheFile, DiGraph, Directedness, Graph, GraphMemoryMap};
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicU64, AtomicUsize, Ordering};

/// Which degree of a node of a directed graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DegreeType {
    /// Its out-degree: arcs out of it. The default for an arc's tail (as networkx's).
    #[default]
    Out,
    /// Its in-degree: arcs into it. The default for an arc's head (as networkx's).
    In,
}

/// Which correlation of the values at either end of each arc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Correlation {
    /// Pearson's: Newman's assortativity coefficient. The default.
    #[default]
    Pearson,
    /// Spearman's: Pearson's of the values' ranks among the arcs' tails' (or heads') values,
    /// ties ranked by their average rank.
    Spearman,
}

/// Nodes per unit of work: threads take them as they go (sums whose order matters are
/// combined by unit, in order, for the same result whatever the threads).
const CHUNK: usize = 1 << 12;

/// Values from which on histograms are counted in shared atomics rather than per thread.
const LOCAL_HISTOGRAM: usize = 1 << 12;

/// `f` over `0..n` in chunks of [`CHUNK`], taken by `threads` as they go: its results, in
/// chunks' order.
fn chunked<T: Send>(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let chunks = n.div_ceil(CHUNK);
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, T)> = thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|_| {
                    let mut mine = vec![];
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= chunks {
                            break mine;
                        }
                        let begin = c * CHUNK;
                        mine.push((c, f(begin, (begin + CHUNK).min(n))));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })?;
    results.sort_unstable_by_key(|(c, _)| *c);
    Ok(results.into_iter().map(|(_, t)| t).collect())
}

/// `f` over each of `threads` ranges splitting `0..n`, in parallel.
fn in_parallel(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    let threads = threads.max(1);
    let load = n.div_ceil(threads);
    thread::scope(|scope| {
        for tid in 0..threads {
            let begin = (tid * load).min(n);
            let end = (begin + load).min(n);
            scope.spawn(move |_| f(begin, end));
        }
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// `f` over each of `threads` ranges splitting `0..n`, in parallel: its results, by range.
fn by_range<T: Send>(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let threads = threads.max(1);
    let load = n.div_ceil(threads);
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let begin = (tid * load).min(n);
                let end = (begin + load).min(n);
                scope.spawn(move |_| f(begin, end))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// An unsigned integer type node values are stored in.
trait Value: Copy + Default + Send + Sync + 'static {
    const MAX: u64;
    fn widen(self) -> u64;
    fn narrow(v: u64) -> Self;
}

macro_rules! value {
    ($($t:ty),*) => {$(
        impl Value for $t {
            const MAX: u64 = <$t>::MAX as u64;
            #[inline(always)]
            fn widen(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn narrow(v: u64) -> Self {
                v as $t
            }
        }
    )*};
}
value!(u8, u16, u32, u64);

/// A graph's CSR, and its scratch files.
struct Arcs<'a, Ix> {
    threads: usize,
    n: usize,
    offsets: &'a [Ix],
    neighbours: &'a [Ix],
    scratch: &'a (dyn Fn(usize) -> Result<String, Box<dyn std::error::Error>> + Sync),
}

/// The integer sums Pearson's correlation is computed from, over the arcs `u -> v` of values
/// `x(u)` and `y(v)`.
#[derive(Clone, Copy, Default, Debug)]
struct Sums {
    x: u128,
    xx: u128,
    y: u128,
    yy: u128,
    xy: u128,
}

impl std::ops::Add for Sums {
    type Output = Sums;
    fn add(self, o: Sums) -> Sums {
        Sums {
            x: self.x + o.x,
            xx: self.xx + o.xx,
            y: self.y + o.y,
            yy: self.yy + o.yy,
            xy: self.xy + o.xy,
        }
    }
}

/// `a * b`, as (high, low) halves of 256 bits.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const M: u128 = u64::MAX as u128;
    let (a1, a0, b1, b0) = (a >> 64, a & M, b >> 64, b & M);
    let (lo, mid1, mid2, hi) = (a0 * b0, a1 * b0, a0 * b1, a1 * b1);
    let (mid, carry) = mid1.overflowing_add(mid2);
    let (lo, c) = lo.overflowing_add(mid << 64);
    let hi = hi + (mid >> 64) + ((carry as u128) << 64) + c as u128;
    (hi, lo)
}

/// `p - q`, for 256 bit values, as a float.
fn sub_to_f64(p: (u128, u128), q: (u128, u128)) -> f64 {
    let wide = |(hi, lo): (u128, u128)| hi as f64 * 2f64.powi(128) + lo as f64;
    let sub = |(ph, pl): (u128, u128), (qh, ql): (u128, u128)| {
        let (lo, borrow) = pl.overflowing_sub(ql);
        (ph - qh - borrow as u128, lo)
    };
    if p >= q {
        wide(sub(p, q))
    } else {
        -wide(sub(q, p))
    }
}

/// Pearson's correlation over `arcs` arcs, from its sums: its numerator and variances computed
/// exactly (in 256 bits), and then divided.
fn pearson(arcs: u128, s: Sums) -> f64 {
    let covariance = sub_to_f64(mul_wide(arcs, s.xy), mul_wide(s.x, s.y));
    let variance_x = sub_to_f64(mul_wide(arcs, s.xx), mul_wide(s.x, s.x));
    let variance_y = sub_to_f64(mul_wide(arcs, s.yy), mul_wide(s.y, s.y));
    if arcs == 0 || variance_x == 0.0 || variance_y == 0.0 {
        return f64::NAN;
    }
    covariance / (variance_x.sqrt() * variance_y.sqrt())
}

/// Arcs ahead whose heads' values are prefetched: the head values' reads are scattered, and
/// the arcs' heads known in advance. Best at 8 to 16 on graphs whose values outgrow the
/// last-level cache (20% faster on a 50 million node graph's, one byte each), and harmless on
/// those that don't.
const PREFETCH: usize = 16;

/// Asks for the cache line holding `*p` (a hint: never faults).
#[inline(always)]
fn prefetch<T>(p: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE is part of x86_64's baseline, and prefetches are hints, valid for any address
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T0>(p as *const T as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// The sums of Pearson's correlation of `x(u)` and `y(v)` over the arcs `u -> v`, where node
/// `v`'s value `y(v)` is `y_map(y[v])` and `heads(v)` is the number of arcs into it.
///
/// The arcs' only scattered reads are their heads' entries of `y`; the rest streams.
#[allow(clippy::too_many_arguments)]
fn sums<Ix, T, X, M>(
    a: &Arcs<Ix>,
    x: &X,
    y: &[T],
    y_map: &M,
    heads: &(dyn Fn(usize) -> u64 + Sync),
    y_max: u64,
    symmetric: bool,
    over_arcs: bool,
) -> Result<Sums, Box<dyn std::error::Error>>
where
    Ix: graph::IndexType,
    T: Value,
    X: Fn(usize) -> u64 + Sync + ?Sized,
    M: Fn(u64) -> u64 + Sync + ?Sized,
{
    let (offsets, neighbours) = (a.offsets, a.neighbours);
    let parts = chunked(a.threads, a.n, &|begin, end| {
        let mut s = Sums::default();
        let last = offsets[end].index().max(1) - 1;
        for u in begin..end {
            let (b, e) = (offsets[u].index(), offsets[u + 1].index());
            let xu = x(u) as u128;
            let tx = (e - b) as u128 * xu;
            s.x += tx;
            s.xx += tx * xu;
            if !symmetric {
                let yu = y_map(y[u].widen()) as u128;
                let ty = heads(u) as u128 * yu;
                s.y += ty;
                s.yy += ty * yu;
            }
            if !over_arcs {
                continue;
            }
            // the heads' values' sum, in 64 bits if it surely fits, their values fetched
            // `PREFETCH` arcs ahead
            let row = &neighbours[b..e];
            let total = if (e - b) as u128 * y_max as u128 <= u64::MAX as u128 {
                let mut t = 0u64;
                for i in b..e {
                    let ahead = (i + PREFETCH).min(last);
                    // SAFETY: `b..e` and `ahead` index the chunk's arcs (`last`, its last), and
                    // arcs' heads are nodes
                    unsafe {
                        let (h, v) = (
                            neighbours.get_unchecked(ahead).index(),
                            neighbours.get_unchecked(i).index(),
                        );
                        debug_assert!(h < y.len() && v < y.len());
                        prefetch(y.get_unchecked(h));
                        t += y_map(y.get_unchecked(v).widen());
                    }
                }
                t as u128
            } else {
                row.iter()
                    .map(|v| y_map(y[v.index()].widen()) as u128)
                    .sum::<u128>()
            };
            s.xy += xu * total;
        }
        s
    })?;
    let mut s = parts.into_iter().fold(Sums::default(), |s, p| s + p);
    if symmetric {
        (s.y, s.yy) = (s.x, s.xx);
    }
    Ok(s)
}

/// [`sums`], the one over arcs here, or by `arc_products`.
#[allow(clippy::too_many_arguments)]
fn all_sums<Ix, T, X, M>(
    a: &Arcs<Ix>,
    x: &X,
    y: &[T],
    y_map: &M,
    heads: &(dyn Fn(usize) -> u64 + Sync),
    y_max: u64,
    symmetric: bool,
    arc_products: Option<ArcProducts>,
) -> Result<Sums, Box<dyn std::error::Error>>
where
    Ix: graph::IndexType,
    T: Value,
    X: Fn(usize) -> u64 + Sync + ?Sized,
    M: Fn(u64) -> u64 + Sync + ?Sized,
{
    match arc_products {
        None => sums(a, x, y, y_map, heads, y_max, symmetric, true),
        Some(products) => {
            let mut s = sums(a, x, y, y_map, heads, y_max, symmetric, false)?;
            s.xy = products(&|u| x(u), &|v| y_map(y[v].widen()))?;
            Ok(s)
        }
    }
}

/// `counts[k]`: the sum of `weight(u)` over the nodes `u` of `key(u) == k` (keys `< len`).
fn histogram(
    threads: usize,
    n: usize,
    len: usize,
    key: &(dyn Fn(usize) -> usize + Sync),
    weight: &(dyn Fn(usize) -> u64 + Sync),
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    // small keys per thread (they're the common ones, contended for if shared), the rest shared
    let local = len.min(LOCAL_HISTOGRAM);
    let shared: Vec<AtomicU64> = (local..len).map(|_| AtomicU64::new(0)).collect();
    let parts = by_range(threads, n, &|begin, end| {
        let mut counts = vec![0u64; local];
        for u in begin..end {
            let (k, w) = (key(u), weight(u));
            if k < local {
                counts[k] += w;
            } else {
                shared[k - local].fetch_add(w, Ordering::Relaxed);
            }
        }
        counts
    })?;
    let mut counts = vec![0u64; len];
    for part in parts {
        for (c, p) in counts.iter_mut().zip(part) {
            *c += p;
        }
    }
    for (c, s) in counts[local..].iter_mut().zip(shared) {
        *c = s.into_inner();
    }
    Ok(counts)
}

/// Each value's rank among the arcs' values, ties ranked by their average rank, doubled (to be
/// an integer): `counts[d]` arcs having value `d`.
fn doubled_ranks(counts: &[u64]) -> Vec<u64> {
    let mut before = 0;
    counts
        .iter()
        .map(|&c| {
            let rank = 2 * before + c + 1;
            before += c;
            rank
        })
        .collect()
}

/// Computes the sum, over every arc `u -> v`, of `x(u) * y(v)` elsewhere (on a GPU).
type ArcProducts<'a> = &'a dyn Fn(
    &(dyn Fn(usize) -> u64 + Sync),
    &(dyn Fn(usize) -> u64 + Sync),
) -> Result<u128, Box<dyn std::error::Error>>;

/// A node's degree as the tail of an arc: the same as its degree as a head (undirected graphs),
/// or another.
#[derive(Clone, Copy)]
enum Tails<'a> {
    AsHeads,
    Of(&'a (dyn Fn(usize) -> u64 + Sync), u64),
}

/// Pearson's or Spearman's correlation of node degrees over the arcs `u -> v`: of `x(u)` and
/// `y(v)`, with `heads(v)` arcs into `v` and `y` at most `y_bound`.
fn degree_correlation<Ix: graph::IndexType, T: Value, Y: Fn(usize) -> u64 + Sync + ?Sized>(
    a: &Arcs<Ix>,
    x: Tails,
    y_of: &Y,
    heads: &(dyn Fn(usize) -> u64 + Sync),
    y_bound: u64,
    correlation: Correlation,
    arc_products: Option<ArcProducts>,
) -> Result<f64, Box<dyn std::error::Error>> {
    // the heads' degrees, narrow
    let y_mem = SharedSliceMut::<T>::abst_mem_mut(&(a.scratch)(0)?, a.n.max(1), true)?;
    {
        let y = y_mem.shared_slice();
        in_parallel(a.threads, a.n, &|begin, end| {
            let mut y = y;
            for v in begin..end {
                *y.get_mut(v) = T::narrow(y_of(v));
            }
        })?;
    }
    let y = &y_mem.as_slice()[..a.n];
    let arcs = a.neighbours.len() as u128;
    // each combination of maps compiled on its own: the arcs' loop calls none
    let s = match (correlation, x) {
        (Correlation::Pearson, Tails::AsHeads) => all_sums(
            a,
            &|u: usize| y[u].widen(),
            y,
            &|d: u64| d,
            heads,
            y_bound,
            true,
            arc_products,
        )?,
        (Correlation::Pearson, Tails::Of(f, _)) => {
            all_sums(a, f, y, &|d: u64| d, heads, y_bound, false, arc_products)?
        }
        (Correlation::Spearman, x) => {
            let offsets = a.offsets;
            let tails = |u: usize| (offsets[u + 1].index() - offsets[u].index()) as u64;
            let max_y = max_of(a.threads, a.n, &|v| y[v].widen())?;
            let rank_y = doubled_ranks(&histogram(
                a.threads,
                a.n,
                max_y as usize + 1,
                &|v| y[v].widen() as usize,
                heads,
            )?);
            let y_max = rank_y.iter().copied().max().unwrap_or(0);
            let rank_y = &rank_y;
            let y_rank = |d: u64| rank_y[d as usize];
            match x {
                // the tails' ranks the heads': the same values, as many times
                Tails::AsHeads => all_sums(
                    a,
                    &|u: usize| rank_y[y[u].widen() as usize],
                    y,
                    &y_rank,
                    heads,
                    y_max,
                    true,
                    arc_products,
                )?,
                Tails::Of(f, max_x) => {
                    let rank_x = doubled_ranks(&histogram(
                        a.threads,
                        a.n,
                        max_x as usize + 1,
                        &|u| f(u) as usize,
                        &tails,
                    )?);
                    let rank_x = &rank_x;
                    all_sums(
                        a,
                        &|u: usize| rank_x[f(u) as usize],
                        y,
                        &y_rank,
                        heads,
                        y_max,
                        false,
                        arc_products,
                    )?
                }
            }
        }
    };
    Ok(pearson(arcs, s))
}

/// `degree_correlation` with `y` in the narrowest type holding `y_bound`.
fn degree_correlation_narrowest<Ix: graph::IndexType, Y: Fn(usize) -> u64 + Sync + ?Sized>(
    a: &Arcs<Ix>,
    x: Tails,
    y_of: &Y,
    heads: &(dyn Fn(usize) -> u64 + Sync),
    y_bound: u64,
    correlation: Correlation,
    arc_products: Option<ArcProducts>,
) -> Result<f64, Box<dyn std::error::Error>> {
    if y_bound <= u8::MAX as u64 {
        degree_correlation::<Ix, u8, Y>(a, x, y_of, heads, y_bound, correlation, arc_products)
    } else if y_bound <= u16::MAX as u64 {
        degree_correlation::<Ix, u16, Y>(a, x, y_of, heads, y_bound, correlation, arc_products)
    } else if y_bound <= u32::MAX as u64 {
        degree_correlation::<Ix, u32, Y>(a, x, y_of, heads, y_bound, correlation, arc_products)
    } else {
        degree_correlation::<Ix, u64, Y>(a, x, y_of, heads, y_bound, correlation, arc_products)
    }
}

/// The largest of `f(u)` over `0..n`.
fn max_of<F: Fn(usize) -> u64 + Sync + ?Sized>(
    threads: usize,
    n: usize,
    f: &F,
) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(chunked(threads, n, &|begin, end| {
        (begin..end).map(f).max().unwrap_or(0)
    })?
    .into_iter()
    .max()
    .unwrap_or(0))
}

/// Runs `f` on a graph's CSR, its scratch files removed after.
fn with_arcs<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness, R>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    f: impl FnOnce(&Arcs<Ix>) -> Result<R, Box<dyn std::error::Error>>,
) -> Result<R, Box<dyn std::error::Error>> {
    let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
    let neighbours = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());
    let scratch = |seq: usize| g.build_cache_filename(CacheFile::Assortativity, Some(seq));
    let a = Arcs {
        threads: g.thread_num(),
        n: g.size(),
        offsets: offsets.as_slice(),
        neighbours: neighbours.as_slice(),
        scratch: &scratch,
    };
    let result = f(&a);
    g.cleanup_cache(CacheFile::Assortativity)?;
    result
}

/// Newman's degree assortativity coefficient of an undirected graph ([`Correlation::Pearson`],
/// networkx's `degree_assortativity_coefficient`), or its rank based counterpart
/// ([`Correlation::Spearman`]): the correlation of the degrees of the ends of each arc (each
/// edge, both ways). Degrees count self-loops twice, as networkx's.
pub fn degree_assortativity<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &Graph<N, E, Ix>,
    correlation: Correlation,
) -> Result<f64, Box<dyn std::error::Error>> {
    degree_assortativity_impl(g, correlation, None)
}

fn degree_assortativity_impl<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &Graph<N, E, Ix>,
    correlation: Correlation,
    arc_products: Option<ArcProducts>,
) -> Result<f64, Box<dyn std::error::Error>> {
    if correlation == Correlation::Pearson && arc_products.is_none() {
        return with_arcs(g, |a| {
            // rows of up to a byte's length, typically: tried first, the narrowest
            if let Some(r) = undirected_pearson::<Ix, u8>(a)? {
                return Ok(r);
            }
            let offsets = a.offsets;
            let max = max_of(a.threads, a.n, &|u: usize| {
                (offsets[u + 1].index() - offsets[u].index()) as u64
            })?;
            Ok(if max <= u16::MAX as u64 {
                undirected_pearson::<Ix, u16>(a)?
            } else if max <= u32::MAX as u64 {
                undirected_pearson::<Ix, u32>(a)?
            } else {
                undirected_pearson::<Ix, u64>(a)?
            }
            .expect("rows fit"))
        });
    }
    with_arcs(g, |a| {
        let (offsets, neighbours) = (a.offsets, a.neighbours);
        // its row's length, plus its self-loops (sorted rows: none unless the node is within
        // its neighbours' range, by binary search then)
        let degree = |u: usize| {
            let row = &neighbours[offsets[u].index()..offsets[u + 1].index()];
            let loops = match (row.first(), row.last()) {
                (Some(lo), Some(hi)) if lo.index() <= u && u <= hi.index() => {
                    let first = row.partition_point(|v| v.index() < u);
                    row[first..].iter().take_while(|v| v.index() == u).count()
                }
                _ => 0,
            };
            (row.len() + loops) as u64
        };
        let arcs_of = |u: usize| (offsets[u + 1].index() - offsets[u].index()) as u64;
        // degrees are at most twice their rows' lengths (all self-loops): bounded without
        // reading the rows
        let bound = 2 * max_of(a.threads, a.n, &arcs_of)?;

        degree_correlation_narrowest(
            a,
            Tails::AsHeads,
            &degree,
            &arcs_of,
            bound,
            correlation,
            arc_products,
        )
    })
}

/// Newman's degree assortativity of an undirected graph, without reading its rows but in the one
/// pass over its arcs: `None` if a row's length doesn't fit a `T`.
///
/// A node's degree `d = r + l` is its row's length `r` plus its self-loops `l`. Heads' `r`s are
/// read from an array of them (from the offsets alone, narrow), each node's `l` counted as its
/// row is, and `Σ_u d_u Σ_{v ∈ N(u)} d_v` is `Σ_u (d_u + l_u) T_u + Σ_v l_v Λ_v`, with
/// `T_u = Σ_{v ∈ N(u)} r_v` and `Λ_v = Σ_{u ∈ N(v)} l_u`: arcs come in reverse pairs (a self-loop
/// its own reverse), so `Σ_u d_u Σ_{v ∈ N(u)} l_v = Σ_v l_v Σ_{u ∈ N(v)} d_u`. The `Λ`s, over the
/// rows of nodes with self-loops only, are summed after.
fn undirected_pearson<Ix: graph::IndexType, T: Value>(
    a: &Arcs<Ix>,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    let (n, offsets, neighbours) = (a.n, a.offsets, a.neighbours);
    let row_len = |u: usize| offsets[u + 1].index() - offsets[u].index();
    // rows' lengths, narrow (failing on one too long)
    let r_mem = SharedSliceMut::<T>::abst_mem_mut(&(a.scratch)(0)?, n.max(1), true)?;
    let fits = {
        let r = r_mem.shared_slice();
        by_range(a.threads, n, &|begin, end| {
            let mut r = r;
            let mut fits = true;
            for u in begin..end {
                let len = row_len(u) as u64;
                fits &= len <= T::MAX;
                *r.get_mut(u) = T::narrow(len);
            }
            fits
        })?
    };
    if !fits.into_iter().all(|f| f) {
        return Ok(None);
    }
    let r = &r_mem.as_slice()[..n];
    let max_r = T::MAX;
    let parts = chunked(a.threads, n, &|begin, end| {
        let mut s = Sums::default();
        let mut looped: Vec<(usize, u64)> = vec![];
        let last = offsets[end].index().max(1) - 1;
        for u in begin..end {
            let (b, e) = (offsets[u].index(), offsets[u + 1].index());
            let mut loops = 0u64;
            // a head's row length, and whether it's a self-loop
            let head = |i: usize| {
                // SAFETY: `i` indexes the chunk's arcs (`last`, its last), and heads are nodes
                unsafe {
                    let ahead = neighbours.get_unchecked((i + PREFETCH).min(last)).index();
                    let v = neighbours.get_unchecked(i).index();
                    debug_assert!(ahead < n && v < n);
                    prefetch(r.get_unchecked(ahead));
                    (r.get_unchecked(v).widen(), (v == u) as u64)
                }
            };
            let t = if (e - b) as u128 * max_r as u128 <= u64::MAX as u128 {
                // in 64 bits, as it surely fits
                let mut t = 0u64;
                for i in b..e {
                    let (rv, l) = head(i);
                    t += rv;
                    loops += l;
                }
                t as u128
            } else {
                let mut t = 0u128;
                for i in b..e {
                    let (rv, l) = head(i);
                    t += rv as u128;
                    loops += l;
                }
                t
            };
            let len = (e - b) as u128;
            let d = len + loops as u128;
            s.x += len * d;
            s.xx += len * d * d;
            s.xy += (d + loops as u128) * t;
            if loops > 0 {
                looped.push((u, loops));
            }
        }
        (s, looped)
    })?;
    let mut s = Sums::default();
    let mut looped = vec![];
    for (p, l) in parts {
        s = s + p;
        looped.extend(l);
    }
    // Σ_v l_v Λ_v: arcs between nodes with self-loops
    if !looped.is_empty() {
        let loops: std::collections::HashMap<usize, u64> = looped.iter().copied().collect();
        for &(v, l) in &looped {
            let lambda: u64 = neighbours[offsets[v].index()..offsets[v + 1].index()]
                .iter()
                .filter_map(|u| loops.get(&u.index()))
                .sum();
            s.xy += l as u128 * lambda as u128;
        }
    }
    (s.y, s.yy) = (s.x, s.xx);
    Ok(Some(pearson(neighbours.len() as u128, s)))
}

/// Newman's degree assortativity coefficient of a directed graph ([`Correlation::Pearson`],
/// networkx's `degree_assortativity_coefficient`), or its rank based counterpart
/// ([`Correlation::Spearman`]): the correlation, over every arc `u -> v`, of `u`'s degree `x`
/// and `v`'s degree `y` (networkx's defaults: [`DegreeType::Out`] and [`DegreeType::In`]).
///
/// Uses the graph's [transpose](DiGraph::transpose) (built, and cached, if it isn't yet) for
/// in-degrees.
pub fn directed_degree_assortativity<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &DiGraph<N, E, Ix>,
    x: DegreeType,
    y: DegreeType,
    correlation: Correlation,
) -> Result<f64, Box<dyn std::error::Error>> {
    directed_degree_assortativity_impl(g, x, y, correlation, None)
}

fn directed_degree_assortativity_impl<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &DiGraph<N, E, Ix>,
    x: DegreeType,
    y: DegreeType,
    correlation: Correlation,
    arc_products: Option<ArcProducts>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let t = g.transpose()?;
    let in_offsets = SharedSlice::<Ix>::new(t.offsets_ptr(), t.offsets_size());
    with_arcs(g, |a| {
        let offsets = a.offsets;
        let out_degree = |u: usize| (offsets[u + 1].index() - offsets[u].index()) as u64;
        let in_degree =
            |u: usize| (in_offsets.get(u + 1).index() - in_offsets.get(u).index()) as u64;
        let degree = |d: DegreeType| -> &(dyn Fn(usize) -> u64 + Sync) {
            match d {
                DegreeType::Out => &out_degree,
                DegreeType::In => &in_degree,
            }
        };
        let (x, y) = (degree(x), degree(y));
        let max_x = max_of(a.threads, a.n, x)?;
        let max_y = max_of(a.threads, a.n, y)?;
        degree_correlation_narrowest(
            a,
            Tails::Of(x, max_x),
            y,
            &in_degree,
            max_y,
            correlation,
            arc_products,
        )
    })
}

/// `f(u)` for every node, as `u32`s for a GPU, failing on values that don't fit.
#[cfg(feature = "gpu")]
fn gpu_values(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize) -> u64 + Sync),
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut values = vec![0u32; n];
    let too_large = AtomicU64::new(0);
    {
        let out = SharedSliceMut::<u32>::new(values.as_mut_ptr(), n);
        in_parallel(threads, n, &|begin, end| {
            let mut out = out;
            for u in begin..end {
                let v = f(u);
                if v > u32::MAX as u64 {
                    too_large.fetch_max(v, Ordering::Relaxed);
                }
                *out.get_mut(u) = v as u32;
            }
        })?;
    }
    match too_large.into_inner() {
        0 => Ok(values),
        v => Err(format!("error a node's value ({v}) is too large for the GPU (32 bits)").into()),
    }
}

/// [`degree_assortativity`] with its sum over arcs computed on a GPU (`graph` being `g` in the
/// `backend`'s device memory, see [`GpuBackend::upload_graph`]): the same result, bit for bit.
///
/// Node values are uploaded as `u32`s: degrees, and with [`Correlation::Spearman`] their
/// (doubled) ranks among the arcs', must fit.
///
/// [`GpuBackend::upload_graph`]: crate::gpu::GpuBackend::upload_graph
#[cfg(feature = "gpu")]
pub fn degree_assortativity_gpu<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    B: crate::gpu::GpuBackend,
>(
    g: &Graph<N, E, Ix>,
    correlation: Correlation,
    backend: &B,
    graph: &B::Graph,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (threads, n) = (g.thread_num(), g.size());
    let products = |x: &(dyn Fn(usize) -> u64 + Sync), y: &(dyn Fn(usize) -> u64 + Sync)| {
        backend.arc_products(
            graph,
            &gpu_values(threads, n, x)?,
            &gpu_values(threads, n, y)?,
        )
    };
    degree_assortativity_impl(g, correlation, Some(&products))
}

/// [`directed_degree_assortativity`] with its sum over arcs computed on a GPU, as
/// [`degree_assortativity_gpu`].
#[cfg(feature = "gpu")]
pub fn directed_degree_assortativity_gpu<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    B: crate::gpu::GpuBackend,
>(
    g: &DiGraph<N, E, Ix>,
    x: DegreeType,
    y: DegreeType,
    correlation: Correlation,
    backend: &B,
    graph: &B::Graph,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (threads, n) = (g.thread_num(), g.size());
    let products = |x: &(dyn Fn(usize) -> u64 + Sync), y: &(dyn Fn(usize) -> u64 + Sync)| {
        backend.arc_products(
            graph,
            &gpu_values(threads, n, x)?,
            &gpu_values(threads, n, y)?,
        )
    };
    directed_degree_assortativity_impl(g, x, y, correlation, Some(&products))
}

/// A sum of floats, compensated (Neumaier's).
#[derive(Clone, Copy, Default)]
struct Sum {
    s: f64,
    c: f64,
}

impl Sum {
    fn add(&mut self, v: f64) {
        let t = self.s + v;
        self.c += if self.s.abs() >= v.abs() {
            (self.s - t) + v
        } else {
            (v - t) + self.s
        };
        self.s = t;
    }
    fn merge(&mut self, o: Sum) {
        self.add(o.s);
        self.add(o.c);
    }
    fn value(self) -> f64 {
        self.s + self.c
    }
}

/// Pearson's correlation of `values` over the arcs: of `values[u]` and `values[v]` over every arc
/// `u -> v` (each undirected edge, both ways), networkx's `numeric_assortativity_coefficient`.
///
/// Computed in two passes, the second over the values minus their means, with compensated
/// sums, in a fixed order: the same whatever the threads.
pub fn numeric_assortativity<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    values: &[f64],
) -> Result<f64, Box<dyn std::error::Error>> {
    if values.len() != g.size() {
        return Err(format!(
            "error computing numeric assortativity: {} values for {} nodes",
            values.len(),
            g.size()
        )
        .into());
    }
    let symmetric = !Dir::DIRECTED || g.is_symmetric()?;
    with_arcs(g, |a| {
        let (offsets, neighbours) = (a.offsets, a.neighbours);
        let arcs = neighbours.len() as f64;
        if neighbours.is_empty() {
            return Ok(f64::NAN);
        }
        // means: of the tails' values, and of the heads' (the same, if arcs all have reverses)
        let means = chunked(a.threads, a.n, &|begin, end| {
            let (mut x, mut y) = (Sum::default(), Sum::default());
            for u in begin..end {
                let row = &neighbours[offsets[u].index()..offsets[u + 1].index()];
                x.add(row.len() as f64 * values[u]);
                if !symmetric {
                    for v in row {
                        y.add(values[v.index()]);
                    }
                }
            }
            (x, y)
        })?;
        let (mut x, mut y) = (Sum::default(), Sum::default());
        for (px, py) in means {
            x.merge(px);
            y.merge(py);
        }
        let mean_x = x.value() / arcs;
        let mean_y = if symmetric { mean_x } else { y.value() / arcs };
        let parts = chunked(a.threads, a.n, &|begin, end| {
            let (mut xy, mut xx, mut yy) = (Sum::default(), Sum::default(), Sum::default());
            let last = offsets[end].index().max(1) - 1;
            for u in begin..end {
                let (b, e) = (offsets[u].index(), offsets[u + 1].index());
                let row = &neighbours[b..e];
                let xu = values[u] - mean_x;
                let mut heads = Sum::default();
                for (i, v) in (b..e).zip(row) {
                    prefetch(&values[neighbours[(i + PREFETCH).min(last)].index()]);
                    let yv = values[v.index()] - mean_y;
                    heads.add(yv);
                    if !symmetric {
                        yy.add(yv * yv);
                    }
                }
                xy.add(xu * heads.value());
                xx.add(row.len() as f64 * xu * xu);
            }
            (xy, xx, yy)
        })?;
        let (mut xy, mut xx, mut yy) = (Sum::default(), Sum::default(), Sum::default());
        for (pxy, pxx, pyy) in parts {
            xy.merge(pxy);
            xx.merge(pxx);
            yy.merge(pyy);
        }
        let (xx, yy) = (xx.value(), if symmetric { xx.value() } else { yy.value() });
        if xx == 0.0 || yy == 0.0 {
            return Ok(f64::NAN);
        }
        Ok(xy.value() / (xx.sqrt() * yy.sqrt()))
    })
}

/// Newman's assortativity coefficient of the categories `labels` (any `usize`s), networkx's
/// `attribute_assortativity_coefficient`: `(Σ_i e_ii − Σ_i a_i b_i) / (1 − Σ_i a_i b_i)`, where
/// `e_ij` is the fraction of arcs from category `i` to category `j` (each undirected edge counts
/// both ways), and `a_i` and `b_i` the fractions of arcs from and to category `i`. Computed
/// exactly, in integers.
pub fn attribute_assortativity<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    labels: &[usize],
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = g.size();
    if labels.len() != n {
        return Err(format!(
            "error computing attribute assortativity: {} labels for {n} nodes",
            labels.len()
        )
        .into());
    }
    let symmetric = !Dir::DIRECTED || g.is_symmetric()?;
    with_arcs(g, |a| {
        let arcs = a.neighbours.len() as u128;
        if arcs == 0 {
            return Ok(f64::NAN);
        }
        // labels as dense indices: as they are if smaller than `n`, by rank among the distinct
        // labels otherwise
        let max = max_of(a.threads, n, &|u| labels[u] as u64)? as usize;
        let distinct = (max >= n).then(|| {
            let mut distinct = labels.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            distinct
        });
        let categories = distinct.as_ref().map_or(max + 1, |d| d.len());
        let dense = |u: usize| match &distinct {
            None => labels[u] as u64,
            Some(d) => d.partition_point(|&l| l < labels[u]) as u64,
        };
        let (from, to, within) = if categories <= 1 << 8 {
            attribute_counts::<Ix, u8>(a, &dense, categories, symmetric)?
        } else if categories <= 1 << 16 {
            attribute_counts::<Ix, u16>(a, &dense, categories, symmetric)?
        } else if categories as u64 <= 1 << 32 {
            attribute_counts::<Ix, u32>(a, &dense, categories, symmetric)?
        } else {
            attribute_counts::<Ix, u64>(a, &dense, categories, symmetric)?
        };
        let to = to.as_ref().unwrap_or(&from);
        let expected: u128 = from
            .iter()
            .zip(to)
            .map(|(&f, &t)| f as u128 * t as u128)
            .sum();
        // (A e − Σ a b) / (A² − Σ a b), with A arcs and e within, in 256 bits
        let num = sub_to_f64(mul_wide(arcs, within as u128), (0, expected));
        let den = sub_to_f64(mul_wide(arcs, arcs), (0, expected));
        if den == 0.0 {
            return Ok(f64::NAN);
        }
        Ok(num / den)
    })
}

/// The arcs from each category, to each (unless the same, `symmetric`), and within categories,
/// node categories `dense(u) < categories` stored as `T`s.
#[allow(clippy::type_complexity)]
fn attribute_counts<Ix: graph::IndexType, T: Value>(
    a: &Arcs<Ix>,
    dense: &(dyn Fn(usize) -> u64 + Sync),
    categories: usize,
    symmetric: bool,
) -> Result<(Vec<u64>, Option<Vec<u64>>, u64), Box<dyn std::error::Error>> {
    let (n, offsets, neighbours) = (a.n, a.offsets, a.neighbours);
    let mem = SharedSliceMut::<T>::abst_mem_mut(&(a.scratch)(0)?, n.max(1), true)?;
    {
        let d = mem.shared_slice();
        in_parallel(a.threads, n, &|begin, end| {
            let mut d = d;
            for u in begin..end {
                *d.get_mut(u) = T::narrow(dense(u));
            }
        })?;
    }
    let c = &mem.as_slice()[..n];
    let from = histogram(a.threads, n, categories, &|u| c[u].widen() as usize, &|u| {
        (offsets[u + 1].index() - offsets[u].index()) as u64
    })?;
    let within: u64 = chunked(a.threads, n, &|begin, end| {
        let mut within = 0u64;
        let last = offsets[end].index().max(1) - 1;
        for u in begin..end {
            let cu = c[u].widen();
            for i in offsets[u].index()..offsets[u + 1].index() {
                prefetch(&c[neighbours[(i + PREFETCH).min(last)].index()]);
                within += (c[neighbours[i].index()].widen() == cu) as u64;
            }
        }
        within
    })?
    .into_iter()
    .sum();
    // arcs to each category: the same as from, if arcs all have reverses
    let to = if symmetric {
        None
    } else {
        let local = categories.min(LOCAL_HISTOGRAM);
        let shared: Vec<AtomicU64> = (local..categories).map(|_| AtomicU64::new(0)).collect();
        let parts = by_range(a.threads, n, &|begin, end| {
            let mut counts = vec![0u64; local];
            for v in &neighbours[offsets[begin].index()..offsets[end].index()] {
                let k = c[v.index()].widen() as usize;
                if k < local {
                    counts[k] += 1;
                } else {
                    shared[k - local].fetch_add(1, Ordering::Relaxed);
                }
            }
            counts
        })?;
        let mut to = vec![0u64; categories];
        for part in parts {
            for (t, p) in to.iter_mut().zip(part) {
                *t += p;
            }
        }
        for (t, s) in to[local..].iter_mut().zip(shared) {
            *t = s.into_inner();
        }
        Some(to)
    };
    Ok((from, to, within))
}

#[cfg(test)]
mod test;
