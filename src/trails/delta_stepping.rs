//! Parallel single-source shortest paths by Δ-stepping (Meyer U. and Sanders P., ["Δ-stepping: a
//! parallelizable shortest path algorithm"](https://doi.org/10.1016/S0196-6774(03)00076-2)), as
//! the GAP Benchmark Suite implements it (Beamer S., Asanović K. and Patterson D.,
//! ["The GAP Benchmark Suite"](https://arxiv.org/abs/1508.03619)).
//!
//! Nodes are kept in buckets of distances `[iΔ, (i + 1)Δ)`, and the threads empty the lowest
//! nonempty bucket together, relaxing its nodes' arcs in parallel: a relaxation is an atomic
//! minimum on the target's distance (non-negative `f64`s order as their bits do, as `u64`s), and
//! a node whose distance drops goes to its new bucket, in the relaxing thread's own buckets (no
//! shared queue). Each bucket ends at a barrier, after which the threads agree on the next one.
//!
//! The distances found are Dijkstra's, bit for bit: a node's is the least, over the arcs into
//! it, of the distance of the arc's tail plus its weight, which every thread computes alike.

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;
use crate::utils::SpinBarrier;

use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The buckets each thread keeps, as a ring: Δ is at least the heaviest arc's weight over
/// `RING - 2`, so that a relaxation lands at most `RING - 1` buckets past the current one.
const RING: usize = 1024;
/// Nodes of a bucket a thread takes at a time, at most: fewer in small buckets, so that every
/// thread gets some (about 8 chunks each).
const CHUNK: usize = 64;
/// Bucket fusion (Zhang Y. et al., ["Optimizing ordered graph algorithms with GraphIt"](https://doi.org/10.1145/3368826.3377909)):
/// the nodes a thread adds to the current bucket while emptying it, it relaxes right away, without
/// waiting for the other threads, as long as they're fewer than this --- on graphs of long paths
/// and narrow fronts (road networks), most buckets then end without a barrier.
const FUSION: usize = 1024;

/// The weights' mean and maximum (`1` and `1` on unweighted graphs); weights must have been
/// checked finite and non-negative.
fn weight_stats<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let arcs = g.width();
    if !E::is_weighted() || arcs == 0 {
        return Ok((1., 1.));
    }
    let total = g.total_weight()?;
    let labels = SharedSlice::<E>::new(g.edge_labels_ptr(), arcs);
    let chunks = g.thread_num().clamp(1, arcs);
    let load = arcs.div_ceil(chunks);
    let max = std::thread::scope(|s| {
        let handles: Vec<_> = (0..chunks)
            .map(|c| {
                s.spawn(move || {
                    ((c * load).min(arcs)..((c + 1) * load).min(arcs))
                        .map(|e| labels.get(e).weight::<f64>())
                        .fold(0f64, f64::max)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(f64::INFINITY))
            .fold(0f64, f64::max)
    });
    Ok((total / arcs as f64, max))
}

/// The bucket width Δ-stepping uses on `g` unless given one: a multiple of the mean weight
/// (see [`delta_stepping`]), at least the heaviest weight over `RING - 2`, and positive.
pub(crate) fn default_delta<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (mean, max) = weight_stats(g)?;
    Ok(clamp_delta(DELTA_PER_MEAN_WEIGHT * mean, max))
}

/// Δ in mean weights, by default.
pub(crate) const DELTA_PER_MEAN_WEIGHT: f64 = 1.;

/// `delta`, raised to the least width the ring of buckets allows for weights up to `max`, and
/// made positive (all weights being zero, any width does).
fn clamp_delta(delta: f64, max: f64) -> f64 {
    let delta = delta.max(max / (RING - 2) as f64);
    if delta.is_normal() { delta } else { 1. }
}

/// Every node's distance from `source` in `g` (following its arcs, as long as their weights),
/// left in `dist` as `f64` bits ([`f64::INFINITY`] for unreachable nodes), by Δ-stepping on
/// `threads` threads with buckets `delta` wide (raised to the least width the buckets allow, if
/// smaller). Weights must have been checked finite and non-negative.
///
/// Δ trades work for parallelism: buckets 1 wide on an unweighted graph are a BFS's levels, and
/// infinitely wide ones make it Bellman-Ford's (relaxing nodes again whenever their distances
/// drop). Memory: `dist`, and each thread's buckets, holding the nodes whose distances dropped
/// (at most once per such drop) until their bucket's turn.
pub(crate) fn delta_stepping<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    dist: &[AtomicU64],
    delta: f64,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = g.size();
    assert_eq!(dist.len(), n);
    let (_, max) = weight_stats(g)?;
    let delta = clamp_delta(delta, max);
    let threads = threads.max(1);
    let infinity = f64::INFINITY.to_bits();

    let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
    let targets = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());
    let labels = SharedSlice::<E>::new(g.edge_labels_ptr(), g.width());

    // each thread's share of the current bucket, read by all while it's emptied
    let handoff: Vec<RwLock<Vec<Ix>>> = (0..threads).map(|_| RwLock::new(Vec::new())).collect();
    let cursor = AtomicUsize::new(0);
    let next_bucket = AtomicUsize::new(usize::MAX);
    let barrier = SpinBarrier::new(threads);

    std::thread::scope(|scope| {
        for tid in 0..threads {
            let (handoff, cursor, next_bucket, barrier) =
                (&handoff, &cursor, &next_bucket, &barrier);
            scope.spawn(move || {
                // every distance infinite, then the source's 0
                let load = n.div_ceil(threads);
                for d in &dist[(tid * load).min(n)..((tid + 1) * load).min(n)] {
                    d.store(infinity, Ordering::Relaxed);
                }
                let mut buckets: Vec<Vec<Ix>> = (0..RING).map(|_| Vec::new()).collect();
                barrier.wait();
                if tid == 0 {
                    dist[source].store(0f64.to_bits(), Ordering::Relaxed);
                    handoff[0].write().unwrap().push(Ix::new(source));
                }
                let mut current = 0usize;
                barrier.wait();

                let mut fused: Vec<Ix> = Vec::new();
                loop {
                    let lower = current as f64 * delta;
                    // relaxes `u`'s arcs, if it's still in the current bucket, adding the nodes
                    // whose distances drop to their buckets
                    let relax = |u: usize, buckets: &mut Vec<Vec<Ix>>| {
                        let du = f64::from_bits(dist[u].load(Ordering::Relaxed));
                        // moved to an earlier bucket, and relaxed there already
                        if du < lower {
                            return;
                        }
                        for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                            let w = if E::is_weighted() {
                                labels.get(e).weight::<f64>()
                            } else {
                                1.
                            };
                            let d = du + w;
                            let v = targets.get(e).index();
                            let bits = d.to_bits();
                            if bits < dist[v].fetch_min(bits, Ordering::Relaxed) {
                                let b = ((d / delta) as usize).clamp(current, current + RING - 1);
                                buckets[b % RING].push(Ix::new(v));
                            }
                        }
                    };
                    // empty the current bucket, a chunk of nodes at a time, from every thread's
                    // share
                    {
                        let shares: Vec<_> = handoff.iter().map(|h| h.read().unwrap()).collect();
                        let mut starts = Vec::with_capacity(threads + 1);
                        starts.push(0);
                        for share in &shares {
                            starts.push(starts.last().unwrap() + share.len());
                        }
                        let total = starts[threads];
                        let chunk = CHUNK;
                        loop {
                            let at = cursor.fetch_add(chunk, Ordering::Relaxed);
                            if at >= total {
                                break;
                            }
                            let mut owner = starts.partition_point(|&s| s <= at) - 1;
                            for i in at..(at + chunk).min(total) {
                                while i >= starts[owner + 1] {
                                    owner += 1;
                                }
                                relax(shares[owner][i - starts[owner]].index(), &mut buckets);
                            }
                        }
                    }
                    // bucket fusion: the thread's own nodes of the current bucket, while few
                    let slot = current % RING;
                    while !buckets[slot].is_empty() && buckets[slot].len() < FUSION {
                        std::mem::swap(&mut fused, &mut buckets[slot]);
                        for u in fused.drain(..) {
                            relax(u.index(), &mut buckets);
                        }
                    }
                    barrier.wait();
                    if tid == 0 {
                        cursor.store(0, Ordering::Relaxed);
                    }
                    // the lowest bucket any thread holds nodes in
                    if let Some(b) =
                        (current..current + RING).find(|b| !buckets[b % RING].is_empty())
                    {
                        next_bucket.fetch_min(b, Ordering::Relaxed);
                    }
                    barrier.wait();
                    let next = next_bucket.load(Ordering::Relaxed);
                    if next == usize::MAX {
                        break;
                    }
                    {
                        let mut share = handoff[tid].write().unwrap();
                        share.clear();
                        std::mem::swap(&mut *share, &mut buckets[next % RING]);
                    }
                    current = next;
                    barrier.wait();
                    if tid == 0 {
                        next_bucket.store(usize::MAX, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    Ok(())
}
