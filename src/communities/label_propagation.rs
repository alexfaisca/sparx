//! Label propagation for large graphs, in parallel and deterministic: [`LabelPropagation`].
//!
//! Label propagation (Raghavan U. N., Albert R. and Kumara S., ["Near linear time algorithm to
//! detect community structures in large-scale networks"](https://doi.org/10.1103/PhysRevE.76.036106))
//! starts every node in its own community and has each, in turn, join the community most of its
//! neighbours' weight is in, until (almost) none moves. Run asynchronously in parallel, as is
//! fastest, its communities depend on the threads' timing; run synchronously (every node at
//! once), it oscillates on bipartite structure (two nodes swapping labels forever).
//!
//! This runs it *semi-synchronously* (Cordasco G. and Gargano L., ["Community detection via
//! semi-synchronous label propagation algorithms"](https://doi.org/10.1109/BASNA.2010.5730298)):
//! nodes are colored so that no two neighbours share a color, and the colors take turns, each's
//! nodes moving at once, in parallel. A node's neighbours are all of other colors, so none
//! moves while it does: what it sees is the same whatever the threads, and it doesn't oscillate.
//! With GVE-LPA's two shortcuts (Sahu S., ["GVE-LPA: Fast Label Propagation Algorithm (LPA) for
//! Community Detection in Shared Memory Setting"](https://arxiv.org/abs/2312.08140)): only nodes
//! a neighbour of which moved are looked at again (*pruning*), and it stops once a pass moves
//! no more than a `tolerance` of the nodes. And of Raghavan et al.'s: in the first passes
//! (*exploration*), a node's own label is only one of the labels of greatest weight, the tie
//! broken as any other's, so that labels spread through sparse graphs rather than freezing in
//! small communities (see [`LpaOptions::exploration`]); semi-synchronous rules after them make
//! it converge.
//!
//! The coloring is Jones and Plassmann's (["A parallel graph coloring
//! heuristic"](https://doi.org/10.1137/0914041)), in rounds: a node whose neighbours of higher
//! (hashed) priority all have colors takes the least color none of them has. Every round reads
//! only colors of rounds before, so the coloring is the same whatever the threads too.
//!
//! A node joins the label of greatest weight among its neighbours' (self-loops aside); of equal
//! ones, it keeps its own if it's one of them, else takes one by a hash of the node, the label
//! and the pass (as a random choice would, so that no label is favoured, but the same every
//! run). Communities are then split
//! into their connected parts (a label may have spread to parts that have since lost the nodes
//! linking them), numbered by their least node.

use crate::graph;
use crate::graph::cache::AccumulatorAtomic;
use crate::graph::csr::Csr;
use crate::graph::*;
use crate::shared_slice::*;
use crate::trails::astar::Reversible;
use crate::utils::SpinBarrier;

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

type Error = Box<dyn std::error::Error>;

/// How [`LabelPropagation`] runs; see [`Default`] for the defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LpaOptions {
    /// Passes over the colors, at most (exploration's included: with fewer than it, it may not
    /// have converged).
    pub max_iterations: usize,
    /// It stops once a pass moves at most this fraction of the nodes (0 to run until none
    /// moves).
    pub tolerance: f64,
    /// Passes in which a node's own label is only one of the labels of greatest weight, if it's
    /// one of them, the tie broken as any other (as Raghavan et al.'s): on sparse graphs, after
    /// the first pass most of a node's neighbours' labels are tied, and keeping its own would
    /// freeze small communities in place. After them, the own label is kept, so that it ends
    /// (and the tolerance applies). On a planted partition of a million nodes (64 blocks of mean
    /// degree 16, 20% of arcs across), 5 passes left 86 thousand communities, 10 left 37
    /// thousand, and 20 found the blocks (an NMI of 0.955); more changed nothing. With
    /// [`exploration_ends_early`](Self::exploration_ends_early), the most of them.
    pub exploration: usize,
    /// Exploration ends at the first of its passes that moves at most `tolerance` of the nodes
    /// (the same every run: every thread sees the pass's moves), rather than after
    /// [`exploration`](Self::exploration) passes: where exploring has stopped paying, as on graphs
    /// whose moves dwindle slowly, a faster run, for a slightly lower modularity (on a protein
    /// graph of 3.2M nodes, 9 passes rather than 21, and 0.394 rather than 0.413); where it
    /// does pay, as on the planted partition above (whose passes move a third of the nodes until
    /// the blocks are found), the same communities.
    pub exploration_ends_early: bool,
    /// On directed graphs, the arcs a node's label follows (on undirected ones, every edge).
    pub direction: LpaDirection,
    /// How nodes take turns (see [`LpaEngine`]).
    pub engine: LpaEngine,
}

/// How [`LabelPropagation`]'s nodes take turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LpaEngine {
    /// Colors take turns, each's nodes moving at once (no two neighbours of the same color):
    /// the same communities whatever the threads (see the [module docs](self)).
    #[default]
    SemiSynchronous,
    /// Every node moves in turn, seeing its neighbours' labels as they are, those moved earlier in
    /// the pass included (as Raghavan et al.'s, NetworKit's PLP and GVE-LPA): threads take the
    /// nodes in chunks, in their ids' order, labels changing in place (atomically). No coloring,
    /// and labels spread within a pass: 1.7x to 3.1x as fast as the semi-synchronous engine on
    /// graphs of millions of nodes. But the communities depend on the threads' timing (the same
    /// on one thread), and on partitions hard to find, their quality with it: on a planted
    /// partition with 30% of its arcs across, NMIs from 0.55 to 0.96 by the threads, where the
    /// semi-synchronous engine's are the same whatever they are.
    Asynchronous,
}

/// Which neighbours' labels a node of a directed graph counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LpaDirection {
    /// Its in- and out-neighbours' (directions ignored): as on the underlying undirected
    /// multigraph.
    #[default]
    Both,
    /// Its in-neighbours': labels spread along the arcs (as information flows).
    In,
    /// Its out-neighbours': labels spread against the arcs.
    Out,
}

impl Default for LpaOptions {
    /// 20 passes of exploration, then until a pass moves 5% of the nodes or fewer (as GVE-LPA
    /// stops), 40 passes at most.
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 0.05,
            exploration: 20,
            exploration_ends_early: false,
            engine: LpaEngine::SemiSynchronous,
            direction: LpaDirection::Both,
        }
    }
}

/// Label propagation's communities of an undirected [`GraphMemoryMap`] (see the [module
/// docs](self)), weighted or not (weights finite and non-negative). For a directed graph, run
/// it on its [`to_undirected`](GraphMemoryMap::to_undirected) one.
///
/// Memory: each node's label, color class position and pending flag, besides the result.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
pub struct LabelPropagation<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness = Undirected,
> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    communities: AbstractedProceduralMemoryMut<Ix>,
    count: usize,
    modularity: f64,
    /// The nodes each pass moved.
    moves: Vec<usize>,
    colors: usize,
}

/// SplitMix64.
#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Nodes a thread takes at a time, at most.
const CHUNK: usize = 1024;

/// The asynchronous engine's chunks of consecutive nodes, taken in a pseudo-random order each
/// pass: in the ids' order, a community settling first floods the next ones (whose nodes'
/// labels are still their own), as it did a planted partition's (NMI 0.906 rather than 0.989,
/// on a million nodes; one community, on one thread, of 1000 nodes in blocks of consecutive
/// ids). At most this many (rows read near each other: in the ids' order, up to 1.8x faster, as
/// labels settle sooner, but at that cost), and, on smaller graphs, down to 16: a 64th of a
/// thread's share of the nodes (a chunk spanning communities floods them as the ids' order).
const ASYNC_CHUNK: usize = 256;

/// A pass's order of `count` chunks: the `k`-th taken is `(k stride + offset) mod count`, a
/// permutation as `stride` is coprime with `count`; both pseudo-random, from the pass.
fn chunk_order(count: usize, pass: usize) -> (usize, usize) {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    if count <= 2 {
        return (1, pass % count.max(1));
    }
    let h = mix(pass as u64 ^ 0x5eed);
    let mut stride = (h as usize % count) | 1;
    while gcd(stride, count) != 1 {
        stride = (stride + 2) % count;
    }
    (stride, mix(h) as usize % count)
}

/// Runs `f(begin, end, scratch)` over `0..n` in chunks taken by `threads` threads as they finish
/// (each with its own `scratch`): `f`'s results, in chunk order.
fn chunked<T: Send, S>(
    n: usize,
    threads: usize,
    scratch: &(dyn Fn() -> S + Sync),
    f: &(dyn Fn(usize, usize, &mut S) -> T + Sync),
) -> Vec<T> {
    let chunk = (n / (threads * 8)).clamp(1, CHUNK);
    let count = n.div_ceil(chunk);
    let threads = threads.clamp(1, count.max(1));
    let next = AtomicUsize::new(0);
    let mut parts: Vec<(usize, T)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut scratch = scratch();
                    let mut mine = vec![];
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= count {
                            break mine;
                        }
                        let (begin, end) = (c * chunk, ((c + 1) * chunk).min(n));
                        mine.push((c, f(begin, end, &mut scratch)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    parts.sort_unstable_by_key(|p| p.0);
    parts.into_iter().map(|p| p.1).collect()
}

/// The nodes by color (Jones and Plassmann's, see the [module docs](self)), no two nodes linked
/// by an arc of any of `csrs` of the same color: the nodes, color after color (by node, within
/// one), and each color's first position, with an end.
fn color_classes<Ix: graph::IndexType, E: graph::E>(
    csrs: &[Csr<Ix, E>],
    n: usize,
    color_file: &str,
    threads: usize,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    const NONE: u32 = u32::MAX;
    let neighbours = |u: usize| {
        csrs.iter()
            .flat_map(move |c| c.targets[c.range(u)].iter().map(|v| v.index()))
    };
    // u before v if u's priority is higher
    let priority = |u: usize| (mix(u as u64), u);
    let mut colors = SharedSliceMut::<u32>::abst_mem_mut(color_file, n.max(1), true)?;
    colors.as_mut_slice().iter_mut().for_each(|c| *c = NONE);
    let shared = colors.shared_slice();
    // each node's neighbours of higher priority (colored before it), which it waits for; the
    // nodes waiting for none are colored first
    let waiting: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let roots: Vec<Vec<usize>> = chunked(n, threads, &|| (), &|begin, end, _| {
        (begin..end)
            .filter(|&u| {
                let pu = priority(u);
                let before = neighbours(u)
                    .filter(|&v| v != u && priority(v) > pu)
                    .count();
                waiting[u].store(before as u32, Ordering::Relaxed);
                before == 0
            })
            .collect::<Vec<_>>()
    });
    let mut ready: Vec<usize> = roots.concat();
    // round by round: the ready nodes (no two neighbours: of two, one waits for the other) each
    // take the least color its neighbours of higher priority (all colored, and only they) don't
    // have, then free those of lower priority it was the last to wait for. O(|E|) work in all,
    // and the same colors as recoloring round by round would give
    while !ready.is_empty() {
        let work = &ready;
        let freed: Vec<Vec<usize>> = chunked(
            work.len(),
            threads,
            &Vec::<usize>::new,
            &|begin, end, taken| {
                let mut colors = shared;
                let mut freed = vec![];
                for &u in &work[begin..end] {
                    for v in neighbours(u) {
                        let c = *colors.get(v);
                        if v != u && c != NONE {
                            let c = c as usize;
                            if c >= taken.len() {
                                taken.resize(c + 1, usize::MAX);
                            }
                            taken[c] = u;
                        }
                    }
                    let c = (0..).find(|&c| taken.get(c) != Some(&u)).unwrap();
                    *colors.get_mut(u) = c as u32;
                    let pu = priority(u);
                    for v in neighbours(u) {
                        if v != u
                            && priority(v) < pu
                            && waiting[v].fetch_sub(1, Ordering::Relaxed) == 1
                        {
                            freed.push(v);
                        }
                    }
                }
                freed
            },
        );
        ready = freed.concat();
    }
    // counting sort, by color then node
    let colors = colors.as_slice();
    let count = colors[..n]
        .iter()
        .map(|&c| c as usize + 1)
        .max()
        .unwrap_or(0);
    let mut starts = vec![0usize; count + 1];
    for &c in &colors[..n] {
        starts[c as usize + 1] += 1;
    }
    for c in 0..count {
        starts[c + 1] += starts[c];
    }
    let mut at = starts.clone();
    let mut order = vec![0usize; n];
    for (u, &c) in colors[..n].iter().enumerate() {
        order[at[c as usize]] = u;
        at[c as usize] += 1;
    }
    Ok((order, starts))
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible>
    LabelPropagation<'a, N, E, Ix, Dir>
{
    /// Finds `g`'s communities with [`LpaOptions::default`].
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix, Dir>) -> Result<Self, Error> {
        Self::with_options(g, LpaOptions::default())
    }

    /// Finds `g`'s communities as `options` say. On directed graphs, the transpose is built if
    /// it isn't yet.
    pub fn with_options(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        options: LpaOptions,
    ) -> Result<Self, Error> {
        if !(options.tolerance.is_finite() && (0. ..=1.).contains(&options.tolerance)) {
            return Err(format!(
                "error the tolerance must be in [0, 1], not {}",
                options.tolerance
            )
            .into());
        }
        // checks the weights finite and non-negative
        g.total_weight()?;
        let (n, threads) = (g.size(), g.thread_num().max(1));
        // the arcs out of nodes, and (directed) into them
        let out = Csr::of(g);
        let into = if Dir::DIRECTED {
            Some(Csr::of(Dir::reverse(g)?))
        } else {
            None
        };
        // whose labels a node counts, and whom it wakes when it moves (those counting its)
        type Csrs<'g, Ix, E> = Vec<Csr<'g, Ix, E>>;
        let (reads, wakes): (Csrs<Ix, E>, Csrs<Ix, E>) = match (into, options.direction) {
            (None, _) => (vec![out], vec![out]),
            (Some(into), LpaDirection::Both) => (vec![out, into], vec![out, into]),
            (Some(into), LpaDirection::In) => (vec![into], vec![out]),
            (Some(into), LpaDirection::Out) => (vec![out], vec![into]),
        };
        let both: Vec<Csr<Ix, E>> = into.map_or(vec![out], |into| vec![out, into]);
        let out_fn = g.build_cache_filename(CacheFile::LabelPropagation, None)?;
        let asynchronous = options.engine == LpaEngine::Asynchronous;
        // the nodes' turns: color after color, or (asynchronous) all at once, by id (no order)
        let (mut order, starts) = if asynchronous {
            (Vec::new(), vec![0, n])
        } else {
            let color_fn = g.build_cache_filename(CacheFile::LabelPropagation, Some(0))?;
            let classes = color_classes(&both, n, &color_fn, threads);
            g.cleanup_cache(CacheFile::LabelPropagation)?;
            classes?
        };
        let colors = starts.len() - 1;

        let mut labels = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, n.max(1), true)?;
        for (u, l) in labels.as_mut_slice().iter_mut().enumerate().take(n) {
            *l = Ix::new(u);
        }
        // whether a node is to be looked at (again), mapped as the labels are
        let pending_fn = g.build_cache_filename(CacheFile::LabelPropagation, Some(1))?;
        let mut pending_mem =
            SharedSliceMut::<AtomicBool>::abst_mem_mut(&pending_fn, n.max(1), true)?;
        for p in pending_mem.as_mut_slice().iter_mut() {
            *p = AtomicBool::new(true);
        }
        let pending = &pending_mem.as_slice()[..n];
        let cursors: Vec<AtomicUsize> = (0..colors).map(|_| AtomicUsize::new(0)).collect();
        // the nodes each pass moves, by its parity (see `shortest_path_tree`'s counts)
        let moved = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let moves = std::sync::Mutex::new(vec![]);
        let barrier = SpinBarrier::new(threads);
        // labels read and written atomically (relaxed: plain loads and stores, on x86), as
        // asynchronous passes move neighbours at once (semi-synchronous ones never do)
        let label_ptr = labels.as_mut_slice().as_mut_ptr();
        assert_eq!(label_ptr as usize % std::mem::align_of::<Ix::Atomic>(), 0);
        // (`Ix::Atomic`'s `Plain` is `Ix`: the same size, and here aligned)
        let atomics: &[Ix::Atomic] =
            unsafe { std::slice::from_raw_parts(label_ptr as *const Ix::Atomic, n) };
        let limit = (options.tolerance * n as f64).floor() as usize;

        std::thread::scope(|s| {
            for tid in 0..threads {
                let (order, starts, cursors, moved, moves, barrier) =
                    (&order, &starts, &cursors, &moved, &moves, &barrier);
                let (reads, wakes) = (&reads, &wakes);
                s.spawn(move || {
                    let labels = atomics;
                    let mut seen: Vec<(usize, f64)> = Vec::new();
                    // (unweighted: the labels alone, counted)
                    let mut seen_labels: Vec<usize> = Vec::new();
                    // (every thread sees each pass's moves: they end exploring together)
                    let mut explored = false;
                    for pass in 0..options.max_iterations {
                        let exploring = pass < options.exploration && !explored;
                        let mut mine = 0;
                        for c in 0..colors {
                            let (first, end) = (starts[c], starts[c + 1]);
                            let len = end - first;
                            // small colors: one thread (the others would mostly wait)
                            let workers = len.div_ceil(256).min(threads);
                            let chunk = if asynchronous {
                                (len / (threads * 64)).clamp(16, ASYNC_CHUNK)
                            } else {
                                (len / (workers * 4)).clamp(1, CHUNK)
                            };
                            let count = len.div_ceil(chunk);
                            // asynchronous passes over small graphs (small chunks, for the threads)
                            // take the chunks in an order of their own (a stride coprime with their
                            // number, from an offset), the same every run: in the ids' order, a
                            // community's label, settled first, could flood the next ones' nodes
                            // (all of a 1,000-node planted partition's, on one thread), and
                            // threads on adjacent small chunks contend for cache lines (permuted,
                            // twice as fast on 0.5M-node graphs on 32 threads). Large graphs' keep
                            // the ids' order: their passes 1.5x to 1.8x as fast (locality), and no
                            // better communities for permuting (planted partitions of 0.3M and 1M
                            // nodes, on 1 to 32 threads)
                            let permute = asynchronous && chunk < ASYNC_CHUNK;
                            let (stride, offset) = if permute {
                                chunk_order(count, pass)
                            } else {
                                (1, 0)
                            };
                            if tid < workers {
                                loop {
                                    let k = cursors[c].fetch_add(1, Ordering::Relaxed);
                                    if k >= count {
                                        break;
                                    }
                                    let at = ((k as u128 * stride as u128 + offset as u128)
                                        % count as u128)
                                        as usize
                                        * chunk;
                                    // (`i` is the node itself, asynchronous: `order` only
                                    // semi-synchronous, so no iterating over it)
                                    #[allow(clippy::needless_range_loop)]
                                    for i in first + at..first + (at + chunk).min(len) {
                                        let u = if asynchronous { i } else { order[i] };
                                        if !pending[u].swap(false, Ordering::Relaxed) {
                                            continue;
                                        }
                                        let own = labels[u].load_widened(Ordering::Relaxed);
                                        let salt = mix(u as u64 ^ mix(pass as u64));
                                        // the best label so far, its weight, and its hash (only
                                        // computed once another label ties it): of equals, the
                                        // least hash of the node, the label and the pass, which
                                        // is unbiased (the least label would flood the graph),
                                        // yet the same every run
                                        let mut best = (own, f64::NEG_INFINITY, None::<u64>);
                                        let mut own_w = f64::NEG_INFINITY;
                                        let mut consider = |l: usize, w: f64| {
                                            if l == own {
                                                own_w = w;
                                            }
                                            if w > best.1 {
                                                best = (l, w, None);
                                            } else if w == best.1 {
                                                let key = mix(mix(salt ^ l as u64));
                                                let best_key = *best.2.get_or_insert_with(|| {
                                                    mix(mix(salt ^ best.0 as u64))
                                                });
                                                if key < best_key {
                                                    best = (l, w, Some(key));
                                                }
                                            }
                                        };
                                        if E::is_weighted() {
                                            seen.clear();
                                            for csr in reads.iter() {
                                                for a in csr.range(u) {
                                                    let v = csr.targets[a].index();
                                                    if v != u {
                                                        let w = csr.weights[a].weight::<f64>();
                                                        seen.push((
                                                            labels[v]
                                                                .load_widened(Ordering::Relaxed),
                                                            w,
                                                        ));
                                                    }
                                                }
                                            }
                                            if seen.is_empty() {
                                                continue;
                                            }
                                            seen.sort_unstable_by_key(|p| p.0);
                                            let mut i = 0;
                                            while i < seen.len() {
                                                let l = seen[i].0;
                                                let mut w = 0.;
                                                while i < seen.len() && seen[i].0 == l {
                                                    w += seen[i].1;
                                                    i += 1;
                                                }
                                                consider(l, w);
                                            }
                                        } else {
                                            seen_labels.clear();
                                            for csr in reads.iter() {
                                                for &v in &csr.targets[csr.range(u)] {
                                                    let v = v.index();
                                                    if v != u {
                                                        seen_labels.push(
                                                            labels[v]
                                                                .load_widened(Ordering::Relaxed),
                                                        );
                                                    }
                                                }
                                            }
                                            if seen_labels.is_empty() {
                                                continue;
                                            }
                                            seen_labels.sort_unstable();
                                            let mut i = 0;
                                            while i < seen_labels.len() {
                                                let l = seen_labels[i];
                                                let run = seen_labels[i..]
                                                    .iter()
                                                    .take_while(|&&x| x == l)
                                                    .count();
                                                i += run;
                                                // (counts, exact as `f64`s: as their unit weights' sums)
                                                consider(l, run as f64);
                                            }
                                        }
                                        let (best, best_w) = (best.0, best.1);
                                        // exploring, ties are broken by the hash, the own
                                        // label included; then, it's kept (so that it ends)
                                        if best == own || (!exploring && own_w == best_w) {
                                            continue;
                                        }
                                        // (a node id: it fits)
                                        let _ = labels[u].checked_store(best, Ordering::Relaxed);
                                        mine += 1;
                                        for csr in wakes.iter() {
                                            for &v in &csr.targets[csr.range(u)] {
                                                // (read first: most are pending already, and
                                                // writing them would bounce their cache lines)
                                                let p = &pending[v.index()];
                                                if !p.load(Ordering::Relaxed) {
                                                    p.store(true, Ordering::Relaxed);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            barrier.wait();
                        }
                        moved[pass % 2].fetch_add(mine, Ordering::Relaxed);
                        barrier.wait();
                        if tid == 0 {
                            for cursor in cursors.iter() {
                                cursor.store(0, Ordering::Relaxed);
                            }
                            moved[(pass + 1) % 2].store(0, Ordering::Relaxed);
                        }
                        let total = moved[pass % 2].load(Ordering::Relaxed);
                        if tid == 0 {
                            moves.lock().unwrap().push(total);
                        }
                        barrier.wait();
                        if !exploring && total <= limit {
                            break;
                        }
                        if exploring && options.exploration_ends_early && total <= limit {
                            explored = true;
                        }
                    }
                });
            }
        });
        let moves = moves.into_inner().unwrap();

        // (weakly) connected parts: arcs' ends are joined either way
        let count = super::split_disconnected(
            threads,
            out.offsets,
            out.targets,
            &mut labels.as_mut_slice()[..n],
        )?;
        labels.flush()?;
        let communities = &labels.as_slice()[..n];
        let modularity = if Dir::DIRECTED {
            super::directed_louvain::directed_modularity(g, communities)?
        } else {
            // `Dir` is `Undirected`: `g` is a `Graph`
            let g =
                unsafe { &*(g as *const GraphMemoryMap<N, E, Ix, Dir> as *const Graph<N, E, Ix>) };
            if E::is_weighted() {
                g.modularity(communities, count)?
            } else if order.len() >= count {
                // the color classes' order is done with: its `n` words (`count` at most needed)
                // hold the communities' totals, rather than memory mapped anew
                let totals: &mut [f64] = bytemuck::cast_slice_mut(&mut order[..]);
                g.modularity_with(communities, count, totals, None)?
            } else {
                // (asynchronous: no order)
                g.modularity(communities, count)?
            }
        };
        Ok(Self {
            g,
            communities: labels,
            count,
            modularity,
            moves,
            colors,
        })
    }

    /// Each node's community, numbered `0..count` by their least node.
    pub fn communities(&self) -> &[Ix] {
        &self.communities.as_slice()[..self.g.size()]
    }

    /// The number of communities.
    pub fn community_count(&self) -> usize {
        self.count
    }

    /// The communities' modularity: directed (Leicht and Newman's), on directed graphs.
    pub fn partition_modularity(&self) -> f64 {
        self.modularity
    }

    /// The nodes each pass moved (as many as passes run).
    pub fn moves(&self) -> &[usize] {
        &self.moves
    }

    /// The colors the nodes took turns by.
    pub fn colors(&self) -> usize {
        self.colors
    }

    /// Removes the communities from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Error> {
        let this = ManuallyDrop::new(self);
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::LabelPropagation, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::communities::spectral::test::{nmi, planted};
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};

    fn graph<E: graph::E>(
        name: &str,
        n: usize,
        edges: &[(usize, usize, f64)],
        threads: u8,
    ) -> TestGraph<VoidLabel, E, usize> {
        let dir = unique_test_dir("lpa");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let field = if E::is_weighted() { "real" } else { "pattern" };
        let mut text = format!(
            "%%MatrixMarket matrix coordinate {field} symmetric\n{n} {n} {}\n",
            edges.len()
        );
        for &(u, v, w) in edges {
            let (u, v) = (u.max(v), u.min(v));
            text += &if E::is_weighted() {
                format!("{} {} {w}\n", u + 1, v + 1)
            } else {
                format!("{} {}\n", u + 1, v + 1)
            };
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("lpa_{name}_{}", unique_run_tag());
        TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    fn labels<Ix: graph::IndexType>(c: &[Ix]) -> Vec<usize> {
        c.iter().map(|x| x.index()).collect()
    }

    /// Communities are connected, numbered by their least node, and their modularity is the
    /// graph's measure of them.
    fn check<E: graph::E>(
        g: &Graph<VoidLabel, E, usize>,
        lpa: &LabelPropagation<VoidLabel, E, usize>,
    ) {
        let c = labels(lpa.communities());
        let mut firsts = vec![usize::MAX; lpa.community_count()];
        for (u, &l) in c.iter().enumerate() {
            firsts[l] = firsts[l].min(u);
        }
        assert!(firsts.windows(2).all(|f| f[0] < f[1]));
        let mut split = c.clone();
        let offsets = SharedSlice::<usize>::new(g.offsets_ptr(), g.offsets_size());
        let targets = SharedSlice::<usize>::new(g.neighbours_ptr(), g.width());
        let count = crate::communities::split_disconnected(
            2,
            offsets.as_slice(),
            targets.as_slice(),
            &mut split,
        )
        .unwrap();
        assert_eq!((count, split), (lpa.community_count(), c));
        assert_eq!(
            lpa.partition_modularity(),
            g.modularity(lpa.communities(), lpa.community_count())
                .unwrap()
        );
    }

    #[test]
    fn finds_planted_partitions_the_same_whatever_the_threads() {
        let (edges, truth) = planted(5, 200, 0.1, 0.003, 11);
        let mut found = vec![];
        for threads in [1u8, 4, 7] {
            let g = graph::<()>(&format!("planted_{threads}"), 1000, &edges, threads);
            let mut lpa = LabelPropagation::new(&g).unwrap();
            check(&g, &lpa);
            assert!(
                nmi(&labels(lpa.communities()), &truth) > 0.95,
                "{threads} threads"
            );
            found.push((
                labels(lpa.communities()),
                lpa.moves().to_vec(),
                lpa.colors(),
            ));
            lpa.drop_cache().unwrap();
        }
        assert!(found.windows(2).all(|f| f[0] == f[1]));
    }

    /// Modularity in lent memory is modularity in mapped memory, unweighted or weighted, the
    /// same whatever the threads; memory too short is an error.
    #[test]
    fn modularity_with_lent_memory() {
        let (edges, _) = planted(4, 150, 0.1, 0.01, 21);
        let weighted: Vec<_> = edges
            .iter()
            .enumerate()
            .map(|(i, &(u, v, _))| (u, v, 0.5 + (i % 7) as f64))
            .collect();
        let labels: Vec<usize> = (0..600).map(|u| (u * 7 + u / 150) % 9).collect();
        let communities: Vec<usize> = labels.clone();
        for threads in [1u8, 3, 8] {
            let g = graph::<()>(&format!("lent_{threads}"), 600, &edges, threads);
            let mapped = g.modularity(&communities, 9).unwrap();
            let mut totals = vec![0f64; 9];
            let lent = g
                .modularity_with(&communities, 9, &mut totals, None)
                .unwrap();
            assert_eq!(mapped, lent);
            assert!(
                g.modularity_with(&communities, 9, &mut [0f64; 8], None)
                    .is_err()
            );

            let w = graph::<Weighted<f64>>(&format!("lent_w_{threads}"), 600, &weighted, threads);
            let mapped = w.modularity(&communities, 9).unwrap();
            let (mut totals, mut degrees) = (vec![0f64; 9], vec![0f64; 600]);
            let lent = w
                .modularity_with(&communities, 9, &mut totals, Some(&mut degrees))
                .unwrap();
            assert_eq!(mapped, lent);
            // weighted edges need the degrees' memory
            assert!(
                w.modularity_with(&communities, 9, &mut totals, None)
                    .is_err()
            );
            assert!(
                w.modularity_with(&communities, 9, &mut totals, Some(&mut [0f64; 10]))
                    .is_err()
            );
        }
    }

    /// Exploration ending early: no longer than exploring every pass, it still finds planted
    /// partitions, the same whatever the threads; and it ends at the first exploring pass that
    /// moves at most `tolerance` of the nodes.
    #[test]
    fn exploration_ending_early() {
        let (edges, truth) = planted(5, 200, 0.1, 0.003, 11);
        let options = LpaOptions {
            exploration_ends_early: true,
            ..LpaOptions::default()
        };
        let mut found = vec![];
        for threads in [1u8, 4, 7] {
            let g = graph::<()>(&format!("early_{threads}"), 1000, &edges, threads);
            let mut lpa = LabelPropagation::with_options(&g, options).unwrap();
            check(&g, &lpa);
            assert!(
                nmi(&labels(lpa.communities()), &truth) > 0.95,
                "{threads} threads"
            );
            let mut full = LabelPropagation::new(&g).unwrap();
            assert!(lpa.moves().len() <= full.moves().len());
            // exploring stops at the first pass moving 5% of the nodes or fewer: the run at
            // the first (exploring or not) after it
            let limit = (options.tolerance * 1000.) as usize;
            let first = lpa.moves().iter().position(|&m| m <= limit).unwrap();
            assert!(lpa.moves().len() <= first + 2, "{:?}", lpa.moves());
            found.push((labels(lpa.communities()), lpa.moves().to_vec()));
            lpa.drop_cache().unwrap();
            full.drop_cache().unwrap();
        }
        assert!(found.windows(2).all(|f| f[0] == f[1]));
    }

    /// The asynchronous engine finds planted partitions (on several threads, whose timing it
    /// depends on), repeats itself on one thread, moves no more than the semi-synchronous
    /// engine passes, and works on directed graphs too.
    #[test]
    fn asynchronous_engine() {
        let (edges, truth) = planted(5, 200, 0.1, 0.003, 11);
        let options = LpaOptions {
            engine: LpaEngine::Asynchronous,
            ..LpaOptions::default()
        };
        let mut repeated = vec![];
        for (run, threads) in [(0, 1u8), (1, 1), (2, 6)] {
            let g = graph::<()>(&format!("async_{run}"), 1000, &edges, threads);
            let mut lpa = LabelPropagation::with_options(&g, options).unwrap();
            check(&g, &lpa);
            assert!(
                nmi(&labels(lpa.communities()), &truth) > 0.95,
                "{threads} threads"
            );
            // no coloring
            assert_eq!(lpa.colors(), 1);
            if threads == 1 {
                repeated.push(labels(lpa.communities()));
            }
            lpa.drop_cache().unwrap();
        }
        assert_eq!(repeated[0], repeated[1]);

        let (arcs, truth) = directed_planted();
        let g = digraph("async_directed", 750, &arcs, 4);
        let mut lpa = LabelPropagation::with_options(&*g, options).unwrap();
        assert!(nmi(&labels(lpa.communities()), &truth) > 0.9);
        let q = crate::communities::directed_louvain::directed_modularity(&*g, lpa.communities())
            .unwrap();
        assert_eq!(lpa.partition_modularity(), q);
        lpa.drop_cache().unwrap();
    }

    #[test]
    fn colorings_are_proper() {
        let (edges, _) = planted(3, 300, 0.05, 0.01, 12);
        let mut edges = edges;
        edges.extend([(0, 0, 1.), (5, 5, 1.)]);
        let g = graph::<()>("coloring", 900, &edges, 5);
        let file = g
            .build_cache_filename(CacheFile::LabelPropagation, Some(0))
            .unwrap();
        let (order, starts) = color_classes(&[Csr::of(&*g)], 900, &file, 5).unwrap();
        g.cleanup_cache(CacheFile::LabelPropagation).unwrap();
        let mut color = vec![usize::MAX; 900];
        for c in 0..starts.len() - 1 {
            for &u in &order[starts[c]..starts[c + 1]] {
                color[u] = c;
            }
        }
        assert!(color.iter().all(|&c| c != usize::MAX));
        for &(u, v, _) in &edges {
            assert!(u == v || color[u] != color[v], "{u} {v}");
        }
    }

    /// Complete bipartite graphs, on which synchronous label propagation swaps the sides'
    /// labels forever: it ends, with no moves left.
    #[test]
    fn bipartite_graphs_dont_oscillate() {
        let edges: Vec<_> = (0..20)
            .flat_map(|u| (20..40).map(move |v| (u, v, 1.)))
            .collect();
        let g = graph::<()>("bipartite", 40, &edges, 4);
        let options = LpaOptions {
            max_iterations: 100,
            tolerance: 0.,
            ..LpaOptions::default()
        };
        let mut lpa = LabelPropagation::with_options(&g, options).unwrap();
        assert_eq!(*lpa.moves().last().unwrap(), 0);
        assert!(lpa.moves().len() < 100);
        assert_eq!(lpa.community_count(), 1);
        lpa.drop_cache().unwrap();
    }

    #[test]
    fn separates_cliques_and_follows_weights() {
        let sizes = [10, 25, 6, 18];
        let (mut edges, mut truth, mut start) = (vec![], vec![], 0);
        for (c, &s) in sizes.iter().enumerate() {
            for u in start..start + s {
                truth.push(c);
                for v in u + 1..start + s {
                    edges.push((u, v, 1.));
                }
            }
            start += s;
        }
        // the cliques, linked by single edges
        edges.extend([(0, 10, 1.), (10, 35, 1.), (35, 41, 1.)]);
        let g = graph::<()>("cliques", start, &edges, 3);
        let mut lpa = LabelPropagation::new(&g).unwrap();
        assert_eq!(labels(lpa.communities()), truth);
        lpa.drop_cache().unwrap();

        // weights alone make the blocks (dense enough for label propagation, which on sparse
        // graphs freezes into many small communities, as networkx's does)
        let (mut edges, truth) = planted(3, 150, 0.12, 0.12, 13);
        for e in edges.iter_mut() {
            e.2 = if truth[e.0] == truth[e.1] { 10. } else { 0.3 };
        }
        let w = graph::<Weighted<f64>>("weights", 450, &edges, 4);
        let mut lpa = LabelPropagation::with_options(
            &w,
            LpaOptions {
                max_iterations: 50,
                tolerance: 0.,
                ..LpaOptions::default()
            },
        )
        .unwrap();
        check(&w, &lpa);
        let score = nmi(&labels(lpa.communities()), &truth);
        assert!(
            score > 0.9,
            "nmi {score}, {} communities",
            lpa.community_count()
        );
        lpa.drop_cache().unwrap();
        let u = graph::<()>("no_weights", 450, &edges, 4);
        let mut lpa = LabelPropagation::new(&u).unwrap();
        assert!(nmi(&labels(lpa.communities()), &truth) < 0.2);
        lpa.drop_cache().unwrap();
    }

    #[test]
    fn bad_tolerances_are_errors() {
        let g = graph::<()>("bad", 3, &[(0, 1, 1.)], 2);
        for tolerance in [-0.1, 1.5, f64::NAN] {
            let options = LpaOptions {
                tolerance,
                ..LpaOptions::default()
            };
            assert!(LabelPropagation::with_options(&g, options).is_err());
        }
        let g = graph::<Weighted<f64>>("negative", 3, &[(0, 1, -1.)], 2);
        assert!(LabelPropagation::new(&g).is_err());
    }

    fn digraph(
        name: &str,
        n: usize,
        arcs: &[(usize, usize)],
        threads: u8,
    ) -> TestGraph<VoidLabel, (), usize, Directed> {
        let dir = unique_test_dir("lpa_directed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut text = format!(
            "%%MatrixMarket matrix coordinate pattern general\n{n} {n} {}\n",
            arcs.len()
        );
        for &(u, v) in arcs {
            text += &format!("{} {}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("lpa_directed_{name}_{}", unique_run_tag());
        TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    /// Planted blocks, each edge an arc one way or the other (by a hash).
    fn directed_planted() -> (Vec<(usize, usize)>, Vec<usize>) {
        let (edges, truth) = planted(5, 150, 0.15, 0.004, 41);
        let arcs = edges
            .iter()
            .map(|&(u, v, _)| {
                if mix((u * 1000 + v) as u64) & 1 == 0 {
                    (u, v)
                } else {
                    (v, u)
                }
            })
            .collect();
        (arcs, truth)
    }

    /// Directed graphs' communities, by any direction: planted blocks found, the same whatever
    /// the threads, with their directed modularity.
    #[test]
    fn directed_graphs_by_every_direction() {
        let (arcs, truth) = directed_planted();
        for direction in [LpaDirection::Both, LpaDirection::In, LpaDirection::Out] {
            let mut found = vec![];
            for threads in [1u8, 5] {
                let g = digraph(&format!("dir_{direction:?}_{threads}"), 750, &arcs, threads);
                let options = LpaOptions {
                    direction,
                    ..LpaOptions::default()
                };
                let mut lpa = LabelPropagation::with_options(&*g, options).unwrap();
                assert!(
                    nmi(&labels(lpa.communities()), &truth) > 0.95,
                    "{direction:?}"
                );
                let q = crate::communities::directed_louvain::directed_modularity(
                    &*g,
                    lpa.communities(),
                )
                .unwrap();
                assert_eq!(lpa.partition_modularity(), q);
                found.push(labels(lpa.communities()));
                lpa.drop_cache().unwrap();
            }
            assert_eq!(found[0], found[1], "{direction:?}");
        }
    }

    /// Both directions are the underlying undirected multigraph's (each arc an edge); out-arcs
    /// are the transpose's in-arcs: the same communities, exactly.
    #[test]
    fn directions_are_the_graphs_they_follow() {
        let (arcs, _) = directed_planted();
        let g = digraph("equivalences", 750, &arcs, 4);
        let both = LabelPropagation::with_options(&*g, LpaOptions::default()).unwrap();
        let undirected = g
            .to_undirected_with(crate::graph::UndirectedMode::Each)
            .unwrap();
        let mut u = LabelPropagation::new(undirected).unwrap();
        assert_eq!(labels(both.communities()), labels(u.communities()));
        u.drop_cache().unwrap();
        let out = LabelPropagation::with_options(
            &*g,
            LpaOptions {
                direction: LpaDirection::Out,
                ..LpaOptions::default()
            },
        )
        .unwrap();
        let transpose = g.transpose().unwrap();
        let mut into = LabelPropagation::with_options(
            transpose,
            LpaOptions {
                direction: LpaDirection::In,
                ..LpaOptions::default()
            },
        )
        .unwrap();
        assert_eq!(labels(out.communities()), labels(into.communities()));
        into.drop_cache().unwrap();
    }
}
