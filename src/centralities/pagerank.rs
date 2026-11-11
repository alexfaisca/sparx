use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicUsize, Ordering};

/// PageRank of every node of a [`GraphMemoryMap`] instance, following its arcs (on undirected
/// graphs, its edges both ways), as networkx's `pagerank`: with damping factor `alpha`, the
/// rank of nodes with no arc out (dangling nodes) spread evenly over all nodes, and iterations
/// until the ranks' total (L1) change is below `tolerance`. Parallel arcs count once each,
/// self-loops too.
///
/// Unlike networkx's, the tolerance isn't multiplied by the number of nodes: ranks sum to 1, so
/// their total change is at most 2, and on graphs of millions of nodes networkx's criterion
/// would hold from the first iteration. As ranks change at most `alpha` times as much from one
/// iteration to the next, the default tolerance (`1e-6`, with `alpha = 0.85`) is reached in at
/// most 89 iterations. networkx's results are those of a tolerance `|V|` times theirs.
///
/// Computed with the *Partition-Centric Processing Methodology* described in ["Accelerating
/// PageRank using Partition-Centric Processing"](https://doi.org/10.48550/arXiv.1709.07122) by
/// Lakhotia K., Kannan R. and Prasanna V.: nodes are split in partitions of consecutive ids
/// whose ranks fit in a core's cache, and each iteration first scatters every node's
/// contribution once to each partition it has arcs into, in bins written sequentially, then
/// gathers each partition's bin into its nodes' ranks, which stay in cache: instead of a random
/// access to memory per arc, as by pulling contributions from neighbours, it streams through
/// memory. The ranks are the same whatever the number of threads, bit for bit.
///
/// Undirected graphs whose nodes' contributions fit in the cache (see [`PULL_MAX_BYTES`]) are
/// ranked by each node summing its neighbours' contributions instead, which needs no bins and is
/// faster there, for the same ranks, bit for bit: see [`PageRankEngine`] to choose either.
///
/// Weighted arcs (see [`E::is_weighted`]) are followed in proportion to
/// their weight, as networkx's `pagerank` does with `weight`: node `u`'s rank goes to `v` in
/// proportion to `w(u, v) / Σ_x w(u, x)`, and nodes whose arcs out all weigh `0` are dangling.
/// Weights must be finite and non-negative. Weighted graphs are always ranked partition-centric:
/// pulling reads a node's in-arcs' weights off its out-arcs, which only holds for symmetric
/// weights.
///
/// Scratch memory: at most `4 * |E| + size_of::<W>() * (|E| + |V|)` bytes and a few MB
/// (partition-centric; `size_of::<W>() * (|E| + |V|)` more for weighted arcs), or
/// `2 * size_of::<W>() * |V|` bytes (pulling).
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct PageRank<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    W: Weight = f64,
    Dir: Directedness = Undirected,
> {
    /// Graph whose nodes are ranked.
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Memmapped slice containing the rank of each node.
    ranks: AbstractedProceduralMemoryMut<W>,
    /// Iterations computed.
    iterations: usize,
}

/// How [`PageRank`] is computed: both give the same ranks, bit for bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PageRankEngine {
    /// [`Pull`](Self::Pull) for undirected graphs whose nodes' contributions (a `W` each) take
    /// at most [`PULL_MAX_BYTES`], [`PartitionCentric`](Self::PartitionCentric) otherwise. The
    /// default.
    #[default]
    Auto,
    /// Each node sums its neighbours' contributions: no other memory than theirs, the fastest
    /// while they fit in the cache, but a random access to memory per edge once they don't.
    /// Undirected graphs only (a node's neighbours, in directed graphs, are those its arcs go to).
    Pull,
    /// The *Partition-Centric Processing Methodology* (see [`PageRank`]): the fastest once the
    /// nodes' contributions don't fit in the cache.
    PartitionCentric,
}

/// The largest contributions (a `W` per node) [`PageRankEngine::Auto`] pulls: 8 MB (with `f64`,
/// 1M nodes). Measured on a CPU with a 20 MB last level cache, pulling was faster up to 8 MB of
/// contributions, partition-centric from 12 MB on (by 1.2 to 1.6 times, on random graphs, but
/// barely on graphs whose neighbours' ids are close, which pulling reads near each other).
pub const PULL_MAX_BYTES: usize = 8 << 20;

/// Flag of an arc's destination in a bin, marking it the first of its source node's arcs into
/// the bin's partition: the source's contribution is the bin's next value.
const FIRST: u32 = 1 << 31;
/// Bounds of the base-2 logarithm of a partition's number of nodes: with 2^15 nodes, a
/// partition's ranks take 128 or 256 KB, a core's L2 cache.
const MIN_SHIFT: u32 = 10;
const MAX_SHIFT: u32 = 15;
/// Partitions to aim for, so that small graphs have enough to share between threads.
const TARGET_PARTITIONS: usize = 256;
/// The most partitions, so that the bins' offsets (one per pair of partitions) take a few MB
/// at most: larger graphs have larger partitions.
const MAX_PARTITIONS: usize = 2048;

/// Base-2 logarithm of the number of nodes of a partition of a graph of `n` nodes: it depends
/// on `n` only (so that the ranks, summed partition by partition, don't depend on the threads).
fn partition_shift(n: usize) -> u32 {
    let log2 = |x: usize| x.next_power_of_two().trailing_zeros();
    log2(n.div_ceil(TARGET_PARTITIONS))
        .clamp(MIN_SHIFT, MAX_SHIFT)
        .max(log2(n.div_ceil(MAX_PARTITIONS)))
}

/// Runs `f(thread)` on `threads` threads.
fn parallel<F: Fn(usize) -> Result<(), String> + Sync>(
    threads: usize,
    f: F,
) -> Result<(), Box<dyn std::error::Error>> {
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let f = &f;
                scope.spawn(move |_| f(t))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().map_err(|e| format!("{e:?}"))?)
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })?
    .map_err(|e| e.into())
}

/// The bins of every partition. For each destination partition: the destinations (their
/// offsets in the partition, the first of each source node's marked [`FIRST`]) of the arcs into
/// it, by source partition and source node; and room for the source nodes' contributions, one
/// per run of arcs.
struct Layout<W: Weight> {
    ids: AbstractedProceduralMemoryMut<u32>,
    values: AbstractedProceduralMemoryMut<W>,
    /// Where each destination partition's bin starts in `ids` and in `values`, and where they
    /// end (`partitions + 1` each).
    id_starts: Vec<usize>,
    value_starts: Vec<usize>,
    /// Where each source partition's contributions start in each destination partition's bin
    /// (source partition major).
    value_cursors: AbstractedProceduralMemoryMut<usize>,
    /// For weighted arcs, each arc's weight, beside its destination in `ids`: the arc carries
    /// its source's contribution times it.
    weights: Option<AbstractedProceduralMemoryMut<W>>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    PageRank<'a, N, E, Ix, W, Dir>
{
    /// networkx's default damping factor.
    pub const DEFAULT_ALPHA: f64 = 0.85;
    /// The default tolerance (networkx's, but not multiplied by the number of nodes).
    pub const DEFAULT_TOLERANCE: f64 = 1e-6;
    /// networkx's default maximum number of iterations.
    pub const DEFAULT_MAX_ITERATIONS: usize = 100;

    /// Computes the PageRank of every node, with damping factor 0.85, tolerance `1e-6` (see the
    /// type's documentation), at most 100 iterations, networkx's defaults.
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix, Dir>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_conf(
            g,
            Self::DEFAULT_ALPHA,
            Self::DEFAULT_TOLERANCE,
            Self::DEFAULT_MAX_ITERATIONS,
        )
    }

    /// Computes the PageRank of every node.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance whose nodes are ranked.
    /// * `alpha` --- the damping factor, in `[0, 1]`.
    /// * `tolerance` --- iterations stop once the ranks' total change is below it.
    /// * `max_iterations` --- not converging within it is an error.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        alpha: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_engine(g, alpha, tolerance, max_iterations, PageRankEngine::Auto)
    }

    /// Computes the PageRank of every node (see [`new_with_conf`](Self::new_with_conf)) with a
    /// given [`PageRankEngine`]: [`PageRankEngine::Pull`] is an error on directed graphs.
    pub fn new_with_engine(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        alpha: f64,
        tolerance: f64,
        max_iterations: usize,
        engine: PageRankEngine,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if engine == PageRankEngine::Pull && Dir::DIRECTED {
            return Err("error PageRank pulls contributions on undirected graphs only".into());
        }
        if engine == PageRankEngine::Pull && E::is_weighted() {
            return Err("error PageRank pulls contributions on unweighted graphs only (a node's in-arcs' weights are read off its out-arcs, right for symmetric weights only): rank weighted graphs partition-centric".into());
        }
        if !(0. ..=1.).contains(&alpha) {
            return Err(
                format!("error PageRank's damping factor must be in [0, 1], not {alpha}").into(),
            );
        }
        if tolerance.is_nan() || tolerance <= 0. {
            return Err(
                format!("error PageRank's tolerance must be positive, not {tolerance}").into(),
            );
        }
        if max_iterations == 0 {
            return Err("error PageRank needs at least one iteration".into());
        }
        let out_fn = g.build_cache_filename(CacheFile::PageRank, None)?;
        let ranks = SharedSliceMut::<W>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut pagerank = Self {
            g,
            ranks,
            iterations: 0,
        };
        let pull = match engine {
            PageRankEngine::Auto => {
                !Dir::DIRECTED
                    && !E::is_weighted()
                    && g.size().saturating_mul(size_of::<W>()) <= PULL_MAX_BYTES
            }
            PageRankEngine::Pull => true,
            PageRankEngine::PartitionCentric => false,
        };
        let computed = pagerank.compute(alpha, tolerance, max_iterations, pull);
        g.cleanup_cache(CacheFile::PageRank)?;
        match computed {
            Ok(iterations) => {
                pagerank.iterations = iterations;
                pagerank.ranks.flush_async()?;
                Ok(pagerank)
            }
            Err(e) => {
                let _ = pagerank.drop_cache();
                Err(e)
            }
        }
    }

    /// Returns the rank of a given node.
    pub fn rank(&self, node: usize) -> W {
        assert!(node < self.g.size());
        *self.ranks.get(node)
    }

    /// Returns a slice containing the rank of each node.
    pub fn ranks(&self) -> &[W] {
        self.ranks.as_slice()
    }

    /// Returns the number of iterations computed until convergence.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        this.g.cleanup_cache(CacheFile::PageRank)?;
        let out_fn = this.g.build_cache_filename(CacheFile::PageRank, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }

    /// A scratch array of `len` (at least one) elements.
    fn scratch<T>(
        &self,
        seq: usize,
        len: usize,
    ) -> Result<AbstractedProceduralMemoryMut<T>, Box<dyn std::error::Error>> {
        let file = self
            .g
            .build_cache_filename(CacheFile::PageRank, Some(seq))?;
        SharedSliceMut::<T>::abst_mem_mut(&file, len.max(1), true)
    }

    /// Each node's total out-weight (summed in `f64`), in scratch file `seq`, after checking every
    /// weight is finite and non-negative.
    fn out_weights(
        &self,
        seq: usize,
    ) -> Result<AbstractedProceduralMemoryMut<W>, Box<dyn std::error::Error>> {
        let n = self.g.size();
        let mem = self.scratch::<W>(seq, n)?;
        let offsets = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let labels = SharedSlice::<E>::new(self.g.edge_labels_ptr(), self.g.width());
        let threads = self.g.thread_num().max(1);
        let load = n.div_ceil(threads);
        let out = mem.shared_slice();
        parallel(threads, |t| {
            let mut out = out;
            for u in (t * load).min(n)..((t + 1) * load).min(n) {
                let mut total = 0f64;
                for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                    let label = labels.get(e);
                    let w = label.weight::<f64>();
                    if !(w.is_finite() && w >= 0.) {
                        return Err(format!(
                            "error PageRank's arc weights must be finite and non-negative, but one is {label:?}"
                        ));
                    }
                    total += w;
                }
                *out.get_mut(u) = W::from_f64(total);
            }
            Ok(())
        })?;
        Ok(mem)
    }

    /// The graph's bins (see [`Layout`]), filled with destinations.
    fn layout(&self) -> Result<Layout<W>, Box<dyn std::error::Error>> {
        let n = self.g.size();
        let shift = partition_shift(n);
        if shift > 31 {
            return Err(format!("error PageRank of {n} nodes: too many").into());
        }
        let k = n.div_ceil(1 << shift);
        let threads = self.g.thread_num().min(k);
        let offsets = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width());
        let partition = |p: usize| (p << shift)..std::cmp::min((p + 1) << shift, n);

        // the arcs, and runs of arcs, from each source partition into each destination one
        let arc_cursors = self.scratch::<usize>(0, k * k)?;
        let value_cursors = self.scratch::<usize>(1, k * k)?;
        let (arcs, runs) = (arc_cursors.shared_slice(), value_cursors.shared_slice());
        let next = AtomicUsize::new(0);
        parallel(threads, |_| {
            let (mut arcs, mut runs) = (arcs, runs);
            loop {
                let ps = next.fetch_add(1, Ordering::Relaxed);
                if ps >= k {
                    return Ok(());
                }
                for u in partition(ps) {
                    let mut last = usize::MAX;
                    for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                        let pd = neighbours.get(e).index() >> shift;
                        *arcs.get_mut(ps * k + pd) += 1;
                        if pd != last {
                            *runs.get_mut(ps * k + pd) += 1;
                            last = pd;
                        }
                    }
                }
            }
        })?;

        // destination partitions' bins one after another, each by source partition
        let (mut arcs, mut runs) = (arcs, runs);
        let (mut id_starts, mut value_starts) =
            (Vec::with_capacity(k + 1), Vec::with_capacity(k + 1));
        let (mut arc_total, mut run_total) = (0, 0);
        for pd in 0..k {
            id_starts.push(arc_total);
            value_starts.push(run_total);
            for ps in 0..k {
                let (a, r) = (*arcs.get(ps * k + pd), *runs.get(ps * k + pd));
                *arcs.get_mut(ps * k + pd) = arc_total;
                *runs.get_mut(ps * k + pd) = run_total;
                arc_total += a;
                run_total += r;
            }
        }
        id_starts.push(arc_total);
        value_starts.push(run_total);

        // the destinations (and weights), in each source partition's place of each bin
        let ids = self.scratch::<u32>(2, arc_total)?;
        let values = self.scratch::<W>(3, run_total)?;
        let weights = if E::is_weighted() {
            Some(self.scratch::<W>(7, arc_total)?)
        } else {
            None
        };
        let labels = weights
            .as_ref()
            .map(|_| SharedSlice::<E>::new(self.g.edge_labels_ptr(), self.g.width()));
        let ids_slice = ids.shared_slice();
        let weights_slice = weights.as_ref().map(|w| w.shared_slice());
        let next = AtomicUsize::new(0);
        parallel(threads, |_| {
            let mut ids = ids_slice;
            let mut weights = weights_slice;
            let mut cursor = vec![0usize; k];
            loop {
                let ps = next.fetch_add(1, Ordering::Relaxed);
                if ps >= k {
                    return Ok(());
                }
                for (pd, c) in cursor.iter_mut().enumerate() {
                    *c = *arcs.get(ps * k + pd);
                }
                for u in partition(ps) {
                    let mut last = usize::MAX;
                    for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                        let v = neighbours.get(e).index();
                        let pd = v >> shift;
                        let id = (v & ((1 << shift) - 1)) as u32;
                        *ids.get_mut(cursor[pd]) = if pd != last { id | FIRST } else { id };
                        if let (Some(weights), Some(labels)) = (&mut weights, &labels) {
                            *weights.get_mut(cursor[pd]) = labels.get(e).weight::<W>();
                        }
                        cursor[pd] += 1;
                        last = pd;
                    }
                }
            }
        })?;
        drop(arc_cursors);

        Ok(Layout {
            ids,
            values,
            id_starts,
            value_starts,
            value_cursors,
            weights,
        })
    }

    /// Iterates until convergence, pulling contributions or partition-centric, returning the
    /// number of iterations.
    fn compute(
        &mut self,
        alpha: f64,
        tolerance: f64,
        max_iterations: usize,
        pull: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let n = self.g.size();
        if n == 0 {
            return Ok(0);
        }
        // weights checked before anything else is built
        let out_weights_mem = if E::is_weighted() {
            Some(self.out_weights(6)?)
        } else {
            None
        };
        let layout = if pull { None } else { Some(self.layout()?) };
        // the same partitions either way: their sums are added up in the same order
        let shift = partition_shift(n);
        let k = n.div_ceil(1 << shift);
        let threads = self.g.thread_num().min(k);
        let offsets = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width());
        // pulled contributions are read while the next are written: those go to another array
        let contributions_mem = self.scratch::<W>(4, n)?;
        let next_mem = if pull {
            Some(self.scratch::<W>(5, n)?)
        } else {
            None
        };
        let ranks = self.ranks.shared_slice();
        let mut contributions = contributions_mem.shared_slice();
        let mut next_contributions = next_mem
            .as_ref()
            .map_or(contributions, |m| m.shared_slice());
        let partition = |p: usize| (p << shift)..std::cmp::min((p + 1) << shift, n);
        // what a node's rank is divided by among its arcs: their number, or for weighted arcs
        // their total weight (checked finite and non-negative, summed in `f64`); `0` for dangling
        // nodes
        let out_weights = out_weights_mem.as_ref().map(|w| w.shared_slice());
        let share = |u: usize| match &out_weights {
            Some(w) => *w.get(u),
            None => W::from_usize(offsets.get(u + 1).index() - offsets.get(u).index()),
        };

        // ranks start even; each partition's sum of its dangling nodes' ranks, and of its
        // ranks' change, are added up in partition order, whatever the threads
        let mut partials = vec![(0f64, 0f64); k];
        let initial = W::one() / W::from_usize(n);
        {
            let (mut ranks, mut contributions) = (ranks, contributions);
            for (p, partial) in partials.iter_mut().enumerate() {
                for u in partition(p) {
                    *ranks.get_mut(u) = initial;
                    let out = share(u);
                    *contributions.get_mut(u) = if out == W::zero() {
                        partial.0 += initial.to_f64();
                        W::zero()
                    } else {
                        initial / out
                    };
                }
            }
        }
        let mut dangling: f64 = partials.iter().map(|p| p.0).sum();
        let alpha_w = W::from_f64(alpha);
        let teleport = (1. - alpha) / n as f64;

        // a partition's nodes' new ranks and contributions, from the sums of the contributions
        // they get: the sums of their dangling nodes' ranks and of their ranks' change
        let update = |nodes: std::ops::Range<usize>,
                      sums: &[W],
                      even_share: W,
                      mut contributions: SharedSliceMut<W>|
         -> (f64, f64) {
            let mut ranks = ranks;
            let (mut dangling, mut change) = (0f64, 0f64);
            for (u, &sum) in nodes.zip(sums.iter()) {
                let rank = alpha_w * sum + even_share;
                change += (rank.to_f64() - ranks.get(u).to_f64()).abs();
                *ranks.get_mut(u) = rank;
                let out = share(u);
                *contributions.get_mut(u) = if out == W::zero() {
                    dangling += rank.to_f64();
                    W::zero()
                } else {
                    rank / out
                };
            }
            (dangling, change)
        };

        for iteration in 1..=max_iterations {
            let even_share = W::from_f64(alpha * dangling / n as f64 + teleport);
            let partials_slice = SharedSliceMut::<(f64, f64)>::new(partials.as_mut_ptr(), k);
            match &layout {
                None => {
                    // pull: each node sums its neighbours' contributions (in order, as the
                    // partition-centric engine does)
                    let next = AtomicUsize::new(0);
                    parallel(threads, |_| {
                        let mut partials = partials_slice;
                        let mut sums = vec![W::zero(); 1 << shift];
                        loop {
                            let p = next.fetch_add(1, Ordering::Relaxed);
                            if p >= k {
                                return Ok(());
                            }
                            let nodes = partition(p);
                            for (v, sum) in nodes.clone().zip(sums.iter_mut()) {
                                *sum = W::zero();
                                for e in offsets.get(v).index()..offsets.get(v + 1).index() {
                                    *sum += *contributions.get(neighbours.get(e).index());
                                }
                            }
                            let len = nodes.len();
                            *partials.get_mut(p) =
                                update(nodes, &sums[..len], even_share, next_contributions);
                        }
                    })?;
                    std::mem::swap(&mut contributions, &mut next_contributions);
                }
                Some(layout) => {
                    let (ids, values) = (layout.ids.shared_slice(), layout.values.shared_slice());
                    let weights = layout.weights.as_ref().map(|w| w.shared_slice());
                    let value_cursors = layout.value_cursors.shared_slice();
                    let (id_starts, value_starts) = (&layout.id_starts, &layout.value_starts);

                    // scatter: each node's contribution, once per partition it has arcs into,
                    // in that partition's bin (after earlier source partitions' nodes')
                    let next = AtomicUsize::new(0);
                    parallel(threads, |_| {
                        let mut values = values;
                        let mut cursor = vec![0usize; k];
                        loop {
                            let ps = next.fetch_add(1, Ordering::Relaxed);
                            if ps >= k {
                                return Ok(());
                            }
                            for (pd, c) in cursor.iter_mut().enumerate() {
                                *c = *value_cursors.get(ps * k + pd);
                            }
                            for u in partition(ps) {
                                let contribution = *contributions.get(u);
                                let mut last = usize::MAX;
                                for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                                    let pd = neighbours.get(e).index() >> shift;
                                    if pd != last {
                                        *values.get_mut(cursor[pd]) = contribution;
                                        cursor[pd] += 1;
                                        last = pd;
                                    }
                                }
                            }
                        }
                    })?;

                    // gather: each partition's bin summed into its nodes', in cache (once read,
                    // contributions are rewritten in place)
                    let next = AtomicUsize::new(0);
                    parallel(threads, |_| {
                        let mut partials = partials_slice;
                        let mut sums = vec![W::zero(); 1 << shift];
                        loop {
                            let pd = next.fetch_add(1, Ordering::Relaxed);
                            if pd >= k {
                                return Ok(());
                            }
                            let nodes = partition(pd);
                            let sums = &mut sums[..nodes.len()];
                            sums.fill(W::zero());
                            let (mut v, mut value) = (value_starts[pd], W::zero());
                            let bin = id_starts[pd]..id_starts[pd + 1];
                            // (two loops: unweighted arcs pay no multiplication)
                            match &weights {
                                None => {
                                    for i in bin {
                                        let id = *ids.get(i);
                                        if id & FIRST != 0 {
                                            value = *values.get(v);
                                            v += 1;
                                        }
                                        sums[(id & !FIRST) as usize] += value;
                                    }
                                }
                                Some(weights) => {
                                    for i in bin {
                                        let id = *ids.get(i);
                                        if id & FIRST != 0 {
                                            value = *values.get(v);
                                            v += 1;
                                        }
                                        sums[(id & !FIRST) as usize] += value * *weights.get(i);
                                    }
                                }
                            }
                            *partials.get_mut(pd) = update(nodes, sums, even_share, contributions);
                        }
                    })?;
                }
            }

            let change: f64 = partials.iter().map(|p| p.1).sum();
            dangling = partials.iter().map(|p| p.0).sum();
            if change < tolerance {
                return Ok(iteration);
            }
            if iteration == max_iterations {
                return Err(format!(
                    "error PageRank didn't converge in {max_iterations} iterations (the ranks \
                     changed by {change:e} in the last, more than {tolerance:e})"
                )
                .into());
            }
        }
        unreachable!("the last iteration returns")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, graph_with_threads, random_arcs};

    /// PageRank by plain power iterations, and the number of iterations.
    fn reference<Dir: Directedness>(
        g: &GraphMemoryMap<(), (), usize, Dir>,
        alpha: f64,
        tolerance: f64,
    ) -> (Vec<f64>, usize) {
        let n = g.size();
        let out: Vec<Vec<usize>> = (0..n).map(|u| g.neighbours(u).unwrap().collect()).collect();
        let mut x = vec![1. / n as f64; n];
        for iteration in 1.. {
            let dangling: f64 = (0..n).filter(|&u| out[u].is_empty()).map(|u| x[u]).sum();
            let mut next = vec![alpha * dangling / n as f64 + (1. - alpha) / n as f64; n];
            for u in 0..n {
                for &v in &out[u] {
                    next[v] += alpha * x[u] / out[u].len() as f64;
                }
            }
            let change: f64 = next.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
            x = next;
            if change < tolerance {
                return (x, iteration);
            }
        }
        unreachable!()
    }

    fn assert_close(got: &[f64], want: &[f64], relative: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (u, (a, b)) in got.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= relative * b.abs(),
                "{what}: node {u}: {a} vs {b}"
            );
        }
    }

    /// The ranks are those of plain power iterations, on graphs of one partition or many,
    /// directed or not, with dangling nodes, self-loops and parallel arcs.
    #[test]
    fn pagerank_matches_power_iterations() {
        let star_in: Vec<_> = (1..50).map(|u| (u, 0)).collect();
        for (name, header, n, arcs) in [
            (
                "cycle",
                "general",
                5,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
            ),
            ("star_in", "general", 50, star_in),
            (
                "loops_and_parallel",
                "general",
                6,
                vec![(0, 0), (0, 1), (0, 1), (1, 2), (2, 0), (3, 3), (4, 5)],
            ),
            ("one_partition", "general", 900, random_arcs(900, 4000, 1)),
            ("partitions", "general", 3000, random_arcs(3000, 9000, 2)),
            (
                "partitions_symmetric",
                "symmetric",
                3000,
                random_arcs(3000, 6000, 3),
            ),
            (
                "many_partitions",
                "general",
                70_000,
                random_arcs(70_000, 350_000, 4),
            ),
            (
                "sparse_dangling",
                "general",
                5000,
                random_arcs(5000, 2500, 5),
            ),
        ] {
            let g = graph(name, header, n, &arcs);
            for (alpha, tolerance) in [(0.85, 1e-6), (0.5, 1e-10), (0.99, 1e-8)] {
                let pr = PageRank::<_, _, _, f64, _>::new_with_conf(&g, alpha, tolerance, 10_000)
                    .unwrap();
                let (want, iterations) = reference(&g, alpha, tolerance);
                assert_eq!(pr.iterations(), iterations, "{name}, alpha {alpha}");
                assert_close(pr.ranks(), &want, 1e-12, &format!("{name}, alpha {alpha}"));
                let total: f64 = pr.ranks().iter().sum();
                assert!((total - 1.).abs() <= 1e-9, "{name}: ranks sum to {total}");
            }
        }
    }

    /// On undirected graphs, both engines give the same ranks, bit for bit, in as many
    /// iterations: of one partition or many, with or without parallel edges and self-loops, with
    /// isolated nodes. Directed graphs can't be pulled.
    #[test]
    fn pulling_gives_the_partition_centric_ranks() {
        for (name, n, arcs) in [
            ("tiny", 5, vec![(0, 1), (1, 2), (2, 0), (3, 3)]),
            ("one_partition", 900, random_arcs(900, 3000, 11)),
            ("partitions", 3000, random_arcs(3000, 9000, 12)),
            ("many_partitions", 70_000, random_arcs(70_000, 300_000, 13)),
            ("isolated", 5000, random_arcs(5000, 1500, 14)),
        ] {
            let d = graph(name, "symmetric", n, &arcs);
            for mode in [UndirectedMode::Collapse, UndirectedMode::Each] {
                let g = d.to_undirected_with(mode).unwrap();
                let run = |engine| {
                    let mut pr =
                        PageRank::<_, _, _, f64, _>::new_with_engine(g, 0.85, 1e-10, 1000, engine)
                            .unwrap();
                    let ranks: Vec<u64> = pr.ranks().iter().map(|r| r.to_bits()).collect();
                    let iterations = pr.iterations();
                    pr.drop_cache().unwrap();
                    (ranks, iterations)
                };
                let (pulled, pcpm) = (
                    run(PageRankEngine::Pull),
                    run(PageRankEngine::PartitionCentric),
                );
                assert_eq!(pulled, pcpm, "{name}, {mode:?}");
                assert_eq!(run(PageRankEngine::Auto), pulled, "{name}, {mode:?}");
                let (want, _) = reference(g, 0.85, 1e-10);
                let got: Vec<f64> = pulled.0.iter().map(|&b| f64::from_bits(b)).collect();
                assert_close(&got, &want, 1e-12, name);
            }
            assert!(
                PageRank::<_, _, _, f64, _>::new_with_engine(
                    &d,
                    0.85,
                    1e-6,
                    100,
                    PageRankEngine::Pull
                )
                .is_err(),
                "{name}: pulling a directed graph"
            );
        }
    }

    /// The ranks don't depend on the number of threads, bit for bit.
    #[test]
    fn pagerank_is_the_same_whatever_the_threads() {
        let arcs = random_arcs(40_000, 200_000, 6);
        let ranks: Vec<Vec<u64>> = [1, 3, 8]
            .into_iter()
            .map(|threads| {
                let g = graph_with_threads("threads", "general", 40_000, &arcs, threads);
                let pr = PageRank::<_, _, _, f64, _>::new(&g).unwrap();
                pr.ranks().iter().map(|r| r.to_bits()).collect()
            })
            .collect();
        assert_eq!(ranks[0], ranks[1]);
        assert_eq!(ranks[0], ranks[2]);
    }

    #[test]
    fn pagerank_of_graphs_without_arcs_and_of_self_loops() {
        // every node dangling: even ranks, at once
        let g = graph("no_arcs", "general", 7, &[]);
        let pr = PageRank::<_, _, _, f64, _>::new(&g).unwrap();
        assert!(pr.ranks().iter().all(|&r| (r - 1. / 7.).abs() < 1e-15));
        assert_eq!(pr.iterations(), 1);
        // a self-loop keeps its node's rank
        let g = graph("loop", "general", 2, &[(0, 0)]);
        let pr = PageRank::<_, _, _, f64, _>::new_with_conf(&g, 0.85, 1e-12, 1000).unwrap();
        let (want, _) = reference(&g, 0.85, 1e-12);
        assert_close(pr.ranks(), &want, 1e-12, "loop");
        assert!(pr.rank(0) > pr.rank(1));
        // no damping: even ranks
        let g = graph("undamped", "general", 4, &[(0, 1), (1, 2)]);
        let pr = PageRank::<_, _, _, f64, _>::new_with_conf(&g, 0., 1e-6, 10).unwrap();
        assert!(pr.ranks().iter().all(|&r| (r - 0.25).abs() < 1e-15));
    }

    #[test]
    fn pagerank_in_f32_is_close() {
        let g = graph("f32", "general", 3000, &random_arcs(3000, 12000, 7));
        let r64 = PageRank::<_, _, _, f64, _>::new(&g)
            .unwrap()
            .ranks()
            .to_vec();
        let pr = PageRank::<_, _, _, f32, _>::new(&g).unwrap();
        let r32: Vec<f64> = pr.ranks().iter().map(|&r| r as f64).collect();
        assert_close(&r32, &r64, 1e-4, "f32");
    }

    #[test]
    fn pagerank_rejects_bad_configurations_and_non_convergence() {
        let g = graph("conf", "general", 100, &random_arcs(100, 400, 8));
        for (alpha, tolerance, iterations) in [
            (1.5, 1e-6, 10),
            (-0.1, 1e-6, 10),
            (0.85, 0., 10),
            (0.85, f64::NAN, 10),
            (0.85, 1e-6, 0),
        ] {
            assert!(
                PageRank::<_, _, _, f64, _>::new_with_conf(&g, alpha, tolerance, iterations)
                    .is_err(),
                "{alpha} {tolerance} {iterations}"
            );
        }
        let err = PageRank::<_, _, _, f64, _>::new_with_conf(&g, 0.85, 1e-15, 2).unwrap_err();
        assert!(err.to_string().contains("didn't converge"), "{err}");
        // no result left behind
        let out = g.build_cache_filename(CacheFile::PageRank, None).unwrap();
        assert!(!std::path::Path::new(&out).exists(), "{out}");
    }

    /// A graph whose arcs `(u, v, value)` carry `value`s, as written in a `real` MatrixMarket
    /// file of the given symmetry, loaded with `f64` weights on `threads` threads.
    fn weighted<Dir: Directedness>(
        name: &str,
        symmetry: &str,
        n: usize,
        arcs: &[(usize, usize, String)],
        threads: u8,
    ) -> crate::graph::test_utils::TestGraph<(), Weighted<f64>, usize, Dir> {
        use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
        let dir = unique_test_dir("pagerank_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut text = format!(
            "%%MatrixMarket matrix coordinate real {symmetry}\n{n} {n} {}\n",
            arcs.len()
        );
        for (u, v, w) in arcs {
            text += &format!("{} {} {w}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let g = GraphMemoryMap::<(), Weighted<f64>, usize, Dir>::from_file(
            &path,
            Some(format!("pr_weights_{name}_{}", unique_run_tag())),
            Some(threads),
        )
        .unwrap();
        crate::graph::test_utils::TestGraph::new(g, Some(dir))
    }

    /// Weighted PageRank by plain power iterations, and the number of iterations.
    fn weighted_reference<Dir: Directedness>(
        g: &GraphMemoryMap<(), Weighted<f64>, usize, Dir>,
        alpha: f64,
        tolerance: f64,
    ) -> (Vec<f64>, usize) {
        let n = g.size();
        let out: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|u| {
                g.weighted_neighbours(u)
                    .unwrap()
                    .map(|(v, w)| (v, w.0))
                    .collect()
            })
            .collect();
        let total: Vec<f64> = out.iter().map(|o| o.iter().map(|a| a.1).sum()).collect();
        let mut x = vec![1. / n as f64; n];
        for iteration in 1.. {
            let dangling: f64 = (0..n).filter(|&u| total[u] == 0.).map(|u| x[u]).sum();
            let mut next = vec![alpha * dangling / n as f64 + (1. - alpha) / n as f64; n];
            for u in 0..n {
                for &(v, w) in &out[u] {
                    if total[u] > 0. {
                        next[v] += alpha * x[u] * w / total[u];
                    }
                }
            }
            let change: f64 = next.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
            x = next;
            if change < tolerance {
                return (x, iteration);
            }
        }
        unreachable!()
    }

    fn random_weights(arcs: &[(usize, usize)], seed: u64) -> Vec<(usize, usize, String)> {
        let mut state = seed;
        arcs.iter()
            .map(|&(u, v)| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // a few zeros, the rest spread over three orders of magnitude
                let r = (state >> 33) % 1000;
                let w = if r < 50 { 0.0 } else { r as f64 / 7.0 };
                (u, v, format!("{w}"))
            })
            .collect()
    }

    /// Weighted ranks are those of plain weighted power iterations, on one partition or many,
    /// directed or not, with zero weights, nodes whose arcs out all weigh nothing (dangling),
    /// self-loops and parallel arcs.
    #[test]
    fn weighted_pagerank_matches_power_iterations() {
        let mut loops =
            random_weights(&[(0, 0), (0, 1), (0, 1), (1, 2), (2, 0), (3, 3), (4, 5)], 1);
        // node 4's only arc weighs nothing: it's dangling
        loops[6].2 = "0".to_string();
        let cases = [
            ("w_loops", 6, loops),
            (
                "w_one_partition",
                900,
                random_weights(&random_arcs(900, 4000, 11), 2),
            ),
            (
                "w_partitions",
                3000,
                random_weights(&random_arcs(3000, 9000, 12), 3),
            ),
            (
                "w_many",
                70_000,
                random_weights(&random_arcs(70_000, 350_000, 13), 4),
            ),
        ];
        for (name, n, arcs) in &cases {
            let g = weighted::<Directed>(name, "general", *n, arcs, 4);
            for (alpha, tolerance) in [(0.85, 1e-6), (0.5, 1e-10)] {
                let pr = PageRank::<_, _, _, f64, _>::new_with_conf(&g, alpha, tolerance, 10_000)
                    .unwrap();
                let (want, iterations) = weighted_reference(&g, alpha, tolerance);
                assert_eq!(pr.iterations(), iterations, "{name}, alpha {alpha}");
                assert_close(pr.ranks(), &want, 1e-10, &format!("{name}, alpha {alpha}"));
                let total: f64 = pr.ranks().iter().sum();
                assert!((total - 1.).abs() <= 1e-9, "{name}: ranks sum to {total}");
            }
        }
        // undirected, symmetric weights: ranked partition-centric (pulling is refused)
        let arcs = random_weights(&random_arcs(2000, 6000, 14), 5);
        let g = weighted::<Undirected>("w_symmetric", "symmetric", 2000, &arcs, 4);
        let pr = PageRank::<_, _, _, f64, _>::new(&g).unwrap();
        let (want, _) = weighted_reference(&g, 0.85, 1e-6);
        assert_close(pr.ranks(), &want, 1e-10, "symmetric");
        assert!(
            PageRank::<_, _, _, f64, _>::new_with_engine(&g, 0.85, 1e-6, 100, PageRankEngine::Pull)
                .is_err()
        );
    }

    /// Uniform weights rank as unit ones do, and ranks don't depend on the threads, bit for bit.
    #[test]
    fn weighted_pagerank_uniform_and_thread_independent() {
        let arcs = random_arcs(3000, 12_000, 21);
        let plain = graph("w_plain", "general", 3000, &arcs);
        let uniform: Vec<_> = arcs
            .iter()
            .map(|&(u, v)| (u, v, "2.5".to_string()))
            .collect();
        let g = weighted::<Directed>("w_uniform", "general", 3000, &uniform, 4);
        let want = PageRank::<_, _, _, f64, _>::new(&plain).unwrap();
        let got = PageRank::<_, _, _, f64, _>::new(&g).unwrap();
        assert_eq!(got.iterations(), want.iterations());
        assert_close(got.ranks(), want.ranks(), 1e-12, "uniform");

        let arcs = random_weights(&arcs, 22);
        let ranks: Vec<Vec<f64>> = [1u8, 3, 8]
            .iter()
            .map(|&t| {
                let g = weighted::<Directed>(&format!("w_threads_{t}"), "general", 3000, &arcs, t);
                PageRank::<_, _, _, f64, _>::new(&g)
                    .unwrap()
                    .ranks()
                    .to_vec()
            })
            .collect();
        assert_eq!(ranks[0], ranks[1]);
        assert_eq!(ranks[0], ranks[2]);
    }

    /// Parallel arcs come out of parsing ordered by their labels, not in the order the parsing
    /// threads happened to reach them: weighted graphs, and what's summed over their arcs (the
    /// ranks above), are the same whatever the threads' number and timing.
    #[test]
    fn parallel_arcs_are_ordered_by_their_labels() {
        let arcs = random_weights(&random_arcs(3000, 12_000, 21), 22);
        let rows = |t: u8| {
            let g = weighted::<Directed>(&format!("parallel_{t}"), "general", 3000, &arcs, t);
            (0..3000)
                .map(|u| {
                    g.weighted_neighbours(u)
                        .unwrap()
                        .map(|(v, l)| (v, l.weight::<f64>().to_bits()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let one = rows(1);
        let parallel = one
            .iter()
            .flat_map(|row| row.windows(2).filter(|w| w[0].0 == w[1].0))
            .count();
        assert!(parallel > 0, "the test graph has no parallel arcs");
        for row in &one {
            assert!(
                row.windows(2)
                    .all(|w| w[0].0 < w[1].0 || f64::from_bits(w[0].1) <= f64::from_bits(w[1].1))
            );
        }
        for t in [3, 8, 3] {
            assert_eq!(rows(t), one, "{t} threads");
        }
    }

    /// Weights that aren't finite and non-negative are refused, saying so.
    #[test]
    fn weighted_pagerank_rejects_bad_weights() {
        for (name, w) in [("w_negative", "-1"), ("w_nan", "nan"), ("w_inf", "inf")] {
            let arcs = vec![(0, 1, "1".to_string()), (1, 0, w.to_string())];
            let g = weighted::<Directed>(name, "general", 2, &arcs, 2);
            let err = PageRank::<_, _, _, f64, _>::new(&g)
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains("finite and non-negative"), "{name}: {err}");
        }
    }
}
