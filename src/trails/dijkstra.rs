//! Weighted shortest paths: [`DijkstraDists`], every node's distance from a source, by Dijkstra's
//! algorithm or, on many threads, Δ-stepping (see [`ShortestPathEngine`]); and the sequential
//! kernel the exact centralities run from every node of weighted graphs (each on a thread of
//! its own).
//!
//! An arc's length is its weight ([`E::weight`], `1` on unweighted graphs): weights must be finite
//! and non-negative (checked once, by [`GraphMemoryMap::total_weight`]). Parallel arcs count as
//! their lightest, and self-loops never shorten a path.
//!
//! [`E::weight`]: crate::graph::E::weight
//! [`GraphMemoryMap::total_weight`]: crate::graph::GraphMemoryMap::total_weight

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use super::delta_stepping::{default_delta, delta_stepping};

use super::shortest_path_visit::{NoOpShortestPathVisitor, ShortestPathVisitor};

use std::mem::ManuallyDrop;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// How [`DijkstraDists`] computes distances. Both engines find the same distances, bit for bit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ShortestPathEngine {
    /// [`Sequential`](Self::Sequential) on one thread or graphs of under
    /// [`PARALLEL_FROM_NODES`] nodes, [`DeltaStepping`](Self::DeltaStepping) otherwise.
    #[default]
    Auto,
    /// Dijkstra's algorithm, on one thread: `O((|V| + |E|) log |V|)` work. Its scratch memory
    /// is `16 + 2 * size_of::<Ix>()` bytes per node.
    Sequential,
    /// Δ-stepping, on the graph's threads, in buckets of distances `delta` wide (by default, the
    /// weights' mean; raised, if need be, to the heaviest weight over 1022). Smaller buckets do
    /// less redundant work, larger ones have more nodes for threads to share: on graphs of
    /// long paths (road networks), a larger `delta` usually pays.
    DeltaStepping { delta: Option<f64> },
}

/// The node count from which [`ShortestPathEngine::Auto`] picks Δ-stepping.
pub const PARALLEL_FROM_NODES: usize = 1 << 16;
type Error = Box<dyn std::error::Error + Send + Sync>;

/// What Dijkstra's algorithm from a node finds: the nodes it reaches (itself, and nodes at
/// distance zero, included), the sum of their distances, and the sum of the inverses of the
/// positive ones (nodes at distance zero being skipped, as networkx's `harmonic_centrality`
/// does).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct WeightedReach {
    pub(crate) reached: usize,
    pub(crate) distances: f64,
    pub(crate) inverse_distances: f64,
}

/// A thread's memory for [`dijkstra`], reused across sources: every node's [`NodeState`], and
/// the heap of found, unsettled nodes --- an indexed 4-ary heap, holding each node at most once
/// (so at most `|V|` entries), with each node's distance in its entry, so that sifting compares
/// entries side by side rather than looking their nodes' distances up.
pub(crate) struct DijkstraMemory<Ix: graph::IndexType> {
    states: AbstractedProceduralMemoryMut<NodeState<Ix>>,
    heap: AbstractedProceduralMemoryMut<HeapEntry<Ix>>,
    round: u32,
}

/// What [`dijkstra`] knows of a node, together, so that relaxing an arc reads one place: its
/// tentative distance, its position in the heap, and its `stamp` --- `2r` once the `r`-th run
/// found it, `2r + 1` once it settled it --- so that nothing is cleared between runs.
#[derive(Clone, Copy)]
#[repr(C)]
struct NodeState<Ix> {
    dist: f64,
    stamp: u32,
    pos: Ix,
}

/// A node of [`DijkstraMemory`]'s heap, and its distance.
#[derive(Clone, Copy)]
#[repr(C)]
struct HeapEntry<Ix> {
    key: f64,
    node: Ix,
}

/// The children of a node of the heap.
const ARITY: usize = 4;

impl<Ix: graph::IndexType> DijkstraMemory<Ix> {
    /// Memory for a graph of `n` nodes, mapped in the 2 `files`: `(16 + 2 * size_of::<Ix>())`
    /// bytes per node, give or take alignment.
    pub(crate) fn new(files: &[String], n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            states: SharedSliceMut::<NodeState<Ix>>::abst_mem_mut(&files[0], n, true)?,
            heap: SharedSliceMut::<HeapEntry<Ix>>::abst_mem_mut(&files[1], n, true)?,
            round: 0,
        })
    }

    /// `v`'s distance in the last run, if it found `v`.
    #[inline(always)]
    pub(crate) fn distance(&self, v: usize) -> Option<f64> {
        let state = self.states.get(v);
        (state.stamp >> 1 == self.round).then_some(state.dist)
    }
}

/// [`DijkstraMemory`]'s slices, for a run.
struct Work<Ix: graph::IndexType> {
    states: SharedSliceMut<NodeState<Ix>>,
    heap: SharedSliceMut<HeapEntry<Ix>>,
}

impl<Ix: graph::IndexType> Work<Ix> {
    #[inline(always)]
    fn place(&mut self, at: usize, e: HeapEntry<Ix>) {
        *self.heap.get_mut(at) = e;
        self.states.get_mut(e.node.index()).pos = Ix::new(at);
    }

    /// Moves `e` up the heap from `at`, to where its key belongs.
    #[inline(always)]
    fn sift_up(&mut self, mut at: usize, e: HeapEntry<Ix>) {
        while at > 0 {
            let parent = (at - 1) / ARITY;
            let p = *self.heap.get(parent);
            if e.key >= p.key {
                break;
            }
            self.place(at, p);
            at = parent;
        }
        self.place(at, e);
    }

    /// Moves `e` down a heap of `len` entries from `at`, to where its key belongs.
    #[inline(always)]
    fn sift_down(&mut self, mut at: usize, len: usize, e: HeapEntry<Ix>) {
        loop {
            let first = ARITY * at + 1;
            if first >= len {
                break;
            }
            let mut least = first;
            let mut key = self.heap.get(first).key;
            for c in first + 1..(first + ARITY).min(len) {
                let k = self.heap.get(c).key;
                if k < key {
                    (least, key) = (c, k);
                }
            }
            if key >= e.key {
                break;
            }
            let c = *self.heap.get(least);
            self.place(at, c);
            at = least;
        }
        self.place(at, e);
    }
}

/// Dijkstra's algorithm from `source` on `g` (following its arcs), with `m`'s memory: leaves the
/// distances of the nodes it reaches in `m` (see [`DijkstraMemory::distance`]) and returns what
/// it found. Weights must have been checked finite and non-negative.
#[inline]
pub(crate) fn dijkstra<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    m: &mut DijkstraMemory<Ix>,
) -> Result<WeightedReach, Error> {
    Ok(dijkstra_visit(g, source, m, &mut NoOpShortestPathVisitor)?.0)
}

/// [`dijkstra`], calling `visitor`'s hooks: also whether it stopped the search (what was found
/// is then what was examined before it did).
pub(crate) fn dijkstra_visit<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
    V: ShortestPathVisitor + ?Sized,
>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    m: &mut DijkstraMemory<Ix>,
    visitor: &mut V,
) -> Result<(WeightedReach, bool), Error> {
    m.round = m
        .round
        .checked_add(1)
        .filter(|&r| r < 1 << 31)
        .ok_or("error too many Dijkstra runs on one scratch memory")?;
    let (found, settled) = (2 * m.round, 2 * m.round + 1);
    let mut s = Work {
        states: m.states.shared_slice(),
        heap: m.heap.shared_slice(),
    };
    let mut reach = WeightedReach::default();
    macro_rules! hook {
        ($call:expr) => {
            if let ControlFlow::Break(()) = $call {
                return Ok((reach, true));
            }
        };
    }

    let state = s.states.get_mut(source);
    (state.dist, state.stamp) = (0., found);
    s.place(
        0,
        HeapEntry {
            key: 0.,
            node: Ix::new(source),
        },
    );
    let mut len = 1;
    hook!(visitor.discover_vertex(source, 0., source));

    while len > 0 {
        // the closest unsettled node
        let top = *s.heap.get(0);
        let (u, du) = (top.node.index(), top.key);
        len -= 1;
        if len > 0 {
            let last = *s.heap.get(len);
            s.sift_down(0, len, last);
        }
        s.states.get_mut(u).stamp = settled;
        reach.reached += 1;
        if du > 0. {
            reach.distances += du;
            reach.inverse_distances += 1. / du;
        }
        hook!(visitor.examine_vertex(u, du));

        let mut relax = |v: usize, w: f64| -> ControlFlow<()> {
            visitor.examine_edge(u, v, w)?;
            let state = *s.states.get(v);
            if state.stamp == settled {
                return visitor.edge_not_relaxed(u, v, w);
            }
            let dv = du + w;
            let e = HeapEntry {
                key: dv,
                node: Ix::new(v),
            };
            if state.stamp != found {
                let state = s.states.get_mut(v);
                (state.dist, state.stamp) = (dv, found);
                len += 1;
                s.sift_up(len - 1, e);
                visitor.edge_relaxed(u, v, w, dv)?;
                visitor.discover_vertex(v, dv, u)
            } else if dv < state.dist {
                s.states.get_mut(v).dist = dv;
                s.sift_up(state.pos.index(), e);
                visitor.edge_relaxed(u, v, w, dv)
            } else {
                visitor.edge_not_relaxed(u, v, w)
            }
        };
        if E::is_weighted() {
            for (v, label) in g.weighted_neighbours(u).map_err(|e| e.to_string())? {
                hook!(relax(v, label.weight::<f64>()));
            }
        } else {
            for v in g.neighbours(u).map_err(|e| e.to_string())? {
                hook!(relax(v, 1.));
            }
        }
        hook!(visitor.finish_vertex(u, du));
    }
    Ok((reach, false))
}

/// Every node's distance from a `source` in a [`GraphMemoryMap`] instance, arcs being as long as
/// their weights (see the [module docs](self)), by Dijkstra's algorithm: the weighted
/// counterpart of [`BFSDists`], which counts arcs. Unreachable nodes are at
/// [`f64::INFINITY`].
///
/// Works on any graph, following arcs out of nodes: distances are from `source`. On a
/// [`DiGraph`], [`new_directed`](Self::new_directed) follows them in another [`Direction`]:
/// [`Direction::In`] gives distances to `source`.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
/// [`BFSDists`]: ../bfs/struct.BFSDists.html
/// [`DiGraph`]: ../../graph/type.DiGraph.html
/// [`Direction`]: ../../graph/direction/enum.Direction.html
/// [`Direction::In`]: ../../graph/direction/enum.Direction.html#variant.In
pub struct DijkstraDists<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness = Undirected,
> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    /// Memmapped slice of each node's distance from the source.
    distances: AbstractedProceduralMemoryMut<f64>,
    /// Number of nodes reachable from the source, the source itself included.
    reachable: usize,
    /// Sum of the reachable nodes' distances from the source.
    total_distances: f64,
    /// Each node's parent in a shortest path tree, once asked for (see
    /// [`predecessors`](Self::predecessors)).
    predecessors: Option<AbstractedProceduralMemoryMut<Ix>>,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DijkstraDists<'a, N, E, Ix, Directed> {
    /// Computes every node's distance from `source`, following arcs in a given [`Direction`]:
    /// the same as [`new`](Self::new) on [`GraphMemoryMap::view`]`(direction)`.
    ///
    /// [`Direction`]: ../../graph/direction/enum.Direction.html
    /// [`GraphMemoryMap::view`]: ../../graph/struct.GraphMemoryMap.html#method.view
    pub fn new_directed(
        g: &'a DiGraph<N, E, Ix>,
        source: usize,
        direction: Direction,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(g.view(direction)?, source)
    }

    /// [`new_directed`](Self::new_directed), with a given [`ShortestPathEngine`].
    pub fn new_directed_with_engine(
        g: &'a DiGraph<N, E, Ix>,
        source: usize,
        direction: Direction,
        engine: ShortestPathEngine,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_engine(g.view(direction)?, source, engine)
    }
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>
    DijkstraDists<'a, N, E, Ix, Dir>
{
    /// Computes every node's distance from `source` ([`ShortestPathEngine::Auto`]).
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_engine(g, source, ShortestPathEngine::Auto)
    }

    /// Computes every node's distance from `source` with a given [`ShortestPathEngine`].
    pub fn with_engine(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
        engine: ShortestPathEngine,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = g.size();
        if source >= n {
            return Err(
                format!("error node with id {source} doesn't exist: max id is |V| = {n}").into(),
            );
        }
        if let ShortestPathEngine::DeltaStepping { delta: Some(d) } = engine
            && !(d.is_finite() && d > 0.)
        {
            return Err(format!("error Δ must be positive and finite, not {d}").into());
        }
        // checks the weights finite and non-negative
        g.total_weight()?;
        let threads = g.thread_num().max(1);
        let delta = resolve_engine(g, engine, threads)?;
        let out_fn = g.build_cache_filename(CacheFile::Dijkstra, None)?;
        let scratch: Vec<String> = (1..3)
            .map(|k| g.build_cache_filename(CacheFile::Dijkstra, Some(k)))
            .collect::<Result<_, _>>()?;
        let mut distances = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, n, true)?;
        distances_into(
            g,
            source,
            distances.as_mut_slice(),
            delta,
            threads,
            &scratch,
        )?;
        // (written back in the background, as other results are: the map is the result)
        distances.flush_async()?;
        let (reachable, total_distances) = totals(distances.as_slice(), threads);
        Ok(Self {
            g,
            source,
            distances,
            reachable,
            total_distances,
            predecessors: None,
        })
    }

    /// The node distances are from.
    pub fn source(&self) -> usize {
        self.source
    }

    /// Every node's distance from the source ([`f64::INFINITY`] for unreachable nodes).
    pub fn get_distances(&self) -> &[f64] {
        self.distances.as_slice()
    }

    /// The number of nodes reachable from the source, the source itself included.
    pub fn reachable(&self) -> usize {
        self.reachable
    }

    /// The sum of the reachable nodes' distances from the source.
    pub fn total_distances(&self) -> f64 {
        self.total_distances
    }

    /// Each node's parent in a shortest path tree (itself, for the source and unreachable
    /// nodes), found the first time it's asked for --- only distances are kept otherwise.
    ///
    /// The tree is the same whatever the engine and the threads: a node's parent is the least of
    /// its in-neighbours on shortest paths to it with the fewest arcs from the source (arcs
    /// `u -> v` with `d(u) + w == d(v)`, exactly), found by a BFS over those arcs, in parallel.
    /// Following parents from a node, the weights' sum is its distance, exactly. It takes one
    /// more pass over the arcs, `size_of::<Ix>()` bytes per node, and 16 bytes per node of
    /// scratch memory while it's found.
    pub fn predecessors(&mut self) -> Result<&[Ix], Box<dyn std::error::Error>> {
        if self.predecessors.is_none() {
            let g = self.g;
            let n = g.size();
            let out_fn = g.build_cache_filename(CacheFile::Dijkstra, Some(3))?;
            let scratch: Vec<String> = (4..6)
                .map(|k| g.build_cache_filename(CacheFile::Dijkstra, Some(k)))
                .collect::<Result<_, _>>()?;
            let result = (|| -> Result<_, Box<dyn std::error::Error>> {
                let parent = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&scratch[0], n, true)?;
                let hops = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&scratch[1], n, true)?;
                let threads = g.thread_num().max(1);
                shortest_path_tree(
                    g,
                    self.source,
                    self.distances.as_slice(),
                    parent.as_slice(),
                    hops.as_slice(),
                    threads,
                );
                let mut out = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, n, true)?;
                let (parent, out_slice) = (parent.as_slice(), out.as_mut_slice());
                let load = n.div_ceil(threads);
                std::thread::scope(|s| {
                    for (t, chunk) in out_slice.chunks_mut(load.max(1)).enumerate() {
                        s.spawn(move || {
                            for (i, p) in chunk.iter_mut().enumerate() {
                                let v = t * load + i;
                                let u = parent[v].load(Ordering::Relaxed);
                                *p = Ix::new(if u == usize::MAX { v } else { u });
                            }
                        });
                    }
                });
                out.flush()?;
                Ok(out)
            })();
            for f in &scratch {
                let _ = std::fs::remove_file(f);
            }
            self.predecessors = Some(result?);
        }
        Ok(self.predecessors.as_ref().unwrap().as_slice())
    }

    /// The nodes of a shortest path from the source to `target` (the one
    /// [`predecessors`](Self::predecessors) gives, found if it wasn't yet), or `None` if
    /// `target` isn't reachable.
    pub fn path_to(
        &mut self,
        target: usize,
    ) -> Result<Option<Vec<usize>>, Box<dyn std::error::Error>> {
        let n = self.g.size();
        if target >= n {
            return Err(
                format!("error node with id {target} doesn't exist: max id is |V| = {n}").into(),
            );
        }
        if self.distances.as_slice()[target].is_infinite() {
            return Ok(None);
        }
        let source = self.source;
        let parents = self.predecessors()?;
        let mut path = vec![target];
        let mut at = target;
        while at != source {
            at = parents[at].index();
            path.push(at);
            if path.len() > n {
                return Err("error the shortest path tree has a cycle".into());
            }
        }
        path.reverse();
        Ok(Some(path))
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        for seq in [None, Some(3)] {
            let file = this.g.build_cache_filename(CacheFile::Dijkstra, seq)?;
            crate::shared_slice::remove_if_exists(file)?;
        }
        Ok(())
    }
}

/// Each node's `parent` in the shortest path tree of `dist`ances from `source` (`usize::MAX` for
/// none) by a level-synchronous BFS over the tight arcs (`d(u) + w == d(v)`): a node's parent is
/// the least of its tight in-neighbours one level up, whatever thread finds it first --- every
/// one of them is at the same level, so each offers itself before the level ends. On
/// `threads` threads, waiting for each other at each level; `hops` is scratch memory.
fn shortest_path_tree<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    dist: &[f64],
    parent: &[AtomicUsize],
    hops: &[AtomicUsize],
    threads: usize,
) {
    use std::sync::RwLock;
    const CHUNK: usize = 64;
    let n = g.size();
    let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
    let targets = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());
    let labels = SharedSlice::<E>::new(g.edge_labels_ptr(), g.width());
    // each thread's share of the current level, read by all while it's expanded
    let handoff: Vec<RwLock<Vec<usize>>> = (0..threads).map(|_| RwLock::new(Vec::new())).collect();
    // the nodes found at each level, by its parity: a level's count is reset while no thread can
    // touch it (during the next level, once every thread has read it)
    let cursor = AtomicUsize::new(0);
    let found = [AtomicUsize::new(0), AtomicUsize::new(0)];
    let barrier = crate::utils::SpinBarrier::new(threads);
    std::thread::scope(|scope| {
        for tid in 0..threads {
            let (handoff, cursor, found, barrier) = (&handoff, &cursor, &found, &barrier);
            scope.spawn(move || {
                let load = n.div_ceil(threads);
                for v in (tid * load).min(n)..((tid + 1) * load).min(n) {
                    parent[v].store(usize::MAX, Ordering::Relaxed);
                    hops[v].store(usize::MAX, Ordering::Relaxed);
                }
                barrier.wait();
                if tid == 0 {
                    hops[source].store(0, Ordering::Relaxed);
                    handoff[0].write().unwrap().push(source);
                }
                barrier.wait();
                let mut level = 0;
                loop {
                    level += 1;
                    let mut next = Vec::new();
                    {
                        let shares: Vec<_> = handoff.iter().map(|h| h.read().unwrap()).collect();
                        let mut starts = Vec::with_capacity(threads + 1);
                        starts.push(0);
                        for share in &shares {
                            starts.push(starts.last().unwrap() + share.len());
                        }
                        let total = starts[threads];
                        loop {
                            let at = cursor.fetch_add(CHUNK, Ordering::Relaxed);
                            if at >= total {
                                break;
                            }
                            let mut owner = starts.partition_point(|&s| s <= at) - 1;
                            for i in at..(at + CHUNK).min(total) {
                                while i >= starts[owner + 1] {
                                    owner += 1;
                                }
                                let u = shares[owner][i - starts[owner]];
                                let du = dist[u];
                                for e in offsets.get(u).index()..offsets.get(u + 1).index() {
                                    let v = targets.get(e).index();
                                    let w = if E::is_weighted() {
                                        labels.get(e).weight::<f64>()
                                    } else {
                                        1.
                                    };
                                    if v == u || du + w != dist[v] {
                                        continue;
                                    }
                                    match hops[v].compare_exchange(
                                        usize::MAX,
                                        level,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    ) {
                                        Ok(_) => {
                                            next.push(v);
                                            parent[v].fetch_min(u, Ordering::Relaxed);
                                        }
                                        Err(l) if l == level => {
                                            parent[v].fetch_min(u, Ordering::Relaxed);
                                        }
                                        Err(_) => {}
                                    }
                                }
                            }
                        }
                    }
                    found[level % 2].fetch_add(next.len(), Ordering::Relaxed);
                    barrier.wait();
                    if tid == 0 {
                        cursor.store(0, Ordering::Relaxed);
                        found[(level + 1) % 2].store(0, Ordering::Relaxed);
                    }
                    let more = found[level % 2].load(Ordering::Relaxed);
                    *handoff[tid].write().unwrap() = next;
                    barrier.wait();
                    if more == 0 {
                        break;
                    }
                }
            });
        }
    });
}

/// What `engine` runs on `g` with `threads` threads: Δ-stepping with buckets `Some(Δ)` wide, or
/// (`None`) Dijkstra's algorithm.
pub(crate) fn resolve_engine<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    engine: ShortestPathEngine,
    threads: usize,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    Ok(match engine {
        ShortestPathEngine::Sequential => None,
        ShortestPathEngine::Auto if threads <= 1 || g.size() < PARALLEL_FROM_NODES => None,
        // (unweighted arcs: a parallel BFS, which takes no buckets; their width, unused, would
        // take a pass over the arcs)
        ShortestPathEngine::Auto if !E::is_weighted() => Some(1.),
        ShortestPathEngine::Auto => Some(default_delta(g)?),
        ShortestPathEngine::DeltaStepping { delta } => Some(match delta {
            Some(d) => d,
            None => default_delta(g)?,
        }),
    })
}

/// Every node's distance from `source` in `g`, into `out` ([`f64::INFINITY`] for unreachable
/// nodes): by Δ-stepping on `threads` threads with buckets `delta` wide, or (`None`) Dijkstra's
/// algorithm, its memory mapped in the 2 `scratch` files (removed after). Weights must have
/// been checked finite and non-negative.
pub(crate) fn distances_into<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    out: &mut [f64],
    delta: Option<f64>,
    threads: usize,
    scratch: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let n = g.size();
    assert_eq!(out.len(), n);
    // unweighted arcs: distances are hop counts, found by a BFS (in parallel, where Δ-stepping
    // would run), the same, exactly
    if !E::is_weighted() {
        let threads = if delta.is_some() { threads } else { 1 };
        return hop_distances(g, source, out, threads, scratch);
    }
    match delta {
        Some(delta) => {
            // `f64`s as their bits, for atomic minima (of the same size, and here aligned)
            let ptr = out.as_mut_ptr();
            assert_eq!(ptr as usize % std::mem::align_of::<AtomicU64>(), 0);
            let atomics = unsafe { std::slice::from_raw_parts(ptr as *const AtomicU64, n) };
            delta_stepping(g, source, atomics, delta, threads)
        }
        None => {
            let result = (|| -> Result<(), Box<dyn std::error::Error>> {
                let mut m = DijkstraMemory::<Ix>::new(scratch, n)?;
                dijkstra(g, source, &mut m).map_err(|e| e.to_string())?;
                for (v, d) in out.iter_mut().enumerate() {
                    *d = m.distance(v).unwrap_or(f64::INFINITY);
                }
                Ok(())
            })();
            for f in scratch {
                let _ = std::fs::remove_file(f);
            }
            result
        }
    }
}

/// Every node's number of arcs from `source` in `g` (as an `f64`), into `out` ([`f64::INFINITY`]
/// for unreachable nodes): a BFS, or, on more `threads` than one, a level-synchronous BFS in
/// parallel (threads take chunks of a level's nodes, each node claimed by the thread that finds
/// it first, with one compare-and-swap); its frontiers mapped in the 2 `scratch` files (removed
/// after). The distances Dijkstra's algorithm and Δ-stepping find on unweighted arcs, exactly.
fn hop_distances<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    out: &mut [f64],
    threads: usize,
    scratch: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let n = g.size();
    let csr = crate::graph::csr::Csr::of(g);
    out.fill(f64::INFINITY);
    out[source] = 0.;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let fronts = [
            SharedSliceMut::<usize>::abst_mem_mut(&scratch[0], n, true)?,
            SharedSliceMut::<usize>::abst_mem_mut(&scratch[1], n, true)?,
        ];
        let views = [fronts[0].shared_slice(), fronts[1].shared_slice()];
        if threads <= 1 {
            let mut queue = views[0];
            let queue = queue.as_mut_slice();
            queue[0] = source;
            let (mut head, mut tail) = (0, 1);
            while head < tail {
                let u = queue[head];
                head += 1;
                let d = out[u] + 1.;
                for &v in &csr.targets[csr.range(u)] {
                    let v = v.index();
                    if out[v] == f64::INFINITY {
                        out[v] = d;
                        queue[tail] = v;
                        tail += 1;
                    }
                }
            }
            return Ok(());
        }
        // `f64`s as their bits, for compare-and-swaps (of the same size, and here aligned)
        let ptr = out.as_mut_ptr();
        assert_eq!(ptr as usize % std::mem::align_of::<AtomicU64>(), 0);
        let dist = unsafe { std::slice::from_raw_parts(ptr as *const AtomicU64, n) };
        const UNREACHED: u64 = f64::INFINITY.to_bits();
        const CHUNK: usize = 256;
        // levels of fewer nodes are searched by one thread (see below)
        const SHARED_FROM: usize = 4 * CHUNK;
        {
            let mut first = views[0];
            *first.get_mut(0) = source;
        }
        let lengths = [AtomicUsize::new(1), AtomicUsize::new(0)];
        let next = AtomicUsize::new(0);
        let barrier = crate::utils::SpinBarrier::new(threads);
        // the level thread 0 reached alone (see below), for the others to resume from
        let resumed = AtomicUsize::new(0);
        let (views, lengths, next, barrier, csr, resumed) =
            (&views, &lengths, &next, &barrier, &csr, &resumed);
        std::thread::scope(|s| {
            for tid in 0..threads {
                s.spawn(move || {
                    // the nodes found, for the next frontier: a batch at a time
                    let mut found: Vec<usize> = Vec::with_capacity(CHUNK * 4);
                    let mut level = 0u64;
                    loop {
                        let (current, upcoming) = ((level & 1) as usize, 1 - (level & 1) as usize);
                        let len = lengths[current].load(Ordering::Acquire);
                        if len == 0 {
                            return;
                        }
                        // small frontiers: thread 0 alone, level after level, until one is
                        // worth sharing (or the search ends), the others waiting once. Sharing a
                        // level costs two barriers: on graphs of long paths (de Bruijn graphs'),
                        // tens of thousands of levels of a few nodes each made the parallel BFS
                        // 4 times as slow as a sequential one
                        if len < SHARED_FROM {
                            // (every thread has read `len` before thread 0 changes the lengths)
                            barrier.wait();
                            if tid == 0 {
                                let (mut current, mut len, mut at) = (current, len, level);
                                loop {
                                    let upcoming = 1 - current;
                                    let d = ((at + 1) as f64).to_bits();
                                    let mut frontier = views[upcoming];
                                    let mut found = 0;
                                    for i in 0..len {
                                        let u = *views[current].get(i);
                                        for &v in &csr.targets[csr.range(u)] {
                                            let v = v.index();
                                            if dist[v].load(Ordering::Relaxed) == UNREACHED {
                                                dist[v].store(d, Ordering::Relaxed);
                                                *frontier.get_mut(found) = v;
                                                found += 1;
                                            }
                                        }
                                    }
                                    lengths[current].store(0, Ordering::Relaxed);
                                    lengths[upcoming].store(found, Ordering::Relaxed);
                                    (current, len, at) = (upcoming, found, at + 1);
                                    if len == 0 || len >= SHARED_FROM {
                                        break;
                                    }
                                }
                                resumed.store(at as usize, Ordering::Relaxed);
                            }
                            barrier.wait();
                            level = resumed.load(Ordering::Relaxed) as u64;
                            continue;
                        }
                        level += 1;
                        let d = (level as f64).to_bits();
                        let flush = |found: &mut Vec<usize>| {
                            let at = lengths[upcoming].fetch_add(found.len(), Ordering::Relaxed);
                            let mut frontier = views[upcoming];
                            for (i, &v) in found.iter().enumerate() {
                                *frontier.get_mut(at + i) = v;
                            }
                            found.clear();
                        };
                        loop {
                            let begin = CHUNK * next.fetch_add(1, Ordering::Relaxed);
                            if begin >= len {
                                break;
                            }
                            for i in begin..(begin + CHUNK).min(len) {
                                let u = *views[current].get(i);
                                for &v in &csr.targets[csr.range(u)] {
                                    let v = v.index();
                                    if dist[v].load(Ordering::Relaxed) == UNREACHED
                                        && dist[v]
                                            .compare_exchange(
                                                UNREACHED,
                                                d,
                                                Ordering::Relaxed,
                                                Ordering::Relaxed,
                                            )
                                            .is_ok()
                                    {
                                        found.push(v);
                                        if found.len() == found.capacity() {
                                            flush(&mut found);
                                        }
                                    }
                                }
                            }
                        }
                        if !found.is_empty() {
                            flush(&mut found);
                        }
                        barrier.wait();
                        if tid == 0 {
                            lengths[current].store(0, Ordering::Relaxed);
                            next.store(0, Ordering::Relaxed);
                        }
                        barrier.wait();
                    }
                });
            }
        });
        Ok(())
    })();
    for f in scratch {
        let _ = std::fs::remove_file(f);
    }
    result
}

/// The number of finite `dist`ances, and their sum: summed in blocks of a fixed size, in
/// parallel, the blocks' sums then added in order, so that it's the same whatever the threads.
fn totals(dist: &[f64], threads: usize) -> (usize, f64) {
    const BLOCK: usize = 1 << 16;
    let blocks = dist.len().div_ceil(BLOCK);
    let threads = threads.clamp(1, blocks.max(1));
    let per = blocks.div_ceil(threads);
    let parts: Vec<Vec<(usize, f64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    ((t * per).min(blocks)..((t + 1) * per).min(blocks))
                        .map(|b| {
                            let block = &dist[b * BLOCK..((b + 1) * BLOCK).min(dist.len())];
                            block
                                .iter()
                                .filter(|d| d.is_finite())
                                .fold((0, 0.), |(r, s), &d| (r + 1, s + d))
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts
        .into_iter()
        .flatten()
        .fold((0, 0.), |(r, s), (br, bs)| (r + br, s + bs))
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::centralities::exact::{
        ExactClosenessCentrality, ExactHarmonicCentrality, ExactLinCentrality,
    };
    use crate::components::test_utils::random_arcs;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};

    pub(crate) type WGraph<Dir> = TestGraph<VoidLabel, Weighted<f64>, usize, Dir>;

    /// A graph of `arcs` `(u, v, weight)`: undirected (each edge listed once) or directed.
    pub(crate) fn weighted<Dir: Directedness>(
        name: &str,
        n: usize,
        arcs: &[(usize, usize, f64)],
    ) -> WGraph<Dir> {
        weighted_on::<Dir>(name, n, arcs, 3)
    }

    pub(crate) fn weighted_on<Dir: Directedness>(
        name: &str,
        n: usize,
        arcs: &[(usize, usize, f64)],
        threads: u8,
    ) -> WGraph<Dir> {
        let dir = unique_test_dir("dijkstra");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let symmetry = if Dir::DIRECTED {
            "general"
        } else {
            "symmetric"
        };
        let mut text = format!(
            "%%MatrixMarket matrix coordinate real {symmetry}\n{n} {n} {}\n",
            arcs.len()
        );
        for &(u, v, w) in arcs {
            let (u, v) = if Dir::DIRECTED {
                (u, v)
            } else {
                (u.max(v), u.min(v))
            };
            text += &format!("{} {} {w}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("dijkstra_{name}_{}", unique_run_tag());
        TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    /// Random arcs with weights: quarters (exact sums) or arbitrary reals, zeros included, and
    /// self-loops and parallel arcs (of `random_arcs`, plus a few more).
    pub(crate) fn weighted_arcs(
        n: usize,
        m: usize,
        seed: u64,
        reals: bool,
    ) -> Vec<(usize, usize, f64)> {
        let mut arcs = random_arcs(n, m, seed);
        arcs.extend([(1, 1), (2, 3), (2, 3), (3, 2)]);
        let mut state = seed;
        arcs.into_iter()
            .map(|(u, v)| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (state >> 33) % 40;
                let w = if reals { r as f64 / 7.3 } else { r as f64 / 4. };
                (u, v, w)
            })
            .collect()
    }

    /// Every pair's distance (`INFINITY` if none), by Floyd–Warshall.
    pub(crate) fn all_pairs(
        n: usize,
        arcs: &[(usize, usize, f64)],
        directed: bool,
    ) -> Vec<Vec<f64>> {
        let mut d = vec![vec![f64::INFINITY; n]; n];
        for (u, row) in d.iter_mut().enumerate() {
            row[u] = 0.;
        }
        for &(u, v, w) in arcs {
            d[u][v] = d[u][v].min(w);
            if !directed {
                d[v][u] = d[v][u].min(w);
            }
        }
        for k in 0..n {
            let via = d[k].clone();
            for row in d.iter_mut() {
                let dk = row[k];
                for (r, &x) in row.iter_mut().zip(&via) {
                    *r = r.min(dk + x);
                }
            }
        }
        d
    }

    pub(crate) fn close(a: f64, b: f64) -> bool {
        a == b || (a - b).abs() <= 1e-9 * (1. + b.abs())
    }

    #[test]
    fn distances_match_floyd_warshall() {
        for seed in 0..6u64 {
            let n = 60;
            let reals = seed % 2 == 1;
            let arcs = weighted_arcs(n, 200, seed, reals);
            for directed in [false, true] {
                let want = all_pairs(n, &arcs, directed);
                let check = |got: &[f64], source: usize, what: &str| {
                    for v in 0..n {
                        let ok = if reals {
                            close(got[v], want[source][v])
                        } else {
                            got[v] == want[source][v]
                        };
                        assert!(
                            ok,
                            "{what} seed {seed}: {source} -> {v}: {} vs {}",
                            got[v], want[source][v]
                        );
                    }
                };
                if directed {
                    let g = weighted::<Directed>(&format!("d{seed}"), n, &arcs);
                    for source in [0, 7, 59] {
                        let mut d = DijkstraDists::new(&*g, source).unwrap();
                        check(d.get_distances(), source, "out");
                        let reach = want[source].iter().filter(|x| x.is_finite()).count();
                        assert_eq!(d.reachable(), reach);
                        d.drop_cache().unwrap();
                        // distances to `source`: along reversed arcs
                        let mut d = DijkstraDists::new_directed(&g, source, Direction::In).unwrap();
                        for (v, (&got, row)) in d.get_distances().iter().zip(&want).enumerate() {
                            assert!(close(got, row[source]), "in seed {seed}: {v} -> {source}");
                        }
                        d.drop_cache().unwrap();
                    }
                } else {
                    let g = weighted::<Undirected>(&format!("u{seed}"), n, &arcs);
                    for source in [0, 7, 59] {
                        let mut d = DijkstraDists::new(&*g, source).unwrap();
                        check(d.get_distances(), source, "undirected");
                        d.drop_cache().unwrap();
                    }
                }
            }
        }
    }

    /// Reference centralities from all pairs' distances, networkx's way: nodes at distance zero
    /// count as reached, but add nothing to harmonic sums.
    fn references(d: &[Vec<f64>]) -> Vec<(usize, f64, f64)> {
        d.iter()
            .map(|row| {
                let reached: Vec<f64> = row.iter().copied().filter(|x| x.is_finite()).collect();
                let total: f64 = reached.iter().sum();
                let inverse: f64 = reached.iter().filter(|&&x| x > 0.).map(|x| 1. / x).sum();
                (reached.len(), total, inverse)
            })
            .collect()
    }

    #[test]
    fn exact_centralities_use_weighted_distances() {
        for seed in 0..4u64 {
            let n = 300;
            let arcs = weighted_arcs(n, 900, 10 + seed, seed % 2 == 0);
            for directed in [false, true] {
                // out-distances, and (on digraphs) in-distances: the transposed matrix
                let d = all_pairs(n, &arcs, directed);
                let d_in: Vec<Vec<f64>> =
                    (0..n).map(|u| (0..n).map(|v| d[v][u]).collect()).collect();
                let check = |c: &[f64], h: &[f64], l: &[f64], d: &[Vec<f64>], what: &str| {
                    for (u, &(r, total, inverse)) in references(d).iter().enumerate() {
                        let closeness = if r <= 1 || total == 0. {
                            0.
                        } else {
                            (r - 1) as f64 / total
                        };
                        let harmonic = if r <= 1 { 0. } else { inverse / (r - 1) as f64 };
                        let lin = if total == 0. {
                            1.
                        } else {
                            (r * r) as f64 / total
                        };
                        assert!(
                            close(c[u], closeness),
                            "{what} closeness {u}: {} vs {closeness}",
                            c[u]
                        );
                        assert!(
                            close(h[u], harmonic),
                            "{what} harmonic {u}: {} vs {harmonic}",
                            h[u]
                        );
                        assert!(close(l[u], lin), "{what} lin {u}: {} vs {lin}", l[u]);
                    }
                };
                macro_rules! run {
                    ($g:expr, $new:ident $(, $dir:expr)?) => {{
                        let mut c = ExactClosenessCentrality::<_, _, _, f64, _>::$new($g, Some(true) $(, $dir)?).unwrap();
                        let mut h = ExactHarmonicCentrality::<_, _, _, f64, _>::$new($g, Some(true) $(, $dir)?).unwrap();
                        let mut l = ExactLinCentrality::<_, _, _, f64, _>::$new($g $(, $dir)?).unwrap();
                        let out = (c.centralities().to_vec(), h.centralities().to_vec(), l.centralities().to_vec());
                        c.drop_cache().unwrap();
                        h.drop_cache().unwrap();
                        l.drop_cache().unwrap();
                        out
                    }};
                }
                if directed {
                    let g = weighted::<Directed>(&format!("cd{seed}"), n, &arcs);
                    let (c, h, l) = run!(&*g, new);
                    check(&c, &h, &l, &d, &format!("out {seed}"));
                    let (c, h, l) = run!(&g, new_directed, Direction::In);
                    check(&c, &h, &l, &d_in, &format!("in {seed}"));
                } else {
                    let g = weighted::<Undirected>(&format!("cu{seed}"), n, &arcs);
                    let (c, h, l) = run!(&*g, new);
                    check(&c, &h, &l, &d, &format!("undirected {seed}"));
                }
            }
        }
    }

    /// The parallel BFS (unweighted arcs, where Δ-stepping would run) finds the sequential
    /// one's distances, on graphs whose frontiers grow past what it shares between threads and
    /// shrink below it, again and again (dense clusters joined by long paths: levels searched by
    /// one thread, then by all, and back), on any number of threads.
    #[test]
    fn parallel_bfs_switches_between_one_thread_and_all() {
        let (clusters, size, path) = (8, 3000, 400);
        let mut arcs = vec![];
        let mut next = 0x5eedu64;
        let mut random = move |m: usize| {
            next = next
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (next >> 33) as usize % m
        };
        let cluster_start = |c: usize| c * (size + path);
        for c in 0..clusters {
            let base = cluster_start(c);
            for _ in 0..size * 4 {
                let (u, v) = (base + random(size), base + random(size));
                arcs.push((u, v));
                arcs.push((v, u));
            }
            if c + 1 < clusters {
                // a path from this cluster to the next, both ways
                let mut prev = base + size - 1;
                for i in 0..path {
                    let v = base + size + i;
                    arcs.push((prev, v));
                    arcs.push((v, prev));
                    prev = v;
                }
                arcs.push((prev, cluster_start(c + 1)));
                arcs.push((cluster_start(c + 1), prev));
            }
        }
        let n = cluster_start(clusters - 1) + size;
        // each source's distances, by the sequential BFS (one thread)
        let mut want: Vec<Vec<u64>> = vec![];
        for threads in [1u8, 2, 7, 16] {
            let g = crate::components::test_utils::graph_with_threads(
                &format!("bfs_switch_{threads}"),
                "general",
                n,
                &arcs,
                threads,
            );
            for (i, source) in [0, cluster_start(3) + 17].into_iter().enumerate() {
                let engine = if threads == 1 {
                    ShortestPathEngine::Sequential
                } else {
                    ShortestPathEngine::DeltaStepping { delta: None }
                };
                let mut d = DijkstraDists::with_engine(&*g, source, engine).unwrap();
                let got: Vec<u64> = d.get_distances().iter().map(|x| x.to_bits()).collect();
                d.drop_cache().unwrap();
                if threads == 1 {
                    want.push(got);
                } else {
                    assert_eq!(got, want[i], "{threads} threads, source {source}");
                }
            }
        }
        // the far end: past every cluster and path
        let far = f64::from_bits(want[0][n - 1]);
        assert!(far > (path * (clusters - 1)) as f64, "{far}");
    }

    /// Unit weights give the hop counts BFSs do: the same centralities (closeness and Lin's bit
    /// for bit, their sums being of integers).
    #[test]
    fn unit_weights_give_hop_distances() {
        let n = 200;
        let arcs: Vec<_> = random_arcs(n, 500, 5)
            .into_iter()
            .map(|(u, v)| (u, v, 1.))
            .collect();
        let w = weighted::<Undirected>("unit", n, &arcs);
        let plain: Vec<_> = arcs.iter().map(|&(u, v, _)| (u, v)).collect();
        let p = crate::components::test_utils::graph("unit_plain", "symmetric", n, &plain);
        let p = p.to_undirected().unwrap();
        let mut a = ExactClosenessCentrality::<_, _, _, f64, _>::new(&*w, Some(false)).unwrap();
        let mut b = ExactClosenessCentrality::<_, _, _, f64, _>::new(p, Some(false)).unwrap();
        assert_eq!(a.centralities(), b.centralities());
        a.drop_cache().unwrap();
        b.drop_cache().unwrap();
        let mut a = ExactHarmonicCentrality::<_, _, _, f64, _>::new(&*w, None).unwrap();
        let mut b = ExactHarmonicCentrality::<_, _, _, f64, _>::new(p, None).unwrap();
        for (x, y) in a.centralities().iter().zip(b.centralities()) {
            assert!(close(*x, *y));
        }
        a.drop_cache().unwrap();
        b.drop_cache().unwrap();
        for source in [0, 50, 199] {
            let mut d = DijkstraDists::new(&*w, source).unwrap();
            let mut bfs = crate::trails::bfs::BFSDists::new(p, source).unwrap();
            for (v, &x) in d.get_distances().iter().enumerate() {
                let hops = bfs.get_distances()[v];
                let reached = v == source || hops != 0;
                assert_eq!(
                    x,
                    if reached { hops as f64 } else { f64::INFINITY },
                    "{source} -> {v}"
                );
            }
            assert_eq!(d.reachable(), bfs.reachable());
            assert_eq!(d.total_distances(), bfs.total_distances());
            d.drop_cache().unwrap();
            bfs.drop_cache().unwrap();
        }
    }

    #[test]
    fn negative_weights_and_bad_sources_are_errors() {
        let g = weighted::<Directed>("negative", 3, &[(0, 1, 1.), (1, 2, -0.5)]);
        let err = DijkstraDists::new(&*g, 0).err().unwrap();
        assert!(err.to_string().contains("non-negative"), "{err}");
        let err = ExactClosenessCentrality::<_, _, _, f64, _>::new(&*g, None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("non-negative"), "{err}");
        let g = weighted::<Directed>("fine", 3, &[(0, 1, 1.)]);
        assert!(DijkstraDists::new(&*g, 3).is_err());
    }

    /// Δ-stepping finds Dijkstra's distances, bit for bit, and the same totals, whatever its
    /// threads and Δ (even Δs smaller than the ring of buckets allows, or than every weight, or
    /// larger than every distance), on graphs with zero weights, self-loops, parallel arcs and
    /// unreachable nodes, weighted or not.
    #[test]
    fn delta_stepping_finds_dijkstras_distances() {
        let engines = [
            ShortestPathEngine::DeltaStepping { delta: None },
            ShortestPathEngine::DeltaStepping { delta: Some(1e-9) },
            ShortestPathEngine::DeltaStepping { delta: Some(0.3) },
            ShortestPathEngine::DeltaStepping { delta: Some(1e6) },
        ];
        fn compare<A: Directedness, B: Directedness, E: graph::E>(
            name: &str,
            a: &DijkstraDists<VoidLabel, E, usize, A>,
            b: &DijkstraDists<VoidLabel, E, usize, B>,
        ) {
            for (v, (x, y)) in a.get_distances().iter().zip(b.get_distances()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{name}: node {v}: {x} vs {y}");
            }
            assert_eq!(a.reachable(), b.reachable(), "{name}");
            assert_eq!(
                a.total_distances().to_bits(),
                b.total_distances().to_bits(),
                "{name}"
            );
        }
        let seq = ShortestPathEngine::Sequential;
        for seed in 0..3u64 {
            let n = 700;
            // two parts, joined by one arc of weight zero
            let mut arcs = weighted_arcs(n / 2, 2500, 30 + seed, seed != 1);
            arcs.extend(
                weighted_arcs(n / 2, 2500, 40 + seed, true)
                    .into_iter()
                    .map(|(u, v, w)| (u + n / 2, v + n / 2, w)),
            );
            arcs.push((10, n / 2 + 3, 0.));
            for threads in [1u8, 2, 5, 8] {
                let g =
                    weighted_on::<Directed>(&format!("ds_d{seed}_{threads}"), n, &arcs, threads);
                let u =
                    weighted_on::<Undirected>(&format!("ds_u{seed}_{threads}"), n, &arcs, threads);
                for source in [0, 10, n - 1] {
                    let mut want = DijkstraDists::with_engine(&*g, source, seq).unwrap();
                    let mut want_in =
                        DijkstraDists::new_directed_with_engine(&g, source, Direction::In, seq)
                            .unwrap();
                    let mut want_u = DijkstraDists::with_engine(&*u, source, seq).unwrap();
                    for engine in engines {
                        let what =
                            format!("seed {seed}, {threads} threads, source {source}, {engine:?}");
                        let mut got = DijkstraDists::with_engine(&*g, source, engine).unwrap();
                        compare(&format!("directed {what}"), &got, &want);
                        got.drop_cache().unwrap();
                        let mut got = DijkstraDists::new_directed_with_engine(
                            &g,
                            source,
                            Direction::In,
                            engine,
                        )
                        .unwrap();
                        compare(&format!("in {what}"), &got, &want_in);
                        got.drop_cache().unwrap();
                        let mut got = DijkstraDists::with_engine(&*u, source, engine).unwrap();
                        compare(&format!("undirected {what}"), &got, &want_u);
                        got.drop_cache().unwrap();
                    }
                    want.drop_cache().unwrap();
                    want_in.drop_cache().unwrap();
                    want_u.drop_cache().unwrap();
                }
            }
        }
        // unweighted: Δ-stepping's buckets are BFS levels
        let arcs = random_arcs(500, 1500, 9);
        let p =
            crate::components::test_utils::graph_with_threads("ds_plain", "general", 500, &arcs, 6);
        for engine in engines {
            let mut a = DijkstraDists::with_engine(&*p, 3, engine).unwrap();
            let mut b = DijkstraDists::with_engine(&*p, 3, seq).unwrap();
            for (x, y) in a.get_distances().iter().zip(b.get_distances()) {
                assert_eq!(x.to_bits(), y.to_bits(), "unweighted {engine:?}");
            }
            assert_eq!(a.reachable(), b.reachable());
            a.drop_cache().unwrap();
            b.drop_cache().unwrap();
        }
        let bad = DijkstraDists::with_engine(
            &*p,
            3,
            ShortestPathEngine::DeltaStepping { delta: Some(0.) },
        );
        assert!(bad.is_err());
    }

    /// Shortest path trees: found only when asked for (no file before), made of arcs on
    /// shortest paths whose weights add up to the distances exactly, acyclic despite zero-weight
    /// cycles, and the same whatever the engine and the threads.
    #[test]
    fn shortest_path_trees() {
        let seq = ShortestPathEngine::Sequential;
        let delta = ShortestPathEngine::DeltaStepping { delta: None };
        for seed in 0..4u64 {
            let n = 150;
            let mut arcs = weighted_arcs(n, 600, 80 + seed, seed % 2 == 1);
            // a zero-weight cycle, entered from both ends
            arcs.extend([(4, 5, 0.), (5, 4, 0.), (0, 4, 1.), (0, 5, 1.)]);
            for directed in [false, true] {
                let d = all_pairs(n, &arcs, directed);
                let mut trees = vec![];
                for threads in [1u8, 3, 8] {
                    macro_rules! check {
                        ($Dir:ty) => {{
                            let g = weighted_on::<$Dir>(
                                &format!("t{seed}_{directed}_{threads}"),
                                n,
                                &arcs,
                                threads,
                            );
                            for engine in [seq, delta] {
                                let mut dd = DijkstraDists::with_engine(&*g, 0, engine).unwrap();
                                let tree_file = g
                                    .build_cache_filename(CacheFile::Dijkstra, Some(3))
                                    .unwrap();
                                assert!(!std::path::Path::new(&tree_file).exists());
                                let dist = dd.get_distances().to_vec();
                                trees.push(
                                    dd.predecessors()
                                        .unwrap()
                                        .iter()
                                        .map(|p| p.index())
                                        .collect::<Vec<_>>(),
                                );
                                for v in 0..n {
                                    match dd.path_to(v).unwrap() {
                                        None => assert!(d[0][v].is_infinite()),
                                        Some(path) => {
                                            assert_eq!((path[0], *path.last().unwrap()), (0, v));
                                            let mut length = 0.;
                                            for pair in path.windows(2) {
                                                let w = arcs
                                                    .iter()
                                                    .filter(|&&(a, b, _)| {
                                                        (a, b) == (pair[0], pair[1])
                                                            || (!directed
                                                                && (b, a) == (pair[0], pair[1]))
                                                    })
                                                    .map(|a| a.2)
                                                    .fold(f64::INFINITY, f64::min);
                                                length += w;
                                            }
                                            assert_eq!(length.to_bits(), dist[v].to_bits(), "{v}");
                                            assert!(close(dist[v], d[0][v]));
                                        }
                                    }
                                }
                                dd.drop_cache().unwrap();
                                assert!(!std::path::Path::new(&tree_file).exists());
                            }
                        }};
                    }
                    if directed {
                        check!(Directed)
                    } else {
                        check!(Undirected)
                    }
                }
                assert!(trees.windows(2).all(|t| t[0] == t[1]), "seed {seed}");
            }
        }
    }

    /// Converting a weighted graph's cache to another index type keeps every arc's weight, so
    /// weighted algorithms find the same results on the converted graph.
    /// Unweighted distances, by the parallel BFS, switching between one thread (the long paths'
    /// levels of a node or two) and all of them (a hub's thousands of neighbours), and back: the
    /// hop counts a sequential BFS finds, whatever the threads.
    #[test]
    fn parallel_hops_through_small_and_large_levels() {
        // a path of 3000 nodes, a hub at its end linked to 3000 nodes, each linked to the next
        // path's start through a node of its own, and a path of 2000 nodes from there
        let (path, fan, tail) = (3000, 3000, 2000);
        let hub = path;
        let join = hub + fan + 1;
        let n = join + tail;
        let mut arcs = vec![];
        for u in 0..path {
            arcs.push((u, u + 1));
        }
        for i in 0..fan {
            arcs.push((hub, hub + 1 + i));
            arcs.push((hub + 1 + i, join));
        }
        for u in join..n - 1 {
            arcs.push((u, u + 1));
        }
        let mut want = vec![f64::INFINITY; n];
        for (u, d) in want.iter_mut().enumerate().take(hub + 1) {
            *d = u as f64;
        }
        for d in &mut want[hub + 1..join] {
            *d = (hub + 1) as f64;
        }
        for (i, d) in want[join..].iter_mut().enumerate() {
            *d = (hub + 2 + i) as f64;
        }
        for threads in [1u8, 3, 8] {
            let g = crate::components::test_utils::graph_with_threads(
                &format!("hops_{threads}"),
                "general",
                n,
                &arcs,
                threads,
            );
            let engine = ShortestPathEngine::DeltaStepping { delta: None };
            let mut d = DijkstraDists::with_engine(&*g, 0, engine).unwrap();
            assert_eq!(d.get_distances(), want.as_slice(), "{threads} threads");
            d.drop_cache().unwrap();
        }
    }

    #[test]
    fn index_conversion_keeps_weights() {
        let arcs = weighted_arcs(200, 900, 61, true);
        fn check<Dir: Directedness, NewIx: graph::IndexType>(g: &WGraph<Dir>) {
            let mut c = g.convert_index_type::<NewIx>().unwrap();
            assert_eq!((c.size(), c.width()), (g.size(), g.width()));
            for u in 0..g.size() {
                let a: Vec<(usize, u64)> = g
                    .weighted_neighbours(u)
                    .unwrap()
                    .map(|(v, l)| (v, l.weight::<f64>().to_bits()))
                    .collect();
                let b: Vec<(usize, u64)> = c
                    .weighted_neighbours(u)
                    .unwrap()
                    .map(|(v, l)| (v, l.weight::<f64>().to_bits()))
                    .collect();
                assert_eq!(a, b, "node {u}");
            }
            assert_eq!(
                c.total_weight().unwrap().to_bits(),
                g.total_weight().unwrap().to_bits()
            );
            let mut d0 = DijkstraDists::new(&**g, 0).unwrap();
            let mut d1 = DijkstraDists::new(&c, 0).unwrap();
            assert_eq!(d0.get_distances(), d1.get_distances());
            d0.drop_cache().unwrap();
            d1.drop_cache().unwrap();
            c.drop_cache().unwrap();
        }
        let g = weighted::<Undirected>("convert_u", 200, &arcs);
        check::<Undirected, u32>(&g);
        check::<Undirected, u16>(&g);
        let d = weighted::<Directed>("convert_d", 200, &arcs);
        check::<Directed, u32>(&d);
        check::<Directed, u16>(&d);
    }
}
