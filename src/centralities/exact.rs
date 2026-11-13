use std::mem::ManuallyDrop;

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crate::graph::csr::Csr;
use crate::trails::dijkstra::{DijkstraMemory, WeightedReach, dijkstra};

use crossbeam::thread;

/// What a BFS from a node finds: the nodes it reaches (itself included), and the sums of their
/// distances and inverse distances to it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Reach {
    pub(crate) reached: usize,
    pub(crate) distances: u64,
    pub(crate) inverse_distances: f64,
}

/// Runs a BFS from every node of `g` (following its arcs), passing each node's [`Reach`] to
/// `store`, on `g`'s threads. Scratch files are `kind`'s.
fn reach_of_every_node<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    kind: CacheFile,
    store: &(dyn Fn(usize, Reach) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    let node_count = g.size();
    // a batch of sources per thread at least, taken as they finish
    let threads = g.thread_num().min(node_count.div_ceil(BATCH)).max(1);
    let next_batch = std::sync::atomic::AtomicUsize::new(0);
    let order_fn = g.build_cache_filename(kind.clone(), Some(6 * threads))?;
    let visited_fn = g.build_cache_filename(kind.clone(), Some(6 * threads + 1))?;
    let order = visit_order(g, &order_fn, &visited_fn);
    let _ = std::fs::remove_file(&visited_fn);
    let order = order?;
    let order_slice = order.as_slice();

    let result = thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let mut handles = vec![];
        for tid in 0..threads {
            let next_batch = &next_batch;
            // each thread's scratch memory, allocated once for all its BFSs
            let files: Vec<String> = (0..5)
                .map(|k| g.build_cache_filename(kind.clone(), Some(6 * tid + k)))
                .collect::<Result<_, _>>()?;
            handles.push(scope.spawn(
                move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let nodes =
                        |k: usize| SharedSliceMut::<Ix>::abst_mem_mut(&files[k], node_count, true);
                    let built = (|| {
                        Ok::<_, Box<dyn std::error::Error>>(BatchScratch {
                            marks: SharedSliceMut::<Marks>::abst_mem_mut(
                                &files[0], node_count, true,
                            )?,
                            visit: SharedSliceMut::<Word>::abst_mem_mut(
                                &files[1], node_count, true,
                            )?,
                            frontier: nodes(2)?,
                            next_frontier: nodes(3)?,
                            touched: nodes(4)?,
                        })
                    })()
                    .map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() },
                    );
                    let result = built.and_then(|mut scratch| {
                        let mut reach = [Reach::default(); BATCH];
                        loop {
                            let first = BATCH
                                * next_batch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if first >= node_count {
                                break Ok(());
                            }
                            let sources =
                                &order_slice[first..std::cmp::min(first + BATCH, node_count)];
                            batch_reach(g, sources, &mut scratch, &mut reach)?;
                            for (source, &r) in sources.iter().zip(&reach) {
                                store(source.index(), r);
                            }
                        }
                    });
                    for f in &files {
                        let _ = std::fs::remove_file(f);
                    }
                    result
                },
            ));
        }
        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error in BFS thread {tid}: {e}").into()
                })?;
        }
        Ok(())
    })
    .map_err(|e| -> Box<dyn std::error::Error> {
        format!("error exact centralities {:?}", e).into()
    });
    drop(order);
    let _ = std::fs::remove_file(&order_fn);
    result?
}

/// Runs Dijkstra's algorithm from every node of `g` (following its arcs, as long as their
/// weights), passing what each finds to `store`, on `g`'s threads: [`reach_of_every_node`]'s
/// counterpart on weighted graphs. Scratch files are `kind`'s.
fn weighted_reach_of_every_node<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    kind: CacheFile,
    store: &(dyn Fn(usize, WeightedReach) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    // checks the weights finite and non-negative
    g.total_weight()?;
    let node_count = g.size();
    // sources taken a few at a time, as threads finish them (their costs vary widely)
    const CHUNK: usize = 16;
    let threads = g.thread_num().min(node_count.div_ceil(CHUNK)).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let mut handles = vec![];
        for tid in 0..threads {
            let next = &next;
            let files: Vec<String> = (0..5)
                .map(|k| g.build_cache_filename(kind.clone(), Some(6 * tid + k)))
                .collect::<Result<_, _>>()?;
            handles.push(scope.spawn(
                move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let memory = DijkstraMemory::<Ix>::new(&files, node_count).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() },
                    );
                    let result = memory.and_then(|mut memory| {
                        loop {
                            let first =
                                CHUNK * next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if first >= node_count {
                                break Ok(());
                            }
                            for source in first..(first + CHUNK).min(node_count) {
                                store(source, dijkstra(g, source, &mut memory)?);
                            }
                        }
                    });
                    for f in &files {
                        let _ = std::fs::remove_file(f);
                    }
                    result
                },
            ));
        }
        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error in Dijkstra thread {tid}: {e}").into()
                })?;
        }
        Ok(())
    })
    .map_err(|e| -> Box<dyn std::error::Error> {
        format!("error exact centralities {:?}", e).into()
    })?
}

/// Every node's centrality, by `set(u, reached, distances, inverse_distances)` from what BFSs
/// from it find, or, on weighted graphs, Dijkstra's algorithm (see [`WeightedReach`]).
fn every_node<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    kind: CacheFile,
    set: &(dyn Fn(usize, usize, f64, f64) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    if E::is_weighted() {
        weighted_reach_of_every_node(g, kind, &|u, r| {
            set(u, r.reached, r.distances, r.inverse_distances)
        })
    } else {
        reach_of_every_node(g, kind, &|u, r| {
            set(u, r.reached, r.distances as f64, r.inverse_distances)
        })
    }
}

/// The nodes of `g` in the order BFSs visit them (from node 0, then from the first node not
/// visited yet, and so on), so that nodes close in the graph tend to be close in it: batched in
/// that order, [`batch_reach`]'s sources' BFSs overlap more, sharing more of their work.
fn visit_order<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    order_fn: &str,
    visited_fn: &str,
) -> Result<AbstractedProceduralMemoryMut<Ix>, Box<dyn std::error::Error>> {
    let node_count = g.size();
    let order_mem = SharedSliceMut::<Ix>::abst_mem_mut(order_fn, node_count, true)?;
    let visited_mem = SharedSliceMut::<bool>::abst_mem_mut(visited_fn, node_count, true)?;
    let (mut order, mut visited) = (order_mem.shared_slice(), visited_mem.shared_slice());
    // the order is the BFSs' queue: nodes are appended as they're found
    let (mut head, mut tail) = (0, 0);
    for start in 0..node_count {
        if *visited.get(start) {
            continue;
        }
        *visited.get_mut(start) = true;
        *order.get_mut(tail) = Ix::new(start);
        tail += 1;
        while head < tail {
            for v in g.neighbours(order.get(head).index())? {
                if !*visited.get(v) {
                    *visited.get_mut(v) = true;
                    *order.get_mut(tail) = Ix::new(v);
                    tail += 1;
                }
            }
            head += 1;
        }
    }
    Ok(order_mem)
}

/// Words of [`BATCH`] bits, a source's each, so that 64-bit lanes are processed together.
const LANES: usize = 4;
/// Sources whose BFSs [`batch_reach`] runs together: a bit of a [`Word`] each.
const BATCH: usize = 64 * LANES;
type Word = [u64; LANES];

#[inline(always)]
fn any(word: &Word) -> bool {
    word.iter().fold(0, |a, &b| a | b) != 0
}

/// Which of a batch's sources reached a node, and which reach it at the next level: together,
/// as a node's neighbour checks read both.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Marks {
    seen: Word,
    next: Word,
}

/// A thread's memory for [`batch_reach`], for every node: its [`Marks`], and which of the batch's
/// sources have it in their current frontier (bit `i` for source `i`); and the nodes of the
/// frontiers, and those any source reached (whose marks to clear for the next batch).
struct BatchScratch<Ix: graph::IndexType> {
    marks: AbstractedProceduralMemoryMut<Marks>,
    visit: AbstractedProceduralMemoryMut<Word>,
    frontier: AbstractedProceduralMemoryMut<Ix>,
    next_frontier: AbstractedProceduralMemoryMut<Ix>,
    touched: AbstractedProceduralMemoryMut<Ix>,
}

/// BFSs from up to [`BATCH`] `sources` at once, as described in ["The More the Merrier: Efficient
/// Multi-Source Graph Traversal"](https://doi.org/10.14778/2735496.2735507) by Then M. et al.:
/// level by level, a node's neighbours are visited once for all the sources reaching it, which
/// are found with bitwise operations. `reach[i]` is set to the [`Reach`] of the `i`-th source.
fn batch_reach<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    sources: &[Ix],
    scratch: &mut BatchScratch<Ix>,
    reach: &mut [Reach; BATCH],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (mut marks, mut visit) = (scratch.marks.shared_slice(), scratch.visit.shared_slice());
    let (mut frontier, mut next_frontier, mut touched) = (
        scratch.frontier.shared_slice(),
        scratch.next_frontier.shared_slice(),
        scratch.touched.shared_slice(),
    );
    let batch = sources.len();
    for (i, source) in sources.iter().enumerate() {
        let source = source.index();
        marks.get_mut(source).seen[i / 64] = 1 << (i % 64);
        visit.get_mut(source)[i / 64] = 1 << (i % 64);
        *frontier.get_mut(i) = Ix::new(source);
        *touched.get_mut(i) = Ix::new(source);
        reach[i] = Reach {
            reached: 1,
            ..Reach::default()
        };
    }
    let (mut frontier_len, mut touched_len) = (batch, batch);
    let mut found = [0u64; BATCH];
    let mut level = 0u64;
    while frontier_len > 0 {
        level += 1;
        let mut next_len = 0;
        for f in 0..frontier_len {
            let v = frontier.get(f).index();
            let sources = std::mem::take(visit.get_mut(v));
            for w in g.neighbours(v).map_err(|e| e.to_string())? {
                let marks_w = marks.get_mut(w);
                let mut new = Word::default();
                for (n, (s, seen)) in new.iter_mut().zip(sources.iter().zip(&marks_w.seen)) {
                    *n = s & !seen;
                }
                if !any(&new) {
                    continue;
                }
                if !any(&marks_w.seen) {
                    *touched.get_mut(touched_len) = Ix::new(w);
                    touched_len += 1;
                }
                if !any(&marks_w.next) {
                    *next_frontier.get_mut(next_len) = Ix::new(w);
                    next_len += 1;
                }
                for (lane, n) in new.iter().enumerate() {
                    marks_w.seen[lane] |= n;
                    marks_w.next[lane] |= n;
                }
            }
        }
        // the nodes each source reached at this level, which are its next frontier
        for f in 0..next_len {
            let w = next_frontier.get(f).index();
            let sources = std::mem::take(&mut marks.get_mut(w).next);
            *visit.get_mut(w) = sources;
            for (lane, &bits) in sources.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    found[64 * lane + bits.trailing_zeros() as usize] += 1;
                    bits &= bits - 1;
                }
            }
        }
        for (r, found) in reach.iter_mut().zip(found.iter_mut()).take(batch) {
            if *found != 0 {
                r.reached += *found as usize;
                r.distances += *found * level;
                r.inverse_distances += *found as f64 / level as f64;
                *found = 0;
            }
        }
        std::mem::swap(&mut frontier, &mut next_frontier);
        frontier_len = next_len;
    }
    for t in 0..touched_len {
        *marks.get_mut(touched.get(t).index()) = Marks::default();
    }
    Ok(())
}

#[cfg(test)]
/// A BFS from `source`, level by level. A node is visited once its `stamp` is `source + 1`, so
/// that the stamps of earlier sources' BFSs needn't be cleared.
fn single_source_reach<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    stamp: &mut SharedSliceMut<u32>,
    queue: &mut SharedSliceMut<Ix>,
) -> Result<Reach, Box<dyn std::error::Error + Send + Sync>> {
    let mark = source as u32 + 1;
    let mut reach = Reach {
        reached: 1,
        ..Reach::default()
    };
    *stamp.get_mut(source) = mark;
    *queue.get_mut(0) = Ix::new(source);
    let (mut level_start, mut tail) = (0, 1);
    let mut level = 0u64;
    while level_start < tail {
        let level_end = tail;
        level += 1;
        for i in level_start..level_end {
            for v in g
                .neighbours(queue.get(i).index())
                .map_err(|e| e.to_string())?
            {
                if *stamp.get(v) != mark {
                    *stamp.get_mut(v) = mark;
                    *queue.get_mut(tail) = Ix::new(v);
                    tail += 1;
                }
            }
        }
        let found = tail - level_end;
        reach.reached += found;
        reach.distances += found as u64 * level;
        reach.inverse_distances += found as f64 / level as f64;
        level_start = level_end;
    }
    Ok(reach)
}

/// Exact closeness centrality of every node, by a BFS (or, weighted, Dijkstra) from every node.
///
/// Computed by BFSs from every node, 256 at a time (multi-source BFSs, sharing their work): about
/// `(96 + 3 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread.
///
/// On weighted graphs (see [`E::is_weighted`]), distances are weighted
/// (as networkx's with `distance="weight"`): by Dijkstra's algorithm from every node instead
/// (see [`DijkstraDists`](crate::trails::dijkstra::DijkstraDists)), with about
/// `(16 + 2 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread. Nodes at distance zero
/// (through zero-weight arcs) count as reached, but add nothing to harmonic sums.
///
/// `normalized` selects the value computed for each node `u`, from the sum `d(u)` of its
/// distances to the nodes it reaches and the number `r(u)` of nodes it reaches (`u` included):
/// * [`None`] --- `d(u)` itself.
/// * `Some(false)` --- `(|V| - 1) / d(u)`.
/// * `Some(true)` --- `(r(u) - 1) / d(u)`.
///
/// Nodes reaching no other node get 0.
/// Works on any graph, following arcs out of nodes ([`Direction::Out`], the default): a node's distances are to the nodes it reaches.
/// On a [`DiGraph`], [`new_directed`](Self::new_directed) follows them in another [`Direction`]:
/// [`Direction::In`] uses distances from the nodes that reach it (the convention of networkx, and of Boldi and Vigna's "Axioms for Centrality"). To ignore their directions, run it on the graph's
/// [`to_undirected`](../../graph/struct.GraphMemoryMap.html#method.to_undirected).
///
/// [`Direction`]: ../../graph/direction/enum.Direction.html
/// [`Direction::Out`]: ../../graph/direction/enum.Direction.html#variant.Out
/// [`Direction::In`]: ../../graph/direction/enum.Direction.html#variant.In
/// [`DiGraph`]: ../../graph/type.DiGraph.html
pub struct ExactClosenessCentrality<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    W: Weight = f64,
    Dir: Directedness = Undirected,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
    /// Normalization flag.
    normalized: Option<bool>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    ExactClosenessCentrality<'a, N, E, Ix, W, Directed>
{
    /// Computes every node's exact closeness centrality, following arcs in a given [`Direction`] (see the type's documentation):
    /// the same as [`new`](Self::new) on [`GraphMemoryMap::view`]`(direction)`, which the
    /// results are then of (its node ids are the graph's).
    ///
    /// [`GraphMemoryMap::view`]: ../../graph/struct.GraphMemoryMap.html#method.view
    pub fn new_directed(
        g: &'a DiGraph<N, E, Ix>,
        normalized: Option<bool>,
        direction: Direction,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(g.view(direction)?, normalized)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactClosenessCentrality<'a, N, E, Ix, W, Dir>
{
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut closeness = Self::new_no_compute(g, normalized)?;

        closeness.compute_with_proc_mem(())?;

        Ok(closeness)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactCloseness, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactClosenessCentrality<'a, N, E, Ix, W, Dir>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactCloseness, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let index = Self::index_by_normalization(self.normalized);
        let norm = W::from_usize(self.g.size().saturating_sub(1));
        let e = self.centralities.shared_slice();
        every_node(
            self.g,
            CacheFile::ExactCloseness,
            &|u, reached, distances, _| {
                let mut e = e;
                let total = W::from_f64(distances);
                *e.get_mut(u) = if reached <= 1 || distances == 0. {
                    W::zero()
                } else {
                    match index {
                        0 => total,
                        1 => norm / total,
                        // reachable nodes other than `u` itself
                        _ => W::from_usize(reached - 1) / total,
                    }
                };
            },
        )?;
        self.g.cleanup_cache(CacheFile::ExactCloseness)?;
        Ok(())
    }
}

/// Exact harmonic centrality of every node, by a BFS (or, weighted, Dijkstra) from every node.
///
/// Computed by BFSs from every node, 256 at a time (multi-source BFSs, sharing their work): about
/// `(96 + 3 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread.
///
/// On weighted graphs (see [`E::is_weighted`]), distances are weighted
/// (as networkx's with `distance="weight"`): by Dijkstra's algorithm from every node instead
/// (see [`DijkstraDists`](crate::trails::dijkstra::DijkstraDists)), with about
/// `(16 + 2 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread. Nodes at distance zero
/// (through zero-weight arcs) count as reached, but add nothing to harmonic sums.
///
/// `normalized` selects the value computed for each node `u`, from the sum `h(u)` of the inverses
/// of its distances to the nodes it reaches and the number `r(u)` of nodes it reaches (`u`
/// included):
/// * [`None`] --- `h(u)` itself.
/// * `Some(false)` --- `h(u) / (|V| - 1)`.
/// * `Some(true)` --- `h(u) / (r(u) - 1)`.
///
/// Nodes reaching no other node get 0.
/// Works on any graph, following arcs out of nodes ([`Direction::Out`], the default): a node's distances are to the nodes it reaches.
/// On a [`DiGraph`], [`new_directed`](Self::new_directed) follows them in another [`Direction`]:
/// [`Direction::In`] uses distances from the nodes that reach it (the convention of networkx, and of Boldi and Vigna's "Axioms for Centrality"). To ignore their directions, run it on the graph's
/// [`to_undirected`](../../graph/struct.GraphMemoryMap.html#method.to_undirected).
///
/// [`Direction`]: ../../graph/direction/enum.Direction.html
/// [`Direction::Out`]: ../../graph/direction/enum.Direction.html#variant.Out
/// [`Direction::In`]: ../../graph/direction/enum.Direction.html#variant.In
/// [`DiGraph`]: ../../graph/type.DiGraph.html
pub struct ExactHarmonicCentrality<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    W: Weight = f64,
    Dir: Directedness = Undirected,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
    /// Normalization flag.
    normalized: Option<bool>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    ExactHarmonicCentrality<'a, N, E, Ix, W, Directed>
{
    /// Computes every node's exact harmonic centrality, following arcs in a given [`Direction`] (see the type's documentation):
    /// the same as [`new`](Self::new) on [`GraphMemoryMap::view`]`(direction)`, which the
    /// results are then of (its node ids are the graph's).
    ///
    /// [`GraphMemoryMap::view`]: ../../graph/struct.GraphMemoryMap.html#method.view
    pub fn new_directed(
        g: &'a DiGraph<N, E, Ix>,
        normalized: Option<bool>,
        direction: Direction,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(g.view(direction)?, normalized)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactHarmonicCentrality<'a, N, E, Ix, W, Dir>
{
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut harmonic = Self::new_no_compute(g, normalized)?;

        harmonic.compute_with_proc_mem(())?;

        Ok(harmonic)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactHarmonic, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactHarmonicCentrality<'a, N, E, Ix, W, Dir>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactHarmonic, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let index = Self::index_by_normalization(self.normalized);
        let norm = W::from_usize(self.g.size().saturating_sub(1));
        let e = self.centralities.shared_slice();
        every_node(
            self.g,
            CacheFile::ExactHarmonic,
            &|u, reached, _, inverse| {
                let mut e = e;
                let inverse = W::from_f64(inverse);
                *e.get_mut(u) = if reached <= 1 {
                    W::zero()
                } else {
                    match index {
                        0 => inverse,
                        1 => inverse / norm,
                        // reachable nodes other than `u` itself
                        _ => inverse / W::from_usize(reached - 1),
                    }
                };
            },
        )?;
        self.g.cleanup_cache(CacheFile::ExactHarmonic)?;
        Ok(())
    }
}

/// Exact Lin's centrality of every node, by a BFS (or, weighted, Dijkstra) from every node: `r(u)^2 / d(u)`, where `r(u)` is the
/// number of nodes `u` reaches (`u` included) and `d(u)` the sum of its distances to them, or 1
/// if `u` reaches no other node.
///
/// Computed by BFSs from every node, 256 at a time (multi-source BFSs, sharing their work): about
/// `(96 + 3 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread.
///
/// On weighted graphs (see [`E::is_weighted`]), distances are weighted
/// (as networkx's with `distance="weight"`): by Dijkstra's algorithm from every node instead
/// (see [`DijkstraDists`](crate::trails::dijkstra::DijkstraDists)), with about
/// `(16 + 2 * size_of::<Ix>()) * |V|` bytes of scratch memory per thread. Nodes at distance zero
/// (through zero-weight arcs) count as reached, but add nothing to harmonic sums.
///
/// Works on any graph, following arcs out of nodes ([`Direction::Out`], the default): a node's distances are to the nodes it reaches.
/// On a [`DiGraph`], [`new_directed`](Self::new_directed) follows them in another [`Direction`]:
/// [`Direction::In`] uses distances from the nodes that reach it (the convention of networkx, and of Boldi and Vigna's "Axioms for Centrality"). To ignore their directions, run it on the graph's
/// [`to_undirected`](../../graph/struct.GraphMemoryMap.html#method.to_undirected).
///
/// [`Direction`]: ../../graph/direction/enum.Direction.html
/// [`Direction::Out`]: ../../graph/direction/enum.Direction.html#variant.Out
/// [`Direction::In`]: ../../graph/direction/enum.Direction.html#variant.In
/// [`DiGraph`]: ../../graph/type.DiGraph.html
pub struct ExactLinCentrality<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    W: Weight = f64,
    Dir: Directedness = Undirected,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    ExactLinCentrality<'a, N, E, Ix, W, Directed>
{
    /// Computes every node's exact Lin's centrality, following arcs in a given [`Direction`] (see the type's documentation):
    /// the same as [`new`](Self::new) on [`GraphMemoryMap::view`]`(direction)`, which the
    /// results are then of (its node ids are the graph's).
    ///
    /// [`GraphMemoryMap::view`]: ../../graph/struct.GraphMemoryMap.html#method.view
    pub fn new_directed(
        g: &'a DiGraph<N, E, Ix>,
        direction: Direction,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(g.view(direction)?)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactLinCentrality<'a, N, E, Ix, W, Dir>
{
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix, Dir>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut lin = Self::new_no_compute(g)?;

        lin.compute_with_proc_mem(())?;

        Ok(lin)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactLin, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight, Dir: Directedness>
    ExactLinCentrality<'a, N, E, Ix, W, Dir>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactLin, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let e = self.centralities.shared_slice();
        every_node(self.g, CacheFile::ExactLin, &|u, reached, distances, _| {
            let mut e = e;
            // as defined by Boldi P. and Vigna S., a node reaching no other node has Lin's
            // centrality 1, and reachable nodes include the node itself
            *e.get_mut(u) = if distances == 0. {
                W::one()
            } else {
                let r = W::from_usize(reached);
                (r * r) / W::from_f64(distances)
            };
        })?;
        self.g.cleanup_cache(CacheFile::ExactLin)?;
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------
// Betweenness
// ------------------------------------------------------------------------------------------------

/// Every node's exact betweenness centrality: how many of the shortest paths between other nodes
/// run through it, as networkx's `betweenness_centrality` (`endpoints=False`), weighted on weighted
/// graphs (whose weights must be positive and finite). Normalized by the `(|V| - 1) (|V| - 2)`
/// ordered pairs of other nodes, or not (the number of shortest paths through a node, each
/// unordered pair of an undirected graph counted once).
///
/// Brandes' algorithm ("A faster algorithm for betweenness centrality", 2001): a search from
/// every node (BFS, or Dijkstra's algorithm on weighted graphs), whose shortest paths' dependencies
/// are then accumulated backward; sources are shared out among the graph's threads, each with its
/// own memory (mapped in the cache). On undirected graphs, it runs on their biconnected
/// components (as Puzis et al.'s "Heuristics for speeding up betweenness centrality computation",
/// 2012, and Sariyüce et al.'s BADIOS, 2013): a shortest path between two nodes of a component
/// never leaves it, so Brandes' algorithm runs within each component only, from each of its nodes,
/// sources and targets weighted by the nodes they stand for (those hanging off them outside the
/// component), and the paths through an articulation point between the parts its removal
/// separates are counted at once. Components of two nodes (bridges, and so the nodes of trees
/// hanging off the graph) cost nothing; the components are found by [`Biconnectivity`], in
/// parallel, and each is searched on a compact copy of its arcs.
///
/// Its results are the same every run on as many threads, and may differ in their last bits with
/// the number of threads (each thread's sources' sums are added separately). Each thread maps
/// memory for its searches and sums (about 40 bytes per node of the largest component searched,
/// and 8 per node); threads are capped to what half the available memory holds. For estimates on graphs too large for `O(|V| |E|)`, see
/// [`Betweenness`](super::betweenness::Betweenness)'s KADABRA engine.
///
/// [`Biconnectivity`]: ../../components/biconnected/struct.Biconnectivity.html
pub struct ExactBetweennessCentrality<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness = Undirected,
> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    centralities: AbstractedProceduralMemoryMut<f64>,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>
    ExactBetweennessCentrality<'a, N, E, Ix, Dir>
{
    /// Computes every node's betweenness, `normalized` or not (see the type's documentation).
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        normalized: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = g.size();
        let out_fn = g.build_cache_filename(CacheFile::ExactBetweenness, None)?;
        let mut centralities = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, n.max(1), true)?;
        let out = &mut centralities.as_mut_slice()[..n];
        let computed = check_positive_weights(g)
            .and_then(|()| betweenness_sums(g, out, g.thread_num().max(1)));
        if let Err(e) = computed {
            // no (partial) result left behind
            drop(centralities);
            let _ = crate::shared_slice::remove_if_exists(&out_fn);
            return Err(e);
        }
        let scale = betweenness_scale(n, normalized, Dir::DIRECTED);
        for c in out.iter_mut() {
            *c *= scale;
        }
        centralities.flush()?;
        Ok(Self { g, centralities })
    }

    /// A node's betweenness.
    pub fn get_node_centrality(&self, idx: usize) -> f64 {
        self.centralities()[idx]
    }

    /// Every node's betweenness.
    pub fn centralities(&self) -> &[f64] {
        &self.centralities.as_slice()[..self.g.size()]
    }

    /// Removes the centralities from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::ExactBetweenness, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

/// What turns a node's sum over ordered pairs into its betweenness: normalized by the
/// `(n - 1) (n - 2)` ordered pairs of other nodes, or not (each unordered pair counted once, on
/// undirected graphs).
pub(crate) fn betweenness_scale(n: usize, normalized: bool, directed: bool) -> f64 {
    if normalized {
        if n > 2 {
            1. / ((n - 1) as f64 * (n - 2) as f64)
        } else {
            0.
        }
    } else if directed {
        1.
    } else {
        0.5
    }
}

/// Checks a weighted graph's weights positive (and finite).
pub(crate) fn check_positive_weights<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Result<(), Box<dyn std::error::Error>> {
    if E::is_weighted() {
        // positive weights: a zero-weight arc would make a path and its extension equally short,
        // and zero-weight cycles infinitely many shortest paths
        g.total_weight()?;
        if let Some(w) = g
            .iter_edge_labels()
            .map(|label| label.weight::<f64>())
            .find(|&w| w <= 0.)
        {
            return Err(format!("error betweenness needs positive weights, not {w}").into());
        }
    }
    Ok(())
}

/// Every node's betweenness as a sum over ordered pairs of other nodes, into `out`, on `threads`
/// threads: by biconnected components on undirected graphs (unless the graph's [`Biconnectivity`]
/// is cached, which isn't overwritten), Brandes' algorithm on the whole graph otherwise. The
/// weights must have been checked positive ([`check_positive_weights`]).
///
/// [`Biconnectivity`]: ../../components/biconnected/struct.Biconnectivity.html
pub(crate) fn betweenness_sums<
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    out: &mut [f64],
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    out.fill(0.);
    if g.size() < 3 {
        return Ok(());
    }
    let result = (|| {
        if let Some(u) = (g as &dyn std::any::Any).downcast_ref::<Graph<N, E, Ix>>() {
            let bcc = u.build_pers_cache_filename(CacheFile::Biconnectivity, Some(0))?;
            if !std::path::Path::new(&bcc).exists() {
                return betweenness_by_blocks(u, out, threads);
            }
        }
        let csr = Csr::of(g);
        brandes_blocks(g, &csr, &[0, g.size()], None, out, threads, 0)
    })();
    g.cleanup_cache(CacheFile::ExactBetweenness)?;
    result
}

/// The arcs Brandes' algorithm searches: a graph's, or a biconnected component's.
trait Arcs: Sync {
    /// Calls `f(v, weight)` for each arc out of `u`, `v` relative to `u`'s block.
    fn arcs(&self, u: usize, f: impl FnMut(usize, f64));
}

impl<Ix: graph::IndexType, E: graph::E> Arcs for Csr<'_, Ix, E> {
    #[inline(always)]
    fn arcs(&self, u: usize, mut f: impl FnMut(usize, f64)) {
        for e in self.range(u) {
            let w = if E::is_weighted() {
                self.weights[e].weight::<f64>()
            } else {
                1.
            };
            f(self.targets[e].index(), w);
        }
    }
}

/// Biconnected components' arcs, each component's nodes numbered from 0 (its head) in a block of
/// their own.
struct BlockArcs<'a> {
    offsets: &'a [usize],
    /// (`u32`s, as most graphs' own: half the bytes of `usize`s to stream through, per search)
    targets: &'a [u32],
    weights: Option<&'a [f64]>,
}

impl Arcs for BlockArcs<'_> {
    #[inline(always)]
    fn arcs(&self, u: usize, mut f: impl FnMut(usize, f64)) {
        for e in self.offsets[u]..self.offsets[u + 1] {
            f(self.targets[e] as usize, self.weights.map_or(1., |w| w[e]));
        }
    }
}

/// A node's state in Brandes' algorithm: its distance, shortest paths and dependency, together
/// (in a 32-byte aligned record, never straddling cache lines), as they're read together: a
/// neighbour's visit costs one cache miss, not three.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Slot {
    dist: f64,
    sigma: f64,
    dep: f64,
    /// Its weight as a target (`reach`'s, or 1), set when it's reached.
    weight: f64,
}

const UNREACHED: Slot = Slot {
    dist: f64::INFINITY,
    sigma: 0.,
    dep: 0.,
    weight: 0.,
};

/// A thread's memory for Brandes' algorithm, for its largest block's nodes.
struct BrandesScratch<'s> {
    slots: &'s mut [Slot],
    /// (`u32`s: nodes are numbered within blocks of fewer than 2^32)
    order: &'s mut [u32],
    heap:
        std::collections::BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f64>, usize)>>,
}

/// Brandes' algorithm from `s` (relative to the block starting at `base`): adds each node's
/// dependency on `s`, weighted by `s`'s weight and its targets' (`reach`, or 1), to `sums`.
fn brandes_source<A: Arcs>(
    arcs: &A,
    weighted: bool,
    base: usize,
    s: usize,
    m: &mut BrandesScratch,
    reach: Option<&[f64]>,
    sums: &mut [f64],
) {
    let (slots, order) = (&mut *m.slots, &mut *m.order);
    let weight_of = |v: usize| reach.map_or(1., |r| r[base + v]);
    slots[s] = Slot {
        dist: 0.,
        sigma: 1.,
        dep: 0.,
        weight: weight_of(s),
    };
    order[0] = s as u32;
    let reached = if weighted {
        use ordered_float::OrderedFloat;
        use std::cmp::Reverse;
        let heap = &mut m.heap;
        heap.clear();
        heap.push(Reverse((OrderedFloat(0.), s)));
        let mut settled = 0;
        while let Some(Reverse((OrderedFloat(d), u))) = heap.pop() {
            // nodes are pushed again only closer: a later entry of a settled node is farther
            if d > slots[u].dist {
                continue;
            }
            order[settled] = u as u32;
            settled += 1;
            let su = slots[u].sigma;
            arcs.arcs(base + u, |v, w| {
                let dv = d + w;
                let sv = &mut slots[v];
                if dv < sv.dist {
                    if sv.dist == f64::INFINITY {
                        sv.weight = weight_of(v);
                    }
                    sv.dist = dv;
                    sv.sigma = su;
                    heap.push(Reverse((OrderedFloat(dv), v)));
                } else if dv == sv.dist {
                    sv.sigma += su;
                }
            });
        }
        settled
    } else {
        let (mut head, mut tail) = (0, 1);
        while head < tail {
            let u = order[head] as usize;
            head += 1;
            let (du, su) = (slots[u].dist + 1., slots[u].sigma);
            arcs.arcs(base + u, |v, _| {
                let sv = &mut slots[v];
                if sv.dist == f64::INFINITY {
                    sv.dist = du;
                    sv.weight = weight_of(v);
                    order[tail] = v as u32;
                    tail += 1;
                }
                if sv.dist == du {
                    sv.sigma += su;
                }
            });
        }
        tail
    };
    // dependencies, farthest nodes first: each node's on its successors'
    let source = slots[s].weight;
    for w in order[..reached].iter().rev().map(|&w| w as usize) {
        let Slot {
            dist: dw,
            sigma: sw,
            ..
        } = slots[w];
        let mut d = 0.;
        arcs.arcs(base + w, |v, weight| {
            let sv = &slots[v];
            if sv.dist == dw + weight {
                d += sw / sv.sigma * (sv.weight + sv.dep);
            }
        });
        slots[w].dep = d;
        if w != s {
            sums[base + w] += source * d;
        }
    }
    for &w in &order[..reached] {
        slots[w as usize] = UNREACHED;
    }
}

/// Brandes' algorithm from every node of every block of `arcs` (block `b`'s nodes are
/// `starts[b]..starts[b + 1]`), on `threads` threads: each node's dependencies, weighted by
/// `reach` (or 1), summed into `out` (one per node of every block). Scratch files are
/// [`CacheFile::ExactBetweenness`]'s, from `first_file` on.
fn brandes_blocks<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness, A: Arcs>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    arcs: &A,
    starts: &[usize],
    reach: Option<&[f64]>,
    out: &mut [f64],
    threads: usize,
    first_file: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let total = *starts.last().unwrap_or(&0);
    let largest = starts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    if total == 0 {
        return Ok(());
    }
    if largest > u32::MAX as usize {
        return Err(format!(
            "error exact betweenness numbers a search's nodes as u32s: {largest} are too many"
        )
        .into());
    }
    // sources a few at a time, each thread's chunks fixed (round robin, in order), so that its
    // sums, and so the results, are the same every run on as many threads
    const CHUNK: usize = 16;
    // each thread's memory: its search's nodes' states and order (`largest` of each), and its
    // sums (`total`); as many threads as half the available memory holds, at most
    let per_thread = (std::mem::size_of::<Slot>() + 4) * largest + 8 * total;
    let affordable = sys_info::mem_info()
        .map(|m| (m.avail as usize * 1024 / 2 / per_thread.max(1)).max(1))
        .unwrap_or(usize::MAX);
    let threads = threads.min(affordable).min(total.div_ceil(CHUNK)).max(1);
    let chunks = total.div_ceil(CHUNK);
    // each thread's nodes' states, visit order and sums
    const FILES: usize = 3;
    let files: Vec<String> = (0..FILES * threads)
        .map(|k| g.build_cache_filename(CacheFile::ExactBetweenness, Some(first_file + k)))
        .collect::<Result<_, _>>()?;
    let sums: Vec<AbstractedProceduralMemoryMut<f64>> = (0..threads)
        .map(|tid| SharedSliceMut::<f64>::abst_mem_mut(&files[FILES * tid + 2], total, true))
        .collect::<Result<_, _>>()?;
    let sum_views: Vec<SharedSliceMut<f64>> = sums.iter().map(|s| s.shared_slice()).collect();
    let weighted = E::is_weighted();
    super::betweenness::on_threads(threads, "Brandes", |tid| {
        let f = &files[FILES * tid..FILES * (tid + 1)];
        let memory = (|| -> Result<_, Box<dyn std::error::Error>> {
            Ok((
                SharedSliceMut::<Slot>::abst_mem_mut(&f[0], largest, true)?,
                SharedSliceMut::<u32>::abst_mem_mut(&f[1], largest, true)?,
            ))
        })()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
        let (mut slots, mut order) = memory;
        let mut m = BrandesScratch {
            slots: slots.as_mut_slice(),
            order: order.as_mut_slice(),
            heap: std::collections::BinaryHeap::new(),
        };
        assert_eq!(m.slots.as_ptr() as usize % std::mem::align_of::<Slot>(), 0);
        m.slots.fill(UNREACHED);
        let mut sum = sum_views[tid];
        let sum = sum.as_mut_slice();
        sum.fill(0.);
        for chunk in (tid..chunks).step_by(threads) {
            let first = CHUNK * chunk;
            for l in first..(first + CHUNK).min(total) {
                let base = starts[starts.partition_point(|&x| x <= l) - 1];
                brandes_source(arcs, weighted, base, l - base, &mut m, reach, sum);
            }
        }
        Ok(())
    })?;
    // the threads' sums, in thread order
    let out_view = SharedSliceMut::new(out.as_mut_ptr(), out.len());
    const PART: usize = 1 << 16;
    let parts = std::sync::atomic::AtomicUsize::new(0);
    super::betweenness::on_threads(threads, "Brandes sum", |_| {
        let mut out = out_view;
        loop {
            let begin = PART * parts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if begin >= total {
                return Ok(());
            }
            let end = (begin + PART).min(total);
            for (i, o) in out.as_mut_slice()[begin..end].iter_mut().enumerate() {
                *o = sum_views.iter().map(|s| s.as_slice()[begin + i]).sum();
            }
        }
    })?;
    drop(sums);
    g.remove_tmp_files(
        CacheFile::ExactBetweenness,
        first_file..first_file + files.len(),
    )?;
    Ok(())
}

/// [`betweenness_sums`] by biconnected components (see [`ExactBetweennessCentrality`]), unless one
/// holds 95% of the nodes or more: then on the whole graph, as searching it alone would save
/// little (the nodes outside it), and copying its arcs would cost more.
///
/// [`Biconnectivity`]'s components form a forest: a component's head is listed in its parent
/// component (or, a root, in none), every other node in exactly one. `below[b]` counts the nodes
/// under component `b` (its nodes but its head, and those under the components they head), so
/// that, within `b`, its head stands for the `component size - below[b]` nodes it doesn't lead to
/// through `b`, and any other node `v` for itself and the `hanging[v]` nodes under the components
/// it heads. An articulation point `v` separates the other nodes of its connected component into
/// parts of `below[c]` nodes (for each `c` it heads) and, if it's listed in a component, the rest:
/// it's on every path between two parts, `(n_c - 1)^2 - Σ part^2` ordered pairs.
///
/// [`Biconnectivity`]: ../../components/biconnected/struct.Biconnectivity.html
fn betweenness_by_blocks<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &Graph<N, E, Ix>,
    out: &mut [f64],
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bcc = crate::components::biconnected::Biconnectivity::new(g)?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let n = g.size();
        let (labels, heads, k) = (bcc.components(), bcc.heads(), bcc.count());
        let none = <Ix as graph::IndexType>::max();
        let block = |v: usize| (labels[v] != none).then(|| labels[v].index());
        let parent = |b: usize| block(heads[b].index());

        // each component's listed nodes, depth (from its forest's root) and nodes below
        let mut listed = vec![0usize; k];
        for v in 0..n {
            if let Some(b) = block(v) {
                listed[b] += 1;
            }
        }
        // nearly all one component: little to gain, a copy of its arcs to pay for
        if 20 * (listed.iter().copied().max().unwrap_or(0) + 1) >= 19 * n {
            return brandes_blocks(g, &Csr::of(g), &[0, n], None, out, threads, 0);
        }
        let mut depth = vec![usize::MAX; k];
        let mut path = vec![];
        for b in 0..k {
            path.clear();
            let mut c = b;
            let mut d = loop {
                if depth[c] != usize::MAX {
                    break depth[c] + 1;
                }
                path.push(c);
                match parent(c) {
                    Some(p) => c = p,
                    None => break 0,
                }
            };
            for &c in path.iter().rev() {
                depth[c] = d;
                d += 1;
            }
        }
        let mut by_depth: Vec<usize> = (0..k).collect();
        by_depth.sort_unstable_by_key(|&b| std::cmp::Reverse(depth[b]));
        let mut below = listed.clone();
        for &b in &by_depth {
            if let Some(p) = parent(b) {
                below[p] += below[b];
            }
        }
        let (mut hanging, mut squares) = (vec![0usize; n], vec![0f64; n]);
        for b in 0..k {
            let h = heads[b].index();
            hanging[h] += below[b];
            squares[h] += below[b] as f64 * below[b] as f64;
        }
        // each component's connected component's size, roots first
        let mut size = vec![0usize; k];
        for &b in by_depth.iter().rev() {
            size[b] = match parent(b) {
                Some(p) => size[p],
                None => 1 + hanging[heads[b].index()],
            };
        }

        // the pairs articulation points separate
        for v in 0..n {
            if hanging[v] == 0 {
                continue;
            }
            let (nc, rest) = match block(v) {
                Some(p) => (size[p], size[p] - 1 - hanging[v]),
                None => (1 + hanging[v], 0),
            };
            let others = (nc - 1) as f64;
            out[v] += others * others - squares[v] - rest as f64 * rest as f64;
        }

        // components of three nodes or more (in pairs, no node is inner to a path), each a block
        // of local ids, its head first
        let mut starts = vec![0usize];
        let mut local_block = vec![usize::MAX; k];
        let mut blocks = vec![];
        for b in 0..k {
            if listed[b] >= 2 {
                local_block[b] = blocks.len();
                blocks.push(b);
                starts.push(starts.last().unwrap() + listed[b] + 1);
            }
        }
        let total = *starts.last().unwrap();
        if total == 0 {
            return Ok(());
        }
        let largest = starts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        if largest > u32::MAX as usize {
            // (local ids are `u32`s) the whole graph instead, the separated pairs not counted twice
            out.fill(0.);
            return brandes_blocks(g, &Csr::of(g), &[0, n], None, out, threads, 0);
        }
        let file = |seq: usize| g.build_cache_filename(CacheFile::ExactBetweenness, Some(seq));
        let mut members_mem = SharedSliceMut::<usize>::abst_mem_mut(&file(0)?, total, true)?;
        let mut reach_mem = SharedSliceMut::<f64>::abst_mem_mut(&file(1)?, total, true)?;
        let mut position_mem = SharedSliceMut::<usize>::abst_mem_mut(&file(2)?, n, true)?;
        let (members, reach, position) = (
            members_mem.as_mut_slice(),
            reach_mem.as_mut_slice(),
            position_mem.as_mut_slice(),
        );
        let mut next: Vec<usize> = vec![1; blocks.len()];
        for (lb, &b) in blocks.iter().enumerate() {
            members[starts[lb]] = heads[b].index();
            reach[starts[lb]] = (size[b] - below[b]) as f64;
        }
        for v in 0..n {
            if let Some(b) = block(v)
                && local_block[b] != usize::MAX
            {
                let lb = local_block[b];
                position[v] = next[lb];
                members[starts[lb] + next[lb]] = v;
                reach[starts[lb] + next[lb]] = (1 + hanging[v]) as f64;
                next[lb] += 1;
            }
        }

        // each block's arcs, between its nodes (an arc between two nodes of a component is its:
        // two components share at most a node)
        let csr = Csr::of(g);
        let local = |lb: usize, v: usize, w: usize| -> Option<usize> {
            let b = blocks[lb];
            if w == v {
                None
            } else if w == heads[b].index() {
                Some(0)
            } else if block(w) == Some(b) {
                Some(position[w])
            } else {
                None
            }
        };
        let mut offsets_mem = SharedSliceMut::<usize>::abst_mem_mut(&file(3)?, total + 1, true)?;
        let offsets = offsets_mem.as_mut_slice();
        offsets[0] = 0;
        for lb in 0..blocks.len() {
            for l in starts[lb]..starts[lb + 1] {
                let v = members[l];
                let degree = csr
                    .range(v)
                    .filter(|&e| local(lb, v, csr.targets[e].index()).is_some())
                    .count();
                offsets[l + 1] = offsets[l] + degree;
            }
        }
        let arcs = offsets[total];
        let mut targets_mem = SharedSliceMut::<u32>::abst_mem_mut(&file(4)?, arcs.max(1), true)?;
        let mut weights_mem = SharedSliceMut::<f64>::abst_mem_mut(
            &file(5)?,
            if E::is_weighted() { arcs.max(1) } else { 1 },
            true,
        )?;
        let (targets, weights) = (targets_mem.as_mut_slice(), weights_mem.as_mut_slice());
        for lb in 0..blocks.len() {
            for l in starts[lb]..starts[lb + 1] {
                let v = members[l];
                let mut at = offsets[l];
                for e in csr.range(v) {
                    if let Some(w) = local(lb, v, csr.targets[e].index()) {
                        targets[at] = w as u32;
                        if E::is_weighted() {
                            weights[at] = csr.weights[e].weight::<f64>();
                        }
                        at += 1;
                    }
                }
            }
        }
        let block_arcs = BlockArcs {
            offsets,
            targets,
            weights: E::is_weighted().then_some(&*weights),
        };
        let mut sums_mem = SharedSliceMut::<f64>::abst_mem_mut(&file(6)?, total, true)?;
        let sums = sums_mem.as_mut_slice();
        brandes_blocks(g, &block_arcs, &starts, Some(reach), sums, threads, 8)?;
        for (l, &s) in sums.iter().enumerate() {
            out[members[l]] += s;
        }
        Ok(())
    })();
    bcc.drop_cache()?;
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use std::io::Write;

    /// Every node's [`Reach`], by [`reach_of_every_node`] (multi-source BFSs).
    fn batched<Dir: Directedness>(g: &GraphMemoryMap<(), (), usize, Dir>) -> Vec<Reach> {
        let reach = std::sync::Mutex::new(vec![Reach::default(); g.size()]);
        reach_of_every_node(g, CacheFile::ExactCloseness, &|u, r| {
            reach.lock().unwrap()[u] = r
        })
        .unwrap();
        g.cleanup_cache(CacheFile::ExactCloseness).unwrap();
        reach.into_inner().unwrap()
    }

    /// Every node's [`Reach`], by one BFS per node.
    fn one_by_one<Dir: Directedness>(g: &GraphMemoryMap<(), (), usize, Dir>) -> Vec<Reach> {
        let n = g.size();
        let mut stamp = vec![0u32; n];
        let mut queue = vec![0usize; n];
        let mut stamp = SharedSliceMut::<u32>::new(stamp.as_mut_ptr(), n);
        let mut queue = SharedSliceMut::<usize>::new(queue.as_mut_ptr(), n);
        (0..n)
            .map(|u| single_source_reach(g, u, &mut stamp, &mut queue).unwrap())
            .collect()
    }

    /// The multi-source BFSs find what BFSs from each node do: the same reached nodes and
    /// distances (and, summing inverse distances in the same order, the same sums, bit for bit),
    /// those of [`BFSDists`] too, on graphs whose node counts aren't multiples of the batches,
    /// with self-loops, parallel arcs, isolated nodes, several components, and more threads than
    /// nodes.
    ///
    /// [`BFSDists`]: ../../trails/bfs/struct.BFSDists.html
    #[test]
    fn multi_source_bfs_finds_what_single_source_ones_do() {
        use crate::components::test_utils::{graph_with_threads, random_arcs};
        use crate::trails::bfs::BFSDists;
        let path: Vec<_> = (0..99).map(|u| (u, u + 1)).collect();
        let mut two_parts = random_arcs(70, 200, 3);
        two_parts.extend(
            random_arcs(70, 200, 4)
                .into_iter()
                .map(|(u, v)| (u + 70, v + 70)),
        );
        for (name, header, n, arcs, threads) in [
            ("single", "general", 1, vec![], 4),
            ("isolated", "general", 5, vec![(0, 0)], 8),
            ("path", "general", 100, path.clone(), 3),
            ("path_symmetric", "symmetric", 100, path, 2),
            (
                "random_batch_less_1",
                "general",
                BATCH - 1,
                random_arcs(BATCH - 1, 5 * BATCH, 1),
                1,
            ),
            (
                "random_batch",
                "general",
                BATCH,
                random_arcs(BATCH, 5 * BATCH, 2),
                1,
            ),
            (
                "random_batch_plus_1",
                "symmetric",
                BATCH + 1,
                random_arcs(BATCH + 1, 2 * BATCH, 5),
                2,
            ),
            (
                "random_batches",
                "general",
                3 * BATCH + 7,
                random_arcs(3 * BATCH + 7, 14 * BATCH, 6),
                3,
            ),
            ("two_parts", "general", 140, two_parts, 5),
            (
                "sparse",
                "general",
                2 * BATCH + 50,
                random_arcs(2 * BATCH + 50, 2 * BATCH, 7),
                32,
            ),
        ] {
            let g = graph_with_threads(name, header, n, &arcs, threads);
            let (got, want) = (batched(&g), one_by_one(&g));
            assert_eq!(got, want, "{name}");
            for (u, got) in got.iter().enumerate() {
                let bfs = BFSDists::new(&g, u).unwrap();
                assert_eq!(got.reached, bfs.reachable(), "{name}: node {u} reaches");
                assert_eq!(
                    got.distances as f64,
                    bfs.total_distances(),
                    "{name}: node {u}'s distances"
                );
                // its distances' inverses, in another order
                let inverse: f64 = bfs
                    .get_distances()
                    .iter()
                    .filter(|&&d| d > 0)
                    .map(|&d| 1. / d as f64)
                    .sum();
                assert!(
                    (got.inverse_distances - inverse).abs() <= 1e-12 * inverse.max(1.),
                    "{name}: node {u}"
                );
            }
        }
    }

    /// A small 4-clique: a connected graph with a trivially known structure (every node reaches
    /// every other node in exactly one hop), used to sanity-check the exact centralities'
    /// accumulation across `Ix` widths and `W` (weight) precisions.
    fn write_clique_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating clique mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "4 4 6").unwrap();
        for &(u, v) in &[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
            writeln!(f, "{u} {v}").unwrap();
        }
        f.flush().unwrap();
    }

    fn open_graph<Ix: graph::IndexType>(
        path: &std::path::Path,
        id: &str,
    ) -> GraphMemoryMap<VoidLabel, VoidLabel, Ix> {
        GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::from_file(path, Some(id.to_string()), None)
            .expect("graph parse should succeed")
    }

    /// Runs all three exact centralities across several `Ix` widths (`W` fixed to `f64`) on a
    /// small clique, asserting every width produces the exact same (bit-for-bit identical)
    /// values --- `Ix`'s width only affects indexing/storage, never the floating-point
    /// computation itself.
    #[test]
    fn exact_centralities_cross_ix_widths_agree() {
        let dir = unique_test_dir("exact_cross_ix_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        macro_rules! run_all {
            ($ixty:ty, $tag:expr) => {{
                let graph = open_graph::<$ixty>(&path, &format!("exact_{}_{}", $tag, run_tag));
                let closeness = ExactClosenessCentrality::<_, _, _, f64>::new(&graph, Some(true))
                    .unwrap_or_else(|e| panic!("closeness({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                let harmonic = ExactHarmonicCentrality::<_, _, _, f64>::new(&graph, Some(true))
                    .unwrap_or_else(|e| panic!("harmonic({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                let lin = ExactLinCentrality::<_, _, _, f64>::new(&graph)
                    .unwrap_or_else(|e| panic!("lin({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                (closeness, harmonic, lin)
            }};
        }

        let (c_usize, h_usize, l_usize) = run_all!(usize, "usize");
        let (c_u32, h_u32, l_u32) = run_all!(u32, "u32");
        let (c_u16, h_u16, l_u16) = run_all!(u16, "u16");

        assert_eq!(
            c_usize, c_u32,
            "closeness: Ix=u32 should match Ix=usize exactly"
        );
        assert_eq!(
            c_usize, c_u16,
            "closeness: Ix=u16 should match Ix=usize exactly"
        );
        assert_eq!(
            h_usize, h_u32,
            "harmonic: Ix=u32 should match Ix=usize exactly"
        );
        assert_eq!(
            h_usize, h_u16,
            "harmonic: Ix=u16 should match Ix=usize exactly"
        );
        assert_eq!(l_usize, l_u32, "lin: Ix=u32 should match Ix=usize exactly");
        assert_eq!(l_usize, l_u16, "lin: Ix=u16 should match Ix=usize exactly");

        // sanity: every value should be positive (exact values are checked in
        // `exact_centralities_match_known_values`).
        for (v, &c) in c_usize.iter().enumerate() {
            assert!(
                c.is_finite() && c > 0.0,
                "node {v}: closeness should be positive, got {c}"
            );
        }
    }

    /// Runs the exact centralities with `W=f32` vs `W=f64` on the same clique, asserting the
    /// results agree closely (not necessarily bit-for-bit, but no catastrophic-cancellation risk
    /// exists here --- see `HyperBallInner`'s analogous test/docs --- so small graphs in
    /// particular should match almost exactly).
    #[test]
    fn exact_centralities_f32_and_f64_agree_closely() {
        let dir = unique_test_dir("exact_cross_weight_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        let graph_f64 = open_graph::<usize>(&path, &format!("exact_f64_{run_tag}"));
        let closeness_f64 = ExactClosenessCentrality::<_, _, _, f64>::new(&graph_f64, Some(true))
            .expect("closeness(f64) should succeed")
            .centralities()
            .to_vec();

        let graph_f32 = open_graph::<usize>(&path, &format!("exact_f32_{run_tag}"));
        let closeness_f32 = ExactClosenessCentrality::<_, _, _, f32>::new(&graph_f32, Some(true))
            .expect("closeness(f32) should succeed")
            .centralities()
            .to_vec();

        assert_eq!(closeness_f64.len(), closeness_f32.len());
        for (v, (&c64, &c32)) in closeness_f64.iter().zip(closeness_f32.iter()).enumerate() {
            assert!(
                (c64 - c32 as f64).abs() < 1e-2,
                "node {v}: f64 closeness {c64} and f32 closeness {c32} should be close"
            );
        }
    }

    /// Path `0 - 1 - 2` plus isolated node `3` (`|V| = 4`, so normalizing by `|V| - 1` and by
    /// reachable nodes differ), with hand-computed values:
    /// * distance sums `d = [3, 2, 3, 0]`, inverse distance sums `h = [1.5, 2, 1.5, 0]`,
    ///   reachable nodes (themselves included) `r = [3, 3, 3, 1]`.
    #[test]
    fn exact_centralities_match_known_values() {
        let dir = unique_test_dir("exact_known_values_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("path_and_isolated.mtx");
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "4 4 2").unwrap();
        writeln!(f, "1 2").unwrap();
        writeln!(f, "2 3").unwrap();
        drop(f);
        let graph = open_graph::<usize>(&path, &format!("exact_known_{run_tag}"));
        assert_eq!(graph.size(), 4);

        let check = |what: &str, got: &[f64], expected: [f64; 4]| {
            for (v, (&g, e)) in got.iter().zip(expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "{what} of node {v}: {g}, expected {e}"
                );
            }
        };
        let closeness = |n| {
            ExactClosenessCentrality::<_, _, _, f64>::new(&graph, n)
                .unwrap()
                .centralities()
                .to_vec()
        };
        let harmonic = |n| {
            ExactHarmonicCentrality::<_, _, _, f64>::new(&graph, n)
                .unwrap()
                .centralities()
                .to_vec()
        };

        check("distance sum", &closeness(None), [3., 2., 3., 0.]);
        check(
            "closeness / (|V| - 1)",
            &closeness(Some(false)),
            [1., 1.5, 1., 0.],
        );
        check(
            "closeness / (r - 1)",
            &closeness(Some(true)),
            [2. / 3., 1., 2. / 3., 0.],
        );
        check("inverse distance sum", &harmonic(None), [1.5, 2., 1.5, 0.]);
        check(
            "harmonic / (|V| - 1)",
            &harmonic(Some(false)),
            [0.5, 2. / 3., 0.5, 0.],
        );
        check(
            "harmonic / (r - 1)",
            &harmonic(Some(true)),
            [0.75, 1., 0.75, 0.],
        );
        check(
            "lin",
            ExactLinCentrality::<_, _, _, f64>::new(&graph)
                .unwrap()
                .centralities(),
            [3., 4.5, 3., 1.],
        );
    }
}
