//! Point-to-point shortest paths by A* (Hart P. E., Nilsson N. J. and Raphael B., ["A formal
//! basis for the heuristic determination of minimum cost paths"](https://doi.org/10.1109/TSSC.1968.300136)):
//! [`astar`] for one query and [`astar_many`] for many, in parallel, guided by any
//! [`Heuristic`] --- one of your own, or [`Landmarks`] (ALT: A*, landmarks and the triangle
//! inequality, Goldberg A. V. and Harrelson C., ["Computing the shortest path: A* search meets
//! graph theory"](https://dl.acm.org/doi/10.5555/1070432.1070455)), which needs no coordinates.
//!
//! Arcs are as long as their weights, which must be finite and non-negative, as for
//! [`DijkstraDists`](super::dijkstra::DijkstraDists).
//!
//! A query's memory grows with the nodes it explores (hash maps and a heap), not with the
//! graph: a well-guided query on a graph of hundreds of millions of nodes touches a sliver of
//! it, and many run at once. For every node's distance from a source, use `DijkstraDists`.

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use super::dijkstra::{ShortestPathEngine, distances_into, resolve_engine};
use super::shortest_path_visit::{NoOpShortestPathVisitor, ShortestPathVisitor};

use std::cmp::Ordering as Cmp;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::ManuallyDrop;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Estimates of distances to a target, guiding [`astar`].
///
/// An estimate must never exceed the distance (*admissible*), and be [`f64::INFINITY`] only if
/// `target` can't be reached from `v` (A* then skips `v`). If it also never drops along an arc
/// by more than the arc's weight (*consistent*: `estimate(u, t) <= w(u, v) + estimate(v, t)`),
/// every node is expanded at most once; otherwise nodes may be expanded again, still finding
/// shortest paths. ([`Landmarks`]' estimates are consistent but for their `f32` rounding, by
/// about `2^-23` of a distance: a node is seldom expanded twice.)
///
/// Any `Fn(usize, usize) -> f64` closure, of `v` and `target`, is one.
pub trait Heuristic: Sync {
    /// A lower bound on the distance from `v` to `target`.
    fn estimate(&self, v: usize, target: usize) -> f64;

    /// Whether estimates are exactly consistent, from and to any node (`estimate(u, t) <= w +
    /// estimate(v, t)` and `estimate(s, v) <= estimate(s, u) + w`, for every arc `u -> v` of
    /// weight `w`, as computed): bidirectional search then stops much sooner (see
    /// [`AStarOptions::bidirectional`]), but could stop too soon if they weren't, by however
    /// much they aren't. `false` unless overridden.
    fn consistent(&self) -> bool {
        false
    }
}

impl<F: Fn(usize, usize) -> f64 + Sync> Heuristic for F {
    #[inline(always)]
    fn estimate(&self, v: usize, target: usize) -> f64 {
        self(v, target)
    }
}

/// No estimate (always 0): A* is then Dijkstra's algorithm, stopping at the target.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoHeuristic;

impl Heuristic for NoHeuristic {
    #[inline(always)]
    fn estimate(&self, _: usize, _: usize) -> f64 {
        0.
    }
    fn consistent(&self) -> bool {
        true
    }
}

/// A shortest path [`astar`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortestPath {
    /// Its length: the sum of its arcs' weights.
    pub distance: f64,
    /// Its nodes, from the source to the target (just the source, if they're the same); empty
    /// if paths weren't asked for (see [`AStarOptions::paths`]).
    pub nodes: Vec<usize>,
    /// The nodes the search expanded (relaxed the arcs of) to find it, in both directions of a
    /// bidirectional one: how well the heuristic guided it.
    pub expanded: usize,
}

/// Hashes node ids by a multiplication, folded (node ids are well spread already).
#[derive(Clone, Copy, Default)]
struct NodeHasher(u64);

impl Hasher for NodeHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    #[inline(always)]
    fn write_usize(&mut self, n: usize) {
        let x = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 32);
    }
}

type NodeMap<V> = HashMap<usize, V, BuildHasherDefault<NodeHasher>>;

/// A node in A*'s heap: least `f` (distance so far plus estimate) first, then most `g`
/// (distance so far: the deeper, the closer to the target, presumably).
#[derive(Clone, Copy)]
struct Entry {
    f: f64,
    g: f64,
    node: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Cmp::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Cmp> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Cmp {
        // reversed: `BinaryHeap` pops the greatest
        other
            .f
            .total_cmp(&self.f)
            .then(self.g.total_cmp(&other.g))
            .then(other.node.cmp(&self.node))
    }
}

/// What a search keeps of each node it finds: its best distance so far and, if paths are asked
/// for, the node it was found through --- only the distance otherwise, 8 bytes less a node.
trait Record: Copy {
    const PATHS: bool;
    fn new(g: f64, parent: usize) -> Self;
    fn g(&self) -> f64;
    fn parent(&self) -> usize;
}

impl Record for f64 {
    const PATHS: bool = false;
    #[inline(always)]
    fn new(g: f64, _: usize) -> Self {
        g
    }
    #[inline(always)]
    fn g(&self) -> f64 {
        *self
    }
    fn parent(&self) -> usize {
        unreachable!("paths weren't asked for")
    }
}

impl Record for (f64, usize) {
    const PATHS: bool = true;
    #[inline(always)]
    fn new(g: f64, parent: usize) -> Self {
        (g, parent)
    }
    #[inline(always)]
    fn g(&self) -> f64 {
        self.0
    }
    #[inline(always)]
    fn parent(&self) -> usize {
        self.1
    }
}

/// One direction of a search, its memory reused by a thread across queries: each found node's
/// [`Record`], and the heap of nodes to expand.
struct Search<R> {
    best: NodeMap<R>,
    heap: BinaryHeap<Entry>,
}

impl<R> Default for Search<R> {
    fn default() -> Self {
        Self {
            best: NodeMap::default(),
            heap: BinaryHeap::new(),
        }
    }
}

/// Calls `f(v, w)` for each of `u`'s arcs (out of `u`, in `g`), as long as it continues.
#[inline(always)]
fn arcs_of<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    u: usize,
    mut f: impl FnMut(usize, f64) -> ControlFlow<()>,
) -> Result<ControlFlow<()>, Box<dyn std::error::Error>> {
    if E::is_weighted() {
        for (v, label) in g.weighted_neighbours(u)? {
            if f(v, label.weight::<f64>()).is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
    } else {
        for v in g.neighbours(u)? {
            if f(v, 1.).is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
    }
    Ok(ControlFlow::Continue(()))
}

impl<R: Record> Search<R> {
    fn reset(&mut self, start: usize, f: f64) {
        self.best.clear();
        self.heap.clear();
        self.best.insert(start, R::new(0., start));
        self.heap.push(Entry {
            f,
            g: 0.,
            node: start,
        });
    }

    /// The nodes from `v` back to `start`, following parents.
    fn chain(&self, start: usize, v: usize) -> Vec<usize> {
        let mut nodes = vec![v];
        let mut at = v;
        while at != start {
            at = self.best[&at].parent();
            nodes.push(at);
        }
        nodes
    }

    /// The least `f` of the heap's live entries (dropping stale ones: nodes found closer since).
    fn top(&mut self) -> f64 {
        while let Some(e) = self.heap.peek() {
            if e.g > self.best[&e.node].g() {
                self.heap.pop();
            } else {
                return e.f;
            }
        }
        f64::INFINITY
    }

    /// A* from `source` to `target` in `g`, calling `visitor`'s hooks (`None` if it stops).
    fn run<N, E, Ix, Dir, H, V>(
        &mut self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
        target: usize,
        h: &H,
        visitor: &mut V,
    ) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
    where
        N: graph::N,
        E: graph::E,
        Ix: graph::IndexType,
        Dir: Directedness,
        H: Heuristic + ?Sized,
        V: ShortestPathVisitor + ?Sized,
    {
        macro_rules! hook {
            ($call:expr) => {
                if let ControlFlow::Break(()) = $call {
                    return Ok(None);
                }
            };
        }
        let mut expanded = 0;
        let h0 = h.estimate(source, target);
        if h0 == f64::INFINITY {
            return Ok(None);
        }
        self.reset(source, h0);
        hook!(visitor.discover_vertex(source, 0., source));
        while let Some(Entry { g: gu, node: u, .. }) = self.heap.pop() {
            // a stale entry: `u` was found closer since
            if gu > self.best[&u].g() {
                continue;
            }
            hook!(visitor.examine_vertex(u, gu));
            if u == target {
                let mut nodes = if R::PATHS {
                    self.chain(source, target)
                } else {
                    vec![]
                };
                nodes.reverse();
                return Ok(Some(ShortestPath {
                    distance: gu,
                    nodes,
                    expanded,
                }));
            }
            expanded += 1;
            let (best, heap) = (&mut self.best, &mut self.heap);
            hook!(arcs_of(g, u, |v, w| {
                visitor.examine_edge(u, v, w)?;
                let gv = gu + w;
                let first = match best.get_mut(&v) {
                    Some(b) if b.g() <= gv => return visitor.edge_not_relaxed(u, v, w),
                    Some(b) => {
                        *b = R::new(gv, u);
                        false
                    }
                    None => {
                        best.insert(v, R::new(gv, u));
                        true
                    }
                };
                let hv = h.estimate(v, target);
                if hv != f64::INFINITY {
                    heap.push(Entry {
                        f: gv + hv,
                        g: gv,
                        node: v,
                    });
                }
                visitor.edge_relaxed(u, v, w, gv)?;
                if first {
                    visitor.discover_vertex(v, gv, u)?;
                }
                ControlFlow::Continue(())
            })?);
            hook!(visitor.finish_vertex(u, gu));
        }
        Ok(None)
    }
}

/// A bidirectional search's two directions: from the source (on the graph) and from the target
/// (on its reverse).
struct BiSearch<R> {
    forward: Search<R>,
    backward: Search<R>,
}

impl<R> Default for BiSearch<R> {
    fn default() -> Self {
        Self {
            forward: Search::default(),
            backward: Search::default(),
        }
    }
}

impl<R: Record> BiSearch<R> {
    /// Bidirectional A* from `source` to `target`: forward on `g`, backward on its `reverse`,
    /// expanding whichever direction's next node has the smaller key, and remembering the
    /// shortest path `mu` through a node both reached.
    ///
    /// With a [consistent](Heuristic::consistent) heuristic, the directions share the potential
    /// `p(v) = (h(v, t) - h(s, v)) / 2` (Ikeda et al.'s average: keys `g + p` forward, `g - p`
    /// backward), so that it's Dijkstra's bidirectional search on arcs whose reduced weights
    /// are non-negative, stopping once the two least keys add up to `mu`. Otherwise, each
    /// direction is guided by its own estimate (`h(v, t)`, `h(s, v)`), and it stops once either
    /// direction's least key is `mu` or more (the symmetric approach, Pohl's: correct for any
    /// admissible heuristic, as nodes found closer are expanded again, but each direction then
    /// searches nearly as far as one-directional A* would).
    fn run<N, E, Ix, Dir, H>(
        &mut self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
        reverse: &GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
        target: usize,
        h: &H,
    ) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
    where
        N: graph::N,
        E: graph::E,
        Ix: graph::IndexType,
        Dir: Directedness,
        H: Heuristic + ?Sized,
    {
        let h0 = h.estimate(source, target);
        if h0 == f64::INFINITY {
            return Ok(None);
        }
        if source == target {
            return Ok(Some(ShortestPath {
                distance: 0.,
                nodes: if R::PATHS { vec![source] } else { vec![] },
                expanded: 0,
            }));
        }
        let consistent = h.consistent();
        // a node's key, from its distance in a direction (`None`: no shorter path through it)
        let key = |forward: bool, v: usize, gv: f64| -> Option<f64> {
            // the estimates of the rest of the way, to the target (forward) or from the source
            let (ahead, behind) = if forward {
                (h.estimate(v, target), h.estimate(source, v))
            } else {
                (h.estimate(source, v), h.estimate(v, target))
            };
            if ahead == f64::INFINITY {
                return None;
            }
            Some(if consistent {
                gv + (ahead - behind) / 2.
            } else {
                gv + ahead
            })
        };
        self.forward
            .reset(source, key(true, source, 0.).unwrap_or(f64::INFINITY));
        self.backward
            .reset(target, key(false, target, 0.).unwrap_or(f64::INFINITY));
        let (mut mu, mut meet) = (f64::INFINITY, source);
        let mut expanded = 0;
        loop {
            let (top_f, top_b) = (self.forward.top(), self.backward.top());
            let done = if consistent {
                top_f + top_b >= mu || top_f == f64::INFINITY || top_b == f64::INFINITY
            } else {
                top_f >= mu || top_b >= mu
            };
            if done {
                break;
            }
            let forward = top_f <= top_b;
            let (this, other, graph) = if forward {
                (&mut self.forward, &self.backward, g)
            } else {
                (&mut self.backward, &self.forward, reverse)
            };
            let Entry { g: gu, node: u, .. } = this.heap.pop().unwrap();
            expanded += 1;
            let (best, heap) = (&mut this.best, &mut this.heap);
            let _ = arcs_of(graph, u, |v, w| {
                let gv = gu + w;
                // a path through `v` with this `g` can't beat `mu`: no need to keep it
                let ahead = if forward {
                    h.estimate(v, target)
                } else {
                    h.estimate(source, v)
                };
                if gv + ahead >= mu {
                    return ControlFlow::Continue(());
                }
                match best.get_mut(&v) {
                    Some(b) if b.g() <= gv => return ControlFlow::Continue(()),
                    Some(b) => *b = R::new(gv, u),
                    None => {
                        best.insert(v, R::new(gv, u));
                    }
                }
                if let Some(f) = key(forward, v, gv) {
                    heap.push(Entry { f, g: gv, node: v });
                }
                if let Some(o) = other.best.get(&v)
                    && gv + o.g() < mu
                {
                    (mu, meet) = (gv + o.g(), v);
                }
                ControlFlow::Continue(())
            })?;
        }
        if mu == f64::INFINITY {
            return Ok(None);
        }
        let nodes = if R::PATHS {
            let mut nodes = self.forward.chain(source, meet);
            nodes.reverse();
            // from `meet` (already in) to the target
            nodes.extend(self.backward.chain(target, meet).into_iter().skip(1));
            nodes
        } else {
            vec![]
        };
        Ok(Some(ShortestPath {
            distance: mu,
            nodes,
            expanded,
        }))
    }
}

/// Graphs whose arcs [`AStarOptions::bidirectional`] search can follow backward: an undirected
/// graph's are its own, a directed graph's are its transpose's (built the first time).
pub trait Reversible: Directedness {
    /// The graph with `g`'s arcs reversed.
    fn reverse<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        g: &GraphMemoryMap<N, E, Ix, Self>,
    ) -> Result<&GraphMemoryMap<N, E, Ix, Self>, Box<dyn std::error::Error>>;
}

impl Reversible for Undirected {
    fn reverse<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        g: &GraphMemoryMap<N, E, Ix, Self>,
    ) -> Result<&GraphMemoryMap<N, E, Ix, Self>, Box<dyn std::error::Error>> {
        Ok(g)
    }
}

impl Reversible for Directed {
    fn reverse<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        g: &GraphMemoryMap<N, E, Ix, Self>,
    ) -> Result<&GraphMemoryMap<N, E, Ix, Self>, Box<dyn std::error::Error>> {
        g.transpose()
    }
}

/// How [`astar_with`] and [`astar_many_with`] search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AStarOptions {
    /// Search from both ends at once (the target's on the graph's reverse, built once on
    /// directed graphs), meeting in the middle. The heuristic must then bound distances between
    /// any two nodes, as it's asked for the source's to each node too (as [`Landmarks`]' and
    /// [`NoHeuristic`]'s do).
    ///
    /// It pays with [consistent](Heuristic::consistent) heuristics: without one
    /// ([`NoHeuristic`]), it's bidirectional Dijkstra, expanding about half the nodes Dijkstra's
    /// algorithm does. With others ([`Landmarks`]', which are consistent only up to rounding),
    /// it's correct but expands more nodes than one-directional search.
    pub bidirectional: bool,
    /// Find paths' nodes: without, only distances are found, keeping 8 bytes less per node
    /// explored ([`ShortestPath::nodes`] is then empty).
    pub paths: bool,
}

impl Default for AStarOptions {
    /// One-directional, with paths.
    fn default() -> Self {
        Self {
            bidirectional: false,
            paths: true,
        }
    }
}

fn check_query<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    target: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = g.size();
    for v in [source, target] {
        if v >= n {
            return Err(
                format!("error node with id {v} doesn't exist: max id is |V| = {n}").into(),
            );
        }
    }
    Ok(())
}

/// A query's memory, of the kind its options ask for, reused by a thread across queries.
enum Searcher {
    Paths(Search<(f64, usize)>),
    Distances(Search<f64>),
    BiPaths(Box<BiSearch<(f64, usize)>>),
    BiDistances(Box<BiSearch<f64>>),
}

impl Searcher {
    fn new(options: AStarOptions) -> Self {
        match (options.bidirectional, options.paths) {
            (false, true) => Searcher::Paths(Search::default()),
            (false, false) => Searcher::Distances(Search::default()),
            (true, true) => Searcher::BiPaths(Box::default()),
            (true, false) => Searcher::BiDistances(Box::default()),
        }
    }

    fn run<N, E, Ix, Dir, H>(
        &mut self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
        reverse: &GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
        target: usize,
        h: &H,
    ) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
    where
        N: graph::N,
        E: graph::E,
        Ix: graph::IndexType,
        Dir: Directedness,
        H: Heuristic + ?Sized,
    {
        let no = &mut NoOpShortestPathVisitor;
        match self {
            Searcher::Paths(s) => s.run(g, source, target, h, no),
            Searcher::Distances(s) => s.run(g, source, target, h, no),
            Searcher::BiPaths(s) => s.run(g, reverse, source, target, h),
            Searcher::BiDistances(s) => s.run(g, reverse, source, target, h),
        }
    }
}

/// A shortest path from `source` to `target` in `g` (following its arcs), by A* guided by `h`,
/// or `None` if `target` can't be reached: [`astar_with`] by default (one-directional, with
/// paths).
pub fn astar<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness, H>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    target: usize,
    h: &H,
) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
where
    H: Heuristic + ?Sized,
{
    check_query(g, source, target)?;
    // checks the weights finite and non-negative (once: the total is kept)
    g.total_weight()?;
    Search::<(f64, usize)>::default().run(g, source, target, h, &mut NoOpShortestPathVisitor)
}

/// [`astar`], searching as `options` say: from both ends at once, and with or without paths.
pub fn astar_with<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible, H>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    target: usize,
    h: &H,
    options: AStarOptions,
) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
where
    H: Heuristic + ?Sized,
{
    check_query(g, source, target)?;
    g.total_weight()?;
    let reverse = if options.bidirectional {
        Dir::reverse(g)?
    } else {
        g
    };
    Searcher::new(options).run(g, reverse, source, target, h)
}

/// A* from `source` to `target` (as [`astar`]) calling `visitor`'s hooks (see
/// [`ShortestPathVisitor`]): the path found, or `None` if `target` can't be reached or the
/// visitor stopped the search. Nodes found closer after they were examined (with an
/// inconsistent heuristic) are examined again.
pub fn astar_visit<N, E, Ix, Dir, H, V>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    source: usize,
    target: usize,
    h: &H,
    visitor: &mut V,
) -> Result<Option<ShortestPath>, Box<dyn std::error::Error>>
where
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness,
    H: Heuristic + ?Sized,
    V: ShortestPathVisitor + ?Sized,
{
    check_query(g, source, target)?;
    g.total_weight()?;
    Search::<(f64, usize)>::default().run(g, source, target, h, visitor)
}

/// [`astar`] for each of `queries` (`(source, target)` pairs), in parallel on `g`'s threads,
/// each thread running whole queries (with memory reused across them): the results, in the
/// queries' order.
pub fn astar_many<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness, H>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    queries: &[(usize, usize)],
    h: &H,
) -> Result<Vec<Option<ShortestPath>>, Box<dyn std::error::Error>>
where
    H: Heuristic + ?Sized,
{
    many(g, g, queries, h, AStarOptions::default())
}

/// [`astar_many`], searching as `options` say (see [`astar_with`]).
pub fn astar_many_with<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible, H>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    queries: &[(usize, usize)],
    h: &H,
    options: AStarOptions,
) -> Result<Vec<Option<ShortestPath>>, Box<dyn std::error::Error>>
where
    H: Heuristic + ?Sized,
{
    let reverse = if options.bidirectional {
        Dir::reverse(g)?
    } else {
        g
    };
    many(g, reverse, queries, h, options)
}

fn many<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness, H>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    reverse: &GraphMemoryMap<N, E, Ix, Dir>,
    queries: &[(usize, usize)],
    h: &H,
    options: AStarOptions,
) -> Result<Vec<Option<ShortestPath>>, Box<dyn std::error::Error>>
where
    H: Heuristic + ?Sized,
{
    for &(s, t) in queries {
        check_query(g, s, t)?;
    }
    g.total_weight()?;
    let threads = g.thread_num().clamp(1, queries.len().max(1));
    let next = AtomicUsize::new(0);
    // each thread's paths, with their queries' indices
    type Found = Result<Vec<(usize, Option<ShortestPath>)>, String>;
    let parts: Vec<Found> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut search = Searcher::new(options);
                    let mut found = vec![];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(s, t)) = queries.get(i) else {
                            break Ok(found);
                        };
                        let path = search.run(g, reverse, s, t, h).map_err(|e| e.to_string())?;
                        found.push((i, path));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| Err(format!("{e:?}"))))
            .collect()
    });
    let mut paths = vec![None; queries.len()];
    for part in parts {
        for (i, path) in part? {
            paths[i] = path;
        }
    }
    Ok(paths)
}

/// A distance whose `f32` doesn't hold it, other than infinity: [`Landmarks`] can't use it.
const UNKNOWN: f32 = f32::MAX;
/// The relative error of a distance's `f32` (2^-24), and of the `f64` sums it was rounded from,
/// with room to spare: estimates are lowered by it, so that they never exceed distances.
const ROUNDING: f64 = 1. / (1u64 << 22) as f64;

/// A lower bound on `a - b`, of distances rounded to `f32`s: infinite if `a` is and `b` isn't
/// (`a` unreachable, `b` reachable), and 0 if either is [`UNKNOWN`] or `b` infinite.
#[inline(always)]
fn bound(a: f32, b: f32) -> f64 {
    if b == f32::INFINITY || a == UNKNOWN || b == UNKNOWN {
        return 0.;
    }
    if a == f32::INFINITY {
        return f64::INFINITY;
    }
    let (a, b) = (a as f64, b as f64);
    (a - b) - (a + b) * ROUNDING
}

/// ALT's [`Heuristic`]: every node's distances from (and, on directed graphs, to) a few
/// *landmarks*, bounding any distance by the triangle inequality: `d(v, t) >= d(L, t) - d(L, v)`
/// and `d(v, t) >= d(v, L) - d(t, L)`, for every landmark `L`. Estimates are infinite where
/// those distances show `t` unreachable from `v`, and lowered by a bound on their `f32` rounding,
/// so that they never exceed distances; they're then consistent up to that rounding (see
/// [`Heuristic`]).
///
/// Landmarks are chosen *farthest* first (Goldberg and Harrelson): the node farthest from node
/// 0, then, repeatedly, the node farthest from the landmarks chosen (a node none reaches, if
/// any: one in a part of the graph they don't cover). Their distances are found by
/// [`DijkstraDists`] (Δ-stepping, on large graphs), one landmark at a time on all of the
/// graph's threads, and kept as `f32`s, node by node: `4 * k * |V|` bytes, twice that on
/// directed graphs, memory-mapped in the graph's cache.
///
/// [`DijkstraDists`]: super::dijkstra::DijkstraDists
pub struct Landmarks<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    nodes: Vec<usize>,
    /// `d(L_i, v)` at `v * k + i`.
    from: AbstractedProceduralMemoryMut<f32>,
    /// `d(v, L_i)` at `v * k + i`, on directed graphs (on undirected ones, it's `from`).
    to: Option<AbstractedProceduralMemoryMut<f32>>,
}

/// What a landmarks' metadata file starts with (and its format's version).
const MAGIC: &[u8; 8] = b"SPXLMK01";

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible>
    Landmarks<'a, N, E, Ix, Dir>
{
    /// Chooses `k` landmarks of `g` (fewer if it has fewer nodes) and finds their distances (on
    /// directed graphs, to every node too, on the transpose, which is built if it isn't yet).
    /// They're kept in the graph's cache, replacing any it held: [`open`](Self::open) maps them
    /// back, without finding them again, until [`drop_cache`](Self::drop_cache) (or the graph's)
    /// removes them.
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let transpose = if Dir::DIRECTED {
            Some(Dir::reverse(g)?)
        } else {
            None
        };
        // whatever was kept is replaced: dropped first, so that a build that fails leaves none
        let meta = g.build_pers_cache_filename(CacheFile::Landmarks, Some(8))?;
        crate::shared_slice::remove_if_exists(&meta)?;
        let landmarks = Self::build(g, transpose, k)?;
        let mut bytes = MAGIC.to_vec();
        let flags = Dir::DIRECTED as u64 | (E::is_weighted() as u64) << 1;
        for x in [g.size() as u64, landmarks.nodes.len() as u64, flags] {
            bytes.extend(x.to_le_bytes());
        }
        for &l in &landmarks.nodes {
            bytes.extend((l as u64).to_le_bytes());
        }
        // written whole, then renamed: a metadata file is only ever there complete
        let tmp = format!("{meta}.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &meta)?;
        Ok(landmarks)
    }

    /// The landmarks [`new`](Self::new) kept in `g`'s cache, mapped back; an error if none are
    /// (or they're another graph's: of another size, directedness or weights --- an unweighted
    /// graph opened on a weighted one's cache entry, say).
    pub fn open(g: &'a GraphMemoryMap<N, E, Ix, Dir>) -> Result<Self, Box<dyn std::error::Error>> {
        let meta = g.build_pers_cache_filename(CacheFile::Landmarks, Some(8))?;
        let bytes = std::fs::read(&meta).map_err(|e| -> Box<dyn std::error::Error> {
            format!("error no landmarks kept for this graph ({meta}: {e})").into()
        })?;
        let word = |i: usize| -> Option<u64> {
            let at = MAGIC.len() + 8 * i;
            Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
        };
        let malformed = || -> Box<dyn std::error::Error> {
            format!("error malformed landmarks metadata {meta}").into()
        };
        if !bytes.starts_with(MAGIC) {
            return Err(malformed());
        }
        let (n, k, flags) = (
            word(0).ok_or_else(malformed)? as usize,
            word(1).ok_or_else(malformed)? as usize,
            word(2).ok_or_else(malformed)?,
        );
        if n != g.size()
            || flags & 1 != Dir::DIRECTED as u64
            || flags >> 1 & 1 != E::is_weighted() as u64
        {
            return Err(format!(
                "error the landmarks kept are another graph's: of {n} nodes, {}directed, {}weighted",
                if flags & 1 == 1 { "" } else { "un" },
                if flags >> 1 & 1 == 1 { "" } else { "un" }
            )
            .into());
        }
        let nodes = (0..k)
            .map(|i| word(3 + i).map(|l| l as usize).filter(|&l| l < n))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(malformed)?;
        if k == 0 || bytes.len() != MAGIC.len() + 8 * (3 + k) {
            return Err(malformed());
        }
        let table =
            |seq: usize| -> Result<AbstractedProceduralMemoryMut<f32>, Box<dyn std::error::Error>> {
                let file = g.build_pers_cache_filename(CacheFile::Landmarks, Some(seq))?;
                let table = AbstractedProceduralMemoryMut::<f32>::from_file_name(&file)?;
                if table.as_slice().len() < n * k {
                    return Err(format!("error landmarks' distances {file} are incomplete").into());
                }
                Ok(table)
            };
        let from = table(0)?;
        let to = if Dir::DIRECTED { Some(table(1)?) } else { None };
        Ok(Self { g, nodes, from, to })
    }
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>
    Landmarks<'a, N, E, Ix, Dir>
{
    fn build(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        transpose: Option<&GraphMemoryMap<N, E, Ix, Dir>>,
        k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = g.size();
        let k = k.min(n);
        if k == 0 {
            return Err("error landmarks need at least one landmark, on a graph with nodes".into());
        }
        let from_fn = g.build_pers_cache_filename(CacheFile::Landmarks, Some(0))?;
        let to_fn = g.build_pers_cache_filename(CacheFile::Landmarks, Some(1))?;
        let mut from = SharedSliceMut::<f32>::abst_mem_mut(&from_fn, n * k, true)?;
        let mut to = match transpose {
            Some(_) => Some(SharedSliceMut::<f32>::abst_mem_mut(&to_fn, n * k, true)?),
            None => None,
        };
        let threads = g.thread_num().max(1);
        // on directed graphs, the distances from and to a landmark are found at once, on half
        // the threads each: Δ-stepping gains little from the last threads on graphs of long
        // paths, where landmarks' searches take longest
        let split = if transpose.is_some() {
            (threads / 2).max(1)
        } else {
            threads
        };
        let file = |seq| g.build_cache_filename(CacheFile::Landmarks, Some(seq));
        let scratch: Vec<String> = (2..8).map(file).collect::<Result<_, _>>()?;
        let result = (|| -> Result<Vec<usize>, Box<dyn std::error::Error>> {
            let mut out = SharedSliceMut::<f64>::abst_mem_mut(&scratch[0], n, true)?;
            let mut back = SharedSliceMut::<f64>::abst_mem_mut(&scratch[1], n, true)?;
            // the first landmark: the node farthest from node 0 (one it doesn't reach, if any)
            let delta = resolve_engine(g, ShortestPathEngine::Auto, threads)?;
            distances_into(g, 0, out.as_mut_slice(), delta, threads, &scratch[2..4])?;
            let d0 = out.as_slice();
            let mut nodes = vec![farthest(n, &|v| d0[v] as f32, threads)];
            for i in 0..k {
                let l = nodes[i];
                let (out, back) = (out.as_mut_slice(), back.as_mut_slice());
                let (from_table, to_table) = (&mut from, to.as_mut());
                let (out_scratch, back_scratch) = (&scratch[2..4], &scratch[4..6]);
                std::thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
                    let forward = s.spawn(move || {
                        store_column(g, l, out, out_scratch, from_table, i, k, split)
                            .map_err(|e| e.to_string())
                    });
                    if let (Some(t), Some(to_table)) = (transpose, to_table) {
                        store_column(t, l, back, back_scratch, to_table, i, k, split)?;
                    }
                    forward
                        .join()
                        .map_err(|e| format!("{e:?}"))?
                        .map_err(|e| e.into())
                })?;
                if i + 1 < k {
                    // the node farthest from the landmarks so far
                    let from = from.as_slice();
                    let next = farthest(
                        n,
                        &|v| {
                            from[v * k..v * k + i + 1]
                                .iter()
                                .copied()
                                .fold(f32::INFINITY, f32::min)
                        },
                        threads,
                    );
                    nodes.push(next);
                }
            }
            Ok(nodes)
        })();
        for f in &scratch {
            let _ = std::fs::remove_file(f);
        }
        let nodes = result?;
        from.flush()?;
        if let Some(to) = to.as_mut() {
            to.flush()?;
        }
        Ok(Self { g, nodes, from, to })
    }

    /// The landmarks, in the order they were chosen.
    pub fn landmarks(&self) -> &[usize] {
        &self.nodes
    }

    /// Removes the landmarks from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        for seq in [8, 0, 1] {
            let file = this
                .g
                .build_pers_cache_filename(CacheFile::Landmarks, Some(seq))?;
            crate::shared_slice::remove_if_exists(file)?;
        }
        Ok(())
    }
}

/// The node of greatest `distance` of `n` (`f32`s): the first infinite one, if any, else the
/// first of the greatest (node 0, if none) --- the same whatever the `threads`.
fn farthest(n: usize, distance: &(dyn Fn(usize) -> f32 + Sync), threads: usize) -> usize {
    let threads = threads.clamp(1, n.max(1));
    let load = n.div_ceil(threads);
    // each range's first infinite node, or its first of the greatest
    let parts: Vec<(Option<usize>, usize, f32)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut best = (None, 0, f32::NEG_INFINITY);
                    for v in (t * load).min(n)..((t + 1) * load).min(n) {
                        let d = distance(v);
                        if d == f32::INFINITY {
                            return (Some(v), v, d);
                        }
                        if d > best.2 {
                            best = (None, v, d);
                        }
                    }
                    best
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if let Some(v) = parts.iter().find_map(|p| p.0) {
        return v;
    }
    let mut best = (0, f32::NEG_INFINITY);
    for &(_, v, d) in &parts {
        if d > best.1 {
            best = (v, d);
        }
    }
    best.0
}

/// Stores every node's distance from `l` in `g` as `table[v * k + i]`, an `f32` ([`UNKNOWN`] if
/// too large for one), found in `out` on `threads` threads (Dijkstra's memory mapped in the 2
/// `scratch` files, if it runs).
#[allow(clippy::too_many_arguments)]
fn store_column<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    l: usize,
    out: &mut [f64],
    scratch: &[String],
    table: &mut AbstractedProceduralMemoryMut<f32>,
    i: usize,
    k: usize,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let delta = resolve_engine(g, ShortestPathEngine::Auto, threads)?;
    distances_into(g, l, out, delta, threads, scratch)?;
    let dist: &[f64] = out;
    let n = dist.len();
    let table = table.shared_slice();
    let load = n.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let mut table = table;
            s.spawn(move || {
                let begin = (t * load).min(n);
                for (v, &x) in dist[begin..((t + 1) * load).min(n)].iter().enumerate() {
                    *table.get_mut((begin + v) * k + i) = if x.is_infinite() {
                        f32::INFINITY
                    } else if x >= UNKNOWN as f64 {
                        UNKNOWN
                    } else {
                        x as f32
                    };
                }
            });
        }
    });
    Ok(())
}

impl<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness> Heuristic
    for Landmarks<'_, N, E, Ix, Dir>
{
    #[inline]
    fn estimate(&self, v: usize, target: usize) -> f64 {
        let k = self.nodes.len();
        let from = self.from.as_slice();
        let (fv, ft) = (&from[v * k..v * k + k], &from[target * k..target * k + k]);
        let mut best = 0f64;
        match &self.to {
            None => {
                for i in 0..k {
                    best = best.max(bound(ft[i], fv[i])).max(bound(fv[i], ft[i]));
                }
            }
            Some(to) => {
                let to = to.as_slice();
                let (tv, tt) = (&to[v * k..v * k + k], &to[target * k..target * k + k]);
                for i in 0..k {
                    best = best.max(bound(ft[i], fv[i])).max(bound(tv[i], tt[i]));
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::random_arcs;
    use crate::graph::label::VoidLabel;
    use crate::trails::dijkstra::test::{all_pairs, close, weighted, weighted_arcs, weighted_on};

    /// Checks `path` is one from `s` to `t` of `arcs`, as long as `want` (the distance).
    fn check(
        path: &ShortestPath,
        s: usize,
        t: usize,
        want: f64,
        arcs: &[(usize, usize, f64)],
        directed: bool,
        what: &str,
    ) {
        assert!(
            close(path.distance, want),
            "{what}: {s} -> {t}: {} vs {want}",
            path.distance
        );
        assert_eq!(
            (path.nodes[0], *path.nodes.last().unwrap()),
            (s, t),
            "{what}"
        );
        // the lightest arc between consecutive nodes
        let mut length = 0.;
        for pair in path.nodes.windows(2) {
            let w = arcs
                .iter()
                .filter(|&&(u, v, _)| {
                    (u, v) == (pair[0], pair[1]) || (!directed && (v, u) == (pair[0], pair[1]))
                })
                .map(|a| a.2)
                .fold(f64::INFINITY, f64::min);
            assert!(w.is_finite(), "{what}: no arc {} -> {}", pair[0], pair[1]);
            length += w;
        }
        assert!(
            close(length, want),
            "{what}: the path is {length} long, not {want}"
        );
    }

    fn queries(n: usize, count: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let s = (state >> 33) as usize % n;
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s, (state >> 33) as usize % n)
            })
            .collect()
    }

    /// A* finds shortest paths with no heuristic, with landmarks, with an admissible but
    /// inconsistent heuristic, one at a time or many in parallel; and nothing where there's no
    /// path (two parts, one-way linked).
    #[test]
    fn astar_finds_shortest_paths() {
        for seed in 0..3u64 {
            let n = 120;
            let mut arcs = weighted_arcs(n / 2, 300, 50 + seed, seed == 1);
            arcs.extend(
                weighted_arcs(n / 2, 300, 60 + seed, false)
                    .into_iter()
                    .map(|(u, v, w)| (u + n / 2, v + n / 2, w)),
            );
            arcs.push((5, n / 2 + 5, 2.5));
            let qs = queries(n, 400, seed);
            for directed in [false, true] {
                let d = all_pairs(n, &arcs, directed);
                macro_rules! run {
                    ($Dir:ty, $name:expr) => {{
                        let g = weighted::<$Dir>(&format!("{}{seed}", $name), n, &arcs);
                        let mut landmarks = Landmarks::<_, _, _, $Dir>::new(&g, 4).unwrap();
                        // admissible (half the distance, or a third), not consistent
                        let wobbly = |v: usize, t: usize| {
                            if d[v][t].is_infinite() {
                                f64::INFINITY
                            } else {
                                d[v][t] / (2 + v % 2) as f64
                            }
                        };
                        let what = format!("seed {seed}, directed {directed}");
                        for (name, h) in [
                            ("none", &NoHeuristic as &dyn Heuristic),
                            ("landmarks", &landmarks as &dyn Heuristic),
                            ("wobbly", &wobbly as &dyn Heuristic),
                        ] {
                            let many = astar_many(&*g, &qs, h).unwrap();
                            for (&(s, t), got) in qs.iter().zip(&many) {
                                let one = astar(&*g, s, t, h).unwrap();
                                assert_eq!(&one, got, "{what} {name}: one vs many");
                                match got {
                                    None => assert!(
                                        d[s][t].is_infinite(),
                                        "{what} {name}: {s} -> {t} not found"
                                    ),
                                    Some(p) => check(
                                        p,
                                        s,
                                        t,
                                        d[s][t],
                                        &arcs,
                                        directed,
                                        &format!("{what} {name}"),
                                    ),
                                }
                            }
                        }
                        landmarks.drop_cache().unwrap();
                    }};
                }
                if directed {
                    run!(Directed, "a_d");
                } else {
                    run!(Undirected, "a_u");
                }
            }
        }
    }

    /// Landmarks' estimates never exceed distances, are infinite only where there's no path, are
    /// 0 at the target, and are consistent but for rounding; their choice doesn't depend on the
    /// threads.
    #[test]
    fn landmark_estimates_are_admissible_and_consistent() {
        for seed in 0..3u64 {
            let n = 150;
            let mut arcs = weighted_arcs(n - 10, 450, 70 + seed, seed % 2 == 0);
            // nodes nothing reaches, and one reaching nothing
            arcs.extend((n - 10..n - 1).map(|u| (u, u % 7, 1.5)));
            for directed in [false, true] {
                let d = all_pairs(n, &arcs, directed);
                let mut chosen = vec![];
                for threads in [1u8, 4, 7] {
                    macro_rules! check {
                        ($Dir:ty) => {{
                            let g = weighted_on::<$Dir>(
                                &format!("l{seed}_{directed}_{threads}"),
                                n,
                                &arcs,
                                threads,
                            );
                            let mut l = Landmarks::<_, _, _, $Dir>::new(&g, 6).unwrap();
                            chosen.push(l.landmarks().to_vec());
                            for v in 0..n {
                                for t in 0..n {
                                    let e = l.estimate(v, t);
                                    assert!(e >= 0., "negative {v} -> {t}");
                                    assert!(
                                        e <= d[v][t],
                                        "directed {directed}: {v} -> {t}: {e} > {}",
                                        d[v][t]
                                    );
                                    if e == f64::INFINITY {
                                        assert!(d[v][t].is_infinite());
                                    }
                                }
                                assert_eq!(l.estimate(v, v), 0.);
                            }
                            // consistent, but for rounding (of distances up to `max`)
                            let max = d
                                .iter()
                                .flatten()
                                .copied()
                                .filter(|x| x.is_finite())
                                .fold(0., f64::max);
                            let slack = 8. * ROUNDING * (1. + max);
                            let both_ways = arcs.iter().flat_map(|&(u, v, w)| {
                                let back = (!directed).then_some((v, u, w));
                                std::iter::once((u, v, w)).chain(back)
                            });
                            for (u, v, w) in both_ways {
                                for t in 0..n {
                                    let (eu, ev) = (l.estimate(u, t), l.estimate(v, t));
                                    if eu.is_finite() {
                                        assert!(
                                            eu <= w + ev + slack,
                                            "inconsistent {u} -> {v} for {t}: {eu} > {w} + {ev}"
                                        );
                                    }
                                }
                            }
                            l.drop_cache().unwrap();
                        }};
                    }
                    if directed {
                        check!(Directed)
                    } else {
                        check!(Undirected)
                    }
                }
                assert!(chosen.windows(2).all(|c| c[0] == c[1]), "{chosen:?}");
            }
        }
    }

    /// Unweighted graphs: A* counts arcs; same nodes; out-of-range nodes are errors.
    #[test]
    fn astar_on_unweighted_graphs_and_errors() {
        let arcs = random_arcs(80, 300, 3);
        let g = crate::components::test_utils::graph("astar_plain", "general", 80, &arcs);
        let unit: Vec<_> = arcs.iter().map(|&(u, v)| (u, v, 1.)).collect();
        let d = all_pairs(80, &unit, true);
        let mut l = Landmarks::<_, _, _, Directed>::new(&g, 3).unwrap();
        for (s, t) in queries(80, 200, 1) {
            let got = astar(&*g, s, t, &l).unwrap();
            assert_eq!(
                got.as_ref().map(|p| p.distance),
                d[s][t].is_finite().then_some(d[s][t])
            );
        }
        let same = astar(&*g, 7, 7, &NoHeuristic).unwrap().unwrap();
        assert_eq!(
            (same.distance, same.nodes.clone(), same.expanded),
            (0., vec![7], 0)
        );
        assert!(astar(&*g, 0, 80, &NoHeuristic).is_err());
        assert!(astar_many(&*g, &[(1, 2), (80, 1)], &NoHeuristic).is_err());
        l.drop_cache().unwrap();
        let neg = weighted::<Directed>("astar_neg", 3, &[(0, 1, -1.)]);
        assert!(astar(&*neg, 0, 1, &NoHeuristic).is_err());
    }

    /// On a grid (long paths, like road networks'), landmarks guide A* to expand a fraction of
    /// the nodes Dijkstra's algorithm does, to the same distances.
    #[test]
    fn landmarks_focus_the_search() {
        let side = 60;
        let mut state = 5u64;
        let mut arcs = vec![];
        for r in 0..side {
            for c in 0..side {
                let u = r * side + c;
                for v in [
                    (c + 1 < side).then_some(u + 1),
                    (r + 1 < side).then_some(u + side),
                ]
                .into_iter()
                .flatten()
                {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    arcs.push((u, v, 1. + ((state >> 33) % 9) as f64));
                }
            }
        }
        let g = weighted::<Undirected>("grid", side * side, &arcs);
        let mut l = Landmarks::<_, _, _, Undirected>::new(&g, 4).unwrap();
        let (mut guided, mut blind) = (0, 0);
        for (s, t) in queries(side * side, 50, 3) {
            let a = astar(&*g, s, t, &l).unwrap().unwrap();
            let b = astar(&*g, s, t, &NoHeuristic).unwrap().unwrap();
            assert!(close(a.distance, b.distance), "{s} -> {t}");
            (guided, blind) = (guided + a.expanded, blind + b.expanded);
        }
        assert!(
            3 * guided <= blind,
            "landmarks expanded {guided} nodes, no heuristic {blind}"
        );
        l.drop_cache().unwrap();
    }

    /// Every combination of options finds the same distances (bidirectional ones within
    /// rounding: they add the two halves), with or without paths, with any heuristic, and
    /// nothing where there's no path.
    #[test]
    fn astar_options_agree() {
        let all = [
            AStarOptions {
                bidirectional: false,
                paths: true,
            },
            AStarOptions {
                bidirectional: false,
                paths: false,
            },
            AStarOptions {
                bidirectional: true,
                paths: true,
            },
            AStarOptions {
                bidirectional: true,
                paths: false,
            },
        ];
        for seed in 0..3u64 {
            let n = 120;
            let mut arcs = weighted_arcs(n / 2, 300, 90 + seed, seed == 2);
            arcs.extend(
                weighted_arcs(n / 2, 300, 95 + seed, false)
                    .into_iter()
                    .map(|(u, v, w)| (u + n / 2, v + n / 2, w)),
            );
            arcs.push((7, n / 2 + 1, 1.25));
            let qs = queries(n, 300, seed + 7);
            for directed in [false, true] {
                let d = all_pairs(n, &arcs, directed);
                macro_rules! run {
                    ($Dir:ty) => {{
                        let g = weighted::<$Dir>(&format!("o{seed}_{directed}"), n, &arcs);
                        let mut l = Landmarks::<_, _, _, $Dir>::new(&g, 3).unwrap();
                        let wobbly = |v: usize, t: usize| {
                            if d[v][t].is_infinite() {
                                f64::INFINITY
                            } else {
                                d[v][t] / (2 + v % 3) as f64
                            }
                        };
                        for (name, h) in [
                            ("none", &NoHeuristic as &dyn Heuristic),
                            ("landmarks", &l as &dyn Heuristic),
                            ("wobbly", &wobbly as &dyn Heuristic),
                        ] {
                            for options in all {
                                let what =
                                    format!("seed {seed} directed {directed} {name} {options:?}");
                                let many = astar_many_with(&*g, &qs, h, options).unwrap();
                                for (&(s, t), got) in qs.iter().zip(&many) {
                                    let one = astar_with(&*g, s, t, h, options).unwrap();
                                    assert_eq!(&one, got, "{what}");
                                    match got {
                                        None => {
                                            assert!(d[s][t].is_infinite(), "{what}: {s} -> {t}")
                                        }
                                        Some(p) if options.paths => {
                                            check(p, s, t, d[s][t], &arcs, directed, &what)
                                        }
                                        Some(p) => {
                                            assert!(p.nodes.is_empty(), "{what}");
                                            assert!(
                                                close(p.distance, d[s][t]),
                                                "{what}: {s} -> {t}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        l.drop_cache().unwrap();
                    }};
                }
                if directed {
                    run!(Directed)
                } else {
                    run!(Undirected)
                }
            }
        }
    }

    /// Kept landmarks reopen as they were built; they're replaced by new ones, refused by graphs
    /// they aren't of, and gone once dropped.
    #[test]
    fn landmarks_reopen() {
        let n = 90;
        let arcs = weighted_arcs(n, 400, 11, true);
        let g = weighted::<Directed>("reopen", n, &arcs);
        assert!(Landmarks::open(&*g).is_err());
        let built = Landmarks::new(&*g, 5).unwrap();
        let estimates: Vec<f64> = (0..n * n).map(|i| built.estimate(i / n, i % n)).collect();
        let nodes = built.landmarks().to_vec();
        drop(built);
        let mut opened = Landmarks::open(&*g).unwrap();
        assert_eq!(opened.landmarks(), nodes);
        for (i, e) in estimates.iter().enumerate() {
            assert_eq!(opened.estimate(i / n, i % n).to_bits(), e.to_bits());
        }
        // replaced
        let fewer = Landmarks::new(&*g, 2).unwrap();
        assert_eq!(Landmarks::open(&*g).unwrap().landmarks(), fewer.landmarks());
        // the same cache entry, unweighted: not its landmarks
        let unweighted =
            DiGraph::<VoidLabel, (), usize>::open(g.cache_filename(), None, Some(2)).unwrap();
        let err = Landmarks::open(&unweighted).err().unwrap();
        assert!(err.to_string().contains("another graph's"), "{err}");
        opened.drop_cache().unwrap();
        assert!(Landmarks::open(&*g).is_err());
    }

    /// Records the hooks a visitor sees, stopping after `stop_after` examined nodes.
    #[derive(Default)]
    struct Recorder {
        discovered: Vec<(usize, f64, usize)>,
        examined: Vec<(usize, f64)>,
        relaxed: usize,
        not_relaxed: usize,
        edges: usize,
        finished: usize,
        stop_after: Option<usize>,
    }

    impl ShortestPathVisitor for Recorder {
        fn discover_vertex(&mut self, v: usize, d: f64, p: usize) -> ControlFlow<()> {
            self.discovered.push((v, d, p));
            ControlFlow::Continue(())
        }
        fn examine_vertex(&mut self, u: usize, d: f64) -> ControlFlow<()> {
            if self.stop_after == Some(self.examined.len()) {
                return ControlFlow::Break(());
            }
            self.examined.push((u, d));
            ControlFlow::Continue(())
        }
        fn examine_edge(&mut self, _: usize, _: usize, _: f64) -> ControlFlow<()> {
            self.edges += 1;
            ControlFlow::Continue(())
        }
        fn edge_relaxed(&mut self, _: usize, _: usize, _: f64, _: f64) -> ControlFlow<()> {
            self.relaxed += 1;
            ControlFlow::Continue(())
        }
        fn edge_not_relaxed(&mut self, _: usize, _: usize, _: f64) -> ControlFlow<()> {
            self.not_relaxed += 1;
            ControlFlow::Continue(())
        }
        fn finish_vertex(&mut self, _: usize, _: f64) -> ControlFlow<()> {
            self.finished += 1;
            ControlFlow::Continue(())
        }
    }

    /// Dijkstra's and A*'s visitors see every reached node discovered once and examined in order
    /// of distance (at its distance), every arc examined and relaxed or not, and can stop them.
    #[test]
    fn shortest_path_visitors() {
        use crate::trails::shortest_path_visit::{DijkstraVisit, ShortestPathBundle};
        let n = 200;
        let arcs = weighted_arcs(n, 900, 21, true);
        let g = weighted::<Directed>("visit", n, &arcs);
        let d = all_pairs(n, &arcs, true);
        let mut r = Recorder::default();
        let mut v = DijkstraVisit::new(&*g, 3, &mut r).unwrap();
        assert!(!v.stopped());
        let reached = d[3].iter().filter(|x| x.is_finite()).count();
        assert_eq!(
            (r.discovered.len(), r.examined.len(), r.finished),
            (reached, reached, reached)
        );
        assert_eq!(r.discovered[0], (3, 0., 3));
        assert!(r.examined.windows(2).all(|e| e[0].1 <= e[1].1));
        for &(u, du) in &r.examined {
            assert!(close(du, d[3][u]));
            assert_eq!(v.get_distances()[u].to_bits(), du.to_bits());
        }
        let arcs_out: usize = r.examined.iter().map(|&(u, _)| g.node_degree(u)).sum();
        assert_eq!((r.edges, r.relaxed + r.not_relaxed), (arcs_out, arcs_out));
        v.drop_cache().unwrap();

        // stopped after 10 nodes, with a bundle: the second visitor sees what the first lets by
        let mut bundle = ShortestPathBundle(
            Recorder {
                stop_after: Some(10),
                ..Default::default()
            },
            Recorder::default(),
        );
        let mut v = DijkstraVisit::new(&*g, 3, &mut bundle).unwrap();
        assert!(v.stopped());
        assert_eq!((bundle.0.examined.len(), bundle.1.examined.len()), (10, 10));
        assert_eq!(
            v.get_distances().iter().filter(|x| x.is_finite()).count(),
            bundle.0.discovered.len()
        );
        v.drop_cache().unwrap();

        // A*: to the farthest reachable node, examined in order of distance plus estimate
        let mut l = Landmarks::new(&*g, 4).unwrap();
        let t = (0..n)
            .filter(|&v| d[3][v].is_finite())
            .max_by(|&a, &b| d[3][a].total_cmp(&d[3][b]))
            .unwrap();
        let mut r = Recorder::default();
        let p = astar_visit(&*g, 3, t, &l, &mut r).unwrap().unwrap();
        assert!(close(p.distance, d[3][t]));
        assert_eq!(r.examined.last().unwrap().0, t);
        assert_eq!(r.finished, p.expanded);
        let mut r = Recorder {
            stop_after: Some(2),
            ..Default::default()
        };
        assert!(astar_visit(&*g, 3, t, &l, &mut r).unwrap().is_none());
        assert_eq!(r.examined.len(), 2);
        l.drop_cache().unwrap();
    }
}
