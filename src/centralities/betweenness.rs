//! Betweenness centrality: how many of the shortest paths between other nodes run through each
//! node. [`Betweenness`] computes it exactly, by Brandes' algorithm, or estimates it for large
//! graphs by KADABRA's adaptive sampling.
//!
//! * **Exact** ([`BetweennessEngine::Exact`]): Brandes' algorithm ("A faster algorithm for
//!   betweenness centrality", 2001), a search from every node (BFS, or Dijkstra's algorithm on
//!   weighted graphs) whose shortest paths' dependencies are then accumulated backward, on
//!   undirected graphs within each biconnected component: [`ExactBetweennessCentrality`]'s.
//!   Sources are shared out among the graph's threads, each with its own memory (mapped in the
//!   cache): `O(|V| |E|)` time at most, for graphs of up to millions of edges.
//! * **KADABRA** ([`BetweennessEngine::Kadabra`]): Borassi and Natale's adaptive sampling ("KADABRA
//!   is an ADaptive Algorithm for Betweenness via Random Approximation", 2016), in parallel as
//!   van der Grinten, Angriman and Meyerhenke's ("Scaling up network centrality computations",
//!   2019). Each sample is a pair of distinct nodes `(s, t)` and one of the shortest paths from
//!   `s` to `t`, drawn uniformly, whose inner nodes are counted: a node's count over the
//!   samples estimates the fraction of pairs whose shortest paths it lies on. The shortest paths
//!   are found by a balanced bidirectional BFS (the side whose frontier has fewer arcs advances),
//!   which on small-world graphs visits a tiny part of the graph; on weighted graphs, by
//!   Dijkstra's algorithm from `s`, until it reaches `t`.
//!
//!   Sampling stops once every node's estimate is within `epsilon` of its betweenness (as a
//!   fraction of the `|V| (|V| - 1)` ordered pairs) with probability `1 - delta`: after at most
//!   `ω` samples (Riondato and Kornaropoulos' bound, from an upper bound on the number of nodes
//!   on a shortest path), or earlier, once KADABRA's per-node confidence intervals are all
//!   narrow enough. With [`BetweennessOptions::top_k`], it stops once the `k` most central nodes
//!   are told apart instead (each node's interval either within `epsilon`, or disjoint from its
//!   neighbours' in the ranking). Threads draw samples in epochs, checking the stopping rule
//!   between them; sample `i` is drawn from a generator seeded by `(seed, i)` alone, so the
//!   result is the same whatever the number of threads.
//!
//!   Memory: a counter per node (4 bytes, mapped in the cache), and each thread's searches'
//!   nodes, in hash maps (the nodes they visit, not `|V|`).
//!
//! Paths follow arcs out of nodes; on a [`DiGraph`], `(s, t)` and `(t, s)` are different pairs,
//! with different paths. On weighted graphs (whose weights must be positive and finite), paths
//! are shortest by weight, and two paths are equally short when their weights' sums, in `f64`,
//! are equal (as networkx's). Parallel arcs are different paths, as in a multigraph.
//!
//! Values follow networkx's `betweenness_centrality` (`endpoints=False`): normalized by the
//! `(|V| - 1) (|V| - 2)` ordered pairs of other nodes, or, unnormalized, the number of shortest
//! paths through a node (each unordered pair counted once, on undirected graphs).
//!
//! [`DiGraph`]: ../../graph/type.DiGraph.html
//! [`ExactBetweennessCentrality`]: super::exact::ExactBetweennessCentrality

use crate::centralities::exact::{betweenness_scale, betweenness_sums, check_positive_weights};
use crate::components::weak::ConnectedComponents;
use crate::graph;
use crate::graph::csr::Csr;
use crate::graph::*;
use crate::shared_slice::*;
use crate::trails::astar::Reversible;
use crate::utils::SpinBarrier;

use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::ManuallyDrop;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

type Error = Box<dyn std::error::Error>;
type SendError = Box<dyn std::error::Error + Send + Sync>;

/// How [`Betweenness`] computes the centralities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BetweennessEngine {
    /// [`Exact`](Self::Exact) on graphs with `|V| |E|` up to [`AUTO_EXACT_WORK`],
    /// [`Kadabra`](Self::Kadabra) on larger ones.
    #[default]
    Auto,
    /// Brandes' algorithm (see the [module docs](self)).
    Exact,
    /// KADABRA's adaptive sampling (see the [module docs](self)).
    Kadabra,
}

/// [`BetweennessEngine::Auto`] computes betweenness exactly on graphs whose `|V| |E|` is at most
/// this: some seconds on a few dozen threads.
pub const AUTO_EXACT_WORK: f64 = 1e10;

/// How [`Betweenness`] runs; see [`Default`] for the defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetweennessOptions {
    pub engine: BetweennessEngine,
    /// KADABRA's error bound, on betweenness as a fraction of ordered pairs (in `(0, 1)`).
    pub epsilon: f64,
    /// KADABRA's failure probability (in `(0, 1)`).
    pub delta: f64,
    /// KADABRA stops once the `k` most central nodes are told apart (see the [module
    /// docs](self)) rather than once every estimate is within `epsilon`; with any engine,
    /// [`Betweenness::top`] lists them.
    pub top_k: Option<usize>,
    /// Normalized by the `(|V| - 1) (|V| - 2)` ordered pairs of other nodes, or not (see the
    /// [module docs](self)).
    pub normalized: bool,
    /// Seeds KADABRA's samples.
    pub seed: u64,
}

impl BetweennessOptions {
    /// Checks the options: `epsilon` and `delta` in `(0, 1)`, `top_k` positive.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [("epsilon", self.epsilon), ("delta", self.delta)] {
            if !(value > 0. && value < 1.) {
                return Err(format!("{name} must be in (0, 1), not {value}"));
            }
        }
        if self.top_k == Some(0) {
            return Err("top_k must be positive".into());
        }
        Ok(())
    }
}

impl Default for BetweennessOptions {
    /// [`BetweennessEngine::Auto`], `epsilon = 0.01` and `delta = 0.1` (NetworKit's), every
    /// node's estimate, normalized, seeded by `0x5eed`.
    fn default() -> Self {
        Self {
            engine: BetweennessEngine::Auto,
            epsilon: 0.01,
            delta: 0.1,
            top_k: None,
            normalized: true,
            seed: 0x5eed,
        }
    }
}

/// Every node's betweenness centrality (see the [module docs](self)), on any graph, weighted
/// (weights positive and finite) or not. On directed graphs, KADABRA follows arcs backward too,
/// on the transpose, which is built if it isn't yet.
pub struct Betweenness<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Dir: Directedness = Undirected,
> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    centralities: AbstractedProceduralMemoryMut<f64>,
    exact: bool,
    samples: usize,
    sample_cap: usize,
    top: Vec<usize>,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible>
    Betweenness<'a, N, E, Ix, Dir>
{
    /// Computes `g`'s betweenness with [`BetweennessOptions::default`].
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix, Dir>) -> Result<Self, Error> {
        Self::with_options(g, BetweennessOptions::default())
    }

    /// Computes `g`'s betweenness as `options` say.
    pub fn with_options(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        options: BetweennessOptions,
    ) -> Result<Self, Error> {
        options
            .validate()
            .map_err(|e| -> Error { format!("error {e}").into() })?;
        check_positive_weights(g)?;
        let n = g.size();
        let threads = g.thread_num().max(1);
        let exact = match options.engine {
            BetweennessEngine::Exact => true,
            BetweennessEngine::Kadabra => false,
            BetweennessEngine::Auto => {
                // an undirected graph's edges are stored as an arc each way
                let edges = if Dir::DIRECTED {
                    g.width()
                } else {
                    g.width() / 2
                };
                n as f64 * edges as f64 <= AUTO_EXACT_WORK
            }
        };
        let out_fn = g.build_cache_filename(CacheFile::Betweenness, None)?;
        let mut centralities = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, n.max(1), true)?;
        let out = &mut centralities.as_mut_slice()[..n];
        out.fill(0.);
        // (the weights checked above)
        let computed = if exact {
            betweenness_sums(g, out, threads).map(|()| (0, 0))
        } else {
            kadabra(g, out, threads, &options)
        };
        g.cleanup_cache(CacheFile::Betweenness)?;
        let (samples, sample_cap) = match computed {
            Ok(computed) => computed,
            Err(e) => {
                // no (partial) result left behind
                drop(centralities);
                let _ = crate::shared_slice::remove_if_exists(&out_fn);
                return Err(e);
            }
        };
        // `out` holds the ordered pairs' sum (exact) or its estimate
        let scale = betweenness_scale(n, options.normalized, Dir::DIRECTED);
        for c in out.iter_mut() {
            *c *= scale;
        }
        let top = options.top_k.map_or_else(Vec::new, |k| top_nodes(out, k));
        centralities.flush()?;
        Ok(Self {
            g,
            centralities,
            exact,
            samples,
            sample_cap,
            top,
        })
    }
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>
    Betweenness<'a, N, E, Ix, Dir>
{
    /// Every node's betweenness.
    pub fn centralities(&self) -> &[f64] {
        &self.centralities.as_slice()[..self.g.size()]
    }

    /// A node's betweenness.
    pub fn get_node_centrality(&self, u: usize) -> f64 {
        self.centralities()[u]
    }

    /// Whether the centralities are exact (Brandes') rather than KADABRA's estimates.
    pub fn is_exact(&self) -> bool {
        self.exact
    }

    /// The samples KADABRA drew (0 when exact).
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The samples KADABRA would have drawn at most, `ω` (0 when exact).
    pub fn sample_cap(&self) -> usize {
        self.sample_cap
    }

    /// With [`BetweennessOptions::top_k`], the `k` most central nodes (or all, if fewer), most
    /// central first (ties by node id); empty otherwise.
    pub fn top(&self) -> &[usize] {
        &self.top
    }

    /// Removes the centralities from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Error> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::Betweenness, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

/// The `k` largest of `values`' indices (or all, if fewer), largest first, ties by index.
fn top_nodes(values: &[f64], k: usize) -> Vec<usize> {
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (u, &v) in values.iter().enumerate() {
        // the heap's top is the least of the largest
        heap.push(Reverse((OrderedFloat(v), Reverse(u))));
        if heap.len() > k {
            heap.pop();
        }
    }
    let mut top: Vec<_> = heap.into_iter().map(|Reverse(t)| t).collect();
    top.sort_unstable_by(|a, b| b.cmp(a));
    top.into_iter().map(|(_, Reverse(u))| u).collect()
}

/// An arc's weight, as paths are measured: `1` on unweighted graphs.
#[inline(always)]
fn weight<E: graph::E>(label: &E) -> f64 {
    if E::is_weighted() {
        label.weight::<f64>()
    } else {
        1.
    }
}

/// Runs `work(tid)` on `threads` scoped threads, failing with the first error.
pub(crate) fn on_threads(
    threads: usize,
    what: &str,
    work: impl Fn(usize) -> Result<(), SendError> + Sync,
) -> Result<(), Error> {
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let work = &work;
                s.spawn(move || work(tid))
            })
            .collect();
        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .map_err(|e| -> Error { format!("error in {what} thread {tid}: {e:?}").into() })?
                .map_err(|e| -> Error { format!("error in {what} thread {tid}: {e}").into() })?;
        }
        Ok(())
    })
}

// ------------------------------------------------------------------------------------------------
// KADABRA
// ------------------------------------------------------------------------------------------------

/// SplitMix64.
#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A sample's random numbers: SplitMix64's sequence, from the seed and the sample's index.
struct SampleRng(u64);

impl SampleRng {
    fn new(seed: u64, sample: usize) -> Self {
        Self(mix(seed ^ mix(sample as u64)))
    }
    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }
    /// Uniform in `0..n`.
    #[inline(always)]
    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
    /// Uniform in `[0, 1)`.
    #[inline(always)]
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Hashes node ids by a multiplication, folded.
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

/// A node a search reached: its distance, and its number of shortest paths.
#[derive(Clone, Copy)]
struct Reached {
    dist: f64,
    sigma: f64,
}

type NodeMap = HashMap<usize, Reached, BuildHasherDefault<NodeHasher>>;

/// A thread's memory for its samples' searches, kept from one to the next.
#[derive(Default)]
struct Sampler {
    forward: NodeMap,
    backward: NodeMap,
    frontier: Vec<usize>,
    other_frontier: Vec<usize>,
    next: Vec<usize>,
    meetings: Vec<(usize, usize, f64)>,
    heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>,
    /// The sampled path's inner nodes.
    path: Vec<usize>,
}

impl Sampler {
    /// Draws a uniformly random shortest path from `s` to `t` (`out` the arcs out of nodes,
    /// `into` into them), and puts its inner nodes in `self.path` (none if `t` is unreachable).
    fn sample<Ix: graph::IndexType, E: graph::E>(
        &mut self,
        out: &Csr<Ix, E>,
        into: &Csr<Ix, E>,
        s: usize,
        t: usize,
        rng: &mut SampleRng,
    ) {
        self.path.clear();
        if E::is_weighted() {
            self.weighted(out, into, s, t, rng);
        } else {
            self.bidirectional(out, into, s, t, rng);
        }
    }

    /// Balanced bidirectional BFS: the side whose frontier has fewer arcs to follow advances a
    /// level. Before they meet, no node was reached by both, so `d(s, t) > df + db`; the first
    /// level at which a side reaches the other's nodes, every shortest path crosses exactly one
    /// of the arcs it does that by, from a node at `df` from `s` to one at `db` from `t`.
    fn bidirectional<Ix: graph::IndexType, E: graph::E>(
        &mut self,
        out: &Csr<Ix, E>,
        into: &Csr<Ix, E>,
        s: usize,
        t: usize,
        rng: &mut SampleRng,
    ) {
        self.forward.clear();
        self.backward.clear();
        self.forward.insert(
            s,
            Reached {
                dist: 0.,
                sigma: 1.,
            },
        );
        self.backward.insert(
            t,
            Reached {
                dist: 0.,
                sigma: 1.,
            },
        );
        // forward's frontier in `frontier`, backward's in `other_frontier`
        self.frontier.clear();
        self.frontier.push(s);
        self.other_frontier.clear();
        self.other_frontier.push(t);
        let (mut forward_arcs, mut backward_arcs) = (out.range(s).len(), into.range(t).len());
        let (mut df, mut db) = (0., 0.);
        loop {
            if self.frontier.is_empty() || self.other_frontier.is_empty() {
                return;
            }
            let forward = forward_arcs <= backward_arcs;
            let (csr, frontier, map, other, d, next_arcs_csr) = if forward {
                (
                    out,
                    &self.frontier,
                    &mut self.forward,
                    &self.backward,
                    df,
                    out,
                )
            } else {
                (
                    into,
                    &self.other_frontier,
                    &mut self.backward,
                    &self.forward,
                    db,
                    into,
                )
            };
            self.next.clear();
            self.meetings.clear();
            let mut arcs = 0;
            for &u in frontier {
                let su = map[&u].sigma;
                for e in csr.range(u) {
                    let v = csr.targets[e].index();
                    if let Some(r) = other.get(&v) {
                        self.meetings.push((u, v, su * r.sigma));
                        continue;
                    }
                    if !self.meetings.is_empty() {
                        continue;
                    }
                    match map.entry(v) {
                        Entry::Vacant(slot) => {
                            slot.insert(Reached {
                                dist: d + 1.,
                                sigma: su,
                            });
                            self.next.push(v);
                            arcs += next_arcs_csr.range(v).len();
                        }
                        Entry::Occupied(mut slot) => {
                            if slot.get().dist == d + 1. {
                                slot.get_mut().sigma += su;
                            }
                        }
                    }
                }
            }
            if !self.meetings.is_empty() {
                // an arc, by the shortest paths crossing it
                let total: f64 = self.meetings.iter().map(|m| m.2).sum();
                let mut r = rng.unit() * total;
                let mut chosen = self.meetings[self.meetings.len() - 1];
                for &m in &self.meetings {
                    if r < m.2 {
                        chosen = m;
                        break;
                    }
                    r -= m.2;
                }
                // `(a, b)`: `a` reached from `s`, `b` from `t`
                let (a, b) = if forward {
                    (chosen.0, chosen.1)
                } else {
                    (chosen.1, chosen.0)
                };
                walk_back(&self.forward, into, a, s, rng, &mut self.path);
                self.path.reverse();
                walk_back(&self.backward, out, b, t, rng, &mut self.path);
                return;
            }
            if forward {
                std::mem::swap(&mut self.frontier, &mut self.next);
                forward_arcs = arcs;
                df += 1.;
            } else {
                std::mem::swap(&mut self.other_frontier, &mut self.next);
                backward_arcs = arcs;
                db += 1.;
            }
        }
    }

    /// Dijkstra's algorithm from `s` until `t` is settled, then a walk back from `t`.
    fn weighted<Ix: graph::IndexType, E: graph::E>(
        &mut self,
        out: &Csr<Ix, E>,
        into: &Csr<Ix, E>,
        s: usize,
        t: usize,
        rng: &mut SampleRng,
    ) {
        let map = &mut self.forward;
        map.clear();
        map.insert(
            s,
            Reached {
                dist: 0.,
                sigma: 1.,
            },
        );
        self.heap.clear();
        self.heap.push(Reverse((OrderedFloat(0.), s)));
        let mut found = false;
        while let Some(Reverse((OrderedFloat(d), u))) = self.heap.pop() {
            let r = map[&u];
            if d > r.dist {
                continue;
            }
            if u == t {
                found = true;
                break;
            }
            for e in out.range(u) {
                let v = out.targets[e].index();
                let dv = d + weight(&out.weights[e]);
                match map.entry(v) {
                    Entry::Vacant(slot) => {
                        slot.insert(Reached {
                            dist: dv,
                            sigma: r.sigma,
                        });
                        self.heap.push(Reverse((OrderedFloat(dv), v)));
                    }
                    Entry::Occupied(mut slot) => {
                        let rv = slot.get_mut();
                        if dv < rv.dist {
                            *rv = Reached {
                                dist: dv,
                                sigma: r.sigma,
                            };
                            self.heap.push(Reverse((OrderedFloat(dv), v)));
                        } else if dv == rv.dist {
                            rv.sigma += r.sigma;
                        }
                    }
                }
            }
        }
        if found {
            // nodes closer than `t` were all settled before it: their counts are final
            walk_back(map, into, t, s, rng, &mut self.path);
            // `t` itself isn't inner
            self.path.remove(0);
        }
    }
}

/// Walks from `from` back to `to` (`map`'s search's source) along a uniformly random shortest
/// path: at each node, to a predecessor (an arc of `back`, reversing the search's), with
/// probability its share of the node's shortest paths. Pushes the nodes walked, `from` included
/// and `to` not.
fn walk_back<Ix: graph::IndexType, E: graph::E>(
    map: &NodeMap,
    back: &Csr<Ix, E>,
    from: usize,
    to: usize,
    rng: &mut SampleRng,
    path: &mut Vec<usize>,
) {
    let mut x = from;
    while x != to {
        path.push(x);
        let rx = map[&x];
        let mut r = rng.unit() * rx.sigma;
        let mut chosen = None;
        for e in back.range(x) {
            let p = back.targets[e].index();
            if let Some(rp) = map.get(&p)
                && rp.dist + weight(&back.weights[e]) == rx.dist
            {
                chosen = Some(p);
                if r < rp.sigma {
                    break;
                }
                r -= rp.sigma;
            }
        }
        // (rounding may leave `r` past the last predecessor: it's chosen)
        x = chosen.expect("a reached node has a predecessor");
    }
}

/// Riondato and Kornaropoulos' sample size for `epsilon` and `delta`, for graphs whose shortest
/// paths have at most `vertex_diameter` nodes (KADABRA's `ω`).
fn sample_cap(vertex_diameter: usize, epsilon: f64, delta: f64) -> usize {
    if vertex_diameter < 3 {
        return 0;
    }
    let log = ((vertex_diameter - 2) as f64).log2().floor();
    (0.5 / (epsilon * epsilon) * (log + 1. + (1. / delta).ln())).ceil() as usize
}

/// An upper bound on the number of nodes of `g`'s shortest paths: on undirected unweighted
/// graphs, `2 e + 1` for the eccentricity `e` of a node of each component (every other node is at
/// most `e` from it), else, and at most, the size of the largest (weakly connected) component.
///
/// Components are found by [`ConnectedComponents`] (in parallel; reused if cached, else dropped
/// after), and the eccentricities by one level-synchronous BFS from every component's smallest
/// node at once, on `threads` threads, with `files` for its levels and frontiers.
fn vertex_diameter_bound<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    threads: usize,
    files: &[String],
) -> Result<usize, Error> {
    let n = g.size();
    let cc_fn = g.build_cache_filename(CacheFile::ConnectedComponents, None)?;
    let cached = std::path::Path::new(&cc_fn).exists();
    let mut cc = ConnectedComponents::get_or_compute(g)?;
    let bound = (|| -> Result<usize, Error> {
        let largest = cc.sizes().iter().copied().max().unwrap_or(0);
        if Dir::DIRECTED || E::is_weighted() {
            return Ok(largest);
        }
        let (components, count) = (cc.components(), cc.count());
        // components are numbered in the order of their smallest nodes
        let mut roots = Vec::with_capacity(count);
        for (u, c) in components.iter().enumerate() {
            if c.index() == roots.len() {
                roots.push(u);
            }
        }
        let eccentricities = parallel_levels(g, &roots, components, count, threads, files)?;
        Ok(eccentricities
            .iter()
            .zip(cc.sizes())
            .map(|(e, &size)| (2 * e.load(Ordering::Relaxed) as usize + 1).min(size))
            .max()
            .unwrap_or(1))
    })();
    if !cached {
        cc.drop_cache()?;
    }
    bound.map(|b| b.min(n))
}

/// A level-synchronous BFS from `roots` (one per component, `components` numbering each node's)
/// on `threads` threads: each component's eccentricity from its root.
fn parallel_levels<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    roots: &[usize],
    components: &[Ix],
    count: usize,
    threads: usize,
    files: &[String],
) -> Result<Vec<AtomicU32>, Error> {
    let n = g.size();
    let csr = Csr::of(g);
    let mut levels_mem = SharedSliceMut::<u32>::abst_mem_mut(&files[0], n, true)?;
    levels_mem.as_mut_slice().fill(u32::MAX);
    let ptr = levels_mem.as_mut_slice().as_mut_ptr();
    assert_eq!(ptr as usize % std::mem::align_of::<AtomicU32>(), 0);
    // `u32`s as atomics (of the same size, and here aligned), each node claimed by one thread
    let levels = unsafe { std::slice::from_raw_parts(ptr as *const AtomicU32, n) };
    let frontiers_mem = [
        SharedSliceMut::<usize>::abst_mem_mut(&files[1], n, true)?,
        SharedSliceMut::<usize>::abst_mem_mut(&files[2], n, true)?,
    ];
    let frontiers = [
        frontiers_mem[0].shared_slice(),
        frontiers_mem[1].shared_slice(),
    ];
    {
        let mut first = frontiers[0];
        for (i, &r) in roots.iter().enumerate() {
            levels[r].store(0, Ordering::Relaxed);
            *first.get_mut(i) = r;
        }
    }
    let eccentricities: Vec<AtomicU32> = (0..count).map(|_| AtomicU32::new(0)).collect();
    let lengths = [AtomicUsize::new(roots.len()), AtomicUsize::new(0)];
    let next_chunk = AtomicUsize::new(0);
    let barrier = SpinBarrier::new(threads);
    const CHUNK: usize = 1024;
    on_threads(threads, "BFS", |tid| {
        let mut found = Vec::with_capacity(CHUNK);
        let mut level = 0u32;
        loop {
            let (current, next) = ((level & 1) as usize, 1 - (level & 1) as usize);
            let len = lengths[current].load(Ordering::Acquire);
            if len == 0 {
                return Ok(());
            }
            level += 1;
            let mut out = frontiers[next];
            let flush = |found: &mut Vec<usize>, out: &mut SharedSliceMut<usize>| {
                let at = lengths[next].fetch_add(found.len(), Ordering::Relaxed);
                for (i, &v) in found.iter().enumerate() {
                    *out.get_mut(at + i) = v;
                }
                found.clear();
            };
            loop {
                let begin = CHUNK * next_chunk.fetch_add(1, Ordering::Relaxed);
                if begin >= len {
                    break;
                }
                for i in begin..(begin + CHUNK).min(len) {
                    let u = *frontiers[current].get(i);
                    for e in csr.range(u) {
                        let v = csr.targets[e].index();
                        if levels[v].load(Ordering::Relaxed) == u32::MAX
                            && levels[v]
                                .compare_exchange(
                                    u32::MAX,
                                    level,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                        {
                            // every write at a level is the level: no need for a maximum
                            let e = &eccentricities[components[v].index()];
                            if e.load(Ordering::Relaxed) != level {
                                e.store(level, Ordering::Relaxed);
                            }
                            found.push(v);
                            if found.len() == CHUNK {
                                flush(&mut found, &mut out);
                            }
                        }
                    }
                }
            }
            if !found.is_empty() {
                flush(&mut found, &mut out);
            }
            barrier.wait();
            if tid == 0 {
                lengths[current].store(0, Ordering::Relaxed);
                next_chunk.store(0, Ordering::Relaxed);
            }
            barrier.wait();
        }
    })?;
    drop(frontiers_mem);
    drop(levels_mem);
    Ok(eccentricities)
}

/// KADABRA's bound on how far below a node's betweenness its estimate `b` may be, after `tau`
/// samples, with probability `1 - delta_l` (Borassi and Natale's `f`).
fn lower_error(b: f64, tau: f64, omega: f64, log_delta: f64) -> f64 {
    let a = omega / tau - 1. / 3.;
    (log_delta / tau * (-a + (a * a + 2. * b * omega / log_delta).sqrt())).min(b)
}

/// KADABRA's bound on how far above its estimate `b` a node's betweenness may be (`g`).
fn upper_error(b: f64, tau: f64, omega: f64, log_delta: f64) -> f64 {
    let a = omega / tau + 1. / 3.;
    (log_delta / tau * (a + (a * a + 2. * b * omega / log_delta).sqrt())).min(1. - b)
}

/// Whether KADABRA can stop after `tau` samples: every node's estimate is within `epsilon`
/// (both errors growing with the estimate, the largest count's are the largest), or, for the
/// top `k`, each of the `k + 1` largest counts' (`top`, largest first) either is, or its interval
/// is apart from its neighbours' in the ranking (the smaller counts' intervals lie below the
/// `k + 1`-th's, and their errors are smaller).
fn can_stop(
    max_count: u32,
    top: Option<&[(u32, usize)]>,
    tau: usize,
    omega: usize,
    epsilon: f64,
    log_delta: f64,
) -> bool {
    let (tau, omega) = (tau as f64, omega as f64);
    let bounds = |count: u32| {
        let b = count as f64 / tau;
        (
            b,
            lower_error(b, tau, omega, log_delta),
            upper_error(b, tau, omega, log_delta),
        )
    };
    let precise = |count| {
        let (_, l, u) = bounds(count);
        l <= epsilon && u <= epsilon
    };
    match top {
        None => precise(max_count),
        Some(top) => {
            let interval = |i: usize| {
                let (b, l, u) = bounds(top[i].0);
                (b - l, b + u)
            };
            (0..top.len()).all(|i| {
                if precise(top[i].0) {
                    return true;
                }
                let (low, high) = interval(i);
                let above = i == 0 || interval(i - 1).0 > high;
                let below = i + 1 == top.len() || interval(i + 1).1 < low;
                above && below
            })
        }
    }
}

/// KADABRA on `g`, on `threads` threads: each node's estimated sum over ordered pairs, into
/// `out`; the samples drawn and `ω`.
fn kadabra<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Reversible>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    out: &mut [f64],
    threads: usize,
    options: &BetweennessOptions,
) -> Result<(usize, usize), Error> {
    let n = g.size();
    if n < 3 {
        return Ok((0, 0));
    }
    let out_csr = Csr::of(g);
    let into_csr = Csr::of(Dir::reverse(g)?);
    let files: Vec<String> = (0..4)
        .map(|k| g.build_cache_filename(CacheFile::Betweenness, Some(k)))
        .collect::<Result<_, _>>()?;
    let diameter = vertex_diameter_bound(g, threads, &files[..3]);
    g.remove_tmp_files(CacheFile::Betweenness, 0..3)?;
    let omega = sample_cap(diameter?, options.epsilon, options.delta / 2.);
    if omega == 0 {
        return Ok((0, 0));
    }
    if omega > u32::MAX as usize {
        return Err(format!(
            "error epsilon {} needs {omega} samples, more than a node's count holds",
            options.epsilon
        )
        .into());
    }
    // `delta` is split in half: for `ω`'s bound, and for the adaptive one's `2 |V|` events
    // (each node's estimate too low, or too high)
    let log_delta = (4. * n as f64 / options.delta).ln();

    let mut counts_mem = SharedSliceMut::<u32>::abst_mem_mut(&files[3], n, true)?;
    counts_mem.as_mut_slice().fill(0);
    let ptr = counts_mem.as_mut_slice().as_mut_ptr();
    assert_eq!(ptr as usize % std::mem::align_of::<AtomicU32>(), 0);
    // `u32`s as atomics (of the same size, and here aligned), for the threads' increments
    let counts = unsafe { std::slice::from_raw_parts(ptr as *const AtomicU32, n) };

    // samples taken a few at a time; epochs grow geometrically, the stopping rule checked after
    // each (the counts of an epoch's end are the same whatever the threads)
    const CHUNK: usize = 16;
    // (independent of the threads, for the same checkpoints whatever their number)
    let first_epoch = 1000.min(omega);
    let next_sample = AtomicUsize::new(0);
    let epoch_end = AtomicUsize::new(first_epoch);
    let stop = AtomicBool::new(false);
    let barrier = SpinBarrier::new(threads);
    let max_count = AtomicU32::new(0);
    let tops: Mutex<Vec<(u32, usize)>> = Mutex::new(vec![]);
    let k = options.top_k;
    let seed = options.seed;
    const PART: usize = 1 << 16;
    let parts = AtomicUsize::new(0);

    on_threads(threads, "KADABRA", |tid| {
        let mut sampler = Sampler::default();
        loop {
            let end = epoch_end.load(Ordering::Acquire);
            let mut local_max = 0;
            loop {
                let first = next_sample.fetch_add(CHUNK, Ordering::Relaxed);
                if first >= end {
                    break;
                }
                for i in first..(first + CHUNK).min(end) {
                    let mut rng = SampleRng::new(seed, i);
                    let s = rng.below(n);
                    let mut t = rng.below(n - 1);
                    if t >= s {
                        t += 1;
                    }
                    sampler.sample(&out_csr, &into_csr, s, t, &mut rng);
                    for &v in &sampler.path {
                        local_max = local_max.max(counts[v].fetch_add(1, Ordering::Relaxed) + 1);
                    }
                }
            }
            max_count.fetch_max(local_max, Ordering::Relaxed);
            barrier.wait();
            // the top `k + 1` counts, each thread's of its parts
            if let Some(k) = k {
                let mut heap = BinaryHeap::with_capacity(k + 2);
                loop {
                    let begin = PART * parts.fetch_add(1, Ordering::Relaxed);
                    if begin >= n {
                        break;
                    }
                    for (v, c) in counts[begin..(begin + PART).min(n)].iter().enumerate() {
                        heap.push(Reverse((c.load(Ordering::Relaxed), Reverse(begin + v))));
                        if heap.len() > k + 1 {
                            heap.pop();
                        }
                    }
                }
                tops.lock()
                    .map_err(|e| -> SendError { e.to_string().into() })?
                    .extend(heap.into_iter().map(|Reverse((c, Reverse(v)))| (c, v)));
                barrier.wait();
            }
            if tid == 0 {
                let top = match k {
                    Some(k) => {
                        let mut top = std::mem::take(
                            &mut *tops
                                .lock()
                                .map_err(|e| -> SendError { e.to_string().into() })?,
                        );
                        top.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                        top.truncate(k + 1);
                        Some(top)
                    }
                    None => None,
                };
                let done = end >= omega
                    || can_stop(
                        max_count.load(Ordering::Relaxed),
                        top.as_deref(),
                        end,
                        omega,
                        options.epsilon,
                        log_delta,
                    );
                if done {
                    stop.store(true, Ordering::Relaxed);
                } else {
                    next_sample.store(end, Ordering::Relaxed);
                    parts.store(0, Ordering::Relaxed);
                    epoch_end.store((end + end / 5).min(omega), Ordering::Release);
                }
            }
            barrier.wait();
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
        }
    })?;
    let samples = epoch_end.load(Ordering::Relaxed);
    // a count's share of the samples estimates the node's share of the `n (n - 1)` ordered pairs
    let scale = n as f64 * (n - 1) as f64 / samples as f64;
    for (o, c) in out.iter_mut().zip(counts) {
        *o = c.load(Ordering::Relaxed) as f64 * scale;
    }
    drop(counts_mem);
    g.remove_tmp_files(CacheFile::Betweenness, 3..4)?;
    Ok((samples, omega))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::unique_test_dir;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_mtx(name: &str, n: usize, edges: &[(usize, usize)], symmetric: bool) -> PathBuf {
        let dir = unique_test_dir("betweenness");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).unwrap();
        let kind = if symmetric { "symmetric" } else { "general" };
        writeln!(
            f,
            "%%MatrixMarket matrix coordinate pattern {kind}\n{n} {n} {}",
            edges.len()
        )
        .unwrap();
        for &(u, v) in edges {
            let (u, v) = if symmetric {
                (u.max(v), u.min(v))
            } else {
                (u, v)
            };
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        path
    }

    /// Brandes' sums over ordered pairs by brute force: every pair's shortest paths, counted
    /// by BFS from both ends.
    fn brute_force(n: usize, arcs: &[(usize, usize)]) -> Vec<f64> {
        let mut adj = vec![vec![]; n];
        for &(u, v) in arcs {
            adj[u].push(v);
        }
        let bfs = |s: usize, adj: &Vec<Vec<usize>>| {
            let (mut dist, mut sigma) = (vec![usize::MAX; n], vec![0f64; n]);
            dist[s] = 0;
            sigma[s] = 1.;
            let mut queue = std::collections::VecDeque::from([s]);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    }
                    if dist[v] == dist[u] + 1 {
                        sigma[v] += sigma[u];
                    }
                }
            }
            (dist, sigma)
        };
        let from: Vec<_> = (0..n).map(|s| bfs(s, &adj)).collect();
        let mut b = vec![0.; n];
        for s in 0..n {
            for t in 0..n {
                let (ds, ss) = &from[s];
                if s == t || ds[t] == usize::MAX {
                    continue;
                }
                for v in 0..n {
                    let dv = &from[v];
                    if v != s
                        && v != t
                        && ds[v] != usize::MAX
                        && dv.0[t] != usize::MAX
                        && ds[v] + dv.0[t] == ds[t]
                    {
                        b[v] += ss[v] * dv.1[t] / ss[t];
                    }
                }
            }
        }
        b
    }

    fn random_arcs(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut rng = SampleRng::new(seed, 0);
        let mut arcs: Vec<_> = (0..m)
            .map(|_| (rng.below(n), rng.below(n)))
            .filter(|(u, v)| u != v)
            .collect();
        arcs.sort_unstable();
        arcs.dedup();
        arcs
    }

    fn options(engine: BetweennessEngine, normalized: bool) -> BetweennessOptions {
        BetweennessOptions {
            engine,
            normalized,
            ..BetweennessOptions::default()
        }
    }

    #[test]
    fn exact_matches_brute_force() {
        for (seed, threads) in [(1, 1), (2, 4), (3, 8)] {
            let n = 40;
            let edges = random_arcs(n, 90, seed);
            let path = write_mtx(&format!("u{seed}"), n, &edges, true);
            let mut g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                None,
                Some(threads),
            )
            .unwrap();
            let both: Vec<_> = edges.iter().flat_map(|&(u, v)| [(u, v), (v, u)]).collect();
            let want = brute_force(n, &both);
            let mut b =
                Betweenness::with_options(&g, options(BetweennessEngine::Exact, false)).unwrap();
            assert!(b.is_exact());
            for (got, want) in b.centralities().iter().zip(&want) {
                assert!((got - want / 2.).abs() < 1e-9, "{got} != {}", want / 2.);
            }
            b.drop_cache().unwrap();
            let mut b =
                Betweenness::with_options(&g, options(BetweennessEngine::Exact, true)).unwrap();
            let scale = 1. / ((n - 1) * (n - 2)) as f64;
            for (got, want) in b.centralities().iter().zip(&want) {
                assert!((got - want * scale).abs() < 1e-12);
            }
            b.drop_cache().unwrap();
            g.drop_cache().unwrap();
        }
    }

    #[test]
    fn exact_directed_matches_brute_force() {
        let n = 35;
        let arcs = random_arcs(n, 120, 7);
        let path = write_mtx("d", n, &arcs, false);
        let mut g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, None, Some(4)).unwrap();
        let want = brute_force(n, &arcs);
        let mut b =
            Betweenness::with_options(&g, options(BetweennessEngine::Exact, false)).unwrap();
        for (got, want) in b.centralities().iter().zip(&want) {
            assert!((got - want).abs() < 1e-9, "{got} != {want}");
        }
        b.drop_cache().unwrap();
        g.drop_cache().unwrap();
    }

    /// KADABRA's estimates are within `epsilon` of the exact (normalized by ordered pairs) on
    /// small graphs, the same whatever the threads.
    #[test]
    fn kadabra_is_within_epsilon_and_deterministic() {
        let n = 300;
        let edges = random_arcs(n, 900, 11);
        let path = write_mtx("k", n, &edges, true);
        let mut results = vec![];
        for threads in [1, 3, 8] {
            let mut g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                None,
                Some(threads),
            )
            .unwrap();
            let mut exact =
                Betweenness::with_options(&g, options(BetweennessEngine::Exact, true)).unwrap();
            let opts = BetweennessOptions {
                engine: BetweennessEngine::Kadabra,
                epsilon: 0.02,
                ..BetweennessOptions::default()
            };
            let mut approx = Betweenness::with_options(&g, opts).unwrap();
            assert!(!approx.is_exact());
            assert!(approx.samples() > 0 && approx.samples() <= approx.sample_cap());
            // normalized values are `n / (n - 2)` times the fraction of ordered pairs
            let bound = 0.02 * n as f64 / (n - 2) as f64;
            for (a, e) in approx.centralities().iter().zip(exact.centralities()) {
                assert!((a - e).abs() <= bound, "{a} vs {e}");
            }
            results.push((approx.samples(), approx.centralities().to_vec()));
            approx.drop_cache().unwrap();
            exact.drop_cache().unwrap();
            g.drop_cache().unwrap();
        }
        assert!(results.windows(2).all(|w| w[0] == w[1]));
    }

    /// Four dense clusters, each joined by one of its nodes to a hub: the hub, on most pairs'
    /// paths (about 0.74 of them, normalized), is told apart from the joining nodes (about 0.35)
    /// long before every estimate is within `epsilon`.
    #[test]
    fn kadabra_top_k() {
        let (clusters, size) = (4, 15);
        let hub = clusters * size;
        let mut edges = vec![];
        for c in 0..clusters {
            let base = c * size;
            for u in 0..size {
                for v in u + 1..size {
                    if (u * 7 + v * 3) % 4 != 0 {
                        edges.push((base + u, base + v));
                    }
                }
            }
            edges.push((base, hub));
        }
        let path = write_mtx("top", hub + 1, &edges, true);
        let mut g =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, None, Some(4)).unwrap();
        let opts = BetweennessOptions {
            engine: BetweennessEngine::Kadabra,
            epsilon: 0.005,
            top_k: Some(1),
            ..BetweennessOptions::default()
        };
        let mut top = Betweenness::with_options(&g, opts).unwrap();
        let mut all = Betweenness::with_options(
            &g,
            BetweennessOptions {
                top_k: None,
                ..opts
            },
        )
        .unwrap();
        assert_eq!(top.top(), &[hub]);
        assert!(all.top().is_empty());
        assert!(
            10 * top.samples() < all.samples(),
            "{} {}",
            top.samples(),
            all.samples()
        );
        top.drop_cache().unwrap();
        all.drop_cache().unwrap();
        g.drop_cache().unwrap();
    }

    /// A sparse graph with much block structure: random trees joined by a few extra edges into
    /// cycles, several components, isolated nodes.
    fn sparse_edges(n: usize, extra: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut rng = SampleRng::new(seed, 1);
        let mut edges = vec![];
        for v in 1..n {
            // a forest: most nodes hang off an earlier one
            if rng.below(10) != 0 {
                edges.push((rng.below(v), v));
            }
        }
        for _ in 0..extra {
            let (u, v) = (rng.below(n), rng.below(n));
            if u != v {
                edges.push((u.min(v), u.max(v)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    #[test]
    fn exact_by_blocks_matches_brute_force() {
        use crate::centralities::exact::ExactBetweennessCentrality;
        for (seed, threads) in [(1, 1), (2, 4), (3, 3), (4, 8), (5, 2)] {
            let n = 90;
            let edges = sparse_edges(n, 12 + 4 * seed as usize, seed);
            let path = write_mtx(&format!("blocks{seed}"), n, &edges, true);
            let mut g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                None,
                Some(threads),
            )
            .unwrap();
            let both: Vec<_> = edges.iter().flat_map(|&(u, v)| [(u, v), (v, u)]).collect();
            let want = brute_force(n, &both);
            let mut b = ExactBetweennessCentrality::new(&g, false).unwrap();
            for (u, (got, want)) in b.centralities().iter().zip(&want).enumerate() {
                assert!(
                    (got - want / 2.).abs() < 1e-9,
                    "seed {seed}, node {u}: {got} != {}",
                    want / 2.
                );
            }
            b.drop_cache().unwrap();
            g.drop_cache().unwrap();
        }
    }

    /// On weighted graphs, by blocks (as an undirected graph) and on the whole graph (as a
    /// directed one, each edge an arc each way) agree.
    #[test]
    fn exact_weighted_by_blocks_matches_whole_graph() {
        use crate::centralities::exact::ExactBetweennessCentrality;
        for seed in 1..4 {
            let n = 70;
            let edges = sparse_edges(n, 25, seed);
            let mut rng = SampleRng::new(seed, 2);
            let dir = unique_test_dir("betweenness");
            std::fs::create_dir_all(&dir).unwrap();
            let (sym, general) = (dir.join("w.mtx"), dir.join("wd.mtx"));
            let weighted: Vec<_> = edges
                .iter()
                .map(|&(u, v)| (u, v, 1 + rng.below(3)))
                .collect();
            let mut text = format!(
                "%%MatrixMarket matrix coordinate real symmetric\n{n} {n} {}\n",
                edges.len()
            );
            let mut directed = format!(
                "%%MatrixMarket matrix coordinate real general\n{n} {n} {}\n",
                2 * edges.len()
            );
            for &(u, v, w) in &weighted {
                text += &format!("{} {} {w}\n", v + 1, u + 1);
                directed += &format!("{} {} {w}\n{} {} {w}\n", u + 1, v + 1, v + 1, u + 1);
            }
            std::fs::write(&sym, text).unwrap();
            std::fs::write(&general, directed).unwrap();
            let mut g =
                GraphMemoryMap::<VoidLabel, Weighted<f64>, usize>::from_file(&sym, None, Some(4))
                    .unwrap();
            let mut d =
                DiGraph::<VoidLabel, Weighted<f64>, usize>::from_file(&general, None, Some(4))
                    .unwrap();
            let mut blocks = ExactBetweennessCentrality::new(&g, true).unwrap();
            let mut whole = ExactBetweennessCentrality::new(&d, true).unwrap();
            for (a, b) in blocks.centralities().iter().zip(whole.centralities()) {
                assert!((a - b).abs() < 1e-12, "{a} != {b}");
            }
            blocks.drop_cache().unwrap();
            whole.drop_cache().unwrap();
            g.drop_cache().unwrap();
            d.drop_cache().unwrap();
        }
    }

    /// Exact runs repeat bit for bit on as many threads (each thread's sources are fixed).
    #[test]
    fn exact_repeats_exactly() {
        use crate::centralities::exact::ExactBetweennessCentrality;
        let n = 300;
        let edges = sparse_edges(n, 400, 9);
        let path = write_mtx("repeat", n, &edges, true);
        let mut g =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, None, Some(6)).unwrap();
        let mut runs = vec![];
        for _ in 0..3 {
            let mut b = ExactBetweennessCentrality::new(&g, true).unwrap();
            runs.push(
                b.centralities()
                    .iter()
                    .map(|c| c.to_bits())
                    .collect::<Vec<_>>(),
            );
            b.drop_cache().unwrap();
        }
        assert!(runs.windows(2).all(|w| w[0] == w[1]));
        g.drop_cache().unwrap();
    }

    /// A failing run leaves no result file behind.
    #[test]
    fn failures_leave_no_result() {
        use crate::centralities::exact::ExactBetweennessCentrality;
        let dir = unique_test_dir("betweenness");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n2 1 1\n3 2 0\n",
        )
        .unwrap();
        let mut g =
            GraphMemoryMap::<VoidLabel, Weighted<f64>, usize>::from_file(&path, None, Some(2))
                .unwrap();
        assert!(ExactBetweennessCentrality::new(&g, true).is_err());
        let exact = g
            .build_cache_filename(CacheFile::ExactBetweenness, None)
            .unwrap();
        assert!(
            !std::path::Path::new(&exact).exists(),
            "{exact} left behind"
        );
        for engine in [BetweennessEngine::Exact, BetweennessEngine::Kadabra] {
            assert!(Betweenness::with_options(&g, options(engine, true)).is_err());
            let out = g
                .build_cache_filename(CacheFile::Betweenness, None)
                .unwrap();
            assert!(!std::path::Path::new(&out).exists(), "{out} left behind");
        }
        g.drop_cache().unwrap();
    }

    #[test]
    fn invalid_options_fail() {
        let path = write_mtx("bad", 3, &[(0, 1), (1, 2)], true);
        let mut g =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, None, Some(1)).unwrap();
        for opts in [
            BetweennessOptions {
                epsilon: 0.,
                ..BetweennessOptions::default()
            },
            BetweennessOptions {
                delta: 1.,
                ..BetweennessOptions::default()
            },
            BetweennessOptions {
                top_k: Some(0),
                ..BetweennessOptions::default()
            },
        ] {
            assert!(Betweenness::with_options(&g, opts).is_err());
        }
        // a path: the middle node is on the one path between the others
        let mut b = Betweenness::new(&g).unwrap();
        assert_eq!(b.centralities(), &[0., 1., 0.]);
        b.drop_cache().unwrap();
        g.drop_cache().unwrap();
    }
}
