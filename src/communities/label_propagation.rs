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
use crate::graph::csr::Csr;
use crate::graph::*;
use crate::shared_slice::*;
use crate::trails::astar::Reversible;
use crate::utils::SpinBarrier;

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    /// thousand, and 20 found the blocks (an NMI of 0.955); more changed nothing.
    pub exploration: usize,
    /// On directed graphs, the arcs a node's label follows (on undirected ones, every edge).
    pub direction: LpaDirection,
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
    let mut remaining: Vec<usize> = (0..n).collect();
    while !remaining.is_empty() {
        let work = &remaining;
        // the nodes all of whose neighbours of higher priority are colored: no two neighbours
        let ready: Vec<Vec<usize>> = chunked(work.len(), threads, &|| (), &|begin, end, _| {
            work[begin..end]
                .iter()
                .copied()
                .filter(|&u| {
                    neighbours(u)
                        .all(|v| v == u || priority(v) < priority(u) || *shared.get(v) != NONE)
                })
                .collect::<Vec<_>>()
        });
        let ready: Vec<usize> = ready.concat();
        // each takes the least color its (colored) neighbours don't have
        chunked(
            ready.len(),
            threads,
            &Vec::<usize>::new,
            &|begin, end, taken| {
                let mut colors = shared;
                for &u in &ready[begin..end] {
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
                }
            },
        );
        let parts: Vec<Vec<usize>> = chunked(work.len(), threads, &|| (), &|begin, end, _| {
            work[begin..end]
                .iter()
                .copied()
                .filter(|&u| *shared.get(u) == NONE)
                .collect::<Vec<_>>()
        });
        remaining = parts.concat();
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
        let color_fn = g.build_cache_filename(CacheFile::LabelPropagation, Some(0))?;
        let classes = color_classes(&both, n, &color_fn, threads);
        g.cleanup_cache(CacheFile::LabelPropagation)?;
        let (order, starts) = classes?;
        let colors = starts.len() - 1;

        let mut labels = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, n.max(1), true)?;
        for (u, l) in labels.as_mut_slice().iter_mut().enumerate().take(n) {
            *l = Ix::new(u);
        }
        let pending: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(true)).collect();
        let cursors: Vec<AtomicUsize> = (0..colors).map(|_| AtomicUsize::new(0)).collect();
        // the nodes each pass moves, by its parity (see `shortest_path_tree`'s counts)
        let moved = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let moves = std::sync::Mutex::new(vec![]);
        let barrier = SpinBarrier::new(threads);
        let shared = labels.shared_slice();
        let limit = (options.tolerance * n as f64).floor() as usize;

        std::thread::scope(|s| {
            for tid in 0..threads {
                let (order, starts, pending, cursors, moved, moves, barrier) = (
                    &order, &starts, &pending, &cursors, &moved, &moves, &barrier,
                );
                let (reads, wakes) = (&reads, &wakes);
                s.spawn(move || {
                    let mut labels = shared;
                    let mut seen: Vec<(usize, f64)> = Vec::new();
                    for pass in 0..options.max_iterations {
                        let exploring = pass < options.exploration;
                        let mut mine = 0;
                        for c in 0..colors {
                            let (first, end) = (starts[c], starts[c + 1]);
                            let len = end - first;
                            // small colors: one thread (the others would mostly wait)
                            let workers = len.div_ceil(256).min(threads);
                            let chunk = (len / (workers * 4)).clamp(1, CHUNK);
                            if tid < workers {
                                loop {
                                    let at = cursors[c].fetch_add(chunk, Ordering::Relaxed);
                                    if at >= len {
                                        break;
                                    }
                                    for &u in &order[first + at..first + (at + chunk).min(len)] {
                                        if !pending[u].swap(false, Ordering::Relaxed) {
                                            continue;
                                        }
                                        seen.clear();
                                        for csr in reads.iter() {
                                            for a in csr.range(u) {
                                                let v = csr.targets[a].index();
                                                if v != u {
                                                    let w = if E::is_weighted() {
                                                        csr.weights[a].weight::<f64>()
                                                    } else {
                                                        1.
                                                    };
                                                    seen.push((labels.get(v).index(), w));
                                                }
                                            }
                                        }
                                        if seen.is_empty() {
                                            continue;
                                        }
                                        seen.sort_unstable_by_key(|p| p.0);
                                        let own = labels.get(u).index();
                                        let (mut best, mut best_w, mut own_w) =
                                            (own, f64::NEG_INFINITY, f64::NEG_INFINITY);
                                        let salt = mix(u as u64 ^ mix(pass as u64));
                                        let mut best_key = u64::MAX;
                                        let mut i = 0;
                                        while i < seen.len() {
                                            let l = seen[i].0;
                                            let mut w = 0.;
                                            while i < seen.len() && seen[i].0 == l {
                                                w += seen[i].1;
                                                i += 1;
                                            }
                                            if l == own {
                                                own_w = w;
                                            }
                                            // of equals, the least hash of the node, the
                                            // label and the pass: unbiased (the least label
                                            // would flood the graph), yet the same every run
                                            let key = mix(mix(salt ^ l as u64));
                                            if w > best_w || (w == best_w && key < best_key) {
                                                (best, best_w, best_key) = (l, w, key);
                                            }
                                        }
                                        // exploring, ties are broken by the hash, the own
                                        // label included; then, it's kept (so that it ends)
                                        if best == own || (!exploring && own_w == best_w) {
                                            continue;
                                        }
                                        *labels.get_mut(u) = Ix::new(best);
                                        mine += 1;
                                        for csr in wakes.iter() {
                                            for &v in &csr.targets[csr.range(u)] {
                                                pending[v.index()].store(true, Ordering::Relaxed);
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
            g.modularity(communities, count)?
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
