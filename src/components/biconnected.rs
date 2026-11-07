use crate::graph;
use crate::graph::cache::AccumulatorAtomic;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicU8, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;

/// The biconnected components, articulation points, bridges and 2-edge-connected components of
/// an undirected graph ([`Graph`]).
///
/// * A biconnected component is a maximal set of nodes linked by at least two paths sharing no
///   node but their ends (or a pair linked by a bridge): every edge is in exactly one, and two
///   share at most one node, an articulation point. Isolated nodes are in none.
/// * An articulation point is a node whose removal disconnects some of its neighbours.
/// * A bridge is an edge whose removal disconnects its ends. One of several parallel edges never
///   is; self-loops never are.
/// * A 2-edge-connected component is a maximal set of nodes linked by at least two paths sharing
///   no edge: the connected components of the graph without its bridges.
///
/// Computed with *FAST-BCC*, described in ["Provably Fast and Space-Efficient Parallel
/// Biconnectivity"](https://doi.org/10.48550/arXiv.2301.01356) by Dong X., Wang L., Gu Y. and
/// Sun Y., which unlike the breadth-first search based algorithms doesn't slow down on graphs
/// of large diameter, and streams through the edges, keeping only per node values: with any
/// spanning forest of the graph, each node's subtree is an interval of its preorder, from which
/// edges going out of subtrees are found. A tree edge from `p` to its child `v` is a *fence* if
/// no edge from `v`'s subtree goes out of `p`'s, and a bridge if none goes out of `v`'s. The
/// biconnected components are the connected components of the graph without fences and edges
/// between a node and one of its ancestors, each with the parent of its top node (its *head*),
/// and articulation points are heads. The edges are read in parallel, three times (spanning
/// forest, biconnected and 2-edge-connected components) plus once (subtrees' edges going out).
/// The trees are rooted in parallel too, by ranking their Euler tours (walks between sampled
/// arcs), and each subtree's edges going out are found by range minimum and maximum queries over
/// its preorder interval (a sparse table over blocks of nodes).
///
/// Components, of either kind, are numbered by their smallest node. Every result but which of
/// its components an articulation point is listed in as a member (see [`component`]) is the
/// same whatever the threads.
///
/// [`component`]: Self::component
#[allow(dead_code)]
#[derive(Debug)]
pub struct Biconnectivity<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph whose components are computed.
    g: &'a Graph<N, E, Ix>,
    /// Each node's biconnected component (`Ix::max()` for none: isolated nodes, and the roots of
    /// the spanning forest, which only head components). Every node is in this component, and
    /// in those it heads.
    components: AbstractedProceduralMemoryMut<Ix>,
    /// Each biconnected component's head: its node outside of `components`.
    heads: AbstractedProceduralMemoryMut<Ix>,
    /// Whether each node is an articulation point.
    articulation_points: AbstractedProceduralMemoryMut<bool>,
    /// The bridges' ends (the smaller first), sorted.
    bridges: AbstractedProceduralMemoryMut<[Ix; 2]>,
    /// Each node's 2-edge-connected component.
    two_edge_components: AbstractedProceduralMemoryMut<Ix>,
    count: usize,
    bridge_count: usize,
    two_edge_count: usize,
}

/// Runs `f(begin, end)` over `threads` ranges of `0..n`.
fn in_parallel(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    let load = n.div_ceil(threads);
    thread::scope(|scope| {
        for tid in 0..threads {
            let begin = std::cmp::min(tid * load, n);
            let end = std::cmp::min(begin + load, n);
            scope.spawn(move |_| f(begin, end));
        }
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// `f` over `0..n` split into `threads` ranges, in parallel: its results, by range.
fn in_parallel_collect<T: Send>(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let load = n.div_ceil(threads);
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let begin = std::cmp::min(tid * load, n);
                let end = std::cmp::min(begin + load, n);
                scope.spawn(move |_| f(begin, end))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// Lock-free union-find over the nodes, as the connected components': roots only link to smaller
/// roots, and finds halve paths, so a component's root is its smallest node.
struct UnionFind<Ix: graph::IndexType> {
    parents: SharedSliceMut<Ix::Atomic>,
}

impl<Ix: graph::IndexType> UnionFind<Ix> {
    /// Every node of `0..n` its own root.
    fn reset(&self, threads: usize, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        let parents = &self.parents;
        in_parallel(threads, n, &|begin, end| {
            for u in begin..end {
                let _ = parents.get(u).checked_store(u, Ordering::Relaxed);
            }
        })
    }

    fn find(&self, mut x: usize) -> usize {
        loop {
            let p = self.parents.get(x).load_widened(Ordering::Acquire);
            if p == x {
                return x;
            }
            let grandparent = self.parents.get(p).load_widened(Ordering::Acquire);
            if grandparent != p {
                self.parents
                    .get(x)
                    .compare_exchange_widened(p, grandparent, Ordering::AcqRel);
            }
            x = p;
        }
    }

    /// Unites the components of `u` and `v`, returning whether they were different.
    fn union(&self, u: usize, v: usize) -> bool {
        loop {
            let (ru, rv) = (self.find(u), self.find(v));
            if ru == rv {
                return false;
            }
            let (high, low) = if ru > rv { (ru, rv) } else { (rv, ru) };
            if self
                .parents
                .get(high)
                .compare_exchange_widened(high, low, Ordering::AcqRel)
            {
                return true;
            }
        }
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Biconnectivity<'a, N, E, Ix> {
    /// Computes the biconnected components, articulation points, bridges and 2-edge-connected
    /// components of the graph.
    ///
    /// The results are stored in memory (in memmapped files).
    pub fn new(g: &'a Graph<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let result = Self::compute(g);
        g.cleanup_cache(CacheFile::Biconnectivity)?;
        if result.is_err() {
            for seq in 0..5 {
                let file = g.build_pers_cache_filename(CacheFile::Biconnectivity, Some(seq))?;
                let _ = crate::shared_slice::remove_if_exists(file);
            }
        }
        result
    }

    /// Returns the number of biconnected components.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the biconnected component a node is in (`None` for isolated nodes, and nodes in
    /// components they all head): an articulation point is also in those it heads (see
    /// [`head`](Self::head)), which of them it's listed in here depending on the threads.
    pub fn component(&self, node: usize) -> Option<usize> {
        assert!(node < self.g.size());
        let c = *self.components.get(node);
        (c != <Ix as graph::IndexType>::max()).then(|| c.index())
    }

    /// Returns each node's biconnected component (`Ix::max()` for none), as [`component`].
    ///
    /// [`component`]: Self::component
    pub fn components(&self) -> &[Ix] {
        self.components.as_slice()
    }

    /// Returns a biconnected component's head: its one node not listed in it by
    /// [`components`](Self::components), an articulation point unless the component is the
    /// only one it's in.
    pub fn head(&self, component: usize) -> usize {
        assert!(component < self.count);
        self.heads.get(component).index()
    }

    /// Returns each biconnected component's head, as [`head`](Self::head).
    pub fn heads(&self) -> &[Ix] {
        &self.heads.as_slice()[..self.count]
    }

    /// Returns the nodes of a biconnected component, sorted (in `O(|V|)`).
    pub fn component_nodes(&self, component: usize) -> Vec<usize> {
        let head = self.head(component);
        let mut nodes: Vec<usize> = (0..self.g.size())
            .filter(|&u| u == head || self.component(u) == Some(component))
            .collect();
        nodes.dedup();
        nodes
    }

    /// Returns whether a node is an articulation point.
    pub fn is_articulation_point(&self, node: usize) -> bool {
        assert!(node < self.g.size());
        *self.articulation_points.get(node)
    }

    /// Returns the articulation points, sorted (in `O(|V|)`).
    pub fn articulation_points(&self) -> Vec<usize> {
        let a = self.articulation_points.as_slice();
        (0..a.len()).filter(|&u| a[u]).collect()
    }

    /// Returns the bridges' ends (the smaller first), sorted.
    pub fn bridges(&self) -> &[[Ix; 2]] {
        &self.bridges.as_slice()[..self.bridge_count]
    }

    /// Returns the number of 2-edge-connected components.
    pub fn two_edge_count(&self) -> usize {
        self.two_edge_count
    }

    /// Returns each node's 2-edge-connected component.
    pub fn two_edge_components(&self) -> &[Ix] {
        self.two_edge_components.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        this.g.cleanup_cache(CacheFile::Biconnectivity)?;
        for seq in 0..5 {
            let file = this
                .g
                .build_pers_cache_filename(CacheFile::Biconnectivity, Some(seq))?;
            crate::shared_slice::remove_if_exists(file)?;
        }
        Ok(())
    }

    fn compute(g: &'a Graph<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let n = g.size();
        let threads = g.thread_num();
        let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
        let neighbours = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());
        let row = |u: usize| offsets.get(u).index()..offsets.get(u + 1).index();
        let none = <Ix as graph::IndexType>::max();
        let scratch = |seq: usize| g.build_cache_filename(CacheFile::Biconnectivity, Some(seq));
        let result = |seq: usize| g.build_pers_cache_filename(CacheFile::Biconnectivity, Some(seq));

        // 1. a spanning forest: the edges by which union-find unites components (each edge, of
        //    the two ways it's stored, from its larger end, self-loops aside), reserved in
        //    batches (a shared counter, taken at every one, would be contended for)
        const BATCH: usize = 1024;
        let parents_mem = SharedSliceMut::<Ix::Atomic>::abst_mem_mut(&scratch(0)?, n.max(1), true)?;
        let uf = UnionFind::<Ix> {
            parents: parents_mem.shared_slice(),
        };
        uf.reset(threads, n)?;
        let tree_mem = SharedSliceMut::<[Ix; 2]>::abst_mem_mut(&scratch(1)?, n.max(1), true)?;
        let tree_count = AtomicUsize::new(0);
        {
            let tree = tree_mem.shared_slice();
            in_parallel(threads, n, &|begin, end| {
                let mut tree = tree;
                let mut local: Vec<[Ix; 2]> = Vec::with_capacity(BATCH);
                let mut flush = |local: &mut Vec<[Ix; 2]>| {
                    let at = tree_count.fetch_add(local.len(), Ordering::Relaxed);
                    for (k, e) in local.drain(..).enumerate() {
                        *tree.get_mut(at + k) = e;
                    }
                };
                for u in begin..end {
                    for e in row(u) {
                        let v = neighbours.get(e).index();
                        if v < u && uf.union(u, v) {
                            local.push([Ix::new(u), Ix::new(v)]);
                            if local.len() == BATCH {
                                flush(&mut local);
                            }
                        }
                    }
                }
                flush(&mut local);
            })?;
        }
        let tree_edges = tree_mem.shared_slice();
        let edge_count = tree_count.load(Ordering::Relaxed);
        let arcs = 2 * edge_count;

        // each node's tree's root (its smallest node), and the trees, by root
        let root_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(2)?, n.max(1), true)?;
        let root_of = root_mem.shared_slice();
        in_parallel(threads, n, &|begin, end| {
            let mut root_of = root_of;
            for u in begin..end {
                *root_of.get_mut(u) = Ix::new(uf.find(u));
            }
        })?;
        let roots: Vec<usize> = in_parallel_collect(threads, n, &|begin, end| {
            (begin..end)
                .filter(|&u| root_of.get(u).index() == u)
                .collect::<Vec<_>>()
        })?
        .concat();
        let tree_mem_id = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(3)?, n.max(1), true)?;
        let tree_id = tree_mem_id.shared_slice();
        in_parallel(threads, roots.len(), &|begin, end| {
            let mut tree_id = tree_id;
            for (k, &r) in roots.iter().enumerate().take(end).skip(begin) {
                *tree_id.get_mut(r) = Ix::new(k);
            }
        })?;

        // 2. the trees rooted, and each node's subtree as the interval of preorder
        //    `first..=last`, from Euler tours: each tree edge, both ways, is an arc in the list of
        //    the node it leaves, knowing its twin, and an arc to `v` is followed by the arc after
        //    its twin in `v`'s list (`v`'s first after its last). Ranks in the tour, computed in
        //    parallel by walks between sampled arcs, give everything else
        let degree_mem = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&scratch(4)?, n.max(1), true)?;
        let start_mem = SharedSliceMut::<usize>::abst_mem_mut(&scratch(5)?, n + 1, true)?;
        let head_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(6)?, arcs.max(1), true)?;
        let twin_mem = SharedSliceMut::<usize>::abst_mem_mut(&scratch(7)?, arcs.max(1), true)?;
        let rank_mem = SharedSliceMut::<usize>::abst_mem_mut(&scratch(8)?, arcs.max(1), true)?;
        let degree = degree_mem.shared_slice();
        let (head, twin, rank) = (
            head_mem.shared_slice(),
            twin_mem.shared_slice(),
            rank_mem.shared_slice(),
        );
        let mut start = start_mem.shared_slice();
        in_parallel(threads, edge_count, &|begin, end| {
            for i in begin..end {
                let [u, v] = *tree_edges.get(i);
                degree.get(u.index()).fetch_add(1, Ordering::Relaxed);
                degree.get(v.index()).fetch_add(1, Ordering::Relaxed);
            }
        })?;
        // (a prefix sum by range: the ranges' sums, then each range's, from the ones before it)
        let sums = in_parallel_collect(threads, n, &|begin, end| {
            (begin..end)
                .map(|u| degree.get(u).load(Ordering::Relaxed))
                .sum::<usize>()
        })?;
        let mut range_starts = Vec::with_capacity(threads);
        let mut at = 0;
        for sum in sums {
            range_starts.push(at);
            at += sum;
        }
        *start.get_mut(n) = at;
        let load = n.div_ceil(threads);
        in_parallel(threads, n, &|begin, end| {
            let mut start = start;
            let mut at = if begin < end {
                range_starts[begin / load]
            } else {
                0
            };
            for u in begin..end {
                let d = degree.get(u).load(Ordering::Relaxed);
                *start.get_mut(u) = at;
                degree.get(u).store(at, Ordering::Relaxed);
                at += d;
            }
        })?;
        in_parallel(threads, edge_count, &|begin, end| {
            let (mut head, mut twin) = (head, twin);
            for e in begin..end {
                let [u, v] = *tree_edges.get(e);
                let i = degree.get(u.index()).fetch_add(1, Ordering::Relaxed);
                let j = degree.get(v.index()).fetch_add(1, Ordering::Relaxed);
                *head.get_mut(i) = v;
                *head.get_mut(j) = u;
                *twin.get_mut(i) = j;
                *twin.get_mut(j) = i;
            }
        })?;
        drop((tree_mem, degree_mem));
        // the arc after each in its tour: an arc into `v` is followed by the arc after its twin
        // in `v`'s list (`v`'s first after its last), but for the tour's last, the arc into its
        // root followed by the root's first (`END`). Arcs followed by a sampled arc, where walks
        // along tours stop, are flagged (`SAMPLED`): every `SAMPLE`th, and each tour's first
        const SAMPLE: usize = 64;
        const END: usize = usize::MAX;
        const SAMPLED: usize = 1 << (usize::BITS - 1);
        let next_mem = SharedSliceMut::<usize>::abst_mem_mut(&scratch(9)?, arcs.max(1), true)?;
        let next = next_mem.shared_slice();
        in_parallel(threads, n, &|begin, end| {
            let mut next = next;
            for v in begin..end {
                let (s, e) = (*start.get(v), *start.get(v + 1));
                let is_root = root_of.get(v).index() == v;
                for j in s..e {
                    let i = *twin.get(if j == s { e - 1 } else { j - 1 });
                    *next.get_mut(i) = if is_root && j == s {
                        END
                    } else if j % SAMPLE == 0 {
                        j | SAMPLED
                    } else {
                        j
                    };
                }
            }
        })?;
        // samples: the regular ones, then each tour's first (if not regular already)
        let regular = arcs.div_ceil(SAMPLE);
        let samples = regular + roots.len();
        let sample_of = |j: usize| {
            if j % SAMPLE == 0 {
                j / SAMPLE
            } else {
                // a tour's first, leaving its root
                regular + tree_id.get(head.get(*twin.get(j)).index()).index()
            }
        };
        let sample_arc = |s: usize| -> Option<usize> {
            if s < regular {
                Some(s * SAMPLE)
            } else {
                let r = roots[s - regular];
                let a = *start.get(r);
                // trees without edges have no tour
                (a < *start.get(r + 1) && a % SAMPLE != 0).then_some(a)
            }
        };
        // each tour's first sample
        let tours = |t: usize| -> Option<usize> {
            let r = roots[t];
            let a = *start.get(r);
            (a < *start.get(r + 1)).then(|| {
                if a % SAMPLE == 0 {
                    a / SAMPLE
                } else {
                    regular + t
                }
            })
        };
        // walks from each sample to the next (the following sample, and the arcs walked),
        // then the arcs before each sample in its tour, and each tour's arcs
        let walk_mem =
            SharedSliceMut::<[usize; 2]>::abst_mem_mut(&scratch(10)?, samples.max(1), true)?;
        let walks = walk_mem.shared_slice();
        // (each arc walked, its successor no longer needed, keeps its walk's sample instead, and
        // its rank in the walk)
        in_parallel(threads, samples, &|begin, end| {
            let (mut walks, mut next, mut rank) = (walks, next, rank);
            for s in begin..end {
                let Some(mut i) = sample_arc(s) else { continue };
                let mut len = 0;
                let following = loop {
                    let j = *next.get(i);
                    *next.get_mut(i) = s;
                    *rank.get_mut(i) = len;
                    len += 1;
                    if j == END {
                        break END;
                    } else if j & SAMPLED != 0 {
                        break sample_of(j & !SAMPLED);
                    }
                    i = j;
                };
                *walks.get_mut(s) = [following, len];
            }
        })?;
        let totals: Vec<AtomicUsize> = (0..roots.len()).map(|_| AtomicUsize::new(0)).collect();
        let offset_mem =
            SharedSliceMut::<usize>::abst_mem_mut(&scratch(11)?, samples.max(1), true)?;
        in_parallel(threads, roots.len(), &|begin, end| {
            let mut offsets = offset_mem.shared_slice();
            for t in begin..end {
                let Some(mut s) = tours(t) else { continue };
                let mut at = 0;
                loop {
                    *offsets.get_mut(s) = at;
                    let [following, len] = *walks.get(s);
                    at += len;
                    if following == END {
                        break;
                    }
                    s = following;
                }
                totals[t].store(at, Ordering::Relaxed);
            }
        })?;
        // each arc's rank in its tour: its walk's sample's, plus its rank in the walk
        in_parallel(threads, arcs, &|begin, end| {
            let mut rank = rank;
            let offsets = offset_mem.shared_slice();
            for i in begin..end {
                *rank.get_mut(i) += *offsets.get(*next.get(i));
            }
        })?;
        drop((next_mem, walk_mem, offset_mem));

        // positions: the trees' tours one after the other, each after its root's; a node's is
        // its parent's arc into it's (the arc's going down: ranked before its twin). Preorder
        // numbers are the number of nodes at positions before a node's, from a bitmap of them,
        // and the last of a subtree's the number at positions up to its node's arc going up
        let mut base = Vec::with_capacity(roots.len());
        let mut positions = 0;
        for total in &totals {
            base.push(positions);
            positions += total.load(Ordering::Relaxed) + 1;
        }
        const W: usize = usize::BITS as usize;
        let words = positions.div_ceil(W);
        let bits_mem =
            SharedSliceMut::<AtomicUsize>::abst_mem_mut(&scratch(12)?, words.max(1), true)?;
        let bits = bits_mem.shared_slice();
        let tree_of = |u: usize| tree_id.get(root_of.get(u).index()).index();
        in_parallel(threads, n, &|begin, end| {
            for u in begin..end {
                let b = base[tree_of(u)];
                if root_of.get(u).index() == u {
                    bits.get(b / W).fetch_or(1 << (b % W), Ordering::Relaxed);
                }
                for i in *start.get(u)..*start.get(u + 1) {
                    let r = *rank.get(i);
                    if r < *rank.get(*twin.get(i)) {
                        let p = b + r + 1;
                        bits.get(p / W).fetch_or(1 << (p % W), Ordering::Relaxed);
                    }
                }
            }
        })?;
        let before_mem = SharedSliceMut::<usize>::abst_mem_mut(&scratch(13)?, words.max(1), true)?;
        let mut before = before_mem.shared_slice();
        let mut nodes = 0;
        for w in 0..words {
            *before.get_mut(w) = nodes;
            nodes += bits.get(w).load(Ordering::Relaxed).count_ones() as usize;
        }
        // the nodes at positions before `p`
        let preorder = |p: usize| {
            let mask = (1usize << (p % W)) - 1;
            *before.get(p / W)
                + (bits.get(p / W).load(Ordering::Relaxed) & mask).count_ones() as usize
        };
        let parent_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(14)?, n.max(1), true)?;
        let first_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(15)?, n.max(1), true)?;
        let last_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(16)?, n.max(1), true)?;
        let (parent, first, last) = (
            parent_mem.shared_slice(),
            first_mem.shared_slice(),
            last_mem.shared_slice(),
        );
        in_parallel(threads, n, &|begin, end| {
            let (mut parent, mut first, mut last) = (parent, first, last);
            for u in begin..end {
                let t = tree_of(u);
                let b = base[t];
                if root_of.get(u).index() == u {
                    let f = preorder(b);
                    *parent.get_mut(u) = none;
                    *first.get_mut(u) = Ix::new(f);
                    *last.get_mut(u) = Ix::new(f + totals[t].load(Ordering::Relaxed) / 2);
                }
                for i in *start.get(u)..*start.get(u + 1) {
                    let (r, rt) = (*rank.get(i), *rank.get(*twin.get(i)));
                    if r < rt {
                        let v = head.get(i).index();
                        *parent.get_mut(v) = Ix::new(u);
                        *first.get_mut(v) = Ix::new(preorder(b + r + 1));
                        *last.get_mut(v) = Ix::new(preorder(b + rt + 1) - 1);
                    }
                }
            }
        })?;
        drop((
            head_mem, twin_mem, rank_mem, start_mem, root_mem, bits_mem, before_mem,
        ));
        drop(tree_mem_id);

        // tree edges: from a node to its parent, or child, the first of their parallel edges in
        // its row (sorted)
        let is_tree_edge = |u: usize, v: usize, e: usize| {
            v != u
                && ((*parent.get(v) != none && parent.get(v).index() == u)
                    || (*parent.get(u) != none && parent.get(u).index() == v))
                && (e == row(u).start || neighbours.get(e - 1).index() != v)
        };

        // 3. each subtree's edges going out, as the smallest and largest preorder of the nodes of
        //    the subtree and their neighbours by non-tree edges: each node's own (in preorder),
        //    and their minimum and maximum over its subtree's interval, by a sparse table over
        //    blocks of `BLOCK` nodes, with minimums and maximums from blocks' starts and to their
        //    ends
        const BLOCK: usize = 64;
        let own_low_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(17)?, n.max(1), true)?;
        let own_high_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(18)?, n.max(1), true)?;
        let (own_low, own_high) = (own_low_mem.shared_slice(), own_high_mem.shared_slice());
        in_parallel(threads, n, &|begin, end| {
            let (mut own_low, mut own_high) = (own_low, own_high);
            for u in begin..end {
                let fu = first.get(u).index();
                let (mut lo, mut hi) = (fu, fu);
                let p = parent.get(u);
                let mut skipped = *p == none;
                for e in row(u) {
                    let v = neighbours.get(e).index();
                    // the tree edge to its parent (its children are in its subtree anyway)
                    if !skipped && v == p.index() {
                        skipped = true;
                        continue;
                    }
                    let fv = first.get(v).index();
                    lo = lo.min(fv);
                    hi = hi.max(fv);
                }
                *own_low.get_mut(fu) = Ix::new(lo);
                *own_high.get_mut(fu) = Ix::new(hi);
            }
        })?;
        let blocks = n.div_ceil(BLOCK);
        let from_start_low = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(19)?, n.max(1), true)?;
        let from_start_high = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(20)?, n.max(1), true)?;
        let to_end_low = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(21)?, n.max(1), true)?;
        let to_end_high = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(22)?, n.max(1), true)?;
        let (fsl, fsh, tel, teh) = (
            from_start_low.shared_slice(),
            from_start_high.shared_slice(),
            to_end_low.shared_slice(),
            to_end_high.shared_slice(),
        );
        let mut table_low: Vec<Vec<Ix>> = vec![vec![Ix::new(0); blocks]];
        let mut table_high: Vec<Vec<Ix>> = vec![vec![Ix::new(0); blocks]];
        {
            let (bl, bh) = (
                SharedSliceMut::<Ix>::new(table_low[0].as_mut_ptr(), blocks),
                SharedSliceMut::<Ix>::new(table_high[0].as_mut_ptr(), blocks),
            );
            in_parallel(threads, blocks, &|begin, end| {
                let (mut fsl, mut fsh, mut tel, mut teh, mut bl, mut bh) =
                    (fsl, fsh, tel, teh, bl, bh);
                for b in begin..end {
                    let (lo_end, hi_end) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
                    let (mut lo, mut hi) = (usize::MAX, 0);
                    for i in lo_end..hi_end {
                        lo = lo.min(own_low.get(i).index());
                        hi = hi.max(own_high.get(i).index());
                        *fsl.get_mut(i) = Ix::new(lo);
                        *fsh.get_mut(i) = Ix::new(hi);
                    }
                    *bl.get_mut(b) = Ix::new(lo);
                    *bh.get_mut(b) = Ix::new(hi);
                    let (mut lo, mut hi) = (usize::MAX, 0);
                    for i in (lo_end..hi_end).rev() {
                        lo = lo.min(own_low.get(i).index());
                        hi = hi.max(own_high.get(i).index());
                        *tel.get_mut(i) = Ix::new(lo);
                        *teh.get_mut(i) = Ix::new(hi);
                    }
                }
            })?;
        }
        let mut span = 1;
        while 2 * span <= blocks {
            let (prev_low, prev_high) = (
                &table_low[table_low.len() - 1],
                &table_high[table_high.len() - 1],
            );
            let len = blocks - 2 * span + 1;
            let (mut next_low, mut next_high) = (vec![Ix::new(0); len], vec![Ix::new(0); len]);
            {
                let (nl, nh) = (
                    SharedSliceMut::<Ix>::new(next_low.as_mut_ptr(), len),
                    SharedSliceMut::<Ix>::new(next_high.as_mut_ptr(), len),
                );
                in_parallel(threads, len, &|begin, end| {
                    let (mut nl, mut nh) = (nl, nh);
                    for b in begin..end {
                        *nl.get_mut(b) =
                            Ix::new(prev_low[b].index().min(prev_low[b + span].index()));
                        *nh.get_mut(b) =
                            Ix::new(prev_high[b].index().max(prev_high[b + span].index()));
                    }
                })?;
            }
            table_low.push(next_low);
            table_high.push(next_high);
            span *= 2;
        }
        // the minimum and maximum over the preorder interval `l..=r`
        let range = |l: usize, r: usize| -> (usize, usize) {
            let (bl, br) = (l / BLOCK, r / BLOCK);
            if bl == br {
                let (mut lo, mut hi) = (usize::MAX, 0);
                for i in l..=r {
                    lo = lo.min(own_low.get(i).index());
                    hi = hi.max(own_high.get(i).index());
                }
                return (lo, hi);
            }
            let mut lo = tel.get(l).index().min(fsl.get(r).index());
            let mut hi = teh.get(l).index().max(fsh.get(r).index());
            if bl + 1 < br {
                let (a, b) = (bl + 1, br - 1);
                let k = (b - a + 1).ilog2() as usize;
                let w = 1 << k;
                lo = lo
                    .min(table_low[k][a].index())
                    .min(table_low[k][b + 1 - w].index());
                hi = hi
                    .max(table_high[k][a].index())
                    .max(table_high[k][b + 1 - w].index());
            }
            (lo, hi)
        };
        let low_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(23)?, n.max(1), true)?;
        let high_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(24)?, n.max(1), true)?;
        let (low, high) = (low_mem.shared_slice(), high_mem.shared_slice());
        in_parallel(threads, n, &|begin, end| {
            let (mut low, mut high) = (low, high);
            for v in begin..end {
                let (lo, hi) = range(first.get(v).index(), last.get(v).index());
                *low.get_mut(v) = Ix::new(lo);
                *high.get_mut(v) = Ix::new(hi);
            }
        })?;
        drop((table_low, table_high));
        drop((
            own_low_mem,
            own_high_mem,
            from_start_low,
            from_start_high,
            to_end_low,
            to_end_high,
        ));
        // whether no edge of `c`'s subtree goes out of `of`'s
        let kept_in = |c: usize, of: usize| {
            first.get(of).index() <= low.get(c).index()
                && high.get(c).index() <= last.get(of).index()
        };
        let is_ancestor = |a: usize, d: usize| {
            first.get(a).index() <= first.get(d).index()
                && first.get(d).index() <= last.get(a).index()
        };

        // bridges: tree edges from a node's parent to it with no edge of its subtree out of it
        let bridge_mem = SharedSliceMut::<bool>::abst_mem_mut(&scratch(25)?, n.max(1), true)?;
        let is_bridge = bridge_mem.shared_slice();
        let bridge_count = AtomicUsize::new(0);
        in_parallel(threads, n, &|begin, end| {
            let mut is_bridge = is_bridge;
            let mut found = 0;
            for v in begin..end {
                let bridge = *parent.get(v) != none && kept_in(v, v);
                *is_bridge.get_mut(v) = bridge;
                found += bridge as usize;
            }
            bridge_count.fetch_add(found, Ordering::Relaxed);
        })?;
        let mut bridges = SharedSliceMut::<[Ix; 2]>::abst_mem_mut(
            &result(3)?,
            bridge_count.load(Ordering::Relaxed).max(1),
            true,
        )?;
        let bridge_count = bridge_count.load(Ordering::Relaxed);
        {
            // (sorted by range, in parallel, then merged: the stable sort merges sorted runs)
            let mut found = in_parallel_collect(threads, n, &|begin, end| {
                let mut found: Vec<[Ix; 2]> = (begin..end)
                    .filter(|&v| *is_bridge.get(v))
                    .map(|v| {
                        let p = parent.get(v).index();
                        [Ix::new(p.min(v)), Ix::new(p.max(v))]
                    })
                    .collect();
                found.sort_unstable();
                found
            })?
            .concat();
            found.sort();
            bridges.as_mut_slice()[..bridge_count].copy_from_slice(&found);
        }

        // 4. biconnected components: the connected components of the graph without fences (tree
        //    edges from `p` to `c` with no edge of `c`'s subtree out of `p`'s) and edges between
        //    a node and one of its ancestors. Roots are alone (their children's edges are fences)
        uf.reset(threads, n)?;
        in_parallel(threads, n, &|begin, end| {
            for u in begin..end {
                for e in row(u) {
                    let v = neighbours.get(e).index();
                    if v >= u {
                        continue;
                    }
                    let skeleton = if is_tree_edge(u, v, e) {
                        let (p, c) = if parent.get(v).index() == u && *parent.get(v) != none {
                            (u, v)
                        } else {
                            (v, u)
                        };
                        !kept_in(c, p)
                    } else {
                        !is_ancestor(u, v) && !is_ancestor(v, u)
                    };
                    if skeleton {
                        uf.union(u, v);
                    }
                }
            }
        })?;

        // each component's head: the parent of its top node, the one node of it whose parent is
        // in another (by component root: its smallest node)
        let head_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(26)?, n.max(1), true)?;
        {
            let heads = head_mem.shared_slice();
            in_parallel(threads, n, &|begin, end| {
                let mut heads = heads;
                for u in begin..end {
                    if *parent.get(u) != none {
                        let p = parent.get(u).index();
                        let r = uf.find(u);
                        if uf.find(p) != r {
                            *heads.get_mut(r) = Ix::new(p);
                        }
                    }
                }
            })?;
        }
        let heads_by_root = head_mem.shared_slice();
        // components numbered by their smallest node, their head included (ties, between
        // components of a common head, broken by their smallest other node)
        // (sorted by range, in parallel, then merged: the stable sort merges sorted runs)
        let mut keys: Vec<(usize, usize)> = in_parallel_collect(threads, n, &|begin, end| {
            let mut keys: Vec<(usize, usize)> = (begin..end)
                .filter(|&u| *parent.get(u) != none && uf.find(u) == u)
                .map(|r| (r.min(heads_by_root.get(r).index()), r))
                .collect();
            keys.sort_unstable();
            keys
        })?
        .concat();
        keys.sort();
        let count = keys.len();
        let number_mem = SharedSliceMut::<Ix>::abst_mem_mut(&scratch(27)?, n.max(1), true)?;
        let number = number_mem.shared_slice();
        let heads = SharedSliceMut::<Ix>::abst_mem_mut(&result(1)?, count.max(1), true)?;
        {
            let (keys, heads) = (&keys, heads.shared_slice());
            in_parallel(threads, count, &|begin, end| {
                let (mut number, mut heads) = (number, heads);
                for id in begin..end {
                    let r = keys[id].1;
                    *number.get_mut(r) = Ix::new(id);
                    *heads.get_mut(id) = *heads_by_root.get(r);
                }
            })?;
        }
        drop(keys);
        let components = SharedSliceMut::<Ix>::abst_mem_mut(&result(0)?, n.max(1), true)?;
        {
            let components = components.shared_slice();
            in_parallel(threads, n, &|begin, end| {
                let mut components = components;
                for u in begin..end {
                    *components.get_mut(u) = if *parent.get(u) != none {
                        *number.get(uf.find(u))
                    } else {
                        none
                    };
                }
            })?;
        }

        // articulation points: heads, of two components or more if they're roots
        let articulation_points =
            SharedSliceMut::<bool>::abst_mem_mut(&result(2)?, n.max(1), true)?;
        {
            // whether each node heads a component (bit 1), and another (bit 2)
            let headed_mem =
                SharedSliceMut::<AtomicU8>::abst_mem_mut(&scratch(28)?, n.max(1), true)?;
            let headed = headed_mem.shared_slice();
            let heads = heads.shared_slice();
            in_parallel(threads, count, &|begin, end| {
                for id in begin..end {
                    let h = headed.get(heads.get(id).index());
                    if h.fetch_or(1, Ordering::Relaxed) & 1 != 0 {
                        h.fetch_or(2, Ordering::Relaxed);
                    }
                }
            })?;
            let a = articulation_points.shared_slice();
            in_parallel(threads, n, &|begin, end| {
                let mut a = a;
                for u in begin..end {
                    let bit = if *parent.get(u) == none { 2 } else { 1 };
                    *a.get_mut(u) = headed.get(u).load(Ordering::Relaxed) & bit != 0;
                }
            })?;
        }

        // 5. 2-edge-connected components: the connected components of the graph without its
        //    bridges, numbered by their smallest node (their root)
        uf.reset(threads, n)?;
        in_parallel(threads, n, &|begin, end| {
            for u in begin..end {
                for e in row(u) {
                    let v = neighbours.get(e).index();
                    if v >= u {
                        continue;
                    }
                    if is_tree_edge(u, v, e) {
                        let c = if parent.get(v).index() == u && *parent.get(v) != none {
                            v
                        } else {
                            u
                        };
                        if *is_bridge.get(c) {
                            continue;
                        }
                    }
                    uf.union(u, v);
                }
            }
        })?;
        let two_edge_components = SharedSliceMut::<Ix>::abst_mem_mut(&result(4)?, n.max(1), true)?;
        let two_edge_count;
        {
            // roots numbered in order (by range, from the roots before it), then their nodes
            let mut t = two_edge_components.shared_slice();
            let found = in_parallel_collect(threads, n, &|begin, end| {
                (begin..end).filter(|&u| uf.find(u) == u).count()
            })?;
            let load = n.div_ceil(threads);
            let mut at = 0;
            for (tid, f) in found.iter().enumerate() {
                let begin = (tid * load).min(n);
                // each range's first id, kept in its first node's slot till numbered
                if begin < n {
                    *t.get_mut(begin) = Ix::new(at);
                }
                at += f;
            }
            two_edge_count = at;
            let number = number_mem.shared_slice();
            in_parallel(threads, n, &|begin, end| {
                let mut number = number;
                let mut id = if begin < end { t.get(begin).index() } else { 0 };
                for u in begin..end {
                    if uf.find(u) == u {
                        *number.get_mut(u) = Ix::new(id);
                        id += 1;
                    }
                }
            })?;
            in_parallel(threads, n, &|begin, end| {
                let mut t = t;
                for u in begin..end {
                    *t.get_mut(u) = *number.get(uf.find(u));
                }
            })?;
        }

        components.flush_async()?;
        heads.flush_async()?;
        articulation_points.flush_async()?;
        bridges.flush_async()?;
        two_edge_components.flush_async()?;
        Ok(Self {
            g,
            components,
            heads,
            articulation_points,
            bridges,
            two_edge_components,
            count,
            bridge_count,
            two_edge_count,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::random_arcs;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use std::collections::BTreeSet;
    use std::io::Write;

    /// An undirected graph of `n` nodes and `edges` (parallel edges and self-loops kept).
    fn undirected(name: &str, n: usize, edges: &[(usize, usize)], threads: u8) -> TestGraph {
        let dir = unique_test_dir("biconnected_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u.max(v) + 1, u.min(v) + 1).unwrap();
        }
        drop(f);
        let id = format!("biconnected_{name}_{}", unique_run_tag());
        TestGraph::new(
            Graph::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    /// The graph as adjacency lists of (neighbour, edge id), self-loops aside.
    fn adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
        let mut adj = vec![vec![]; n];
        for (id, &(u, v)) in edges.iter().enumerate() {
            if u != v {
                adj[u].push((v, id));
                adj[v].push((u, id));
            }
        }
        adj
    }

    /// Whether `a` reaches `b` without `node` or the edge `edge`.
    fn reaches(
        adj: &[Vec<(usize, usize)>],
        a: usize,
        b: usize,
        node: Option<usize>,
        edge: Option<usize>,
    ) -> bool {
        let mut seen = vec![false; adj.len()];
        let mut stack = vec![a];
        seen[a] = true;
        while let Some(u) = stack.pop() {
            if u == b {
                return true;
            }
            for &(v, id) in &adj[u] {
                if !seen[v] && Some(v) != node && Some(id) != edge {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        false
    }

    /// Biconnected components' node sets, by Hopcroft and Tarjan's algorithm (an edge stack,
    /// the parent edge skipped by its id, so that parallel edges count).
    fn reference_components(adj: &[Vec<(usize, usize)>]) -> BTreeSet<Vec<usize>> {
        let n = adj.len();
        let (mut disc, mut low) = (vec![usize::MAX; n], vec![0; n]);
        let mut time = 0;
        let mut edge_stack: Vec<(usize, usize, usize)> = vec![];
        let mut out = BTreeSet::new();
        fn dfs(
            u: usize,
            parent_edge: Option<usize>,
            adj: &[Vec<(usize, usize)>],
            disc: &mut Vec<usize>,
            low: &mut Vec<usize>,
            time: &mut usize,
            stack: &mut Vec<(usize, usize, usize)>,
            out: &mut BTreeSet<Vec<usize>>,
        ) {
            disc[u] = *time;
            low[u] = *time;
            *time += 1;
            for &(v, id) in &adj[u] {
                if Some(id) == parent_edge {
                    continue;
                }
                if disc[v] == usize::MAX {
                    stack.push((u, v, id));
                    dfs(v, Some(id), adj, disc, low, time, stack, out);
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        let mut nodes = BTreeSet::new();
                        while let Some((a, b, i)) = stack.pop() {
                            nodes.insert(a);
                            nodes.insert(b);
                            if i == id {
                                break;
                            }
                        }
                        out.insert(nodes.into_iter().collect());
                    }
                } else if disc[v] < disc[u] {
                    stack.push((u, v, id));
                    low[u] = low[u].min(disc[v]);
                }
            }
        }
        for u in 0..n {
            if disc[u] == usize::MAX {
                dfs(
                    u,
                    None,
                    adj,
                    &mut disc,
                    &mut low,
                    &mut time,
                    &mut edge_stack,
                    &mut out,
                );
            }
        }
        out
    }

    /// Checks every result against brute force and the reference.
    fn check(name: &str, n: usize, edges: &[(usize, usize)]) {
        let adj = adjacency(n, edges);
        let want_components = reference_components(&adj);
        let want_articulation: Vec<usize> = (0..n)
            .filter(|&v| {
                let nbrs: Vec<usize> = adj[v].iter().map(|&(w, _)| w).collect();
                nbrs.iter()
                    .any(|&a| nbrs.iter().any(|&b| !reaches(&adj, a, b, Some(v), None)))
            })
            .collect();
        let mut want_bridges: Vec<[usize; 2]> = edges
            .iter()
            .enumerate()
            .filter(|&(id, &(u, v))| u != v && !reaches(&adj, u, v, None, Some(id)))
            .map(|(_, &(u, v))| [u.min(v), u.max(v)])
            .collect();
        want_bridges.sort();
        let bridge_ids: BTreeSet<usize> = edges
            .iter()
            .enumerate()
            .filter(|&(id, &(u, v))| u != v && !reaches(&adj, u, v, None, Some(id)))
            .map(|(id, _)| id)
            .collect();
        // 2-edge-connected components, numbered by their smallest node
        let mut want_two_edge = vec![usize::MAX; n];
        let mut two_edge_count = 0;
        for s in 0..n {
            if want_two_edge[s] != usize::MAX {
                continue;
            }
            let mut stack = vec![s];
            want_two_edge[s] = two_edge_count;
            while let Some(u) = stack.pop() {
                for &(v, id) in &adj[u] {
                    if want_two_edge[v] == usize::MAX && !bridge_ids.contains(&id) {
                        want_two_edge[v] = two_edge_count;
                        stack.push(v);
                    }
                }
            }
            two_edge_count += 1;
        }

        for threads in [1, 3, 8] {
            let g = undirected(name, n, edges, threads);
            let mut b = Biconnectivity::new(&g).unwrap();
            let what = format!("{name}, {threads} threads");
            let got: Vec<Vec<usize>> = (0..b.count()).map(|c| b.component_nodes(c)).collect();
            // numbered by their smallest node
            assert!(
                got.windows(2).all(|w| w[0][0] <= w[1][0]),
                "{what}: numbering {got:?}"
            );
            let got_set: BTreeSet<Vec<usize>> = got.into_iter().collect();
            assert_eq!(got_set, want_components, "{what}: components");
            assert_eq!(
                b.articulation_points(),
                want_articulation,
                "{what}: articulation points"
            );
            let bridges: Vec<[usize; 2]> = b
                .bridges()
                .iter()
                .map(|e| [e[0].index(), e[1].index()])
                .collect();
            assert_eq!(bridges, want_bridges, "{what}: bridges");
            let two: Vec<usize> = b.two_edge_components().iter().map(|c| c.index()).collect();
            assert_eq!(two, want_two_edge, "{what}: 2-edge-connected components");
            assert_eq!(b.two_edge_count(), two_edge_count, "{what}");
            // a node's listed component holds it, and heads are its other nodes
            for u in 0..n {
                if let Some(c) = b.component(u) {
                    assert_ne!(b.head(c), u, "{what}: node {u} heads its own component");
                }
            }
            b.drop_cache().unwrap();
        }
    }

    #[test]
    fn biconnectivity_of_small_shapes() {
        for (name, n, edges) in [
            ("empty", 3, vec![]),
            ("edge", 2, vec![(0, 1)]),
            ("path", 5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]),
            ("cycle", 5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
            ("star", 5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]),
            // two triangles at node 2, and a pendant edge
            (
                "bowtie",
                6,
                vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5)],
            ),
            // parallel edges aren't bridges; self-loops change nothing
            ("parallel", 4, vec![(0, 1), (0, 1), (1, 2), (2, 3), (3, 3)]),
            ("triple", 3, vec![(0, 1), (0, 1), (0, 1), (1, 2)]),
            // components, isolated nodes
            (
                "parts",
                9,
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 0),
                    (4, 5),
                    (6, 7),
                    (7, 8),
                    (8, 6),
                    (6, 4),
                ],
            ),
            // a cycle through the root's children (one component at the root)
            ("root_cycle", 4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]),
            (
                "two_cycles_at_root",
                5,
                vec![(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
            ),
        ] {
            check(name, n, &edges);
        }
    }

    #[test]
    fn biconnectivity_of_random_graphs() {
        for seed in 0..40u64 {
            let n = 5 + (seed as usize * 7) % 60;
            // sparse ones have many bridges and articulation points, denser ones few
            let m = n * (1 + seed as usize % 3) / 2 + seed as usize % 5;
            let edges: Vec<(usize, usize)> = random_arcs(n, m, seed + 1);
            check(&format!("random_{seed}"), n, &edges);
        }
    }

    /// The same results whatever the threads, on a larger graph, but which of their components
    /// articulation points are listed in.
    #[test]
    fn biconnectivity_is_the_same_whatever_the_threads() {
        let n = 20_000;
        let mut edges = random_arcs(n, 18_000, 7);
        edges.extend((0..2000).map(|u| (u, u + 1)));
        let run = |threads| {
            let g = undirected("threads", n, &edges, threads);
            let mut b = Biconnectivity::new(&g).unwrap();
            let sets: Vec<(usize, Vec<usize>)> = (0..b.count())
                .map(|c| (b.head(c).min(b.component_nodes(c)[0]), b.component_nodes(c)))
                .collect();
            let out = (
                sets,
                b.articulation_points(),
                b.bridges().to_vec(),
                b.two_edge_components().to_vec(),
            );
            b.drop_cache().unwrap();
            out
        };
        let one = run(1);
        assert_eq!(run(4), one);
        assert_eq!(run(16), one);
    }
}
