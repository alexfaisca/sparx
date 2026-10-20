use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use std::cell::Cell;
use std::mem::ManuallyDrop;

/// Default maximum number of local moving phases of [`DirectedLouvain`].
const DEFAULT_MAX_PASSES: usize = 256;
/// Default minimum modularity gain for a local moving phase to count as an improvement.
const DEFAULT_TOLERANCE: f64 = 0.0;
/// First sequence number of [`directed_modularity`]'s scratch files (those of [`DirectedLouvain`]
/// start at `0`): the two never share one, even run at once on the same graph.
const MODULARITY_SCRATCH: usize = 1 << (usize::BITS - 1);

/// Scratch arrays of one run, each in a memmapped file of its own (see [`ScratchMem`]): working
/// memory as large as the graph stays out of the heap.
struct Scratch<'a> {
    name: &'a dyn Fn(usize) -> Result<String, Box<dyn std::error::Error>>,
    next: Cell<usize>,
}

impl<'a> Scratch<'a> {
    fn new(
        name: &'a dyn Fn(usize) -> Result<String, Box<dyn std::error::Error>>,
        first: usize,
    ) -> Self {
        Self {
            name,
            next: Cell::new(first),
        }
    }

    fn file_name(&self) -> Result<String, Box<dyn std::error::Error>> {
        let seq = self.next.get();
        self.next.set(seq + 1);
        (self.name)(seq)
    }

    /// A zeroed array.
    fn zeroed<T>(&self, len: usize) -> Result<ScratchMem<T>, Box<dyn std::error::Error>> {
        ScratchMem::new(self.file_name()?, len)
    }

    /// An array of `value`s.
    fn filled<T: Copy>(
        &self,
        len: usize,
        value: T,
    ) -> Result<ScratchMem<T>, Box<dyn std::error::Error>> {
        ScratchMem::filled(self.file_name()?, len, value)
    }

    /// The array `0, 1, .., len - 1`: every node in a community of its own.
    fn singletons(&self, len: usize) -> Result<ScratchMem<usize>, Box<dyn std::error::Error>> {
        let mut mem = self.zeroed::<usize>(len)?;
        for (i, c) in mem.iter_mut().enumerate() {
            *c = i;
        }
        Ok(mem)
    }

    /// An empty stack of room for `capacity` elements.
    fn stack<T>(&self, capacity: usize) -> Result<ScratchStack<T>, Box<dyn std::error::Error>> {
        ScratchStack::new(self.file_name()?, capacity)
    }
}

/// Leicht and Newman's directed modularity of a partition of a [`GraphMemoryMap`] instance
/// (["Community structure in directed networks"](https://doi.org/10.1103/PhysRevLett.100.118703)):
///
/// `Q = (1/m) Σ_ij [A_ij − k_i^out k_j^in / m] δ(c_i, c_j)`,
///
/// where `m` is the number of arcs, `A_ij` the number of arcs `i -> j` (parallel arcs add up)
/// and `k_i^out`, `k_i^in` node `i`'s out- and in-degree. A self-loop is one arc, counted in
/// both its node's out- and in-degree: this is what networkx's `community.modularity` computes
/// on a `DiGraph`/`MultiDiGraph`. On an undirected graph (stored, as every graph here, with each
/// edge both ways) it is the usual undirected modularity, except for self-loops: stored once,
/// they count as one arc here, where undirected modularity counts them twice (networkx's
/// `modularity` of a `Graph` with self-loops differs from this, that of `G.to_directed()` doesn't).
///
/// `communities[v]` is node `v`'s community; any labels will do. Computed in `O(|V| + |E|)`
/// from each community's total out- and in-degree and number of internal arcs. A graph without
/// arcs has modularity `0`.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
pub fn directed_modularity<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    communities: &[Ix],
) -> Result<f64, Box<dyn std::error::Error>> {
    let node_count = g.size();
    if communities.len() != node_count {
        return Err(format!(
            "error computing directed modularity: {} community labels for {node_count} nodes",
            communities.len()
        )
        .into());
    }
    let arc_count = g.width();
    if node_count == 0 || arc_count == 0 {
        return Ok(0.0);
    }

    let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
    let scratch = Scratch::new(&name, MODULARITY_SCRATCH);

    // labels as dense indices: as they are if small enough, by rank among the distinct labels
    // otherwise (sorted, in a scratch file)
    let distinct = if communities.iter().all(|c| c.index() < node_count) {
        None
    } else {
        let mut labels = scratch.zeroed::<usize>(node_count)?;
        for (l, c) in labels.iter_mut().zip(communities) {
            *l = c.index();
        }
        labels.sort_unstable();
        let mut len = 0;
        for i in 0..node_count {
            if i == 0 || labels[i] != labels[len - 1] {
                labels[len] = labels[i];
                len += 1;
            }
        }
        Some((labels, len))
    };
    let dense = |u: usize| -> usize {
        let c = communities[u].index();
        match &distinct {
            None => c,
            Some((labels, len)) => labels[..*len].partition_point(|&l| l < c),
        }
    };

    let offsets = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
    let neighbours = SharedSlice::<Ix>::new(g.neighbours_ptr(), arc_count);
    let (offsets, neighbours) = (offsets.as_slice(), neighbours.as_slice());

    let mut sigma_out = scratch.zeroed::<u64>(node_count)?;
    let mut sigma_in = scratch.zeroed::<u64>(node_count)?;
    let mut internal = 0u64;
    for u in 0..node_count {
        let cu = dense(u);
        for v in &neighbours[offsets[u].index()..offsets[u + 1].index()] {
            let cv = dense(v.index());
            sigma_out[cu] += 1;
            sigma_in[cv] += 1;
            if cu == cv {
                internal += 1;
            }
        }
    }
    let expected: u128 = sigma_out
        .iter()
        .zip(sigma_in.iter())
        .map(|(&o, &i)| o as u128 * i as u128)
        .sum();
    let m = arc_count as f64;
    Ok(internal as f64 / m - expected as f64 / (m * m))
}

/// Directed Louvain community detection on a [`GraphMemoryMap`] instance: a partition of its
/// nodes (locally) maximizing [directed modularity](directed_modularity), following Dugué and
/// Perez's ["Directed Louvain: maximizing modularity in directed
/// networks"](https://hal.science/hal-01231784).
///
/// This is the community detection algorithm for [`DiGraph`]s, as [`AlgoGVELouvain`] (faster and
/// parallel) is for undirected [`Graph`]s; on a symmetric `DiGraph` (e.g. a `Graph`'s
/// [`into_directed`](crate::graph::GraphMemoryMap::into_directed)) it is ordinary, sequential,
/// Louvain. The graph is unweighted: each arc weighs `1`, so parallel arcs add up.
///
/// [`DiGraph`]: crate::graph::DiGraph
/// [`Graph`]: crate::graph::Graph
///
/// The algorithm alternates two phases, sequentially and in node order (so results are
/// deterministic):
///
/// * *local moving*: each node in turn is moved to the community, among its in- and
///   out-neighbours' (or its own), whose modularity gain
///   `ΔQ = (w_{i→c} + w_{c→i})/m − (k_i^out Σ_c^in + k_i^in Σ_c^out)/m²` for inserting it
///   (once taken out of its own) is largest, until a sweep over all nodes moves none (or gains
///   no more than the tolerance). Gains are compared exactly, in integers.
/// * *aggregation*: communities become the nodes of a weighted directed supergraph (arc weights
///   are the number of arcs between communities, internal arcs becoming self-loops), on which
///   local moving starts again from singletons.
///
/// When a level no longer improves, local moving is run once more on the original graph,
/// starting from the partition found; if it moves any node, aggregation resumes from there.
/// The result is thus a local optimum for single node moves: no node gains modularity by moving
/// to a neighbour's community.
///
/// The supergraphs (weighted CSRs of out- and in-arcs) and every working array are held in
/// memmapped scratch files, removed as soon as they're no longer needed; the original graph is
/// read through its CSR and its [`transpose`]. Communities are numbered by their
/// smallest node, as connected components are.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
/// [`transpose`]: ../../graph/struct.GraphMemoryMap.html#method.transpose
/// [`AlgoGVELouvain`]: ../gve_louvain/struct.AlgoGVELouvain.html
#[allow(dead_code)]
#[derive(Debug)]
pub struct DirectedLouvain<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph whose nodes are partitioned.
    g: &'a DiGraph<N, E, Ix>,
    /// Memmapped slice containing the community of each node.
    communities: AbstractedProceduralMemoryMut<Ix>,
    /// Number of communities.
    community_count: usize,
    /// Directed modularity of the partition.
    modularity: f64,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedLouvain<'a, N, E, Ix> {
    /// Partitions the graph's nodes by directed Louvain, with default configuration: local
    /// moving phases (at most 256) are repeated until they no longer improve modularity.
    ///
    /// The resulting community of each node is stored in memory (in a memmapped file).
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance to be partitioned.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a DiGraph<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_conf(g, DEFAULT_MAX_PASSES, DEFAULT_TOLERANCE)
    }

    /// Partitions the graph's nodes by directed Louvain.
    ///
    /// The resulting community of each node is stored in memory (in a memmapped file).
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance to be partitioned.
    /// * `max_passes` --- the maximum number of local moving phases (on the graph or on a
    ///   supergraph), at least `1`.
    /// * `tolerance` --- the modularity gain a local moving phase must exceed to be followed by
    ///   another (its moves are kept either way); within a phase, sweeps over the nodes stop
    ///   once one gains no more than it. With `0` (the default), phases go on as long as some
    ///   node move improves modularity.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a DiGraph<N, E, Ix>,
        max_passes: usize,
        tolerance: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if max_passes == 0 {
            return Err("error running directed louvain: max_passes must be at least 1".into());
        }
        if tolerance.is_nan() || tolerance < 0.0 {
            return Err(format!(
                "error running directed louvain: tolerance must be non-negative, got {tolerance}"
            )
            .into());
        }
        let out_fn = g.build_cache_filename(CacheFile::DirectedLouvain, None)?;
        let communities = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut louvain = Self {
            g,
            communities,
            community_count: 0,
            modularity: 0.0,
        };
        louvain.compute(max_passes, tolerance)?;
        Ok(louvain)
    }

    /// Returns the community of a given node.
    pub fn community(&self, node: usize) -> usize {
        assert!(node < self.g.size());
        self.communities.get(node).index()
    }

    /// Returns a slice containing the community of each node.
    pub fn communities(&self) -> &[Ix] {
        self.communities.as_slice()
    }

    /// Returns the number of communities.
    pub fn community_count(&self) -> usize {
        self.community_count
    }

    /// Returns the [directed modularity](directed_modularity) of the partition.
    pub fn partition_modularity(&self) -> f64 {
        self.modularity
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        this.g.cleanup_cache(CacheFile::DirectedLouvain)?;
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::DirectedLouvain, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }

    fn compute(
        &mut self,
        max_passes: usize,
        tolerance: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        if node_count == 0 {
            return Ok(());
        }
        let base = BaseArcs::new(self.g)?;
        let m = self.g.width() as u64;
        let g = self.g;
        let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
        let scratch = Scratch::new(&name, 0);

        let mut part = scratch.singletons(node_count)?;
        let mut passes = 0;
        'outer: while passes < max_passes {
            // local moving on the graph itself: from singletons at first, later refining the
            // partition found on the supergraphs
            let (moves, gain) = local_move(&base, &mut part, m, tolerance, &scratch)?;
            passes += 1;
            let mut count = renumber(&mut part, &scratch)?;
            if moves == 0 || gain <= tolerance {
                break;
            }
            let mut level = Supergraph::aggregate(&base, &part, count, &scratch)?;
            loop {
                if passes >= max_passes {
                    break 'outer;
                }
                let mut comm = scratch.singletons(level.len())?;
                let (moves, gain) = local_move(&level, &mut comm, m, tolerance, &scratch)?;
                passes += 1;
                if moves == 0 {
                    break;
                }
                count = renumber(&mut comm, &scratch)?;
                for p in part.iter_mut() {
                    *p = comm[*p];
                }
                if gain <= tolerance {
                    break;
                }
                level = Supergraph::aggregate(&level, &comm, count, &scratch)?;
            }
        }

        // numbered by smallest node
        self.community_count = renumber(&mut part, &scratch)?;
        let out = self.communities.as_mut_slice();
        for (o, &p) in out.iter_mut().zip(part.iter()) {
            *o = Ix::new(p);
        }
        drop(part);
        self.communities.flush_async()?;
        self.modularity = directed_modularity(self.g, self.communities.as_slice())?;
        Ok(())
    }
}

/// Renumbers `comm`'s labels (all smaller than `comm.len()`) densely, in order of first
/// appearance, and returns how many there are.
fn renumber(comm: &mut [usize], scratch: &Scratch) -> Result<usize, Box<dyn std::error::Error>> {
    let mut ids = scratch.filled(comm.len(), usize::MAX)?;
    let mut count = 0;
    for c in comm.iter_mut() {
        if ids[*c] == usize::MAX {
            ids[*c] = count;
            count += 1;
        }
        *c = ids[*c];
    }
    Ok(count)
}

/// A weighted directed graph local moving and aggregation run on: the original graph, or a
/// supergraph of communities.
trait Arcs {
    /// Number of nodes.
    fn len(&self) -> usize;
    /// Number of (stored) out-arcs, as many as in-arcs: bounds a supergraph's.
    fn arc_count(&self) -> usize;
    /// Total weight of a node's out-arcs (self-loops included).
    fn k_out(&self, i: usize) -> u64;
    /// Total weight of a node's in-arcs (self-loops included).
    fn k_in(&self, i: usize) -> u64;
    /// Calls `f(j, w)` for a node's out-arcs `i -> j` of weight `w` (possibly repeated `j`s).
    fn out_arcs(&self, i: usize, f: impl FnMut(usize, u64));
    /// Calls `f(j, w)` for a node's in-arcs `j -> i` of weight `w` (possibly repeated `j`s).
    fn in_arcs(&self, i: usize, f: impl FnMut(usize, u64));
}

/// The original graph's arcs, each of weight `1`: out-arcs from its CSR, in-arcs from its
/// transpose's.
struct BaseArcs<Ix: graph::IndexType> {
    out_offsets: SharedSlice<Ix>,
    out_neighbours: SharedSlice<Ix>,
    in_offsets: SharedSlice<Ix>,
    in_neighbours: SharedSlice<Ix>,
}

impl<Ix: graph::IndexType> BaseArcs<Ix> {
    /// Valid while `g` (which owns its transpose) is.
    fn new<N: graph::N, E: graph::E>(
        g: &DiGraph<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.transpose()?;
        Ok(Self {
            out_offsets: SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size()),
            out_neighbours: SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width()),
            in_offsets: SharedSlice::<Ix>::new(t.offsets_ptr(), t.offsets_size()),
            in_neighbours: SharedSlice::<Ix>::new(t.neighbours_ptr(), t.width()),
        })
    }

    fn range(offsets: &SharedSlice<Ix>, i: usize) -> std::ops::Range<usize> {
        offsets.get(i).index()..offsets.get(i + 1).index()
    }
}

impl<Ix: graph::IndexType> Arcs for BaseArcs<Ix> {
    fn len(&self) -> usize {
        self.out_offsets.len() - 1
    }

    fn arc_count(&self) -> usize {
        self.out_neighbours.len()
    }

    fn k_out(&self, i: usize) -> u64 {
        Self::range(&self.out_offsets, i).len() as u64
    }

    fn k_in(&self, i: usize) -> u64 {
        Self::range(&self.in_offsets, i).len() as u64
    }

    fn out_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        for j in &self.out_neighbours.as_slice()[Self::range(&self.out_offsets, i)] {
            f(j.index(), 1);
        }
    }

    fn in_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        for j in &self.in_neighbours.as_slice()[Self::range(&self.in_offsets, i)] {
            f(j.index(), 1);
        }
    }
}

/// A weighted directed supergraph, as CSRs of `(neighbour, weight)` out- and in-arcs (one per
/// neighbour, self-loops included), in scratch files.
struct Supergraph {
    out_offsets: ScratchMem<usize>,
    out_arcs: ScratchStack<(usize, u64)>,
    in_offsets: ScratchMem<usize>,
    in_arcs: ScratchStack<(usize, u64)>,
    k_out: ScratchMem<u64>,
    k_in: ScratchMem<u64>,
}

impl Supergraph {
    /// The supergraph of `a`'s communities `comm` (dense, `0..count`): an arc `c -> d` weighs
    /// as much as all arcs from `c`'s nodes to `d`'s.
    fn aggregate<A: Arcs>(
        a: &A,
        comm: &[usize],
        count: usize,
        scratch: &Scratch,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // nodes by community
        let mut start = scratch.zeroed::<usize>(count + 1)?;
        for &c in comm {
            start[c + 1] += 1;
        }
        for c in 0..count {
            start[c + 1] += start[c];
        }
        let mut next = scratch.zeroed::<usize>(count)?;
        next.copy_from_slice(&start[..count]);
        let mut members = scratch.zeroed::<usize>(comm.len())?;
        for (i, &c) in comm.iter().enumerate() {
            members[next[c]] = i;
            next[c] += 1;
        }
        drop(next);

        // no more arcs between communities than between their nodes
        let mut s = Self {
            out_offsets: scratch.zeroed(count + 1)?,
            out_arcs: scratch.stack(a.arc_count())?,
            in_offsets: scratch.zeroed(count + 1)?,
            in_arcs: scratch.stack(a.arc_count())?,
            k_out: scratch.zeroed(count)?,
            k_in: scratch.zeroed(count)?,
        };
        let mut acc = scratch.zeroed::<u64>(count)?;
        let mut touched = scratch.stack::<usize>(count)?;
        fn flush(
            acc: &mut [u64],
            touched: &mut ScratchStack<usize>,
            arcs: &mut ScratchStack<(usize, u64)>,
        ) {
            touched.as_mut_slice().sort_unstable();
            for &d in touched.as_slice() {
                arcs.push((d, acc[d]));
                acc[d] = 0;
            }
            touched.clear();
        }
        for c in 0..count {
            let nodes = &members[start[c]..start[c + 1]];
            for &i in nodes {
                s.k_out[c] += a.k_out(i);
                a.out_arcs(i, |j, w| {
                    let d = comm[j];
                    if acc[d] == 0 {
                        touched.push(d);
                    }
                    acc[d] += w;
                });
            }
            flush(&mut acc, &mut touched, &mut s.out_arcs);
            s.out_offsets[c + 1] = s.out_arcs.len();
            for &i in nodes {
                s.k_in[c] += a.k_in(i);
                a.in_arcs(i, |j, w| {
                    let d = comm[j];
                    if acc[d] == 0 {
                        touched.push(d);
                    }
                    acc[d] += w;
                });
            }
            flush(&mut acc, &mut touched, &mut s.in_arcs);
            s.in_offsets[c + 1] = s.in_arcs.len();
        }
        Ok(s)
    }
}

impl Arcs for Supergraph {
    fn len(&self) -> usize {
        self.k_out.len()
    }

    fn arc_count(&self) -> usize {
        self.out_arcs.len()
    }

    fn k_out(&self, i: usize) -> u64 {
        self.k_out[i]
    }

    fn k_in(&self, i: usize) -> u64 {
        self.k_in[i]
    }

    fn out_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        for &(j, w) in &self.out_arcs.as_slice()[self.out_offsets[i]..self.out_offsets[i + 1]] {
            f(j, w);
        }
    }

    fn in_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        for &(j, w) in &self.in_arcs.as_slice()[self.in_offsets[i]..self.in_offsets[i + 1]] {
            f(j, w);
        }
    }
}

/// Louvain's local moving phase on `a` (whose arcs weigh `m` in all), from the partition
/// `comm` (labels smaller than `a.len()`): sweeps over the nodes in order, moving each to the
/// neighbouring community (or its own) where inserting it gains most modularity, until a sweep
/// moves no node or gains no more than `tolerance`.
///
/// For node `i` taken out of its community, inserting it into `c` gains
/// `(w_{i→c} + w_{c→i})/m − (k_i^out Σ_c^in + k_i^in Σ_c^out)/m²`; gains are compared as
/// their (integer) multiples by `m²`, so exactly. Ties keep a node where it is, or else go to
/// the smallest community.
///
/// Returns the number of moves and the modularity gained.
fn local_move<A: Arcs>(
    a: &A,
    comm: &mut [usize],
    m: u64,
    tolerance: f64,
    scratch: &Scratch,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    let n = a.len();
    let mut sigma_out = scratch.zeroed::<u64>(n)?;
    let mut sigma_in = scratch.zeroed::<u64>(n)?;
    for i in 0..n {
        sigma_out[comm[i]] += a.k_out(i);
        sigma_in[comm[i]] += a.k_in(i);
    }
    let mut size = scratch.zeroed::<usize>(n)?;
    for i in 0..n {
        size[comm[i]] += 1;
    }
    // labels of no community: at most every one, but that of the community left
    let mut free = scratch.stack::<usize>(n)?;
    for c in 0..n {
        if size[c] == 0 {
            free.push(c);
        }
    }
    let mut acc = scratch.zeroed::<u64>(n)?;
    let mut touched = scratch.stack::<usize>(n)?;
    let m_sq = (m as f64) * (m as f64);
    let (mut moves, mut gain) = (0, 0.0);
    loop {
        let (mut sweep_moves, mut sweep_gain) = (0, 0i128);
        for i in 0..n {
            let (d, k_out, k_in) = (comm[i], a.k_out(i) as i128, a.k_in(i) as i128);
            // w_{i→c} + w_{c→i} for each neighbouring community c, self-loops excepted
            let mut link = |j: usize, w: u64| {
                if j != i {
                    let c = comm[j];
                    if acc[c] == 0 {
                        touched.push(c);
                    }
                    acc[c] += w;
                }
            };
            a.out_arcs(i, &mut link);
            a.in_arcs(i, &mut link);

            sigma_out[d] -= k_out as u64;
            sigma_in[d] -= k_in as u64;
            let score = |c: usize, w: u64| {
                m as i128 * w as i128 - (k_out * sigma_in[c] as i128 + k_in * sigma_out[c] as i128)
            };
            let stay = score(d, acc[d]);
            let (mut best, mut best_score) = (d, stay);
            for &c in touched.as_slice() {
                let s = score(c, acc[c]);
                if s > best_score || (s == best_score && best != d && c < best) {
                    (best, best_score) = (c, s);
                }
            }
            // a community of its own (score 0) beats staying where it is at a loss
            if best_score < 0 {
                let empty = free.pop().expect("a free label");
                (best, best_score) = (empty, 0);
            }
            sigma_out[best] += k_out as u64;
            sigma_in[best] += k_in as u64;
            size[d] -= 1;
            size[best] += 1;
            if size[d] == 0 && best != d {
                free.push(d);
            }
            if best != d {
                comm[i] = best;
                sweep_moves += 1;
                sweep_gain += best_score - stay;
            }
            for &c in touched.as_slice() {
                acc[c] = 0;
            }
            touched.clear();
        }
        let sweep_gain = sweep_gain as f64 / m_sq;
        moves += sweep_moves;
        gain += sweep_gain;
        if sweep_moves == 0 || sweep_gain <= tolerance {
            break;
        }
    }
    Ok((moves, gain))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, label, random_arcs};

    /// The formula, in `O(n²)`.
    fn brute_force_modularity(n: usize, arcs: &[(usize, usize)], comm: &[usize]) -> f64 {
        let m = arcs.len() as f64;
        if arcs.is_empty() {
            return 0.0;
        }
        let mut a = vec![vec![0.0; n]; n];
        let (mut k_out, mut k_in) = (vec![0.0; n], vec![0.0; n]);
        for &(u, v) in arcs {
            a[u][v] += 1.0;
            k_out[u] += 1.0;
            k_in[v] += 1.0;
        }
        let mut q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if comm[i] == comm[j] {
                    q += a[i][j] - k_out[i] * k_in[j] / m;
                }
            }
        }
        q / m
    }

    /// Usual undirected modularity of simple undirected `edges` (each listed once).
    fn brute_force_undirected_modularity(
        n: usize,
        edges: &[(usize, usize)],
        comm: &[usize],
    ) -> f64 {
        let two_m = 2.0 * edges.len() as f64;
        let mut a = vec![vec![0.0; n]; n];
        let mut k = vec![0.0; n];
        for &(u, v) in edges {
            a[u][v] += 1.0;
            a[v][u] += 1.0;
            k[u] += 1.0;
            k[v] += 1.0;
        }
        let mut q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if comm[i] == comm[j] {
                    q += a[i][j] - k[i] * k[j] / two_m;
                }
            }
        }
        q / two_m
    }

    fn ix(comm: &[usize]) -> Vec<usize> {
        comm.to_vec()
    }

    fn as_usize(comm: &[usize]) -> Vec<usize> {
        comm.iter().map(|c| c.index()).collect()
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!((a - b).abs() < 1e-12, "{what}: {a} != {b}");
    }

    fn simple_edges(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut edges: Vec<_> = random_arcs(n, m, seed)
            .into_iter()
            .filter(|(u, v)| u != v)
            .map(|(u, v)| (u.min(v), u.max(v)))
            .collect();
        edges.sort();
        edges.dedup();
        edges
    }

    #[test]
    fn directed_modularity_matches_brute_force() {
        for seed in 0..6u64 {
            let n = 12 + seed as usize * 3;
            let arcs = random_arcs(n, 4 * n, 100 + seed);
            let g = graph(&format!("dm_random_{seed}"), "general", n, &arcs);
            let partitions = [
                (0..n).collect::<Vec<_>>(),
                vec![0; n],
                (0..n).map(|v| v % 3).collect(),
                random_arcs(n, n, 200 + seed)
                    .iter()
                    .map(|&(c, _)| c % 4)
                    .collect(),
                // labels beyond |V|
                (0..n).map(|v| 1000 + v % 2).collect(),
            ];
            for (p, comm) in partitions.iter().enumerate() {
                let got = directed_modularity(&g, &ix(comm)).unwrap();
                assert_close(
                    got,
                    brute_force_modularity(n, &arcs, comm),
                    &format!("seed {seed}, partition {p}"),
                );
            }
            assert_close(
                directed_modularity(&g, &vec![0; n]).unwrap(),
                0.0,
                "one community",
            );
        }
    }

    #[test]
    fn directed_modularity_of_linked_cycles_matches_networkx() {
        let arcs = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)];
        let g = graph("dm_cycles", "general", 6, &arcs);
        assert!(!g.is_symmetric().unwrap());
        let q = directed_modularity(&g, &[0, 0, 0, 1, 1, 1]).unwrap();
        assert_close(q, 18.0 / 49.0, "two cycles");
        assert!(directed_modularity(&g, &[0, 0, 0]).is_err());
    }

    #[test]
    fn directed_modularity_of_undirected_graphs_is_the_usual_one() {
        let n = 30;
        let edges = simple_edges(n, 90, 7);
        let g = graph("dm_symmetric", "symmetric", n, &edges);
        assert!(g.is_symmetric().unwrap());
        for comm in [
            (0..n).map(|v| v % 4).collect::<Vec<_>>(),
            (0..n).map(|v| v / 10).collect(),
        ] {
            let got = directed_modularity(&g, &ix(&comm)).unwrap();
            assert_close(
                got,
                brute_force_undirected_modularity(n, &edges, &comm),
                "symmetric",
            );
        }
    }

    /// `groups` groups of `size` nodes, each a directed cycle with chords `i -> i+2, i+3`,
    /// group `k`'s first node linked to group `k+1`'s middle one.
    fn planted(groups: usize, size: usize) -> Vec<(usize, usize)> {
        let mut arcs = Vec::new();
        for k in 0..groups {
            let base = k * size;
            for i in 0..size {
                for step in 1..=3 {
                    arcs.push((base + i, base + (i + step) % size));
                }
            }
            arcs.push((base, ((k + 1) % groups) * size + size / 2));
        }
        arcs
    }

    #[test]
    fn directed_louvain_recovers_planted_communities() {
        let (groups, size) = (4, 8);
        let arcs = planted(groups, size);
        let g = graph("dl_planted", "general", groups * size, &arcs);
        assert!(!g.is_symmetric().unwrap());
        let louvain = DirectedLouvain::new(&g).unwrap();
        let expected: Vec<usize> = (0..groups * size).map(|v| v / size).collect();
        assert_eq!(as_usize(louvain.communities()), expected);
        assert_eq!(louvain.community_count(), groups);
        assert_eq!(louvain.community(size + 1), 1);
        let q = louvain.partition_modularity();
        assert_close(
            q,
            directed_modularity(&g, louvain.communities()).unwrap(),
            "result",
        );
        assert_close(
            q,
            brute_force_modularity(groups * size, &arcs, &expected),
            "brute force",
        );
    }

    #[test]
    fn directed_louvain_is_a_deterministic_local_optimum() {
        for seed in 0..4u64 {
            let n = 40;
            // a few planted groups plus noise, so that there is structure to find
            let mut arcs = planted(4, 10);
            arcs.extend(random_arcs(n, 30, 300 + seed));
            let g = graph(&format!("dl_local_{seed}"), "general", n, &arcs);
            let louvain = DirectedLouvain::new(&g).unwrap();
            let comm = as_usize(louvain.communities());
            assert_eq!(
                comm,
                as_usize(DirectedLouvain::new(&g).unwrap().communities()),
                "deterministic"
            );

            // numbered by smallest node
            assert_eq!(comm, label(n, |u, v| comm[u] == comm[v]));
            assert_eq!(louvain.community_count(), comm.iter().max().unwrap() + 1);

            let q = brute_force_modularity(n, &arcs, &comm);
            assert_close(louvain.partition_modularity(), q, "result");
            let singletons: Vec<usize> = (0..n).collect();
            assert!(q >= brute_force_modularity(n, &arcs, &singletons));

            // no node gains by moving to a neighbour's community
            for &(u, v) in &arcs {
                for (i, j) in [(u, v), (v, u)] {
                    if comm[i] != comm[j] {
                        let mut moved = comm.clone();
                        moved[i] = comm[j];
                        let q_moved = brute_force_modularity(n, &arcs, &moved);
                        assert!(
                            q_moved <= q + 1e-12,
                            "seed {seed}: moving {i} to {j}'s gains {}",
                            q_moved - q
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn directed_louvain_on_random_graphs_beats_singletons() {
        for seed in 0..4u64 {
            let n = 60;
            let arcs = random_arcs(n, 150, 400 + seed);
            let g = graph(&format!("dl_random_{seed}"), "general", n, &arcs);
            let louvain = DirectedLouvain::new_with_conf(&g, 8, 1e-6).unwrap();
            let comm = as_usize(louvain.communities());
            let q = brute_force_modularity(n, &arcs, &comm);
            assert_close(louvain.partition_modularity(), q, "result");
            assert!(q >= brute_force_modularity(n, &arcs, &(0..n).collect::<Vec<_>>()));
            assert!(q > 0.0, "seed {seed}: {q}");
        }
    }

    #[test]
    fn directed_louvain_on_undirected_graphs() {
        // two 5-cliques joined by an edge
        let mut edges = Vec::new();
        for base in [0, 5] {
            for u in 0..5 {
                for v in u + 1..5 {
                    edges.push((base + u, base + v));
                }
            }
        }
        edges.push((4, 5));
        let g = graph("dl_cliques", "symmetric", 10, &edges);
        assert!(g.is_symmetric().unwrap());
        let louvain = DirectedLouvain::new(&g).unwrap();
        let comm = as_usize(louvain.communities());
        assert_eq!(comm, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
        assert_close(
            louvain.partition_modularity(),
            brute_force_undirected_modularity(10, &edges, &comm),
            "cliques",
        );
    }

    #[test]
    fn directed_louvain_without_arcs_between_nodes() {
        let g = graph("dl_self_loop", "general", 5, &[(0, 0)]);
        let mut louvain = DirectedLouvain::new(&g).unwrap();
        assert_eq!(as_usize(louvain.communities()), vec![0, 1, 2, 3, 4]);
        assert_eq!(louvain.community_count(), 5);
        assert_close(louvain.partition_modularity(), 0.0, "self loop");
        louvain.drop_cache().unwrap();
    }

    #[test]
    fn directed_louvain_rejects_bad_configuration() {
        let g = graph("dl_conf", "general", 3, &[(0, 1), (1, 2)]);
        assert!(DirectedLouvain::new_with_conf(&g, 0, 0.0).is_err());
        assert!(DirectedLouvain::new_with_conf(&g, 4, -1.0).is_err());
        assert!(DirectedLouvain::new_with_conf(&g, 4, f64::NAN).is_err());
    }

    #[test]
    fn isolating_a_node_does_not_gain() {
        let n = 4;
        let arcs = [(2, 1), (0, 0), (3, 0), (0, 2), (3, 3)];
        let g = graph("isolate", "general", n, &arcs);
        let louvain = DirectedLouvain::new(&g).unwrap();
        let comm = as_usize(louvain.communities());
        let q = brute_force_modularity(n, &arcs, &comm);
        for i in 0..n {
            let mut moved = comm.clone();
            moved[i] = n; // a community of its own
            let q_moved = brute_force_modularity(n, &arcs, &moved);
            assert!(
                q_moved <= q + 1e-12,
                "{comm:?}: isolating {i} gains {}",
                q_moved - q
            );
        }
    }

    #[test]
    fn narrow_index_types() {
        use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
        use std::io::Write;
        let (groups, size) = (4, 8);
        let arcs = planted(groups, size);
        let dir = unique_test_dir("review_u16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.mtx");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "{} {} {}", groups * size, groups * size, arcs.len()).unwrap();
        for &(u, v) in &arcs {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let expected: Vec<usize> = (0..groups * size).map(|v| v / size).collect();
        let g16 = TestGraph::new(
            crate::graph::DiGraph::<(), (), u16>::from_file(
                &path,
                Some(format!("r16_{}", unique_run_tag())),
                Some(2),
            )
            .unwrap(),
            None,
        );
        let l = DirectedLouvain::new(&g16).unwrap();
        assert_eq!(
            l.communities()
                .iter()
                .map(|c| c.index())
                .collect::<Vec<_>>(),
            expected
        );
        let lab: Vec<u16> = (0..groups * size)
            .map(|v| 60000 + (v / size) as u16)
            .collect();
        assert_close(
            directed_modularity(&g16, &lab).unwrap(),
            l.partition_modularity(),
            "u16 big labels",
        );
        let g8 = TestGraph::new(
            crate::graph::DiGraph::<(), (), u8>::from_file(
                &path,
                Some(format!("r8_{}", unique_run_tag())),
                Some(2),
            )
            .unwrap(),
            Some(dir),
        );
        let l = DirectedLouvain::new(&g8).unwrap();
        assert_eq!(
            l.communities()
                .iter()
                .map(|c| c.index())
                .collect::<Vec<_>>(),
            expected
        );
    }
}
