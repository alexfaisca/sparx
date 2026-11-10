use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use portable_atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
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
    fn singletons(
        &self,
        len: usize,
        threads: usize,
    ) -> Result<ScratchMem<usize>, Box<dyn std::error::Error>> {
        let mut mem = self.zeroed::<usize>(len)?;
        let out = SharedSliceMut::<usize>::new(mem.as_mut_ptr(), len);
        for_each(threads, len, &|i| {
            let mut out = out;
            *out.get_mut(i) = i;
        })?;
        Ok(mem)
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
/// With weighted arcs (see [`E::is_weighted`](graph::E::is_weighted)), `m` is their total weight,
/// `A_ij` that of the arcs `i -> j`, and `k_i^out`, `k_i^in` that of `i`'s out- and in-arcs.
/// Weights must be non-negative integers (`3.0` is one; see
/// [`integer_weight`](graph::E::integer_weight)), totalling at most `2^60`: the sums are exact.
///
/// `communities[v]` is node `v`'s community; any labels will do. Computed in `O(|V| + |E|)`
/// from each community's total out- and in-weight and weight of internal arcs. A graph without
/// arcs (or whose arcs all weigh `0`) has modularity `0`.
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
    let threads = g.thread_num().max(1);

    // weighted arcs weigh their labels' integer weights (checked, and their total, first)
    let labels = E::is_weighted().then(|| SharedSlice::<E>::new(g.edge_labels_ptr(), arc_count));
    let m = match &labels {
        None => arc_count as u128,
        Some(labels) => {
            let mut m = 0u128;
            for part in by_range(threads, arc_count, &|begin, end| {
                (begin..end)
                    .map(|e| exact_weight(labels.get(e)).map(u128::from))
                    .sum::<Result<u128, _>>()
            })? {
                m += part.map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            }
            m
        }
    };
    let m = check_total_weight(m)?;
    if m == 0 {
        return Ok(0.0);
    }
    let weight = |e: usize| {
        labels
            .as_ref()
            .map_or(1, |l| l.get(e).integer_weight().unwrap_or(0))
    };

    // communities' total out- and in-weights, and weight within communities, in parallel
    let sigma_out_mem = scratch.zeroed::<u64>(node_count)?;
    let sigma_in_mem = scratch.zeroed::<u64>(node_count)?;
    let (sigma_out, sigma_in) = (atomics(&sigma_out_mem), atomics(&sigma_in_mem));
    let internal: u64 = by_range(threads, node_count, &|begin, end| {
        let mut internal = 0u64;
        for u in begin..end {
            let cu = dense(u);
            let mut k_out = 0u64;
            let row = offsets[u].index()..offsets[u + 1].index();
            for (e, v) in row.clone().zip(&neighbours[row]) {
                let (cv, w) = (dense(v.index()), weight(e));
                k_out += w;
                sigma_in[cv].fetch_add(w, Ordering::Relaxed);
                if cu == cv {
                    internal += w;
                }
            }
            sigma_out[cu].fetch_add(k_out, Ordering::Relaxed);
        }
        internal
    })?
    .into_iter()
    .sum();
    let expected: u128 = by_range(threads, node_count, &|begin, end| {
        (begin..end)
            .map(|c| sigma_out_mem[c] as u128 * sigma_in_mem[c] as u128)
            .sum::<u128>()
    })?
    .into_iter()
    .sum();
    let m = m as f64;
    Ok(internal as f64 / m - expected as f64 / (m * m))
}

pub use super::Refinement;

/// Directed Louvain community detection on a [`GraphMemoryMap`] instance: a partition of its
/// nodes (locally) maximizing [directed modularity](directed_modularity), following Dugué and
/// Perez's ["Directed Louvain: maximizing modularity in directed
/// networks"](https://hal.science/hal-01231784).
///
/// This is the community detection algorithm for [`DiGraph`]s, as [`AlgoGVELouvain`] is for
/// undirected [`Graph`]s, and runs in parallel as it does, following Sahu's ["GVE-Louvain: Fast
/// Louvain Algorithm for Community Detection in Shared Memory
/// Setting"](https://doi.org/10.48550/arXiv.2312.04876); on a symmetric `DiGraph` (e.g. a
/// `Graph`'s [`into_directed`](crate::graph::GraphMemoryMap::into_directed)) it is ordinary
/// Louvain. Parallel arcs add up. Unweighted arcs weigh `1`; weighted ones (see
/// [`E::is_weighted`](graph::E::is_weighted)) their labels' weights, which must be non-negative
/// integers totalling at most `2^60` (see [`directed_modularity`]): gains are compared exactly.
///
/// [`DiGraph`]: crate::graph::DiGraph
/// [`Graph`]: crate::graph::Graph
///
/// The algorithm alternates two phases:
///
/// * *local moving*: each node is moved to the community, among its in- and out-neighbours'
///   (or its own), whose modularity gain
///   `ΔQ = (w_{i→c} + w_{c→i})/m − (k_i^out Σ_c^in + k_i^in Σ_c^out)/m²` for inserting it
///   (once taken out of its own) is largest. Gains are compared exactly, in integers.
///   Threads take nodes as they go and move them *asynchronously*, judging communities as they
///   are when they look (communities' total degrees are atomics), and sweeps visit only the
///   nodes one of whose neighbours moved since they were last visited (GVE-Louvain's vertex
///   pruning).
/// * *aggregation*: communities become the nodes of a weighted directed supergraph (arc weights
///   are the total weight of arcs between communities, internal arcs becoming self-loops), on which
///   local moving starts again from singletons. Supergraphs are built in parallel too, each
///   community's arcs within room reserved from prefix sums.
///
/// When a level no longer improves, local moving is run once more on the original graph,
/// starting from the partition found; if it moves any node, aggregation resumes from there.
///
/// Graphs too small for every thread to take several chunks of nodes (and the supergraphs of
/// larger ones, once that small) aren't moved asynchronously: several threads would mostly move
/// strongly linked nodes at once, back and forth. Those with many arcs (a large graph's first
/// supergraphs, typically) are moved by color classes, as Lu, Halappanavar and Kalyanaraman's
/// [Grappolo](https://doi.org/10.1016/j.parco.2015.03.003) does: colored so that no two linked
/// nodes share a color, in parallel, they're moved a color at a time, each color's nodes by
/// several threads at once, which then wait for each other. The others are moved on one thread. Graphs this small are partitioned as by sequential
/// Louvain: each phase goes on until a sweep over all nodes moves none (or gains no more than
/// the tolerance), so the result is a local optimum for single node moves (no node gains
/// modularity by moving to a neighbour's community), the same whatever the threads. Larger
/// graphs, moved asynchronously, have their phases end as GVE-Louvain's do (asynchronous moves
/// may never stop finding tiny gains): after 20 sweeps, or a sweep gaining less than a
/// threshold starting at `0.01` and dropping tenfold each pass, and the algorithm ends once
/// refining the partition on the original graph gains less than a thousandth of what was gained.
/// Their partitions depend on the threads' timing, from run to run, with modularity within a
/// fraction of a percent of sequential Louvain's, or above it (and faster by 40 to 60 times on
/// 32 cores, on graphs of 4 to 22 million nodes).
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
    /// Sweeps over the nodes local moving took, on the graph and its supergraphs.
    sweeps: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedLouvain<'a, N, E, Ix> {
    /// Partitions the graph's nodes by directed Louvain, with default configuration: local
    /// moving phases (at most 256) are repeated until they no longer improve modularity (on
    /// large graphs, until they improve it little, see [`DirectedLouvain`]).
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

    /// Partitions the graph's nodes by directed Louvain with Leiden's refinement (see
    /// [`Refinement::Leiden`]), with default configuration: communities are connected.
    pub fn new_leiden(g: &'a DiGraph<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_refinement(g, DEFAULT_MAX_PASSES, DEFAULT_TOLERANCE, Refinement::Leiden)
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
    ///   node move improves modularity. Large graphs' phases are held to GVE-Louvain's
    ///   thresholds too, whichever is larger (see [`DirectedLouvain`]).
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a DiGraph<N, E, Ix>,
        max_passes: usize,
        tolerance: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_refinement(g, max_passes, tolerance, Refinement::None)
    }

    /// Partitions the graph's nodes by directed Louvain, its communities refined or not before
    /// they're aggregated (see [`Refinement`]), with `max_passes` and `tolerance` as
    /// [`new_with_conf`](Self::new_with_conf)'s.
    pub fn new_with_refinement(
        g: &'a DiGraph<N, E, Ix>,
        max_passes: usize,
        tolerance: f64,
        refinement: Refinement,
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
            sweeps: 0,
        };
        match refinement {
            Refinement::None => louvain.compute(max_passes, tolerance)?,
            Refinement::Leiden => louvain.compute_leiden(max_passes, tolerance)?,
        }
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

    /// Returns the number of sweeps over the nodes local moving took, on the graph and its
    /// supergraphs: how long the algorithm took to converge, independently of the machine.
    pub fn sweeps(&self) -> usize {
        self.sweeps
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
        let g = self.g;
        let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
        let scratch = Scratch::new(&name, 0);
        let base = BaseArcs::new(g, &scratch)?;
        // total arc weight
        let m = base.m;
        if m == 0 {
            return self.all_alone();
        }

        let threads = g.thread_num().max(1);
        // graphs large enough for parallel local moving have every phase held to GVE-Louvain's
        // stopping rules (see `local_move`), their supergraphs' too
        let gve = parallel_threads(node_count, threads) > 1;
        let mut part = scratch.singletons(node_count, threads)?;
        let mut passes = 0;
        // modularity gained so far
        let mut total_gain = 0.0;
        'outer: while passes < max_passes {
            // local moving on the graph itself: from singletons at first, later refining the
            // partition found on the supergraphs
            // (each phase's gain compared to its own tolerance: parallel phases' scale down by
            // pass, see `local_move`)
            let (moves, gain, phase_tolerance, sweeps) = local_move(
                &base, &mut part, m, tolerance, gve, passes, threads, &scratch,
            )?;
            passes += 1;
            self.sweeps += sweeps;
            let mut count = renumber(&mut part, threads, &scratch)?;
            // parallel phases refining the partition found keep finding a little more (their
            // moves judged on communities as they were): they end once that's a negligible part
            // of what was gained
            if moves == 0
                || gain <= phase_tolerance
                || (gve && total_gain > 0.0 && gain <= REFINEMENT_FRACTION * total_gain)
            {
                break;
            }
            total_gain += gain;
            let mut level = Supergraph::aggregate(&base, &part, count, threads, &scratch)?;
            loop {
                if passes >= max_passes {
                    break 'outer;
                }
                let mut comm = scratch.singletons(level.len(), threads)?;
                let (moves, gain, phase_tolerance, sweeps) = local_move(
                    &level, &mut comm, m, tolerance, gve, passes, threads, &scratch,
                )?;
                passes += 1;
                self.sweeps += sweeps;
                if moves == 0 {
                    break;
                }
                total_gain += gain;
                count = renumber(&mut comm, threads, &scratch)?;
                {
                    let p = SharedSliceMut::<usize>::new(part.as_mut_ptr(), part.len());
                    let comm = &comm;
                    for_each(threads, part.len(), &|u| {
                        let mut p = p;
                        *p.get_mut(u) = comm[*p.get(u)];
                    })?;
                }
                if gain <= phase_tolerance || (gve && gain <= REFINEMENT_FRACTION * total_gain) {
                    break;
                }
                level = Supergraph::aggregate(&level, &comm, count, threads, &scratch)?;
            }
        }

        // numbered by smallest node
        // communities split into their (weakly) connected parts (numbered by smallest node):
        // connected, and each split raising modularity
        self.community_count = super::split_disconnected(
            threads,
            base.out_offsets.as_slice(),
            base.out_neighbours.as_slice(),
            &mut part,
        )?;
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

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedLouvain<'a, N, E, Ix> {
    /// The result when no arc weighs anything (no arcs, or all of weight `0`): no move gains
    /// anything, and every node stays in a community of its own.
    fn all_alone(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (u, c) in self.communities.as_mut_slice().iter_mut().enumerate() {
            *c = Ix::new(u);
        }
        self.community_count = self.g.size();
        self.communities.flush_async()?;
        self.modularity = 0.0;
        Ok(())
    }

    /// [`compute`](Self::compute), with Leiden's refinement: each pass's communities (its
    /// *bounds*) are refined into connected sub-communities (see [`refine`]), which are
    /// aggregated, each supernode starting the next pass in its bound.
    fn compute_leiden(
        &mut self,
        max_passes: usize,
        tolerance: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        if node_count == 0 {
            return Ok(());
        }
        let g = self.g;
        let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
        let scratch = Scratch::new(&name, 0);
        let base = BaseArcs::new(g, &scratch)?;
        // total arc weight
        let m = base.m;
        if m == 0 {
            return self.all_alone();
        }

        let threads = g.thread_num().max(1);
        let gve = parallel_threads(node_count, threads) > 1;
        // each node's community, as local moving finds them
        let mut part = scratch.singletons(node_count, threads)?;
        let mut passes = 0;
        let mut total_gain = 0.0;
        'outer: while passes < max_passes {
            let (moves, gain, phase_tolerance, sweeps) = local_move(
                &base, &mut part, m, tolerance, gve, passes, threads, &scratch,
            )?;
            passes += 1;
            self.sweeps += sweeps;
            renumber(&mut part, threads, &scratch)?;
            if moves == 0
                || gain <= phase_tolerance
                || (gve && total_gain > 0.0 && gain <= REFINEMENT_FRACTION * total_gain)
            {
                break;
            }
            total_gain += gain;
            // each node's supernode, and the supernodes' communities: refined, unless refining
            // shrinks the graph too little (its communities, then, as Louvain's)
            let (mut level, mut comm, mut node_of) = {
                let mut sub = scratch.singletons(node_count, threads)?;
                refine(&base, &part, &mut sub, m, threads, &scratch)?;
                let count = renumber(&mut sub, threads, &scratch)?;
                if count as f64 <= AGGREGATION_TOLERANCE * node_count as f64 {
                    let level = Supergraph::aggregate(&base, &sub, count, threads, &scratch)?;
                    (
                        level,
                        bounds_of(&sub, &part, count, threads, &scratch)?,
                        sub,
                    )
                } else {
                    drop(sub);
                    let count = part.iter().max().map_or(0, |&c| c + 1);
                    let level = Supergraph::aggregate(&base, &part, count, threads, &scratch)?;
                    let mut node_of = scratch.zeroed::<usize>(node_count)?;
                    node_of.copy_from_slice(&part);
                    (level, scratch.singletons(count, threads)?, node_of)
                }
            };
            loop {
                if passes >= max_passes {
                    break;
                }
                let (moves, gain, phase_tolerance, sweeps) = local_move(
                    &level, &mut comm, m, tolerance, gve, passes, threads, &scratch,
                )?;
                passes += 1;
                self.sweeps += sweeps;
                if moves == 0 {
                    break;
                }
                total_gain += gain;
                // (supergraphs' passes, too, end once they gain a negligible part of what was
                // gained: the last, small and dense, ones take whole sweeps for next to nothing)
                if gain <= phase_tolerance || (gve && gain <= REFINEMENT_FRACTION * total_gain) {
                    break;
                }
                let mut sub = scratch.singletons(level.len(), threads)?;
                refine(&level, &comm, &mut sub, m, threads, &scratch)?;
                let count = renumber(&mut sub, threads, &scratch)?;
                if count as f64 <= AGGREGATION_TOLERANCE * level.len() as f64 {
                    let next = Supergraph::aggregate(&level, &sub, count, threads, &scratch)?;
                    comm = bounds_of(&sub, &comm, count, threads, &scratch)?;
                    compose(&mut node_of, &sub, threads)?;
                    level = next;
                } else {
                    // refined too little: the communities aggregated, as Louvain's
                    drop(sub);
                    let count = renumber(&mut comm, threads, &scratch)?;
                    if count == level.len() {
                        break;
                    }
                    let next = Supergraph::aggregate(&level, &comm, count, threads, &scratch)?;
                    compose(&mut node_of, &comm, threads)?;
                    comm = scratch.singletons(count, threads)?;
                    level = next;
                }
            }
            // the nodes' communities, as the last pass found them
            compose(&mut node_of, &comm, threads)?;
            part = node_of;
            if passes >= max_passes {
                break 'outer;
            }
        }

        // communities split into their (weakly) connected parts (numbered by smallest node):
        // connected, and each split raising modularity
        self.community_count = super::split_disconnected(
            threads,
            base.out_offsets.as_slice(),
            base.out_neighbours.as_slice(),
            &mut part,
        )?;
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

/// Shrink in supernodes below which Leiden's passes stop aggregating (GVE-Leiden's): refinement
/// keeping more than this fraction of a graph's nodes apart.
const AGGREGATION_TOLERANCE: f64 = 0.8;

/// `map[u] = by[map[u]]` for every `u`, in parallel.
fn compose(
    map: &mut ScratchMem<usize>,
    by: &[usize],
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = SharedSliceMut::<usize>::new(map.as_mut_ptr(), map.len());
    for_each(threads, map.len(), &|u| {
        let mut p = p;
        *p.get_mut(u) = by[*p.get(u)];
    })
}

/// The supernodes' communities: each supernode's (`sub[u]`, of `count`) the community
/// (`bound[u]`) of its nodes, labelled by its first supernode (labels smaller than `count`).
fn bounds_of(
    sub: &[usize],
    bound: &[usize],
    count: usize,
    threads: usize,
    scratch: &Scratch,
) -> Result<ScratchMem<usize>, Box<dyn std::error::Error>> {
    let n = sub.len();
    let first = scratch.filled(n, usize::MAX)?;
    let first_of = atomics(&first);
    for_each(threads, n, &|u| {
        first_of[bound[u]].fetch_min(sub[u], Ordering::Relaxed);
    })?;
    let mut comm = scratch.zeroed::<usize>(count)?;
    let c = SharedSliceMut::<usize>::new(comm.as_mut_ptr(), count);
    for_each(threads, n, &|u| {
        let mut c = c;
        *c.get_mut(sub[u]) = first[bound[u]];
    })?;
    Ok(comm)
}

/// Leiden's refinement of `a`'s communities `bound` into sub-communities `sub` (from
/// singletons), as GVE-Leiden's: each node still alone in its sub-community joins the
/// neighbouring sub-community of its community (by in- or out-arcs) where it gains most
/// modularity, if it gains any, once, in parallel. A node leaves its sub-community only while it
/// is alone in it (by a compare-and-swap of its size, from 1), so nodes join sub-communities only
/// through an arc to one of their nodes, which stay: sub-communities are (weakly) connected.
///
/// Returns the number of nodes that joined another's sub-community.
fn refine<A: Arcs>(
    a: &A,
    bound: &[usize],
    sub: &mut [usize],
    m: u64,
    threads: usize,
    scratch: &Scratch,
) -> Result<usize, Box<dyn std::error::Error>> {
    let n = a.len();
    let threads = parallel_threads(n, threads);
    let sigma_mem = scratch.zeroed::<u64>(2 * n)?;
    let size_mem = scratch.filled::<usize>(n, 1)?;
    let (sigma, size) = (atomics(&sigma_mem), atomics(&size_mem));
    let sigma_out = |c: usize| &sigma[2 * c];
    let sigma_in = |c: usize| &sigma[2 * c + 1];
    for_each(threads, n, &|i| {
        sigma_out(i).store(a.k_out(i), Ordering::Relaxed);
        sigma_in(i).store(a.k_in(i), Ordering::Relaxed);
    })?;
    // SAFETY: `AtomicUsize` has `usize`'s layout, and `sub` is borrowed mutably for the phase
    let sub: &[AtomicUsize] =
        unsafe { std::slice::from_raw_parts(sub.as_mut_ptr() as *const AtomicUsize, n) };
    let merges = chunked(
        threads,
        n,
        &|begin, end, (acc, merges): &mut (Accumulator, usize)| {
            for i in begin..end {
                let c = sub[i].load(Ordering::Relaxed);
                if size[c].load(Ordering::Relaxed) != 1 {
                    continue;
                }
                let b = bound[i];
                let mut link = |j: usize, w: u64| {
                    if j != i && bound[j] == b {
                        acc.add(sub[j].load(Ordering::Relaxed), w);
                    }
                };
                a.out_arcs(i, &mut link);
                a.in_arcs(i, &mut link);
                let (k_out, k_in) = (a.k_out(i) as i128, a.k_in(i) as i128);
                // its gain for joining `d`, alone (its own sub-community's score is `0`)
                let (mut best, mut best_score) = (c, 0i128);
                for (&d, &w) in acc.keys.iter().zip(&acc.weights) {
                    if d == c {
                        continue;
                    }
                    let score = m as i128 * w as i128
                        - (k_out * sigma_in(d).load(Ordering::Relaxed) as i128
                            + k_in * sigma_out(d).load(Ordering::Relaxed) as i128);
                    if score > best_score || (score == best_score && best != c && d < best) {
                        (best, best_score) = (d, score);
                    }
                }
                acc.clear();
                if best != c
                    && size[c]
                        .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                {
                    size[best].fetch_add(1, Ordering::Relaxed);
                    sigma_out(c).fetch_sub(k_out as u64, Ordering::Relaxed);
                    sigma_in(c).fetch_sub(k_in as u64, Ordering::Relaxed);
                    sigma_out(best).fetch_add(k_out as u64, Ordering::Relaxed);
                    sigma_in(best).fetch_add(k_in as u64, Ordering::Relaxed);
                    sub[i].store(best, Ordering::Relaxed);
                    *merges += 1;
                }
            }
        },
        &|| (Accumulator::for_keys(n), 0usize),
    )?;
    Ok(merges.iter().map(|(_, m)| m).sum())
}

/// Nodes per unit of work, taken by threads as they go: degrees vary too much for even splits.
/// Tiny in tests, for the small graphs' nodes to be moved by several threads at once.
const CHUNK: usize = if cfg!(test) { 4 } else { 1 << 10 };

/// Sweeps of a parallel local moving phase, at most (GVE-Louvain's).
const PARALLEL_SWEEPS: usize = 20;

/// Gain below which a sweep ends a parallel local moving phase, on the first pass, and how much
/// it drops each pass after (GVE-Louvain's threshold scaling).
const PARALLEL_TOLERANCE: f64 = 1e-2;
const PARALLEL_TOLERANCE_DROP: f64 = 10.0;

/// Fraction of the modularity gained so far below which a parallel phase refining the partition
/// found on the graph itself ends the algorithm, and a supergraph's pass, its level.
const REFINEMENT_FRACTION: f64 = 1e-3;

/// Sweeps of a local moving phase, at most: asynchronous moves converge, but nothing bounds how
/// fast.
const MAX_SWEEPS: usize = 500;

/// `f` over `0..n` in chunks of [`CHUNK`], taken by `threads` as they go: its results, by thread.
fn chunked<T: Send>(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize, &mut T) + Sync),
    init: &(dyn Fn() -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    chunked_by(threads, n, CHUNK, f, init)
}

/// [`chunked`], in chunks of `chunk` items.
fn chunked_by<T: Send>(
    threads: usize,
    n: usize,
    chunk: usize,
    f: &(dyn Fn(usize, usize, &mut T) + Sync),
    init: &(dyn Fn() -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let next = AtomicUsize::new(0);
    crossbeam::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|_| {
                    let mut state = init();
                    loop {
                        let begin = next.fetch_add(chunk, Ordering::Relaxed);
                        if begin >= n {
                            break state;
                        }
                        f(begin, (begin + chunk).min(n), &mut state);
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// `f` over each of `threads` ranges splitting `0..n`, in parallel: its results, in order.
fn by_range<T: Send>(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize, usize) -> T + Sync),
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let threads = threads.max(1);
    let load = n.div_ceil(threads);
    crossbeam::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let begin = (tid * load).min(n);
                let end = (begin + load).min(n);
                scope.spawn(move |_| f(begin, end))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

/// `f` over `0..n`, in parallel.
fn for_each(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    by_range(threads, n, &|begin, end| (begin..end).for_each(f))?;
    Ok(())
}

/// The total weight by key (community) of a node's arcs, in insertion order (`keys` and
/// `weights`), found by an open addressing table of their indices that grows as needed (at most
/// half full) and is cleared through the entries it holds: constant time per arc, and per key
/// to clear. One per thread.
///
/// With few enough possible keys (supergraphs' communities, soon), indices are kept in an array
/// by key instead (`direct`): in the cache, and without hashing.
#[derive(Default)]
struct Accumulator {
    keys: Vec<usize>,
    weights: Vec<u64>,
    /// Indices into `keys` (`u32::MAX`: empty), `2^bits` of them.
    table: Vec<u32>,
    /// Each key's slot in `table`.
    slots: Vec<usize>,
    bits: u32,
    /// Indices into `keys` by key (`u32::MAX`: empty), if keys are few.
    direct: Vec<u32>,
}

impl Accumulator {
    const EMPTY: u32 = u32::MAX;

    /// An accumulator of keys smaller than `keys`.
    fn for_keys(keys: usize) -> Self {
        let mut acc = Self::default();
        if keys <= DIRECT_KEYS {
            acc.direct = vec![Self::EMPTY; keys];
        }
        acc
    }

    #[inline(always)]
    fn home(&self, key: usize) -> usize {
        ((key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - self.bits)) as usize
    }

    /// The slot holding `key`, or the empty one where it would go.
    #[inline(always)]
    fn find(&self, key: usize) -> usize {
        let mask = self.table.len() - 1;
        let mut slot = self.home(key);
        loop {
            let i = self.table[slot];
            if i == Self::EMPTY || self.keys[i as usize] == key {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }

    fn grow(&mut self) {
        self.bits = (self.bits + 1).max(4);
        self.table.clear();
        self.table.resize(1 << self.bits, Self::EMPTY);
        for i in 0..self.keys.len() {
            let slot = self.find(self.keys[i]);
            self.table[slot] = i as u32;
            self.slots[i] = slot;
        }
    }

    #[inline(always)]
    fn add(&mut self, key: usize, w: u64) {
        if !self.direct.is_empty() {
            match self.direct[key] {
                Self::EMPTY => {
                    self.direct[key] = self.keys.len() as u32;
                    self.keys.push(key);
                    self.weights.push(w);
                }
                i => self.weights[i as usize] += w,
            }
            return;
        }
        if 2 * (self.keys.len() + 1) > self.table.len() {
            self.grow();
        }
        let slot = self.find(key);
        match self.table[slot] {
            Self::EMPTY => {
                self.table[slot] = self.keys.len() as u32;
                self.slots.push(slot);
                self.keys.push(key);
                self.weights.push(w);
            }
            i => self.weights[i as usize] += w,
        }
    }

    fn get(&self, key: usize) -> u64 {
        if !self.direct.is_empty() {
            return match self.direct[key] {
                Self::EMPTY => 0,
                i => self.weights[i as usize],
            };
        }
        if self.table.is_empty() {
            return 0;
        }
        match self.table[self.find(key)] {
            Self::EMPTY => 0,
            i => self.weights[i as usize],
        }
    }

    fn clear(&mut self) {
        if !self.direct.is_empty() {
            for &k in &self.keys {
                self.direct[k] = Self::EMPTY;
            }
        }
        for &slot in &self.slots {
            self.table[slot] = Self::EMPTY;
        }
        self.keys.clear();
        self.weights.clear();
        self.slots.clear();
    }
}

/// Renumbers `comm`'s labels (all smaller than `comm.len()`) densely, by their first node, and
/// returns how many there are: in parallel, from each label's first node and a prefix count of
/// the nodes first in their community.
fn renumber(
    comm: &mut [usize],
    threads: usize,
    scratch: &Scratch,
) -> Result<usize, Box<dyn std::error::Error>> {
    let n = comm.len();
    let first = scratch.filled(n, usize::MAX)?;
    {
        // SAFETY: `AtomicUsize` has `usize`'s layout
        let first: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(first.as_ptr() as *const AtomicUsize, n) };
        let comm = &*comm;
        for_each(threads, n, &|u| {
            first[comm[u]].fetch_min(u, Ordering::Relaxed);
        })?;
    }
    // labels by first node: the number of first nodes before theirs
    let mut ids = scratch.zeroed::<usize>(n)?;
    let counts = by_range(threads, n, &|begin, end| {
        (begin..end).filter(|&u| first[comm[u]] == u).count()
    })?;
    let mut starts = Vec::with_capacity(counts.len());
    let mut count = 0;
    for c in &counts {
        starts.push(count);
        count += c;
    }
    {
        let ids_out = SharedSliceMut::<usize>::new(ids.as_mut_ptr(), n);
        let load = n.div_ceil(threads.max(1));
        let comm = &*comm;
        by_range(threads, n, &|begin, end| {
            let mut ids_out = ids_out;
            let mut id = if begin < end { starts[begin / load] } else { 0 };
            for u in begin..end {
                if first[comm[u]] == u {
                    *ids_out.get_mut(comm[u]) = id;
                    id += 1;
                }
            }
        })?;
    }
    let out = SharedSliceMut::<usize>::new(comm.as_mut_ptr(), n);
    for_each(threads, n, &|u| {
        let mut out = out;
        *out.get_mut(u) = ids[*out.get(u)];
    })?;
    Ok(count)
}

/// A weighted directed graph local moving and aggregation run on: the original graph, or a
/// supergraph of communities.
trait Arcs: Sync {
    /// Number of nodes.
    fn len(&self) -> usize;
    /// Number of (stored) out-arcs, as many as in-arcs: bounds a supergraph's.
    fn arc_count(&self) -> usize;
    /// Total weight of a node's out-arcs (self-loops included).
    fn k_out(&self, i: usize) -> u64;
    /// Total weight of a node's in-arcs (self-loops included).
    fn k_in(&self, i: usize) -> u64;
    /// Number of a node's out-arcs: bounds the out-arcs of a community it's in.
    fn out_degree(&self, i: usize) -> usize;
    /// Number of a node's in-arcs.
    fn in_degree(&self, i: usize) -> usize;
    /// Calls `f(j, w)` for a node's out-arcs `i -> j` of weight `w` (possibly repeated `j`s).
    fn out_arcs(&self, i: usize, f: impl FnMut(usize, u64));
    /// Calls `f(j, w)` for a node's in-arcs `j -> i` of weight `w` (possibly repeated `j`s).
    fn in_arcs(&self, i: usize, f: impl FnMut(usize, u64));
}

/// Largest total arc weight `m` local moving accepts: its gains, `m·w − (k_out Σ_in + k_in Σ_out)`
/// (each term at most `m²`), are summed exactly in `i128`, with room for many.
const MAX_TOTAL_WEIGHT: u64 = 1 << 60;

/// An arc's weight, as an exact integer: what [`DirectedLouvain`] and [`directed_modularity`]
/// compute with.
fn exact_weight<L: graph::E>(label: &L) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    label.integer_weight().ok_or_else(|| {
        format!(
            "error directed modularity is computed exactly, in integers, so arc weights must be non-negative integers, but one is {label:?}"
        )
        .into()
    })
}

/// Checks a graph's total arc weight is within [`MAX_TOTAL_WEIGHT`].
fn check_total_weight(m: u128) -> Result<u64, Box<dyn std::error::Error>> {
    if m > MAX_TOTAL_WEIGHT as u128 {
        return Err(format!(
            "error the arcs' total weight, {m}, exceeds the {MAX_TOTAL_WEIGHT} directed modularity is computed exactly up to: scale the weights down"
        )
        .into());
    }
    Ok(m as u64)
}

/// The original graph's arcs: out-arcs from its CSR, in-arcs from its transpose's. Unweighted
/// arcs (see [`E::is_weighted`](graph::E::is_weighted)) weigh `1`, and nodes' total weights are
/// their degrees; weighted ones weigh their labels' [`integer_weight`](graph::E::integer_weight),
/// and nodes' total weights are summed once, up front.
struct BaseArcs<Ix: graph::IndexType, L: graph::E> {
    out_offsets: SharedSlice<Ix>,
    out_neighbours: SharedSlice<Ix>,
    in_offsets: SharedSlice<Ix>,
    in_neighbours: SharedSlice<Ix>,
    /// Out- and in-arcs' labels, and nodes' total out- and in-weights: only for weighted arcs.
    weights: Option<BaseWeights<L>>,
    /// Total arc weight.
    m: u64,
}

struct BaseWeights<L> {
    out_labels: SharedSlice<L>,
    in_labels: SharedSlice<L>,
    k_out: ScratchMem<u64>,
    k_in: ScratchMem<u64>,
}

impl<Ix: graph::IndexType, L: graph::E> BaseArcs<Ix, L> {
    /// Valid while `g` (which owns its transpose) is. Fails on arc weights local moving can't
    /// compute with exactly (see [`exact_weight`], [`MAX_TOTAL_WEIGHT`]).
    fn new<N: graph::N>(
        g: &DiGraph<N, L, Ix>,
        scratch: &Scratch,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.transpose()?;
        let mut base = Self {
            out_offsets: SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size()),
            out_neighbours: SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width()),
            in_offsets: SharedSlice::<Ix>::new(t.offsets_ptr(), t.offsets_size()),
            in_neighbours: SharedSlice::<Ix>::new(t.neighbours_ptr(), t.width()),
            weights: None,
            m: check_total_weight(g.width() as u128)?,
        };
        if L::is_weighted() {
            let n = g.size();
            let mut w = BaseWeights {
                // (a transpose's arcs keep their labels)
                out_labels: SharedSlice::<L>::new(g.edge_labels_ptr(), g.width()),
                in_labels: SharedSlice::<L>::new(t.edge_labels_ptr(), t.width()),
                k_out: scratch.zeroed(n)?,
                k_in: scratch.zeroed(n)?,
            };
            let threads = g.thread_num().max(1);
            let totals = |offsets: &SharedSlice<Ix>,
                          labels: &SharedSlice<L>,
                          k: &mut ScratchMem<u64>|
             -> Result<u128, Box<dyn std::error::Error>> {
                let k = SharedSliceMut::<u64>::new(k.as_mut_ptr(), n);
                let mut m = 0u128;
                for part in by_range(threads, n, &|begin, end| {
                    let mut k = k;
                    let mut m = 0u128;
                    for i in begin..end {
                        let mut k_i = 0u128;
                        for e in Self::range(offsets, i) {
                            k_i += exact_weight(labels.get(e))? as u128;
                        }
                        // (within the total's bound, checked below, if the total is)
                        *k.get_mut(i) = k_i.min(u64::MAX as u128) as u64;
                        m += k_i;
                    }
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(m)
                })? {
                    m +=
                        part.map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                }
                Ok(m)
            };
            let m = totals(&base.out_offsets, &w.out_labels, &mut w.k_out)?;
            // the same arcs, seen from their heads
            totals(&base.in_offsets, &w.in_labels, &mut w.k_in)?;
            base.m = check_total_weight(m)?;
            base.weights = Some(w);
        }
        Ok(base)
    }

    fn range(offsets: &SharedSlice<Ix>, i: usize) -> std::ops::Range<usize> {
        offsets.get(i).index()..offsets.get(i + 1).index()
    }
}

impl<Ix: graph::IndexType, L: graph::E> Arcs for BaseArcs<Ix, L> {
    fn len(&self) -> usize {
        self.out_offsets.len() - 1
    }

    fn arc_count(&self) -> usize {
        self.out_neighbours.len()
    }

    fn k_out(&self, i: usize) -> u64 {
        match &self.weights {
            Some(w) => w.k_out[i],
            None => Self::range(&self.out_offsets, i).len() as u64,
        }
    }

    fn k_in(&self, i: usize) -> u64 {
        match &self.weights {
            Some(w) => w.k_in[i],
            None => Self::range(&self.in_offsets, i).len() as u64,
        }
    }

    fn out_degree(&self, i: usize) -> usize {
        Self::range(&self.out_offsets, i).len()
    }

    fn in_degree(&self, i: usize) -> usize {
        Self::range(&self.in_offsets, i).len()
    }

    fn out_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        let range = Self::range(&self.out_offsets, i);
        match &self.weights {
            // (weights checked in `new`)
            Some(w) => {
                for e in range {
                    let weight = w.out_labels.get(e).integer_weight().unwrap_or(0);
                    f(self.out_neighbours.get(e).index(), weight);
                }
            }
            None => {
                for j in &self.out_neighbours.as_slice()[range] {
                    f(j.index(), 1);
                }
            }
        }
    }

    fn in_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        let range = Self::range(&self.in_offsets, i);
        match &self.weights {
            Some(w) => {
                for e in range {
                    let weight = w.in_labels.get(e).integer_weight().unwrap_or(0);
                    f(self.in_neighbours.get(e).index(), weight);
                }
            }
            None => {
                for j in &self.in_neighbours.as_slice()[range] {
                    f(j.index(), 1);
                }
            }
        }
    }
}

/// A weighted directed supergraph, as CSRs of `(neighbour, weight)` out- and in-arcs (one per
/// neighbour, self-loops included), in scratch files. Each community's arcs are written within
/// room reserved for its nodes' (as many as their arcs, in parallel, from prefix sums), so rows
/// have their own length.
struct Supergraph {
    out_offsets: ScratchMem<usize>,
    out_lens: ScratchMem<usize>,
    out_arcs: ScratchMem<(usize, u64)>,
    in_offsets: ScratchMem<usize>,
    in_lens: ScratchMem<usize>,
    in_arcs: ScratchMem<(usize, u64)>,
    k_out: ScratchMem<u64>,
    k_in: ScratchMem<u64>,
    arc_count: usize,
}

/// Exclusive prefix sums of `f(c)` over `0..count`, in parallel: the `count + 1` offsets.
fn offsets_of(
    threads: usize,
    count: usize,
    f: &(dyn Fn(usize) -> usize + Sync),
    out: &mut [usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let sums = by_range(threads, count, &|begin, end| {
        (begin..end).map(f).sum::<usize>()
    })?;
    let mut starts = Vec::with_capacity(sums.len());
    let mut at = 0;
    for s in sums {
        starts.push(at);
        at += s;
    }
    out[count] = at;
    let load = count.div_ceil(threads.max(1));
    let o = SharedSliceMut::<usize>::new(out.as_mut_ptr(), count + 1);
    by_range(threads, count, &|begin, end| {
        let mut o = o;
        let mut at = if begin < end { starts[begin / load] } else { 0 };
        for c in begin..end {
            *o.get_mut(c) = at;
            at += f(c);
        }
    })?;
    Ok(())
}

impl Supergraph {
    /// The supergraph of `a`'s communities `comm` (dense, `0..count`): an arc `c -> d` weighs
    /// as much as all arcs from `c`'s nodes to `d`'s.
    fn aggregate<A: Arcs>(
        a: &A,
        comm: &[usize],
        count: usize,
        threads: usize,
        scratch: &Scratch,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = comm.len();
        // nodes by community, and their communities' arcs, at most theirs
        let sizes = scratch.zeroed::<usize>(count)?;
        let out_room = scratch.zeroed::<usize>(count)?;
        let in_room = scratch.zeroed::<usize>(count)?;
        {
            // SAFETY: `AtomicUsize` has `usize`'s layout
            let atomic = |m: &ScratchMem<usize>| -> &[AtomicUsize] {
                unsafe { std::slice::from_raw_parts(m.as_ptr() as *const AtomicUsize, count) }
            };
            let (sz, o, i) = (atomic(&sizes), atomic(&out_room), atomic(&in_room));
            for_each(threads, n, &|u| {
                let c = comm[u];
                sz[c].fetch_add(1, Ordering::Relaxed);
                o[c].fetch_add(a.out_degree(u), Ordering::Relaxed);
                i[c].fetch_add(a.in_degree(u), Ordering::Relaxed);
            })?;
        }
        let mut start = scratch.zeroed::<usize>(count + 1)?;
        offsets_of(threads, count, &|c| sizes[c], &mut start)?;
        let mut members = scratch.zeroed::<usize>(n)?;
        {
            let next = scratch.zeroed::<usize>(count)?;
            // SAFETY: `AtomicUsize` has `usize`'s layout
            let next: &[AtomicUsize] =
                unsafe { std::slice::from_raw_parts(next.as_ptr() as *const AtomicUsize, count) };
            let m = SharedSliceMut::<usize>::new(members.as_mut_ptr(), n);
            let start = &start;
            for_each(threads, n, &|u| {
                let mut m = m;
                let c = comm[u];
                *m.get_mut(start[c] + next[c].fetch_add(1, Ordering::Relaxed)) = u;
            })?;
        }

        let mut s = Self {
            out_offsets: scratch.zeroed(count + 1)?,
            out_lens: scratch.zeroed(count)?,
            out_arcs: scratch.zeroed(a.arc_count().max(1))?,
            in_offsets: scratch.zeroed(count + 1)?,
            in_lens: scratch.zeroed(count)?,
            in_arcs: scratch.zeroed(a.arc_count().max(1))?,
            k_out: scratch.zeroed(count)?,
            k_in: scratch.zeroed(count)?,
            arc_count: 0,
        };
        offsets_of(threads, count, &|c| out_room[c], &mut s.out_offsets)?;
        offsets_of(threads, count, &|c| in_room[c], &mut s.in_offsets)?;
        drop((sizes, out_room, in_room));

        let arcs_len = s.out_arcs.len();
        let (out_arcs, in_arcs) = (
            SharedSliceMut::<(usize, u64)>::new(s.out_arcs.as_mut_ptr(), arcs_len),
            SharedSliceMut::<(usize, u64)>::new(s.in_arcs.as_mut_ptr(), arcs_len),
        );
        let (out_lens, in_lens) = (
            SharedSliceMut::<usize>::new(s.out_lens.as_mut_ptr(), count),
            SharedSliceMut::<usize>::new(s.in_lens.as_mut_ptr(), count),
        );
        let (k_out, k_in) = (
            SharedSliceMut::<u64>::new(s.k_out.as_mut_ptr(), count),
            SharedSliceMut::<u64>::new(s.k_in.as_mut_ptr(), count),
        );
        let (out_offsets, in_offsets) = (&s.out_offsets, &s.in_offsets);
        let (start, members) = (&start, &members);
        // each community's arcs, sorted by neighbour
        let write =
            |acc: &mut Accumulator, mut arcs: SharedSliceMut<(usize, u64)>, at: usize| -> usize {
                let mut row: Vec<(usize, u64)> = acc
                    .keys
                    .iter()
                    .copied()
                    .zip(acc.weights.iter().copied())
                    .collect();
                row.sort_unstable();
                for (k, &e) in row.iter().enumerate() {
                    *arcs.get_mut(at + k) = e;
                }
                acc.clear();
                row.len()
            };
        // communities in chunks small enough for every thread to take several: supergraphs'
        // are few, and heavy
        let totals = chunked_by(
            threads,
            count,
            (count / (threads * 16)).clamp(1, CHUNK),
            &|begin, end, (acc, total): &mut (Accumulator, usize)| {
                let (mut out_lens, mut in_lens, mut k_out, mut k_in) =
                    (out_lens, in_lens, k_out, k_in);
                for c in begin..end {
                    let nodes = &members[start[c]..start[c + 1]];
                    let (mut ko, mut ki) = (0, 0);
                    for &i in nodes {
                        ko += a.k_out(i);
                        a.out_arcs(i, |j, w| acc.add(comm[j], w));
                    }
                    let len = write(acc, out_arcs, out_offsets[c]);
                    *out_lens.get_mut(c) = len;
                    *total += len;
                    for &i in nodes {
                        ki += a.k_in(i);
                        a.in_arcs(i, |j, w| acc.add(comm[j], w));
                    }
                    *in_lens.get_mut(c) = write(acc, in_arcs, in_offsets[c]);
                    *k_out.get_mut(c) = ko;
                    *k_in.get_mut(c) = ki;
                }
            },
            &|| (Accumulator::for_keys(count), 0usize),
        )?;
        s.arc_count = totals.iter().map(|(_, t)| t).sum();
        Ok(s)
    }
}

impl Arcs for Supergraph {
    fn len(&self) -> usize {
        self.k_out.len()
    }

    fn arc_count(&self) -> usize {
        self.arc_count
    }

    fn k_out(&self, i: usize) -> u64 {
        self.k_out[i]
    }

    fn k_in(&self, i: usize) -> u64 {
        self.k_in[i]
    }

    fn out_degree(&self, i: usize) -> usize {
        self.out_lens[i]
    }

    fn in_degree(&self, i: usize) -> usize {
        self.in_lens[i]
    }

    fn out_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        let at = self.out_offsets[i];
        for &(j, w) in &self.out_arcs[at..at + self.out_lens[i]] {
            f(j, w);
        }
    }

    fn in_arcs(&self, i: usize, mut f: impl FnMut(usize, u64)) {
        let at = self.in_offsets[i];
        for &(j, w) in &self.in_arcs[at..at + self.in_lens[i]] {
            f(j, w);
        }
    }
}

/// `mem` as atomics.
fn atomics<T>(mem: &ScratchMem<T>) -> &[<T as AtomicOf>::Atomic]
where
    T: AtomicOf,
{
    // SAFETY: atomics have their integer's layout
    unsafe { std::slice::from_raw_parts(mem.as_ptr() as *const T::Atomic, mem.len()) }
}

/// The atomic of an integer type.
trait AtomicOf {
    type Atomic;
}
impl AtomicOf for usize {
    type Atomic = AtomicUsize;
}
impl AtomicOf for u64 {
    type Atomic = AtomicU64;
}
impl AtomicOf for u8 {
    type Atomic = AtomicU8;
}

/// Nodes from which local moving prefetches communities (of `8` bytes each): fewer fit in the
/// caches.
const PREFETCH_NODES: usize = 1 << 19;

/// Arcs from which a graph too small to move asynchronously is moved by color classes.
const COLORED_ARCS: usize = if cfg!(test) { 64 } else { 1 << 20 };

/// Nodes per color for each thread moving them by color classes, on average.
const COLORED_NODES_PER_THREAD: usize = if cfg!(test) { 2 } else { 8 };

/// Keys an [`Accumulator`] indexes directly, at most: `4` bytes each, per thread.
const DIRECT_KEYS: usize = 1 << 16;

/// Locks the labels of empty communities are split across, by label.
const FREE_SHARDS: usize = 64;

/// Asks for the cache line holding `*p` (a hint: never faults).
#[inline(always)]
fn prefetch<T>(p: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE is part of x86_64's baseline, and prefetches are hints, valid for any address
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T0>(p as *const T as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// A barrier threads spin at (yielding after a while): color classes take microseconds each,
/// less than putting threads to sleep and waking them up, hundreds of times a sweep.
struct SpinBarrier {
    threads: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(threads: usize) -> Self {
        Self {
            threads,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.threads {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return;
        }
        let mut spins = 0u32;
        while self.generation.load(Ordering::Acquire) == generation {
            if spins < 1 << 12 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// A graph's nodes by color, no two linked nodes of the same: nodes of a color can be moved
/// at once without any of them joining another's community as it leaves it (Lu, Halappanavar
/// and Kalyanaraman's ["Parallel heuristics for scalable community detection"](https://doi.org/10.1016/j.parco.2015.03.003)).
struct ColorClasses {
    /// Nodes, color after color.
    order: Vec<usize>,
    /// Each color's first node in `order`, and its end.
    starts: Vec<usize>,
    /// The threads moving each color's nodes.
    threads: usize,
}

impl ColorClasses {
    /// A sweep over the nodes, color after color, each color's nodes moved by `self.threads` at
    /// once (the same threads throughout, waiting for each other between colors): their moves
    /// and gains.
    fn sweep(
        &self,
        visit: &(dyn Fn(usize, &mut Accumulator, &mut usize, &mut i128) + Sync),
        keys: usize,
    ) -> Result<Vec<(Accumulator, usize, i128)>, Box<dyn std::error::Error>> {
        let threads = self.threads;
        let colors = self.starts.len() - 1;
        let next: Vec<AtomicUsize> = (0..colors).map(|_| AtomicUsize::new(0)).collect();
        let barrier = SpinBarrier::new(threads);
        crossbeam::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|_| {
                        let (mut acc, mut moves, mut gain) = (Accumulator::for_keys(keys), 0, 0);
                        for c in 0..colors {
                            let (begin, end) = (self.starts[c], self.starts[c + 1]);
                            let chunk = ((end - begin) / (threads * 4)).max(1);
                            loop {
                                let at = begin + next[c].fetch_add(chunk, Ordering::Relaxed);
                                if at >= end {
                                    break;
                                }
                                for &i in &self.order[at..(at + chunk).min(end)] {
                                    visit(i, &mut acc, &mut moves, &mut gain);
                                }
                            }
                            barrier.wait();
                        }
                        (acc, moves, gain)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
    }
}

/// `a`'s nodes by color (in- and out-neighbours of different colors), colored greedily in
/// parallel (each uncolored node taking the smallest color none of its neighbours has, the
/// larger of two linked nodes of a color uncolored again, until none are), or `None` if colors
/// would hold too few nodes each for `threads` to share (dense graphs).
fn color_classes<A: Arcs>(
    a: &A,
    threads: usize,
) -> Result<Option<ColorClasses>, Box<dyn std::error::Error>> {
    const NONE: usize = usize::MAX;
    let n = a.len();
    let color: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(NONE)).collect();
    let mut uncolored: Vec<usize> = (0..n).collect();
    while !uncolored.is_empty() {
        let work = &uncolored;
        // colors taken by a node's neighbours, marked with the node (+ 1)
        chunked_by(
            threads,
            work.len(),
            (work.len() / (threads * 8)).clamp(1, CHUNK),
            &|begin, end, taken: &mut Vec<usize>| {
                for &i in &work[begin..end] {
                    let mut mark = |j: usize, _| {
                        let c = color[j].load(Ordering::Relaxed);
                        if j != i && c != NONE {
                            if c >= taken.len() {
                                taken.resize(c + 1, 0);
                            }
                            taken[c] = i + 1;
                        }
                    };
                    a.out_arcs(i, &mut mark);
                    a.in_arcs(i, &mut mark);
                    let c = (0..).find(|&c| taken.get(c) != Some(&(i + 1))).unwrap();
                    color[i].store(c, Ordering::Relaxed);
                }
            },
            &Vec::new,
        )?;
        let conflicts = chunked_by(
            threads,
            work.len(),
            (work.len() / (threads * 8)).clamp(1, CHUNK),
            &|begin, end, again: &mut Vec<usize>| {
                for &i in &work[begin..end] {
                    let c = color[i].load(Ordering::Relaxed);
                    let mut clash = false;
                    let mut check = |j: usize, _| {
                        clash |= j < i && color[j].load(Ordering::Relaxed) == c;
                    };
                    a.out_arcs(i, &mut check);
                    a.in_arcs(i, &mut check);
                    if clash {
                        again.push(i);
                    }
                }
            },
            &Vec::new,
        )?;
        uncolored = conflicts.concat();
    }
    let colors = color
        .iter()
        .map(|c| c.load(Ordering::Relaxed) + 1)
        .max()
        .unwrap_or(0);
    // a thread per few nodes of a color, on average: more would mostly wait at the barriers
    // between colors (and on a busy machine, for each other)
    let threads = (n / colors.max(1) / COLORED_NODES_PER_THREAD).min(threads);
    if colors == 0 || threads < 2 {
        return Ok(None);
    }
    let mut starts = vec![0usize; colors + 1];
    for c in &color {
        starts[c.load(Ordering::Relaxed) + 1] += 1;
    }
    for c in 0..colors {
        starts[c + 1] += starts[c];
    }
    let mut at = starts.clone();
    let mut order = vec![0usize; n];
    for (i, c) in color.iter().enumerate() {
        let c = c.load(Ordering::Relaxed);
        order[at[c]] = i;
        at[c] += 1;
    }
    Ok(Some(ColorClasses {
        order,
        starts,
        threads,
    }))
}

/// The threads local moving runs on, on a graph of `n` nodes: one on graphs too small for every
/// thread to take several chunks.
fn parallel_threads(n: usize, threads: usize) -> usize {
    if n < threads * CHUNK * 4 { 1 } else { threads }
}

/// Louvain's local moving phase on `a` (whose arcs weigh `m` in all), from the partition
/// `comm` (labels smaller than `a.len()`), asynchronously, as GVE-Louvain's: threads take nodes
/// as they go, each moving its node to the neighbouring community (or its own) where inserting
/// it gains most modularity, as communities are when it looks. Communities' total in- and
/// out-degrees are atomics, updated as nodes move.
///
/// For node `i` taken out of its community, inserting it into `c` gains
/// `(w_{i→c} + w_{c→i})/m − (k_i^out Σ_c^in + k_i^in Σ_c^out)/m²`; gains are compared as
/// their (integer) multiples by `m²`, so exactly. Ties keep a node where it is, or else go to
/// the smallest community.
///
/// Nodes are pruned (GVE-Louvain's vertex pruning): a sweep visits only those one of whose in-
/// or out-neighbours moved since they were last visited. Once a pruned sweep moves no node, a
/// sweep over all nodes follows (communities' totals change as distant nodes move too), and
/// local moving stops only after one of those moves none: no node then gains by moving to a
/// neighbour's community, as with sequential local moving. It stops too after a sweep gaining
/// no more than `tolerance`, or [`MAX_SWEEPS`] sweeps.
///
/// Returns the number of moves, the modularity gained, the tolerance the phase's sweeps were
/// held to, and the number of sweeps.
fn local_move<A: Arcs>(
    a: &A,
    comm: &mut [usize],
    m: u64,
    tolerance: f64,
    gve: bool,
    pass: usize,
    threads: usize,
    scratch: &Scratch,
) -> Result<(usize, f64, f64, usize), Box<dyn std::error::Error>> {
    let n = a.len();
    // on graphs too small for every thread to take several chunks (supergraphs, soon), nodes
    // aren't moved asynchronously: several threads would mostly move strongly linked nodes at
    // once, back and forth (supergraphs of a few thousand nodes then take hundreds of sweeps, and
    // their communities merge too much). Those with many arcs are moved by color classes
    // instead (see `color_classes`), the others on one thread
    let requested = threads;
    let threads = parallel_threads(n, threads);
    let classes = if threads == 1 && requested > 1 && a.arc_count() >= COLORED_ARCS {
        color_classes(a, requested)?
    } else {
        None
    };
    // communities' total out- and in-degrees side by side (`2c` and `2c + 1`): read together
    // for every candidate community, one cache miss rather than two
    let sigma_mem = scratch.zeroed::<u64>(2 * n)?;
    let size_mem = scratch.zeroed::<usize>(n)?;
    let active_mem = scratch.filled::<u8>(n, 1)?;
    let (sigma, size, active) = (
        atomics(&sigma_mem),
        atomics(&size_mem),
        atomics(&active_mem),
    );
    let sigma_out = |c: usize| &sigma[2 * c];
    let sigma_in = |c: usize| &sigma[2 * c + 1];
    // SAFETY: `AtomicUsize` has `usize`'s layout, and `comm` is borrowed mutably for the phase
    let comm: &[AtomicUsize] =
        unsafe { std::slice::from_raw_parts(comm.as_mut_ptr() as *const AtomicUsize, n) };
    for_each(threads, n, &|i| {
        let c = comm[i].load(Ordering::Relaxed);
        sigma_out(c).fetch_add(a.k_out(i), Ordering::Relaxed);
        sigma_in(c).fetch_add(a.k_in(i), Ordering::Relaxed);
        size[c].fetch_add(1, Ordering::Relaxed);
    })?;
    // labels of no community (a node isolating itself takes one): pushed as communities empty,
    // so the stack may hold labels that were taken again since, which only makes a node
    // isolating itself join a community instead (its gain counted from the community as it is)
    //
    // sharded by label (`FREE_SHARDS` locks): the first sweep, from singletons, empties a
    // community with nearly every move, and one lock taken that often by every thread costs as
    // much as the moves
    let free: Vec<std::sync::Mutex<Vec<usize>>> =
        (0..FREE_SHARDS).map(|_| Default::default()).collect();
    for part in by_range(threads, n, &|begin, end| {
        (begin..end)
            .filter(|&c| size[c].load(Ordering::Relaxed) == 0)
            .collect::<Vec<_>>()
    })? {
        for c in part {
            free[c % FREE_SHARDS].lock().unwrap().push(c);
        }
    }
    // a free label, from node `i`'s shard first
    let pop_free =
        |i: usize| (0..FREE_SHARDS).find_map(|k| free[(i + k) % FREE_SHARDS].lock().unwrap().pop());
    let m_sq = (m as f64) * (m as f64);
    let (mut moves, mut gain) = (0, 0.0);
    // the next node's neighbours' communities are prefetched only when they outgrow the caches:
    // a supergraph's, in them, would only have its heavy nodes' arcs read twice
    let prefetching = n > PREFETCH_NODES;
    // whether every node is active in this sweep
    let mut full = true;
    // asynchronous moves are judged on communities as they were when a thread looked, so they
    // may not converge (on graphs without structure, sweeps keep finding tiny gains): parallel
    // phases stop as GVE-Louvain's do, after a sweep gaining less than a threshold dropping
    // tenfold by pass, or `PARALLEL_SWEEPS` sweeps
    let (max_sweeps, sweep_tolerance) = if gve {
        let scaled = PARALLEL_TOLERANCE / PARALLEL_TOLERANCE_DROP.powi(pass.min(64) as i32);
        (PARALLEL_SWEEPS, tolerance.max(scaled))
    } else {
        (MAX_SWEEPS, tolerance)
    };
    // moves node `i` (if active), the next node's neighbours' communities prefetched meanwhile
    let visit = |i: usize,
                 next: Option<usize>,
                 acc: &mut Accumulator,
                 moves: &mut usize,
                 gain: &mut i128| {
        if active[i].swap(0, Ordering::Relaxed) == 0 {
            return;
        }
        // the next node's neighbours' communities, fetched while this one moves
        if let Some(next) = next {
            a.out_arcs(next, |j, _| prefetch(&comm[j]));
            a.in_arcs(next, |j, _| prefetch(&comm[j]));
        }
        let d = comm[i].load(Ordering::Relaxed);
        let (k_out, k_in) = (a.k_out(i) as i128, a.k_in(i) as i128);
        // w_{i→c} + w_{c→i} for each neighbouring community c, self-loops excepted
        let mut link = |j: usize, w: u64| {
            if j != i {
                acc.add(comm[j].load(Ordering::Relaxed), w);
            }
        };
        a.out_arcs(i, &mut link);
        a.in_arcs(i, &mut link);

        // communities' totals, its own's without it: taken out of it here only, the shared
        // totals changed only if it moves (others would see its community without it meanwhile,
        // and join it, as smaller than it is)
        let score = |c: usize, w: u64| {
            let (mut out, mut into) = (
                sigma_out(c).load(Ordering::Relaxed) as i128,
                sigma_in(c).load(Ordering::Relaxed) as i128,
            );
            if c == d {
                (out, into) = (out - k_out, into - k_in);
            }
            m as i128 * w as i128 - (k_out * into + k_in * out)
        };
        // every candidate's totals asked for at once, before they're read
        for &c in &acc.keys {
            prefetch(&sigma[2 * c]);
        }
        let stay = score(d, acc.get(d));
        let (mut best, mut best_score) = (d, stay);
        for (&c, &w) in acc.keys.iter().zip(&acc.weights) {
            let s = score(c, w);
            if s > best_score || (s == best_score && best != d && c < best) {
                (best, best_score) = (c, s);
            }
        }
        // a community of its own (score 0) beats staying where it is at a loss
        if best_score < 0
            && let Some(empty) = pop_free(i)
        {
            (best, best_score) = (empty, 0);
        }
        if best != d {
            sigma_out(d).fetch_sub(k_out as u64, Ordering::Relaxed);
            sigma_in(d).fetch_sub(k_in as u64, Ordering::Relaxed);
            sigma_out(best).fetch_add(k_out as u64, Ordering::Relaxed);
            sigma_in(best).fetch_add(k_in as u64, Ordering::Relaxed);
            size[best].fetch_add(1, Ordering::Relaxed);
            if size[d].fetch_sub(1, Ordering::Relaxed) == 1 {
                free[d % FREE_SHARDS].lock().unwrap().push(d);
            }
            comm[i].store(best, Ordering::Relaxed);
            *moves += 1;
            *gain += best_score - stay;
            // its neighbours' best communities may have changed
            // (written only if not set already: on small graphs every thread shares the
            // flags' few cache lines, and writes would bounce them between cores)
            let wake = |j: usize, _| {
                if active[j].load(Ordering::Relaxed) == 0 {
                    active[j].store(1, Ordering::Relaxed);
                }
            };
            a.out_arcs(i, wake);
            a.in_arcs(i, wake);
        }
        acc.clear();
    };
    let mut sweeps = 0;
    for _ in 0..max_sweeps {
        sweeps += 1;
        let parts = match &classes {
            None => chunked(
                threads,
                n,
                &|begin, end, (acc, moves, gain): &mut (Accumulator, usize, i128)| {
                    for i in begin..end {
                        let next = (prefetching && i + 1 < end).then_some(i + 1);
                        visit(i, next, acc, moves, gain);
                    }
                },
                &|| (Accumulator::for_keys(n), 0usize, 0i128),
            )?,
            Some(classes) => {
                classes.sweep(&|i, acc, moves, gain| visit(i, None, acc, moves, gain), n)?
            }
        };
        let sweep_moves: usize = parts.iter().map(|p| p.1).sum();
        let sweep_gain = parts.iter().map(|p| p.2).sum::<i128>() as f64 / m_sq;
        moves += sweep_moves;
        gain += sweep_gain;
        if sweep_moves == 0 {
            if full {
                break;
            }
            for_each(threads, n, &|i| active[i].store(1, Ordering::Relaxed))?;
            full = true;
            continue;
        }
        if sweep_gain <= sweep_tolerance {
            break;
        }
        full = false;
    }
    Ok((moves, gain, sweep_tolerance, sweeps))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, graph_with_threads, label, random_arcs};

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
    fn directed_louvain_is_a_local_optimum() {
        for (seed, threads) in (0..8u64).zip([1, 2, 4, 8].into_iter().cycle()) {
            let n = 40;
            // a few planted groups plus noise, so that there is structure to find
            let mut arcs = planted(4, 10);
            arcs.extend(random_arcs(n, 30, 300 + seed));
            let g = graph_with_threads(&format!("dl_local_{seed}"), "general", n, &arcs, threads);
            let louvain = DirectedLouvain::new(&g).unwrap();
            let comm = as_usize(louvain.communities());

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
    fn directed_louvain_recovers_planted_communities_whatever_the_threads() {
        let (groups, size) = (40, 12);
        let arcs = planted(groups, size);
        let n = groups * size;
        let expected: Vec<usize> = (0..n).map(|v| v / size).collect();
        for threads in [1, 3, 8, 16] {
            let g = graph_with_threads(
                &format!("dl_planted_t{threads}"),
                "general",
                n,
                &arcs,
                threads,
            );
            for _ in 0..3 {
                let louvain = DirectedLouvain::new(&g).unwrap();
                assert_eq!(
                    as_usize(louvain.communities()),
                    expected,
                    "{threads} threads"
                );
                assert_close(
                    louvain.partition_modularity(),
                    brute_force_modularity(n, &arcs, &expected),
                    "modularity",
                );
            }
        }
    }

    /// Whether every community's nodes are linked by arcs between them (either way).
    fn communities_are_connected(n: usize, arcs: &[(usize, usize)], comm: &[usize]) -> bool {
        let mut adj = vec![vec![]; n];
        for &(u, v) in arcs {
            if comm[u] == comm[v] {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
        let mut seen = vec![false; n];
        let mut reached_community = std::collections::HashSet::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            // a second component of a community already reached: disconnected
            if !reached_community.insert(comm[start]) {
                return false;
            }
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(u) = stack.pop() {
                for &v in &adj[u] {
                    if !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        true
    }

    #[test]
    fn communities_are_always_connected() {
        for (seed, threads) in (0..24u64).zip([1, 2, 4, 8].into_iter().cycle()) {
            let n = 60 + seed as usize * 10;
            // random graphs, and planted groups with noise
            let mut arcs = random_arcs(n, 3 * n, 500 + seed);
            if seed % 2 == 0 {
                arcs.extend(planted(n / 12, 12));
            }
            let g =
                graph_with_threads(&format!("leiden_conn_{seed}"), "general", n, &arcs, threads);
            for refinement in [Refinement::None, Refinement::Leiden] {
                let l = DirectedLouvain::new_with_refinement(&g, 256, 0.0, refinement).unwrap();
                let comm = as_usize(l.communities());
                assert!(
                    communities_are_connected(n, &arcs, &comm),
                    "seed {seed}, {threads} threads, {refinement:?}: a community is disconnected"
                );
                // numbered by smallest node, its modularity the formula's
                assert_eq!(comm, label(n, |u, v| comm[u] == comm[v]));
                assert_close(
                    l.partition_modularity(),
                    brute_force_modularity(n, &arcs, &comm),
                    "q",
                );
                let singletons = brute_force_modularity(n, &arcs, &(0..n).collect::<Vec<_>>());
                assert!(l.partition_modularity() >= singletons);
            }
        }
    }

    #[test]
    fn leiden_recovers_planted_communities() {
        let (groups, size) = (40, 12);
        let arcs = planted(groups, size);
        let n = groups * size;
        let expected: Vec<usize> = (0..n).map(|v| v / size).collect();
        for threads in [1, 3, 8] {
            let g = graph_with_threads(
                &format!("leiden_planted_t{threads}"),
                "general",
                n,
                &arcs,
                threads,
            );
            let leiden = DirectedLouvain::new_leiden(&g).unwrap();
            assert_eq!(
                as_usize(leiden.communities()),
                expected,
                "{threads} threads"
            );
        }
    }

    #[test]
    fn leiden_modularity_is_louvains() {
        // about the same: refinement changes what's aggregated, not what's optimized
        for seed in 0..6u64 {
            let n = 400;
            let mut arcs = planted(20, 20);
            arcs.extend(random_arcs(n, 400, 700 + seed));
            let g = graph_with_threads(&format!("leiden_q_{seed}"), "general", n, &arcs, 4);
            let louvain = DirectedLouvain::new(&g).unwrap().partition_modularity();
            let leiden = DirectedLouvain::new_leiden(&g)
                .unwrap()
                .partition_modularity();
            assert!(
                leiden >= louvain - 0.02,
                "seed {seed}: leiden {leiden} vs louvain {louvain}"
            );
        }
    }

    #[test]
    fn splitting_disconnected_communities() {
        // two triangles and an isolated pair, all labelled one community
        let arcs = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7)];
        let g = graph("split", "general", 9, &arcs);
        let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
        let scratch = Scratch::new(&name, 0);
        let base = BaseArcs::new(&g, &scratch).unwrap();
        let (offsets, neighbours) = (base.out_offsets.as_slice(), base.out_neighbours.as_slice());
        let mut comm = vec![0usize; 9];
        let count = super::super::split_disconnected(3, offsets, neighbours, &mut comm).unwrap();
        assert_eq!((count, comm.clone()), (4, vec![0, 0, 0, 1, 1, 1, 2, 2, 3]));
        assert!(
            brute_force_modularity(9, &arcs, &comm) > brute_force_modularity(9, &arcs, &[0; 9])
        );
        // connected communities stay as they are, renumbered by smallest node
        let mut comm: Vec<usize> = vec![5, 5, 5, 2, 2, 2, 7, 7, 1];
        let count = super::super::split_disconnected(2, offsets, neighbours, &mut comm).unwrap();
        assert_eq!((count, comm), (4, vec![0, 0, 0, 1, 1, 1, 2, 2, 3]));
    }

    #[test]
    fn color_classes_never_hold_linked_nodes() {
        for seed in 0..10u64 {
            let n = 300;
            let arcs = random_arcs(n, 3000, 900 + seed);
            for threads in [2, 4, 8] {
                let g = graph_with_threads(
                    &format!("colors_{seed}_{threads}"),
                    "general",
                    n,
                    &arcs,
                    threads,
                );
                let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
                let scratch = Scratch::new(&name, 0);
                let base = BaseArcs::new(&g, &scratch).unwrap();
                let Some(classes) = color_classes(&base, threads as usize).unwrap() else {
                    continue;
                };
                let mut color = vec![usize::MAX; n];
                for c in 0..classes.starts.len() - 1 {
                    for &i in &classes.order[classes.starts[c]..classes.starts[c + 1]] {
                        assert_eq!(color[i], usize::MAX, "node {i} colored twice");
                        color[i] = c;
                    }
                }
                assert!(color.iter().all(|&c| c != usize::MAX), "a node uncolored");
                for &(u, v) in &arcs {
                    assert!(
                        u == v || color[u] != color[v],
                        "seed {seed}: {u} -> {v} share a color"
                    );
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

    /// A directed graph whose arcs `(u, v, value)` carry `value`s, as written in a `field`
    /// MatrixMarket file, loaded with `T` weights on `threads` threads.
    fn weighted_graph<T: crate::graph::EdgeWeight>(
        name: &str,
        field: &str,
        n: usize,
        arcs: &[(usize, usize, String)],
        threads: u8,
    ) -> Result<
        crate::graph::test_utils::TestGraph<label::VoidLabel, Weighted<T>, usize, Directed>,
        String,
    > {
        use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
        let dir = unique_test_dir("directed_louvain_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut text = format!(
            "%%MatrixMarket matrix coordinate {field} general\n{n} {n} {}\n",
            arcs.len()
        );
        for (u, v, w) in arcs {
            text += &format!("{} {} {w}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let g = DiGraph::<label::VoidLabel, Weighted<T>, usize>::from_file(
            &path,
            Some(format!("dl_weights_{name}_{}", unique_run_tag())),
            Some(threads),
        )
        .map_err(|e| e.to_string());
        let g = match g {
            Ok(g) => g,
            Err(e) => {
                std::fs::remove_dir_all(&dir).ok();
                return Err(e);
            }
        };
        Ok(crate::graph::test_utils::TestGraph::new(g, Some(dir)))
    }

    /// The weighted formula, in `O(n²)`.
    fn brute_force_weighted_modularity(
        n: usize,
        arcs: &[(usize, usize, u64)],
        comm: &[usize],
    ) -> f64 {
        let m: f64 = arcs.iter().map(|a| a.2 as f64).sum();
        if m == 0.0 {
            return 0.0;
        }
        let mut a = vec![vec![0.0; n]; n];
        let (mut k_out, mut k_in) = (vec![0.0; n], vec![0.0; n]);
        for &(u, v, w) in arcs {
            a[u][v] += w as f64;
            k_out[u] += w as f64;
            k_in[v] += w as f64;
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

    fn with_weights(arcs: &[(usize, usize)], seed: u64) -> Vec<(usize, usize, u64)> {
        let mut state = seed;
        arcs.iter()
            .map(|&(u, v)| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // zeros included
                (u, v, (state >> 33) % 20)
            })
            .collect()
    }

    fn as_text(arcs: &[(usize, usize, u64)]) -> Vec<(usize, usize, String)> {
        arcs.iter()
            .map(|&(u, v, w)| (u, v, w.to_string()))
            .collect()
    }

    fn assert_close_rel(a: f64, b: f64, what: &str) {
        assert!((a - b).abs() < 1e-9, "{what}: {a} != {b}");
    }

    /// Weighted modularity is the formula, for any partition, parallel arcs and zero weights
    /// included.
    #[test]
    fn weighted_directed_modularity_is_the_formula() {
        for seed in 0..6u64 {
            let n = 40;
            let arcs = with_weights(&random_arcs(n, 300, 50 + seed), seed);
            let g =
                weighted_graph::<u32>(&format!("wmod_{seed}"), "integer", n, &as_text(&arcs), 3)
                    .unwrap();
            for parts in [1, 3, 7, n] {
                let comm: Vec<usize> = (0..n).map(|u| (u * 7 + seed as usize) % parts).collect();
                let labels: Vec<usize> = comm.clone();
                assert_close_rel(
                    directed_modularity(&g, &labels).unwrap(),
                    brute_force_weighted_modularity(n, &arcs, &comm),
                    &format!("seed {seed}, {parts} parts"),
                );
            }
        }
    }

    /// Scaling every weight alike scales every gain alike: uniform weights find exactly what
    /// unit ones do, with Louvain and Leiden.
    #[test]
    fn uniform_weights_find_what_unit_ones_do() {
        for seed in 0..4u64 {
            let n = 120;
            let arcs = random_arcs(n, 600, 300 + seed);
            let plain = graph_with_threads(&format!("uni_plain_{seed}"), "general", n, &arcs, 1);
            let uniform: Vec<_> = arcs.iter().map(|&(u, v)| (u, v, "7".to_string())).collect();
            let weighted =
                weighted_graph::<u16>(&format!("uni_{seed}"), "integer", n, &uniform, 1).unwrap();
            for refinement in [Refinement::None, Refinement::Leiden] {
                let want =
                    DirectedLouvain::new_with_refinement(&plain, 256, 0.0, refinement).unwrap();
                let got =
                    DirectedLouvain::new_with_refinement(&weighted, 256, 0.0, refinement).unwrap();
                assert_eq!(
                    as_usize(got.communities()),
                    as_usize(want.communities()),
                    "seed {seed}, {refinement:?}"
                );
                assert_close_rel(
                    got.partition_modularity(),
                    want.partition_modularity(),
                    &format!("seed {seed}, {refinement:?}"),
                );
            }
        }
    }

    /// Two directed 4-cliques (arcs both ways), linked both ways between nodes 3 and 4:
    /// unweighted, the cliques are the communities; a heavy enough link pulls its ends together. The modularity reported is
    /// the weighted one.
    #[test]
    fn weights_steer_communities() {
        let mut arcs = vec![];
        for base in [0, 4] {
            for u in 0..4 {
                for v in 0..4 {
                    if u != v {
                        arcs.push((base + u, base + v));
                    }
                }
            }
        }
        arcs.extend([(3, 4), (4, 3)]);
        let weighted: Vec<_> = arcs
            .iter()
            .map(|&(u, v)| {
                (
                    u,
                    v,
                    if (u, v) == (3, 4) || (u, v) == (4, 3) {
                        60
                    } else {
                        1
                    },
                )
            })
            .collect();
        let plain = graph("steer_plain", "general", 8, &arcs);
        let g = weighted_graph::<u64>("steer", "integer", 8, &as_text(&weighted), 3).unwrap();
        for refinement in [Refinement::None, Refinement::Leiden] {
            let unweighted = DirectedLouvain::new_with_refinement(&plain, 256, 0.0, refinement)
                .unwrap()
                .communities()
                .to_vec();
            assert_ne!(unweighted[3], unweighted[4], "{refinement:?}");
            let l = DirectedLouvain::new_with_refinement(&g, 256, 0.0, refinement).unwrap();
            let comm = as_usize(l.communities());
            assert_eq!(comm[3], comm[4], "{refinement:?}: {comm:?}");
            assert_close_rel(
                l.partition_modularity(),
                brute_force_weighted_modularity(8, &weighted, &comm),
                &format!("{refinement:?}"),
            );
        }
    }

    /// Integral floats are integers; other weights directed modularity can't be exact with are
    /// refused, by Louvain and by the modularity alike, saying why.
    #[test]
    fn weights_must_be_exact_integers() {
        let arcs = |w: &str| {
            vec![
                (0, 1, w.to_string()),
                (1, 2, "1".to_string()),
                (2, 0, "1".to_string()),
            ]
        };
        let ints = weighted_graph::<u32>("exact_int", "integer", 3, &arcs("3"), 2).unwrap();
        let floats = weighted_graph::<f64>("exact_float", "real", 3, &arcs("3.0"), 2).unwrap();
        assert_close_rel(
            DirectedLouvain::new(&floats)
                .unwrap()
                .partition_modularity(),
            DirectedLouvain::new(&ints).unwrap().partition_modularity(),
            "3.0 is 3",
        );
        let check = |err: String, expected: &str| assert!(err.contains(expected), "{err}");
        let fractional = weighted_graph::<f64>("exact_half", "real", 3, &arcs("0.5"), 2).unwrap();
        let negative = weighted_graph::<i32>("exact_neg", "integer", 3, &arcs("-2"), 2).unwrap();
        let huge = weighted_graph::<u64>(
            "exact_huge",
            "integer",
            3,
            &arcs(&(1u64 << 60).to_string()),
            2,
        )
        .unwrap();
        for (what, err) in [
            (
                "fractional",
                DirectedLouvain::new(&fractional)
                    .err()
                    .map(|e| e.to_string()),
            ),
            (
                "negative",
                DirectedLouvain::new(&negative).err().map(|e| e.to_string()),
            ),
        ] {
            check(
                err.unwrap_or_else(|| panic!("{what}: ran")),
                "non-negative integers",
            );
        }
        check(
            DirectedLouvain::new(&huge).err().unwrap().to_string(),
            "exceeds",
        );
        let comm = [0usize, 0, 1];
        check(
            directed_modularity(&fractional, &comm)
                .unwrap_err()
                .to_string(),
            "non-negative integers",
        );
        check(
            directed_modularity(&huge, &comm).unwrap_err().to_string(),
            "exceeds",
        );
    }

    /// Arcs all weighing nothing leave every node alone, with modularity `0` (not `0 / 0`).
    #[test]
    fn weightless_arcs_leave_nodes_alone() {
        let arcs: Vec<_> = random_arcs(30, 200, 7)
            .into_iter()
            .map(|(u, v)| (u, v, "0".to_string()))
            .collect();
        let g = weighted_graph::<u8>("weightless", "integer", 30, &arcs, 2).unwrap();
        for refinement in [Refinement::None, Refinement::Leiden] {
            let l = DirectedLouvain::new_with_refinement(&g, 256, 0.0, refinement).unwrap();
            assert_eq!(as_usize(l.communities()), (0..30).collect::<Vec<_>>());
            assert_eq!(l.community_count(), 30);
            assert_eq!(l.partition_modularity(), 0.0);
        }
        assert_eq!(directed_modularity(&g, &[0; 30]).unwrap(), 0.0);
    }

    /// In-arcs carry the weights of the arcs they are, also on graphs whose arcs all have a
    /// reverse (whose transpose was once the graph itself, labels and all).
    #[test]
    fn in_arcs_weigh_what_their_arcs_do() {
        let n = 30;
        let simple = simple_edges(n, 120, 77);
        // each edge both ways, weighing differently each way
        let arcs: Vec<(usize, usize, u64)> = simple
            .iter()
            .enumerate()
            .flat_map(|(k, &(u, v))| [(u, v, 1 + k as u64), (v, u, 100 + k as u64)])
            .collect();
        let g = weighted_graph::<u32>("in_weights", "integer", n, &as_text(&arcs), 2).unwrap();
        assert!(g.is_symmetric().unwrap());
        let name = |seq| g.build_cache_filename(CacheFile::DirectedLouvain, Some(seq));
        let scratch = Scratch::new(&name, 0);
        let base = BaseArcs::new(&g, &scratch).unwrap();
        for i in 0..n {
            let mut want: Vec<(usize, u64)> = arcs
                .iter()
                .filter(|a| a.1 == i)
                .map(|a| (a.0, a.2))
                .collect();
            let mut got = vec![];
            base.in_arcs(i, |j, w| got.push((j, w)));
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want, "node {i}");
            assert_eq!(
                base.k_in(i),
                want.iter().map(|a| a.1).sum::<u64>(),
                "node {i}"
            );
        }
    }
}
