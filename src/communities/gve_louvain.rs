use crate::graph;
use crate::graph::cache::AccumulatorAtomic;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
#[allow(unused_imports)]
use portable_atomic::{AtomicBool, AtomicF64, AtomicUsize, Ordering};
use std::collections::HashMap as StdHashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Hashes community ids by a multiplication, folded: unlike `std`'s default (randomly seeded
/// per map), the same ids hash alike every run, so the maps list them in the same order, and
/// gains tied between communities are broken alike (single-threaded runs repeat exactly).
#[derive(Clone, Copy, Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
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

type HashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<IdHasher>>;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
#[cfg(feature = "bench")]
use std::time::Instant;

/// `f(u)` for every `u` in `0..n`, on `threads` threads.
fn par_for(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize) + Sync),
) -> Result<(), Box<dyn std::error::Error>> {
    let load = n.div_ceil(threads.max(1));
    thread::scope(|scope| {
        for tid in 0..threads.max(1) {
            let (begin, end) = ((tid * load).min(n), ((tid + 1) * load).min(n));
            scope.spawn(move |_| (begin..end).for_each(f));
        }
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
}

type ProceduralMemoryGVELouvain<Ix, W> = (
    // K' --- renamed to k
    AbstractedProceduralMemoryMut<<W as Weight>::Atomic>,
    // Σ' --- renamed to sigma
    AbstractedProceduralMemoryMut<<W as Weight>::Atomic>,
    // first CSR
    // G'.index --- renamed to gdi, `Ix`-width: during aggregation (`louvain_aggregate`) this
    // array's `next_index` role is unsafely reinterpreted, in-place, as an `Ix::Atomic` array
    // (`next_index.cast::<Ix::Atomic>()`) to serve as a lock-free edge counter --- sound because
    // `Ix::Atomic`'s `Plain` is `Ix` by construction (see `graph::IndexType::Atomic`).
    AbstractedProceduralMemoryMut<Ix>,
    // G'.edges --- renamed to gde
    AbstractedProceduralMemoryMut<Ix>,
    // G'.weights --- renamed to gdw
    AbstractedProceduralMemoryMut<W>,
    // second CSR
    // G''.index --- renamed to gddi (see `gdi`'s note: swaps roles with it across passes)
    AbstractedProceduralMemoryMut<Ix>,
    // G''.edges --- renamed to gdde
    AbstractedProceduralMemoryMut<Ix>,
    // G''.weights --- renamed to gddw
    AbstractedProceduralMemoryMut<W>,
    // processed --- flags node as processed
    AbstractedProceduralMemoryMut<AtomicBool>,
    // C' --- renamed to coms (C is stored as a member of the struct), `Ix`-width: community ids
    // are bounded exactly like node ids.
    AbstractedProceduralMemoryMut<Ix>,
    // Helper array for C' renumbering, dendrogram lookup & CSR aggregation, `Ix`-width for the
    // same reason as `coms`.
    AbstractedProceduralMemoryMut<Ix>,
);

/// For the computation of Louvain partitions as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) on [`GraphMemoryMap`] instances.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoGVELouvain<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight = f64> {
    /// Graph for which the partition is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped array containing each node's community.
    community: AbstractedProceduralMemoryMut<Ix>,
    /// Cardinality of distinct communities in the final partition.
    community_count: usize,
    /// Partition modularity.
    modularity: f64,
    /// Number of threads to be used in parallel environments.
    threads: usize,
    /// Whether communities are refined (Leiden's) before they're aggregated.
    refinement: super::Refinement,
    #[cfg(feature = "bench")]
    /// If feature = "bench" is enabled: store number of iterations, number of nodes (communities),
    /// time for louvain_move() for each pass performed on the graph.
    iters: Vec<(usize, usize, u128)>,
    _marker: std::marker::PhantomData<W>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    AlgoGVELouvain<'a, N, E, Ix, W>
{
    /// Constants set according to the optimizeds parameters described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876).
    /// Described in 4.1.2 Limiting the number of iterations per pass.
    const MAX_ITERATIONS: usize = 20;
    /// Described in 4.1.3 Adjusting tolerance drop rate (threshold scaling).
    const TOLERANCE_DROP: f64 = 10.;
    /// Described in 4.1.4 Adjusting initial tolerance (where it's 0.01: the local-moving phase
    /// stops once an iteration's modularity gains add up to less). On a sparse graph's first
    /// pass, though, nodes gain about `1 / m` each, and 0.01 stops it while its communities are
    /// still pairs, about a fifth of them across communities, which aggregating then makes
    /// permanent: on a planted partition of 10 thousand nodes, modularity 0.587 (Louvain) and
    /// 0.584 (Leiden), where networkx's Louvain finds 0.65 to 0.68. This runs local moving
    /// (nearly) to convergence instead, for 0.657 and 0.660, at no measurable cost on graphs of
    /// millions of edges (which it changed by 0.005 at most).
    const INITIAL_TOLERANCE: f64 = 1e-6;
    /// Described in 4.1.4 Adjusting aggregation tolerance.
    const AGGREGATION_TOLERANCE: f64 = 0.8;
    /// Maximum number of passes to be performed (a pass is an iteration of the Louvain() function described in "GVE-Louvain: Fast Louvain Algorithm for
    /// Community Detection in Shared Memory Setting" p. 5).
    const MAX_PASSES: usize = 30;

    /// Performs the Louvain() function as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) p. 5.
    ///
    /// The resulting graph partition and its corresponding modularity are stored in memory (in
    /// the partition's case in a memmapped file).
    ///
    /// * Note: isolated nodes remain in their own isolated community, in the final partition.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the louvain partition is to be computed.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gve_louvain = Self::new_no_compute(g, g.thread_num())?;
        let proc_mem = gve_louvain.init_cache_mem()?;

        gve_louvain.compute_with_proc_mem(proc_mem)?;

        Ok(gve_louvain)
    }

    /// Performs the Louvain() function as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) p. 5.
    ///
    /// The resulting graph partition and its corresponding modularity are stored in memory (in
    /// the partition's case in a memmapped file).
    ///
    /// * Note: isolated nodes remain in their own isolated community, in the final partition.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the louvain partition is to be computed.
    /// * `threads` --- the number of threads to be used in the computation.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_refinement(g, threads, super::Refinement::None)
    }

    /// Partitions the graph's nodes as [`new`](Self::new) does, with Leiden's refinement (see
    /// [`Refinement::Leiden`](super::Refinement::Leiden)): GVE-Leiden.
    pub fn new_leiden(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_refinement(g, g.thread_num(), super::Refinement::Leiden)
    }

    /// Partitions the graph's nodes on `threads` threads, its communities refined or not before
    /// they're aggregated (see [`Refinement`](super::Refinement)).
    ///
    /// With [`Refinement::Leiden`](super::Refinement::Leiden), this is GVE-Leiden (Sahu's
    /// ["GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory
    /// Setting"](https://doi.org/10.48550/arXiv.2312.13936)): after each pass's local moving,
    /// each node still alone in its sub-community (from singletons) joins the neighbouring
    /// sub-community, within its community, it gains most modularity joining (once, in
    /// parallel, by a compare-and-swap of its sub-community's weight checking it's still alone),
    /// sub-communities are aggregated, and each supernode starts the next pass in its community.
    /// A pass whose refinement keeps more than the aggregation tolerance's fraction of nodes apart
    /// aggregates its communities instead, as Louvain's.
    pub fn new_with_refinement(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
        refinement: super::Refinement,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gve_louvain = Self::new_no_compute(g, threads)?;
        gve_louvain.refinement = refinement;
        let proc_mem = gve_louvain.init_cache_mem()?;

        gve_louvain.compute_with_proc_mem(proc_mem)?;

        Ok(gve_louvain)
    }

    /// Searches for a previously cached result, and if not found performs the Louvain() function as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) p. 5.
    ///
    /// The resulting graph partition and its corresponding modularity are stored in memory (in
    /// the partition's case in a memmapped file).
    ///
    /// * Note: isolated nodes remain in their own isolated community, in the final partition.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the louvain partition is to be computed.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let l_fn = g.build_cache_filename(CacheFile::GVELouvain, None)?;
        if Path::new(&l_fn).exists()
            && let Ok(comms) = AbstractedProceduralMemoryMut::<Ix>::from_file_name(&l_fn)
        {
            let mut max = 0;
            comms.as_slice().iter().for_each(|&c| {
                let c = c.index();
                if c > max {
                    max = c;
                }
            });
            if let Ok(modularity) = g.modularity(comms.as_slice(), max) {
                return Ok(Self {
                    g,
                    community: comms,
                    community_count: max,
                    modularity,
                    threads: g.thread_num().max(1),
                    refinement: super::Refinement::None,
                    #[cfg(feature = "bench")]
                    iters: vec![],
                    _marker: std::marker::PhantomData,
                });
            }
        }
        Self::new(g)
    }

    pub fn coalesce_isolated_nodes(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.community_count = self.g.coallesce_isolated_nodes_community(
            self.community.shared_slice(),
            self.community_count,
        )?;
        Ok(())
    }

    /// Returns the number of communities in the Louvain partition.
    pub fn community_count(&self) -> usize {
        self.community_count
    }

    /// Returns the modularity of the Louvain partition.
    pub fn partition_modularity(&self) -> f64 {
        self.modularity
    }

    /// Returns the community label of a given node.
    pub fn node_community(&self, idx: usize) -> usize {
        assert!(idx < self.g.size());
        self.community.get(idx).index()
    }

    /// Returns a slice where each element corresponds (by index) to each node's community label.
    pub fn communities(&self) -> &[Ix] {
        self.community.as_slice()
    }

    #[cfg(feature = "bench")]
    pub fn get_iters(&self) -> &[(usize, usize, u128)] {
        self.iters.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::GVELouvain, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    AlgoGVELouvain<'a, N, E, Ix, W>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, threads)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, threads)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryGVELouvain<Ix, W>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryGVELouvain<Ix, W>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix, W>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix, W>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threads = threads.max(1);
        let out_fn = g.build_cache_filename(CacheFile::GVELouvain, None)?;
        let coms = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?;
        Ok(Self {
            g,
            community: coms,
            community_count: 0,
            modularity: 0.,
            threads,
            refinement: super::Refinement::None,
            #[cfg(feature = "bench")]
            iters: Vec::new(),
            _marker: std::marker::PhantomData,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<ProceduralMemoryGVELouvain<Ix, W>, Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let k_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(0))?;
        let s_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(1))?;
        let gdi_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(2))?;
        let gde_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(3))?;
        let gdw_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(4))?;
        let gddi_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(5))?;
        let gdde_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(6))?;
        let gddw_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(7))?;
        let a_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(8))?;
        let c_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(9))?;
        let h_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(10))?;

        let k = SharedSliceMut::<W::Atomic>::abst_mem_mut(&k_fn, node_count, true)?;
        let sigma = SharedSliceMut::<W::Atomic>::abst_mem_mut(&s_fn, node_count, true)?;
        let gdi = SharedSliceMut::<Ix>::abst_mem_mut(&gdi_fn, 2 * node_count, true)?;
        let gde = SharedSliceMut::<Ix>::abst_mem_mut(&gde_fn, edge_count, true)?;
        let gdw = SharedSliceMut::<W>::abst_mem_mut(&gdw_fn, edge_count, true)?;
        let gddi = SharedSliceMut::<Ix>::abst_mem_mut(&gddi_fn, 2 * node_count, true)?;
        let gdde = SharedSliceMut::<Ix>::abst_mem_mut(&gdde_fn, edge_count, true)?;
        let gddw = SharedSliceMut::<W>::abst_mem_mut(&gddw_fn, edge_count, true)?;
        let processed = SharedSliceMut::<AtomicBool>::abst_mem_mut(&a_fn, node_count, true)?;
        let coms = SharedSliceMut::<Ix>::abst_mem_mut(&c_fn, node_count, true)?;
        // has to be size max(|V|, |E|) as it is both used to store the 'holey' CSR (O(|E|) space) and
        // renumbered community memberships (O(|V|) space)
        let help = SharedSliceMut::<Ix>::abst_mem_mut(&h_fn, edge_count.max(node_count), true)?;

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);
        // only read for weighted edges: unweighted ones all weigh one
        let labels_ptr =
            E::is_weighted().then(|| SharedSlice::<E>::new(self.g.edge_labels_ptr(), edge_count));

        // initialize
        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            // initializations always uses at least two threads per core
            let threads = self.threads;
            let node_load = node_count.div_ceil(threads);

            let mut threads_res = vec![];
            for tid in 0..threads {
                let processed = processed.shared_slice();

                let mut gdi = gdi.shared_slice();
                let mut gde = gde.shared_slice();
                let mut gdw = gdw.shared_slice();
                let mut k = k.shared_slice();
                let mut sigma = sigma.shared_slice();
                let mut coms = coms.shared_slice();
                let mut helper = help.shared_slice();
                let mut communities = self.community.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // unit weights, written a chunk at a time (degrees may exceed it)
                        let ones: Vec<W> = if labels_ptr.is_none() {
                            vec![W::one(); u16::MAX as usize]
                        } else {
                            Vec::new()
                        };
                        for u in begin..end {
                            let u_start = index_ptr.get(u).index();
                            let u_end = index_ptr.get(u + 1).index();
                            let out_offset = u * 2;
                            let deg_u = match u_end.overflowing_sub(u_start) {
                                (r, false) => r,
                                (_, true) => {
                                    return Err(format!(
                                        "overflow calculating degree of {u}: {u_end} - {u_start}"
                                    )
                                    .into());
                                }
                            };
                            let edges = neighbours_ptr.slice(u_start, u_end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error reading node {u} in init").into()
                                },
                            )?;

                            // mark all nodes unprocessed
                            processed.get(u).swap(false, Ordering::Relaxed);

                            // initilize helper, communities, coms and CSR vectors
                            *helper.get_mut(u) = <Ix as graph::IndexType>::max();
                            *communities.get_mut(u) = Ix::new(u);
                            *coms.get_mut(u) = Ix::new(u);
                            *gdi.get_mut(out_offset) = Ix::new(u_start);
                            *gdi.get_mut(out_offset + 1) = Ix::new(u_end);

                            // copy G onto G' --- no copy/alloc needed: `edges` already borrows the
                            // `Ix`-width neighbours straight out of the original graph's mmap, and
                            // `gde` now stores them at the same width, so this is a plain memcpy.
                            gde.write_slice(u_start, edges);
                            // 𝐾′[u]: u's total edge weight
                            let k_u = match &labels_ptr {
                                Some(labels) => {
                                    let mut k_u = W::zero();
                                    for e in u_start..u_end {
                                        let label = labels.get(e);
                                        let w = label.weight::<W>();
                                        // (modularity is meaningless otherwise)
                                        if !(w.is_finite() && w >= W::zero()) {
                                            return Err(format!(
                                                "error louvain's edge weights must be finite and non-negative, but one is {label:?}"
                                            )
                                            .into());
                                        }
                                        *gdw.get_mut(e) = w;
                                        k_u += w;
                                    }
                                    k_u
                                }
                                None => {
                                    for at in (u_start..u_end).step_by(ones.len()) {
                                        gdw.write_slice(at, &ones[..ones.len().min(u_end - at)]);
                                    }
                                    W::from_usize(deg_u)
                                }
                            };

                            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                            *k.get_mut(u) = W::Atomic::new(k_u);
                            *sigma.get_mut(u) = W::Atomic::new(k_u);
                        }
                        Ok(())
                    },
                ));
            }

            // check for errors
            for (tid, r) in threads_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
            }

            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        Ok((
            k, sigma, gdi, gde, gdw, gddi, gdde, gddw, processed, coms, help,
        ))
    }

    /// Computes the differencial modularity[^1], 𝛿𝑄, upon moving a node `u`, from a community `d` to a community `c`.
    ///
    /// [^1]: 𝛿𝑄 = (Ku->c - Ku->d) / m - (Ku * (Σc - Σd) / (2m^2)).
    ///
    /// # Arguments
    ///
    /// * `k_u_c` --- weight of edges from node `u` to nodes in community `c`.
    /// * `k_u_d` --- weight of edges from node `u` to nodes in community `d`.
    /// * `k_u` --- total weight of edges from node `u`.
    /// * `sig_c` --- total weight of edges in community `c`.
    /// * `sig_d` --- total weight of edges in community `d`.
    /// * `m` --- total weight of edges in the graph (or half of it if graphis directed).
    ///
    /// # Returns
    ///
    /// Difference in modularity upon moving node `u` from community `d` to community `c`.
    ///
    /// Always computed in `f64`, regardless of the algorithm's own `W`: `sig_c`/`sig_d`/`m` scale
    /// with the *total* graph weight, so `sig_c - sig_d` (and `k_u_c - k_u_d`) is a subtraction of
    /// two large, often near-equal values --- classic catastrophic cancellation. In `f32`, once
    /// community weights exceed ~2^24 (~16.7M, easily reached by real graphs), this cancellation
    /// destroys essentially all of `𝛿𝑄`'s precision, making every computed gain round down to
    /// (near) zero and silently stalling the whole algorithm (confirmed: on a 68M-edge graph,
    /// `f32`-stored weights with this computation also done in `f32` left *every* node in its own
    /// singleton community). Doing just this one, numerically-sensitive computation in `f64`
    /// fixes that while leaving `k`/`sigma`/edge weights narrow (`W`) everywhere else, preserving
    /// the memory-bandwidth benefit that narrower storage was for.
    #[inline(always)]
    fn delta_q(k_u_c: f64, k_u_d: f64, k_u: f64, sig_c: f64, sig_d: f64, m: f64) -> f64 {
        assert!(
            k_u_c >= 0.0 && k_u_d >= 0.0 && k_u >= 0.0 && sig_c >= 0.0 && sig_d >= 0.0 && m >= 0.0
        );
        assert!(
            k_u_c.is_finite()
                && k_u_d.is_finite()
                && k_u.is_finite()
                && sig_c.is_finite()
                && sig_d.is_finite()
                && m.is_normal()
        );
        (k_u_c - k_u_d) / m - k_u * (k_u + sig_c - sig_d) / (2.0 * m * m)
    }

    /// Determines the total weight of edges from a given node `u` to every community `c` it is linked to.
    ///
    /// # Arguments
    ///
    /// * `nmap` (updated) --- hashmap for entries `(c, k_u_c)`, where `c` is a community and `k_u_c` is the total weight of edges from node u into nodes in `c`.
    /// * `index` --- index vector.
    /// * `edges` --- edges vector.
    /// * `weights` --- weights vector.
    /// * `coms` --- communities vector.
    /// * `u` --- index of node `u`.
    /// * `self_allowed` --- flag signaling whether self-loops should be accepted.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn scan_communities(
        comm_count: usize,
        nmap: &mut HashMap<usize, W>,
        index: SharedSliceMut<Ix>,
        edges: SharedSliceMut<Ix>,
        weights: SharedSliceMut<W>,
        coms: SharedSliceMut<Ix>,
        u: usize,
        self_allowed: bool,
    ) {
        for e_idx in index.get(2 * u).index()..index.get(2 * u + 1).index() {
            let v = edges.get(e_idx).index();
            if !self_allowed && v == u {
                continue;
            }
            let coms_v = coms.get(v).index();
            if coms_v > comm_count {
                println!(
                    "WARNING!! Impossible neighbour commnity for node {u} (neighbour {}): {{{} > {comm_count}}} but c: {}",
                    v,
                    coms_v,
                    coms.get(coms_v).index()
                );
                println!(
                    "{}..{} at {e_idx}\n",
                    index.get(2 * u).index(),
                    index.get(2 * u + 1).index()
                );
            }
            // 𝐻𝑡[𝐶'[v]] ← 𝐻𝑡[𝐶'[v]] + w
            *nmap.entry(coms_v).or_insert(W::zero()) += *weights.get(e_idx);
        }
    }

    /// Determines the best community a node `u` may be switched into from its neighbours community set.
    ///
    /// # Arguments
    ///
    /// * `nmap` --- `Hashmap` of entries `(c, k_u_c)`, where `c` is a community and `k_u_c` is the total weight of edges from node `u` into nodes in `c`.
    /// * `sigma` --- total community weights vector.
    /// * `c_u` --- community of node `u`.
    /// * `k:u` --- total weight of node `u` .
    /// * `m` --- total weight of edges in the graph (or half of it if graph is directed).
    ///
    /// # Returns
    ///
    /// `(c*: usize, 𝛿𝑄*: f64)`, where:
    /// * `c*` is the best community node u may be switched into
    /// * `𝛿𝑄*` is the delta modularity arising thereof.
    #[inline(always)]
    fn choose_community(
        nmap: &mut HashMap<usize, W>,
        sigma: SharedSliceMut<W::Atomic>,
        c_u: usize,
        k_u: W,
        m: W,
    ) -> (usize, f64) {
        // houses c*
        let mut best_gain = 0.0f64;
        // houses 𝛿𝑄*
        let mut best_comm = c_u;

        let k_u_f64 = k_u.to_f64();
        let m_f64 = m.to_f64();

        // Ku->d
        let k_u_d = nmap.get(&c_u).cloned().unwrap_or(W::zero()).to_f64();
        // Σd
        let sig_d = sigma.get(c_u).load(Ordering::Relaxed).to_f64();

        for (&c_v, &k_u_c) in nmap.iter() {
            if c_v == c_u {
                continue;
            }
            // Σc
            let sig_c = sigma.get(c_v).load(Ordering::Relaxed).to_f64();

            // 𝛿𝑄 = (Ku->c - Ku->d) / m - (Ku * (Σc - Σd) / (2m^2))
            let dq = Self::delta_q(k_u_c.to_f64(), k_u_d, k_u_f64, sig_c, sig_d, m_f64);
            if dq > best_gain {
                // 𝛿𝑄∗ = 𝛿𝑄
                best_gain = dq;
                // *c = C'[v]
                best_comm = c_v;
            }
        }
        // (c*, 𝛿𝑄*)
        (best_comm, best_gain)
    }

    /// Performs the LouvainMove() function as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) p. 6.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn louvain_move(
        l_pass: usize,
        comm_count: usize,
        k: SharedSliceMut<W::Atomic>,
        sigma: SharedSliceMut<W::Atomic>,
        processed: SharedSliceMut<AtomicBool>,
        idx: SharedSliceMut<Ix>,
        mut next_index: SharedSliceMut<Ix>,
        e: SharedSliceMut<Ix>,
        w: SharedSliceMut<W>,
        mut c: SharedSliceMut<Ix>,
        mut helper: SharedSliceMut<Ix>,
        m: W,
        tolerance: f64,
        threads: usize,
        keep_communities: bool,
    ) -> Result<(usize, bool), Box<dyn std::error::Error>> {
        // recalculate node load per thread
        let node_load = comm_count.div_ceil(threads);

        let mut l_iter = 0;
        // whether any node moved (an iteration gaining less than the tolerance may have)
        let mut moved = false;
        let synchronize = Arc::new(Barrier::new(threads));

        for i in 0..Self::MAX_ITERATIONS {
            let global_delta_q = Arc::new(AtomicF64::new(0.0));

            // reset iteration metrics
            thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                // use dynamic style scheduling by splitting range [0, cur_node_count) for threads
                let mut thread_res = Vec::with_capacity(threads);
                for tid in 0..threads {
                    let mut local_delta_q_sum = 0.0;
                    let mut nmap: HashMap<usize, W> = HashMap::default();

                    let k = k.clone();
                    let s = sigma.clone();
                    let processed = processed.clone();

                    let global_delta_q = global_delta_q.clone();
                    let synchronize = synchronize.clone();

                    let begin = std::cmp::min(tid * node_load, comm_count);
                    let end = std::cmp::min(begin + node_load, comm_count);

                    // each thread gets its own local HashMap for accumulating neighbor community weights
                    thread_res.push(scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            // if pass > 0, we need to initialize communities for the new graph (each super-vertex in its own community)
                            if l_pass != 0 && i == 0 {
                                for u in begin..end {
                                    // unprune all nodes
                                    *helper.get_mut(u) = <Ix as graph::IndexType>::max();
                                    *next_index.get_mut(u * 2) = Ix::new(0);
                                    // reassign node community to itself (unless supernodes start
                                    // in their communities: Leiden's)
                                    if !keep_communities {
                                        *c.get_mut(u) = Ix::new(u);
                                    }
                                }
                            }

                            // wait for all threads to finish current pass initialization
                            synchronize.wait();

                            for u in begin..end {
                                // skip this vertex if it was marked as pruned (no need to process)
                                if processed.get(u).swap(true, Ordering::Relaxed) {
                                    continue;
                                }
                                let c_u = c.get(u).index();
                                // get weight of u k[u]
                                let k_u = k.get(u).load(Ordering::Relaxed);
                                // gather neighbour communities and edge weight sums using the adjacency list of u

                                nmap.clear();
                                let nm = &mut nmap;
                                Self::scan_communities(comm_count, nm, idx, e, w, c, u, false);

                                let (b_c, b_g) = Self::choose_community(nm, s.clone(), c_u, k_u, m);

                                // move u to the best community if it yields positive gain. the
                                // comparison is always done in `f64` (see `delta_q`'s docs), so
                                // this threshold is independent of `W`.
                                if b_c != c_u && b_g > 1e-12 {
                                    // atomic updates to community totals --- if node has already
                                    // been moved don't do anything
                                    if s
                                        .get(c_u)
                                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                                            // real-valued weights' totals (summed in any order)
                                            // may round a hair below a member's own: only a
                                            // larger shortfall means it isn't a member
                                            let slack = W::from_f64(
                                                k_u.to_f64() * W::epsilon().to_f64().sqrt(),
                                            );
                                            if s + slack < k_u {
                                                None
                                            } else if s < k_u {
                                                Some(W::zero())
                                            } else {
                                                Some(s - k_u)
                                            }
                                        })
                                        .is_ok()
                                    {
                                        s.get(b_c).fetch_add(k_u, Ordering::Relaxed);
                                        // update community assignment of u
                                        *c.get_mut(u) = Ix::new(b_c);
                                        local_delta_q_sum += b_g;
                                        // mark all neighbors of u to be processed in the next iteration
                                        for idx in idx.get(u * 2).index()..idx.get(u * 2 + 1).index() {
                                            processed
                                                .get(e.get(idx).index())
                                                .store(false, Ordering::Relaxed);
                                        }
                                    } else {
                                        return Err(
                                            format!("error louvain_move(): {} {u} can't be removed from its community {c_u} as node weight is {k_u} but community total weight is only {}",
                                                    if l_pass == 0 {"node"} else {"super-node"}, s.get(c_u).load(Ordering::Relaxed)
                                                    ).into()
                                            );
                                    }
                                }
                            }
                            // atomically accumulate thread-local result to global
                            #[allow(clippy::unnecessary_cast)]
                            global_delta_q.fetch_add(local_delta_q_sum as f64, Ordering::Relaxed);
                            Ok(())
                        },
                    ));
                }
                // check for errors
                for (tid, r) in thread_res.into_iter().enumerate() {
                    r.join()
                        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                        .map_err(|e| -> Box<dyn std::error::Error> {
                            format!("error in move (thread {tid}): {:?}", e).into()
                        })?;
                }
                Ok(())
            })
            .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

            moved |= global_delta_q.load(Ordering::Relaxed) > 0.;
            // check convergence criteria for the local-moving phase
            if global_delta_q.load(Ordering::Relaxed).abs() < tolerance {
                break;
            }
            l_iter += 1;
        }

        Ok((l_iter, moved))
    }

    /// Leiden's refinement (GVE-Leiden's `leidenRefine`): each node `u` still alone in its
    /// sub-community (`sub`, from singletons, weighing `sub_sigma`) joins the neighbouring
    /// sub-community within its community (`bounds`) it gains most modularity joining, if any,
    /// once, in parallel. It leaves its sub-community only if a compare-and-swap of the
    /// sub-community's weight, from its own to `0`, finds it still alone: nodes join
    /// sub-communities through an edge to one of their nodes, which stay, so sub-communities are
    /// connected.
    #[allow(clippy::too_many_arguments)]
    fn leiden_refine(
        comm_count: usize,
        k: SharedSliceMut<W::Atomic>,
        bounds: SharedSliceMut<Ix>,
        sub: SharedSliceMut<Ix>,
        sub_sigma: SharedSliceMut<W::Atomic>,
        index: SharedSliceMut<Ix>,
        edges: SharedSliceMut<Ix>,
        weights: SharedSliceMut<W>,
        m: W,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let load = comm_count.div_ceil(threads);
        let m_f64 = m.to_f64();
        thread::scope(|scope| {
            for tid in 0..threads {
                let (k, sub_sigma) = (k.clone(), sub_sigma.clone());
                let mut sub = sub;
                let begin = (tid * load).min(comm_count);
                let end = (begin + load).min(comm_count);
                scope.spawn(move |_| {
                    let mut nmap: HashMap<usize, W> = HashMap::default();
                    for u in begin..end {
                        let c = sub.get(u).index();
                        let k_u = k.get(u).load(Ordering::Relaxed);
                        // alone in its sub-community (or without edges)
                        if sub_sigma.get(c).load(Ordering::Relaxed) != k_u || k_u == W::zero() {
                            continue;
                        }
                        let b = bounds.get(u).index();
                        nmap.clear();
                        for e in index.get(2 * u).index()..index.get(2 * u + 1).index() {
                            let v = edges.get(e).index();
                            if v != u && bounds.get(v).index() == b {
                                *nmap.entry(sub.get(v).index()).or_insert(W::zero()) +=
                                    *weights.get(e);
                            }
                        }
                        let k_u_f64 = k_u.to_f64();
                        let (mut best, mut best_gain) = (c, 0.0f64);
                        for (&d, &k_u_d) in nmap.iter() {
                            if d == c {
                                continue;
                            }
                            let sig_d = sub_sigma.get(d).load(Ordering::Relaxed).to_f64();
                            // from its own sub-community, alone (`Σ = K_u`, no edge to it)
                            let dq =
                                Self::delta_q(k_u_d.to_f64(), 0.0, k_u_f64, sig_d, k_u_f64, m_f64);
                            if dq > best_gain || (dq == best_gain && best != c && d < best) {
                                (best, best_gain) = (d, dq);
                            }
                        }
                        if best != c
                            && best_gain > 1e-12
                            && sub_sigma
                                .get(c)
                                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |s| {
                                    (s == k_u).then_some(W::zero())
                                })
                                .is_ok()
                        {
                            sub_sigma.get(best).fetch_add(k_u, Ordering::Relaxed);
                            *sub.get_mut(u) = Ix::new(best);
                        }
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
    }

    /// Performs the LouvainAggregate() function as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) p. 6.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn louvain_aggregate(
        l_pass: usize,
        comm_count: usize,
        new_comm_count: usize,
        k: SharedSliceMut<W::Atomic>,
        sigma: SharedSliceMut<W::Atomic>,
        processed: SharedSliceMut<AtomicBool>,
        index: SharedSliceMut<Ix>,
        mut next_index: SharedSliceMut<Ix>,
        edges: SharedSliceMut<Ix>,
        mut next_edges: SharedSliceMut<Ix>,
        weights: SharedSliceMut<W>,
        mut next_weights: SharedSliceMut<W>,
        coms: SharedSliceMut<Ix>,
        mut helper: SharedSliceMut<Ix>,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // recalculate comm load per thread
        let prev_comm_load = comm_count.div_ceil(threads);
        let new_comm_load = new_comm_count.div_ceil(threads);

        let synchronize = Arc::new(Barrier::new(threads));
        let counter = unsafe {
            next_index
                .cast::<Ix::Atomic>()
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "error getting atomic edge counters".into()
                })?
        };

        // exclusive scan
        let mut sum = 0usize;
        for u in 0..new_comm_count {
            let degree = next_index.get(u * 2).index();
            *next_index.get_mut(u * 2) = Ix::new(sum);
            *next_index.get_mut(u * 2 + 1) = Ix::new(sum);
            sum += degree;
        }

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            // use dynamic style scheduling by splitting range [0, cur_node_count) for threads
            let mut thread_res = Vec::with_capacity(threads);
            for tid in 0..threads {

                // hashmap to store edges for each community
                let mut nm: HashMap<usize, W> = HashMap::default();

                let counter = counter.clone();
                let k = k.clone();
                let sigma = sigma.clone();
                let processed = processed.clone();

                let synchronize = synchronize.clone();

                let prev_begin = std::cmp::min(tid * prev_comm_load, comm_count);
                let prev_end = std::cmp::min(prev_begin + prev_comm_load, comm_count);
                let new_begin = std::cmp::min(tid * new_comm_load, new_comm_count);
                let new_end = std::cmp::min(new_begin + new_comm_load, new_comm_count);

                thread_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // add com - node edges atomically
                        for u in prev_begin..prev_end {
                            let degree = index.get(u * 2 + 1).index() - index.get(u * 2).index();
                            // guard against isolated nodes to not go over community index boundary
                            if degree > 0 {
                                // add edge (C'[u], u)
                                let idx = counter
                                    .get(coms.get(u).index() * 2 + 1)
                                    .fetch_add_one_widened(Ordering::Relaxed);
                                *helper.get_mut(idx) = Ix::new(u);
                            }
                        }

                        // wait for every thread to finish adding com - node edges
                        // synchronize.wait();
                        //
                        // for c in new_begin..new_end {
                        //     *next_index.get_mut(c * 2 + 1) = k.get(c).load(Ordering::Relaxed);
                        // }

                        // wait for every thread to finish com - node CSR indexing
                        synchronize.wait();

                        for c in new_begin..new_end {
                            // local accumulator variables
                            let mut total = 0;
                            let mut total_weight = W::zero();

                            // mark unprocessed
                            processed.get(c).store(false, Ordering::Relaxed);

                            // clear previous community's neighbour map
                            nm.clear();

                            // for every node in the community scan edges to accumulate supergraph
                            // edge weights
                            for e_idx in next_index.get(c * 2).index()..next_index.get(c * 2 + 1).index() {
                                Self::scan_communities(comm_count, &mut nm, index, edges, weights, coms, helper.get(e_idx).index(), true);
                            }

                            // for every linked community store (dest, weight) in CSR
                            for (&d, &w) in nm.iter() {
                                // troubleshoot edge
                                if d > new_comm_count {
                                    return Err(
                                        format!("error building supergraph: found edge to community {d} but max_id for communities is {comm_count} in the {l_pass}th pass"
                                                ).into()
                                        );
                                }

                                // store edge
                                let base = next_index.get(c * 2).index();
                                *next_edges.get_mut(base + total) = Ix::new(d);
                                *next_weights.get_mut(base + total) = w;

                                // increment counts
                                total += 1;
                                total_weight += w;
                            }

                            // set cmommunity index upper boundary
                            *next_index.get_mut(c * 2 + 1) = Ix::new(next_index.get(c * 2).index() + total);

                            // check for indexing missalignment errors
                            if c > 0 && next_index.get(c * 2).index() < next_index.get(c * 2 - 1).index() {
                                println!("error {} {} {}", new_begin, next_index.get(c * 2).index(), next_index.get(c * 2 - 1).index());
                                return Err(
                                    format!("error building supergraph: index missalignment, node {} ends at {} and node {c} starts at {}",
                                            c - 1, next_index.get(c * 2 - 1).index(), next_index.get(c * 2).index()
                                            ).into()
                                    );
                            }
                            // store node weight & community weight
                            k.get(c).store(total_weight, Ordering::Relaxed);
                            sigma.get(c).store(total_weight, Ordering::Relaxed);
                        }
                        Ok(())
                    },
                ));
            }
            // check for errors
            for (tid, r) in thread_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in aggregation (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        Ok(())
    }

    /// The partition's modularity ([`GraphMemoryMap::modularity_with`], in parallel), in memory
    /// the run is done with: `tot_deg` for the communities' total degrees, and, with weighted
    /// edges, `int_deg` for the nodes' weighted degrees, if it holds them (else, mapped anew).
    ///
    /// Generic over `T: Weight`, independently of the algorithm's own `W`: the caller picks
    /// whichever `T` lets it reuse memory in place (see call site); the division and the
    /// summation are in `f64` regardless.
    ///
    /// [`GraphMemoryMap::modularity_with`]: crate::graph::GraphMemoryMap::modularity_with
    fn finish_modularity<T: Weight>(
        &self,
        mut int_deg: SharedSliceMut<T>,
        mut tot_deg: SharedSliceMut<T>,
        communities: SharedSliceMut<Ix>,
        node_count: usize,
        prev_comm_count: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let communities = &communities.as_slice()[..node_count];
        let degrees = (int_deg.len() >= node_count).then(|| int_deg.as_mut_slice());
        if E::is_weighted() && degrees.is_none() {
            return self.g.modularity(communities, prev_comm_count);
        }
        self.g.modularity_with(
            communities,
            prev_comm_count,
            tot_deg.as_mut_slice(),
            degrees,
        )
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix, W>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let threads = self.threads;

        let (k, sigma, gdi, gde, gdw, gddi, gdde, gddw, processed, coms, helper) = proc_mem;

        // C --- the community vector of the original nodes
        let mut communities = self.community.shared_slice();

        let mut tolerance = Self::INITIAL_TOLERANCE;

        let mut prev_comm_count = node_count;
        let mut new_comm_count: usize = 0;

        // 2m (since each undirected edge has weight 2): the total of the nodes' weights, 𝐾′, for
        // weighted edges (summed in `f64`), the arc count for unweighted ones
        let m2 = if E::is_weighted() {
            let k = k.shared_slice();
            W::from_f64(
                (0..node_count)
                    .map(|u| k.get(u).load(Ordering::Relaxed).to_f64())
                    .sum(),
            )
        } else {
            W::from_usize(edge_count)
        };
        let m = m2 / (W::one() + W::one());

        let mut coms = coms.shared_slice();
        let mut helper = helper.shared_slice();

        let mut index = gdi.shared_slice();
        let mut next_index = gddi.shared_slice();

        let mut edges = gde.shared_slice();
        let mut next_edges = gdde.shared_slice();

        let mut weights = gdw.shared_slice();
        let mut next_weights = gddw.shared_slice();

        let k = k.shared_slice();
        let sigma = sigma.shared_slice();
        let processed = processed.shared_slice();

        // Leiden's: each pass's communities (bounds), its nodes' sub-communities and their
        // weights, the supernodes' starting communities, and their communities' first supernodes
        let leiden_mem = match self.refinement {
            super::Refinement::None => None,
            super::Refinement::Leiden => {
                let file = |seq| self.build_cache_filename(CacheFile::GVELouvain, Some(seq));
                Some((
                    SharedSliceMut::<Ix>::abst_mem_mut(&file(13)?, node_count.max(1), true)?,
                    SharedSliceMut::<Ix>::abst_mem_mut(&file(14)?, node_count.max(1), true)?,
                    SharedSliceMut::<W::Atomic>::abst_mem_mut(&file(15)?, node_count.max(1), true)?,
                    SharedSliceMut::<Ix>::abst_mem_mut(&file(16)?, node_count.max(1), true)?,
                    SharedSliceMut::<AtomicUsize>::abst_mem_mut(
                        &file(17)?,
                        node_count.max(1),
                        true,
                    )?,
                ))
            }
        };
        // whether `coms` holds the communities the current graph's (super)nodes start the pass in
        // (Leiden's): the nodes' communities, if the algorithm ends before moving them
        let mut pending = false;

        // outer loop: louvain passes
        for l_pass in 0..Self::MAX_PASSES {
            if prev_comm_count <= 1 {
                break;
            }

            #[cfg(feature = "bench")]
            let before = Instant::now();

            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′) ; 𝐶′ ← [0..|𝑉′|) (initialization inside louvain_move())
            // li <- louvain_move(G', C', K', Σ′)
            // if 𝑙𝑖 ≤ 1 then break --- adjusted to 𝑙𝑖 ≤ 0 as we don't increment before loop ends
            // as is teh case with the algorithm author
            let (_count_iter, moved) = Self::louvain_move(
                l_pass,
                prev_comm_count,
                k.clone(),
                sigma.clone(),
                processed.clone(),
                index,
                next_index,
                edges,
                weights,
                coms,
                helper,
                m,
                tolerance,
                threads,
                pending,
            )?;

            #[cfg(feature = "bench")]
            {
                let elapsed = before.elapsed().as_micros();
                println!(
                    "{l_pass} did {_count_iter} iterations on {prev_comm_count} nodes, in {elapsed} micros"
                );
                self.iters.push((_count_iter, prev_comm_count, elapsed));
            }

            // nothing moved: done (if some nodes moved, though less than the tolerance, their
            // moves are kept, and the aggregation tolerance ends it below)
            if !moved {
                // (Leiden's supernodes are in their communities, moved or not)
                pending = leiden_mem.is_some();
                break;
            }
            pending = false;

            // Leiden's refinement: the communities local moving found become the bounds, their
            // nodes' sub-communities (refined, connected) what's aggregated, unless refining
            // keeps too many nodes apart
            let mut leiden_level = false;
            if let Some((bounds, sub, sub_sigma, _, marks)) = &leiden_mem {
                let (bounds, sub, sub_sigma, marks) = (
                    bounds.shared_slice(),
                    sub.shared_slice(),
                    sub_sigma.shared_slice(),
                    marks.shared_slice(),
                );
                par_for(threads, prev_comm_count, &|u| {
                    let (mut bounds, mut sub) = (bounds, sub);
                    *bounds.get_mut(u) = *coms.get(u);
                    *sub.get_mut(u) = Ix::new(u);
                    sub_sigma
                        .get(u)
                        .store(k.get(u).load(Ordering::Relaxed), Ordering::Relaxed);
                    marks.get(u).store(0, Ordering::Relaxed);
                })?;
                Self::leiden_refine(
                    prev_comm_count,
                    k.clone(),
                    bounds,
                    sub,
                    sub_sigma,
                    index,
                    edges,
                    weights,
                    m,
                    threads,
                )?;
                let pieces = AtomicUsize::new(0);
                par_for(threads, prev_comm_count, &|u| {
                    if marks.get(sub.get(u).index()).swap(1, Ordering::Relaxed) == 0 {
                        pieces.fetch_add(1, Ordering::Relaxed);
                    }
                })?;
                let pieces = pieces.load(Ordering::Relaxed);
                if pieces as f64 <= Self::AGGREGATION_TOLERANCE * prev_comm_count as f64 {
                    par_for(threads, prev_comm_count, &|u| {
                        let mut coms = coms;
                        *coms.get_mut(u) = *sub.get(u);
                    })?;
                    leiden_level = true;
                }
            }

            for u in 0..prev_comm_count {
                let degree = index.get(u * 2 + 1).index() - index.get(u * 2).index();
                let c = coms.get(u).index();
                let new_c = helper.get(c).index();
                if new_c == <Ix as graph::IndexType>::max().index() {
                    *helper.get_mut(c) = Ix::new(new_comm_count);
                    *coms.get_mut(u) = Ix::new(new_comm_count);
                    *next_index.get_mut(new_comm_count * 2) = Ix::new(degree);
                    new_comm_count += 1;
                } else {
                    *coms.get_mut(u) = Ix::new(new_c);
                    let prev = next_index.get(new_c * 2).index();
                    *next_index.get_mut(new_c * 2) = Ix::new(prev + degree);
                }
            }

            // 𝐶 ← Lookup dendrogram using 𝐶 to 𝐶
            for orig in 0..node_count {
                let c = communities.get(orig).index();
                if c < prev_comm_count {
                    *communities.get_mut(orig) = *coms.get(c);
                } else {
                    return Err(
                        format!(
                            "error performing dendrogram lookup: node {orig} is in community {c} but max_id for communities is {prev_comm_count} in the {l_pass}th pass"
                            ).into()
                        );
                }
            }

            // Leiden's: each supernode's community, labelled by its first supernode
            if leiden_level && let Some((bounds, _, _, init, firsts)) = &leiden_mem {
                let (bounds, init, firsts) = (
                    bounds.shared_slice(),
                    init.shared_slice(),
                    firsts.shared_slice(),
                );
                par_for(threads, prev_comm_count, &|u| {
                    firsts
                        .get(bounds.get(u).index())
                        .store(usize::MAX, Ordering::Relaxed);
                })?;
                par_for(threads, prev_comm_count, &|u| {
                    firsts
                        .get(bounds.get(u).index())
                        .fetch_min(coms.get(u).index(), Ordering::Relaxed);
                })?;
                par_for(threads, prev_comm_count, &|u| {
                    let mut init = init;
                    let first = firsts.get(bounds.get(u).index()).load(Ordering::Relaxed);
                    *init.get_mut(coms.get(u).index()) = Ix::new(first);
                })?;
            }
            // the supernodes' starting communities, in `coms` (no longer needed once aggregated)
            let start_in_communities = |coms: &mut SharedSliceMut<Ix>| {
                if let Some((_, _, _, init, _)) = &leiden_mem {
                    let (init, coms) = (init.shared_slice(), *coms);
                    let _ = par_for(threads, new_comm_count, &|s| {
                        let mut coms = coms;
                        *coms.get_mut(s) = *init.get(s);
                    });
                }
            };

            // if |Γ|/|Γ𝑜𝑙𝑑 | > 𝜏_𝑎𝑔𝑔 then break
            if new_comm_count as f64 / prev_comm_count as f64 > Self::AGGREGATION_TOLERANCE {
                if leiden_level {
                    start_in_communities(&mut coms);
                    pending = true;
                }
                prev_comm_count = new_comm_count;
                break;
            }

            #[cfg(feature = "bench")]
            let before = Instant::now();

            // build the aggregated graph (super-vertex graph) in the 'next_index' and 'next_edges' buffers
            Self::louvain_aggregate(
                l_pass,
                prev_comm_count,
                new_comm_count,
                k.clone(),
                sigma.clone(),
                processed.clone(),
                index,
                next_index,
                edges,
                next_edges,
                weights,
                next_weights,
                coms,
                helper,
                threads,
            )?;

            #[cfg(feature = "bench")]
            {
                let _elapsed = before.elapsed().as_micros();
                println!("{l_pass} aggregation took {_elapsed} micros");
            }

            // Leiden's supernodes start the next pass in their communities
            if leiden_level {
                start_in_communities(&mut coms);
                par_for(threads, new_comm_count, &|s| {
                    sigma.get(s).store(W::zero(), Ordering::Relaxed);
                })?;
                par_for(threads, new_comm_count, &|s| {
                    sigma
                        .get(coms.get(s).index())
                        .fetch_add(k.get(s).load(Ordering::Relaxed), Ordering::Relaxed);
                })?;
                pending = true;
            }

            // check aggregation tolerance: if communities did not reduce sufficiently, break cycle
            if new_comm_count as f64 >= (prev_comm_count as f64 * Self::AGGREGATION_TOLERANCE) {
                prev_comm_count = new_comm_count;
                break;
            }

            // tighten tolerance for next pass (threshold scaling)
            // 𝜏 = 𝜏 / TOLERANCE_DROP
            tolerance /= Self::TOLERANCE_DROP;

            // prepare for next loop by resetting community count and switching CSRs
            std::mem::swap(&mut index, &mut next_index);
            std::mem::swap(&mut edges, &mut next_edges);
            std::mem::swap(&mut weights, &mut next_weights);

            prev_comm_count = new_comm_count;
            new_comm_count = 0;
        }

        // Leiden's last supernodes' communities: the nodes'
        if pending {
            par_for(threads, node_count, &|orig| {
                let mut communities = communities;
                let s = communities.get(orig).index();
                *communities.get_mut(orig) = *coms.get(s);
            })?;
        }

        // communities split into their connected parts (numbered by smallest node): Louvain may
        // leave some disconnected (a node holding one together may move out of it), and
        // splitting them only raises modularity
        if node_count > 0 {
            let offsets = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
            let neighbours = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);
            prev_comm_count = super::split_disconnected(
                threads,
                offsets.as_slice(),
                neighbours.as_slice(),
                &mut self.community.as_mut_slice()[..node_count],
            )?;
        }
        self.community_count = prev_comm_count;

        // compute partition modularity
        if m2 == W::zero() {
            self.modularity = 0.0;
            self.g.cleanup_cache(CacheFile::GVELouvain)?;
            return Ok(());
        }

        self.modularity = if self.g.size() <= self.g.width() {
            // common case (|E| >= |V|): reuse the already-`W`-typed weight buffers directly ---
            // no cast, no extra allocation.
            let int_deg = weights.par_zero_out_bytes();
            let tot_deg = next_weights.par_zero_out_bytes();
            self.finish_modularity::<W>(int_deg, tot_deg, communities, node_count, prev_comm_count)?
        } else if let (Some(mut int_deg), Some(mut tot_deg)) =
            (unsafe { helper.cast::<f64>() }, unsafe {
                coms.cast::<f64>()
            })
        {
            // rare case (|V| > |E|): `Ix` happens to be 8 bytes wide, matching `f64` --- reuse
            // `helper`/`coms` in place, no extra allocation.
            self.finish_modularity::<f64>(
                int_deg.par_zero_out_bytes(),
                tot_deg.par_zero_out_bytes(),
                communities,
                node_count,
                prev_comm_count,
            )?
        } else if let (Some(mut int_deg), Some(mut tot_deg)) =
            (unsafe { helper.cast::<f32>() }, unsafe {
                coms.cast::<f32>()
            })
        {
            // rare case (|V| > |E|): `Ix` happens to be 4 bytes wide, matching `f32` --- reuse
            // `helper`/`coms` in place, no extra allocation.
            self.finish_modularity::<f32>(
                int_deg.par_zero_out_bytes(),
                tot_deg.par_zero_out_bytes(),
                communities,
                node_count,
                prev_comm_count,
            )?
        } else {
            // rare case (|V| > |E|) *and* `Ix` is 1 or 2 bytes wide (`u8`/`u16`): no float width
            // matches, so there's nothing sound to reinterpret in place --- fall back to a
            // small, dedicated scratch allocation. Both conditions needed to reach here are
            // uncommon, so this is not a hot path.
            let id_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(11))?;
            let td_fn = self.build_cache_filename(CacheFile::GVELouvain, Some(12))?;
            let int_deg_owner = SharedSliceMut::<W>::abst_mem_mut(&id_fn, prev_comm_count, true)?;
            let tot_deg_owner = SharedSliceMut::<W>::abst_mem_mut(&td_fn, prev_comm_count, true)?;
            let modularity = self.finish_modularity::<W>(
                int_deg_owner.shared_slice(),
                tot_deg_owner.shared_slice(),
                communities,
                node_count,
                prev_comm_count,
            )?;
            drop(int_deg_owner);
            drop(tot_deg_owner);
            std::fs::remove_file(&id_fn)?;
            std::fs::remove_file(&td_fn)?;
            modularity
        };

        self.community.flush()?;

        self.g.cleanup_cache(CacheFile::GVELouvain)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{AlgoGVELouvain, Weight};
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use crate::graph::{GraphMemoryMap, IndexType, label::VoidLabel};
    use std::io::Write;

    /// Two 4-cliques (nodes 1-4, 5-8) bridged by a single edge (4-5): a classic community-
    /// detection fixture expected to split into (at least) two communities. Dense enough
    /// (|E| = 26 directed internal edges >> |V| = 8) that modularity computation always takes
    /// the common (`weights`-reuse) path, not the rare fallback.
    fn write_two_cliques_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating cliques mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "8 8 13").unwrap();
        for &(u, v) in &[
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (7, 8),
            (4, 5),
        ] {
            writeln!(f, "{u} {v}").unwrap();
        }
        f.flush().unwrap();
    }

    /// A long path graph (|V| = 20, |E| = 19*2 = 38 directed edges, still |E| > |V| here, so
    /// this alone doesn't exercise the rare fallback --- see
    /// `write_sparse_more_nodes_than_edges_mtx` for that).
    fn write_path_mtx(path: &std::path::Path, n: usize) {
        let mut f = std::fs::File::create(path).expect("error creating path mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "{n} {n} {}", n - 1).unwrap();
        for i in 1..n {
            writeln!(f, "{i} {}", i + 1).unwrap();
        }
        f.flush().unwrap();
    }

    /// A graph with strictly more nodes than edges (most nodes isolated): exercises Louvain's
    /// rare modularity fallback path (`self.g.size() > self.g.width()`), which --- depending on
    /// `Ix`'s width --- either reinterprets `helper`/`coms` in place as a same-width float, or,
    /// for `u8`/`u16`, falls back further to a dedicated scratch allocation.
    fn write_sparse_more_nodes_than_edges_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating sparse mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        // 40 nodes, only 2 edges (one connected pair, 38 isolated nodes): |V| = 40 > |E| = 4
        // (after symmetric doubling).
        writeln!(f, "40 40 2").unwrap();
        writeln!(f, "1 2").unwrap();
        writeln!(f, "3 4").unwrap();
        f.flush().unwrap();
    }

    fn run_louvain<Ix: IndexType, W: Weight>(path: &std::path::Path, id: &str) -> (usize, f64) {
        let graph =
            GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::from_file(path, Some(id.to_string()), None)
                .expect("graph parse should succeed");
        let louvain = AlgoGVELouvain::<_, _, _, W>::new(&graph)
            .unwrap_or_else(|e| panic!("louvain({id}) should succeed: {e}"));
        (louvain.community_count(), louvain.partition_modularity())
    }

    /// Runs Louvain across every supported `(Ix, W)` combination on a graph with a real community
    /// structure, asserting all combinations agree on the qualitative result (nodes 1-4 end up
    /// together, nodes 5-8 end up together, in a different community) and produce a finite,
    /// non-negative-looking modularity. `W=f32` vs `W=f64` aren't expected to be bit-identical
    /// (different floating-point precision throughout accumulation), but should be close.
    #[test]
    fn gve_louvain_cross_ix_and_weight_combinations() {
        let dir = unique_test_dir("gve_louvain_cross_ix_weight_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("cliques.mtx");
        write_two_cliques_mtx(&path);

        macro_rules! check {
            ($ix:ty, $w:ty, $tag:expr) => {{
                let (comm_count, modularity) =
                    run_louvain::<$ix, $w>(&path, &format!("gve_cross_{}_{}", $tag, run_tag));
                assert!(
                    comm_count >= 2,
                    "expected at least 2 communities for the two-cliques fixture with Ix={}, W={}, got {comm_count}",
                    stringify!($ix),
                    stringify!($w)
                );
                assert!(
                    modularity.is_finite() && modularity > 0.0,
                    "expected a positive, finite modularity for the two-cliques fixture with Ix={}, W={}, got {modularity}",
                    stringify!($ix),
                    stringify!($w)
                );
                (comm_count, modularity)
            }};
        }

        let (comm_usize_f64, mod_usize_f64) = check!(usize, f64, "usize_f64");
        let (comm_u32_f32, mod_u32_f32) = check!(u32, f32, "u32_f32");
        let (comm_u16_f32, _mod_u16_f32) = check!(u16, f32, "u16_f32");
        let (comm_u8_f32, _mod_u8_f32) = check!(u8, f32, "u8_f32");

        // all widths should find the same number of communities on this small, unambiguous
        // fixture (2 cliques bridged by one edge almost always resolves to exactly 2
        // communities regardless of float precision).
        assert_eq!(comm_usize_f64, comm_u32_f32);
        assert_eq!(comm_usize_f64, comm_u16_f32);
        assert_eq!(comm_usize_f64, comm_u8_f32);
        // f32 vs f64 modularity should be close (not necessarily bit-identical).
        assert!(
            (mod_usize_f64 - mod_u32_f32).abs() < 1e-3,
            "f64 modularity {mod_usize_f64} and f32 modularity {mod_u32_f32} diverge too much"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Exercises the rare modularity-computation fallback path (more nodes than edges) across
    /// every `Ix` width, including the narrowest ones (`u8`/`u16`) that have no same-width float
    /// to reinterpret in place, forcing the dedicated-scratch-allocation sub-fallback.
    #[test]
    fn gve_louvain_sparse_more_nodes_than_edges_fallback() {
        let dir = unique_test_dir("gve_louvain_sparse_fallback_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("sparse.mtx");
        write_sparse_more_nodes_than_edges_mtx(&path);

        macro_rules! check {
            ($ix:ty, $w:ty, $tag:expr) => {{
                let (comm_count, modularity) =
                    run_louvain::<$ix, $w>(&path, &format!("gve_sparse_{}_{}", $tag, run_tag));
                assert!(
                    modularity.is_finite(),
                    "modularity should be finite, got {modularity}"
                );
                comm_count
            }};
        }

        // every isolated node starts (and, with no positive-gain moves possible, stays) in its
        // own community, plus the one connected pair in each of two more communities ---
        // whatever the exact count, it must be identical across every (Ix, W) combination.
        let comm_usize_f64 = check!(usize, f64, "usize_f64");
        let comm_u32_f64 = check!(u32, f64, "u32_f64");
        let comm_u32_f32 = check!(u32, f32, "u32_f32");
        // `u16`/`u8` have no same-width float (4 or 8 bytes) to reinterpret `helper`/`coms`
        // into, forcing the dedicated-scratch-allocation sub-fallback.
        let comm_u16_f32 = check!(u16, f32, "u16_f32");
        let comm_u8_f32 = check!(u8, f32, "u8_f32");

        assert_eq!(comm_usize_f64, comm_u32_f64);
        assert_eq!(comm_usize_f64, comm_u32_f32);
        assert_eq!(comm_usize_f64, comm_u16_f32);
        assert_eq!(comm_usize_f64, comm_u8_f32);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Sanity check that a slightly larger, still-dense path graph also works end to end across
    /// `Ix`/`W` combinations (a different topology than the clique fixture, to catch anything
    /// specific to highly-clustered graphs).
    #[test]
    fn gve_louvain_path_graph_cross_ix_and_weight() {
        let dir = unique_test_dir("gve_louvain_path_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("path.mtx");
        write_path_mtx(&path, 20);

        let (comm_a, mod_a) =
            run_louvain::<usize, f64>(&path, &format!("gve_path_usize_f64_{run_tag}"));
        let (comm_b, mod_b) =
            run_louvain::<u32, f32>(&path, &format!("gve_path_u32_f32_{run_tag}"));

        assert!(comm_a >= 1 && mod_a.is_finite());
        assert!(comm_b >= 1 && mod_b.is_finite());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An undirected graph of `n` nodes and `edges` (each once), loaded with `threads` threads.
    fn undirected(
        name: &str,
        n: usize,
        edges: &[(usize, usize)],
        threads: u8,
    ) -> crate::graph::test_utils::TestGraph<(), (), usize> {
        let dir = unique_test_dir("gve_leiden");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u.max(v) + 1, u.min(v) + 1).unwrap();
        }
        drop(f);
        let id = format!("gve_leiden_{name}_{}", unique_run_tag());
        crate::graph::test_utils::TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    /// Whether every community's nodes are linked by edges between them.
    fn connected(n: usize, edges: &[(usize, usize)], comm: &[usize]) -> bool {
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                x = p[x];
            }
            x
        }
        for &(u, v) in edges {
            if comm[u] == comm[v] {
                let (a, b) = (find(&mut parent, u), find(&mut parent, v));
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut root_of = std::collections::HashMap::new();
        (0..n).all(|u| {
            let r = find(&mut parent, u);
            *root_of.entry(comm[u]).or_insert(r) == r
        })
    }

    /// `groups` cliques of `size` nodes, each linked to the next by an edge, plus random edges.
    fn planted_with_noise(
        groups: usize,
        size: usize,
        noise: usize,
        seed: u64,
    ) -> Vec<(usize, usize)> {
        let mut edges = vec![];
        for k in 0..groups {
            for i in 0..size {
                for j in i + 1..size {
                    edges.push((k * size + i, k * size + j));
                }
            }
            edges.push((k * size, ((k + 1) % groups) * size + 1));
        }
        let n = groups * size;
        edges.extend(
            crate::components::test_utils::random_arcs(n, noise, seed)
                .into_iter()
                .filter(|(u, v)| u != v),
        );
        edges
    }

    #[test]
    fn gve_communities_are_connected_and_leiden_is_as_good() {
        use crate::communities::Refinement;
        for (seed, threads) in (0..16u64).zip([1u8, 2, 4, 8].into_iter().cycle()) {
            let edges = if seed % 2 == 0 {
                planted_with_noise(12, 8, 60, 300 + seed)
            } else {
                crate::components::test_utils::random_arcs(150, 450, 300 + seed)
                    .into_iter()
                    .filter(|(u, v)| u != v)
                    .collect()
            };
            let n = edges.iter().map(|&(u, v)| u.max(v) + 1).max().unwrap();
            let g = undirected(&format!("conn_{seed}"), n, &edges, threads);
            let mut q = vec![];
            for refinement in [Refinement::None, Refinement::Leiden] {
                let mut l = AlgoGVELouvain::<_, _, _, f64>::new_with_refinement(
                    &g,
                    threads as usize,
                    refinement,
                )
                .unwrap();
                let comm: Vec<usize> = l.communities().iter().map(|c| c.index()).collect();
                assert!(
                    connected(n, &edges, &comm),
                    "seed {seed}, {refinement:?}: disconnected"
                );
                assert_eq!(l.community_count(), comm.iter().max().unwrap() + 1);
                let expected = g.modularity(l.communities(), l.community_count()).unwrap();
                assert!(
                    (l.partition_modularity() - expected).abs() < 1e-9,
                    "seed {seed}, {refinement:?}: {} vs {expected}",
                    l.partition_modularity()
                );
                q.push(l.partition_modularity());
                l.drop_cache().unwrap();
            }
            assert!(
                q[1] >= q[0] - 0.03,
                "seed {seed}: leiden {} vs louvain {}",
                q[1],
                q[0]
            );
        }
    }

    #[test]
    fn gve_leiden_recovers_planted_cliques() {
        let edges = planted_with_noise(20, 10, 0, 0);
        let g = undirected("planted", 200, &edges, 4);
        let mut l = AlgoGVELouvain::<_, _, _, f64>::new_leiden(&g).unwrap();
        let comm: Vec<usize> = l.communities().iter().map(|c| c.index()).collect();
        for k in 0..20 {
            assert!(
                (k * 10..k * 10 + 10).all(|u| comm[u] == comm[k * 10]),
                "clique {k} split: {comm:?}"
            );
        }
        assert_eq!(l.community_count(), 20);
        l.drop_cache().unwrap();
    }

    /// A partition's modularity, from scratch: Σ_c in_c / 2m − (tot_c / 2m)², arcs weighing what
    /// their labels say.
    fn reference_modularity<L: crate::graph::E>(
        g: &GraphMemoryMap<VoidLabel, L, usize>,
        com: &[usize],
    ) -> f64 {
        let n = g.size();
        let (mut inside, mut total) = (vec![0f64; n], vec![0f64; n]);
        let mut m2 = 0f64;
        for u in 0..n {
            for (v, label) in g.weighted_neighbours(u).unwrap() {
                let w = label.weight::<f64>();
                m2 += w;
                total[com[u]] += w;
                if com[u] == com[v] {
                    inside[com[u]] += w;
                }
            }
        }
        (0..n)
            .map(|c| inside[c] / m2 - (total[c] / m2).powi(2))
            .sum()
    }

    /// The two-cliques fixture with every edge weighing `weight`, except the bridge (4-5).
    fn write_weighted_cliques(path: &std::path::Path, weight: &str, bridge: &str) {
        let mut text = "%%MatrixMarket matrix coordinate real symmetric\n8 8 13\n".to_string();
        for (u, v) in [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (7, 8),
        ] {
            text += &format!("{u} {v} {weight}\n");
        }
        text += &format!("4 5 {bridge}\n");
        std::fs::write(path, text).unwrap();
    }

    fn louvain_of<L: crate::graph::E, W: Weight>(
        g: &GraphMemoryMap<VoidLabel, L, usize>,
        leiden: bool,
    ) -> (Vec<usize>, f64) {
        let l = if leiden {
            AlgoGVELouvain::<_, _, _, W>::new_leiden(g)
        } else {
            AlgoGVELouvain::<_, _, _, W>::new(g)
        }
        .unwrap();
        let com: Vec<usize> = (0..g.size()).map(|u| l.node_community(u)).collect();
        (com, l.partition_modularity())
    }

    /// Whether two partitions group nodes alike, whatever their community ids.
    fn same_grouping(a: &[usize], b: &[usize]) -> bool {
        (0..a.len()).all(|u| (0..a.len()).all(|v| (a[u] == a[v]) == (b[u] == b[v])))
    }

    /// Scaling every weight alike leaves modularity, and so the partition, unchanged: uniform
    /// weights must give exactly what unit (unweighted) ones do.
    #[test]
    fn gve_louvain_uniform_weights_match_unweighted() {
        use crate::graph::Weighted;
        let dir = unique_test_dir("gve_louvain_uniform_weights");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let tag = unique_run_tag();
        let (plain, weighted) = (dir.join("plain.mtx"), dir.join("weighted.mtx"));
        write_two_cliques_mtx(&plain);
        write_weighted_cliques(&weighted, "3.5", "3.5");
        let unweighted = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
            &plain,
            Some(format!("gve_uniform_plain_{tag}")),
            None,
        )
        .unwrap();
        let g = GraphMemoryMap::<VoidLabel, Weighted<f32>, usize>::from_file(
            &weighted,
            Some(format!("gve_uniform_weighted_{tag}")),
            None,
        )
        .unwrap();
        for leiden in [false, true] {
            let (want, want_q) = louvain_of::<_, f64>(&unweighted, leiden);
            for (com, q) in [
                louvain_of::<_, f64>(&g, leiden),
                louvain_of::<_, f32>(&g, leiden),
            ] {
                assert!(
                    same_grouping(&com, &want),
                    "leiden {leiden}: {com:?} vs {want:?}"
                );
                assert!(
                    (q - want_q).abs() < 1e-6,
                    "leiden {leiden}: {q} vs {want_q}"
                );
                assert!((q - reference_modularity(&g, &com)).abs() < 1e-6);
            }
        }
    }

    /// A heavy enough bridge pulls its ends together, which unit weights never would; the
    /// modularity reported is the weighted one.
    #[test]
    fn gve_louvain_follows_edge_weights() {
        use crate::graph::Weighted;
        let dir = unique_test_dir("gve_louvain_weights");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let tag = unique_run_tag();
        let path = dir.join("bridged.mtx");
        write_weighted_cliques(&path, "1", "50");
        let weighted = GraphMemoryMap::<VoidLabel, Weighted<f64>, usize>::from_file(
            &path,
            Some(format!("gve_weights_{tag}")),
            None,
        )
        .unwrap();
        // the same file, its values ignored
        let unweighted = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("gve_weights_ignored_{tag}")),
            None,
        )
        .unwrap();
        for leiden in [false, true] {
            let (com, q) = louvain_of::<_, f64>(&weighted, leiden);
            assert_eq!(
                com[3], com[4],
                "leiden {leiden}: bridge ends apart: {com:?}"
            );
            assert!((q - reference_modularity(&weighted, &com)).abs() < 1e-9);
            // the graph's own modularity is the weighted one too
            let count = com.iter().max().unwrap() + 1;
            let q_graph = weighted.modularity(&com, count).unwrap();
            assert!(
                (q_graph - q).abs() < 1e-9,
                "leiden {leiden}: {q_graph} vs {q}"
            );
            let (plain, _) = louvain_of::<_, f64>(&unweighted, leiden);
            assert_ne!(plain[3], plain[4], "leiden {leiden}: {plain:?}");
            assert!(!same_grouping(&com, &plain));
        }
    }

    /// A node of degree past `u16::MAX` (the unit-weight buffer's length) gets every edge's
    /// weight: the buffer used to be sliced to the degree, panicking.
    #[test]
    fn gve_louvain_handles_hubs_past_the_unit_weight_buffer() {
        let dir = unique_test_dir("gve_louvain_hub");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let leaves = u16::MAX as usize + 10;
        let mut text = format!(
            "%%MatrixMarket matrix coordinate pattern symmetric\n{0} {0} {leaves}\n",
            leaves + 1
        );
        for v in 2..=leaves + 1 {
            text += &format!("1 {v}\n");
        }
        let path = dir.join("star.mtx");
        std::fs::write(&path, text).unwrap();
        let (count, q) = run_louvain::<usize, f64>(&path, &format!("gve_hub_{}", unique_run_tag()));
        // a star is one community
        assert_eq!(count, 1);
        assert!(q.abs() < 1e-9, "{q}");
    }

    /// Weights modularity can't be computed with are refused, saying so.
    #[test]
    fn gve_louvain_rejects_bad_weights() {
        use crate::graph::Weighted;
        let dir = unique_test_dir("gve_louvain_bad_weights");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        for (name, w) in [("negative", "-1"), ("nan", "nan"), ("inf", "inf")] {
            let path = dir.join(format!("{name}.mtx"));
            write_weighted_cliques(&path, "1", w);
            let g = GraphMemoryMap::<VoidLabel, Weighted<f64>, usize>::from_file(
                &path,
                Some(format!("gve_bad_{name}_{}", unique_run_tag())),
                None,
            )
            .unwrap();
            let Err(err) = AlgoGVELouvain::<_, _, _, f64>::new(&g) else {
                panic!("{name}: ran");
            };
            assert!(
                err.to_string().contains("finite and non-negative"),
                "{name}: {err}"
            );
        }
    }

    /// Real-valued weights' community totals, summed by threads in any order, can round a hair
    /// below a member's own weight: moving it out mustn't fail (it once did), and the
    /// modularity found is still the weighted one.
    #[test]
    fn gve_louvain_tolerates_rounding_of_real_weights() {
        use crate::graph::Weighted;
        let dir = unique_test_dir("gve_louvain_rounding");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        for seed in 0..8u64 {
            let n = 300;
            let mut state = seed;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                state >> 33
            };
            let mut edges = std::collections::BTreeMap::new();
            for _ in 0..2000 {
                let (u, v) = (next() as usize % n, next() as usize % n);
                if u != v {
                    let w = 0.1 + (next() % 1_000_000) as f64 / 211_111.0;
                    edges.insert((u.max(v), u.min(v)), w);
                }
            }
            let mut text = format!(
                "%%MatrixMarket matrix coordinate real symmetric\n{n} {n} {}\n",
                edges.len()
            );
            for ((u, v), w) in &edges {
                text += &format!("{} {} {w}\n", u + 1, v + 1);
            }
            let path = dir.join(format!("r{seed}.mtx"));
            std::fs::write(&path, text).unwrap();
            let g = GraphMemoryMap::<VoidLabel, Weighted<f64>, usize>::from_file(
                &path,
                Some(format!("gve_rounding_{seed}_{}", unique_run_tag())),
                Some(16),
            )
            .unwrap();
            for leiden in [false, true] {
                let (com, q) = louvain_of::<_, f64>(&g, leiden);
                assert!(
                    (q - reference_modularity(&g, &com)).abs() < 1e-9,
                    "seed {seed}"
                );
                let (com, q) = louvain_of::<_, f32>(&g, leiden);
                assert!(
                    (q - reference_modularity(&g, &com)).abs() < 1e-5,
                    "seed {seed}"
                );
            }
        }
    }

    /// Louvain's (or Leiden's) communities and modularity.
    fn run(g: &GraphMemoryMap<(), (), usize>, leiden: bool) -> (Vec<usize>, f64) {
        let l = if leiden {
            AlgoGVELouvain::<_, _, _, f64>::new_leiden(g)
        } else {
            AlgoGVELouvain::<_, _, _, f64>::new(g)
        }
        .unwrap();
        let com: Vec<usize> = l.communities().iter().map(|c| c.index()).collect();
        (com, l.partition_modularity())
    }

    /// On a sparse planted partition (modularity 0.674), Louvain and Leiden find as much as
    /// networkx's Louvain does (0.60 to 0.63, by its seed): 0.597 and 0.616 here, single-threaded.
    /// With GVE-Louvain's initial tolerance of 0.01, local moving stopped while communities were
    /// still pairs (a fifth of them across blocks), which aggregating made permanent: 0.503.
    #[test]
    fn sparse_planted_partitions_are_found() {
        let (weighted, truth) =
            crate::communities::spectral::test::planted(8, 1250, 0.0102, 0.000366, 21);
        let edges: Vec<(usize, usize)> = weighted.iter().map(|e| (e.0, e.1)).collect();
        let g = undirected("sparse_planted", 10_000, &edges, 1);
        let truth_ix: Vec<usize> = truth.clone();
        let planted_q = g.modularity(&truth_ix, 8).unwrap();
        for leiden in [false, true] {
            let (_, q) = run(&g, leiden);
            assert!(
                q > planted_q - 0.12,
                "leiden {leiden}: {q} vs planted {planted_q}"
            );
        }
    }

    /// Single-threaded runs repeat exactly: communities tied in gain are listed (and the tie
    /// broken) in the same order every run.
    #[test]
    fn single_threaded_runs_repeat() {
        let (weighted, _) = crate::communities::spectral::test::planted(6, 300, 0.04, 0.002, 22);
        let edges: Vec<(usize, usize)> = weighted.iter().map(|e| (e.0, e.1)).collect();
        let g = undirected("repeat", 1800, &edges, 1);
        for leiden in [false, true] {
            let first = run(&g, leiden);
            for _ in 0..3 {
                assert_eq!(run(&g, leiden), first, "leiden {leiden}");
            }
        }
    }
}
