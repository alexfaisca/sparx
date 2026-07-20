use crate::graph;
use crate::graph::cache::AccumulatorAtomic;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
#[allow(unused_imports)]
use portable_atomic::{AtomicBool, AtomicF64, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
#[cfg(feature = "bench")]
use std::time::Instant;

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
    #[cfg(feature = "bench")]
    /// If feature = "bench" is enabled: store number of iterations, number of nodes (communities),
    /// time for louvain_move() for each pass performed on the graph.
    iters: Vec<(usize, usize, u128)>,
    _marker: std::marker::PhantomData<W>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> AlgoGVELouvain<'a, N, E, Ix, W> {
    /// Constants set according to the optimizeds parameters described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876).
    /// Described in 4.1.2 Limiting the number of iterations per pass.
    const MAX_ITERATIONS: usize = 20;
    /// Described in 4.1.3 Adjusting tolerance drop rate (threshold scaling).
    const TOLERANCE_DROP: f64 = 10.;
    /// Described in 4.1.4 Adjusting initial tolerance.
    const INITIAL_TOLERANCE: f64 = 0.01;
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
        let mut gve_louvain = Self::new_no_compute(g, threads)?;
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
        if Path::new(&l_fn).exists() {
            if let Ok(comms) = AbstractedProceduralMemoryMut::<Ix>::from_file_name(&l_fn) {
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
                        #[cfg(feature = "bench")]
                        iters: vec![],
                        _marker: std::marker::PhantomData,
                    });
                }
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
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> AlgoGVELouvain<'a, N, E, Ix, W> {
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
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryGVELouvain<Ix, W>, Box<dyn std::error::Error>> {
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
        g.require_symmetric("GVE-Louvain", None)?;
        let threads = threads.max(1);
        let out_fn = g.build_cache_filename(CacheFile::GVELouvain, None)?;
        let coms = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?;
        Ok(Self {
            g,
            community: coms,
            community_count: 0,
            modularity: 0.,
            threads,
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
                        let weights: Vec<W> = vec![W::one(); u16::MAX as usize];
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
                            gdw.write_slice(u_start, &weights[0..(u_end - u_start)]);

                            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                            *k.get_mut(u) = W::Atomic::new(W::from_usize(deg_u));
                            *sigma.get_mut(u) = W::Atomic::new(W::from_usize(deg_u));
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
    fn delta_q(
        k_u_c: f64,
        k_u_d: f64,
        k_u: f64,
        sig_c: f64,
        sig_d: f64,
        m: f64,
    ) -> f64 {
        assert!(k_u_c >= 0.0 && k_u_d >= 0.0 && k_u >= 0.0 && sig_c >= 0.0 && sig_d >= 0.0 && m >= 0.0);
        assert!(
            k_u_c.is_finite()
                && k_u_d.is_finite()
                && k_u.is_finite()
                && sig_c.is_finite()
                && sig_d.is_finite()
                && m.is_normal()
        );
        let dq = (k_u_c - k_u_d) / m - k_u * (k_u + sig_c - sig_d) / (2.0 * m * m);
        if dq > 1.0 {
            println!(
                "Deviant dq: ({k_u_c} - {k_u_d}) / {m} - {k_u} * ({k_u} + {sig_c} - {sig_d}) / (2. *{m} * {m}) = {dq}"
            );
        }
        dq
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
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // recalculate node load per thread
        let node_load = comm_count.div_ceil(threads);

        let mut l_iter = 0;
        let synchronize = Arc::new(Barrier::new(threads));

        for i in 0..Self::MAX_ITERATIONS {
            let global_delta_q = Arc::new(AtomicF64::new(0.0));

            // reset iteration metrics
            thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                // use dynamic style scheduling by splitting range [0, cur_node_count) for threads
                let mut thread_res = Vec::with_capacity(threads);
                for tid in 0..threads {
                    let mut local_delta_q_sum = 0.0;
                    let mut nmap: HashMap<usize, W> = HashMap::new();

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
                                    // reassign node community to itself
                                    *c.get_mut(u) = Ix::new(u);
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
                                            if s < k_u {
                                                None
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

            // check convergence criteria for the local-moving phase
            if global_delta_q.load(Ordering::Relaxed).abs() < tolerance {
                break;
            }
            l_iter += 1;
        }

        Ok(l_iter)
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
                let mut nm: HashMap<usize, W> = HashMap::new();

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

    /// Accumulates each community's internal/total degree (into `int_deg`/`tot_deg`, zeroed by
    /// the caller) and computes the resulting partition modularity.
    ///
    /// Generic over `T: Weight`, independently of the algorithm's own `W`: `int_deg`/`tot_deg`
    /// are purely a transient scratch computation local to this one function, so the caller is
    /// free to pick whichever `T` lets it reuse already-allocated memory in place (see call
    /// site), with the division/summation always happening in `f64` regardless, so `T` and `W`
    /// never need to match.
    #[allow(clippy::too_many_arguments)]
    fn finish_modularity<T: Weight>(
        &self,
        mut int_deg: SharedSliceMut<T>,
        mut tot_deg: SharedSliceMut<T>,
        communities: SharedSliceMut<Ix>,
        node_count: usize,
        prev_comm_count: usize,
        m2: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // sum internal degree ---> int_deg | sum total degree ---> tot_deg
        for u in 0..node_count {
            let comm_u = communities.get(u).index();
            let iter = self.g.neighbours(u)?;
            *tot_deg.get_mut(comm_u) += T::from_usize(iter.remaining_neighbours());
            for v in iter {
                if v >= node_count {
                    continue;
                } // safety
                if communities.get(v).index() == comm_u {
                    *int_deg.get_mut(comm_u) += T::one();
                }
            }
        }

        // modularity: sum_c( internal_dir_edges[c]/m2 - (Kc[c]/m2)^2 )
        let mut partition_modularity = 0.0f64;
        for c in 0..prev_comm_count {
            let lc_over_m2 = int_deg.get(c).to_f64() / m2;
            let kc_over_m2 = tot_deg.get(c).to_f64() / m2;
            partition_modularity += lc_over_m2 - kc_over_m2 * kc_over_m2;
        }
        Ok(partition_modularity)
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

        // 2m (since each undirected edge has weight 2)
        let m2 = W::from_usize(edge_count);
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
            let count_iter = Self::louvain_move(
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
            )?;

            #[cfg(feature = "bench")]
            {
                let elapsed = before.elapsed().as_micros();
                println!(
                    "{l_pass} did {count_iter} iterations on {prev_comm_count} nodes, in {elapsed} micros"
                );
                self.iters.push((count_iter, prev_comm_count, elapsed));
            }

            if count_iter == 0 {
                break;
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

            // if |Γ|/|Γ𝑜𝑙𝑑 | > 𝜏_𝑎𝑔𝑔 then break
            if new_comm_count as f64 / prev_comm_count as f64 > Self::AGGREGATION_TOLERANCE {
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

        self.community_count = prev_comm_count;

        // compute partition modularity
        if m2 == W::zero() {
            self.modularity = 0.0;
            self.g.cleanup_cache(CacheFile::GVELouvain)?;
            return Ok(());
        }
        let m2_f64 = m2.to_f64();

        self.modularity = if self.g.size() <= self.g.width() {
            // common case (|E| >= |V|): reuse the already-`W`-typed weight buffers directly ---
            // no cast, no extra allocation.
            let int_deg = weights.par_zero_out_bytes();
            let tot_deg = next_weights.par_zero_out_bytes();
            self.finish_modularity::<W>(
                int_deg,
                tot_deg,
                communities,
                node_count,
                prev_comm_count,
                m2_f64,
            )?
        } else if let (Some(mut int_deg), Some(mut tot_deg)) =
            (unsafe { helper.cast::<f64>() }, unsafe { coms.cast::<f64>() })
        {
            // rare case (|V| > |E|): `Ix` happens to be 8 bytes wide, matching `f64` --- reuse
            // `helper`/`coms` in place, no extra allocation.
            self.finish_modularity::<f64>(
                int_deg.par_zero_out_bytes(),
                tot_deg.par_zero_out_bytes(),
                communities,
                node_count,
                prev_comm_count,
                m2_f64,
            )?
        } else if let (Some(mut int_deg), Some(mut tot_deg)) =
            (unsafe { helper.cast::<f32>() }, unsafe { coms.cast::<f32>() })
        {
            // rare case (|V| > |E|): `Ix` happens to be 4 bytes wide, matching `f32` --- reuse
            // `helper`/`coms` in place, no extra allocation.
            self.finish_modularity::<f32>(
                int_deg.par_zero_out_bytes(),
                tot_deg.par_zero_out_bytes(),
                communities,
                node_count,
                prev_comm_count,
                m2_f64,
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
                m2_f64,
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
            (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
            (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8),
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
        let graph = GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::from_file(
            path,
            Some(id.to_string()),
            None,
        )
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
                assert!(modularity.is_finite(), "modularity should be finite, got {modularity}");
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

        let (comm_a, mod_a) = run_louvain::<usize, f64>(&path, &format!("gve_path_usize_f64_{run_tag}"));
        let (comm_b, mod_b) = run_louvain::<u32, f32>(&path, &format!("gve_path_u32_f32_{run_tag}"));

        assert!(comm_a >= 1 && mod_a.is_finite());
        assert!(comm_b >= 1 && mod_b.is_finite());

        std::fs::remove_dir_all(&dir).ok();
    }
}
