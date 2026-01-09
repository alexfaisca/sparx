use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
#[allow(unused_imports)]
use portable_atomic::{AtomicBool, AtomicF32, AtomicF64, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
#[cfg(feature = "bench")]
use std::time::Instant;

type Weight = f64;
type AtomicWeight = AtomicF64;

type ProceduralMemoryGVELouvain<Ix> = (
    // K' --- renamed to k
    AbstractedProceduralMemoryMut<AtomicWeight>,
    // Σ' --- renamed to sigma
    AbstractedProceduralMemoryMut<AtomicWeight>,
    // first CSR
    // G'.index --- renamed to gdi
    // NB: kept `usize`-width (not `Ix`), rather than `node_count`-bounded, because during
    // aggregation (`louvain_aggregate`) this array's `next_index` role is unsafely reinterpreted,
    // in-place, as an `AtomicUsize` array (`next_index.cast::<AtomicUsize>()`) to serve as a
    // lock-free edge counter --- a trick that requires the element size to match `AtomicUsize`'s.
    AbstractedProceduralMemoryMut<usize>,
    // G'.edges --- renamed to gde
    AbstractedProceduralMemoryMut<Ix>,
    // G'.weights --- renamed to gdw
    AbstractedProceduralMemoryMut<Weight>,
    // second CSR
    // G''.index --- renamed to gddi (see `gdi`'s note: swaps roles with it across passes, so it
    // is subject to the same `usize`-width constraint)
    AbstractedProceduralMemoryMut<usize>,
    // G''.edges --- renamed to gdde
    AbstractedProceduralMemoryMut<Ix>,
    // G''.weights --- renamed to gddw
    AbstractedProceduralMemoryMut<Weight>,
    // processed --- flags node as processed
    AbstractedProceduralMemoryMut<AtomicBool>,
    // C' --- renamed to coms (C is stored as a member of the struct)
    // NB: kept `usize`-width for the same reason as `gdi`/`gddi`: reinterpreted, in-place, as an
    // `f64` array (`coms.cast::<f64>()`) when computing the final partition's modularity.
    AbstractedProceduralMemoryMut<usize>,
    // Helper array for C' renumbering, dendrogram lookup & CSR aggregation
    // NB: kept `usize`-width for the same reason (reinterpreted as `f64` via `helper.cast::<f64>()`).
    AbstractedProceduralMemoryMut<usize>,
);

/// For the computation of Louvain partitions as described in ["GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.04876) on [`GraphMemoryMap`] instances.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoGVELouvain<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which the partition is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped array containing each node's community.
    community: AbstractedProceduralMemoryMut<usize>,
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
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoGVELouvain<'a, N, E, Ix> {
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
            if let Ok(comms) = AbstractedProceduralMemoryMut::from_file_name(&l_fn) {
                let mut max = 0;
                comms.as_slice().iter().for_each(|&c| {
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
        *self.community.get(idx)
    }

    /// Returns a slice where each element corresponds (by index) to each node's community label.
    pub fn communities(&self) -> &[usize] {
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
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoGVELouvain<'a, N, E, Ix> {
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
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryGVELouvain<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryGVELouvain<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix>,
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
        let coms = SharedSliceMut::<usize>::abst_mem_mut(&out_fn, g.size(), true)?;
        Ok(Self {
            g,
            community: coms,
            community_count: 0,
            modularity: 0.,
            threads,
            #[cfg(feature = "bench")]
            iters: Vec::new(),
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<ProceduralMemoryGVELouvain<Ix>, Box<dyn std::error::Error>> {
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

        let k = SharedSliceMut::<AtomicWeight>::abst_mem_mut(&k_fn, node_count, true)?;
        let sigma = SharedSliceMut::<AtomicWeight>::abst_mem_mut(&s_fn, node_count, true)?;
        let gdi = SharedSliceMut::<usize>::abst_mem_mut(&gdi_fn, 2 * node_count, true)?;
        let gde = SharedSliceMut::<Ix>::abst_mem_mut(&gde_fn, edge_count, true)?;
        let gdw = SharedSliceMut::<Weight>::abst_mem_mut(&gdw_fn, edge_count, true)?;
        let gddi = SharedSliceMut::<usize>::abst_mem_mut(&gddi_fn, 2 * node_count, true)?;
        let gdde = SharedSliceMut::<Ix>::abst_mem_mut(&gdde_fn, edge_count, true)?;
        let gddw = SharedSliceMut::<Weight>::abst_mem_mut(&gddw_fn, edge_count, true)?;
        let processed = SharedSliceMut::<AtomicBool>::abst_mem_mut(&a_fn, node_count, true)?;
        let coms = SharedSliceMut::<usize>::abst_mem_mut(&c_fn, node_count, true)?;
        // has to be size max(|V|, |E|) as it is both used to store the 'holey' CSR (O(|E|) space) and
        // renumbered community memberships (O(|V|) space)
        let help = SharedSliceMut::<usize>::abst_mem_mut(&h_fn, edge_count.max(node_count), true)?;

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
                        let weights: Vec<Weight> = vec![1.; u16::MAX as usize];
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
                            *helper.get_mut(u) = usize::MAX;
                            *communities.get_mut(u) = u;
                            *coms.get_mut(u) = u;
                            *gdi.get_mut(out_offset) = u_start;
                            *gdi.get_mut(out_offset + 1) = u_end;

                            // copy G onto G' --- no copy/alloc needed: `edges` already borrows the
                            // `Ix`-width neighbours straight out of the original graph's mmap, and
                            // `gde` now stores them at the same width, so this is a plain memcpy.
                            gde.write_slice(u_start, edges);
                            gdw.write_slice(u_start, &weights[0..(u_end - u_start)]);

                            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                            *k.get_mut(u) = AtomicWeight::new(deg_u as Weight);
                            *sigma.get_mut(u) = AtomicWeight::new(deg_u as Weight);
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
    #[inline(always)]
    fn delta_q(
        k_u_c: Weight,
        k_u_d: Weight,
        k_u: Weight,
        sig_c: Weight,
        sig_d: Weight,
        m: Weight,
    ) -> Weight {
        assert!(k_u_c >= 0. && k_u_d >= 0. && k_u >= 0. && sig_c >= 0. && sig_d >= 0. && m >= 0.);
        assert!(
            k_u_c.is_finite()
                && k_u_d.is_finite()
                && k_u.is_finite()
                && sig_c.is_finite()
                && sig_d.is_finite()
                && m.is_normal()
        );
        let dq = (k_u_c - k_u_d) / m - k_u * (k_u + sig_c - sig_d) / (2. * m * m);
        if dq > 1. {
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
        nmap: &mut HashMap<usize, Weight>,
        index: SharedSliceMut<usize>,
        edges: SharedSliceMut<Ix>,
        weights: SharedSliceMut<Weight>,
        coms: SharedSliceMut<usize>,
        u: usize,
        self_allowed: bool,
    ) {
        for e_idx in *index.get(2 * u)..*index.get(2 * u + 1) {
            let v = edges.get(e_idx).index();
            if !self_allowed && v == u {
                continue;
            }
            if *coms.get(v) > comm_count {
                println!(
                    "WARNING!! Impossible neighbour commnity for node {u} (neighbour {}): {{{} > {comm_count}}} but c: {}",
                    v,
                    *coms.get(v),
                    *coms.get(*coms.get(v))
                );
                println!(
                    "{}..{} at {e_idx}\n",
                    *index.get(2 * u),
                    *index.get(2 * u + 1)
                );
            }
            // 𝐻𝑡[𝐶'[v]] ← 𝐻𝑡[𝐶'[v]] + w
            *nmap.entry(*coms.get(v)).or_insert(0.) += *weights.get(e_idx);
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
        nmap: &mut HashMap<usize, Weight>,
        sigma: SharedSliceMut<AtomicWeight>,
        c_u: usize,
        k_u: Weight,
        m: Weight,
    ) -> (usize, Weight) {
        // houses c*
        let mut best_gain = 0.;
        // houses 𝛿𝑄*
        let mut best_comm = c_u;

        // Ku->d
        let k_u_d = nmap.get(&c_u).cloned().unwrap_or(0.);
        // Σd
        let sig_d = sigma.get(c_u).load(Ordering::Relaxed);

        for (&c_v, &k_u_c) in nmap.iter() {
            if c_v == c_u {
                continue;
            }
            // Σc
            let sig_c = sigma.get(c_v).load(Ordering::Relaxed);

            // 𝛿𝑄 = (Ku->c - Ku->d) / m - (Ku * (Σc - Σd) / (2m^2))
            let dq = Self::delta_q(k_u_c, k_u_d, k_u, sig_c, sig_d, m);
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
        k: SharedSliceMut<AtomicWeight>,
        sigma: SharedSliceMut<AtomicWeight>,
        processed: SharedSliceMut<AtomicBool>,
        idx: SharedSliceMut<usize>,
        mut next_index: SharedSliceMut<usize>,
        e: SharedSliceMut<Ix>,
        w: SharedSliceMut<Weight>,
        mut c: SharedSliceMut<usize>,
        mut helper: SharedSliceMut<usize>,
        m: Weight,
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
                    let mut nmap: HashMap<usize, Weight> = HashMap::new();

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
                                    *helper.get_mut(u) = usize::MAX;
                                    *next_index.get_mut(u * 2) = 0;
                                    // reassign node community to itself
                                    *c.get_mut(u) = u;
                                }
                            }

                            // wait for all threads to finish current pass initialization
                            synchronize.wait();

                            for u in begin..end {
                                // skip this vertex if it was marked as pruned (no need to process)
                                if processed.get(u).swap(true, Ordering::Relaxed) {
                                    continue;
                                }
                                let c_u = *c.get(u);
                                // get weight of u k[u]
                                let k_u = k.get(u).load(Ordering::Relaxed);
                                // gather neighbour communities and edge weight sums using the adjacency list of u

                                nmap.clear();
                                let nm = &mut nmap;
                                Self::scan_communities(comm_count, nm, idx, e, w, c, u, false);

                                let (b_c, b_g) = Self::choose_community(nm, s.clone(), c_u, k_u, m);

                                // move u to the best community if it yields positive gain
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
                                        *c.get_mut(u) = b_c;
                                        local_delta_q_sum += b_g;
                                        // mark all neighbors of u to be processed in the next iteration
                                        for idx in *idx.get(u * 2)..*idx.get(u * 2 + 1) {
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
        k: SharedSliceMut<AtomicWeight>,
        sigma: SharedSliceMut<AtomicWeight>,
        processed: SharedSliceMut<AtomicBool>,
        index: SharedSliceMut<usize>,
        mut next_index: SharedSliceMut<usize>,
        edges: SharedSliceMut<Ix>,
        mut next_edges: SharedSliceMut<Ix>,
        weights: SharedSliceMut<Weight>,
        mut next_weights: SharedSliceMut<Weight>,
        coms: SharedSliceMut<usize>,
        mut helper: SharedSliceMut<usize>,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // recalculate comm load per thread
        let prev_comm_load = comm_count.div_ceil(threads);
        let new_comm_load = new_comm_count.div_ceil(threads);

        let synchronize = Arc::new(Barrier::new(threads));
        let counter = unsafe {
            next_index
                .cast::<AtomicUsize>()
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "error getting atomic edge counters".into()
                })?
        };

        // exclusive scan
        let mut sum = 0;
        for u in 0..new_comm_count {
            let degree = *next_index.get(u * 2);
            *next_index.get_mut(u * 2) = sum;
            *next_index.get_mut(u * 2 + 1) = sum;
            sum += degree;
        }

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            // use dynamic style scheduling by splitting range [0, cur_node_count) for threads
            let mut thread_res = Vec::with_capacity(threads);
            for tid in 0..threads {

                // hashmap to store edges for each community
                let mut nm: HashMap<usize, Weight> = HashMap::new();

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
                            let degree = *index.get(u * 2 + 1) - *index.get(u * 2);
                            // guard against isolated nodes to not go over community index boundary
                            if degree > 0 {
                                // add edge (C'[u], u)
                                let idx = counter.get(*coms.get(u) * 2 + 1).fetch_add(1, Ordering::Relaxed);
                                *helper.get_mut(idx) = u;
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
                            let mut total_weight = 0.;

                            // mark unprocessed
                            processed.get(c).store(false, Ordering::Relaxed);

                            // clear previous community's neighbour map
                            nm.clear();

                            // for every node in the community scan edges to accumulate supergraph
                            // edge weights
                            for e_idx in *next_index.get(c * 2)..*next_index.get(c * 2 + 1) {
                                Self::scan_communities(comm_count, &mut nm, index, edges, weights, coms, *helper.get(e_idx), true);
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
                                *next_edges.get_mut(*next_index.get(c * 2) + total) = Ix::new(d);
                                *next_weights.get_mut(*next_index.get(c * 2) + total) = w;

                                // increment counts
                                total += 1;
                                total_weight += w;
                            }

                            // set cmommunity index upper boundary
                            *next_index.get_mut(c * 2 + 1) = *next_index.get(c * 2) + total;

                            // check for indexing missalignment errors
                            if c > 0 && *next_index.get(c * 2) < *next_index.get(c * 2 - 1) {
                                println!("error {} {} {}", new_begin, *next_index.get(c * 2), *next_index.get(c * 2 - 1));
                                return Err(
                                    format!("error building supergraph: index missalignment, node {} ends at {} and node {c} starts at {}",
                                            c - 1, *next_index.get(c * 2 - 1), *next_index.get(c * 2)
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

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryGVELouvain<Ix>,
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
        let m2 = edge_count as Weight;
        let m = m2 / 2.;

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
                let degree = *index.get(u * 2 + 1) - *index.get(u * 2);
                let c = *coms.get(u);
                let new_c = *helper.get(c);
                if new_c == usize::MAX {
                    *helper.get_mut(c) = new_comm_count;
                    *coms.get_mut(u) = new_comm_count;
                    *next_index.get_mut(new_comm_count * 2) = degree;
                    new_comm_count += 1;
                } else {
                    *coms.get_mut(u) = new_c;
                    *next_index.get_mut(new_c * 2) += degree;
                }
            }

            // 𝐶 ← Lookup dendrogram using 𝐶 to 𝐶
            for orig in 0..node_count {
                let c = *communities.get(orig);
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
        if m2 == 0. {
            self.modularity = 0.0;
            self.g.cleanup_cache(CacheFile::GVELouvain)?;
            return Ok(());
        }

        let mut int_deg = if self.g.size() <= self.g.width() {
            weights.par_zero_out_bytes()
        } else {
            unsafe {
                helper
                    .cast::<f64>()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error getting atomic edge counters".into()
                    })?
                    .par_zero_out_bytes()
            }
        };
        let mut tot_deg = if self.g.size() <= self.g.width() {
            next_weights.par_zero_out_bytes()
        } else {
            unsafe {
                coms.cast::<f64>()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error getting atomic edge counters".into()
                    })?
                    .par_zero_out_bytes()
            }
        };

        // sum internal degree ---> sigma | sum total degree ---> k
        for u in 0..node_count {
            let comm_u = *communities.get(u);
            let iter = self.g.neighbours(u)?;
            *tot_deg.get_mut(*communities.get(u)) += iter.remaining_neighbours() as Weight;
            for v in iter {
                if v >= node_count {
                    continue;
                } // safety
                if *communities.get(v) == comm_u {
                    *int_deg.get_mut(comm_u) += 1.;
                }
            }
        }

        // modularity: sum_c( internal_dir_edges[c]/m2 - (Kc[c]/m2)^2 )
        let mut partition_modularity = 0.0f64;
        #[allow(clippy::unnecessary_cast)]
        for c in 0..prev_comm_count {
            let lc_over_m2 = *int_deg.get(c) / m2;
            let kc_over_m2 = *tot_deg.get(c) / m2;
            partition_modularity += (lc_over_m2 - kc_over_m2 * kc_over_m2) as f64;
        }

        self.modularity = partition_modularity;
        self.community.flush()?;

        self.g.cleanup_cache(CacheFile::GVELouvain)?;

        Ok(())
    }
}
