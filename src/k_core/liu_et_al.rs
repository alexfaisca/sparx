use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Barrier};

type ProceduralMemoryLiuEtAL<Ix> = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<AtomicBool>,
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<Ix>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoLiuEtAl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which node/edge coreness is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the coreness of each edge.
    k_cores: AbstractedProceduralMemoryMut<u8>,
    threads: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoLiuEtAl<'a, N, E, Ix> {
    /// Performs k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
    ///
    /// * Note: we did not implement the *Node Sampling*[^1] scheme optimization (used for high degree nodes), as our objective is the decomposition of very large sparse graphs.
    ///
    /// [^1]: details on how to implement this potimization and how it works can be found in the *4.1.2 Details about the Sampling Scheme.* section of the aforementioned paper in pp. 6-7.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance in which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut liu_et_al = Self::new_no_compute(g, g.thread_num())?;
        let proc_mem = liu_et_al.init_cache_mem()?;

        liu_et_al.compute_with_proc_mem(proc_mem)?;

        Ok(liu_et_al)
    }
    /// Searches for a previously cached result, and if not found performs k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
    ///
    /// * Note: we did not implement the *Node Sampling*[^1] scheme optimization (used for high degree nodes), as our objective is the decomposition of very large sparse graphs.
    ///
    /// [^1]: details on how to implement this potimization and how it works can be found in the *4.1.2 Details about the Sampling Scheme.* section of the aforementioned paper in pp. 6-7.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance in which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let c_fn = g.build_cache_filename(CacheFile::KCoreLEA, None)?;
        if Path::new(&c_fn).exists() {
            if let Ok(k_cores) = AbstractedProceduralMemoryMut::from_file_name(&c_fn) {
                return Ok(Self {
                    g,
                    k_cores,
                    threads: g.thread_num().max(1),
                });
            }
        }
        Self::new(g)
    }

    /// Performs k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
    ///
    /// * Note: we did not implement the *Node Sampling*[^1] scheme optimization (used for high degree nodes), as our objective is the decomposition of very large sparse graphs.
    ///
    /// [^1]: details on how to implement this potimization and how it works can be found in the *4.1.2 Details about the Sampling Scheme.* section of the aforementioned paper in pp. 6-7.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance in which k-core decomposition is to be performed in.
    /// * `threads` --- the number of threads to be used in the computation.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut liu_et_al = Self::new_no_compute(g, threads)?;
        let proc_mem = liu_et_al.init_cache_mem()?;

        liu_et_al.compute_with_proc_mem(proc_mem)?;

        Ok(liu_et_al)
    }

    /// Returns the coreness of a given edge of a [`GraphMemoryMap`] instance.
    ///
    /// # Arguments
    ///
    /// * `e_idx` --- the index of the edge whose coreness is to be returned.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn coreness(&self, e_idx: usize) -> u8 {
        assert!(e_idx < self.g.width());
        *self.k_cores.get(e_idx)
    }

    /// Returns a slice containing the coreness of each edge of the [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn k_cores(&self) -> &[u8] {
        self.k_cores.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        this.g.cleanup_cache(CacheFile::KCoreLEA)?;
        let out_fn = this.g.build_cache_filename(CacheFile::KCoreLEA, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoLiuEtAl<'a, N, E, Ix> {
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
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryLiuEtAL<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryLiuEtAL<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryLiuEtAL<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryLiuEtAL<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threads = threads.max(1);
        let out_fn = g.build_cache_filename(CacheFile::KCoreLEA, None)?;
        let k_cores = SharedSliceMut::<u8>::abst_mem_mut(&out_fn, g.size(), true)?;
        Ok(Self {
            g,
            k_cores,
            threads,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryLiuEtAL<Ix>, Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let d_fn = self.g.build_cache_filename(CacheFile::KCoreLEA, Some(0))?;
        let ni_fn = self.g.build_cache_filename(CacheFile::KCoreLEA, Some(1))?;
        let a_fn = self.g.build_cache_filename(CacheFile::KCoreLEA, Some(2))?;
        let f_fn = self.g.build_cache_filename(CacheFile::KCoreLEA, Some(3))?;
        let fs_fn = self.g.build_cache_filename(CacheFile::KCoreLEA, Some(4))?;

        let degree = SharedSliceMut::<AtomicU8>::abst_mem_mut(&d_fn, node_count, true)?;
        let node_index = SharedSliceMut::<Ix>::abst_mem_mut(&ni_fn, node_count, true)?;
        let alive = SharedSliceMut::<AtomicBool>::abst_mem_mut(&a_fn, node_count, true)?;
        let frontier = SharedSliceMut::<Ix>::abst_mem_mut(&f_fn, edge_count, true)?;
        let frontier_swap = SharedSliceMut::<Ix>::abst_mem_mut(&fs_fn, edge_count, true)?;

        Ok((degree, node_index, alive, frontier, frontier_swap))
    }

    fn compute_with_proc_mem_impl(
        &self,
        proc_mem: ProceduralMemoryLiuEtAL<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        if node_count == 0 {
            return Ok(());
        }

        let threads = self.threads;
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbour_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        let (degree, _node_index, alive, frontier, swap) = proc_mem;
        let mut coreness = self.k_cores.shared_slice();

        // Initialize
        let total_dead_nodes = thread::scope(|s| -> usize {
            let mut dead_nodes = vec![];
            for tid in 0..threads {
                let mut degree = degree.shared_slice();
                let mut alive = alive.shared_slice();

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                dead_nodes.push(s.spawn(move |_| -> usize {
                    let mut dead_nodes = 0;

                    for u in start..end {
                        let deg_u = index_ptr.get(u + 1).index() - index_ptr.get(u).index();
                        *degree.get_mut(u) = AtomicU8::new(deg_u as u8);

                        if deg_u == 0 {
                            dead_nodes += 1;
                            *alive.get_mut(u) = AtomicBool::new(false);
                        } else {
                            *alive.get_mut(u) = AtomicBool::new(true);
                        }
                    }

                    dead_nodes
                }));
            }
            let mut total_dead_nodes = 0;
            dead_nodes
                .into_iter()
                .map(|e| e.join().expect("error"))
                .for_each(|e| total_dead_nodes += e);
            let _ = degree.flush_async();
            let _ = alive.flush_async();
            total_dead_nodes
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // ditch node sampling as graphs are inherintely sparse
        // use veertical granularity control:
        // When peeling a low-degree vertex 𝑣, we
        // place all its active neighbors in a FIFO queue, referred to as the local
        // queue of 𝑣, and process all vertices in the local queue sequentially.
        // When we decrementing the induced degree of a neighbor 𝑢, if 𝑑˜ [𝑢]
        // drops to 𝑘 (line 6 in Alg. 3), instead of adding 𝑢 to Fnext , we add 𝑢 to
        // the local queue. This allows 𝑢 to be processed in the same subround
        // as 𝑣, rather than waiting for the next subround. We refer to this
        // process as a local search at 𝑣.

        // --- core-peeling loop (Liu et al. algorithm) ---
        // for nodes with degree <= 16 no bucketing is used
        let mut k = 1u8;
        // number of vertices not yet peeled
        let remaining_global = Arc::new(AtomicUsize::new(node_count - total_dead_nodes));
        let frontier = SharedQueueMut::<Ix>::from_shared_slice(frontier.shared_slice());
        let swap = SharedQueueMut::<Ix>::from_shared_slice(swap.shared_slice());
        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            for tid in 0..threads {
                let degree = &degree;
                let alive = alive.shared_slice();
                let mut frontier = frontier.clone();
                let mut swap = swap.clone();

                let remaining_global = Arc::clone(&remaining_global);
                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        let mut local_stack: Vec<Ix> = vec![];
                        let max_len = 256;

                        while remaining_global.load(Ordering::Relaxed) > 0 {
                            synchronize.wait();
                            // build initial frontier = all vertices with degree <= k that are still active.
                            for u in start..end {
                                if alive.get(u).load(Ordering::Relaxed)
                                    && degree.get(u).load(Ordering::Relaxed) <= k
                                {
                                    alive.get(u).store(false, Ordering::Relaxed);
                                    let _ = frontier.push(Ix::new(u));
                                }
                            }

                            synchronize.wait();

                            if frontier.is_empty() {
                                k = match k.overflowing_add(1) {
                                    (r, false) => r,
                                    _ => {
                                        return Err(format!(
                                            "error overflow when adding to k ({k} + 1)"
                                        )
                                        .into());
                                    }
                                };
                                continue;
                            }

                            // process subrounds for current k: peel all vertices with degree k.
                            while !frontier.is_empty() {
                                if tid == 0 {
                                    remaining_global.fetch_sub(frontier.len(), Ordering::Relaxed);
                                }

                                let chunk_size = frontier.len().div_ceil(threads);
                                let start = std::cmp::min(tid * chunk_size, frontier.len());
                                let end = std::cmp::min(start + chunk_size, frontier.len());

                                if let Some(chunk) = frontier.slice(start, end) {
                                    for i in 0..end - start {
                                        // set coreness and decrement neighbour degrees
                                        let u = chunk.get(i).index();
                                        *coreness.get_mut(u) = k;

                                        // for each neighbor v of u:
                                        for idx in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                                            let v = neighbour_ptr.get(idx).index();
                                            if let Ok(old) = degree.get(v).fetch_update(
                                                Ordering::Relaxed,
                                                Ordering::Relaxed,
                                                |x| {
                                                    if x > k {
                                                        match x.overflowing_sub(1) {
                                                            (r, false) => Some(r),
                                                            _ => None,
                                                        }
                                                    } else {
                                                        None
                                                    }
                                                },
                                            ) {
                                                if old == k + 1 {
                                                    let life =
                                                        alive.get(v).swap(false, Ordering::Relaxed);
                                                    if life {
                                                        if local_stack.len() < max_len {
                                                            local_stack.push(Ix::new(v));
                                                        } else {
                                                            let _ = swap.push(Ix::new(v));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // process local stack
                                        let mut read_in_stack: usize = 0;
                                        while let Some(u) = local_stack.pop() {
                                            // set coreness and decrement neighbour degrees
                                            read_in_stack += 1;
                                            let u = u.index();
                                            *coreness.get_mut(u) = k;

                                            for idx in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                                                let v = neighbour_ptr.get(idx).index();
                                                if let Ok(old) = degree.get(v).fetch_update(
                                                    Ordering::Relaxed,
                                                    Ordering::Relaxed,
                                                    |x| {
                                                        if x > k {
                                                            match x.overflowing_sub(1) {
                                                                (r, false) => Some(r),
                                                                _ => None,
                                                            }
                                                        } else {
                                                            None
                                                        }
                                                    },
                                                ) {
                                                    if old == k + 1 {
                                                        let life = alive
                                                            .get(v)
                                                            .swap(false, Ordering::Relaxed);
                                                        if life {
                                                            if local_stack.len() < max_len {
                                                                local_stack.push(Ix::new(v));
                                                            } else {
                                                                let _ = swap.push(Ix::new(v));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        remaining_global
                                            .fetch_sub(read_in_stack, Ordering::Relaxed);
                                    }
                                }

                                synchronize.wait();

                                swap = std::mem::replace(&mut frontier, swap).clear();

                                synchronize.wait();
                            }
                            k = match k.overflowing_add(1) {
                                (r, false) => r,
                                _ => {
                                    return Err(format!(
                                        "error overflow when adding to k ({k} + 1)"
                                    )
                                    .into());
                                }
                            };
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        let env_verbose_val = std::env::var("BRUIJNX_VERBOSE").unwrap_or_else(|_| "0".to_string());
        let verbose: bool = env_verbose_val == "1";
        if verbose {
            let mut r = vec![0usize; u8::MAX as usize];
            let mut max = 0;
            for i in 0..node_count {
                if *coreness.get(i) > max {
                    max = *coreness.get(i);
                }
                r[*coreness.get(i) as usize] += 1;
            }
            r.resize(max as usize + 1, 0);
            println!("k-cores {:?}", r);
        }

        // flush output to ensure all data is written to disk
        self.k_cores.flush_async()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::KCoreLEA)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{k_core::verify_k_cores, test_common::get_or_init_dataset_cache_entry};

    use super::*;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use paste::paste;
    use std::io::Write;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<k_cores_liu_et_al_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        let graph = GraphMemoryMap::init_from_cache(
            get_or_init_dataset_cache_entry(path.as_ref())?,
            Some(16),
        )?;
        let liu_et_al_k_cores = AlgoLiuEtAl::new(&graph)?;

        verify_k_cores(&graph, liu_et_al_k_cores.k_cores)
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => "../ggcat/graphs/random_graph_1_5.lz4",
        ggcat_2_5 => "../ggcat/graphs/random_graph_2_5.lz4",
        ggcat_3_5 => "../ggcat/graphs/random_graph_3_5.lz4",
        ggcat_4_5 => "../ggcat/graphs/random_graph_4_5.lz4",
        ggcat_5_5 => "../ggcat/graphs/random_graph_5_5.lz4",
        ggcat_6_5 => "../ggcat/graphs/random_graph_6_5.lz4",
        ggcat_7_5 => "../ggcat/graphs/random_graph_7_5.lz4",
        ggcat_8_5 => "../ggcat/graphs/random_graph_8_5.lz4",
        ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }

    /// Same nested k-core fixture used by `batagelj_zaversnik`'s cross-Ix test --- duplicated
    /// locally (rather than shared) to keep each algorithm's test module self-contained, matching
    /// this crate's existing per-file fixture convention.
    fn write_nested_cores_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating nested-cores mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "9 9 15").unwrap();
        for &(u, v) in &[
            (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5),
            (5, 6),
            (6, 7), (6, 8), (7, 8),
            (8, 9),
        ] {
            writeln!(f, "{u} {v}").unwrap();
        }
        f.flush().unwrap();
    }

    /// See `batagelj_zaversnik`'s identically-named helper for rationale (this algorithm's
    /// `k_cores` is likewise node-indexed, not edge-indexed as `verify_k_cores` assumes).
    fn assert_valid_node_indexed_k_cores<Ix: crate::graph::IndexType>(
        graph: &GraphMemoryMap<crate::graph::label::VoidLabel, crate::graph::label::VoidLabel, Ix>,
        coreness: &[u8],
    ) {
        assert_eq!(coreness.len(), graph.size(), "k_cores should be node-indexed");
        for v in 0..graph.size() {
            let k_v = coreness[v];
            let qualifying_neighbours = graph
                .neighbours(v)
                .expect("neighbours should succeed")
                .filter(|&u| coreness[u] >= k_v)
                .count();
            assert!(
                qualifying_neighbours >= k_v as usize,
                "node {v} claims coreness {k_v} but only has {qualifying_neighbours} neighbours with coreness >= {k_v}"
            );
        }
    }

    fn run_liu<Ix: crate::graph::IndexType>(path: &std::path::Path, id: &str) -> Vec<u8> {
        let graph = GraphMemoryMap::<crate::graph::label::VoidLabel, crate::graph::label::VoidLabel, Ix>::from_file(
            path,
            Some(id.to_string()),
            None,
        )
        .expect("graph parse should succeed");
        let liu = AlgoLiuEtAl::new(&graph)
            .unwrap_or_else(|e| panic!("liu-et-al({id}) should succeed: {e}"));
        let coreness = liu.k_cores().to_vec();
        assert_valid_node_indexed_k_cores(&graph, &coreness);
        coreness
    }

    /// Runs the Liu et al. k-core decomposition across several `Ix` widths on a graph with
    /// genuinely nested core structure, asserting every width agrees on the exact (bit-for-bit
    /// identical) per-node coreness assignment, that the decomposition is itself structurally
    /// valid, and that it agrees with the (independently implemented) Batagelj-Zaversnik
    /// algorithm's result on the same graph --- a cross-algorithm agreement check, since the
    /// canonical k-core decomposition is unique regardless of which peeling strategy computes it.
    #[test]
    fn k_cores_liu_et_al_cross_ix_widths_and_cross_algorithm() {
        let dir = unique_test_dir("liu_cross_ix_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("nested_cores.mtx");
        write_nested_cores_mtx(&path);

        let usize_cores = run_liu::<usize>(&path, &format!("liu_usize_{run_tag}"));
        let u32_cores = run_liu::<u32>(&path, &format!("liu_u32_{run_tag}"));
        let u16_cores = run_liu::<u16>(&path, &format!("liu_u16_{run_tag}"));

        assert_eq!(usize_cores, u32_cores, "Ix=u32 coreness should match Ix=usize exactly");
        assert_eq!(usize_cores, u16_cores, "Ix=u16 coreness should match Ix=usize exactly");

        // exact expected values, independently verified via standard k-core peeling (see
        // batagelj_zaversnik's identically-structured test for the full derivation).
        assert_eq!(
            usize_cores,
            vec![4, 4, 4, 4, 4, 2, 2, 2, 1],
            "unexpected coreness values for the nested-cores fixture"
        );

        // cross-algorithm agreement with batagelj_zaversnik on the exact same graph/cache entry
        let graph = GraphMemoryMap::<crate::graph::label::VoidLabel, crate::graph::label::VoidLabel, usize>::from_file(
            &path,
            Some(format!("liu_vs_bz_{run_tag}")),
            None,
        )
        .expect("graph parse should succeed");
        let bz_cores = crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik::new(&graph)
            .expect("batagelj-zaversnik should succeed")
            .k_cores()
            .to_vec();
        assert_eq!(
            usize_cores, bz_cores,
            "Liu et al. and Batagelj-Zaversnik should agree on the canonical k-core decomposition"
        );
    }
}
