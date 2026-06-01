use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use num_cpus::get_physical;
use portable_atomic::Ordering;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;

use crate::utils::OneOrMany;

use super::triangles::Triangles;

type ProceduralMemoryBurkhardtEtAl<Ix> = (
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<(Ix, Ix)>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoBurkhardtEtAl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which edge trussness is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the trussness of each edge.
    k_trusses: AbstractedProceduralMemoryMut<u8>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoBurkhardtEtAl<'a, N, E, Ix> {
    /// Performs k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut burkhardt_et_al = Self::new_no_compute(g)?;
        let proc_mem = burkhardt_et_al.init_cache_mem()?;

        burkhardt_et_al.compute_with_proc_mem(proc_mem)?;

        Ok(burkhardt_et_al)
    }

    /// Searches for a previously cached result, and if not found performs k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::KTrussBEA, None)?;
        if Path::new(&t_fn).exists() {
            if let Ok(k_trusses) = AbstractedProceduralMemoryMut::from_file_name(&t_fn) {
                return Ok(Self { g, k_trusses });
            }
        }
        Self::new(g)
    }

    /// Returns the trussness of a given edge of a [`GraphMemoryMap`] instance.
    ///
    /// # Arguments
    ///
    /// * `e_idx` --- the index of the edge whose trussness is to be returned.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn trussness(&self, e_idx: usize) -> u8 {
        assert!(e_idx < self.g.width());
        *self.k_trusses.get(e_idx)
    }

    /// Returns a slice containing the trussness of each edge of the [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn k_trusses(&self) -> &[u8] {
        self.k_trusses.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::KTrussBEA, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoBurkhardtEtAl<'a, N, E, Ix> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::KTrussBEA, None)?;
        let k_trusses = SharedSliceMut::<u8>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self { g, k_trusses })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        let edge_count = self.g.width();
        let edge_count2 = self.g.width() * 2;

        let el_fn = self.g.build_cache_filename(CacheFile::KTrussBEA, Some(0))?;
        let ei_fn = self.g.build_cache_filename(CacheFile::KTrussBEA, Some(1))?;
        let s_fn = self.g.build_cache_filename(CacheFile::KTrussBEA, Some(2))?;

        let edge_list = SharedSliceMut::<Ix>::abst_mem_mut(&el_fn, edge_count, true)?;
        let edge_index = SharedSliceMut::<Ix>::abst_mem_mut(&ei_fn, edge_count, true)?;
        let stack = SharedSliceMut::<(Ix, Ix)>::abst_mem_mut(&s_fn, edge_count2, true)?;

        // pre-initialize the memmapped files if they don't exist
        let _edge_reciprocal = self.g.edge_reciprocal()?;
        let _edge_out = self.g.edge_over()?;

        Ok((edge_list, edge_index, stack))
    }

    fn compute_with_proc_mem_impl(
        &self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let edge_count = self.g.width();

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let (edges, edge_index, e_stack) = proc_mem;
        let mut trussness = self.k_trusses.shared_slice();

        let edge_reciprocal = self.g.edge_reciprocal()?;

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.g.thread_num().max(get_physical() * 2);
            let edge_load = edge_count.div_ceil(threads);

            for tid in 0..threads {
                let mut edges = edges.shared_slice();
                let mut edge_index = edge_index.shared_slice();

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);

                scope.spawn(move |_| {
                    for idx in begin..end {
                        // minimum possible trussness
                        *trussness.get_mut(idx) = 2;
                        *edges.get_mut(idx) = Ix::new(idx);
                        *edge_index.get_mut(idx) = Ix::new(idx);
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // Algorithm 1 - adjusted for directed scheme, computed by Triangles
        let mut triangles = Triangles::new(self.g)?;
        let tris = triangles.triangles_shares_slice();
        let mut stack = SharedQueueMut::<(Ix, Ix)>::from_shared_slice(e_stack.shared_slice());

        // Algorithm 2 - sentinel value is 0
        // blank (u64, usize) value
        let mut edges = edges.shared_slice();
        let mut edge_index = edge_index.shared_slice();
        let mut edge_count = edge_count;
        let er = edge_reciprocal.shared_slice();
        let mut test = vec![0usize; u8::MAX as usize];

        for k in 1..u8::MAX {
            if edge_count == 0 {
                break;
            }
            let mut idx = 0;
            while idx < edge_count {
                let edge_offset = edge_index.get(idx).index();
                let t_count = tris.get(edge_offset).load(Ordering::Relaxed);
                if t_count == k {
                    let u = neighbours_ptr.get(er.get(edge_offset).index()).index();
                    stack.push((Ix::new(u), Ix::new(edge_offset)));
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                } else if t_count == 0 {
                    edge_count = match edge_count.overflowing_sub(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!(
                                "error overflow add to edge_count ({edge_count} + 1)",
                            )
                            .into());
                        }
                    };

                    let e_index = edge_index.get(edge_count).index();
                    let r_index = edges.get(edge_offset).index();
                    *edge_index.get_mut(edge_count) = *edge_index.get(r_index);
                    *edge_index.get_mut(r_index) = Ix::new(e_index);
                    *edges.get_mut(edge_offset) = Ix::new(edge_count);
                    *edges.get_mut(e_index) = Ix::new(r_index);
                    // store edge trussness
                    *trussness.get_mut(edge_offset) = k + 1;
                    if k == 1 {
                        test[2] += 1;
                    }
                    continue;
                } else {
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                }
            }

            let mut neighbours = HashMap::<usize, OneOrMany<usize>>::new();

            while let Some((u, offset)) = stack.pop() {
                let u = u.index();
                let offset = offset.index();
                tris.get(offset).store(0, Ordering::Relaxed);
                // by definition any non-trivial subgraph is at least a 2-truss: other
                // values go up from there.
                test[k as usize + 2] += 1;
                let v = neighbours_ptr.get(offset).index();

                // build u's neighour map
                for u_w in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                    let w = neighbours_ptr.get(u_w).index();
                    if w != u && w != v {
                        match neighbours.entry(w) {
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(OneOrMany::One(u_w));
                            }
                            std::collections::hash_map::Entry::Occupied(mut e) => {
                                match e.get_mut() {
                                    OneOrMany::One(prev_u_w) => {
                                        let mut u_ws: SmallVec<[usize; 4]> = SmallVec::new();
                                        u_ws.push(*prev_u_w);
                                        u_ws.push(u_w);
                                        *e.get_mut() = OneOrMany::Many(u_ws);
                                    }
                                    OneOrMany::Many(u_ws) => u_ws.push(u_w),
                                }
                            }
                        }
                    }
                }

                // search v for neighbours in u's neighour map
                for v_w in index_ptr.get(v).index()..index_ptr.get(v + 1).index() {
                    let w = neighbours_ptr.get(v_w).index();
                    match neighbours.get(&w) {
                        Some(i) => match i {
                            OneOrMany::One(u_w) => {
                                let w_u = er.get(*u_w).index();
                                if tris.get(v_w).load(Ordering::Relaxed) != 0
                                    && tris.get(w_u).load(Ordering::Relaxed) != 0
                                {
                                    if tris.get(w_u).fetch_sub(1, Ordering::Relaxed) == k + 1 {
                                        stack.push((Ix::new(w), Ix::new(w_u)));
                                    }
                                    if tris.get(v_w).fetch_sub(1, Ordering::Relaxed) == k + 1 {
                                        stack.push((Ix::new(v), Ix::new(v_w)));
                                    }
                                }
                            }
                            OneOrMany::Many(u_ws) => {
                                for u_w in u_ws {
                                    let w_u = er.get(*u_w).index();
                                    if tris.get(v_w).load(Ordering::Relaxed) != 0
                                        && tris.get(w_u).load(Ordering::Relaxed) != 0
                                    {
                                        if tris.get(w_u).fetch_sub(1, Ordering::Relaxed) == k + 1 {
                                            stack.push((Ix::new(w), Ix::new(w_u)));
                                        }
                                        if tris.get(v_w).fetch_sub(1, Ordering::Relaxed) == k + 1 {
                                            stack.push((Ix::new(v), Ix::new(v_w)));
                                        }
                                    }
                                }
                            }
                        },
                        None => continue,
                    };
                }
                neighbours.clear();
            }
        }
        let env_verbose_val = std::env::var("BRUIJNX_VERBOSE").unwrap_or_else(|_| "0".to_string());
        let verbose: bool = env_verbose_val == "1";
        if verbose {
            let mut max = 0;
            test.iter().enumerate().for_each(|(i, v)| {
                if *v != 0 && i > max {
                    max = i;
                }
            });
            test.resize(max + 1, 0);
            println!("k-trussness {:?}", test);
        }
        self.k_trusses.flush_async()?;

        triangles.drop_cache()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::KTrussBEA)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{k_truss::verify_k_trusses, test_common::get_or_init_dataset_cache_entry};

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<k_trusses_burkhardt_et_al_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
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
        let burkhardt_et_al_k_trusses = AlgoBurkhardtEtAl::new(&graph)?;

        verify_k_trusses(&graph, burkhardt_et_al_k_trusses.k_trusses)
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => "../ggcat/graphs/random_graph_1_5.lz4",
        ggcat_2_5 => "../ggcat/graphs/random_graph_2_5.lz4",
        ggcat_3_5 => "../ggcat/graphs/random_graph_3_5.lz4",
        ggcat_4_5 => "../ggcat/graphs/random_graph_4_5.lz4",
        // ggcat_5_5 => "../ggcat/graphs/random_graph_5_5.lz4",
        // ggcat_6_5 => "../ggcat/graphs/random_graph_6_5.lz4",
        // ggcat_7_5 => "../ggcat/graphs/random_graph_7_5.lz4",
        // ggcat_8_5 => "../ggcat/graphs/random_graph_8_5.lz4",
        // ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        // ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        // ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        // ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        // ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        // ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        // ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        // ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        // ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        // ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
