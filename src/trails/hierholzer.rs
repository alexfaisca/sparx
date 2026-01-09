use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::mem::ManuallyDrop;
use std::path::Path;

type ProceduralMemoryHierholzers = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<u8>,
);

type AllEulerTrailsConcatenatedWithBounds<'a> = (&'a [usize], Box<[(usize, usize)]>);

/// For the computation of the euler trail(s) of a [`GraphMemoryMap`] instance using *Hierholzer's Algorithm*.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct AlgoHierholzer<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which the euler trail(s) is(are) computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the euler trails.
    euler_trails: AbstractedProceduralMemoryMut<usize>,
    /// Memmapped slice containing the starting position of each euler trail.
    euler_indices: AbstractedProceduralMemoryMut<usize>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoHierholzer<'a, N, E, Ix> {
    /// Performs graph traversal using *Hierholzer's Algorithm*, computing the euler trails of a [`GraphMemoryMap`] instance.
    ///
    /// The resulting Euler trail(s) is(are) stored in memory (in a memmapped file) nodewise[^1].
    ///
    /// * Note *1*: by definition, isolated nodes won't be present in the euler trail unless they have *self-loops*[^2].
    /// * Note *2*: in the case of a non-eulerian graph, trails are stored sequentially in memory, with their respective starting indexes stored in the aboce mentioned `trail_index` array.
    /// * Note *3*: the last edge of each euler trail, connecting the trail into an euler cycle is intentionally left out of the resulting euler trail(s).
    ///
    /// [^1]: for each two consecutive node entries, u and v, the(an) edge (u,v) between them is traversed.
    /// [^2]: edges of the type (u, u), for a given node u.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which *Hierholzer's Algorithm* is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut euler = Self::new_no_compute(g)?;
        let proc_mem = euler.init_cache_mem()?;

        euler.compute_with_proc_mem(proc_mem)?;

        Ok(euler)
    }

    /// Searches for a previously cached result, and if not found performs graph traversal using *Hierholzer's Algorithm*, computing the euler trails of a [`GraphMemoryMap`] instance.
    ///
    /// The resulting Euler trail(s) is(are) stored in memory (in a memmapped file) nodewise[^1].
    ///
    /// * Note *1*: by definition, isolated nodes won't be present in the euler trail unless they have *self-loops*[^2].
    /// * Note *2*: in the case of a non-eulerian graph, trails are stored sequentially in memory, with their respective starting indexes stored in the aboce mentioned `trail_index` array.
    /// * Note *3*: the last edge of each euler trail, connecting the trail into an euler cycle is intentionally left out of the resulting euler trail(s).
    ///
    /// [^1]: for each two consecutive node entries, u and v, the(an) edge (u,v) between them is traversed.
    /// [^2]: edges of the type (u, u), for a given node u.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which *Hierholzer's Algorithm* is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::EulerTrail, None)?;
        if Path::new(&t_fn).exists() {
            if let Ok(euler_trails) = AbstractedProceduralMemoryMut::from_file_name(&t_fn) {
                let i_fn = g.build_cache_filename(CacheFile::EulerIndex, None)?;
                if Path::new(&i_fn).exists() {
                    if let Ok(euler_indices) = AbstractedProceduralMemoryMut::from_file_name(&i_fn)
                    {
                        return Ok(Self {
                            g,
                            euler_trails,
                            euler_indices,
                        });
                    }
                }
            }
        }
        Self::new(g)
    }

    /// Returns the number of euler trails found upon performing *Hierholzer's* graph traversal algorithm.
    pub fn trail_number(&self) -> usize {
        self.euler_indices.len()
    }

    /// Returns the number of (strongly[^1]) connected components on the [`GraphMemoryMap`] instance.
    ///
    /// [^1]: Given the topologic properties of de Bruijn graphs, the number of strongly connected
    /// components always equals the number of weakly connected components.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn count_connected_components(&self) -> usize {
        self.trail_number()
    }

    /// Returns the euler trail corresponding to a given `idx` index for the [`GraphMemoryMap`] instance.
    ///
    /// # Arguments
    ///
    /// * `idx` --- the index of the euler trail which is to be returned.
    ///
    /// * Returns
    ///
    /// This method will return [`Some`] if `idx` is smaller than the number of euler trails found upon performing *Hierholzer's* graph traversal algorithm and [`None`] otherwise, i.e. iff `idx < trail_index.len()`.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_trail(&self, idx: usize) -> Option<&[usize]> {
        if idx < self.euler_indices.len() {
            // get trail start index
            let start = match idx.overflowing_sub(1) {
                (_, true) => 0,
                (i, false) => *self.euler_indices.get(i),
            };
            // get trail end index
            let end = *self.euler_indices.get(idx);
            // slice concatenated trails based on start & end indexes
            self.euler_trails.slice(start, end)
        } else {
            None
        }
    }

    /// Returns all euler trails of the [`GraphMemoryMap`] instance.
    ///
    /// The euler trails are concatenated and a slice of bounds[^1] is provided with indexing
    /// purposes.
    ///
    /// [^1]: bounds are given as a tuple (start_idx: [`usize`], end_idx: [`usize`]).
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_all_trails(&'a self) -> AllEulerTrailsConcatenatedWithBounds<'a> {
        let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(self.euler_indices.len());
        self.euler_indices
            .as_slice()
            .iter()
            .enumerate()
            .for_each(|(idx, &end_pos)| {
                let start = match idx.overflowing_sub(1) {
                    (_, true) => 0,
                    (i, false) => *self.euler_indices.get(i),
                };
                bounds.push((start, end_pos));
            });
        (self.euler_trails.as_slice(), bounds.into_boxed_slice())
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let t_fn = this.g.build_cache_filename(CacheFile::EulerTrail, None)?;
        let i_fn = this.g.build_cache_filename(CacheFile::EulerIndex, None)?;
        this.g.cleanup_cache(CacheFile::EulerTrail)?;
        std::fs::remove_file(t_fn)?;
        std::fs::remove_file(i_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoHierholzer<'a, N, E, Ix> {
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
    ) -> Result<ProceduralMemoryHierholzers, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryHierholzers, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryHierholzers,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryHierholzers,
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
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let i_fn = g.build_cache_filename(CacheFile::EulerIndex, None)?;
        let t_fn = g.build_cache_filename(CacheFile::EulerTrail, None)?;
        Ok(Self {
            g,
            euler_trails: SharedSliceMut::<usize>::abst_mem_mut(&t_fn, g.width(), true)?,
            euler_indices: SharedSliceMut::<usize>::abst_mem_mut(&i_fn, 1, true)?,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<ProceduralMemoryHierholzers, Box<dyn std::error::Error>> {
        let node_count = self.g.size();

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());

        let e_fn = self.build_cache_filename(CacheFile::EulerTrail, Some(0))?;
        let c_fn = self.build_cache_filename(CacheFile::EulerTrail, Some(1))?;

        let edges = SharedSliceMut::<usize>::abst_mem_mut(&e_fn, node_count, true)?;
        let count = SharedSliceMut::<u8>::abst_mem_mut(&c_fn, node_count, true)?;

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.g.thread_num().max(get_physical() * 2);
            let node_load = node_count.div_ceil(threads);

            for i in 0..threads {
                let mut edges = edges.shared_slice();
                let mut count = count.shared_slice();
                let begin = std::cmp::min(i * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                scope.spawn(move |_| {
                    for k in begin..end {
                        *edges.get_mut(k) = index_ptr.get(k).index();
                        *count.get_mut(k) = (index_ptr.get(k + 1).index() - index_ptr.get(k).index()) as u8;
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        edges.flush()?;
        count.flush()?;

        Ok((edges, count))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryHierholzers,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = match self.g.size() {
            0 => return Ok(()),
            i => i,
        };
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width());

        let (mut edges, mut edge_count) = proc_mem;

        let mut start_vertex_counter = 0usize;
        let mut cycles = self.euler_trails.shared_slice();
        let mut write_idx = 0usize;
        let mut indices = Vec::<usize>::with_capacity(128);
        // find cycles until no unused edges remain
        loop {
            let start_v = loop {
                let idx = if start_vertex_counter >= node_count {
                    break None;
                } else {
                    start_vertex_counter
                };
                if idx >= node_count {
                    break None;
                }
                if *edge_count.get(idx) > 0 {
                    break Some(idx);
                }
                start_vertex_counter += 1;
            };
            let start_v = match start_v {
                Some(v) => v,
                None => {
                    break;
                }
            };

            // Hierholzer's DFS
            let mut stack: Vec<usize> = Vec::with_capacity(node_count);
            let mut cycle: Vec<usize> = Vec::with_capacity(node_count);

            stack.push(start_v);
            while let Some(&v) = stack.last() {
                if *edge_count.get(v) > 0 {
                    *edge_count.get_mut(v) -= 1;
                    stack.push(neighbours_ptr.get(*edges.get(v)).index());
                    *edges.get_mut(v) += 1;
                } else {
                    stack.pop();
                    cycle.push(v);
                }
            }
            if !cycle.is_empty() {
                cycle.reverse();

                let cycle_slice = cycle.as_slice();
                let end = cycle_slice.len() - 1;

                write_idx = cycles
                    .write_slice(write_idx, &cycle_slice[..end])
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error couldn't slice mmap to write cycle".into()
                    })?;

                indices.push(write_idx);
                cycle.clear();
            }
        }

        if indices.len() == 1 {
            *self.euler_indices.get_mut(0) = indices[0];
        } else {
            let i_fn = self.build_cache_filename(CacheFile::EulerIndex, None)?;
            self.euler_indices = SharedSliceMut::abst_mem_mut(&i_fn, indices.len(), true)?;
            indices.iter().enumerate().for_each(|(idx, &val)| {
                *self.euler_indices.get_mut(idx) = val;
            });
        }

        self.euler_trails.flush()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::EulerTrail)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::test_common::get_or_init_dataset_cache_entry;
    use crate::trails::verify_trails;

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<hierholzers_euler_trails_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
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
        let het = AlgoHierholzer::new(&graph)?;

        verify_trails(&graph, het.euler_trails, het.euler_indices)
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
        // ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        // ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        // ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        // ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        // ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
