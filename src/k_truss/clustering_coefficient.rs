use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicF64, AtomicU8, AtomicUsize, Ordering};
use std::{
    collections::HashSet,
    sync::{Arc, Barrier},
};

use super::triangles::Triangles;

type ProceduralMemoryClusteringCoefficient = ();

/// For the computation of a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering coefficient.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct ClusteringCoefficient<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which local clustering coefficient, transitivity and average local clustering coefficient.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's local clustering coefficient.
    local: AbstractedProceduralMemoryMut<f64>,
    /// Global clustering coefficient --- graph transitivity.
    transitivity: f64,
    /// Overall level of clustering in a network is measured by Watts and Strogatz as the average of the local clustering coefficients of all the nodes.
    local_average: f64,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ClusteringCoefficient<'a, N, E, Ix> {
    /// Computes a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering
    /// coefficient.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance for which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::ClusteringCoefficient, None)?;
        let local = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut clustering_coefficient = Self {
            g,
            local,
            transitivity: 0.,
            local_average: 0.,
        };
        clustering_coefficient.compute()?;
        Ok(clustering_coefficient)
    }

    /// Getter for the average of clustering coefficients of all nodes.
    pub fn get_average_clustering_coefficient(&self) -> f64 {
        self.local_average
    }

    /// Getter for individual node's clustering coefficient.
    pub fn get_node_clusteringcoefficient(&self, id: usize) -> f64 {
        assert!(id < self.g.size(), "id < |V| --- not met");
        *self.local.get(id)
    }

    /// Getter for graph's global clustering coefficient --- graph transitivity.
    pub fn get_graph_transitivity(&self) -> f64 {
        self.transitivity
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    fn init_procedural_memory_(
        &self,
    ) -> Result<ProceduralMemoryClusteringCoefficient, Box<dyn std::error::Error>> {
        Ok(())
    }

    /// Computes a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering
    /// coefficient.
    ///
    /// # Arguments
    ///
    /// * `mmap` --- the level of memmapping to be used during the computation (*experimental feature*).
    ///
    pub fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let threads = self.g.thread_num();
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let mut triangles = Triangles::new(self.g)?;
        let tris = triangles.triangles_shares_slice();

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        let total_possible_triangles = Arc::new(AtomicF64::new(0.));
        let local_clustering_sum = Arc::new(AtomicF64::new(0.));
        let total_triangles = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for tid in 0..threads {
                let mut local = self.local.shared_slice();
                let mut tris = tris.clone();

                let total_possible_triangles = total_possible_triangles.clone();
                let total_triangles = total_triangles.clone();
                let local_clustering_sum = local_clustering_sum.clone();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                scope.spawn(move |_| {
                    let mut total_triangles_t = 0;
                    // initialize triangle_count with zeroes
                    let edge_begin = index_ptr.get(begin).index();
                    let edge_end = index_ptr.get(end).index();
                    for idx in edge_begin..edge_end {
                        *tris.get_mut(idx) = AtomicU8::new(0);
                    }

                    synchronize.wait();

                    let mut all_neighbours = HashSet::<usize>::new();
                    for u in begin..end {
                        for j in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                            let v = neighbours_ptr.get(j).index();
                            all_neighbours.insert(v);
                            total_triangles_t += tris.get(j).load(Ordering::Relaxed);
                        }
                        let n_len = all_neighbours.len();
                        // #possible edges between neighbours = k * (k - 1) / 2
                        let n_possible_triangles = if n_len > 1 {
                            n_len as f64 * (n_len - 1) as f64 / 2.
                        } else {
                            0.
                        };
                        total_possible_triangles.fetch_add(n_possible_triangles, Ordering::Relaxed);
                        *local.get_mut(u) = n_possible_triangles;
                        all_neighbours.clear();
                    }

                    total_triangles.add((total_triangles_t / 3) as usize, Ordering::Relaxed);
                    synchronize.wait();

                    for u in begin..end {
                        let mut u_triangles: usize = 0;
                        for edge_idx in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                            u_triangles += tris.get(edge_idx).load(Ordering::Relaxed) as usize;
                        }
                        let undirected_triangles = u_triangles.div_ceil(2) as f64;
                        let possible_triangles = *local.get(u);
                        let local_conductivity = if possible_triangles.is_normal() {
                            undirected_triangles / possible_triangles
                        } else {
                            0.
                        };
                        *local.get_mut(u) = local_conductivity;
                        local_clustering_sum.fetch_add(local_conductivity, Ordering::Relaxed);
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        self.local_average = local_clustering_sum.load(Ordering::Relaxed) / node_count as f64;
        self.transitivity = total_triangles.load(Ordering::Relaxed) as f64
            / 2.
            / total_possible_triangles.load(Ordering::Relaxed);

        // cleanup cache
        triangles.drop_cache()?;
        self.g.cleanup_cache(CacheFile::ClusteringCoefficient)?;

        Ok(())
    }
}
