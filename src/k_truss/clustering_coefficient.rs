use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicF64, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Barrier};

use super::count::{TriangleCount, TriangleCounter};
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

    /// Computes a [`GraphMemoryMap`] instance's clustering coefficients from precomputed
    /// `triangles` of `g` (e.g. counted on a GPU, with [`Triangles::new_gpu`]), which are consumed.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    /// [`Triangles::new_gpu`]: ../triangles/struct.Triangles.html#method.new_gpu
    pub fn new_with_triangles<Tx: TriangleCount>(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::ClusteringCoefficient, None)?;
        let local = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut clustering_coefficient = Self {
            g,
            local,
            transitivity: 0.,
            local_average: 0.,
        };
        clustering_coefficient.compute_with_triangles(triangles)?;
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

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::ClusteringCoefficient, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
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
        self.compute_with_triangles(Triangles::new(self.g)?)
    }

    /// Computes the clustering coefficients from the graph's `triangles` (which are consumed).
    fn compute_with_triangles<Tx: TriangleCount>(
        &mut self,
        mut triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if triangles.all_triangles().len() != self.g.width() {
            return Err(format!(
                "error expected the triangles of a graph with {} edges, got {}",
                self.g.width(),
                triangles.all_triangles().len()
            )
            .into());
        }

        let node_count = self.g.size();
        let edge_count = self.g.width();

        let threads = self.g.thread_num();
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let tris = triangles.triangles_shares_slice();

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        let total_possible_triangles = Arc::new(AtomicF64::new(0.));
        let local_clustering_sum = Arc::new(AtomicF64::new(0.));
        let total_triangles = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for tid in 0..threads {
                let mut local = self.local.shared_slice();
                let tris = tris.clone();

                let total_possible_triangles = total_possible_triangles.clone();
                let total_triangles = total_triangles.clone();
                let local_clustering_sum = local_clustering_sum.clone();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                scope.spawn(move |_| {
                    // sum of the thread's edges' triangle counts
                    let mut total_triangles_t: usize = 0;

                    for u in begin..end {
                        // distinct neighbours: lists are sorted, so parallel edges are adjacent
                        let mut n_len = 0;
                        let mut prev = None;
                        for j in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                            let v = neighbours_ptr.get(j).index();
                            // a self-loop doesn't make `u` its own neighbour
                            if v != u && prev != Some(v) {
                                n_len += 1;
                            }
                            prev = Some(v);
                            total_triangles_t += tris.get(j).count();
                        }
                        // #possible edges between neighbours = k * (k - 1) / 2
                        let n_possible_triangles = if n_len > 1 {
                            n_len as f64 * (n_len - 1) as f64 / 2.
                        } else {
                            0.
                        };
                        total_possible_triangles.fetch_add(n_possible_triangles, Ordering::Relaxed);
                        *local.get_mut(u) = n_possible_triangles;
                    }

                    total_triangles.add(total_triangles_t, Ordering::Relaxed);
                    synchronize.wait();

                    for u in begin..end {
                        let mut u_triangles: usize = 0;
                        for edge_idx in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                            u_triangles += tris.get(edge_idx).count();
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
        // every triangle counts once for each of its 6 (directed) edges, so the sum of all
        // edges' counts is 6 * triangles, and transitivity = 3 * triangles / connected triples
        // (0 without connected triples, as networkx's `transitivity`)
        let triples = total_possible_triangles.load(Ordering::Relaxed);
        self.transitivity = if triples > 0. {
            total_triangles.load(Ordering::Relaxed) as f64 / 2. / triples
        } else {
            0.
        };

        // cleanup cache
        triangles.drop_cache()?;

        Ok(())
    }
}

/// For the computation of a directed [`GraphMemoryMap`] instance's local clustering
/// coefficients and their average, as defined by Fagiolo in ["Clustering in complex directed
/// networks"](https://doi.org/10.1103/PhysRevE.76.026107) (and computed by networkx's
/// `clustering` on directed graphs):
///
/// `C(i) = t(i) / (d(i) (d(i) - 1) - 2 d↔(i))`
///
/// where `t(i)` counts the directed triangles through `i`, of any orientation, `d(i)` is `i`'s
/// in- plus out-degree and `d↔(i)` the number of its reciprocated arcs (with multiplicity: the
/// product of the arcs each way). Self-loops are ignored. On undirected graphs it's the usual
/// clustering coefficient ([`ClusteringCoefficient`]).
///
/// Built on the counts of [`DirectedTriangles`] of every type
/// ([`DirectedTriangleType::CycleFeedForward`]): an arc `u -> v`'s count is
/// `Σ_w (a(u,w) + a(w,u)) (a(v,w) + a(w,v))`, and summing it over a node's arcs, in and out,
/// gives `2 t(i)`.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
/// [`DirectedTriangles`]: ../directed/struct.DirectedTriangles.html
/// [`DirectedTriangleType::CycleFeedForward`]: ../directed/enum.DirectedTriangleType.html#variant.CycleFeedForward
#[allow(dead_code)]
#[derive(Debug)]
pub struct DirectedClusteringCoefficient<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which the clustering coefficients are computed.
    g: &'a DiGraph<N, E, Ix>,
    /// Memmapped slice containing each node's local clustering coefficient.
    local: AbstractedProceduralMemoryMut<f64>,
    /// Average of the local clustering coefficients.
    local_average: f64,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedClusteringCoefficient<'a, N, E, Ix> {
    /// Computes the directed clustering coefficients of a [`GraphMemoryMap`] instance,
    /// counting triangles as `u32`s (see [`new_tx`](Self::new_tx) for other widths).
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a DiGraph<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx::<u32>(g)
    }

    /// Computes the directed clustering coefficients of a [`GraphMemoryMap`] instance,
    /// counting each arc's triangles as `Tx` integers.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_tx<Tx: TriangleCount>(
        g: &'a DiGraph<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use super::directed::{DirectedTriangleType, DirectedTriangles};

        // what can fail first, so that failing leaves no result file behind
        let transpose = g.transpose()?;
        let mut triangles =
            DirectedTriangles::<_, _, _, Tx>::new_tx(g, DirectedTriangleType::CycleFeedForward)?;
        let out_fn = g.build_cache_filename(CacheFile::DirectedClusteringCoefficient, None)?;
        let mut local = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, g.size(), true)?;

        let node_count = g.size();
        let index_ptr = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(g.neighbours_ptr(), g.width());

        // 2 t(i): every arc's triangles, at both of its ends
        let t = local.as_mut_slice();
        let counts = triangles.all_triangles();
        for u in 0..node_count {
            for e in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                let v = neighbours_ptr.get(e).index();
                let c = counts[e].count() as f64;
                t[u] += c;
                t[v] += c;
            }
        }

        let mut sum = 0f64;
        for (i, t_i) in t.iter_mut().enumerate() {
            let outs: Vec<usize> = g.neighbours(i)?.filter(|&j| j != i).collect();
            let ins: Vec<usize> = transpose.neighbours(i)?.filter(|&j| j != i).collect();
            let degree = (outs.len() + ins.len()) as f64;
            // d↔(i): the product of the runs of each neighbour in both sorted lists
            let mut reciprocated = 0usize;
            let (mut a, mut b) = (0, 0);
            while a < outs.len() && b < ins.len() {
                match outs[a].cmp(&ins[b]) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        let j = outs[a];
                        let run_out = outs[a..].iter().take_while(|&&x| x == j).count();
                        let run_in = ins[b..].iter().take_while(|&&x| x == j).count();
                        reciprocated += run_out * run_in;
                        a += run_out;
                        b += run_in;
                    }
                }
            }
            let possible = degree * (degree - 1.) - 2. * reciprocated as f64;
            *t_i = if possible > 0. { *t_i / 2. / possible } else { 0. };
            sum += *t_i;
        }
        triangles.drop_cache()?;
        local.flush()?;

        Ok(Self {
            g,
            local,
            local_average: if node_count > 0 { sum / node_count as f64 } else { 0. },
        })
    }

    /// Returns the average of the local clustering coefficients (nodes without triangles
    /// included, as zeros).
    pub fn get_average_clustering_coefficient(&self) -> f64 {
        self.local_average
    }

    /// Returns a node's local clustering coefficient.
    pub fn get_node_clustering_coefficient(&self, id: usize) -> f64 {
        assert!(id < self.g.size());
        *self.local.get(id)
    }

    /// Returns a slice containing each node's local clustering coefficient.
    pub fn local_clustering_coefficients(&self) -> &[f64] {
        self.local.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::DirectedClusteringCoefficient, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[cfg(test)]
mod directed_test {
    use super::*;
    use crate::components::test_utils::{graph, random_arcs};

    /// Fagiolo's formula, with adjacency matrices (`a[i][j]`: arcs `i -> j`, self-loops ignored).
    fn brute_force(n: usize, arcs: &[(usize, usize)]) -> Vec<f64> {
        let mut a = vec![vec![0f64; n]; n];
        for &(u, v) in arcs {
            if u != v {
                a[u][v] += 1.;
            }
        }
        let s = |i: usize, j: usize| a[i][j] + a[j][i];
        (0..n)
            .map(|i| {
                let mut t = 0.;
                for j in 0..n {
                    for h in 0..n {
                        t += s(i, j) * s(i, h) * s(j, h);
                    }
                }
                t /= 2.;
                let degree: f64 = (0..n).map(|j| s(i, j)).sum();
                let reciprocated: f64 = (0..n).map(|j| a[i][j] * a[j][i]).sum();
                let possible = degree * (degree - 1.) - 2. * reciprocated;
                if possible > 0. { t / possible } else { 0. }
            })
            .collect()
    }

    #[test]
    fn directed_clustering_coefficients_match_brute_force() {
        for (name, n, arcs) in [
            // networkx: clustering(nx.DiGraph([(0,1),(0,2),(1,2)])) == {0: 1, 1: 1, 2: 1}
            ("feed_forward", 3, vec![(0, 1), (0, 2), (1, 2)]),
            ("cycle_with_reciprocal", 4, vec![(0, 1), (1, 2), (2, 0), (1, 0), (2, 3)]),
            ("multi_and_loops", 5, vec![(0, 1), (0, 1), (1, 2), (2, 0), (3, 3), (2, 4), (4, 0)]),
            ("random", 30, random_arcs(30, 160, 9)),
        ] {
            let g = graph(name, "general", n, &arcs);
            let cc = DirectedClusteringCoefficient::new(&g).unwrap();
            let want = brute_force(n, &arcs);
            for (i, (got, want)) in cc.local_clustering_coefficients().iter().zip(&want).enumerate() {
                assert!((got - want).abs() < 1e-12, "{name}: node {i}: {got} != {want}");
            }
            let average = want.iter().sum::<f64>() / n as f64;
            assert!((cc.get_average_clustering_coefficient() - average).abs() < 1e-12, "{name}");
        }
    }

    #[test]
    fn directed_clustering_coefficients_of_undirected_graphs_are_the_usual_ones() {
        let edges = random_arcs(30, 120, 4).into_iter().filter(|(u, v)| u < v).collect::<Vec<_>>();
        let mut simple = edges.clone();
        simple.sort();
        simple.dedup();
        let g = graph("undirected", "symmetric", 30, &simple);
        let directed = DirectedClusteringCoefficient::new(&g).unwrap();
        let undirected = ClusteringCoefficient::new(g.to_undirected().unwrap()).unwrap();
        for i in 0..30 {
            let (d, u) = (directed.get_node_clustering_coefficient(i), undirected.get_node_clusteringcoefficient(i));
            assert!((d - u).abs() < 1e-12, "node {i}: {d} != {u}");
        }
    }

    /// Without connected triples, transitivity is 0 (as networkx's), not 0 / 0.
    #[test]
    fn transitivity_without_triples_is_zero() {
        let g = graph("one_edge", "symmetric", 3, &[(2, 1)]);
        let cc = ClusteringCoefficient::new(g.to_undirected().unwrap()).unwrap();
        assert_eq!(cc.get_graph_transitivity(), 0.);
        assert_eq!(cc.get_average_clustering_coefficient(), 0.);
    }
}
