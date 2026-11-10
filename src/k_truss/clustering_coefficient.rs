use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Barrier};

use super::count::{TriangleCount, TriangleCounter};
use super::triangles::Triangles;

type ProceduralMemoryClusteringCoefficient = ();

/// A node's neighbours other than itself, sorted, with the weight of its arcs to each (parallel
/// arcs' added up).
fn merged_row<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
    u: usize,
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    let mut row: Vec<(usize, f64)> = Vec::new();
    for (v, label) in g.weighted_neighbours(u)? {
        let w = label.weight::<f64>();
        match row.last_mut() {
            Some((last, total)) if *last == v => *total += w,
            _ if v == u => {}
            _ => row.push((v, w)),
        }
    }
    Ok(row)
}

/// The largest weight between two nodes (parallel arcs' added up, self-loops included): what
/// networkx normalizes weights by. Weights are checked finite and non-negative.
fn max_merged_weight<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Result<f64, Box<dyn std::error::Error>> {
    g.total_weight()?;
    let mut max = 0f64;
    for u in 0..g.size() {
        let (mut last, mut total) = (None, 0f64);
        for (v, label) in g.weighted_neighbours(u)? {
            if last != Some(v) {
                max = max.max(total);
                (last, total) = (Some(v), 0.);
            }
            total += label.weight::<f64>();
        }
        max = max.max(total);
    }
    Ok(max)
}

/// `f(u)` for every node, in parallel, in node order.
fn per_node(
    threads: usize,
    n: usize,
    f: &(dyn Fn(usize) -> Result<f64, Box<dyn std::error::Error>> + Sync),
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let threads = threads.clamp(1, n.max(1));
    let load = n.div_ceil(threads);
    let parts: Vec<Result<Vec<f64>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    ((t * load).min(n)..((t + 1) * load).min(n))
                        .map(|u| f(u).map_err(|e| e.to_string()))
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| Err(format!("{e:?}"))))
            .collect()
    });
    let mut values = Vec::with_capacity(n);
    for part in parts {
        values.extend(part?);
    }
    Ok(values)
}

/// Each node's weighted clustering coefficient, networkx's `clustering` with `weight` (Onnela et
/// al.'s): `c(i) = 2 / (d(i) (d(i) - 1)) Σ_{j < k} (ŵ_ij ŵ_ik ŵ_jk)^(1/3)`, over the pairs of
/// `i`'s `d(i)` neighbours (self-loops ignored), weights `ŵ` divided by the largest. Parallel
/// edges weigh their weights' sum.
fn weighted_local_clustering<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &GraphMemoryMap<N, E, Ix>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let max = max_merged_weight(g)?;
    per_node(g.thread_num(), g.size(), &|i| {
        let row = merged_row(g, i)?;
        let d = row.len() as f64;
        if row.len() < 2 || max == 0. {
            return Ok(0.);
        }
        let mut sum = 0f64;
        for &(j, w_ij) in &row {
            let row_j = merged_row(g, j)?;
            // the k > j both neighbour, by merging the sorted rows
            let (mut a, mut b) = (row.partition_point(|&(k, _)| k <= j), 0);
            while a < row.len() && b < row_j.len() {
                match row[a].0.cmp(&row_j[b].0) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        sum += (w_ij / max * (row[a].1 / max) * (row_j[b].1 / max)).cbrt();
                        a += 1;
                        b += 1;
                    }
                }
            }
        }
        Ok(2. * sum / (d * (d - 1.)))
    })
}

/// A neighbour, with the weights of the arcs to and from it (`None` for none).
type BothWays = (usize, Option<f64>, Option<f64>);

/// A node's neighbours other than itself, sorted, with the weights of its arcs to and from each
/// (parallel arcs' added up; `None` for none).
fn merged_both_ways<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &DiGraph<N, E, Ix>,
    t: &DiGraph<N, E, Ix>,
    u: usize,
) -> Result<Vec<BothWays>, Box<dyn std::error::Error>> {
    let (outs, ins) = (merged_row(g, u)?, merged_row(t, u)?);
    let mut both = Vec::with_capacity(outs.len() + ins.len());
    let (mut a, mut b) = (0, 0);
    while a < outs.len() || b < ins.len() {
        let next = match (outs.get(a), ins.get(b)) {
            (Some(o), Some(i)) if o.0 == i.0 => (o.0, Some(o.1), Some(i.1)),
            (Some(o), Some(i)) if o.0 < i.0 => (o.0, Some(o.1), None),
            (Some(o), None) => (o.0, Some(o.1), None),
            (_, Some(i)) => (i.0, None, Some(i.1)),
            (None, None) => unreachable!(),
        };
        if next.1.is_some() {
            a += 1;
        }
        if next.2.is_some() {
            b += 1;
        }
        both.push(next);
    }
    Ok(both)
}

/// Each node's weighted directed clustering coefficient, networkx's `clustering` with `weight`
/// on directed graphs (Fagiolo's): `c(i) = t(i) / (2 (d(i) (d(i) - 1) - 2 d↔(i)))`, `t(i)` the
/// sum over ordered pairs of `i`'s neighbours `j ≠ k`, and over every arc between `i` and `j`,
/// `i` and `k`, `j` and `k` (any direction), of the cube root of the product of their weights
/// (divided by the largest), `d(i)` the number of `i`'s in- and out-neighbours and `d↔(i)` of
/// those both. Self-loops are ignored; parallel arcs weigh their weights' sum.
fn weighted_directed_local_clustering<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &DiGraph<N, E, Ix>,
    t: &DiGraph<N, E, Ix>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let max = max_merged_weight(g)?;
    // the weights of the arcs either way, normalized
    let arcs = |out: Option<f64>, into: Option<f64>| {
        [out, into].into_iter().flatten().map(move |w| w / max)
    };
    per_node(g.thread_num(), g.size(), &|i| {
        let row = merged_both_ways(g, t, i)?;
        let degree = row
            .iter()
            .map(|r| r.1.is_some() as usize + r.2.is_some() as usize)
            .sum::<usize>() as f64;
        let reciprocated = row
            .iter()
            .filter(|r| r.1.is_some() && r.2.is_some())
            .count() as f64;
        let possible = degree * (degree - 1.) - 2. * reciprocated;
        if possible <= 0. || max == 0. {
            return Ok(0.);
        }
        let mut sum = 0f64;
        for &(j, o_ij, i_ij) in &row {
            let row_j = merged_both_ways(g, t, j)?;
            let (mut a, mut b) = (0, 0);
            while a < row.len() && b < row_j.len() {
                match row[a].0.cmp(&row_j[b].0) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        let (k, o_ik, i_ik) = row[a];
                        let (_, o_jk, i_jk) = row_j[b];
                        if k != i {
                            for x in arcs(o_ij, i_ij) {
                                for y in arcs(o_ik, i_ik) {
                                    for z in arcs(o_jk, i_jk) {
                                        sum += (x * y * z).cbrt();
                                    }
                                }
                            }
                        }
                        a += 1;
                        b += 1;
                    }
                }
            }
        }
        Ok(sum / (2. * possible))
    })
}

/// For the computation of a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering coefficient.
///
/// With weighted edges (see [`E::is_weighted`]), local coefficients (and
/// their average) are networkx's weighted ones (Onnela et al.'s geometric means of triangles'
/// normalized weights, see `weighted_local_clustering`); transitivity, as networkx's, ignores
/// weights.
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
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::ClusteringCoefficient, None)?;
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

        // each thread's sums: of its nodes' possible triangles (connected triples) and local
        // coefficients, and of its edges' triangle counts --- added up once, in thread order (a
        // shared atomic sum would have every thread contend for it at every node, and add up in a
        // different order every run)
        let sums = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for tid in 0..threads {
                let mut local = self.local.shared_slice();
                let tris = tris.clone();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                handles.push(scope.spawn(move |_| -> (f64, f64, usize) {
                    // sum of the thread's edges' triangle counts
                    let mut total_triangles_t: usize = 0;
                    let mut possible_triangles_t = 0.;
                    let mut clustering_sum_t = 0.;

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
                        possible_triangles_t += n_possible_triangles;
                        *local.get_mut(u) = n_possible_triangles;
                    }

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
                        clustering_sum_t += local_conductivity;
                    }
                    (possible_triangles_t, clustering_sum_t, total_triangles_t)
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join())
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;
        let (mut triples, mut local_clustering_sum, mut total_triangles) = (0., 0., 0usize);
        for (possible, clustering, triangles) in sums {
            triples += possible;
            local_clustering_sum += clustering;
            total_triangles += triangles;
        }

        self.local_average = local_clustering_sum / node_count as f64;
        // every triangle counts once for each of its 6 (directed) edges, so the sum of all
        // edges' counts is 6 * triangles, and transitivity = 3 * triangles / connected triples
        // (0 without connected triples, as networkx's `transitivity`)
        self.transitivity = if triples > 0. {
            total_triangles as f64 / 2. / triples
        } else {
            0.
        };

        // weighted edges: the local coefficients are the weighted ones (transitivity, as
        // networkx's, isn't)
        if E::is_weighted() {
            let weighted = weighted_local_clustering(self.g)?;
            self.local.as_mut_slice()[..node_count].copy_from_slice(&weighted);
            self.local_average = if node_count > 0 {
                weighted.iter().sum::<f64>() / node_count as f64
            } else {
                0.
            };
        }

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
/// With weighted arcs, networkx's weighted directed coefficients instead (Fagiolo's, see
/// `weighted_directed_local_clustering`): parallel arcs weigh their weights' sum.
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
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType>
    DirectedClusteringCoefficient<'a, N, E, Ix>
{
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
            let (begin, end) = (index_ptr.get(u).index(), index_ptr.get(u + 1).index());
            for (e, count) in (begin..end).zip(&counts[begin..end]) {
                let v = neighbours_ptr.get(e).index();
                let c = count.count() as f64;
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
            *t_i = if possible > 0. {
                *t_i / 2. / possible
            } else {
                0.
            };
            sum += *t_i;
        }
        triangles.drop_cache()?;
        // weighted arcs: the weighted coefficients
        if E::is_weighted() {
            let weighted = weighted_directed_local_clustering(g, transpose)?;
            sum = weighted.iter().sum();
            local.as_mut_slice()[..node_count].copy_from_slice(&weighted);
        }
        local.flush()?;

        Ok(Self {
            g,
            local,
            local_average: if node_count > 0 {
                sum / node_count as f64
            } else {
                0.
            },
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
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::DirectedClusteringCoefficient, None)?;
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
            (
                "cycle_with_reciprocal",
                4,
                vec![(0, 1), (1, 2), (2, 0), (1, 0), (2, 3)],
            ),
            (
                "multi_and_loops",
                5,
                vec![(0, 1), (0, 1), (1, 2), (2, 0), (3, 3), (2, 4), (4, 0)],
            ),
            ("random", 30, random_arcs(30, 160, 9)),
        ] {
            let g = graph(name, "general", n, &arcs);
            let cc = DirectedClusteringCoefficient::new(&g).unwrap();
            let want = brute_force(n, &arcs);
            for (i, (got, want)) in cc
                .local_clustering_coefficients()
                .iter()
                .zip(&want)
                .enumerate()
            {
                assert!(
                    (got - want).abs() < 1e-12,
                    "{name}: node {i}: {got} != {want}"
                );
            }
            let average = want.iter().sum::<f64>() / n as f64;
            assert!(
                (cc.get_average_clustering_coefficient() - average).abs() < 1e-12,
                "{name}"
            );
        }
    }

    #[test]
    fn directed_clustering_coefficients_of_undirected_graphs_are_the_usual_ones() {
        let edges = random_arcs(30, 120, 4)
            .into_iter()
            .filter(|(u, v)| u < v)
            .collect::<Vec<_>>();
        let mut simple = edges.clone();
        simple.sort();
        simple.dedup();
        let g = graph("undirected", "symmetric", 30, &simple);
        let directed = DirectedClusteringCoefficient::new(&g).unwrap();
        let undirected = ClusteringCoefficient::new(g.to_undirected().unwrap()).unwrap();
        for i in 0..30 {
            let (d, u) = (
                directed.get_node_clustering_coefficient(i),
                undirected.get_node_clusteringcoefficient(i),
            );
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

#[cfg(test)]
mod weighted_test {
    use super::*;
    use crate::components::test_utils::random_arcs;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};

    fn weighted<Dir: Directedness>(
        name: &str,
        n: usize,
        arcs: &[(usize, usize, f64)],
    ) -> TestGraph<(), Weighted<f64>, usize, Dir> {
        let dir = unique_test_dir("clustering_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let symmetry = if Dir::DIRECTED {
            "general"
        } else {
            "symmetric"
        };
        let mut text = format!(
            "%%MatrixMarket matrix coordinate real {symmetry}\n{n} {n} {}\n",
            arcs.len()
        );
        for &(u, v, w) in arcs {
            let (u, v) = if Dir::DIRECTED {
                (u, v)
            } else {
                (u.max(v), u.min(v))
            };
            text += &format!("{} {} {w}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("clustering_w_{name}_{}", unique_run_tag());
        TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(3)).unwrap(),
            Some(dir),
        )
    }

    /// Simple (no parallel arcs, no self-loops) random arcs, weighted, zeros included.
    fn simple_weighted(
        n: usize,
        m: usize,
        seed: u64,
        undirected: bool,
    ) -> Vec<(usize, usize, f64)> {
        let mut arcs: Vec<(usize, usize)> = random_arcs(n, m, seed)
            .into_iter()
            .filter(|(u, v)| u != v)
            .map(|(u, v)| {
                if undirected {
                    (u.min(v), u.max(v))
                } else {
                    (u, v)
                }
            })
            .collect();
        arcs.sort_unstable();
        arcs.dedup();
        let mut state = seed;
        arcs.into_iter()
            .map(|(u, v)| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (u, v, ((state >> 33) % 40) as f64 / 8.0)
            })
            .collect()
    }

    /// networkx's weighted clustering, in `O(n³)`: `a` whether arcs exist, `w` their weights.
    fn brute_force(n: usize, arcs: &[(usize, usize, f64)], directed: bool) -> Vec<f64> {
        let (mut a, mut w) = (vec![vec![false; n]; n], vec![vec![0.0; n]; n]);
        for &(u, v, x) in arcs {
            a[u][v] = true;
            w[u][v] = x;
            if !directed {
                a[v][u] = true;
                w[v][u] = x;
            }
        }
        let max = arcs.iter().map(|x| x.2).fold(0.0, f64::max);
        let w = |u: usize, v: usize| w[u][v] / max;
        (0..n)
            .map(|i| {
                if !directed {
                    let nb: Vec<usize> = (0..n).filter(|&j| j != i && a[i][j]).collect();
                    let d = nb.len() as f64;
                    if nb.len() < 2 {
                        return 0.0;
                    }
                    let mut s = 0.0;
                    for (x, &j) in nb.iter().enumerate() {
                        for &k in &nb[x + 1..] {
                            if a[j][k] {
                                s += (w(i, j) * w(i, k) * w(j, k)).cbrt();
                            }
                        }
                    }
                    return 2.0 * s / (d * (d - 1.0));
                }
                let d: usize = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| a[i][j] as usize + a[j][i] as usize)
                    .sum();
                let both = (0..n).filter(|&j| j != i && a[i][j] && a[j][i]).count();
                let possible = (d * d.saturating_sub(1)) as f64 - 2.0 * both as f64;
                if possible <= 0.0 {
                    return 0.0;
                }
                let arcs = |x: usize, y: usize| {
                    [(a[x][y], w(x, y)), (a[y][x], w(y, x))]
                        .into_iter()
                        .filter(|p| p.0)
                        .map(|p| p.1)
                        .collect::<Vec<f64>>()
                };
                let mut t = 0.0;
                for j in (0..n).filter(|&j| j != i) {
                    for k in (0..n).filter(|&k| k != i && k != j) {
                        for x in arcs(i, j) {
                            for y in arcs(i, k) {
                                for z in arcs(j, k) {
                                    t += (x * y * z).cbrt();
                                }
                            }
                        }
                    }
                }
                t / (2.0 * possible)
            })
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        for (u, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() <= 1e-12, "{what}: node {u}: {a} vs {b}");
        }
    }

    #[test]
    fn weighted_clustering_matches_networkx_formulas() {
        for seed in 0..4u64 {
            let n = 40;
            let arcs = simple_weighted(n, 300, 10 + seed, true);
            let g = weighted::<Undirected>(&format!("u_{seed}"), n, &arcs);
            let cc = ClusteringCoefficient::new(&g).unwrap();
            let got: Vec<f64> = (0..n)
                .map(|u| cc.get_node_clusteringcoefficient(u))
                .collect();
            let want = brute_force(n, &arcs, false);
            assert_close(&got, &want, &format!("undirected {seed}"));
            let average = want.iter().sum::<f64>() / n as f64;
            assert!((cc.get_average_clustering_coefficient() - average).abs() < 1e-12);

            let arcs = simple_weighted(n, 400, 20 + seed, false);
            let g = weighted::<Directed>(&format!("d_{seed}"), n, &arcs);
            let cc = DirectedClusteringCoefficient::new(&g).unwrap();
            let want = brute_force(n, &arcs, true);
            assert_close(
                cc.local_clustering_coefficients(),
                &want,
                &format!("directed {seed}"),
            );
        }
    }

    /// Uniform weights cluster as unit ones do; transitivity ignores weights.
    #[test]
    fn uniform_weights_cluster_as_unit_ones() {
        let n = 60;
        let arcs: Vec<_> = simple_weighted(n, 500, 3, true)
            .into_iter()
            .map(|(u, v, _)| (u, v, 7.5))
            .collect();
        let g = weighted::<Undirected>("uniform", n, &arcs);
        let plain: Vec<_> = arcs.iter().map(|&(u, v, _)| (u, v)).collect();
        let p = crate::components::test_utils::graph("uniform_plain", "symmetric", n, &plain);
        let p = p.to_undirected().unwrap();
        let (w, u) = (
            ClusteringCoefficient::new(&g).unwrap(),
            ClusteringCoefficient::new(p).unwrap(),
        );
        for x in 0..n {
            let (a, b) = (
                w.get_node_clusteringcoefficient(x),
                u.get_node_clusteringcoefficient(x),
            );
            assert!((a - b).abs() < 1e-12, "node {x}: {a} vs {b}");
        }
        assert_eq!(w.get_graph_transitivity(), u.get_graph_transitivity());
    }
}
