use super::{Community, GraphMemoryMap};
use crate::shared_slice::{SharedSlice, SharedSliceMut};

use crossbeam::thread;
use num_cpus::get_physical;
use ordered_float::OrderedFloat;
use portable_atomic::{AtomicUsize, Ordering};
use std::{
    collections::HashSet,
    sync::{Arc, Barrier},
};

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    pub(super) fn modularity_impl(
        &self,
        communities: &[Ix],
        comms_cardinality: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let m2 = match self.width() {
            0 => return Ok(0.),
            edge_count => edge_count as f64,
        };
        let node_count = self.size();

        // simple input invariants validatation
        if communities.len() != node_count {
            return Err(format!("error can't calculate the modularity of a partition with length {}: partition length must equal |V| = {node_count}", communities.len()).into());
        } else if comms_cardinality > node_count {
            return Err(format!("error community cardinality {comms_cardinality} can't be bigger than |V| = {node_count}").into());
        }

        // helper internal degree accumulator
        let id_fn = self.build_helper_filename(0)?;
        // helper total degree accumulator
        let td_fn = self.build_helper_filename(1)?;

        if E::is_weighted() {
            // edges weigh their labels' weights (finite and non-negative), summed in `f64`
            let mut internal =
                SharedSliceMut::<f64>::abst_mem_mut(&id_fn, comms_cardinality, true)?;
            let mut total = SharedSliceMut::<f64>::abst_mem_mut(&td_fn, comms_cardinality, true)?;
            let mut m2 = 0f64;
            for u in 0..node_count {
                let comm_u = communities[u].index();
                for (v, label) in self.weighted_neighbours(u)? {
                    let w = label.weight::<f64>();
                    if !(w.is_finite() && w >= 0.) {
                        drop((internal, total));
                        self.cleanup_helpers()?;
                        return Err(format!(
                            "error modularity's edge weights must be finite and non-negative, but one is {label:?}"
                        )
                        .into());
                    }
                    m2 += w;
                    *total.get_mut(comm_u) += w;
                    if v < node_count && communities[v].index() == comm_u {
                        *internal.get_mut(comm_u) += w;
                    }
                }
            }
            let modularity = if m2 == 0. {
                0.
            } else {
                (0..comms_cardinality)
                    .map(|c| internal.get(c) / m2 - (total.get(c) / m2).powi(2))
                    .sum()
            };
            drop((internal, total));
            self.cleanup_helpers()?;
            return Ok(modularity);
        }

        let mut internal_degree =
            SharedSliceMut::<usize>::abst_mem_mut(&id_fn, comms_cardinality, true)?;
        let mut total_degree =
            SharedSliceMut::<usize>::abst_mem_mut(&td_fn, comms_cardinality, true)?;

        for u in 0..node_count {
            let iter = self.neighbours(u)?;
            let comm_u = communities[u].index();
            *total_degree.get_mut(comm_u) += iter.remaining_neighbours();
            for v in iter {
                if v >= node_count {
                    continue;
                } // safety
                if communities[v].index() == comm_u {
                    *internal_degree.get_mut(comm_u) += 1;
                }
            }
        }

        // modularity: sum_c( internal_dir_edges[c]/m2 - (Kc[c]/m2)^2 )
        let mut partition_modularity = 0.0f64;
        for c in 0..comms_cardinality {
            let lc_over_m2 = (*internal_degree.get(c) as f64) / m2;
            let kc_over_m2 = (*total_degree.get(c) as f64) / m2;
            partition_modularity += lc_over_m2 - kc_over_m2 * kc_over_m2;
        }

        self.cleanup_helpers()?;

        Ok(partition_modularity)
    }

    pub(crate) fn coallesce_isolated_nodes_community(
        &self,
        communities: SharedSliceMut<Ix>,
        comm_count: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let node_count = self.size();
        if node_count <= 1 {
            return Ok(node_count);
        }
        let threads = self.thread_num().max(get_physical() * 2);
        let node_load = node_count.div_ceil(threads);

        let offsets_ptr = SharedSlice::<Ix>::new(self.offsets_ptr(), self.offsets_size());
        // helper indexer
        let h_fn = self.build_helper_filename(1)?;
        // allocate |V| + 1 usize's to store the beginning and end offsets for each node's edges
        let counter = SharedSliceMut::<usize>::abst_mem_mut(&h_fn, comm_count, true)?;

        let isolated_nodes_comm = Arc::new(AtomicUsize::new(0));
        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            // Thread syncronization
            let synchronize = Arc::new(Barrier::new(threads));
            let mut handles = vec![];

            for tid in 0..threads {
                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let mut counter = counter.shared_slice();
                let mut communities = communities;

                let synchronize = synchronize.clone();
                let isolated_nodes_comm = isolated_nodes_comm.clone();

                handles.push(scope.spawn(move |_| {
                    // mark isolated nodes
                    for u in begin..end {
                        if offsets_ptr.get(u + 1).index() - offsets_ptr.get(u).index() == 0 {
                            let comm_u = communities.get(u).index();
                            assert!(comm_u <= comm_count);
                            *counter.get_mut(comm_u) = 1;
                        }
                    }

                    synchronize.wait();
                    if tid == 0 {
                        // prefix sum marks
                        let mut sum = 0;
                        for u in 0..comm_count {
                            let mark = *counter.get(u);
                            *counter.get_mut(u) = sum;
                            sum += mark;
                        }
                        let isolated_comm = comm_count - sum;
                        isolated_nodes_comm.store(isolated_comm, Ordering::Relaxed);
                        println!("new max comm {isolated_comm}");
                    }
                    synchronize.wait();

                    let isolated_nodes_comm = isolated_nodes_comm.load(Ordering::Relaxed);

                    for u in begin..end {
                        if offsets_ptr.get(u + 1).index() - offsets_ptr.get(u).index() == 0 {
                            *communities.get_mut(u) = Ix::new(isolated_nodes_comm);
                        } else {
                            let comm_u = communities.get(u).index();
                            let new_comm_u = comm_u - *counter.get(comm_u);
                            *communities.get_mut(u) = Ix::new(new_comm_u);
                            if new_comm_u > 1000000000 {
                                println!("{u}");
                            }
                        }
                    }
                }));
            }
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join().map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error in thread {tid}: {:?}", e).into()
                })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(isolated_nodes_comm.load(Ordering::Relaxed) + 1)
    }

    /// Performs a sweep cut over a given diffusion vector[^1] by partition conductance.
    ///
    /// # Arguments
    ///
    /// * `diffusion` --- the diffusion vector[^1][^2].
    /// * `target_size` --- the partition's target size[^3].
    /// * `target_volume` --- the partition's target volume[^4].
    ///
    /// [^1]: diffusion vector entries must be of type (node_id: [`usize`], heat: [`f64`]).
    /// [^2]: entries must be descendingly ordered by diffusion.
    /// [^3]: if [`None`] is provided defaults to `|V|`, effectively, the overall best partition by conducatance is returned independent on the number of nodes in it.
    /// [^4]: if [`None`] is provided defaults to `|E|`, effectively, the overall best partition by conducatance is returned independent on the number of edges in it.
    /// The sweep cut of a diffusion vector: its nodes, by decreasing diffusion, added one by one
    /// to a set `S`, the best prefix by conductance `ϕ(S) = cut(S) / min(vol(S), vol(V \ S))`.
    /// Weighted edges (see [`E::is_weighted`](crate::graph::E::is_weighted)) count their weight
    /// in cuts and volumes (and must be finite, non-negative, and symmetric: each edge's weight
    /// the same both ways); unweighted ones count one each. The sweep stops once `S` holds twice
    /// `target_size` nodes or twice `target_volume` arcs.
    pub(super) fn sweep_cut_over_diffusion_vector_by_conductance(
        &self,
        diffusion: &mut [(usize, f64)],
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        diffusion.sort_unstable_by_key(|(_, mass)| std::cmp::Reverse(OrderedFloat(*mass)));
        let target_size = target_size.map_or(self.size(), |s| s);
        let target_volume = target_volume.map_or(self.width(), |s| s);
        let weighted = E::is_weighted();
        // (checks the weights, for weighted edges)
        let total_weight = self.total_weight()?;

        // arcs out of S, and of V \ S, and arcs across: counted, and weighed
        let (mut vol_s, mut cut_s) = (0usize, 0usize);
        let mut vol_v_minus_s = self.width();
        let (mut w_vol_s, mut w_cut_s) = (0f64, 0f64);
        let mut community: HashSet<usize> = HashSet::new();
        let mut best_conductance = 1f64;
        let mut best_community: Vec<(usize, f64)> = Vec::new();
        let mut best_size = 0usize;
        let mut best_width = 0usize;
        let mut best_volume = 0f64;

        for (idx, (u, _)) in diffusion.iter().enumerate() {
            let u_n = self
                .weighted_neighbours(*u)
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error sweep cut couldn't get {u} neighbours: {e}").into()
                })?;
            let neighbour_count = u_n.len();

            vol_s = vol_s.checked_add(neighbour_count).ok_or_else(
                || -> Box<dyn std::error::Error> {
                    format!("error sweep cut overflow_add in vol_s at node {u}").into()
                },
            )?;
            vol_v_minus_s = vol_v_minus_s.checked_sub(neighbour_count).ok_or_else(
                || -> Box<dyn std::error::Error> {
                    format!("error sweep cut overflow_sub in vol_v_minus_s at node {u}").into()
                },
            )?;

            if !community.insert(*u) {
                return Err(
                    format!("error sweepcut diffusion vector: {u} present multiple times").into(),
                );
            }

            for (v, label) in u_n {
                let w = if weighted { label.weight::<f64>() } else { 1. };
                w_vol_s += w;
                if v == *u {
                    continue;
                }
                // an edge to S stops crossing the cut, one to V \ S starts
                if community.contains(&v) {
                    cut_s = cut_s
                        .checked_sub(1)
                        .ok_or_else(|| -> Box<dyn std::error::Error> {
                            format!("error sweepcut overflow_sub at node {u} in neighbour {v}")
                                .into()
                        })?;
                    w_cut_s -= w;
                } else {
                    cut_s = cut_s
                        .checked_add(1)
                        .ok_or_else(|| -> Box<dyn std::error::Error> {
                            format!("error sweepcut overflow_add at node {u} in neighbour {v}")
                                .into()
                        })?;
                    w_cut_s += w;
                }
            }

            let conductance = if weighted {
                // (rounding may leave a cut of nothing a hair below zero)
                w_cut_s.max(0.) / w_vol_s.min(total_weight - w_vol_s)
            } else {
                (cut_s as f64) / (std::cmp::min(vol_s, vol_v_minus_s) as f64)
            };
            if conductance < best_conductance {
                best_conductance = conductance;
                best_community = diffusion[0..=idx].to_vec();
                best_width = vol_s;
                best_volume = if weighted { w_vol_s } else { vol_s as f64 };
                best_size = community.len();
            }

            // truncate sweep if vol or size go over double the target value
            if community.len() > target_size * 2 || vol_s > target_volume * 2 {
                println!(
                    "Sweep cut truncated with size: {} and volume {}\n\tTarget size: {target_size}\n\tTarget volume: {target_volume}",
                    community.len(),
                    vol_s
                );
                break;
            }
        }

        Ok(Community {
            nodes: best_community,
            size: best_size,
            width: best_width,
            volume: best_volume,
            conductance: best_conductance,
        })
    }
}
