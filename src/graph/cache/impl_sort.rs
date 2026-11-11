use crate::shared_slice::SharedSliceMut;

use super::{AccumulatorAtomic, GraphCache, utils::apply_order_in_place};

use crossbeam::thread;

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: super::IndexType> GraphCache<N, E, Ix> {
    #[inline(always)]
    pub(super) fn sort_edges<D: AccumulatorAtomic>(
        offsets: SharedSliceMut<D::Plain>,
        neighbors: SharedSliceMut<Ix>,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = offsets.len().saturating_sub(1);
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();
            for tid in 0..threads {
                let node_load = node_count.div_ceil(threads);
                let begin = std::cmp::min(node_load * tid, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let mut neighbors = neighbors;

                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for node in begin..end {
                            let begin = D::widen(*offsets.get(node));
                            let end = D::widen(*offsets.get(node + 1));
                            let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            node_edges.sort();
                        }
                        Ok(())
                });
                handles.push(handle);
            }

            for (idx, handle) in handles.into_iter().enumerate() {
                handle
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error joining thread {idx}: {:?}", e).into()
                    })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in thread {idx}: {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }

    #[inline(always)]
    pub(super) fn sort_edges_with_labels<D: AccumulatorAtomic>(
        offsets: SharedSliceMut<D::Plain>,
        neighbors: SharedSliceMut<Ix>,
        edge_labels: SharedSliceMut<E>,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = offsets.len().saturating_sub(1);
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();
            for tid in 0..threads {
                let node_load = node_count.div_ceil(threads);
                let begin = std::cmp::min(node_load * tid, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let mut neighbors = neighbors;
                let mut edge_labels = edge_labels;

                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        let mut idx = Vec::with_capacity(u16::MAX as usize);
                        let mut destination: Vec<usize> = Vec::new();
                        for node in begin..end {
                            let begin = D::widen(*offsets.get(node));
                            let end = D::widen(*offsets.get(node + 1));
                            // this node's permutation only
                            idx.clear();
                            idx.extend(0..end - begin);
                            let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            let edge_labels = edge_labels.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges' edge labels as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            // idx[k]: the edge that goes to position k
                            idx.sort_by_key(|&i| node_edges[i]);
                            apply_order_in_place(&idx, &mut destination, node_edges, edge_labels);
                        }
                        Ok(())
                });
                handles.push(handle);
            }

            for (idx, handle) in handles.into_iter().enumerate() {
                handle
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error joining thread {idx}: {:?}", e).into()
                    })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in thread {idx}: {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }
}
