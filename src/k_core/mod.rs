pub mod batagelj_zaversnik;
pub mod liu_et_al;

#[cfg(test)]
use _verify::verify_k_cores;

#[cfg(test)]
mod _verify {
    use crate::{
        graph::{self, GraphMemoryMap, cache::utils::{FileType, H, cache_file_name}},
        shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut},
    };

    use crossbeam::thread;
    use num_cpus::get_physical;
    use std::sync::{Arc, Barrier};

    #[allow(dead_code)]
    pub(super) fn verify_k_cores<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        graph: &GraphMemoryMap<N, E, Ix>,
        edge_coreness: AbstractedProceduralMemoryMut<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = get_physical() * 2;
        let node_count = graph.size();
        let node_load = node_count.div_ceil(threads);

        // coreness length == graph width
        if edge_coreness.len() != graph.width() {
            return Err(format!(
                "k-cores invalid: graph width is {} but coreness vec has length {}",
                graph.width(),
                edge_coreness.len()
            )
            .into());
        }
        if graph.width() == 0 {
            return Ok(());
        }

        let e_coreness = edge_coreness.shared_slice();
        // coreness check test filename
        let cct_fn = cache_file_name("", &FileType::Test(H::H), None)?;
        let node_coreness = SharedSliceMut::<u8>::abst_mem_mut(
            &cct_fn,
            node_count,
            true,
        )?;
        let mut n_coreness = node_coreness.shared_slice();

        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(
            |scope| -> Result<(), Box<dyn std::error::Error>> {
                let mut res = vec![];
                for tid in 0..threads {
                    let synchronize = synchronize.clone();

                    let start = std::cmp::min(tid * node_load, node_count);
                    let end = std::cmp::min(start + node_load, node_count);

                    res.push(scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            for u in start..end {
                                let mut n_core = 0;
                                let mut cardinality = 0;
                                for e_idx in graph.index_node(u) {
                                    match e_coreness.get(e_idx).cmp(&n_core) {
                                        std::cmp::Ordering::Greater => {
                                        n_core = *e_coreness.get(e_idx);
                                        cardinality = 1;
                                        }
                                        std::cmp::Ordering::Equal => {
                                        cardinality += 1;
                                        }
                                        std::cmp::Ordering::Less => {}
                                    };
                                }
                                if cardinality < n_core {
                                    return Err(format!("invalid k-core: cardinality is {cardinality} but node coreness is {n_core}").into());
                                }
                                *n_coreness.get_mut(u) = n_core;
                            }

                            synchronize.wait();

                            let mut higher_deg_than_core;
                            for u in start..end {
                                higher_deg_than_core = 0;
                                let core = *n_coreness.get(u);
                                for e_idx in graph.index_node(u) {
                                    let dest_node = unsafe { graph.neighbours_ptr().add(e_idx).read() }.index();
                                    let n_core = *n_coreness.get(dest_node);
                                    // ensure coreness((u,v)) == min(coreness(u), coreness(v))
                                    if *e_coreness.get(e_idx) > n_core {
                                        return Err(
                                            format!(
                                                "invalid k-core: edge ({u}, {dest_node}) has coreness {} which is bigger than {dest_node}'s coreness {}'",
                                                *e_coreness.get(e_idx), 
                                                *n_coreness.get(dest_node)
                                                ).into()
                                            );
                                    }
                                    // ensure k-core membership maximality
                                    if n_core > core {
                                        higher_deg_than_core += 1;
                                    }
                                }
                                if higher_deg_than_core > core as usize {
                                    return Err(format!("invalid k-core: node {u}: deg_≥{core} = {higher_deg_than_core}, {u} should be in a higher k-core").into());
                                }
                            }
                            Ok(())
                        },
                    ));
                }
                // join results
                for r in res {
                    let joined_r = match r.join() {
                        Ok(b) => b,
                        Err(e) => {
                            return Err(format!("{:?}", e).into());
                        }
                    };
                    joined_r.map_err(|e| -> Box<dyn std::error::Error> {format!("{:?}", e).into()})?;
                }
                Ok(())
            },
        ).map_err(|e| -> Box<dyn std::error::Error> {format!("{:?}", e).into()})?
    }
}
