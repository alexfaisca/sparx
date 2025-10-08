use super::{CacheFile, GraphMemoryMap};
use crate::{
    shared_slice::{
        AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut,
    },
    utils::OneOrMany,
};

use crossbeam::thread;
use smallvec::SmallVec;
use std::{
    fs::OpenOptions,
    sync::{Arc, Barrier},
};

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    pub(super) fn get_edge_reciprocal_impl(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        let er_fn = self.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let dud = Vec::new();
        match OpenOptions::new().read(true).open(er_fn.as_str()) {
            Ok(i) => {
                let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
                SharedSlice::<usize>::abstract_mem(&er_fn, dud, len, true)
            }
            Err(_) => {
                self.build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(er_fn.as_str()) {
                    Ok(i) => {
                        let len =
                            i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
                        SharedSlice::<usize>::abstract_mem(&er_fn, dud, len, true)
                    }
                    Err(e) => Err(format!("error can't abst mem for edge_reciprocal {e}").into()),
                }
            }
        }
    }

    pub(super) fn get_edge_over_impl(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        let eo_fn = self.build_cache_filename(CacheFile::EdgeOver, None)?;
        let dud = Vec::new();
        match OpenOptions::new().read(true).open(eo_fn.as_str()) {
            Ok(i) => {
                let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
                SharedSlice::<usize>::abstract_mem(&eo_fn, dud, len, true)
            }
            Err(_) => {
                self.build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(eo_fn.as_str()) {
                    Ok(i) => {
                        let len =
                            i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
                        SharedSlice::<usize>::abstract_mem(&eo_fn, dud, len, true)
                    }
                    Err(e) => Err(format!("error can't abst mem for edge_over {e}").into()),
                }
            }
        }
    }

    fn init_procedural_memory_build_reciprocal(
        &self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<usize>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.size();
        let edge_count = self.width();

        let er_fn = self.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let eo_fn = self.build_cache_filename(CacheFile::EdgeOver, None)?;

        let edge_reciprocal = SharedSliceMut::<usize>::abst_mem_mut(&er_fn, edge_count, true)?;
        let edge_out = SharedSliceMut::<usize>::abst_mem_mut(&eo_fn, node_count, true)?;

        Ok((edge_reciprocal, edge_out))
    }

    fn build_reciprocal_edge_index(
        &self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<usize>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.size();
        let edge_count = self.width();

        let threads = self.thread_count.max(1) as usize;
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<usize>::new(self.offsets_ptr(), self.offsets_size());
        let neighbours_ptr = SharedSlice::<usize>::new(self.neighbours_ptr(), edge_count);

        let (er, eo) = self.init_procedural_memory_build_reciprocal()?;

        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|scope|-> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let mut er = er.shared_slice();
                let mut eo = eo.shared_slice();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                threads_res.push(scope.spawn(move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

                    let mut edges_start = *index_ptr.get(begin);

                    for u in begin..end {
                        let mut eo_at_end = true;
                        let edges_stop = *index_ptr.get(u + 1);
                        for edge_offset in edges_start..edges_stop {
                            let v = *neighbours_ptr.get(edge_offset);
                            if v > u {
                                eo_at_end = false;
                                *eo.get_mut(u) = edge_offset;
                                break;
                            }
                        }
                        if eo_at_end {
                            *eo.get_mut(u) = edges_stop;
                        }
                        edges_start = edges_stop;
                    }

                    synchronize.wait();

                    let mut prev = OneOrMany::<(usize, usize)>::One((node_count, 0usize));
                    for u in begin..end {
                        for edge_offset in *index_ptr.get(u)..*eo.get(u) {
                            let v = *neighbours_ptr.get(edge_offset);
                            // binary search on neighbours w, where w < v
                            if u == v {
                                // self loops are their own reciprocals
                                *er.get_mut(edge_offset) = edge_offset;
                                // reset
                                prev = OneOrMany::One((v, edge_offset));
                                continue;
                            } else if let OneOrMany::One((prev_v, prev_offset)) = prev {
                                if prev_v == v {
                                    // find reciprocal, dest is the same, offset is different
                                    // make entry into list and record (never actually loops)
                                    let reciprocal = if prev_offset < edge_count - 1 && *neighbours_ptr.get(prev_offset + 1) == u {
                                        prev_offset + 1
                                    } /*else if prev_offset < edge_count - 1 {
                                            println!("after {prev_offset}->{u} comes {}", graph_ptr.get(prev_offset + 1).dest());
                                        }*/
                                        // search backwards
                                    else if prev_offset > 0 && *neighbours_ptr.get(prev_offset - 1) == u {
                                        prev_offset - 1
                                    } /*else if prev_offset > 0 {
                                            println!("before {prev_offset}->{u} comes {}", graph_ptr.get(prev_offset - 1).dest());
                                        }*/
                                    else {
                                        return Err(format!("error couldn't find reciprocal for 2nd multi edge {edge_offset}, u: ({u}) -> v: ({v}), all options were already taken").into());
                                    };
                                    let mut vs: SmallVec<[(usize, usize); 4]> = SmallVec::new();
                                    vs.push((prev_v, prev_offset));
                                    vs.push((v, reciprocal));
                                    prev = OneOrMany::Many(vs);
                                    *er.get_mut(edge_offset) = reciprocal;
                                    *er.get_mut(reciprocal) = edge_offset;
                                    continue;
                                }
                            } else if let OneOrMany::Many(ref mut vs) = prev {
                                if vs[0].0 == v {
                                    // find reciprocal, dest is the same, offset is different from
                                    // any of the recorded offsets: greedy lookup
                                    let reciprocal = 'search_adjacents: {
                                        // search forwards
                                        let mut res = vs[0].1;
                                        while res < edge_count - 1 && *neighbours_ptr.get(res + 1) == u {
                                            res += 1;
                                            if vs.iter().any(|&(_, b)| b == res) {
                                                continue;
                                            } else {
                                                break 'search_adjacents res;
                                            }
                                        }
                                        // reset and search backwards
                                        res = vs[0].1;
                                        while res > 0 && *neighbours_ptr.get(res - 1) == u {
                                            res -= 1;
                                            if vs.iter().any(|&(_, b)| b == res) {
                                                continue;
                                            } else {
                                                break 'search_adjacents res;
                                            }
                                        }

                                        return Err(format!("error couldn't find reciprocal for nth multi edge {edge_offset}, u: ({u}) -> v: ({v}), all options were already taken").into());
                                    };

                                    // extend entry's list
                                    vs.push((v, reciprocal));
                                    *er.get_mut(edge_offset) = reciprocal;
                                    *er.get_mut(reciprocal) = edge_offset;
                                    continue;
                                }
                            }

                                let mut floor = *eo.get(v);
                                let mut ceil = *index_ptr.get(v + 1);
                                let reciprocal =  loop {
                                    if floor > ceil {
                                        return Err(format!("error couldn't find reciprocal for edge {edge_offset}, u: ({u}) -> v: ({v})").into());
                                    }

                                    let m = floor + (ceil - floor) / 2;
                                    let dest = *neighbours_ptr.get(m);

                                    match dest.cmp(&u) {
                                        std::cmp::Ordering::Equal => break m,
                                        std::cmp::Ordering::Greater => ceil = m - 1,
                                        std::cmp::Ordering::Less => floor = m + 1,
                                    }
                                };

                                *er.get_mut(edge_offset) = reciprocal;
                                *er.get_mut(reciprocal) = edge_offset;
                                // reset
                                prev = OneOrMany::One((v, reciprocal));
                        }

                        prev = OneOrMany::One((node_count, 0));
                    }

                    Ok(())
                }));
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

        er.flush()?;
        eo.flush()?;

        Ok((er, eo))
    }
}
