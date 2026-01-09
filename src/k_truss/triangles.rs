use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicU8, Ordering};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;

use crate::utils::OneOrMany;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Triangles<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which each edge's triangles are computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the number of triangles of each edge.
    triangles: AbstractedProceduralMemoryMut<AtomicU8>,
    /// Threads to be used
    threads: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Triangles<'a, N, E, Ix> {
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut triangles = Self::new_no_compute(g, g.thread_num())?;
        triangles.compute_with_proc_mem(triangles.init_cache_mem()?)?;

        Ok(triangles)
    }

    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::Triangles, None)?;
        if Path::new(&t_fn).exists() {
            if let Ok(triangles) = AbstractedProceduralMemoryMut::from_file_name(&t_fn) {
                return Ok(Self {
                    g,
                    triangles,
                    threads: g.thread_num().max(1),
                });
            }
        }
        Self::new(g)
    }

    pub fn triangles(&self, e_idx: usize) -> u8 {
        assert!(e_idx < self.g.width());
        self.triangles.get(e_idx).load(Ordering::Relaxed)
    }

    pub(crate) fn triangles_shares_slice(&self) -> SharedSliceMut<AtomicU8> {
        self.triangles.shared_slice()
    }

    pub fn all_triangles(&self) -> &[AtomicU8] {
        self.triangles.as_slice()
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::Triangles, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Triangles<'a, N, E, Ix> {
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
    pub fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
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
        let out_fn = g.build_cache_filename(CacheFile::Triangles, None)?;
        let triangles = SharedSliceMut::<AtomicU8>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self {
            g,
            triangles,
            threads,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn compute_with_proc_mem_impl(&self, _proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        let tris = self.triangles.shared_slice();

        let edge_reciprocal = self.g.edge_reciprocal()?;
        let edge_out = self.g.edge_over()?;

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.threads;
            let node_load = node_count.div_ceil(threads);

            for tid in 0..threads {
                let eo = edge_out.shared_slice();
                let er = edge_reciprocal.shared_slice();

                let tris = tris.clone();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                scope.spawn(move |_| {
                    let mut neighbours = HashMap::<usize, OneOrMany<usize>>::new();
                    for u in begin..end {
                        for j in eo.get(u).index()..index_ptr.get(u + 1).index() {
                            let w = neighbours_ptr.get(j).index();
                            match neighbours.entry(w) {
                                std::collections::hash_map::Entry::Vacant(e) => {
                                    e.insert(OneOrMany::One(j));
                                }
                                std::collections::hash_map::Entry::Occupied(mut e) => {
                                    match e.get_mut() {
                                        OneOrMany::One(i) => {
                                            let mut sv: SmallVec<[usize; 4]> = SmallVec::new();
                                            sv.push(*i);
                                            sv.push(j);
                                            *e.get_mut() = OneOrMany::Many(sv);
                                        }
                                        OneOrMany::Many(sv) => sv.push(j),
                                    }
                                }
                            }
                        }
                        for u_v in index_ptr.get(u).index()..eo.get(u).index() {
                            let v = neighbours_ptr.get(u_v).index();
                            if u == v {
                                continue;
                            }
                            for v_w in (eo.get(v).index()..index_ptr.get(v + 1).index()).rev() {
                                let w = neighbours_ptr.get(v_w).index();
                                if w <= u {
                                    break;
                                }
                                match neighbours.get(&w) {
                                    Some(i) => match i {
                                        OneOrMany::One(u_w) => {
                                            let w_u = er.get(*u_w).index();
                                            tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                            tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                            tris.get(*u_w).fetch_add(1, Ordering::Relaxed);
                                            tris.get(er.get(u_v).index()).fetch_add(1, Ordering::Relaxed);
                                            tris.get(er.get(v_w).index()).fetch_add(1, Ordering::Relaxed);
                                            tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                        }
                                        OneOrMany::Many(u_ws) => {
                                            for &u_w in u_ws {
                                                let w_u = er.get(u_w).index();
                                                tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                                tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                                tris.get(u_w).fetch_add(1, Ordering::Relaxed);
                                                tris.get(er.get(u_v).index())
                                                    .fetch_add(1, Ordering::Relaxed);
                                                tris.get(er.get(v_w).index())
                                                    .fetch_add(1, Ordering::Relaxed);
                                                tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                            }
                                        }
                                    },
                                    None => continue,
                                };
                            }
                        }
                        neighbours.clear();
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        self.triangles.flush_async()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::Triangles)?;

        Ok(())
    }
}
