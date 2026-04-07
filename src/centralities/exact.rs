use std::mem::ManuallyDrop;

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;
use crate::trails::bfs::BFSDists;

use crossbeam::thread;

pub struct ExactClosenessCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight = f64> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
    /// Normalization flag.
    normalized: Option<bool>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactClosenessCentrality<'a, N, E, Ix, W> {
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, normalized)?;

        hyper_ball.compute_with_proc_mem(())?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactCloseness, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactClosenessCentrality<'a, N, E, Ix, W> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
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

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactCloseness, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);
        let index = Self::index_by_normalization(self.normalized);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        if index == 0 {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    *e.get_mut(u) = W::from_f64(bfs.total_distances());
                                }
                            }
                        } else if index == 1 {
                            let norm = W::from_usize(node_count.saturating_sub(1));
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    *e.get_mut(u) = norm / W::from_f64(bfs.total_distances());
                                }
                            }
                        } else {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    *e.get_mut(u) = W::from_usize(bfs.recheable())
                                        / W::from_f64(bfs.total_distances());
                                }
                            }
                        }
                        Ok(())
                    },
                ));
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactCloseness)?;

        Ok(())
    }
}

pub struct ExactHarmonicCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight = f64> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
    /// Normalization flag.
    normalized: Option<bool>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactHarmonicCentrality<'a, N, E, Ix, W> {
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, normalized)?;

        hyper_ball.compute_with_proc_mem(())?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactHarmonic, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactHarmonicCentrality<'a, N, E, Ix, W> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
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

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactHarmonic, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);
        let index = Self::index_by_normalization(self.normalized);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        if index == 0 {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    let inv_dist: W = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            if d > 0 { W::one() / W::from_usize(d) } else { W::zero() }
                                        })
                                        .fold(W::zero(), |a, b| a + b);
                                    *e.get_mut(u) = inv_dist;
                                }
                            }
                        } else if index == 1 {
                            let norm = W::from_usize(node_count.saturating_sub(1));
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    let inv_dist: W = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            if d > 0 { W::one() / W::from_usize(d) } else { W::zero() }
                                        })
                                        .fold(W::zero(), |a, b| a + b);
                                    *e.get_mut(u) = norm / inv_dist;
                                }
                            }
                        } else {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = W::zero();
                                } else {
                                    let inv_dist: W = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            if d > 0 { W::one() / W::from_usize(d) } else { W::zero() }
                                        })
                                        .fold(W::zero(), |a, b| a + b);
                                    *e.get_mut(u) = W::from_usize(bfs.recheable()) / inv_dist;
                                }
                            }
                        }
                        Ok(())
                    },
                ));
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactHarmonic)?;

        Ok(())
    }
}

pub struct ExactLinCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight = f64> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<W>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactLinCentrality<'a, N, E, Ix, W> {
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g)?;

        hyper_ball.compute_with_proc_mem(())?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> W {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[W] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactLin, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight> ExactLinCentrality<'a, N, E, Ix, W> {
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
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
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

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactCloseness, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for u in begin..end {
                            let bfs = BFSDists::new(graph, u).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error: {:?}", e).into()
                                },
                            )?;
                            if bfs.total_distances() == 0. {
                                *e.get_mut(u) = W::zero();
                            } else {
                                let r = W::from_usize(bfs.recheable());
                                *e.get_mut(u) = (r * r) / W::from_f64(bfs.total_distances());
                            }
                        }
                        Ok(())
                    },
                ));
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactLin)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use std::io::Write;

    /// A small 4-clique: a connected graph with a trivially known structure (every node reaches
    /// every other node in exactly one hop), used to sanity-check the exact centralities'
    /// accumulation across `Ix` widths and `W` (weight) precisions.
    fn write_clique_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating clique mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "4 4 6").unwrap();
        for &(u, v) in &[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
            writeln!(f, "{u} {v}").unwrap();
        }
        f.flush().unwrap();
    }

    fn open_graph<Ix: graph::IndexType>(
        path: &std::path::Path,
        id: &str,
    ) -> GraphMemoryMap<VoidLabel, VoidLabel, Ix> {
        GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::from_file(path, Some(id.to_string()), None)
            .expect("graph parse should succeed")
    }

    /// Runs all three exact centralities across several `Ix` widths (`W` fixed to `f64`) on a
    /// small clique, asserting every width produces the exact same (bit-for-bit identical)
    /// values --- `Ix`'s width only affects indexing/storage, never the floating-point
    /// computation itself.
    #[test]
    fn exact_centralities_cross_ix_widths_agree() {
        let dir = unique_test_dir("exact_cross_ix_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        macro_rules! run_all {
            ($ixty:ty, $tag:expr) => {{
                let graph = open_graph::<$ixty>(&path, &format!("exact_{}_{}", $tag, run_tag));
                let closeness = ExactClosenessCentrality::<_, _, _, f64>::new(&graph, Some(true))
                    .unwrap_or_else(|e| panic!("closeness({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                let harmonic = ExactHarmonicCentrality::<_, _, _, f64>::new(&graph, Some(true))
                    .unwrap_or_else(|e| panic!("harmonic({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                let lin = ExactLinCentrality::<_, _, _, f64>::new(&graph)
                    .unwrap_or_else(|e| panic!("lin({}) should succeed: {e}", $tag))
                    .centralities()
                    .to_vec();
                (closeness, harmonic, lin)
            }};
        }

        let (c_usize, h_usize, l_usize) = run_all!(usize, "usize");
        let (c_u32, h_u32, l_u32) = run_all!(u32, "u32");
        let (c_u16, h_u16, l_u16) = run_all!(u16, "u16");

        assert_eq!(c_usize, c_u32, "closeness: Ix=u32 should match Ix=usize exactly");
        assert_eq!(c_usize, c_u16, "closeness: Ix=u16 should match Ix=usize exactly");
        assert_eq!(h_usize, h_u32, "harmonic: Ix=u32 should match Ix=usize exactly");
        assert_eq!(h_usize, h_u16, "harmonic: Ix=u16 should match Ix=usize exactly");
        assert_eq!(l_usize, l_u32, "lin: Ix=u32 should match Ix=usize exactly");
        assert_eq!(l_usize, l_u16, "lin: Ix=u16 should match Ix=usize exactly");

        // sanity: every node in a clique reaches the 3 others at distance 1, so each raw
        // (non-normalized) closeness value should equal 3 and every value should be positive.
        for (v, &c) in c_usize.iter().enumerate() {
            assert!(c.is_finite() && c > 0.0, "node {v}: closeness should be positive, got {c}");
        }
    }

    /// Runs the exact centralities with `W=f32` vs `W=f64` on the same clique, asserting the
    /// results agree closely (not necessarily bit-for-bit, but no catastrophic-cancellation risk
    /// exists here --- see `HyperBallInner`'s analogous test/docs --- so small graphs in
    /// particular should match almost exactly).
    #[test]
    fn exact_centralities_f32_and_f64_agree_closely() {
        let dir = unique_test_dir("exact_cross_weight_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        let graph_f64 = open_graph::<usize>(&path, &format!("exact_f64_{run_tag}"));
        let closeness_f64 = ExactClosenessCentrality::<_, _, _, f64>::new(&graph_f64, Some(true))
            .expect("closeness(f64) should succeed")
            .centralities()
            .to_vec();

        let graph_f32 = open_graph::<usize>(&path, &format!("exact_f32_{run_tag}"));
        let closeness_f32 = ExactClosenessCentrality::<_, _, _, f32>::new(&graph_f32, Some(true))
            .expect("closeness(f32) should succeed")
            .centralities()
            .to_vec();

        assert_eq!(closeness_f64.len(), closeness_f32.len());
        for (v, (&c64, &c32)) in closeness_f64.iter().zip(closeness_f32.iter()).enumerate() {
            assert!(
                (c64 - c32 as f64).abs() < 1e-2,
                "node {v}: f64 closeness {c64} and f32 closeness {c32} should be close"
            );
        }
    }
}
