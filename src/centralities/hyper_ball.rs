use crate::graph::{self, *};
use crate::shared_slice::*;

use crossbeam::thread;
use crate::centralities::hyper_log_log::{HyperLogLog, Precision};
use portable_atomic::{AtomicBool, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Barrier};

// Re-export precisions for UX enhancement
pub use crate::centralities::hyper_log_log::{
    Precision4, Precision5, Precision6, Precision7, Precision8, Precision9, Precision10,
    Precision11, Precision12, Precision13, Precision14, Precision15, Precision16,
};

type ProceduralMemoryHB<P, W> = (
    AbstractedProceduralMemoryMut<HyperLogLog<P>>,
    AbstractedProceduralMemoryMut<W>,
    AbstractedProceduralMemoryMut<W>,
);

pub type HyperBall4<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision4>;
pub type HyperBall5<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision5>;
pub type HyperBall6<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision6>;
pub type HyperBall7<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision7>;
pub type HyperBall8<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision8>;
pub type HyperBall9<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision9>;
pub type HyperBall10<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision10>;
pub type HyperBall11<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision11>;
pub type HyperBall12<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision12>;
pub type HyperBall13<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision13>;
pub type HyperBall14<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision14>;
pub type HyperBall15<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision15>;
pub type HyperBall16<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision16>;

/// Enum for centralities' caching filenames' creation manager logic.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
enum Centrality {
    Harmonic,
    NHarmonic,
    NCHarmonic,
    Closeness,
    NCloseness,
    NCCloseness,
    Lin,
}

/// For the *HyperBall Algorithm* described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144)
/// by Boldi P. and Vigna S. on [`GraphMemoryMap`] instances.
///
/// * Note: De Bruijn graphs are symmetric graphs, so running the HyperBall algorithm on the graph is the same as running it on its transpose. Hence, we can run HyperBall ona De Bruinjn graph to obtain the necessary components to determine the closeness, harmonic and Lin's centralities for every node of the graph.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct HyperBallInner<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    P: Precision = Precision8,
    W: Weight = f64,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's *HyperLogLog* counter.
    counters: AbstractedProceduralMemoryMut<HyperLogLog<P>>,
    /// Memmapped slice containing each node's distance accumulator.
    distances: AbstractedProceduralMemoryMut<W>,
    /// Memmapped slice containing each node's inverse distance accumulator.
    inverse_distances: AbstractedProceduralMemoryMut<W>,
    /// Maximum number of iteration --- default is 100, max is 1024 (details on how these values are ludicrously big are found in the abovementioned paper).
    max_t: usize,
    /// Closeness centralities' cached values --- under various degrees of normalization.
    closeness: [Option<AbstractedProceduralMemoryMut<W>>; 2],
    /// Harmonic centralities' cached values --- under various degrees of normalization.
    harmonic: [Option<AbstractedProceduralMemoryMut<W>>; 2],
    /// Lin's centrality's cached values.
    lin: Option<AbstractedProceduralMemoryMut<W>>,
    threads: usize,
    #[cfg(any(test, feature = "bench"))]
    iters: usize,
    /// Wall time of each iteration (measured by thread 0, up to the iteration's barrier).
    #[cfg(feature = "bench")]
    iter_times: Vec<std::time::Duration>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, P: Precision, W: Weight>
    HyperBallInner<'a, N, E, Ix, P, W>
{
    const DEAFULT_MAX_DEPTH: usize = 100;
    const MAX_MAX_DEPTH: usize = 1024;

    /// Performs the *HyperBall Algorithm* as described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144) by Boldi P. and Vigna S.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the *HyperBall Algorithm* is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, None, g.thread_num())?;

        let proc_mem = hyper_ball.init_cache_mem()?;
        hyper_ball.compute_with_proc_mem(proc_mem)?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    /// Performs the *HyperBall Algorithm* as described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144) by Boldi P. and Vigna S.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the *HyperBall Algorithm* is to be performed in.
    /// * `max_depth` --- the maximum number of iterations of the *HyperBall Algorithm* to tolerate before convergence is achieved (defaults to 100, clamped to `1..=1024`).
    /// * `threads` --- the number of threads to be used in the computation.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a GraphMemoryMap<N, E, Ix>,
        max_depth: Option<usize>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, max_depth, threads.max(1))?;

        let proc_mem = hyper_ball.init_cache_mem()?;
        hyper_ball.compute_with_proc_mem(proc_mem)?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }
    /// Searches for a previously cached result, and if not found performs the *HyperBall Algorithm* as described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144) by Boldi P. and Vigna S.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the *HyperBall Algorithm* is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let h_fn = g.build_cache_filename(CacheFile::HyperBall, None)?;
        if Path::new(&h_fn).exists() {
            // cached files from runs with a different precision or weight type have a different
            // length and are recomputed rather than reinterpreted
            if let Some(counters) = Self::load_cached(&h_fn, node_count) {
                let d_fn = g.build_cache_filename(CacheFile::HyperBallDistances, None)?;
                let id_fn = g.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;
                if let Some(distances) = Self::load_cached(&d_fn, node_count) {
                    if let Some(inverse_distances) = Self::load_cached(&id_fn, node_count) {
                        let mut hb = Self {
                            g,
                            counters,
                            distances,
                            inverse_distances,
                            max_t: Self::DEAFULT_MAX_DEPTH,
                            threads: g.thread_num().max(1),
                            closeness: [None, None],
                            harmonic: [None, None],
                            lin: None,
                            #[cfg(any(test, feature = "bench"))]
                            iters: 0,
                            #[cfg(feature = "bench")]
                            iter_times: vec![],
                        };
                        let _ = hb.search_cache_closeness_centrality(false);
                        let _ = hb.search_cache_harmonic_centrality(false);
                        let _ = hb.search_cache_closeness_centrality(true);
                        let _ = hb.search_cache_harmonic_centrality(true);
                        let _ = hb.search_cache_lins_centrality();
                        return Ok(hb);
                    }
                }
            }
        }
        Self::new(g)
    }

    /// Manager logic for centralities' caching filenames' creation.
    fn centrality_cache_file_name(
        &self,
        centrality: Centrality,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let (cache_file_type, seq) = match centrality {
            Centrality::Closeness => (CacheFile::HyperBallClosenessCentrality, Some(0)),
            Centrality::NCloseness => (CacheFile::HyperBallClosenessCentrality, Some(1)),
            Centrality::NCCloseness => (CacheFile::HyperBallClosenessCentrality, Some(2)),
            Centrality::Harmonic => (CacheFile::HyperBallHarmonicCentrality, Some(0)),
            Centrality::NHarmonic => (CacheFile::HyperBallHarmonicCentrality, Some(1)),
            Centrality::NCHarmonic => (CacheFile::HyperBallHarmonicCentrality, Some(2)),
            Centrality::Lin => (CacheFile::HyperBallLinCentrality, Some(0)),
        };

        self.build_pers_cache_filename(cache_file_type, seq)
    }

    fn index_by_normalization(normalization: bool) -> usize {
        match normalization {
            false => 0,
            true => 1,
        }
    }

    fn centrality_normalization(centrality: Centrality, normalization: bool) -> Centrality {
        let index = Self::index_by_normalization(normalization);
        match centrality {
            Centrality::Lin => Centrality::Lin,
            Centrality::Harmonic => match index {
                0 => Centrality::NHarmonic,
                1 => Centrality::NCHarmonic,
                // dummy
                _ => Centrality::Harmonic,
            },
            Centrality::Closeness => match index {
                0 => Centrality::NCloseness,
                1 => Centrality::NCCloseness,
                // dummy
                _ => Centrality::Closeness,
            },
            a => a,
        }
    }

    /// Maps a cached file holding exactly `len` elements of `T`.
    fn load_cached<T>(file_name: &str, len: usize) -> Option<AbstractedProceduralMemoryMut<T>> {
        AbstractedProceduralMemoryMut::<T>::from_file_name(file_name)
            .ok()
            .filter(|mem| mem.as_slice().len() == len)
    }

    fn get_from_cent_file(
        &self,
        target: Centrality,
    ) -> Result<AbstractedProceduralMemoryMut<W>, Box<dyn std::error::Error>> {
        let t_fn = self.centrality_cache_file_name(target)?;
        if Path::new(&t_fn).exists() {
            Self::load_cached(&t_fn, self.g.size())
                .ok_or_else(|| format!("error {t_fn} centrality file has the wrong length").into())
        } else {
            Err(format!("error no {t_fn} centrality file").into())
        }
    }

    /// Gets (from a previously cached result) the approximation of a given node's total distance to other nodes from their respective distance accumulator.
    pub fn get_geodesic_distance(&self, node_id: usize) -> W {
        assert!(node_id < self.g.size());
        *self.distances.get(node_id)
    }

    /// Gets (from a previously cached result) the approximation of a given node's total inverse distance to other nodes from their respective inverse distance accumulator.
    pub fn get_harmonic_geodesic_distance(&self, node_id: usize) -> W {
        assert!(node_id < self.g.size());
        *self.inverse_distances.get(node_id)
    }

    /// Gets (from a previously cached result) the approximation of each node's total distance to other nodes from their respective distance accumulator.
    pub fn geodesic_distances(&self) -> &[W] {
        self.distances.as_slice()
    }

    /// Gets (from a previously cached result) the approximation of each node's total inverse distance to other nodes from their respective inverse distance accumulator.
    pub fn harmonic_geodesic_distances(&self) -> &[W] {
        self.inverse_distances.as_slice()
    }

    /// Gets (from a previously cached result) the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn get_closeness_centrality(
        &self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if let Some(mem) = self.closeness[idx].as_ref() {
            return Ok(mem.as_slice());
        }
        Err("error no such cached centrality".into())
    }

    /// Searches cache for the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog* counter estimation).
    pub fn search_cache_closeness_centrality(
        &mut self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        if let Ok(mem) = self.get_from_cent_file(Self::centrality_normalization(
            Centrality::Closeness,
            normalization,
        )) {
            let idx = Self::index_by_normalization(normalization);
            self.closeness[idx] = Some(mem);
            if let Some(mem) = self.closeness[idx].as_ref() {
                return Ok(mem.as_slice());
            }
        }
        Err("error no such cached centrality".into())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn get_or_compute_closeness_centrality(
        &mut self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.closeness[idx].is_none() {
            self.compute_closeness_centrality(normalization)
        } else {
            self.get_closeness_centrality(normalization)
        }
    }

    /// Gets (from a previously cached result) the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn get_hamonic_centrality(
        &self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if let Some(mem) = self.harmonic[idx].as_ref() {
            return Ok(mem.as_slice());
        }
        Err("no such cached centrality".into())
    }

    /// Searches cache for the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog* counter estimation).
    pub fn search_cache_harmonic_centrality(
        &mut self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        if let Ok(mem) = self.get_from_cent_file(Self::centrality_normalization(
            Centrality::Harmonic,
            normalization,
        )) {
            let idx = Self::index_by_normalization(normalization);
            self.harmonic[idx] = Some(mem);
            if let Some(mem) = self.harmonic[idx].as_ref() {
                return Ok(mem.as_slice());
            }
        }
        Err("error no such cached centrality".into())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn get_or_compute_harmonic_centrality(
        &mut self,
        normalization: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.harmonic[idx].is_none() {
            self.compute_harmonic_centrality(normalization)
        } else {
            self.get_hamonic_centrality(normalization)
        }
    }

    /// Gets (from a previously cached result) the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog* counter estimation.
    pub fn get_lins_centrality(&self) -> Result<&[W], Box<dyn std::error::Error>> {
        if let Some(mem) = self.lin.as_ref() {
            return Ok(mem.as_slice());
        }
        Err("no such cached centrality".into())
    }

    /// Searches cache for the approximation of each node's *Lin's Centrality* from their respective distance accumulator.
    pub fn search_cache_lins_centrality(&mut self) -> Result<&[W], Box<dyn std::error::Error>> {
        if let Ok(mem) =
            self.get_from_cent_file(Self::centrality_normalization(Centrality::Lin, false))
        {
            self.lin = Some(mem);
            if let Some(mem) = self.lin.as_ref() {
                return Ok(mem.as_slice());
            }
        }
        Err("error no such cached centrality".into())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog* counter estimation.
    pub fn get_or_compute_lins_centrality(&mut self) -> Result<&[W], Box<dyn std::error::Error>> {
        if self.lin.is_none() {
            self.compute_lins_centrality()
        } else {
            self.get_lins_centrality()
        }
    }

    /// Computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn compute_closeness_centrality(
        &mut self,
        normalize: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn = self.centrality_cache_file_name(Self::centrality_normalization(
            Centrality::Closeness,
            normalize,
        ))?;
        let mut mem = SharedSliceMut::<W>::abst_mem_mut(&c_fn, node_count, true)?;

        // normalized by number of reacheable nodes
        if normalize {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = W::zero();
                } else {
                    *mem.get_mut(idx) = (W::from_f64(self.counters.get(idx).estimate_cardinality())
                        - W::one())
                        / *self.distances.get(idx);
                }
            }
        // normalized by node count (|V| - 1)
        } else {
            let normalize_factor = W::from_usize(node_count) - W::one(); // |V| - 1
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = W::zero();
                } else {
                    *mem.get_mut(idx) = normalize_factor / *self.distances.get(idx);
                }
            }
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        let idx = Self::index_by_normalization(normalize);
        self.closeness[idx] = Some(mem);
        Ok(self.closeness[idx]
            .as_ref()
            .ok_or("closeness not computed")?
            .as_slice())
    }

    /// Computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by total number of nodes, `|V|`.
    ///     - ([`true`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///
    pub fn compute_harmonic_centrality(
        &mut self,
        normalize: bool,
    ) -> Result<&[W], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn = self.centrality_cache_file_name(Self::centrality_normalization(
            Centrality::Harmonic,
            normalize,
        ))?;
        let mut mem = SharedSliceMut::<W>::abst_mem_mut(&c_fn, node_count, true)?;

        // normalized by number of reacheable nodes
        if normalize {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = W::zero();
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx)
                        / (W::from_f64(self.counters.get(idx).estimate_cardinality()) - W::one());
                }
            }
        // normalized by node count (|v| - 1)
        } else {
            let normalize_factor = W::from_usize(node_count) - W::one(); // |V| - 1
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = W::zero();
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx) / normalize_factor;
                }
            }
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        let idx = Self::index_by_normalization(normalize);
        self.harmonic[idx] = Some(mem);
        Ok(self.harmonic[idx]
            .as_ref()
            .ok_or("harmonic not computed")?
            .as_slice())
    }

    /// Computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog* counter estimation.
    pub fn compute_lins_centrality(&mut self) -> Result<&[W], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn = self.centrality_cache_file_name(Centrality::Lin)?;
        let mut mem = SharedSliceMut::<W>::abst_mem_mut(&c_fn, node_count, true)?;

        // lin's centrality is like a closeness centraility doubly normalized by number of
        // reacheable nodes
        for idx in 0..node_count {
            if !self.distances.get(idx).is_normal() {
                *mem.get_mut(idx) = W::one();
            } else {
                // as defined by Boldi P. and Vigna S., reachable nodes include the node itself
                let reachable = W::from_f64(self.counters.get(idx).estimate_cardinality());
                *mem.get_mut(idx) = (reachable * reachable) / *self.distances.get(idx);
            }
        }

        self.lin = Some(mem);
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        Ok(self.lin.as_ref().ok_or("lin's not computed")?.as_slice())
    }

    #[cfg(feature = "bench")]
    pub fn get_iters(&self) -> usize {
        self.iters
    }

    /// Wall time of each iteration of the last computation.
    #[cfg(feature = "bench")]
    pub fn get_iter_times(&self) -> &[std::time::Duration] {
        &self.iter_times
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::HyperBall, None)?;
        let d_fn = this.build_cache_filename(CacheFile::HyperBallDistances, None)?;
        let i_fn = this.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;
        std::fs::remove_file(out_fn)?;
        std::fs::remove_file(d_fn)?;
        std::fs::remove_file(i_fn)?;

        // remove any closeness centralities' cached values
        for (i, t) in [(0, Centrality::NCloseness), (1, Centrality::NCCloseness)] {
            if this.closeness[i].is_some() {
                std::fs::remove_file(this.centrality_cache_file_name(t)?)?;
            }
        }

        // remove any harmonic centralities' cached values
        for (i, t) in [(0, Centrality::NHarmonic), (1, Centrality::NCHarmonic)] {
            if this.harmonic[i].is_some() {
                std::fs::remove_file(this.centrality_cache_file_name(t)?)?;
            }
        }

        // remove any lin's centrality's cached values
        if this.lin.is_some() {
            std::fs::remove_file(this.centrality_cache_file_name(Centrality::Lin)?)?;
        }

        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, P: Precision, W: Weight>
    HyperBallInner<'a, N, E, Ix, P, W>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        max_depth: Option<usize>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, max_depth, threads)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        max_depth: Option<usize>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, max_depth, threads)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<HyperLogLog<P>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<HyperLogLog<P>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<HyperLogLog<P>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(swap)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<HyperLogLog<P>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(swap)
    }

    #[inline(always)]
    fn build_pers_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_pers_cache_filename(file_type, seq)
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
        self.iters
    }

    fn init_cache(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<ProceduralMemoryHB<P, W>, Box<dyn std::error::Error>> {
        let node_count = g.size();

        let c_fn = g.build_cache_filename(CacheFile::HyperBall, None)?;
        let d_fn = g.build_cache_filename(CacheFile::HyperBallDistances, None)?;
        let i_fn = g.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;

        let counters = SharedSliceMut::<HyperLogLog<P>>::abst_mem_mut(
            &c_fn, node_count, true,
        )?;
        let distances = SharedSliceMut::<W>::abst_mem_mut(&d_fn, node_count, true)?;
        let inverse_distances = SharedSliceMut::<W>::abst_mem_mut(&i_fn, node_count, true)?;

        Ok((counters, distances, inverse_distances))
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        max_depth: Option<usize>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threads = threads.max(1);
        let node_count = g.size();
        // make sure depth is within bounds
        let max_t = max_depth.map_or(Self::DEAFULT_MAX_DEPTH, |p| {
            p.clamp(1, Self::MAX_MAX_DEPTH)
        });
        // init cached vecs for distances and inverse distances accumulation
        let (counters, distances, inverse_distances) = Self::init_cache(g)?;
        // init counters: foreach v in 0..=n { add(c[v], v) } & distances and inverse_distances
        let node_load = node_count.div_ceil(threads);
        thread::scope(|scope| {
            for tid in 0..threads {
                let mut counters = counters.shared_slice();
                let mut distances = distances.shared_slice();
                let mut inverse_distances = inverse_distances.shared_slice();
                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                scope.spawn(move |_| {
                    for u in begin..end {
                        *counters.get_mut(u) = HyperLogLog::<P>::with_element(u as u64);
                        *distances.get_mut(u) = W::zero();
                        *inverse_distances.get_mut(u) = W::zero();
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> {
            format!("error initializing HyperBall {:?}", e).into()
        })?;

        Ok(Self {
            g,
            counters,
            distances,
            inverse_distances,
            max_t,
            closeness: [None, None],
            harmonic: [None, None],
            lin: None,
            #[cfg(any(test, feature = "bench"))]
            iters: 0,
            #[cfg(feature = "bench")]
            iter_times: vec![],
            threads,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<HyperLogLog<P>>,
        Box<dyn std::error::Error>,
    > {
        let node_count = self.g.size();

        let s_fn = self.g.build_cache_filename(CacheFile::HyperBall, Some(0))?;
        let swap = SharedSliceMut::<HyperLogLog<P>>::abst_mem_mut(
            &s_fn, node_count, true,
        )?;

        Ok(swap)
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        swap: AbstractedProceduralMemoryMut<HyperLogLog<P>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.threads;
        let node_load = node_count.div_ceil(threads);

        let max_t = self.max_t;

        // one changed counter per iteration parity: iteration `i` accumulates into slot `i % 2`
        // while slot `(i + 1) % 2` (last read during iteration `i - 1`) is reset for the next one
        let global_changed = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let synchronize = Arc::new(Barrier::new(threads));
        // set by a thread that failed, so that all threads stop together at the next barrier
        // instead of waiting on the failed one forever
        let failed = Arc::new(AtomicBool::new(false));

        let counters = self.counters.shared_slice();
        let swap = swap.shared_slice();

        let mut t: W = W::one(); // first iteration is initialization
        let mut inv_t: W = W::one(); // first iteration is initialization
        let mut changed = 0;

        #[cfg(feature = "bench")]
        let iters = Arc::new(AtomicUsize::new(0));
        #[cfg(feature = "bench")]
        let iter_times = std::sync::Mutex::new(Vec::<std::time::Duration>::new());

        thread::scope(
            |scope| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let mut handles = Vec::with_capacity(threads);
                for tid in 0..threads {
                    let graph = &self.g;

                    let mut counters = counters;
                    let mut swap = swap;
                    let mut distance_accumulator = self.distances.shared_slice();
                    let mut inverse_distance_accumulator = self.inverse_distances.shared_slice();

                    let global_changed = Arc::clone(&global_changed);
                    let synchronize = Arc::clone(&synchronize);
                    let failed = Arc::clone(&failed);
                    #[cfg(feature = "bench")]
                    let iters = iters.clone();
                    #[cfg(feature = "bench")]
                    let iter_times = &iter_times;

                    let begin = std::cmp::min(tid * node_load, node_count);
                    let end = std::cmp::min(begin + node_load, node_count);
                    handles.push(scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            let mut error = None;
                            let mut iter: usize = 0;
                            // `counters` initially aliases `self.counters`, results are written
                            // to `swap`, after each iteration both are swapped
                            let mut swapped = false;
                            loop {
                                #[cfg(feature = "bench")]
                                let iter_start = std::time::Instant::now();
                                #[cfg(feature = "bench")]
                                if tid == 0 {
                                    iters.add(1, Ordering::Relaxed);
                                }
                                for u in begin..end {
                                    let mut a: HyperLogLog<P> = *counters.get(u);

                                    let u_n = match graph.neighbours(u) {
                                        Ok(u_n) => u_n,
                                        Err(e) => {
                                            error = Some(format!(
                                                "couldn't find neighbours of {u}: {e}"
                                            ));
                                            failed.store(true, Ordering::Relaxed);
                                            break;
                                        }
                                    };

                                    let mut ball_changed = false;
                                    for v in u_n {
                                        ball_changed |= a.union(counters.get(v));
                                    }

                                    // unchanged registers imply an unchanged estimate
                                    if ball_changed {
                                        let count_diff = W::from_f64(
                                            a.estimate_cardinality()
                                                - counters.get(u).estimate_cardinality(),
                                        );
                                        if count_diff > W::zero() {
                                            // update accumulators
                                            *distance_accumulator.get_mut(u) += t * count_diff;
                                            *inverse_distance_accumulator.get_mut(u) +=
                                                inv_t * count_diff;
                                        }
                                        // update local changed count
                                        changed += 1;
                                    }

                                    *swap.get_mut(u) = a;
                                }

                                global_changed[iter % 2].fetch_add(changed, Ordering::Relaxed);
                                synchronize.wait();

                                #[cfg(feature = "bench")]
                                if tid == 0 {
                                    iter_times.lock().unwrap().push(iter_start.elapsed());
                                }

                                if failed.load(Ordering::Relaxed) {
                                    break match error {
                                        Some(e) => Err(e.into()),
                                        None => Ok(()),
                                    };
                                }

                                iter += 1;
                                if global_changed[(iter - 1) % 2].load(Ordering::Relaxed) == 0
                                    || iter >= max_t
                                {
                                    // newest counters are in `swap`, make sure they end up in
                                    // `self.counters` (only this thread's node range)
                                    if !swapped {
                                        for u in begin..end {
                                            *counters.get_mut(u) = *swap.get(u);
                                        }
                                    }
                                    break Ok(());
                                }

                                swap = std::mem::replace(&mut counters, swap);
                                swapped = !swapped;
                                changed = 0;
                                t += W::one();
                                inv_t = W::one() / t;

                                if tid == 0 {
                                    global_changed[iter % 2].store(0, Ordering::Relaxed);
                                }

                                synchronize.wait();
                            }
                        },
                    ));
                }
                for (tid, handle) in handles.into_iter().enumerate() {
                    handle
                        .join()
                        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("thread {tid} panicked: {:?}", e).into()
                        })??;
                }
                Ok(())
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        #[cfg(feature = "bench")]
        {
            self.iters = iters.load(Ordering::Relaxed);
            self.iter_times = iter_times.into_inner().unwrap();
        }

        // results stay in the page cache (visible to later processes) whether or not they reach
        // the disk, so writeback is scheduled rather than waited for
        self.counters.flush_async()?;
        self.distances.flush_async()?;
        self.inverse_distances.flush_async()?;
        self.g.cleanup_cache(CacheFile::HyperBall)?;

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
    /// every other node in exactly one hop), used to sanity-check `HyperBall`'s distance/
    /// centrality accumulation across `Ix` widths and `W` (weight) precisions.
    fn write_clique_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating clique mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "4 4 6").unwrap();
        for &(u, v) in &[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
            writeln!(f, "{u} {v}").unwrap();
        }
        f.flush().unwrap();
    }

    /// A 10-node path `1 - 2 - ... - 10`: its diameter (9) exceeds the iteration caps used below,
    /// so a capped run must stop before convergence.
    fn write_path_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating path mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "10 10 9").unwrap();
        for u in 1..10 {
            writeln!(f, "{u} {}", u + 1).unwrap();
        }
        f.flush().unwrap();
    }

    fn run_hyperball<Ix: graph::IndexType, W: Weight>(
        path: &std::path::Path,
        id: &str,
    ) -> (Vec<f64>, usize) {
        let graph = GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::from_file(
            path,
            Some(id.to_string()),
            None,
        )
        .expect("graph parse should succeed");
        let hb = HyperBallInner::<VoidLabel, VoidLabel, Ix, Precision8, W>::new(&graph)
            .unwrap_or_else(|e| panic!("hyperball({id}) should succeed: {e}"));
        let distances: Vec<f64> = hb.geodesic_distances().iter().map(|d| d.to_f64()).collect();
        (distances, graph.size())
    }

    /// Runs `HyperBall` across several `Ix` widths (with `W` fixed to `f64`) on a small clique,
    /// asserting every width produces the exact same (bit-for-bit identical) geodesic distances
    /// --- `Ix`'s width only affects indexing/storage, never the floating-point computation
    /// itself, so narrower `Ix` must never change the result.
    #[test]
    fn hyperball_cross_ix_widths_agree() {
        let dir = unique_test_dir("hyperball_cross_ix_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        let (usize_dist, node_count) =
            run_hyperball::<usize, f64>(&path, &format!("hb_usize_{run_tag}"));
        let (u32_dist, _) = run_hyperball::<u32, f64>(&path, &format!("hb_u32_{run_tag}"));
        let (u16_dist, _) = run_hyperball::<u16, f64>(&path, &format!("hb_u16_{run_tag}"));

        assert_eq!(usize_dist, u32_dist, "Ix=u32 distances should match Ix=usize exactly");
        assert_eq!(usize_dist, u16_dist, "Ix=u16 distances should match Ix=usize exactly");

        // sanity: in a 4-clique every node reaches the 3 others in exactly one hop each, so each
        // node's total geodesic distance should be close to 3 (subject to HyperLogLog's own
        // cardinality-estimation error, independent of `Ix`/`W`).
        for (v, &d) in usize_dist.iter().enumerate() {
            assert!(
                d.is_finite() && d > 0.0,
                "node {v} should have a positive, finite geodesic distance, got {d}"
            );
        }
        assert_eq!(node_count, 4);
    }

    /// Runs `HyperBall` with `W=f32` vs `W=f64` on the same clique, asserting the resulting
    /// geodesic distances agree closely (not necessarily bit-for-bit, since `f32`/`f64` differ in
    /// precision, but the accumulation itself is pure-addition/division --- see `compute_closeness
    /// _centrality`'s module docs --- so no catastrophic-cancellation risk exists here, unlike
    /// GVE-Louvain's `delta_q`; small graphs in particular should match almost exactly).
    #[test]
    fn hyperball_f32_and_f64_agree_closely() {
        let dir = unique_test_dir("hyperball_cross_weight_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("clique.mtx");
        write_clique_mtx(&path);

        let (f64_dist, _) = run_hyperball::<usize, f64>(&path, &format!("hb_f64_{run_tag}"));
        let (f32_dist, _) = run_hyperball::<usize, f32>(&path, &format!("hb_f32_{run_tag}"));

        assert_eq!(f64_dist.len(), f32_dist.len());
        for (v, (&d64, &d32)) in f64_dist.iter().zip(f32_dist.iter()).enumerate() {
            assert!(
                (d64 - d32).abs() < 1e-2,
                "node {v}: f64 distance {d64} and f32 distance {d32} should be close"
            );
        }
    }

    /// Runs `HyperBall` on a path with `max_depth` caps below its diameter, asserting the run
    /// stops after exactly `max_depth` iterations and that the persisted counters hold the newest
    /// iteration's balls (whichever of the two swapped buffers they were last written to).
    #[test]
    fn hyperball_respects_max_depth() {
        let dir = unique_test_dir("hyperball_max_depth_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("path.mtx");
        write_path_mtx(&path);

        // odd and even caps end with the newest counters in different buffers
        for max_depth in [2usize, 3] {
            let graph = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                Some(format!("hb_depth{max_depth}_{run_tag}")),
                None,
            )
            .expect("graph parse should succeed");
            let hb = HyperBallInner::<VoidLabel, VoidLabel, usize, Precision8, f64>::new_with_conf(
                &graph,
                Some(max_depth),
                2,
            )
            .unwrap_or_else(|e| panic!("hyperball(max_depth = {max_depth}) should succeed: {e}"));

            // node 0 is the path's end: after `max_depth` iterations its ball holds itself plus
            // `max_depth` nodes, at distances 1..=max_depth
            let ball = hb.counters.get(0).estimate_cardinality();
            assert!(
                (ball - (max_depth + 1) as f64).abs() < 0.5,
                "max_depth {max_depth}: node 0's ball should hold {} nodes, got {ball}",
                max_depth + 1
            );
            let expected = (max_depth * (max_depth + 1) / 2) as f64;
            let d = hb.get_geodesic_distance(0).to_f64();
            assert!(
                (d - expected).abs() < 0.5,
                "max_depth {max_depth}: node 0's distance sum should be {expected}, got {d}"
            );
        }
    }

    /// Writes `edges` (0-based, undirected) as a symmetric MatrixMarket file over `n` nodes.
    fn write_symmetric_mtx(path: &std::path::Path, n: usize, edges: &[(usize, usize)]) {
        let mut f = std::fs::File::create(path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        f.flush().unwrap();
    }

    /// Undirected DNA de Bruijn graph of order `k` (`4^k` nodes, `u -- (4u + c) mod 4^k`, without
    /// self-loops or duplicate edges), the kind of graph *SparX* targets.
    fn de_bruijn_edges(k: u32) -> (usize, Vec<(usize, usize)>) {
        let n = 4usize.pow(k);
        let mut edges = std::collections::BTreeSet::new();
        for u in 0..n {
            for c in 0..4 {
                let v = (4 * u + c) % n;
                if u != v {
                    edges.insert((u.min(v), u.max(v)));
                }
            }
        }
        (n, edges.into_iter().collect())
    }

    /// `side x side` grid: a large diameter (`2 * (side - 1)`), hence many iterations.
    fn grid_edges(side: usize) -> (usize, Vec<(usize, usize)>) {
        let mut edges = vec![];
        for r in 0..side {
            for c in 0..side {
                let u = r * side + c;
                if c + 1 < side {
                    edges.push((u, u + 1));
                }
                if r + 1 < side {
                    edges.push((u, u + side));
                }
            }
        }
        (side * side, edges)
    }

    /// Exact (sum of distances, sum of inverse distances, reachable nodes including itself) for
    /// every node, by BFS.
    fn exact_balls(n: usize, edges: &[(usize, usize)]) -> Vec<(f64, f64, f64)> {
        let mut adj = vec![vec![]; n];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        (0..n)
            .map(|s| {
                let mut dist = vec![usize::MAX; n];
                let mut queue = std::collections::VecDeque::from([s]);
                dist[s] = 0;
                let (mut sum, mut inv, mut reachable) = (0., 0., 1.);
                while let Some(u) = queue.pop_front() {
                    for &v in &adj[u] {
                        if dist[v] == usize::MAX {
                            dist[v] = dist[u] + 1;
                            sum += dist[v] as f64;
                            inv += 1. / dist[v] as f64;
                            reachable += 1.;
                            queue.push_back(v);
                        }
                    }
                }
                (sum, inv, reachable)
            })
            .collect()
    }

    /// Mean relative error of `approx` w.r.t. `exact`.
    fn mean_relative_error(approx: impl Iterator<Item = f64>, exact: impl Iterator<Item = f64>) -> f64 {
        let (mut total, mut count) = (0., 0.);
        for (a, e) in approx.zip(exact) {
            total += (a - e).abs() / e;
            count += 1.;
        }
        total / count
    }

    /// Runs `HyperBall` with precision `P` and returns the mean relative errors of the distance
    /// sums, inverse distance sums, closeness (normalized by reachable nodes) and Lin's centrality
    /// w.r.t. their exact values.
    fn hyperball_errors<P: Precision>(
        path: &std::path::Path,
        id: &str,
        exact: &[(f64, f64, f64)],
    ) -> [f64; 4] {
        let graph = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
            path,
            Some(id.to_string()),
            Some(4),
        )
        .expect("graph parse should succeed");
        let mut hb = HyperBallInner::<VoidLabel, VoidLabel, usize, P, f64>::new(&graph)
            .unwrap_or_else(|e| panic!("hyperball({id}) should succeed: {e}"));
        let distances = mean_relative_error(
            hb.geodesic_distances().iter().copied(),
            exact.iter().map(|e| e.0),
        );
        let inverse = mean_relative_error(
            hb.harmonic_geodesic_distances().iter().copied(),
            exact.iter().map(|e| e.1),
        );
        let closeness = mean_relative_error(
            hb.compute_closeness_centrality(true).unwrap().to_vec().into_iter(),
            exact.iter().map(|e| (e.2 - 1.) / e.0),
        );
        let lin = mean_relative_error(
            hb.compute_lins_centrality().unwrap().to_vec().into_iter(),
            exact.iter().map(|e| e.2 * e.2 / e.0),
        );
        hb.drop_cache().expect("drop cache should succeed");
        [distances, inverse, closeness, lin]
    }

    /// Runs `HyperBall` at every precision on a de Bruijn graph and a grid, asserting the mean
    /// relative errors of the distance sums, inverse distance sums, closeness and Lin's
    /// centrality stay within a small multiple of *HyperLogLog*'s standard error
    /// (`1.04 / sqrt(2^p)`).
    #[test]
    fn hyperball_matches_exact_across_precisions() {
        let dir = unique_test_dir("hyperball_precisions_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();

        for (name, (n, edges)) in [("debruijn", de_bruijn_edges(5)), ("grid", grid_edges(24))] {
            let path = dir.join(format!("{name}.mtx"));
            write_symmetric_mtx(&path, n, &edges);
            let exact = exact_balls(n, &edges);

            macro_rules! check {
                ($($p:ident),*) => {$(
                    let errors = hyperball_errors::<$p>(
                        &path,
                        &format!("hb_{name}_{}_{run_tag}", stringify!($p)),
                        &exact,
                    );
                    // all nodes of a small connected graph end with the same ball (hence the
                    // same estimation error), so errors don't average out across nodes
                    let std_error = 1.04 / ((1usize << $p::EXPONENT) as f64).sqrt();
                    for (what, error) in ["distances", "inverse distances", "closeness", "lin"]
                        .into_iter()
                        .zip(errors)
                    {
                        assert!(
                            error <= 2.5 * std_error,
                            "{name} {}: mean relative error of {what} {error} above 2.5 * {std_error}",
                            stringify!($p)
                        );
                    }
                )*};
            }
            check!(
                Precision4, Precision5, Precision6, Precision7, Precision8, Precision9,
                Precision10, Precision11, Precision12, Precision13, Precision14, Precision15,
                Precision16
            );
        }
    }

    fn clique_graph(id: &str) -> GraphMemoryMap<VoidLabel, VoidLabel, usize> {
        let dir = unique_test_dir("hyperball_clique_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{id}.mtx"));
        write_clique_mtx(&path);
        GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id.to_string()), None)
            .expect("graph parse should succeed")
    }

    /// On a 4-clique every node reaches 3 others at distance 1, so: closeness (either
    /// normalization) = 3 / 3, harmonic (either normalization) = 3 / 3 and Lin's centrality =
    /// 4^2 / 3 (reachable nodes count the node itself). Precision 16 makes estimates of such
    /// small balls practically exact.
    #[test]
    fn hyperball_centralities_on_clique() {
        let graph = clique_graph(&format!("hb_clique_cent_{}", unique_run_tag()));
        let mut hb =
            HyperBallInner::<VoidLabel, VoidLabel, usize, Precision16, f64>::new(&graph).unwrap();
        let close = |values: &[f64], expected: f64, what: &str| {
            for (v, &x) in values.iter().enumerate() {
                assert!((x - expected).abs() < 1e-2, "{what} of node {v}: {x}, expected {expected}");
            }
        };

        // computing harmonic first used to make `get_or_compute_closeness_centrality` look up
        // the harmonic cache, and fail
        close(&hb.compute_harmonic_centrality(true).unwrap().to_vec(), 1., "harmonic");
        close(&hb.get_or_compute_closeness_centrality(true).unwrap().to_vec(), 1., "closeness");
        close(&hb.get_or_compute_closeness_centrality(false).unwrap().to_vec(), 1., "closeness");
        close(&hb.get_or_compute_harmonic_centrality(false).unwrap().to_vec(), 1., "harmonic");
        close(&hb.compute_lins_centrality().unwrap().to_vec(), 16. / 3., "lin");

        // every centrality file must be removed, normalized closeness (`NCCloseness`) and
        // harmonic files included
        let files: Vec<String> = [
            Centrality::NCloseness,
            Centrality::NCCloseness,
            Centrality::NHarmonic,
            Centrality::NCHarmonic,
            Centrality::Lin,
        ]
        .into_iter()
        .map(|c| hb.centrality_cache_file_name(c).unwrap())
        .collect();
        for f in &files {
            assert!(Path::new(f).exists(), "{f} should exist before drop_cache");
        }
        hb.drop_cache().expect("drop cache should succeed");
        for f in &files {
            assert!(!Path::new(f).exists(), "{f} should have been removed by drop_cache");
        }
    }

    /// A cached result of another precision must be recomputed, not reinterpreted --- a larger
    /// precision's file length is a multiple of a smaller precision's counter size, so it would
    /// otherwise map fine (as more, garbage, counters).
    #[test]
    fn hyperball_get_or_compute_ignores_other_precisions_cache() {
        let graph = clique_graph(&format!("hb_clique_cache_{}", unique_run_tag()));
        HyperBallInner::<VoidLabel, VoidLabel, usize, Precision8, f64>::new(&graph).unwrap();
        let mut hb =
            HyperBallInner::<VoidLabel, VoidLabel, usize, Precision4, f64>::get_or_compute(&graph)
                .unwrap();
        assert_eq!(hb.counters.as_slice().len(), graph.size());
        // every node's ball is the whole clique
        let mut clique = HyperLogLog::<Precision4>::default();
        (0..4).for_each(|v| clique.insert(v));
        for v in 0..graph.size() {
            assert_eq!(*hb.counters.get(v), clique, "node {v}'s ball should be the whole clique");
        }
        hb.drop_cache().expect("drop cache should succeed");
    }

    /// `HyperBall`'s centralities and the exact ones (`crate::centralities::exact`) must follow
    /// the same conventions (normalizations, reachable nodes counting the node itself, isolated
    /// nodes): at precision 16 every value matches the exact one closely.
    #[test]
    fn hyperball_agrees_with_exact_centralities() {
        use crate::centralities::exact::*;
        let dir = unique_test_dir("hyperball_vs_exact_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();

        for (name, (n, edges)) in [("debruijn", de_bruijn_edges(4)), ("grid", grid_edges(12))] {
            let path = dir.join(format!("{name}.mtx"));
            write_symmetric_mtx(&path, n, &edges);
            let graph = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                Some(format!("hb_vs_exact_{name}_{run_tag}")),
                Some(4),
            )
            .expect("graph parse should succeed");
            let mut hb =
                HyperBallInner::<VoidLabel, VoidLabel, usize, Precision16, f64>::new(&graph)
                    .unwrap();

            let closeness = |n| ExactClosenessCentrality::<_, _, _, f64>::new(&graph, n).unwrap();
            let harmonic = |n| ExactHarmonicCentrality::<_, _, _, f64>::new(&graph, n).unwrap();
            let pairs: [(&str, Vec<f64>, Vec<f64>); 7] = [
                ("distance sums", hb.geodesic_distances().to_vec(), closeness(None).centralities().to_vec()),
                ("inverse distance sums", hb.harmonic_geodesic_distances().to_vec(), harmonic(None).centralities().to_vec()),
                ("closeness / (|V| - 1)", hb.compute_closeness_centrality(false).unwrap().to_vec(), closeness(Some(false)).centralities().to_vec()),
                ("closeness / (r - 1)", hb.compute_closeness_centrality(true).unwrap().to_vec(), closeness(Some(true)).centralities().to_vec()),
                ("harmonic / (|V| - 1)", hb.compute_harmonic_centrality(false).unwrap().to_vec(), harmonic(Some(false)).centralities().to_vec()),
                ("harmonic / (r - 1)", hb.compute_harmonic_centrality(true).unwrap().to_vec(), harmonic(Some(true)).centralities().to_vec()),
                ("lin", hb.compute_lins_centrality().unwrap().to_vec(), ExactLinCentrality::<_, _, _, f64>::new(&graph).unwrap().centralities().to_vec()),
            ];
            for (what, approx, exact) in pairs {
                let error = mean_relative_error(approx.into_iter(), exact.into_iter());
                assert!(error < 0.02, "{name}: {what} mean relative error {error}");
            }
            hb.drop_cache().expect("drop cache should succeed");
        }
    }
}
