use crate::graph::{self, *};
use crate::shared_slice::*;

use crossbeam::thread;
use hyperloglog_rs::prelude::WordType;
use hyperloglog_rs::prelude::{HyperLogLog, HyperLogLogTrait};
use portable_atomic::{AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Barrier};

// Re-export precision consts from hyperloglog_rs  for UX enhancement and portability
/// For HyperLogLog++ registers with 2⁴ bits.
pub use hyperloglog_rs::prelude::Precision4;
/// For HyperLogLog++ registers with 2⁵ bits.
pub use hyperloglog_rs::prelude::Precision5;
/// For HyperLogLog++ registers with 2⁶ bits.
pub use hyperloglog_rs::prelude::Precision6;
/// For HyperLogLog++ registers with 2⁷ bits.
pub use hyperloglog_rs::prelude::Precision7;
/// For HyperLogLog++ registers with 2⁸ bits.
pub use hyperloglog_rs::prelude::Precision8;
/// For HyperLogLog++ registers with 2⁹ bits.
pub use hyperloglog_rs::prelude::Precision9;
/// For HyperLogLog++ registers with 2¹⁰ bits.
pub use hyperloglog_rs::prelude::Precision10;
/// For HyperLogLog++ registers with 2¹¹ bits.
pub use hyperloglog_rs::prelude::Precision11;
/// For HyperLogLog++ registers with 2¹² bits.
pub use hyperloglog_rs::prelude::Precision12;
/// For HyperLogLog++ registers with 2¹³ bits.
pub use hyperloglog_rs::prelude::Precision13;
/// For HyperLogLog++ registers with 2¹⁴ bits.
pub use hyperloglog_rs::prelude::Precision14;
/// For HyperLogLog++ registers with 2¹⁵ bits.
pub use hyperloglog_rs::prelude::Precision15;
/// For HyperLogLog++ registers with 2¹⁶ bits.
pub use hyperloglog_rs::prelude::Precision16;

type ProceduralMemoryHB<P, const B: usize, W> = (
    AbstractedProceduralMemoryMut<HyperLogLog<P, B>>,
    AbstractedProceduralMemoryMut<W>,
    AbstractedProceduralMemoryMut<W>,
);

pub type HyperBall4<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision4, 6>;
pub type HyperBall5<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision5, 6>;
pub type HyperBall6<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision6, 6>;
pub type HyperBall7<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision7, 6>;
pub type HyperBall8<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision8, 6>;
pub type HyperBall9<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision9, 6>;
pub type HyperBall10<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision10, 6>;
pub type HyperBall11<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision11, 6>;
pub type HyperBall12<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision12, 6>;
pub type HyperBall13<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision13, 6>;
pub type HyperBall14<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision14, 6>;
pub type HyperBall15<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision15, 6>;
pub type HyperBall16<'a, N, E, Ix> = HyperBallInner<'a, N, E, Ix, Precision16, 6>;

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
    P: WordType<B> = Precision8,
    const B: usize = 6,
    W: Weight = f64,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's *HyperLogLog++* counter.
    counters: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    /// Memmapped slice containing each node's distance accumulator.
    distances: AbstractedProceduralMemoryMut<W>,
    /// Memmapped slice containing each node's inverse distance accumulator.
    inverse_distances: AbstractedProceduralMemoryMut<W>,
    /// Maximum number of iteration --- default is 128, max is 1024 (details on how these values are ludicrously big are found in the abovementioned paper).
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
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, P: WordType<B>, const B: usize, W: Weight>
    HyperBallInner<'a, N, E, Ix, P, B, W>
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
    /// * `max_depth` --- the maximum number of iterations of the *HyperBall Algorithm* to tolerate before convergence is achieved (defaults to 128, max is 1024).
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
        let h_fn = g.build_cache_filename(CacheFile::HyperBall, None)?;
        if Path::new(&h_fn).exists() {
            if let Ok(counters) = AbstractedProceduralMemoryMut::from_file_name(&h_fn) {
                let d_fn = g.build_cache_filename(CacheFile::HyperBallDistances, None)?;
                let id_fn = g.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;
                if let Ok(distances) = AbstractedProceduralMemoryMut::from_file_name(&d_fn) {
                    if let Ok(inverse_distances) =
                        AbstractedProceduralMemoryMut::from_file_name(&id_fn)
                    {
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

    fn get_from_cent_file(
        &self,
        target: Centrality,
    ) -> Result<AbstractedProceduralMemoryMut<W>, Box<dyn std::error::Error>> {
        let t_fn = self.centrality_cache_file_name(target)?;
        if Path::new(&t_fn).exists() {
            AbstractedProceduralMemoryMut::from_file_name(&t_fn)
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
    pub fn geodesic_distances(&'a self) -> &'a [W] {
        self.distances.as_slice()
    }

    /// Gets (from a previously cached result) the approximation of each node's total inverse distance to other nodes from their respective inverse distance accumulator.
    pub fn harmonic_geodesic_distances(&'a self) -> &'a [W] {
        self.inverse_distances.as_slice()
    }

    /// Gets (from a previously cached result) the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_closeness_centrality(
        &'a self,
        normalization: bool,
    ) -> Result<&'a [W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if let Some(mem) = self.closeness[idx].as_ref() {
            return Ok(mem.as_slice());
        }
        Err("error no such cached centrality".into())
    }

    /// Searches cache for the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
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

    /// Gets (from a previously cached result) or computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_or_compute_closeness_centrality(
        &'a mut self,
        normalization: bool,
    ) -> Result<&'a [W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.harmonic[idx].is_none() {
            self.compute_closeness_centrality(normalization)
        } else {
            self.get_closeness_centrality(normalization)
        }
    }

    /// Gets (from a previously cached result) the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_hamonic_centrality(
        &'a self,
        normalization: bool,
    ) -> Result<&'a [W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if let Some(mem) = self.harmonic[idx].as_ref() {
            return Ok(mem.as_slice());
        }
        Err("no such cached centrality".into())
    }

    /// Searches cache for the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
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

    /// Gets (from a previously cached result) or computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_or_compute_harmonic_centrality(
        &'a mut self,
        normalization: bool,
    ) -> Result<&'a [W], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.harmonic[idx].is_none() {
            self.compute_harmonic_centrality(normalization)
        } else {
            self.get_hamonic_centrality(normalization)
        }
    }

    /// Gets (from a previously cached result) the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
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

    /// Gets (from a previously cached result) or computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
    pub fn get_or_compute_lins_centrality(&mut self) -> Result<&[W], Box<dyn std::error::Error>> {
        if self.lin.is_none() {
            self.compute_lins_centrality()
        } else {
            self.get_lins_centrality()
        }
    }

    /// Computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn compute_closeness_centrality(
        &'a mut self,
        normalize: bool,
    ) -> Result<&'a [W], Box<dyn std::error::Error>> {
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
                    *mem.get_mut(idx) = (W::from_f32(self.counters.get_mut(idx).estimate_cardinality())
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

    /// Computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- flag determining the type of normalization that was used for the computation.
    ///     - ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
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
                        / (W::from_f32(self.counters.get_mut(idx).estimate_cardinality()) - W::one());
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

    /// Computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
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
                let reachable_minus_one =
                    W::from_f32(self.counters.get_mut(idx).estimate_cardinality()) - W::one();
                *mem.get_mut(idx) =
                    (reachable_minus_one * reachable_minus_one) / *self.distances.get(idx);
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
        for (i, t) in [(0, Centrality::NCloseness), (1, Centrality::NCloseness)] {
            if this.closeness[i].is_some() {
                std::fs::remove_file(this.centrality_cache_file_name(t)?)?;
            }
        }

        // remove any harmonic centralities' cached values
        for (i, t) in [(0, Centrality::NHarmonic), (1, Centrality::NCHarmonic)] {
            if this.closeness[i].is_some() {
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
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, P: WordType<B>, const B: usize, W: Weight>
    HyperBallInner<'a, N, E, Ix, P, B, W>
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
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(swap)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
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
    ) -> Result<ProceduralMemoryHB<P, B, W>, Box<dyn std::error::Error>> {
        let node_count = g.size();

        let c_fn = g.build_cache_filename(CacheFile::HyperBall, None)?;
        let d_fn = g.build_cache_filename(CacheFile::HyperBallDistances, None)?;
        let i_fn = g.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;

        let counters = SharedSliceMut::<hyperloglog_rs::prelude::HyperLogLog<P, B>>::abst_mem_mut(
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
            std::cmp::max(Self::MAX_MAX_DEPTH, p)
        });
        // init cached vecs for distances and inverse distances accumulation
        let (mut counters, mut distances, mut inverse_distances) = Self::init_cache(g)?;
        // init counters: foreach v in 0..=n { add(c[v], v) } & distances and inverse_distances
        (0..node_count).for_each(|u| {
            *counters.get_mut(u) = hyperloglog_rs::prelude::HyperLogLog::<P, B>::default();
            counters.get_mut(u).insert(u);
            *distances.get_mut(u) = W::zero();
            *inverse_distances.get_mut(u) = W::zero();
        });

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
            threads,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        let node_count = self.g.size();

        let s_fn = self.g.build_cache_filename(CacheFile::HyperBall, Some(0))?;
        let swap = SharedSliceMut::<hyperloglog_rs::prelude::HyperLogLog<P, B>>::abst_mem_mut(
            &s_fn, node_count, true,
        )?;

        Ok(swap)
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.threads;
        let node_load = node_count.div_ceil(threads);

        let global_changed = Arc::new(AtomicUsize::new(0));
        let synchronize = Arc::new(Barrier::new(threads));

        let counters = self.counters.shared_slice();
        let swap = swap.shared_slice();

        let mut t: W = W::one(); // first iteration is initialization
        let mut inv_t: W = W::one(); // first iteration is initialization
        let mut changed = 0;

        #[cfg(feature = "bench")]
        let iters = Arc::new(AtomicUsize::new(0));

        thread::scope(
            |scope| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                for tid in 0..threads {
                    let graph = &self.g;

                    let mut counters = counters;
                    let mut swap = swap;
                    let mut distance_accumulator = self.distances.shared_slice();
                    let mut inverse_distance_accumulator = self.inverse_distances.shared_slice();

                    let global_changed = Arc::clone(&global_changed);
                    let synchronize = Arc::clone(&synchronize);
                    #[cfg(feature = "bench")]
                    let iters = iters.clone();

                    let begin = std::cmp::min(tid * node_load, node_count);
                    let end = std::cmp::min(begin + node_load, node_count);
                    scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            loop {
                                #[cfg(feature = "bench")]
                                if tid == 0 {
                                    iters.add(1, Ordering::Relaxed);
                                }
                                {
                                    // println!("HyperBall tid {tid} iteration {t}");
                                }

                                for u in begin..end {
                                    let mut a: HyperLogLog<P, B> = *counters.get_mut(u);

                                    let prev_count = a.estimate_cardinality();

                                    let u_n = graph.neighbours(u).map_err(
                                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                                            format!("couldn't find neighbours of {u}: {e}").into()
                                        },
                                    )?;

                                    for v in u_n {
                                        a |= counters.get(v);
                                    }

                                    let curr_count = a.estimate_cardinality();
                                    let count_diff = W::from_f32(curr_count - prev_count);

                                    if count_diff > W::zero() {
                                        // update accumulators
                                        *distance_accumulator.get_mut(u) += t * count_diff;
                                        *inverse_distance_accumulator.get_mut(u) +=
                                            inv_t * count_diff;
                                        // update local changed count
                                        changed += 1;
                                    }

                                    *swap.get_mut(u) = a;
                                }

                                global_changed.fetch_add(changed, Ordering::Relaxed);
                                synchronize.wait();

                                if changed == 0 && global_changed.load(Ordering::Relaxed) == 0 {
                                    break Ok(());
                                }

                                swap = std::mem::replace(&mut counters, swap);
                                changed = 0;
                                t += W::one();
                                inv_t = W::one() / t;

                                synchronize.wait();

                                if tid == 0 {
                                    global_changed.store(0, Ordering::Relaxed);
                                }
                            }
                        },
                    );
                }
                Ok(())
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        #[cfg(feature = "bench")]
        {
            self.iters = iters.load(Ordering::Relaxed);
        }

        self.counters.flush()?;
        self.g.cleanup_cache(CacheFile::HyperBall)?;

        // for i in 0..node_count {
        //     print!("\t|id: {i} ==> counter: {}|", self.counters.get(i).estimate_cardinality());
        // }
        // println!();

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
        let hb = HyperBallInner::<VoidLabel, VoidLabel, Ix, Precision8, 6, W>::new(&graph)
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
}
