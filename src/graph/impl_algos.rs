use crate::{
    centralities::hyper_ball::HyperBall8,
    communities::{gve_louvain::AlgoGVELouvain, hk_relax::HKRelax},
    k_core::liu_et_al::AlgoLiuEtAl,
    k_truss::{burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT},
    trails::hierholzer::AlgoHierholzer,
};

use super::GraphMemoryMap;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    /// Performs *Hierholzer's DFS* on the graph.
    pub fn euler_trail(&self) -> Result<AlgoHierholzer<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoHierholzer::new(self)
    }
    /// Performs k-core decomposition on the graph.
    pub fn k_core(&self) -> Result<AlgoLiuEtAl<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoLiuEtAl::new(self)
    }
    /// Performs k-core decomposition on the graph using Batagelj & Zavernik's algorithm.
    pub fn k_core_bz(&self) -> Result<AlgoBurkhardtEtAl<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    /// Performs k-core decomposition on the graph using Liu et al's algorithm.
    pub fn k_core_lea(&self) -> Result<AlgoLiuEtAl<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoLiuEtAl::new(self)
    }
    /// Performs k-truss decomposition on the graph.
    pub fn k_truss(&self) -> Result<AlgoBurkhardtEtAl<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    /// Performs k-truss decomposition on the graph using Burkhardt et al's algorithm.
    pub fn k_truss_bea(&self) -> Result<AlgoBurkhardtEtAl<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    /// Performs k-truss decomposition on the graph using the Parallel K-Truss algorithm.
    pub fn k_truss_pkt(&self) -> Result<AlgoPKT<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoPKT::new(self)
    }
    /// Performs *Louvain's Partition Algorithm* on the graph.
    pub fn louvain(&self) -> Result<AlgoGVELouvain<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoGVELouvain::new(self)
    }
    /// Performs *HyperBall* on the graph.
    pub fn hyperball(&self) -> Result<HyperBall8<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        HyperBall8::new(self)
    }
    /// Performs the *Heat Kernel Relax (HK-Relax)* algortihm on the graph.
    ///
    /// # Arguments
    ///
    /// * `seed` --- the diffusion seed node.
    /// * `t` --- the diffusion temperature parameter.
    /// * `eps` --- the diffusion error parameter.
    pub fn hk_relax(
        &self,
        seed: usize,
        t: f64,
        eps: f64,
    ) -> Result<HKRelax<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        HKRelax::new(self, t, eps, vec![seed], None, None)
    }
    /// Performs the *Heat Kernel Relax (HK-Relax)* algortihm on the graph, with multiple diffusion sources.
    ///
    /// # Arguments
    ///
    /// * `seed` --- the diffusion seed nodes.
    /// * `t` --- the diffusion temperature parameter.
    /// * `eps` --- the diffusion error parameter.
    pub fn hk_relax_ms(
        &self,
        seed: &[usize],
        t: f64,
        eps: f64,
    ) -> Result<HKRelax<'_, N, E, Ix>, Box<dyn std::error::Error>> {
        HKRelax::new(self, t, eps, seed.to_vec(), None, None)
    }
}
