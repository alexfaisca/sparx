//! GPU compute backends (enabled by the `gpu` feature).
//!
//! A backend uploads a [`GraphMemoryMap`]'s CSR (offsets and neighbours, as `u32` node ids) once
//! with [`GpuBackend::upload_graph`], after which algorithms run on the device copy. Only the CSR
//! is uploaded --- node and edge labels stay on the host.
//!
//! Algorithms' device state (e.g. *HyperBall*'s counters) lives in GPU memory for the whole run,
//! so it must fit there: backends check their device's limits up front and fail with an error
//! (rather than run out of memory midway) when it doesn't.
//!
//! # Example
//!
//! (The README's GPU example, compiled here as it needs this feature.)
//!
//! ```no_run
//! use sparx::centralities::hyper_ball::{HyperBallInner, Precision6};
//! use sparx::gpu::{GpuBackend, WgpuBackend};
//! use sparx::graph::Graph;
//! use sparx::k_truss::{pkt::AlgoPKT, triangles::Triangles};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let g = Graph::<(), (), u32>::from_file("./datasets/graphs/kmer_V2a.mtx", None, None)?;
//!     let backend = WgpuBackend::new()?;
//!     let device_graph = backend.upload_graph(&g)?;
//!
//!     let mut hyperball =
//!         HyperBallInner::<_, _, _, Precision6>::new_gpu(&g, &backend, &device_graph, None)?;
//!     let closeness = hyperball.get_or_compute_closeness_centrality(true)?;
//!     println!("closeness of node 0: {:.4}", closeness[0]);
//!
//!     let triangles = Triangles::new_gpu(&g, &backend, &device_graph)?;
//!     let pkt = AlgoPKT::new_with_triangles(&g, triangles)?;
//!     println!("max trussness: {:?}", pkt.k_trusses().iter().max());
//!     Ok(())
//! }
//! ```
//!
//! [`GraphMemoryMap`]: ../graph/struct.GraphMemoryMap.html#
use crate::graph::{self, GraphMemoryMap};

mod wgpu_backend;

pub use wgpu_backend::{WgpuBackend, WgpuGraph};

/// Configuration of a *HyperBall* run, independent of the counters' Rust type.
#[derive(Clone, Copy, Debug)]
pub struct HyperBallGpuParams {
    /// Registers per counter (`2^precision`).
    pub registers: usize,
    /// Largest register value (`64 - precision + 1`).
    pub max_rank: u8,
    /// Maximum number of iterations.
    pub max_iterations: usize,
}

/// A GPU compute backend.
pub trait GpuBackend {
    /// A graph's CSR in device memory.
    type Graph;

    /// Copies `g`'s CSR to device memory, to be reused by any number of algorithm runs.
    fn upload_graph<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>(
        &self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<Self::Graph, Box<dyn std::error::Error>>;

    /// Runs *HyperBall* on `graph`.
    ///
    /// `counters` holds every node's *HyperLogLog* counter (`params.registers` bytes each, node
    /// after node) and is updated in place to the final balls. `accumulators` receives every
    /// node's (sum of distances, sum of inverse distances). Returns the number of iterations
    /// performed (the last one being the first that changed no counter, unless
    /// `params.max_iterations` was reached).
    fn hyperball(
        &self,
        graph: &Self::Graph,
        params: HyperBallGpuParams,
        counters: &mut [u8],
        accumulators: &mut [[f32; 2]],
    ) -> Result<usize, Box<dyn std::error::Error>>;

    /// Counts, for every edge of `graph`, the triangles it is in (once per combination of
    /// parallel edges), as [`Triangles`] does.
    ///
    /// `edge_over` and `edge_reciprocal` are the graph's [`edge_over`] and [`edge_reciprocal`]
    /// (neighbour lists must be sorted, and every edge must have a reciprocal). `counts`
    /// receives every edge's count.
    ///
    /// [`Triangles`]: ../k_truss/triangles/struct.Triangles.html#
    /// [`edge_over`]: ../graph/struct.GraphMemoryMap.html#method.edge_over
    /// [`edge_reciprocal`]: ../graph/struct.GraphMemoryMap.html#method.edge_reciprocal
    fn triangles<Ix: graph::IndexType>(
        &self,
        graph: &Self::Graph,
        edge_over: &[Ix],
        edge_reciprocal: &[Ix],
        counts: &mut [u32],
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// The sum, over every arc `u -> v` of `graph` (each parallel arc counting), of
    /// `x[u] * y[v]`: the only sum over arcs of the [assortativity] coefficients'. Computed
    /// exactly: `y`'s values are summed by node in 64 bits, the sums multiplied by `x`'s in 96 and
    /// added up in 128.
    ///
    /// [assortativity]: ../assortativity/index.html
    fn arc_products(
        &self,
        graph: &Self::Graph,
        x: &[u32],
        y: &[u32],
    ) -> Result<u128, Box<dyn std::error::Error>>;
}

/// The GPU backend tests run on, or `None` (the test then skips) on machines without any GPU
/// adapter --- unless `SPARX_REQUIRE_GPU` is set, non-empty (as in CI, which installs a software Vulkan
/// driver), where a missing adapter fails the test rather than letting it pass unrun.
#[cfg(test)]
pub(crate) fn test_backend() -> Option<WgpuBackend> {
    match WgpuBackend::new() {
        Ok(backend) => Some(backend),
        Err(e) if std::env::var_os("SPARX_REQUIRE_GPU").is_some_and(|v| !v.is_empty()) => {
            panic!("no GPU adapter, while SPARX_REQUIRE_GPU is set: {e}")
        }
        Err(e) => {
            eprintln!("skipping GPU test, no GPU adapter: {e}");
            None
        }
    }
}
