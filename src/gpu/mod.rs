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
}
