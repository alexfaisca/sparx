//! GPU compute backends: `WgpuBackend` (the `gpu` feature: any vendor, through
//! Vulkan/Metal/DX12) and `CudaBackend` (the `nvidia` feature: NVIDIA GPUs, through CUDA).
//!
//! A backend uploads a [`GraphMemoryMap`]'s CSR (offsets and neighbours, as `u32` node ids) once
//! with [`GpuBackend::upload_graph`], after which algorithms run on the device copy. Only the CSR
//! is uploaded --- node and edge labels stay on the host.
//!
//! `Backend` is the backend to use: `CudaBackend` when the `nvidia` feature is enabled (even
//! with `gpu` too), `WgpuBackend` otherwise. `CudaBackend` has CUDA kernels for *HyperBall* and
//! per-edge triangle counts: with the `gpu` feature it hands degree assortativity to wgpu (on the
//! same GPU), and fails it without.
//!
//! Algorithms' device state (e.g. *HyperBall*'s counters) lives in GPU memory for the whole run,
//! so it must fit there: backends check their device's limits up front and fail with an error
//! (rather than run out of memory midway) when it doesn't.
//!
//! [`GraphMemoryMap`]: ../graph/struct.GraphMemoryMap.html#
#![cfg_attr(
    feature = "gpu",
    doc = r#"
# Example

(The README's GPU example, compiled here as it needs this feature.)

```no_run
use sparx::centralities::hyper_ball::{HyperBallInner, Precision6};
use sparx::gpu::{GpuBackend, WgpuBackend};
use sparx::graph::Graph;
use sparx::k_truss::{pkt::AlgoPKT, triangles::Triangles};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let g = Graph::<(), (), u32>::from_file("./datasets/graphs/kmer_V2a.mtx", None, None)?;
    let backend = WgpuBackend::new()?;
    let device_graph = backend.upload_graph(&g)?;

    let mut hyperball =
        HyperBallInner::<_, _, _, Precision6>::new_gpu(&g, &backend, &device_graph, None)?;
    let closeness = hyperball.get_or_compute_closeness_centrality(true)?;
    println!("closeness of node 0: {:.4}", closeness[0]);

    let triangles = Triangles::new_gpu(&g, &backend, &device_graph)?;
    let pkt = AlgoPKT::new_with_triangles(&g, triangles)?;
    println!("max trussness: {:?}", pkt.k_trusses().iter().max());
    Ok(())
}
```
"#
)]
#![cfg_attr(
    feature = "nvidia",
    doc = r#"
# Example (CUDA)

```no_run
use sparx::gpu::{Backend, GpuBackend};
use sparx::graph::Graph;
use sparx::k_truss::triangles::Triangles;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let g = Graph::<(), (), u32>::from_file("./datasets/graphs/kmer_V2a.mtx", None, None)?;
    // `CudaBackend`, as the `nvidia` feature is enabled
    let backend = Backend::new()?;
    let device_graph = backend.upload_graph(&g)?;
    let triangles = Triangles::new_gpu(&g, &backend, &device_graph)?;
    println!("triangles of edge 0: {}", triangles.triangles(0));
    Ok(())
}
```
"#
)]
use crate::graph::{self, GraphMemoryMap};

#[cfg(feature = "nvidia")]
mod cuda_backend;
#[cfg(feature = "gpu")]
mod wgpu_backend;

#[cfg(feature = "nvidia")]
pub use cuda_backend::{CudaBackend, CudaGraph};
#[cfg(feature = "gpu")]
pub use wgpu_backend::{WgpuBackend, WgpuGraph};

/// The GPU backend to use: `CudaBackend` when the `nvidia` feature is enabled (whether or not
/// `gpu` is), `WgpuBackend` otherwise.
#[cfg(feature = "nvidia")]
pub type Backend = CudaBackend;
/// The GPU backend to use: `CudaBackend` when the `nvidia` feature is enabled (whether or not
/// `gpu` is), `WgpuBackend` otherwise.
#[cfg(all(feature = "gpu", not(feature = "nvidia")))]
pub type Backend = WgpuBackend;

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

/// A count a GPU computes in its own width (see [`GpuBackend::triangles`]): `u8`, `u16` or
/// `u32`. Sealed.
pub trait GpuCount: bytemuck::Pod + Send + Sync + private::Sealed {
    /// Width, in bits.
    const BITS: u32;

    /// `self` read from little-endian bytes (as devices store it).
    fn from_le(self) -> Self;
}

mod private {
    pub trait Sealed {}
}

macro_rules! impl_gpu_count {
    ($($t:ty),*) => {$(
        impl private::Sealed for $t {}

        impl GpuCount for $t {
            const BITS: u32 = <$t>::BITS;

            #[inline(always)]
            fn from_le(self) -> Self {
                <$t>::from_le(self)
            }
        }
    )*};
}

impl_gpu_count!(u8, u16, u32);

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
    /// receives every edge's count, counted in `C`'s width on the device (so that narrower
    /// counts move less data): an edge in more triangles than `C` holds fails the count.
    ///
    /// [`Triangles`]: ../k_truss/triangles/struct.Triangles.html#
    /// [`edge_over`]: ../graph/struct.GraphMemoryMap.html#method.edge_over
    /// [`edge_reciprocal`]: ../graph/struct.GraphMemoryMap.html#method.edge_reciprocal
    fn triangles<Ix: graph::IndexType, C: GpuCount>(
        &self,
        graph: &Self::Graph,
        edge_over: &[Ix],
        edge_reciprocal: &[Ix],
        counts: &mut [C],
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

/// `g`'s CSR: its `|V| + 1` offsets and `|E|` neighbours, after checking devices can index them
/// with `u32`s.
pub(crate) fn csr<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Result<(&[Ix], &[Ix]), Box<dyn std::error::Error>> {
    let (nodes, edges) = (g.size(), g.width());
    // offsets index edges, so both must be representable as `u32`
    if nodes >= u32::MAX as usize || edges > u32::MAX as usize {
        return Err(format!(
            "error GPU graphs are indexed by u32: |V| = {nodes} and |E| = {edges} must be \
             smaller than {}",
            u32::MAX
        )
        .into());
    }
    // SAFETY: the graph's index holds `|V| + 1` offsets and its neighbours `|E|` node ids, both
    // mapped for as long as `g` lives
    Ok(unsafe {
        (
            std::slice::from_raw_parts(g.offsets_ptr(), nodes + 1),
            std::slice::from_raw_parts(g.neighbours_ptr(), edges),
        )
    })
}

/// The error of an edge in more triangles than a `C` holds.
pub(crate) fn count_overflow<C: GpuCount>() -> Box<dyn std::error::Error> {
    format!(
        "error an edge is in more than {} triangles, more than can be counted ({})",
        (1u64 << C::BITS) - 1,
        std::any::type_name::<C>()
    )
    .into()
}

/// Completes triangle counts a device computed for upward edges (`u -> v`, `u < v`) only:
/// downward edges are in their reciprocals' triangles. A downward edge's reciprocal, in an earlier
/// node's neighbour list, comes before it (and an upward one's after it), so only downward edges
/// are written and only upward ones read: threads split the edges freely.
pub(crate) fn copy_to_downward_edges<Ix: graph::IndexType, C: Copy + Send + Sync>(
    edge_reciprocal: &[Ix],
    counts: &mut [C],
) {
    struct Counts<C>(*mut C);
    // SAFETY: threads never access the same count, one writing it, at once (see above)
    unsafe impl<C> Send for Counts<C> {}
    unsafe impl<C> Sync for Counts<C> {}
    assert_eq!(edge_reciprocal.len(), counts.len());
    let shared = Counts(counts.as_mut_ptr());
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let chunk = counts.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (c, reciprocals) in edge_reciprocal.chunks(chunk).enumerate() {
            let shared = &shared;
            scope.spawn(move || {
                for (e, r) in (c * chunk..).zip(reciprocals) {
                    let r = r.index();
                    if r < e {
                        // SAFETY: `e` and `r` are edges (below `counts.len()`), `r` upward, only
                        // read, and `e` downward, written by this thread only
                        unsafe { *shared.0.add(e) = *shared.0.add(r) };
                    }
                }
            });
        }
    });
}

/// The CUDA backend tests run on, or `None` (the test then skips) on machines without an NVIDIA
/// GPU, its driver or NVRTC --- unless `SPARX_REQUIRE_CUDA` is set, non-empty, where that fails the
/// test rather than letting it pass unrun.
#[cfg(all(test, feature = "nvidia"))]
pub(crate) fn test_cuda_backend() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(backend) => Some(backend),
        Err(e) if std::env::var_os("SPARX_REQUIRE_CUDA").is_some_and(|v| !v.is_empty()) => {
            panic!("no CUDA device, while SPARX_REQUIRE_CUDA is set: {e}")
        }
        Err(e) => {
            eprintln!("skipping CUDA test, no CUDA device: {e}");
            None
        }
    }
}

/// The GPU backend tests run on, or `None` (the test then skips) on machines without any GPU
/// adapter --- unless `SPARX_REQUIRE_GPU` is set, non-empty (as in CI, which installs a software Vulkan
/// driver), where a missing adapter fails the test rather than letting it pass unrun.
#[cfg(all(test, feature = "gpu"))]
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
