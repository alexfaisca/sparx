//! [`GpuBackend`] on CUDA (NVIDIA GPUs), through [cudarc](https://docs.rs/cudarc): kernels are
//! compiled at runtime by NVRTC, for the device found, so building needs no CUDA toolkit, while
//! running needs the NVIDIA driver and NVRTC (the CUDA toolkit's, or its redistributable).
//!
//! *HyperBall* and per-edge triangle counts have CUDA kernels: with the `gpu` feature, arc products
//! (degree assortativity) are handed to a [`WgpuBackend`](super::WgpuBackend) (created, and given
//! the graph, on first use); without it, they fail.
use super::{GpuBackend, GpuCount, HyperBallGpuParams};
use crate::graph::{self, GraphMemoryMap, IndexType};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Threads per block of every kernel.
const BLOCK_SIZE: u32 = 256;
/// Most blocks launched: kernels loop over the rest of their work (grid-stride loops).
const MAX_BLOCKS: u64 = 1 << 20;

/// A graph's CSR in a CUDA device's memory (see [`GpuBackend::upload_graph`]).
pub struct CudaGraph {
    nodes: usize,
    edges: usize,
    // Offsets and neighbours are `u32`s on the device, whatever the graph's `Ix`: `u32` ids are
    // borrowed as is, wider ones narrowed, and graphs past `u32` rejected (see `super::csr`).
    // Choosing each graph's width at upload instead (`u64` only past the `u32` limit, the kernel
    // compiled for it) would lift that limit at no cost below it, but gains little in practice:
    // a graph over the `u32` limit needs at least 4 billion `u64` neighbours, over 32 GB of GPU
    // memory. That's datacenter-GPU territory, so on a GPU of a few GB (e.g. 6) it only adds
    // headroom that can't be used. Below the limit, the per-graph choice keeps today's
    // performance exactly.
    offsets: CudaSlice<u32>,
    neighbours: CudaSlice<u32>,
    /// The graph in wgpu's device memory, uploaded on its first use.
    #[cfg(feature = "gpu")]
    wgpu: std::sync::OnceLock<super::WgpuGraph>,
}

impl CudaGraph {
    /// Number of nodes of the uploaded graph.
    pub fn size(&self) -> usize {
        self.nodes
    }
}

/// A CUDA device (the first one), its stream, and the kernels compiled for it.
pub struct CudaBackend {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// `--gpu-architecture` for NVRTC: the device's own.
    arch: String,
    /// Compiled kernels, by source file and compile options.
    modules: Mutex<HashMap<(&'static str, Vec<String>), Arc<CudaModule>>>,
    /// The backend non-CUDA algorithms are handed to, created on their first use.
    #[cfg(feature = "gpu")]
    wgpu: std::sync::OnceLock<Result<super::WgpuBackend, String>>,
}

impl CudaBackend {
    /// The first CUDA device: fails if there's none, or the NVIDIA driver or NVRTC can't be
    /// found (NVRTC's library, e.g. `nvrtc64_130_0.dll` or `libnvrtc.so.13`, must be on the
    /// library search path).
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // SAFETY: only tries loading the libraries (rather than panicking when they're missing)
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("error the NVIDIA driver's CUDA library wasn't found".into());
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("error NVRTC (the CUDA toolkit's runtime compiler) wasn't found".into());
        }
        if CudaContext::device_count()? < 1 {
            return Err("error no CUDA device found".into());
        }
        let context = CudaContext::new(0)?;
        let (major, minor) = context.compute_capability()?;
        Ok(Self {
            stream: context.default_stream(),
            context,
            arch: format!("--gpu-architecture=compute_{major}{minor}"),
            modules: Mutex::new(HashMap::new()),
            #[cfg(feature = "gpu")]
            wgpu: std::sync::OnceLock::new(),
        })
    }

    /// The device's name.
    pub fn device_name(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.context.name()?)
    }

    /// Function `function` of `source` (file `file`) compiled with `defines` (`-D` options),
    /// compiled on first use. Floating point operations are left unfused (as in the wgpu
    /// kernels: each rounded on its own).
    fn function(
        &self,
        file: &'static str,
        source: &'static str,
        defines: &[String],
        function: &str,
    ) -> Result<CudaFunction, Box<dyn std::error::Error>> {
        let mut modules = self.modules.lock().map_err(|e| e.to_string())?;
        let key = (file, defines.to_vec());
        let module = match modules.get(&key) {
            Some(module) => module.clone(),
            None => {
                let mut options = vec![self.arch.clone()];
                options.extend(defines.iter().map(|d| format!("-D{d}")));
                let ptx = compile_ptx_with_opts(
                    source,
                    CompileOptions {
                        options,
                        fmad: Some(false),
                        name: Some(file.into()),
                        ..Default::default()
                    },
                )?;
                let module = self.context.load_module(ptx)?;
                modules.insert(key, module.clone());
                module
            }
        };
        Ok(module.load_function(function)?)
    }

    /// Copies node or edge ids (of any index type) to the device, as `u32`s.
    fn upload_ids<Ix: IndexType>(
        &self,
        ids: &[Ix],
    ) -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let ids: Cow<[u32]> = if std::any::TypeId::of::<Ix>() == std::any::TypeId::of::<u32>() {
            // SAFETY: `Ix` is `u32`
            Cow::Borrowed(unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len()) })
        } else {
            Cow::Owned(ids.iter().map(|id| id.index() as u32).collect())
        };
        // (empty arrays can't be allocated)
        if ids.is_empty() {
            return Ok(self.stream.alloc_zeros(1)?);
        }
        Ok(self.stream.clone_htod(ids.as_ref())?)
    }

    /// A launch of `threads` threads (fewer, if more than [`MAX_BLOCKS`] blocks: kernels loop).
    fn launch_config(threads: u64) -> LaunchConfig {
        let blocks = threads.div_ceil(BLOCK_SIZE as u64).clamp(1, MAX_BLOCKS) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// The wgpu backend, and `graph` in its memory, uploaded on first use (from the CUDA copy).
    #[cfg(feature = "gpu")]
    fn wgpu<'a>(
        &'a self,
        graph: &'a CudaGraph,
    ) -> Result<(&'a super::WgpuBackend, &'a super::WgpuGraph), Box<dyn std::error::Error>> {
        let backend = self
            .wgpu
            .get_or_init(|| super::WgpuBackend::new().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| {
                format!("error no wgpu adapter for the CUDA backend to hand over to: {e}")
            })?;
        if graph.wgpu.get().is_none() {
            let offsets = self.stream.clone_dtoh(&graph.offsets)?;
            let neighbours = self.stream.clone_dtoh(&graph.neighbours)?;
            self.stream.synchronize()?;
            let uploaded = backend.upload_csr(&offsets, &neighbours[..graph.edges])?;
            // (another thread may have uploaded it meanwhile: either copy does)
            let _ = graph.wgpu.set(uploaded);
        }
        Ok((backend, graph.wgpu.get().expect("set above")))
    }
}

/// The error of an algorithm without CUDA kernels, when the `gpu` feature is disabled.
#[cfg(not(feature = "gpu"))]
fn without_wgpu(algorithm: &str) -> Box<dyn std::error::Error> {
    format!(
        "error {algorithm} has no CUDA kernels: enable the `gpu` feature for the CUDA backend to \
         run it through wgpu"
    )
    .into()
}

impl GpuBackend for CudaBackend {
    type Graph = CudaGraph;

    fn upload_graph<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>(
        &self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<CudaGraph, Box<dyn std::error::Error>> {
        let (offsets, neighbours) = super::csr(g)?;
        Ok(CudaGraph {
            nodes: offsets.len() - 1,
            edges: neighbours.len(),
            offsets: self.upload_ids(offsets)?,
            neighbours: self.upload_ids(neighbours)?,
            #[cfg(feature = "gpu")]
            wgpu: std::sync::OnceLock::new(),
        })
    }

    fn hyperball(
        &self,
        graph: &CudaGraph,
        params: HyperBallGpuParams,
        counters: &mut [u8],
        accumulators: &mut [[f32; 2]],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let nodes = graph.nodes;
        let registers = params.registers;
        if registers < 4 || !registers.is_multiple_of(4) {
            return Err(format!("error {registers} registers aren't a multiple of 4").into());
        }
        if counters.len() != nodes * registers || accumulators.len() != nodes {
            return Err(format!(
                "error expected {nodes} counters of {registers} registers and {nodes} \
                 accumulators, got {} bytes of counters and {} accumulators",
                counters.len(),
                accumulators.len()
            )
            .into());
        }
        let words = (registers / 4) as u32;
        let source = include_str!("hyperball.cu");
        let union_balls = self.function("hyperball.cu", source, &[], "union_balls")?;
        let accumulate = self.function("hyperball.cu", source, &[], "accumulate")?;

        // counters before (`current`) and after (`next`) each iteration, swapped after it; every
        // buffer holds at least one element (empty ones can't be allocated)
        let mut current = if counters.is_empty() {
            self.stream.alloc_zeros::<u8>(4)?
        } else {
            self.stream.clone_htod(&*counters)?
        };
        let mut next = self.stream.alloc_zeros::<u8>(counters.len().max(4))?;
        let mut changed = self.stream.alloc_zeros::<u32>(nodes.max(1))?;
        let mut sums = self.stream.alloc_zeros::<f32>(2 * nodes.max(1))?;
        let mut changed_count = self.stream.alloc_zeros::<u32>(1)?;

        let (node_count, max_rank, register_count) =
            (nodes as u32, params.max_rank as u32, registers as f32);
        let union_threads = Self::launch_config(nodes as u64 * words as u64);
        let accumulate_threads = Self::launch_config(nodes as u64);
        let mut iteration = 0;
        let mut t = 1f32;
        loop {
            let inv_t = 1. / t;
            self.stream.memset_zeros(&mut changed_count)?;
            let mut launch = self.stream.launch_builder(&union_balls);
            launch
                .arg(&graph.offsets)
                .arg(&graph.neighbours)
                .arg(&current)
                .arg(&mut next)
                .arg(&mut changed)
                .arg(&node_count)
                .arg(&words);
            // SAFETY: `union_balls`'s arguments (`hyperball.cu`), in order: the CSR, counters
            // before and after the iteration (`nodes * words` words each), one changed flag per
            // node, the node count and words per counter
            unsafe { launch.launch(union_threads) }?;
            let mut launch = self.stream.launch_builder(&accumulate);
            launch
                .arg(&current)
                .arg(&next)
                .arg(&mut changed)
                .arg(&mut sums)
                .arg(&mut changed_count)
                .arg(&node_count)
                .arg(&words)
                .arg(&max_rank)
                .arg(&register_count)
                .arg(&t)
                .arg(&inv_t);
            // SAFETY: `accumulate`'s arguments, in order: the counters before and after the
            // iteration, the changed flags, two accumulators per node, the count of changed
            // counters, the node count, words and registers per counter, the largest register
            // value, the iteration's distance and its inverse
            unsafe { launch.launch(accumulate_threads) }?;

            // (copies are asynchronous: synchronized before reading)
            let count = self.stream.clone_dtoh(&changed_count)?;
            self.stream.synchronize()?;
            iteration += 1;
            if count[0] == 0 || iteration >= params.max_iterations {
                break;
            }
            std::mem::swap(&mut current, &mut next);
            t += 1.;
        }

        // the last iteration's counters are in `next`
        if !counters.is_empty() {
            self.stream
                .memcpy_dtoh(&next.slice(..counters.len()), counters)?;
        }
        if nodes > 0 {
            let sums_out: &mut [f32] = bytemuck::cast_slice_mut(accumulators);
            self.stream
                .memcpy_dtoh(&sums.slice(..2 * nodes), sums_out)?;
        }
        self.stream.synchronize()?;
        Ok(iteration)
    }

    fn triangles<Ix: graph::IndexType, C: GpuCount>(
        &self,
        graph: &CudaGraph,
        edge_over: &[Ix],
        edge_reciprocal: &[Ix],
        counts: &mut [C],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (nodes, edges) = (graph.nodes, graph.edges);
        if edge_over.len() != nodes || edge_reciprocal.len() != edges || counts.len() != edges {
            return Err(format!(
                "error expected {nodes} edge_over offsets, and {edges} reciprocals and counts, \
                 got {}, {} and {}",
                edge_over.len(),
                edge_reciprocal.len(),
                counts.len()
            )
            .into());
        }
        if edges == 0 {
            return Ok(());
        }
        let kernel = self.function(
            "triangles.cu",
            include_str!("triangles.cu"),
            &[format!("COUNT_BITS={}", C::BITS)],
            "count_triangles",
        )?;

        // counts are packed in `C`'s width, little-endian in 32-bit words: the buffer's bytes are
        // those of `counts`
        let count_bytes = std::mem::size_of_val(counts);
        let edge_over = self.upload_ids(edge_over)?;
        let mut device_counts = self
            .stream
            .alloc_zeros::<u8>(count_bytes.next_multiple_of(4))?;
        let mut overflow = self.stream.alloc_zeros::<u32>(1)?;
        let (nodes, edges) = (nodes as u32, edges as u32);
        let mut launch = self.stream.launch_builder(&kernel);
        launch
            .arg(&graph.offsets)
            .arg(&graph.neighbours)
            .arg(&edge_over)
            .arg(&mut device_counts)
            .arg(&mut overflow)
            .arg(&nodes)
            .arg(&edges);
        // SAFETY: the arguments are the kernel's (`triangles.cu`), in order: the CSR, `edge_over`
        // (one per node), counts (`count_bytes` rounded up to whole words, as it writes) and the
        // overflow flag, the node and edge counts
        unsafe { launch.launch(Self::launch_config(edges as u64)) }?;

        // (copies are asynchronous: synchronized before reading)
        let overflow = self.stream.clone_dtoh(&overflow)?;
        self.stream.synchronize()?;
        if overflow[0] != 0 {
            return Err(super::count_overflow::<C>());
        }
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(counts);
        self.stream
            .memcpy_dtoh(&device_counts.slice(..count_bytes), bytes)?;
        self.stream.synchronize()?;
        if cfg!(target_endian = "big") {
            for count in counts.iter_mut() {
                *count = count.from_le();
            }
        }

        // the device counted upward edges only
        super::copy_to_downward_edges(edge_reciprocal, counts);
        Ok(())
    }

    fn arc_products(
        &self,
        graph: &CudaGraph,
        x: &[u32],
        y: &[u32],
    ) -> Result<u128, Box<dyn std::error::Error>> {
        #[cfg(feature = "gpu")]
        {
            let (backend, graph) = self.wgpu(graph)?;
            backend.arc_products(graph, x, y)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (graph, x, y);
            Err(without_wgpu("degree assortativity"))
        }
    }
}
