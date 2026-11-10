//! [`GpuBackend`] on [wgpu](https://wgpu.rs) (Vulkan, Metal, DX12), so on any GPU vendor ---
//! or on a CPU through a software driver (e.g. Mesa's lavapipe), useful for testing.
use super::{GpuBackend, HyperBallGpuParams};
use crate::graph::{self, GraphMemoryMap, IndexType};

use std::sync::mpsc;

/// Must match `WORKGROUP_SIZE` in the shaders.
const WORKGROUP_SIZE: u64 = 64;
/// Host <-> device transfers are split in chunks of (at most) this many bytes, bounding the host
/// memory used for conversion and staging.
const TRANSFER_CHUNK: u64 = 64 << 20;

/// [`GpuBackend`] on the highest performance adapter wgpu finds (the `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` environment variables may be used to choose another).
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    /// Largest buffer bound to a shader, in bytes: the device's limit (lowered by tests, to split
    /// data in several buffers on small inputs).
    max_binding: u64,
    /// Most storage buffers a shader may bind: the device's limit (lowered by tests).
    max_storage_buffers: usize,
}

/// A graph's CSR in a [`WgpuBackend`]'s device memory.
pub struct WgpuGraph {
    nodes: usize,
    edges: usize,
    offsets: wgpu::Buffer,
    neighbours: wgpu::Buffer,
}

impl WgpuGraph {
    /// Number of nodes of the uploaded graph.
    pub fn size(&self) -> usize {
        self.nodes
    }
}

/// Compute pipelines sharing a bind group layout.
struct Kernels {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, Box<dyn std::error::Error>> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        // `wgpu::util::initialize_adapter_from_env` panics when no adapter matches the name, so
        // the name is matched here instead
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(name) => {
                let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
                let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
                adapters
                    .into_iter()
                    .find(|a| a.get_info().name.to_lowercase().contains(&name.to_lowercase()))
                    .ok_or_else(|| {
                        format!("error no GPU adapter matches WGPU_ADAPTER_NAME={name:?} (available: {names:?})")
                    })?
            }
            Err(_) => {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..Default::default()
                    })
                    .await?
            }
        };
        let info = adapter.get_info();
        // device state is sized by the graph, so ask for everything the adapter offers
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sparx"),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await?;
        let max_binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);

        let max_storage_buffers = limits.max_storage_buffers_per_shader_stage as usize;

        Ok(Self {
            device,
            queue,
            info,
            limits,
            max_binding,
            max_storage_buffers,
        })
    }

    /// The adapter (device and driver) computations run on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// Lowers the largest buffer bound to a shader to `bytes` (a multiple of 4), so that tests
    /// can exercise data split in several buffers on small inputs.
    #[cfg(test)]
    pub(crate) fn with_max_binding(mut self, bytes: u64) -> Self {
        self.max_binding = self.max_binding.min(bytes);
        self
    }

    /// Lowers the most storage buffers a shader may bind to `count`.
    #[cfg(test)]
    pub(crate) fn with_max_storage_buffers(mut self, count: usize) -> Self {
        self.max_storage_buffers = self.max_storage_buffers.min(count);
        self
    }

    /// Compiles `entry_points` of the WGSL `source`, whose bindings (all in group 0, numbered in
    /// order) are buffers of the given types.
    fn kernels(
        &self,
        label: &str,
        source: &str,
        bindings: &[wgpu::BufferBindingType],
        entry_points: &[&str],
    ) -> Kernels {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(binding, &ty)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipelines = entry_points
            .iter()
            .map(|&entry_point| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .collect();
        Kernels { layout, pipelines }
    }

    /// Binds `buffers` in order (binding `i` is `buffers[i]`).
    fn bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    /// Checks a shader may use `count` storage buffers.
    fn check_storage_buffers(&self, count: usize) -> Result<(), Box<dyn std::error::Error>> {
        let max = self.max_storage_buffers;
        if count > max {
            return Err(format!(
                "error {count} GPU buffers are needed, but {} supports at most {max} per shader",
                self.info.name
            )
            .into());
        }
        Ok(())
    }

    /// Creates a storage buffer of `size` bytes (zero initialized), failing if it exceeds what
    /// the device can bind.
    fn storage_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, Box<dyn std::error::Error>> {
        // bindings may not be empty
        let size = size.max(4);
        if size > self.max_binding {
            return Err(format!(
                "error {label} need {size} bytes of GPU memory in a single buffer, but {} \
                 supports at most {} bytes per buffer",
                self.info.name, self.max_binding
            )
            .into());
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        }))
    }

    fn uniform_buffer<T: bytemuck::Pod>(&self, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn wait(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    /// Writes `data` to `buffer` in chunks, submitting each so that staging memory is bounded.
    fn upload(&self, buffer: &wgpu::Buffer, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        for (i, chunk) in data.chunks(TRANSFER_CHUNK as usize).enumerate() {
            self.queue
                .write_buffer(buffer, i as u64 * TRANSFER_CHUNK, chunk);
            self.queue.submit([]);
            self.wait()?;
        }
        Ok(())
    }

    /// Uploads node or edge ids (of any index type) as `u32`.
    fn upload_ids<Ix: IndexType>(
        &self,
        buffer: &wgpu::Buffer,
        ids: &[Ix],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let per_chunk = (TRANSFER_CHUNK / 4) as usize;
        let mut converted = Vec::with_capacity(per_chunk.min(ids.len()));
        for (i, chunk) in ids.chunks(per_chunk).enumerate() {
            converted.clear();
            converted.extend(chunk.iter().map(|id| id.index() as u32));
            self.queue.write_buffer(
                buffer,
                (i * per_chunk * 4) as u64,
                bytemuck::cast_slice(&converted),
            );
            self.queue.submit([]);
            self.wait()?;
        }
        Ok(())
    }

    /// Reads `buffer`'s first `out.len()` bytes into `out`, in chunks.
    fn download(
        &self,
        buffer: &wgpu::Buffer,
        out: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: TRANSFER_CHUNK.min(out.len() as u64).max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for (i, chunk) in out.chunks_mut(TRANSFER_CHUNK as usize).enumerate() {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                buffer,
                i as u64 * TRANSFER_CHUNK,
                &staging,
                0,
                chunk.len() as u64,
            );
            self.queue.submit([encoder.finish()]);
            self.read_staging(&staging, chunk)?;
        }
        Ok(())
    }

    /// Maps `staging`, copies its first `out.len()` bytes into `out` and unmaps it.
    fn read_staging(
        &self,
        staging: &wgpu::Buffer,
        out: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let slice = staging.slice(0..out.len() as u64);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait()?;
        receiver.recv()??;
        out.copy_from_slice(&slice.get_mapped_range()?);
        staging.unmap();
        Ok(())
    }

    /// Workgroups to dispatch over (at least) `threads` invocations indexed linearly as
    /// `id.x + id.y * num_workgroups.x * WORKGROUP_SIZE`, in rows when they exceed the per
    /// dimension limit.
    fn linear_workgroups(&self, threads: u64) -> Result<(u32, u32), Box<dyn std::error::Error>> {
        let max = self.limits.max_compute_workgroups_per_dimension as u64;
        let groups = threads.div_ceil(WORKGROUP_SIZE).max(1);
        let x = groups.min(max);
        let y = groups.div_ceil(x);
        if y > max || x * y * WORKGROUP_SIZE > u32::MAX as u64 {
            return Err(
                format!("error {threads} GPU invocations exceed what can be dispatched").into(),
            );
        }
        Ok((x as u32, y as u32))
    }

    fn dispatch(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        (x, y): (u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
}

/// `hyperball.wgsl`'s uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HyperBallUniforms {
    nodes: u32,
    words: u32,
    max_rank: u32,
    segment_nodes: u32,
    row_nodes: u32,
    _pad0: [u32; 3],
    t: f32,
    inv_t: f32,
    registers: f32,
    _pad1: f32,
}

/// `hyperball.wgsl` with bindings and accessors for `segments` counter segments.
fn hyperball_shader(segments: usize) -> String {
    let (mut bindings, mut read_current, mut read_next, mut write_next) =
        (String::new(), String::new(), String::new(), String::new());
    for k in 0..segments {
        bindings += &format!(
            "@group(0) @binding({}) var<storage, read> current_{k}: array<u32>;\n\
             @group(0) @binding({}) var<storage, read_write> next_{k}: array<u32>;\n",
            HYPERBALL_FIXED_BINDINGS + k,
            HYPERBALL_FIXED_BINDINGS + segments + k
        );
        read_current += &format!("        case {k}u: {{ return current_{k}[i]; }}\n");
        read_next += &format!("        case {k}u: {{ return next_{k}[i]; }}\n");
        write_next += &format!("        case {k}u: {{ next_{k}[i] = value; }}\n");
    }
    include_str!("hyperball.wgsl")
        .replace("{{SEGMENT_BINDINGS}}", &bindings)
        .replace("{{READ_CURRENT_CASES}}\n", &read_current)
        .replace("{{READ_NEXT_CASES}}\n", &read_next)
        .replace("{{WRITE_NEXT_CASES}}\n", &write_next)
}

/// `triangles.wgsl`'s uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TrianglesUniforms {
    nodes: u32,
    edges: u32,
    _pad: [u32; 2],
}

/// `arc_products.wgsl`'s uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArcProductsUniforms {
    nodes: u32,
    _pad: [u32; 3],
}

/// Bindings of `hyperball.wgsl` before the counter segments'.
const HYPERBALL_FIXED_BINDINGS: usize = 6;

impl GpuBackend for WgpuBackend {
    type Graph = WgpuGraph;

    fn upload_graph<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>(
        &self,
        g: &GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<WgpuGraph, Box<dyn std::error::Error>> {
        let nodes = g.size();
        let edges = g.width();
        // offsets index edges, so both must be representable as `u32`
        if nodes >= u32::MAX as usize || edges > u32::MAX as usize {
            return Err(format!(
                "error GPU graphs are indexed by u32: |V| = {nodes} and |E| = {edges} must be \
                 smaller than {}",
                u32::MAX
            )
            .into());
        }
        // SAFETY: the graph's index holds `|V| + 1` offsets and its neighbours `|E|` node ids,
        // both mapped for as long as `g` lives
        let (offsets, neighbours) = unsafe {
            (
                std::slice::from_raw_parts(g.offsets_ptr(), nodes + 1),
                std::slice::from_raw_parts(g.neighbours_ptr(), edges),
            )
        };

        let copy = wgpu::BufferUsages::COPY_DST;
        let offsets_buffer = self.storage_buffer("offsets", (nodes as u64 + 1) * 4, copy)?;
        let neighbours_buffer = self.storage_buffer("neighbours", edges as u64 * 4, copy)?;
        self.upload_ids(&offsets_buffer, offsets)?;
        self.upload_ids(&neighbours_buffer, neighbours)?;

        Ok(WgpuGraph {
            nodes,
            edges,
            offsets: offsets_buffer,
            neighbours: neighbours_buffer,
        })
    }

    fn hyperball(
        &self,
        graph: &WgpuGraph,
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
        let words = (registers / 4) as u64;

        // counters are split in as few segments (one buffer each) as the device allows
        let segment_nodes = (self.max_binding / registers as u64).min(nodes.max(1) as u64);
        if segment_nodes == 0 {
            return Err(format!(
                "error a {registers} register counter exceeds the largest GPU buffer ({} bytes)",
                self.max_binding
            )
            .into());
        }
        let segments = (nodes as u64).div_ceil(segment_nodes).max(1) as usize;
        self.check_storage_buffers(HYPERBALL_FIXED_BINDINGS - 1 + 2 * segments)?;

        // `union_balls` runs in rows of whole counters: `row_threads` (a multiple of both
        // `words` and the workgroup size) invocations cover `row_nodes` counters
        let max_groups = self.limits.max_compute_workgroups_per_dimension as u64;
        let row_unit = words.max(WORKGROUP_SIZE);
        let row_threads = (max_groups * WORKGROUP_SIZE / row_unit * row_unit)
            .min((nodes as u64 * words).div_ceil(row_unit) * row_unit)
            .max(row_unit);
        let row_nodes = row_threads / words;
        let rows = (nodes as u64).div_ceil(row_nodes).max(1);
        if rows > max_groups || row_threads / WORKGROUP_SIZE > max_groups {
            return Err(format!("error {nodes} counters exceed what can be dispatched").into());
        }
        let union_groups = ((row_threads / WORKGROUP_SIZE) as u32, rows as u32);
        let accumulate_groups = self.linear_workgroups(nodes as u64)?;

        use wgpu::BufferBindingType as B;
        let storage = |read_only| B::Storage { read_only };
        let mut bindings = vec![
            storage(true),
            storage(true),
            storage(false),
            storage(false),
            storage(false),
            B::Uniform,
        ];
        bindings.extend(std::iter::repeat_n(storage(true), segments));
        bindings.extend(std::iter::repeat_n(storage(false), segments));
        let kernels = self.kernels(
            "hyperball",
            &hyperball_shader(segments),
            &bindings,
            &["union_balls", "accumulate"],
        );

        use wgpu::BufferUsages as U;
        let segment_bytes = segment_nodes as usize * registers;
        let mut a = Vec::with_capacity(segments);
        let mut b = Vec::with_capacity(segments);
        for chunk in counters.chunks(segment_bytes) {
            let segment =
                self.storage_buffer("counters", chunk.len() as u64, U::COPY_DST | U::COPY_SRC)?;
            self.upload(&segment, chunk)?;
            a.push(segment);
            b.push(self.storage_buffer("counters swap", chunk.len() as u64, U::COPY_SRC)?);
        }
        let changed = self.storage_buffer("changed", nodes as u64 * 4, U::empty())?;
        let sums = self.storage_buffer("accumulators", nodes as u64 * 8, U::COPY_SRC)?;
        let changed_count = self.storage_buffer("changed count", 4, U::COPY_DST | U::COPY_SRC)?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("changed count readback"),
            size: 4,
            usage: U::MAP_READ | U::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.uniform_buffer::<HyperBallUniforms>("hyperball params");

        // `current -> next` bind groups for both directions
        let bind_group = |current: &[wgpu::Buffer], next: &[wgpu::Buffer]| {
            let mut buffers = vec![
                &graph.offsets,
                &graph.neighbours,
                &changed,
                &sums,
                &changed_count,
                &uniform,
            ];
            buffers.extend(current);
            buffers.extend(next);
            self.bind_group(&kernels.layout, &buffers)
        };
        let bind_groups = [bind_group(&a, &b), bind_group(&b, &a)];

        let mut iteration = 0;
        let mut t = 1f32;
        loop {
            let uniforms = HyperBallUniforms {
                nodes: nodes as u32,
                words: words as u32,
                max_rank: params.max_rank as u32,
                segment_nodes: segment_nodes as u32,
                row_nodes: row_nodes as u32,
                _pad0: [0; 3],
                t,
                inv_t: 1. / t,
                registers: registers as f32,
                _pad1: 0.,
            };
            self.queue
                .write_buffer(&uniform, 0, bytemuck::bytes_of(&uniforms));
            self.queue
                .write_buffer(&changed_count, 0, &0u32.to_ne_bytes());

            let bind_group = &bind_groups[iteration % 2];
            let mut encoder = self.device.create_command_encoder(&Default::default());
            Self::dispatch(
                &mut encoder,
                &kernels.pipelines[0],
                bind_group,
                union_groups,
            );
            Self::dispatch(
                &mut encoder,
                &kernels.pipelines[1],
                bind_group,
                accumulate_groups,
            );
            encoder.copy_buffer_to_buffer(&changed_count, 0, &readback, 0, 4);
            self.queue.submit([encoder.finish()]);

            let mut count = [0u8; 4];
            self.read_staging(&readback, &mut count)?;
            iteration += 1;
            if u32::from_ne_bytes(count) == 0 || iteration >= params.max_iterations {
                break;
            }
            t += 1.;
        }

        // the last iteration wrote into `b` if it was the 1st, 3rd, ... one
        let newest = if iteration % 2 == 1 { &b } else { &a };
        for (segment, chunk) in newest.iter().zip(counters.chunks_mut(segment_bytes)) {
            self.download(segment, chunk)?;
        }
        self.download(&sums, bytemuck::cast_slice_mut(accumulators))?;

        Ok(iteration)
    }

    fn triangles<Ix: graph::IndexType>(
        &self,
        graph: &WgpuGraph,
        edge_over: &[Ix],
        edge_reciprocal: &[Ix],
        counts: &mut [u32],
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
        self.check_storage_buffers(5)?;
        let groups = self.linear_workgroups(edges as u64)?;

        use wgpu::BufferUsages as U;
        let edge_over_buffer = self.storage_buffer("edge over", nodes as u64 * 4, U::COPY_DST)?;
        let reciprocal_buffer =
            self.storage_buffer("reciprocals", edges as u64 * 4, U::COPY_DST)?;
        let counts_buffer =
            self.storage_buffer("triangle counts", edges as u64 * 4, U::COPY_SRC)?;
        let uniform = self.uniform_buffer::<TrianglesUniforms>("triangles params");
        self.upload_ids(&edge_over_buffer, edge_over)?;
        self.upload_ids(&reciprocal_buffer, edge_reciprocal)?;
        let uniforms = TrianglesUniforms {
            nodes: nodes as u32,
            edges: edges as u32,
            _pad: [0; 2],
        };
        self.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&uniforms));

        use wgpu::BufferBindingType as B;
        let storage = |read_only| B::Storage { read_only };
        let kernels = self.kernels(
            "triangles",
            include_str!("triangles.wgsl"),
            &[
                storage(true),
                storage(true),
                storage(true),
                storage(true),
                storage(false),
                B::Uniform,
            ],
            &["count_triangles"],
        );
        let bind_group = self.bind_group(
            &kernels.layout,
            &[
                &graph.offsets,
                &graph.neighbours,
                &edge_over_buffer,
                &reciprocal_buffer,
                &counts_buffer,
                &uniform,
            ],
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        Self::dispatch(&mut encoder, &kernels.pipelines[0], &bind_group, groups);
        self.queue.submit([encoder.finish()]);

        self.download(&counts_buffer, bytemuck::cast_slice_mut(counts))
    }

    fn arc_products(
        &self,
        graph: &WgpuGraph,
        x: &[u32],
        y: &[u32],
    ) -> Result<u128, Box<dyn std::error::Error>> {
        let nodes = graph.nodes;
        if x.len() != nodes || y.len() != nodes {
            return Err(format!(
                "error expected {nodes} values of each end, got {} and {}",
                x.len(),
                y.len()
            )
            .into());
        }
        self.check_storage_buffers(5)?;
        let groups = self.linear_workgroups(nodes as u64)?;
        let group_count = groups.0 as u64 * groups.1 as u64;

        use wgpu::BufferUsages as U;
        let x_buffer = self.storage_buffer("tails' values", nodes as u64 * 4, U::COPY_DST)?;
        let y_buffer = self.storage_buffer("heads' values", nodes as u64 * 4, U::COPY_DST)?;
        let partials_buffer = self.storage_buffer("arc products", group_count * 16, U::COPY_SRC)?;
        let uniform = self.uniform_buffer::<ArcProductsUniforms>("arc products params");
        self.upload(&x_buffer, bytemuck::cast_slice(x))?;
        self.upload(&y_buffer, bytemuck::cast_slice(y))?;
        let uniforms = ArcProductsUniforms {
            nodes: nodes as u32,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&uniforms));

        use wgpu::BufferBindingType as B;
        let storage = |read_only| B::Storage { read_only };
        let kernels = self.kernels(
            "arc products",
            include_str!("arc_products.wgsl"),
            &[
                storage(true),
                storage(true),
                storage(true),
                storage(true),
                storage(false),
                B::Uniform,
            ],
            &["arc_products"],
        );
        let bind_group = self.bind_group(
            &kernels.layout,
            &[
                &graph.offsets,
                &graph.neighbours,
                &x_buffer,
                &y_buffer,
                &partials_buffer,
                &uniform,
            ],
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        Self::dispatch(&mut encoder, &kernels.pipelines[0], &bind_group, groups);
        self.queue.submit([encoder.finish()]);

        let mut partials = vec![[0u32; 4]; group_count as usize];
        self.download(&partials_buffer, bytemuck::cast_slice_mut(&mut partials))?;
        Ok(partials
            .iter()
            .map(|l| {
                l.iter()
                    .rev()
                    .fold(0u128, |acc, &limb| (acc << 32) | limb as u128)
            })
            .sum())
    }
}
