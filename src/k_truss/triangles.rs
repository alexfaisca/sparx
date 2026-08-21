use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicBool, AtomicUsize, Ordering};

use super::count::{TriangleCount, TriangleCounter};
use std::mem::ManuallyDrop;
use std::path::Path;

/// Per-edge triangle counts of an undirected graph (for directed graphs see
/// [`DirectedTriangles`]), as `Tx` integers (`u8` by default, see [`TriangleCount`]).
///
/// [`DirectedTriangles`]: ../directed/struct.DirectedTriangles.html#
/// [`TriangleCount`]: ../count/trait.TriangleCount.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct Triangles<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Tx: TriangleCount = u8,
> {
    /// Graph for which each edge's triangles are computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the number of triangles of each edge.
    triangles: AbstractedProceduralMemoryMut<Tx::Atomic>,
    /// Threads to be used
    threads: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Triangles<'a, N, E, Ix> {
    /// Counts every edge's triangles, as `u8`s (see [`new_tx`](Self::new_tx) for other widths).
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx(g)
    }

    /// Searches for previously cached `u8` triangle counts, computing them if not found.
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::get_or_compute_tx(g)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    Triangles<'a, N, E, Ix, Tx>
{
    /// Counts every edge's triangles, as `Tx` integers.
    pub fn new_tx(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut triangles = Self::new_no_compute(g, g.thread_num())?;
        triangles.compute_with_proc_mem(triangles.init_cache_mem()?)?;

        Ok(triangles)
    }

    /// Searches for previously cached `Tx` triangle counts, computing them if not found.
    pub fn get_or_compute_tx(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::Triangles, None)?;
        if Path::new(&t_fn).exists() {
            // a cache of another graph (version) or count width would have another length
            if let Ok(triangles) =
                AbstractedProceduralMemoryMut::<Tx::Atomic>::from_file_name(&t_fn)
                && triangles.as_slice().len() == g.width()
            {
                return Ok(Self {
                    g,
                    triangles,
                    threads: g.thread_num().max(1),
                });
            }
        }
        Self::new_tx(g)
    }

    pub fn triangles(&self, e_idx: usize) -> Tx {
        assert!(e_idx < self.g.width());
        Tx::from_usize(self.triangles.get(e_idx).count())
    }

    pub(crate) fn triangles_shares_slice(&self) -> SharedSliceMut<Tx::Atomic> {
        self.triangles.shared_slice()
    }

    pub fn all_triangles(&self) -> &[Tx::Atomic] {
        self.triangles.as_slice()
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::Triangles, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    Triangles<'a, N, E, Ix, Tx>
{
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
        let triangles = SharedSliceMut::<Tx::Atomic>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self {
            g,
            triangles,
            threads,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    /// Counts every edge's triangles. Neighbour lists are sorted, so every triangle `x < y < z`
    /// is found exactly once, from its lowest edge `x -> y`, by merging `x`'s and `y`'s neighbours
    /// above `y`; it then counts once for each of its 6 (directed) edges --- once per combination
    /// of parallel edges.
    fn compute_with_proc_mem_impl(&self, _proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        // nodes claimed by a thread at a time: dynamic scheduling balances uneven degrees
        const CHUNK: usize = 1024;

        let node_count = self.g.size();
        let edge_count = self.g.width();

        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        let tris = self.triangles.shared_slice();

        let edge_reciprocal = self.g.edge_reciprocal()?;
        let edge_over = self.g.edge_over()?;

        let next_chunk = AtomicUsize::new(0);
        let overflow = AtomicBool::new(false);

        thread::scope(|scope| {
            for _ in 0..self.threads {
                let eo = edge_over.shared_slice();
                let er = edge_reciprocal.shared_slice();
                let tris = tris.clone();
                let (next_chunk, overflow) = (&next_chunk, &overflow);

                scope.spawn(move |_| {
                    // counts a triangle for edge `e` and its reciprocal
                    let count = |e: usize| {
                        for e in [e, er.get(e).index()] {
                            if tris.get(e).add_count(1) == Tx::MAX {
                                overflow.store(true, Ordering::Relaxed);
                            }
                        }
                    };
                    loop {
                        let begin = next_chunk.fetch_add(CHUNK, Ordering::Relaxed);
                        if begin >= node_count {
                            break;
                        }
                        let end = std::cmp::min(begin + CHUNK, node_count);
                        for x in begin..end {
                            let x_end = index_ptr.get(x + 1).index();
                            for x_y in eo.get(x).index()..x_end {
                                let y = neighbours_ptr.get(x_y).index();
                                // `x`'s neighbours above `y` (past `y` itself and edges parallel
                                // to `x -> y`) and `y`'s neighbours above `y`
                                let mut p = x_y + 1;
                                while p < x_end && neighbours_ptr.get(p).index() == y {
                                    p += 1;
                                }
                                let mut q = eo.get(y).index();
                                let y_end = index_ptr.get(y + 1).index();
                                while p < x_end && q < y_end {
                                    let a = neighbours_ptr.get(p).index();
                                    let b = neighbours_ptr.get(q).index();
                                    match a.cmp(&b) {
                                        std::cmp::Ordering::Less => p += 1,
                                        std::cmp::Ordering::Greater => q += 1,
                                        std::cmp::Ordering::Equal => {
                                            // runs of parallel edges to the common neighbour
                                            let mut p_run = p;
                                            while p_run < x_end && neighbours_ptr.get(p_run).index() == a {
                                                p_run += 1;
                                            }
                                            let mut q_run = q;
                                            while q_run < y_end && neighbours_ptr.get(q_run).index() == a {
                                                q_run += 1;
                                            }
                                            for x_z in p..p_run {
                                                for y_z in q..q_run {
                                                    count(x_y);
                                                    count(x_z);
                                                    count(y_z);
                                                }
                                            }
                                            p = p_run;
                                            q = q_run;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        if overflow.load(Ordering::Relaxed) {
            return Err(format!(
                "error an edge is in more than {} triangles, more than can be counted ({})",
                Tx::MAX,
                std::any::type_name::<Tx>()
            )
            .into());
        }

        // (counting creates no temporary files)
        self.triangles.flush_async()?;

        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Triangles<'a, N, E, Ix> {
    /// Computes each edge's triangles on a GPU, as `u8`s (see [`new_gpu_tx`](Self::new_gpu_tx)
    /// for other widths).
    pub fn new_gpu<B: crate::gpu::GpuBackend>(
        g: &'a GraphMemoryMap<N, E, Ix>,
        backend: &B,
        graph: &B::Graph,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_gpu_tx(g, backend, graph)
    }
}

#[cfg(feature = "gpu")]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    Triangles<'a, N, E, Ix, Tx>
{
    /// Computes each edge's triangles on a GPU, with the same results (and cached file) as
    /// [`new`].
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance whose edges' triangles are counted.
    /// * `backend` --- the GPU backend to compute on.
    /// * `graph` --- `g` in the backend's device memory, see [`GpuBackend::upload_graph`].
    ///
    /// [`new`]: #method.new
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    /// [`GpuBackend::upload_graph`]: ../../gpu/trait.GpuBackend.html#tymethod.upload_graph
    pub fn new_gpu_tx<B: crate::gpu::GpuBackend>(
        g: &'a GraphMemoryMap<N, E, Ix>,
        backend: &B,
        graph: &B::Graph,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let triangles = Self::new_no_compute_impl(g, g.thread_num())?;
        let edge_over = g.edge_over()?;
        let edge_reciprocal = g.edge_reciprocal()?;

        // read back into a scratch file rather than the heap: one count per arc
        let c_fn = g.build_cache_filename(CacheFile::Triangles, Some(0))?;
        let mut counts = ScratchMem::<u32>::new(c_fn, g.width())?;
        backend.triangles(graph, edge_over.as_slice(), edge_reciprocal.as_slice(), &mut counts)?;
        for (t, &c) in triangles.triangles.as_slice().iter().zip(counts.iter()) {
            if c as usize > Tx::MAX {
                return Err(format!(
                    "error an edge is in {c} triangles, more than can be counted ({})",
                    std::any::type_name::<Tx>()
                )
                .into());
            }
            t.set_count(c as usize);
        }

        triangles.triangles.flush_async()?;
        Ok(triangles)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use std::collections::BTreeSet;
    use std::io::Write;

    /// A test graph (removing its cached files and fixture when dropped).
    pub(crate) type TestGraph = crate::graph::test_utils::TestGraph<VoidLabel, VoidLabel, usize>;
    /// The graphs the brute force references take.
    pub(crate) type Graph = GraphMemoryMap<VoidLabel, VoidLabel, usize>;
    /// The directed graphs the brute force references take, and the test graphs that deref to them.
    pub(crate) type DiGraph = crate::graph::DiGraph<VoidLabel, VoidLabel, usize>;
    pub(crate) type DiTestGraph =
        crate::graph::test_utils::TestGraph<VoidLabel, VoidLabel, usize, crate::graph::Directed>;

    /// Parses `edges` (0-based, undirected) as a symmetric MatrixMarket graph over `n` nodes.
    pub(crate) fn undirected_graph(name: &str, n: usize, edges: &[(usize, usize)]) -> TestGraph {
        let dir = unique_test_dir("triangles_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let graph = GraphMemoryMap::from_file(
            &path,
            Some(format!("tri_{name}_{}", unique_run_tag())),
            Some(4),
        )
        .expect("graph parse should succeed");
        TestGraph::new(graph, Some(dir))
    }

    pub(crate) fn clique_edges(n: usize) -> Vec<(usize, usize)> {
        (0..n).flat_map(|u| (u + 1..n).map(move |v| (u, v))).collect()
    }

    /// Undirected DNA de Bruijn graph of order `k`, without self-loops or duplicate edges.
    pub(crate) fn de_bruijn_edges(k: u32) -> Vec<(usize, usize)> {
        let n = 4usize.pow(k);
        let mut edges = BTreeSet::new();
        for u in 0..n {
            for c in 0..4 {
                let v = (4 * u + c) % n;
                if u != v {
                    edges.insert((u.min(v), u.max(v)));
                }
            }
        }
        edges.into_iter().collect()
    }

    /// Erdos-Renyi `G(n, p)` graph (fixed seed).
    pub(crate) fn random_edges(n: usize, p: f64, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed | 1;
        let mut edges = vec![];
        for u in 0..n {
            for v in u + 1..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if (state >> 11) as f64 / (1u64 << 53) as f64 <= p {
                    edges.push((u, v));
                }
            }
        }
        edges
    }

    /// `side x side` grid with one diagonal per cell (every inner edge in 2 triangles).
    pub(crate) fn triangulated_grid_edges(side: usize) -> Vec<(usize, usize)> {
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
                if c + 1 < side && r + 1 < side {
                    edges.push((u, u + side + 1));
                }
            }
        }
        edges
    }

    /// Each CSR edge's triangle count by brute force: for edge `u -> v`, the sum over the
    /// endpoints' common neighbours `w` of the number of `u -- w` and `v -- w` edge combinations
    /// (so 1 per common neighbour without parallel edges), and 0 for self-loops.
    pub(crate) fn brute_force_triangles(g: &Graph) -> Vec<usize> {
        let multiplicities: Vec<std::collections::BTreeMap<usize, usize>> = (0..g.size())
            .map(|u| {
                let mut m = std::collections::BTreeMap::new();
                for v in g.neighbours(u).unwrap().filter(|&v| v != u) {
                    *m.entry(v).or_insert(0) += 1;
                }
                m
            })
            .collect();
        let mut counts = vec![];
        for u in 0..g.size() {
            for v in g.neighbours(u).unwrap() {
                counts.push(if u == v {
                    0
                } else {
                    multiplicities[u]
                        .iter()
                        .filter(|&(&w, _)| w != v)
                        .map(|(w, m)| m * multiplicities[v].get(w).unwrap_or(&0))
                        .sum()
                });
            }
        }
        counts
    }

    pub(crate) fn fixtures() -> Vec<(&'static str, TestGraph)> {
        vec![
            ("clique6", undirected_graph("clique6", 6, &clique_edges(6))),
            ("debruijn4", undirected_graph("debruijn4", 256, &de_bruijn_edges(4))),
            ("grid", undirected_graph("grid", 400, &triangulated_grid_edges(20))),
            ("random", undirected_graph("random", 300, &random_edges(300, 0.08, 7))),
            // isolated nodes and a self-loop
            (
                "sparse",
                undirected_graph("sparse", 30, &[(0, 1), (1, 2), (0, 2), (5, 5), (5, 6), (20, 21)]),
            ),
            // parallel edges: every 5th edge of a random graph tripled
            ("multi", undirected_graph("multi", 80, &{
                let mut edges = random_edges(80, 0.2, 11);
                let parallel: Vec<_> = edges.iter().step_by(5).copied().collect();
                edges.extend(parallel.iter().chain(&parallel));
                edges
            })),
        ]
    }

    fn counts(t: &Triangles<VoidLabel, VoidLabel, usize>) -> Vec<usize> {
        t.all_triangles().iter().map(|c| c.load(Ordering::Relaxed) as usize).collect()
    }

    #[test]
    fn triangles_match_brute_force() {
        for (name, g) in fixtures() {
            let mut t = Triangles::new(&g).unwrap();
            assert_eq!(counts(&t), brute_force_triangles(&g), "{name}");
            t.drop_cache().unwrap();
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn triangles_gpu_match_brute_force() {
        use crate::gpu::GpuBackend;
        let backend = match crate::gpu::WgpuBackend::new() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("skipping GPU test, no GPU adapter: {e}");
                return;
            }
        };
        for (name, g) in fixtures() {
            let device_graph = backend.upload_graph(&g).unwrap();
            let mut t = Triangles::new_gpu(&g, &backend, &device_graph).unwrap();
            assert_eq!(counts(&t), brute_force_triangles(&g), "{name}");
            t.drop_cache().unwrap();
        }
    }
}

