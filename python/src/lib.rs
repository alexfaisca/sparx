//! Python bindings of `sparx`: a [`Graph`] class whose methods run the library's algorithms and
//! return their results as `numpy` arrays.
//!
//! Every algorithm caches its results in files next to the graph's; the bindings copy them out and
//! drop those files right away, so that the cache only ever holds the graph itself (dropped too
//! when the graph is closed, unless it was built with `keep_cache=True`).

use std::error::Error;
use std::path::Path;
use std::sync::Mutex;

use numpy::{Element, IntoPyArray};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sparx::centralities::exact::{
    ExactClosenessCentrality, ExactHarmonicCentrality, ExactLinCentrality,
};
use sparx::centralities::hyper_ball::*;
use sparx::centralities::hyper_log_log::Precision;
use sparx::communities::directed_louvain::DirectedLouvain;
use sparx::communities::gve_louvain::AlgoGVELouvain;
use sparx::components::{strong::StronglyConnectedComponents, weak::ConnectedComponents};
use sparx::graph::{Direction, GraphMemoryMap, IndexType};
use sparx::k_core::{
    batagelj_zaversnik::AlgoBatageljZaversnik, directed::DirectedKCores, liu_et_al::AlgoLiuEtAl,
};
use sparx::k_truss::clustering_coefficient::{ClusteringCoefficient, DirectedClusteringCoefficient};
use sparx::k_truss::count::TriangleCount;
use sparx::k_truss::directed::{DirectedTriangleType, DirectedTriangles};
use sparx::k_truss::{burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT, triangles::Triangles};
use sparx::trails::bfs::BFSDists;

type G<Ix> = GraphMemoryMap<(), (), Ix>;
type Res<T> = Result<T, Box<dyn Error>>;

/// A result, built into Python objects once the GIL is held again.
type Out = Box<dyn for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>> + Send>;

fn array<T: Element + Send + 'static>(v: Vec<T>) -> Out {
    Box::new(move |py| Ok(v.into_pyarray(py).into_any()))
}

/// A dictionary of results (arrays or numbers).
fn dict(entries: Vec<(&'static str, Out)>) -> Out {
    Box::new(move |py| {
        let d = PyDict::new(py);
        for (k, v) in entries {
            d.set_item(k, v(py)?)?;
        }
        Ok(d.into_any())
    })
}

fn pair(a: Out, b: Out) -> Out {
    Box::new(move |py| Ok((a(py)?, b(py)?).into_pyobject(py)?.into_any()))
}

fn number<T: for<'py> IntoPyObject<'py> + Send + 'static>(x: T) -> Out {
    Box::new(move |py| x.into_bound_py_any(py))
}

/// Index types the bindings are built for.
trait Ix: IndexType {}
impl Ix for u32 {}
impl Ix for usize {}

/// Node and edge indices are `int64` in Python, whatever the graph's index type, like `numpy`'s
/// own indices.
fn ix_array<I: Ix>(s: &[I]) -> Out {
    array(s.iter().map(|x| x.index() as i64).collect::<Vec<_>>())
}

enum AnyGraph {
    U32(G<u32>),
    U64(G<usize>),
}

/// Runs `$body` on the graph, whatever its index type, as `$g: &G<Ix>`.
macro_rules! with_graph {
    ($any:expr, $g:ident => $body:expr) => {
        match $any {
            AnyGraph::U32($g) => $body,
            AnyGraph::U64($g) => $body,
        }
    };
}

#[derive(Clone, Copy)]
enum Width {
    U8,
    U16,
    U32,
    U64,
}

fn width(name: &str) -> PyResult<Width> {
    match name {
        "u8" => Ok(Width::U8),
        "u16" => Ok(Width::U16),
        "u32" => Ok(Width::U32),
        "u64" => Ok(Width::U64),
        _ => Err(PyValueError::new_err(format!(
            "unknown width {name:?}: expected 'u8', 'u16', 'u32' or 'u64'"
        ))),
    }
}

/// Runs `$body` with `$T` the counter type of width `$w`.
macro_rules! with_width {
    ($w:expr, $T:ident => $body:expr) => {
        match $w {
            Width::U8 => {
                type $T = u8;
                $body
            }
            Width::U16 => {
                type $T = u16;
                $body
            }
            Width::U32 => {
                type $T = u32;
                $body
            }
            Width::U64 => {
                type $T = u64;
                $body
            }
        }
    };
}

/// Runs `$body` with `$P` the *HyperLogLog* precision `$p`.
macro_rules! with_precision {
    ($p:expr, $P:ident => $body:expr) => {
        with_precision!(@ $p, $P, $body, 4 Precision4, 5 Precision5, 6 Precision6, 7 Precision7,
            8 Precision8, 9 Precision9, 10 Precision10, 11 Precision11, 12 Precision12,
            13 Precision13, 14 Precision14, 15 Precision15, 16 Precision16)
    };
    (@ $p:expr, $P:ident, $body:expr, $($n:literal $ty:ident),*) => {
        match $p {
            $($n => {
                type $P = $ty;
                Ok($body)
            })*
            p => Err(PyValueError::new_err(format!("precision must be in 4..=16, got {p}"))),
        }
    };
}

fn direction(name: &str) -> PyResult<Direction> {
    match name {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        "both" => Ok(Direction::Both),
        _ => Err(PyValueError::new_err(format!(
            "unknown direction {name:?}: expected 'out', 'in' or 'both'"
        ))),
    }
}

fn triangle_type(name: &str) -> PyResult<DirectedTriangleType> {
    let key: String = name
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .collect::<String>()
        .to_lowercase();
    DirectedTriangleType::ALL
        .into_iter()
        .find(|t| t.name() == key)
        .ok_or_else(|| {
            let names: Vec<_> = DirectedTriangleType::ALL.iter().map(|t| t.name()).collect();
            PyValueError::new_err(format!(
                "unknown triangle type {name:?}: expected one of {}",
                names.join(", ")
            ))
        })
}

fn triangles<I: Ix, Tx: TriangleCount + Element>(g: &G<I>) -> Res<Out> {
    let mut t = Triangles::<_, _, _, Tx>::new_tx(g)?;
    let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
    t.drop_cache()?;
    Ok(array(v))
}

#[cfg(feature = "gpu")]
fn triangles_gpu<I: Ix, Tx: TriangleCount + Element>(g: &G<I>) -> Res<Out> {
    use sparx::gpu::{GpuBackend, WgpuBackend};
    let backend = WgpuBackend::new()?;
    let device_graph = backend.upload_graph(g)?;
    let mut t = Triangles::<_, _, _, Tx>::new_gpu_tx(g, &backend, &device_graph)?;
    let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
    t.drop_cache()?;
    Ok(array(v))
}

fn directed_triangles<I: Ix, Tx: TriangleCount + Element>(
    g: &G<I>,
    kind: DirectedTriangleType,
) -> Res<Out> {
    let mut t = DirectedTriangles::<_, _, _, Tx>::new_tx(g, kind)?;
    let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
    t.drop_cache()?;
    Ok(array(v))
}

fn k_truss<I: Ix, Tx: TriangleCount + Element>(
    g: &G<I>,
    pkt: bool,
    kind: Option<DirectedTriangleType>,
) -> Res<Out> {
    let v = if pkt {
        let mut a = match kind {
            None => AlgoPKT::<_, _, _, Tx>::new_tx(g, g.thread_num())?,
            Some(kind) => AlgoPKT::<_, _, _, Tx>::new_directed_tx(g, kind, g.thread_num())?,
        };
        let v = a.k_trusses().to_vec();
        a.drop_cache()?;
        v
    } else {
        let mut a = match kind {
            None => AlgoBurkhardtEtAl::<_, _, _, Tx>::new_tx(g)?,
            Some(kind) => AlgoBurkhardtEtAl::<_, _, _, Tx>::new_directed_tx(g, kind)?,
        };
        let v = a.k_trusses().to_vec();
        a.drop_cache()?;
        v
    };
    Ok(array(v))
}

fn k_core<I: Ix>(g: &G<I>, bz: bool) -> Res<Out> {
    let v = if bz {
        let mut a = AlgoBatageljZaversnik::new(g)?;
        let v = a.k_cores().to_vec();
        a.drop_cache()?;
        v
    } else {
        let mut a = AlgoLiuEtAl::new(g)?;
        let v = a.k_cores().to_vec();
        a.drop_cache()?;
        v
    };
    Ok(array(v))
}

fn k_core_directed<I: Ix>(g: &G<I>, direction: Direction) -> Res<Out> {
    let mut a = DirectedKCores::new(g, direction)?;
    let v = ix_array(a.k_cores());
    a.drop_cache()?;
    Ok(v)
}

fn components<I: Ix>(g: &G<I>, strong: bool) -> Res<Out> {
    let (components, count, sizes) = if strong {
        let mut c = StronglyConnectedComponents::new(g)?;
        let r = (ix_array(c.components()), c.count(), c.sizes());
        c.drop_cache()?;
        r
    } else {
        let mut c = ConnectedComponents::new(g)?;
        let r = (ix_array(c.components()), c.count(), c.sizes());
        c.drop_cache()?;
        r
    };
    let sizes: Vec<i64> = sizes.into_iter().map(|s| s as i64).collect();
    Ok(dict(vec![
        ("components", components),
        ("count", number(count)),
        ("sizes", array(sizes)),
    ]))
}

fn directed_clustering<I: Ix>(g: &G<I>) -> Res<Out> {
    let mut c = DirectedClusteringCoefficient::new(g)?;
    let local = c.local_clustering_coefficients().to_vec();
    let average = c.get_average_clustering_coefficient();
    c.drop_cache()?;
    Ok(dict(vec![("local", array(local)), ("average", number(average))]))
}

fn clustering<I: Ix>(g: &G<I>) -> Res<Out> {
    let mut c = ClusteringCoefficient::new(g)?;
    let local: Vec<f64> = (0..g.size())
        .map(|u| c.get_node_clusteringcoefficient(u))
        .collect();
    let (average, transitivity) = (
        c.get_average_clustering_coefficient(),
        c.get_graph_transitivity(),
    );
    c.drop_cache()?;
    Ok(dict(vec![
        ("local", array(local)),
        ("average", number(average)),
        ("transitivity", number(transitivity)),
    ]))
}

fn louvain<I: Ix>(g: &G<I>) -> Res<Out> {
    let mut l = AlgoGVELouvain::<_, _, _, f64>::new(g)?;
    let communities = ix_array(l.communities());
    let (count, modularity) = (l.community_count(), l.partition_modularity());
    l.drop_cache()?;
    Ok(dict(vec![
        ("communities", communities),
        ("count", number(count)),
        ("modularity", number(modularity)),
    ]))
}

fn directed_louvain<I: Ix>(g: &G<I>) -> Res<Out> {
    let mut l = DirectedLouvain::new(g)?;
    let communities = ix_array(l.communities());
    let (count, modularity) = (l.community_count(), l.partition_modularity());
    l.drop_cache()?;
    Ok(dict(vec![
        ("communities", communities),
        ("count", number(count)),
        ("modularity", number(modularity)),
    ]))
}

fn hyperball_results<I: Ix, P: Precision>(
    mut hb: HyperBallInner<'_, (), (), I, P>,
    normalized: bool,
) -> Res<Out> {
    let distances = hb.geodesic_distances().to_vec();
    let harmonic_distances = hb.harmonic_geodesic_distances().to_vec();
    let closeness = hb.compute_closeness_centrality(normalized)?.to_vec();
    let harmonic = hb.compute_harmonic_centrality(normalized)?.to_vec();
    let lin = hb.compute_lins_centrality()?.to_vec();
    hb.drop_cache()?;
    Ok(dict(vec![
        ("distances", array(distances)),
        ("harmonic_distances", array(harmonic_distances)),
        ("closeness", array(closeness)),
        ("harmonic", array(harmonic)),
        ("lin", array(lin)),
    ]))
}

fn hyperball<I: Ix, P: Precision>(g: &G<I>, normalized: bool, direction: Direction) -> Res<Out> {
    hyperball_results(HyperBallInner::<_, _, _, P>::new_directed(g, direction)?, normalized)
}

#[cfg(feature = "gpu")]
fn hyperball_gpu<I: Ix, P: Precision>(
    g: &G<I>,
    normalized: bool,
    direction: Direction,
) -> Res<Out> {
    use sparx::gpu::{GpuBackend, WgpuBackend};
    let g = g.view(direction)?;
    let backend = WgpuBackend::new()?;
    let device_graph = backend.upload_graph(g)?;
    let hb = HyperBallInner::<_, _, _, P>::new_gpu(g, &backend, &device_graph, None)?;
    hyperball_results(hb, normalized)
}

fn bfs<I: Ix>(g: &G<I>, source: usize, direction: Direction) -> Res<Out> {
    let mut b = BFSDists::new_directed(g, source, direction)?;
    // unreachable nodes are left at distance 0, like the source
    let v: Vec<i64> = b
        .get_distances()
        .iter()
        .enumerate()
        .map(|(u, d)| match d.index() {
            0 if u != source => -1,
            d => d as i64,
        })
        .collect();
    b.drop_cache()?;
    Ok(array(v))
}

fn open_graph(
    path: &str,
    index: &str,
    threads: Option<u8>,
    id: Option<String>,
    reopen: bool,
) -> Res<AnyGraph> {
    macro_rules! open {
        ($I:ty) => {
            match reopen {
                true => G::<$I>::open(path, None, threads)?,
                false => G::<$I>::from_file(path, id, threads)?,
            }
        };
    }
    Ok(match index {
        "u32" => AnyGraph::U32(open!(u32)),
        "u64" => AnyGraph::U64(open!(usize)),
        _ => return Err(format!("unknown index type {index:?}: expected 'u32' or 'u64'").into()),
    })
}

/// Removes a graph's cache entry (with every file left in the cache under its id).
fn drop_graph<I: Ix>(mut g: G<I>) -> Res<()> {
    g.drop_cache()
}

fn default_threads() -> u8 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(u8::MAX as usize) as u8)
        .unwrap_or(1)
}

/// Names a temporary cache entry uniquely, so that graphs built from the same file never share
/// (and drop) each other's.
fn unique_id(path: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let stem = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!(
        "{stem}_py{}_{}_{nanos}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// A static graph, memory-mapped from `sparx`'s cache.
///
/// Algorithms run with the GIL released, one at a time per graph (they're parallel themselves).
#[pyclass(module = "sparx")]
struct Graph {
    /// `None` once closed.
    inner: Mutex<Option<AnyGraph>>,
    keep_cache: bool,
}

impl Graph {
    /// Runs `f` on the graph with the GIL released, then builds its result into Python objects.
    fn run<'py>(
        &self,
        py: Python<'py>,
        f: impl FnOnce(&AnyGraph) -> Res<Out> + Send,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| -> Result<Out, String> {
            let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let g = guard.as_ref().ok_or("the graph is closed")?;
            f(g).map_err(|e| e.to_string())
        });
        out.map_err(PyRuntimeError::new_err)?(py)
    }

    fn with<T>(&self, f: impl FnOnce(&AnyGraph) -> Res<T>) -> PyResult<T> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let g = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the graph is closed"))?;
        f(g).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn close_impl(&self) -> Res<()> {
        let taken = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
        match taken {
            Some(g) if !self.keep_cache => with_graph!(g, g => drop_graph(g)),
            _ => Ok(()),
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.close_impl();
    }
}

#[pymethods]
impl Graph {
    /// Builds a graph from a dataset (`.mtx`, `.lz4` ggcat graph, or nodes/edges files).
    ///
    /// `index` is the width of node and edge indices (`"u32"` halves memory traffic for graphs
    /// with under 4 billion nodes and edges). The graph's cache is dropped when it is closed (or
    /// garbage collected), unless `keep_cache` is true: then `Graph.open(graph.cache_file)`
    /// reopens it later without parsing the dataset again.
    #[new]
    #[pyo3(signature = (path, *, index = "u64", threads = None, keep_cache = false, id = None))]
    fn new(
        py: Python<'_>,
        path: std::path::PathBuf,
        index: &str,
        threads: Option<u8>,
        keep_cache: bool,
        id: Option<String>,
    ) -> PyResult<Self> {
        let path = path.to_string_lossy().into_owned();
        let id = id.or_else(|| (!keep_cache).then(|| unique_id(&path)));
        let threads = Some(threads.unwrap_or_else(default_threads));
        let index = index.to_owned();
        let g = py
            .detach(|| open_graph(&path, &index, threads, id, false).map_err(|e| e.to_string()))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Mutex::new(Some(g)),
            keep_cache,
        })
    }

    /// Reopens a graph kept in the cache (see `keep_cache`), from its `cache_file`.
    #[staticmethod]
    #[pyo3(signature = (cache_file, *, index = "u64", threads = None))]
    fn open(
        py: Python<'_>,
        cache_file: std::path::PathBuf,
        index: &str,
        threads: Option<u8>,
    ) -> PyResult<Self> {
        let path = cache_file.to_string_lossy().into_owned();
        let threads = Some(threads.unwrap_or_else(default_threads));
        let index = index.to_owned();
        let g = py
            .detach(|| open_graph(&path, &index, threads, None, true).map_err(|e| e.to_string()))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Mutex::new(Some(g)),
            keep_cache: true,
        })
    }

    /// Closes the graph, dropping its cache unless it was built with `keep_cache=True`.
    fn close(&self) -> PyResult<()> {
        self.close_impl()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _ty: Option<Bound<'_, PyAny>>,
        _value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    /// Number of nodes.
    #[getter]
    fn num_nodes(&self) -> PyResult<usize> {
        self.with(|g| Ok(with_graph!(g, g => g.size())))
    }

    /// Number of (directed) edges: an undirected edge is stored once in each direction.
    #[getter]
    fn num_edges(&self) -> PyResult<usize> {
        self.with(|g| Ok(with_graph!(g, g => g.width())))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_nodes()
    }

    /// Width of node and edge indices, `"u32"` or `"u64"`.
    #[getter]
    fn index(&self) -> PyResult<&'static str> {
        self.with(|g| {
            Ok(match g {
                AnyGraph::U32(_) => "u32",
                AnyGraph::U64(_) => "u64",
            })
        })
    }

    /// Number of threads algorithms run on.
    #[getter]
    fn threads(&self) -> PyResult<usize> {
        self.with(|g| Ok(with_graph!(g, g => g.thread_num())))
    }

    /// The graph's id in the cache.
    #[getter]
    fn cache_id(&self) -> PyResult<String> {
        self.with(|g| with_graph!(g, g => g.graph_id()))
    }

    /// The graph's cache file, which `Graph.open` accepts.
    #[getter]
    fn cache_file(&self) -> PyResult<String> {
        self.with(|g| Ok(with_graph!(g, g => g.cache_filename())))
    }

    /// The graph's metadata, as text.
    fn metadata(&self) -> PyResult<String> {
        self.with(|g| with_graph!(g, g => g.metadata().map(|m| m.to_string())))
    }

    fn __repr__(&self) -> String {
        match self.with(|g| Ok(with_graph!(g, g => (g.size(), g.width())))) {
            Ok((n, m)) => format!(
                "<sparx.Graph |V|={n} |E|={m} index={}>",
                self.index().unwrap_or("?")
            ),
            Err(_) => "<sparx.Graph (closed)>".to_owned(),
        }
    }

    /// Whether the graph is directed: whether some arc `u -> v` has no reverse `v -> u`
    /// (counting parallel arcs). Recorded when the graph is built. Undirected-only methods
    /// (`triangles`, `k_truss()`, `k_core()`, `clustering`, `louvain()`) refuse directed graphs.
    #[getter]
    fn is_directed(&self) -> PyResult<bool> {
        self.with(|g| Ok(with_graph!(g, g => g.is_directed()?)))
    }

    /// In-degree of node `u` (its degree, on undirected graphs).
    fn in_degree(&self, u: usize) -> PyResult<usize> {
        self.check_node(u)?;
        self.with(|g| Ok(with_graph!(g, g => g.in_degree(u)?)))
    }

    /// Out-degree of node `u`.
    fn degree(&self, u: usize) -> PyResult<usize> {
        self.check_node(u)?;
        self.with(|g| Ok(with_graph!(g, g => g.node_degree(u))))
    }

    /// Out-neighbours of node `u`, in edge order: edge `e` of `csr()[1]` is `neighbours(u)[e - offsets[u]]`.
    fn neighbours<'py>(&self, py: Python<'py>, u: usize) -> PyResult<Bound<'py, PyAny>> {
        self.check_node(u)?;
        self.run(py, |g| {
            with_graph!(g, g => {
                let v: Vec<i64> = g.neighbours(u)?.map(|v| v as i64).collect();
                Ok(array(v))
            })
        })
    }

    /// The graph's compressed sparse rows, `(offsets, targets)`: node `u`'s edges are
    /// `offsets[u]..offsets[u + 1]` and edge `e` points to `targets[e]`. Per-edge results are in
    /// this order.
    fn csr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, |g| {
            with_graph!(g, g => {
                let mut offsets = Vec::with_capacity(g.size() + 1);
                let mut targets = Vec::with_capacity(g.width());
                offsets.push(0i64);
                for u in 0..g.size() {
                    targets.extend(g.neighbours(u)?.map(|v| v as i64));
                    offsets.push(targets.len() as i64);
                }
                Ok(pair(array(offsets), array(targets)))
            })
        })
    }

    /// Number of triangles each edge is in (per edge, in `csr()` order), as `width` integers.
    #[pyo3(signature = (*, width = "u8", gpu = false))]
    fn triangles<'py>(
        &self,
        py: Python<'py>,
        width: &str,
        gpu: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let w = self::width(width)?;
        if gpu {
            return self.triangles_gpu(py, w);
        }
        self.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => triangles::<_, T>(g))),
        )
    }

    /// Number of directed triangles each edge is in, in the roles `kind` selects: `"cycle"`,
    /// `"out"`, `"in"`, `"middleman"`, `"feedforward"`, `"cyclefeedforward"`, or any other
    /// combination of roles (`"cycle_out"`, `"out_in_middleman"`, ...).
    #[pyo3(signature = (kind, *, width = "u8"))]
    fn directed_triangles<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (w, kind) = (self::width(width)?, triangle_type(kind)?);
        self.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => directed_triangles::<_, T>(g, kind))),
        )
    }

    /// Trussness of each edge: by `"pkt"` or `"burkhardt"`, of the undirected graph, or of
    /// directed triangles of a `kind` (see `directed_triangles`).
    #[pyo3(signature = (*, algorithm = "pkt", kind = None, width = "u8"))]
    fn k_truss<'py>(
        &self,
        py: Python<'py>,
        algorithm: &str,
        kind: Option<&str>,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pkt = match algorithm {
            "pkt" => true,
            "burkhardt" => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown k-truss algorithm {algorithm:?}: expected 'pkt' or 'burkhardt'"
                )));
            }
        };
        let kind = kind.map(triangle_type).transpose()?;
        let w = self::width(width)?;
        self.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => k_truss::<_, T>(g, pkt, kind))),
        )
    }

    /// Coreness of each node, by `"bz"` (Batagelj–Zaversnik) or `"liu"` (Liu et al.), of an
    /// undirected graph. With a `direction`, the cores of a directed graph by in-degree
    /// (`"in"`), out-degree (`"out"`) or both (`"both"`, networkx's `core_number`), whatever
    /// `algorithm`.
    #[pyo3(signature = (*, algorithm = "bz", direction = None))]
    fn k_core<'py>(
        &self,
        py: Python<'py>,
        algorithm: &str,
        direction: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bz = match algorithm {
            "bz" => true,
            "liu" => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown k-core algorithm {algorithm:?}: expected 'bz' or 'liu'"
                )));
            }
        };
        if let Some(d) = direction {
            let d = self::direction(d)?;
            return self.run(py, |g| with_graph!(g, g => k_core_directed(g, d)));
        }
        self.run(py, |g| with_graph!(g, g => k_core(g, bz)))
    }

    /// Clustering coefficients: `{"local": per node, "average": ..., "transitivity": ...}`.
    fn clustering<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, |g| with_graph!(g, g => clustering(g)))
    }

    /// Fagiolo's clustering coefficients of a directed graph (networkx's `clustering` on
    /// directed graphs): `{"local": per node, "average": ...}`. On undirected graphs, the same as
    /// `clustering`'s.
    fn directed_clustering<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, |g| with_graph!(g, g => directed_clustering(g)))
    }

    /// Connected components, `"weak"` (ignoring arc directions) or `"strong"`:
    /// `{"components": per node, "count": ..., "sizes": per component}`, components numbered by
    /// their smallest node. The same on undirected graphs.
    #[pyo3(signature = (*, kind = "weak"))]
    fn components<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyAny>> {
        let strong = match kind {
            "weak" => false,
            "strong" => true,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown kind of components {kind:?}: expected 'weak' or 'strong'"
                )));
            }
        };
        self.run(py, |g| with_graph!(g, g => components(g, strong)))
    }

    /// GVE-Louvain communities: `{"communities": per node, "count": ..., "modularity": ...}`.
    fn louvain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, |g| with_graph!(g, g => louvain(g)))
    }

    /// Louvain communities of a directed graph, by Leicht and Newman's directed modularity (the
    /// modularity networkx computes on directed graphs): `{"communities": per node, numbered by
    /// smallest node, "count": ..., "modularity": ...}`. Also works on undirected graphs, where
    /// `louvain` is faster.
    fn directed_louvain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.run(py, |g| with_graph!(g, g => directed_louvain(g)))
    }

    /// HyperBall estimates (per node): `distances` and `harmonic_distances` (sums of distances
    /// and of their inverses to reachable nodes), and `closeness`, `harmonic` and `lin`
    /// centralities. `normalized` normalizes closeness and harmonic centralities by each node's
    /// reachable set rather than by `|V|`. `precision` (4 to 16) trades memory (`2^precision`
    /// bytes per node) for accuracy. On directed graphs, distances follow arcs in `direction`:
    /// `"out"` (the default) from each node, `"in"` to it (networkx's convention), `"both"`
    /// ignoring directions.
    #[pyo3(signature = (*, precision = 8, normalized = true, gpu = false, direction = "out"))]
    fn hyperball<'py>(
        &self,
        py: Python<'py>,
        precision: u8,
        normalized: bool,
        gpu: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self::direction(direction)?;
        if gpu {
            return self.hyperball_gpu(py, precision, normalized, d);
        }
        with_precision!(precision, P => self.run(py, |g| {
            with_graph!(g, g => hyperball::<_, P>(g, normalized, d))
        }))?
    }

    /// Exact closeness centrality of each node. On directed graphs, distances follow arcs in
    /// `direction` (see `hyperball`).
    #[pyo3(signature = (*, normalized = true, direction = "out"))]
    fn closeness<'py>(
        &self,
        py: Python<'py>,
        normalized: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self::direction(direction)?;
        self.run(py, |g| {
            with_graph!(g, g => {
                let mut c =
                    ExactClosenessCentrality::<_, _, _, f64>::new_directed(g, Some(normalized), d)?;
                let v = c.centralities().to_vec();
                c.drop_cache()?;
                Ok(array(v))
            })
        })
    }

    /// Exact harmonic centrality of each node. On directed graphs, distances follow arcs in
    /// `direction` (see `hyperball`).
    #[pyo3(signature = (*, normalized = true, direction = "out"))]
    fn harmonic<'py>(
        &self,
        py: Python<'py>,
        normalized: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self::direction(direction)?;
        self.run(py, |g| {
            with_graph!(g, g => {
                let mut c =
                    ExactHarmonicCentrality::<_, _, _, f64>::new_directed(g, Some(normalized), d)?;
                let v = c.centralities().to_vec();
                c.drop_cache()?;
                Ok(array(v))
            })
        })
    }

    /// Exact Lin's centrality of each node. On directed graphs, distances follow arcs in
    /// `direction` (see `hyperball`).
    #[pyo3(signature = (*, direction = "out"))]
    fn lin<'py>(&self, py: Python<'py>, direction: &str) -> PyResult<Bound<'py, PyAny>> {
        let d = self::direction(direction)?;
        self.run(py, |g| {
            with_graph!(g, g => {
                let mut c = ExactLinCentrality::<_, _, _, f64>::new_directed(g, d)?;
                let v = c.centralities().to_vec();
                c.drop_cache()?;
                Ok(array(v))
            })
        })
    }

    /// BFS distance of each node from `source` (`-1` for nodes it doesn't reach). On directed
    /// graphs it follows arcs in `direction`: `"out"` (the default) gives distances from
    /// `source`, `"in"` to it, `"both"` ignores directions.
    #[pyo3(signature = (source, *, direction = "out"))]
    fn bfs<'py>(
        &self,
        py: Python<'py>,
        source: usize,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_node(source)?;
        let d = self::direction(direction)?;
        self.run(py, |g| with_graph!(g, g => bfs(g, source, d)))
    }
}

impl Graph {
    fn check_node(&self, u: usize) -> PyResult<()> {
        let n = self.num_nodes()?;
        if u >= n {
            return Err(PyValueError::new_err(format!(
                "node {u} out of range: the graph has {n} nodes"
            )));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn triangles_gpu<'py>(&self, py: Python<'py>, w: Width) -> PyResult<Bound<'py, PyAny>> {
        self.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => triangles_gpu::<_, T>(g))),
        )
    }

    #[cfg(not(feature = "gpu"))]
    fn triangles_gpu<'py>(&self, _py: Python<'py>, _w: Width) -> PyResult<Bound<'py, PyAny>> {
        Err(no_gpu())
    }

    #[cfg(feature = "gpu")]
    fn hyperball_gpu<'py>(
        &self,
        py: Python<'py>,
        precision: u8,
        normalized: bool,
        direction: Direction,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_precision!(precision, P => self.run(py, |g| {
            with_graph!(g, g => hyperball_gpu::<_, P>(g, normalized, direction))
        }))?
    }

    #[cfg(not(feature = "gpu"))]
    fn hyperball_gpu<'py>(
        &self,
        _py: Python<'py>,
        _precision: u8,
        _normalized: bool,
        _direction: Direction,
    ) -> PyResult<Bound<'py, PyAny>> {
        Err(no_gpu())
    }
}

#[cfg(not(feature = "gpu"))]
fn no_gpu() -> PyErr {
    PyValueError::new_err("sparx was built without GPU support (the `gpu` feature)")
}

#[pymodule]
fn _sparx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Graph>()?;
    m.add(
        "DIRECTED_TRIANGLE_TYPES",
        DirectedTriangleType::ALL.map(|t| t.name()).to_vec(),
    )?;
    m.add("GPU", cfg!(feature = "gpu"))?;
    Ok(())
}
