//! Python bindings of `sparx`: an undirected `Graph` and a directed `DiGraph` class, whose
//! methods run the library's algorithms and return their results as `numpy` arrays.
//!
//! Both classes wrap an [`Inner`], generic over the graph's [`Directedness`], and share their
//! common methods through [`graph_class!`], which emits one `#[pymethods]` block per class with
//! the shared methods and the class's own ones.
//!
//! Every algorithm caches its results in files next to the graph's; the bindings copy them out and
//! drop those files right away, so that the cache only ever holds the graph itself (and graphs
//! derived from it: a directed graph's transpose and underlying undirected graph), dropped too
//! when the graph is closed, unless it was built with `keep_cache=True`.

use std::collections::HashSet;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
use sparx::graph::{
    CacheFile, Directed, Directedness, GraphMemoryMap, IndexType, Undirected, UndirectedMode,
};
use sparx::k_core::{
    batagelj_zaversnik::AlgoBatageljZaversnik,
    count::CoreCount,
    directed::{CoreDegree, DirectedKCores},
    liu_et_al::AlgoLiuEtAl,
};
use sparx::k_truss::clustering_coefficient::{
    ClusteringCoefficient, DirectedClusteringCoefficient,
};
use sparx::k_truss::count::TriangleCount;
use sparx::k_truss::directed::{DirectedTriangleType, DirectedTriangles};
use sparx::k_truss::{burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT, triangles::Triangles};
use sparx::trails::bfs::BFSDists;

type G<Ix, Dir> = GraphMemoryMap<(), (), Ix, Dir>;
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

/// A graph of either index type.
enum AnyG<Dir: Directedness> {
    U32(G<u32, Dir>),
    U64(G<usize, Dir>),
}

/// Runs `$body` on the graph, whatever its index type, as `$g: &G<Ix, Dir>`.
macro_rules! with_graph {
    ($any:expr, $g:ident => $body:expr) => {
        match $any {
            AnyG::U32($g) => $body,
            AnyG::U64($g) => $body,
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

/// The arcs a directed graph's traversals and centralities follow: out of nodes, into them, or
/// either way (on the underlying undirected graph).
#[derive(Clone, Copy)]
enum Follow {
    Out,
    In,
    Both,
}

fn follow(name: &str) -> PyResult<Follow> {
    match name {
        "out" => Ok(Follow::Out),
        "in" => Ok(Follow::In),
        "both" => Ok(Follow::Both),
        _ => Err(PyValueError::new_err(format!(
            "unknown direction {name:?}: expected 'out', 'in' or 'both'"
        ))),
    }
}

/// Runs `$body` on the graph whose out-neighbours are `$g`'s neighbours in direction `$d`: `$g`
/// itself, its transpose or its (collapsed: distances don't depend on multiplicity) underlying
/// undirected graph, as `$v` (whose node ids are `$g`'s).
macro_rules! with_view {
    ($g:expr, $d:expr, $v:ident => $body:expr) => {
        match $d {
            Follow::Out => {
                let $v = $g;
                $body
            }
            Follow::In => {
                let $v = $g.transpose()?;
                $body
            }
            Follow::Both => {
                let $v = $g.to_undirected()?;
                $body
            }
        }
    };
}

fn undirected_mode(name: &str) -> PyResult<UndirectedMode> {
    match name {
        "collapse" => Ok(UndirectedMode::Collapse),
        "each" => Ok(UndirectedMode::Each),
        "mutual" => Ok(UndirectedMode::Mutual),
        _ => Err(PyValueError::new_err(format!(
            "unknown undirected mode {name:?}: expected 'collapse', 'each' or 'mutual'"
        ))),
    }
}

fn core_degree(name: &str) -> PyResult<CoreDegree> {
    match name {
        "in" => Ok(CoreDegree::In),
        "out" => Ok(CoreDegree::Out),
        "total" => Ok(CoreDegree::Total),
        _ => Err(PyValueError::new_err(format!(
            "unknown core degree {name:?}: expected 'in', 'out' or 'total'"
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

/// Whether a k-truss algorithm's name is PKT's (else Burkhardt et al.'s).
fn truss_algorithm(algorithm: &str) -> PyResult<bool> {
    match algorithm {
        "pkt" => Ok(true),
        "burkhardt" => Ok(false),
        _ => Err(PyValueError::new_err(format!(
            "unknown k-truss algorithm {algorithm:?}: expected 'pkt' or 'burkhardt'"
        ))),
    }
}

fn component_kind(kind: &str) -> PyResult<bool> {
    match kind {
        "weak" => Ok(false),
        "strong" => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "unknown kind of components {kind:?}: expected 'weak' or 'strong'"
        ))),
    }
}

// --- algorithms -------------------------------------------------------------------------------

/// The files in the cache named after `g`'s cache entry (`<kind>[_<seq>]_<hashed id>.<ext>`, in
/// the cache directory or its subdirectories for some kinds): its own, and its algorithms'
/// results and scratch files.
fn files_of<I: Ix, Dir: Directedness>(g: &G<I, Dir>) -> Res<HashSet<PathBuf>> {
    let mut files = HashSet::new();
    // a file of each directory algorithm files go to, named after the entry's hashed id
    for kind in [CacheFile::General, CacheFile::ExactCloseness] {
        let file = PathBuf::from(g.build_cache_filename(kind, None)?);
        let (Some(dir), Some(stem)) = (file.parent(), file.file_stem().and_then(|s| s.to_str()))
        else {
            continue;
        };
        let Some((_, hashed)) = stem.rsplit_once('_') else {
            continue;
        };
        let suffix = format!("_{hashed}.");
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        files.extend(entries.flatten().map(|e| e.path()).filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(&suffix))
        }));
    }
    Ok(files)
}

/// Runs an algorithm on `g`, removing the files it created in the cache if it fails (its result
/// and scratch files: algorithms only drop them on success). Files that were there before (the
/// graph's own among them) and other graphs' (derived ones included, named after their own ids)
/// are left alone. Algorithms on one graph run one at a time (see `Inner`), so the new files
/// are this call's.
fn cleaning<I: Ix, Dir: Directedness>(g: &G<I, Dir>, f: impl FnOnce() -> Res<Out>) -> Res<Out> {
    let before = files_of(g)?;
    let result = f();
    if result.is_err()
        && let Ok(after) = files_of(g)
    {
        // but the registry of graphs derived from `g` (views an algorithm built before failing,
        // valid still): without it, they'd outlive `g`
        let registry = |f: &PathBuf| {
            f.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("derived_"))
        };
        for file in after.difference(&before).filter(|f| !registry(f)) {
            let _ = std::fs::remove_file(file);
        }
    }
    result
}

fn triangles<I: Ix, Tx: TriangleCount + Element>(g: &G<I, Undirected>) -> Res<Out> {
    cleaning(g, || {
        let mut t = Triangles::<_, _, _, Tx>::new_tx(g)?;
        let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
        t.drop_cache()?;
        Ok(array(v))
    })
}

#[cfg(feature = "gpu")]
fn triangles_gpu<I: Ix, Tx: TriangleCount + Element>(g: &G<I, Undirected>) -> Res<Out> {
    cleaning(g, || {
        use sparx::gpu::{GpuBackend, WgpuBackend};
        let backend = WgpuBackend::new()?;
        let device_graph = backend.upload_graph(g)?;
        let mut t = Triangles::<_, _, _, Tx>::new_gpu_tx(g, &backend, &device_graph)?;
        let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
        t.drop_cache()?;
        Ok(array(v))
    })
}

fn directed_triangles<I: Ix, Tx: TriangleCount + Element>(
    g: &G<I, Directed>,
    kind: DirectedTriangleType,
) -> Res<Out> {
    cleaning(g, || {
        let mut t = DirectedTriangles::<_, _, _, Tx>::new_tx(g, kind)?;
        let v: Vec<Tx> = (0..g.width()).map(|e| t.triangles(e)).collect();
        t.drop_cache()?;
        Ok(array(v))
    })
}

fn k_truss<I: Ix, Tx: TriangleCount + Element>(g: &G<I, Undirected>, pkt: bool) -> Res<Out> {
    cleaning(g, || {
        let v = if pkt {
            let mut a = AlgoPKT::<_, _, _, Tx>::new_tx(g, g.thread_num())?;
            let v = a.k_trusses().to_vec();
            a.drop_cache()?;
            v
        } else {
            let mut a = AlgoBurkhardtEtAl::<_, _, _, Tx>::new_tx(g)?;
            let v = a.k_trusses().to_vec();
            a.drop_cache()?;
            v
        };
        Ok(array(v))
    })
}

fn directed_k_truss<I: Ix, Tx: TriangleCount + Element>(
    g: &G<I, Directed>,
    pkt: bool,
    kind: DirectedTriangleType,
) -> Res<Out> {
    cleaning(g, || {
        let v = if pkt {
            let mut a = AlgoPKT::<_, _, _, Tx, _>::new_directed_tx(g, kind, g.thread_num())?;
            let v = a.k_trusses().to_vec();
            a.drop_cache()?;
            v
        } else {
            let mut a = AlgoBurkhardtEtAl::<_, _, _, Tx, _>::new_directed_tx(g, kind)?;
            let v = a.k_trusses().to_vec();
            a.drop_cache()?;
            v
        };
        Ok(array(v))
    })
}

fn k_core<I: Ix, Cx: CoreCount + Element>(g: &G<I, Undirected>, bz: bool) -> Res<Out> {
    cleaning(g, || {
        let v = if bz {
            let mut a = AlgoBatageljZaversnik::<_, _, _, Cx>::new_cx(g)?;
            let v = a.k_cores().to_vec();
            a.drop_cache()?;
            v
        } else {
            let mut a = AlgoLiuEtAl::<_, _, _, Cx>::new_cx(g, g.thread_num())?;
            let v = a.k_cores().to_vec();
            a.drop_cache()?;
            v
        };
        Ok(array(v))
    })
}

fn directed_k_core<I: Ix, Cx: CoreCount + Element>(
    g: &G<I, Directed>,
    degree: CoreDegree,
) -> Res<Out> {
    cleaning(g, || {
        let mut a = DirectedKCores::<_, _, _, Cx>::new_cx(g, degree)?;
        let v = array(a.k_cores().to_vec());
        a.drop_cache()?;
        Ok(v)
    })
}

fn components<I: Ix, Dir: Directedness>(g: &G<I, Dir>, strong: bool) -> Res<Out> {
    cleaning(g, || {
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
    })
}

fn clustering<I: Ix>(g: &G<I, Undirected>) -> Res<Out> {
    cleaning(g, || {
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
    })
}

fn directed_clustering<I: Ix>(g: &G<I, Directed>) -> Res<Out> {
    cleaning(g, || {
        let mut c = DirectedClusteringCoefficient::new(g)?;
        let local = c.local_clustering_coefficients().to_vec();
        let average = c.get_average_clustering_coefficient();
        c.drop_cache()?;
        Ok(dict(vec![
            ("local", array(local)),
            ("average", number(average)),
        ]))
    })
}

fn louvain<I: Ix>(g: &G<I, Undirected>) -> Res<Out> {
    cleaning(g, || {
        let mut l = AlgoGVELouvain::<_, _, _, f64>::new(g)?;
        let communities = ix_array(l.communities());
        let (count, modularity) = (l.community_count(), l.partition_modularity());
        l.drop_cache()?;
        Ok(dict(vec![
            ("communities", communities),
            ("count", number(count)),
            ("modularity", number(modularity)),
        ]))
    })
}

fn directed_louvain<I: Ix>(g: &G<I, Directed>) -> Res<Out> {
    cleaning(g, || {
        let mut l = DirectedLouvain::new(g)?;
        let communities = ix_array(l.communities());
        let (count, modularity) = (l.community_count(), l.partition_modularity());
        l.drop_cache()?;
        Ok(dict(vec![
            ("communities", communities),
            ("count", number(count)),
            ("modularity", number(modularity)),
        ]))
    })
}

fn hyperball_results<I: Ix, P: Precision, Dir: Directedness>(
    mut hb: HyperBallInner<'_, (), (), I, P, f64, Dir>,
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

fn hyperball<I: Ix, P: Precision, Dir: Directedness>(g: &G<I, Dir>, normalized: bool) -> Res<Out> {
    cleaning(g, || {
        hyperball_results(HyperBallInner::<_, _, _, P, f64, _>::new(g)?, normalized)
    })
}

#[cfg(feature = "gpu")]
fn hyperball_gpu<I: Ix, P: Precision, Dir: Directedness>(
    g: &G<I, Dir>,
    normalized: bool,
) -> Res<Out> {
    cleaning(g, || {
        use sparx::gpu::{GpuBackend, WgpuBackend};
        let backend = WgpuBackend::new()?;
        let device_graph = backend.upload_graph(g)?;
        let hb = HyperBallInner::<_, _, _, P, f64, _>::new_gpu(g, &backend, &device_graph, None)?;
        hyperball_results(hb, normalized)
    })
}

fn closeness<I: Ix, Dir: Directedness>(g: &G<I, Dir>, normalized: bool) -> Res<Out> {
    cleaning(g, || {
        let mut c = ExactClosenessCentrality::<_, _, _, f64, _>::new(g, Some(normalized))?;
        let v = c.centralities().to_vec();
        c.drop_cache()?;
        Ok(array(v))
    })
}

fn harmonic<I: Ix, Dir: Directedness>(g: &G<I, Dir>, normalized: bool) -> Res<Out> {
    cleaning(g, || {
        let mut c = ExactHarmonicCentrality::<_, _, _, f64, _>::new(g, Some(normalized))?;
        let v = c.centralities().to_vec();
        c.drop_cache()?;
        Ok(array(v))
    })
}

fn lin<I: Ix, Dir: Directedness>(g: &G<I, Dir>) -> Res<Out> {
    cleaning(g, || {
        let mut c = ExactLinCentrality::<_, _, _, f64, _>::new(g)?;
        let v = c.centralities().to_vec();
        c.drop_cache()?;
        Ok(array(v))
    })
}

fn bfs<I: Ix, Dir: Directedness>(g: &G<I, Dir>, source: usize) -> Res<Out> {
    cleaning(g, || {
        let mut b = BFSDists::new(g, source)?;
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
    })
}

// --- graphs -----------------------------------------------------------------------------------

fn open_graph<Dir: Directedness>(
    path: &str,
    index: &str,
    threads: Option<u8>,
    id: Option<String>,
    reopen: bool,
) -> Res<AnyG<Dir>> {
    macro_rules! open {
        ($I:ty) => {
            match reopen {
                true => G::<$I, Dir>::open(path, None, threads)?,
                false => G::<$I, Dir>::from_file(path, id, threads)?,
            }
        };
    }
    Ok(match index {
        "u32" => AnyG::U32(open!(u32)),
        "u64" => AnyG::U64(open!(usize)),
        _ => return Err(format!("unknown index type {index:?}: expected 'u32' or 'u64'").into()),
    })
}

/// Opens the cache entry of `g`'s graph as a graph of type `D2`, with `g`'s index type and threads.
fn reopen_as<D2: Directedness, Dir: Directedness>(
    g: &AnyG<Dir>,
    cache_file: impl Fn(&AnyG<Dir>) -> Res<String>,
) -> Res<AnyG<D2>> {
    let file = cache_file(g)?;
    let threads = Some(with_graph!(g, g => g.thread_num()).min(u8::MAX as usize) as u8);
    Ok(match g {
        AnyG::U32(_) => AnyG::U32(G::<u32, D2>::open(file, None, threads)?),
        AnyG::U64(_) => AnyG::U64(G::<usize, D2>::open(file, None, threads)?),
    })
}

/// Removes a graph's cache entry (with every file left in the cache under its id, and the graphs
/// derived from it).
fn drop_graph<I: Ix, Dir: Directedness>(mut g: G<I, Dir>) -> Res<()> {
    g.drop_cache()
}

fn check_index(index: &str) -> PyResult<()> {
    match index {
        "u32" | "u64" => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "unknown index type {index:?}: expected 'u32' or 'u64'"
        ))),
    }
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

/// A graph another one was converted from (by `to_directed` or `to_undirected`), whose cache
/// entry the converted one maps, without owning it.
trait Parent: Send + Sync {
    /// Runs `f` while holding the parent's lock (so that it isn't closed, nor runs an algorithm
    /// writing the same cache files, meanwhile), if it's still open.
    fn while_open(&self, f: &mut dyn FnMut()) -> bool;
}

impl<Dir: Directedness> Parent for Mutex<Option<AnyG<Dir>>> {
    fn while_open(&self, f: &mut dyn FnMut()) -> bool {
        let guard = self.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            return false;
        }
        f();
        true
    }
}

/// Runs `f` holding the locks of every one of `ancestors` (parent first: a graph is always
/// locked before the ones it was converted from, so locks are always taken in the same order),
/// if they're all open.
fn while_open(ancestors: &[Arc<dyn Parent>], f: &mut dyn FnMut()) -> bool {
    match ancestors.split_first() {
        None => {
            f();
            true
        }
        Some((parent, rest)) => {
            let mut open = false;
            parent.while_open(&mut || open = while_open(rest, &mut *f)) && open
        }
    }
}

/// What both graph classes hold: the graph (`None` once closed), and whether to keep its cache
/// entry when closing.
struct Inner<Dir: Directedness> {
    graph: Arc<Mutex<Option<AnyG<Dir>>>>,
    keep_cache: bool,
    /// The graphs this one was converted from, nearest first: it maps their cache entry, which
    /// they own, so it's only usable while they're open.
    ancestors: Vec<Arc<dyn Parent>>,
}

impl<Dir: Directedness> Inner<Dir> {
    fn new(g: AnyG<Dir>, keep_cache: bool) -> Self {
        Self {
            graph: Arc::new(Mutex::new(Some(g))),
            keep_cache,
            ancestors: Vec::new(),
        }
    }

    /// Builds or reopens a graph with the GIL released.
    fn load(
        py: Python<'_>,
        path: &Path,
        index: &str,
        threads: Option<u8>,
        keep_cache: bool,
        id: Option<String>,
        reopen: bool,
    ) -> PyResult<Self> {
        check_index(index)?;
        let path = path.to_string_lossy().into_owned();
        let id = match reopen {
            true => None,
            false => id.or_else(|| (!keep_cache).then(|| unique_id(&path))),
        };
        let threads = Some(threads.unwrap_or_else(default_threads));
        let g = py
            .detach(|| {
                open_graph::<Dir>(&path, index, threads, id, reopen).map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self::new(g, keep_cache || reopen))
    }

    /// Runs `f` on the graph, if it and the graphs it was converted from are open.
    fn access<T>(&self, f: impl FnOnce(&AnyG<Dir>) -> Res<T>) -> Result<T, String> {
        let guard = self.graph.lock().unwrap_or_else(|e| e.into_inner());
        let g = guard.as_ref().ok_or("the graph is closed")?;
        let mut f = Some(f);
        let mut out = None;
        if !while_open(&self.ancestors, &mut || {
            out = f.take().map(|f| f(g));
        }) {
            return Err("the graph this one was converted from is closed".to_owned());
        }
        match out {
            Some(r) => r.map_err(|e| e.to_string()),
            None => Err("the graph this one was converted from is closed".to_owned()),
        }
    }

    /// Runs `f` on the graph with the GIL released, then builds its result into Python objects.
    fn run<'py>(
        &self,
        py: Python<'py>,
        f: impl FnOnce(&AnyG<Dir>) -> Res<Out> + Send,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| self.access(f));
        out.map_err(PyRuntimeError::new_err)?(py)
    }

    /// Runs `f` on the graph, holding the GIL: for what's instant.
    fn with<T>(&self, f: impl FnOnce(&AnyG<Dir>) -> Res<T>) -> PyResult<T> {
        self.access(f).map_err(PyRuntimeError::new_err)
    }

    /// Runs `f` on the graph with the GIL released: for what may take a while (reading files,
    /// building a derived graph, scanning the arcs).
    fn with_detached<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&AnyG<Dir>) -> Res<T> + Send,
    ) -> PyResult<T> {
        py.detach(|| self.access(f))
            .map_err(PyRuntimeError::new_err)
    }

    fn check_node(&self, u: usize) -> PyResult<()> {
        let n = self.with(|g| Ok(with_graph!(g, g => g.size())))?;
        if u >= n {
            return Err(PyValueError::new_err(format!(
                "node {u} out of range: the graph has {n} nodes"
            )));
        }
        Ok(())
    }

    /// A graph of type `D2` mapping the cache entry `cache_file` gives, owned by this graph: it
    /// never drops it, and is only usable while this graph is open.
    fn convert<D2: Directedness>(
        &self,
        py: Python<'_>,
        cache_file: impl Fn(&AnyG<Dir>) -> Res<String> + Send,
    ) -> PyResult<Inner<D2>> {
        let g = py
            .detach(|| self.access(|g| reopen_as::<D2, Dir>(g, cache_file)))
            .map_err(PyRuntimeError::new_err)?;
        let mut ancestors: Vec<Arc<dyn Parent>> = vec![self.graph.clone()];
        ancestors.extend(self.ancestors.iter().cloned());
        Ok(Inner {
            graph: Arc::new(Mutex::new(Some(g))),
            keep_cache: true,
            ancestors,
        })
    }

    fn close(&self) -> Res<()> {
        let taken = self.graph.lock().unwrap_or_else(|e| e.into_inner()).take();
        match taken {
            Some(g) if !self.keep_cache => with_graph!(g, g => drop_graph(g)),
            _ => Ok(()),
        }
    }
}

impl<Dir: Directedness> Drop for Inner<Dir> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(not(feature = "gpu"))]
fn no_gpu() -> PyErr {
    PyValueError::new_err("sparx was built without GPU support (the `gpu` feature)")
}

/// Emits a graph class's `#[pymethods]`: the methods both classes share, followed by the class's
/// own (`$own`), in a single block (pyo3 allows one per class without its `multiple-pymethods`
/// feature).
macro_rules! graph_class {
    ($Class:ident, $name:literal, { $($own:tt)* }) => {
        #[pymethods]
        impl $Class {
            /// Builds a graph from a dataset (`.mtx`, `.lz4` ggcat graph, or nodes/edges files).
            ///
            /// `index` is the width of node and edge indices (`"u32"` halves memory traffic for
            /// graphs with under 4 billion nodes and edges). The graph's cache is dropped when it
            /// is closed (or garbage collected), unless `keep_cache` is true: then
            /// `open(graph.cache_file)` reopens it later without parsing the dataset again.
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
                let inner = Inner::load(py, &path, index, threads, keep_cache, id, false)?;
                Ok(Self { inner })
            }

            /// Reopens a graph kept in the cache (see `keep_cache`), from its `cache_file`. The
            /// reopened graph keeps its cache too.
            #[staticmethod]
            #[pyo3(signature = (cache_file, *, index = "u64", threads = None))]
            fn open(
                py: Python<'_>,
                cache_file: std::path::PathBuf,
                index: &str,
                threads: Option<u8>,
            ) -> PyResult<Self> {
                let inner = Inner::load(py, &cache_file, index, threads, true, None, true)?;
                Ok(Self { inner })
            }

            /// Closes the graph, dropping its cache unless it was built with `keep_cache=True`
            /// (or opened from the cache, or converted from another graph).
            fn close(&self) -> PyResult<()> {
                self.inner
                    .close()
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
                self.inner.with(|g| Ok(with_graph!(g, g => g.size())))
            }

            /// Number of arcs: an undirected edge is stored as an arc each way (a self-loop once).
            #[getter]
            fn num_edges(&self) -> PyResult<usize> {
                self.inner.with(|g| Ok(with_graph!(g, g => g.width())))
            }

            fn __len__(&self) -> PyResult<usize> {
                self.num_nodes()
            }

            /// Width of node and edge indices, `"u32"` or `"u64"`.
            #[getter]
            fn index(&self) -> PyResult<&'static str> {
                self.inner.with(|g| {
                    Ok(match g {
                        AnyG::U32(_) => "u32",
                        AnyG::U64(_) => "u64",
                    })
                })
            }

            /// Number of threads algorithms run on.
            #[getter]
            fn threads(&self) -> PyResult<usize> {
                self.inner.with(|g| Ok(with_graph!(g, g => g.thread_num())))
            }

            /// The graph's id in the cache.
            #[getter]
            fn cache_id(&self) -> PyResult<String> {
                self.inner.with(|g| with_graph!(g, g => g.graph_id()))
            }

            /// The graph's cache file, which `open` accepts.
            #[getter]
            fn cache_file(&self) -> PyResult<String> {
                self.inner.with(|g| Ok(with_graph!(g, g => g.cache_filename())))
            }

            /// The graph's metadata, as text.
            fn metadata(&self, py: Python<'_>) -> PyResult<String> {
                self.inner.with_detached(py, |g| with_graph!(g, g => g.metadata().map(|m| m.to_string())))
            }

            fn __repr__(&self) -> String {
                match self.inner.with(|g| Ok(with_graph!(g, g => (g.size(), g.width())))) {
                    Ok((n, m)) => format!(
                        concat!("<sparx.", $name, " |V|={} |E|={} index={}>"),
                        n,
                        m,
                        self.index().unwrap_or("?")
                    ),
                    Err(_) => concat!("<sparx.", $name, " (closed)>").to_owned(),
                }
            }

            /// Whether this is a directed graph class (`DiGraph`): see `is_symmetric` for whether
            /// its arcs are.
            #[getter]
            fn is_directed(&self) -> bool {
                <Self as GraphClass>::Dir::DIRECTED
            }

            /// Whether every arc `u -> v` has a reverse `v -> u`, counting parallel arcs
            /// (always, for a `Graph`). Recorded when the graph is built.
            #[getter]
            fn is_symmetric(&self, py: Python<'_>) -> PyResult<bool> {
                self.inner.with_detached(py, |g| with_graph!(g, g => g.is_symmetric()))
            }

            /// Degree of node `u`: its number of out-arcs, on a `DiGraph`.
            fn degree(&self, u: usize) -> PyResult<usize> {
                self.inner.check_node(u)?;
                self.inner.with(|g| Ok(with_graph!(g, g => g.node_degree(u))))
            }

            /// Neighbours (out-neighbours, on a `DiGraph`) of node `u`, in edge order: edge `e` of
            /// `csr()[1]` is `neighbours(u)[e - offsets[u]]`.
            fn neighbours<'py>(&self, py: Python<'py>, u: usize) -> PyResult<Bound<'py, PyAny>> {
                self.inner.check_node(u)?;
                self.inner.run(py, |g| {
                    with_graph!(g, g => {
                        let v: Vec<i64> = g.neighbours(u)?.map(|v| v as i64).collect();
                        Ok(array(v))
                    })
                })
            }

            /// The graph's compressed sparse rows, `(offsets, targets)`: node `u`'s arcs are
            /// `offsets[u]..offsets[u + 1]` and arc `e` points to `targets[e]`. Per-edge results
            /// are in this order.
            fn csr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.inner.run(py, |g| {
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

            /// Connected components, `"weak"` (ignoring arc directions) or `"strong"`:
            /// `{"components": per node, "count": ..., "sizes": per component}`, components
            /// numbered by their smallest node. Both are the same on a `Graph`.
            #[pyo3(signature = (*, kind = "weak"))]
            fn components<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyAny>> {
                let strong = component_kind(kind)?;
                self.inner.run(py, |g| with_graph!(g, g => components(g, strong)))
            }

            $($own)*
        }
    };
}

/// The directedness of a graph class's graphs.
trait GraphClass {
    type Dir: Directedness;
}

/// An undirected graph, memory-mapped from `sparx`'s cache: every edge `{u, v}` is stored as the
/// arcs `u -> v` and `v -> u`. Loading a directed dataset (one with an arc without a reverse)
/// fails: load it as a `DiGraph`.
///
/// Algorithms run with the GIL released, one at a time per graph (they're parallel themselves).
#[pyclass(module = "sparx", name = "Graph")]
struct UGraph {
    inner: Inner<Undirected>,
}

impl GraphClass for UGraph {
    type Dir = Undirected;
}

/// A directed graph, memory-mapped from `sparx`'s cache. Any dataset loads as one (an undirected
/// one with each edge as an arc each way). Its transpose and underlying undirected graph are
/// built once, when first needed, and cached alongside it.
///
/// Algorithms run with the GIL released, one at a time per graph (they're parallel themselves).
#[pyclass(module = "sparx", name = "DiGraph")]
struct DGraph {
    inner: Inner<Directed>,
}

impl GraphClass for DGraph {
    type Dir = Directed;
}

graph_class!(UGraph, "Graph", {
    /// Number of triangles each edge is in (per arc, in `csr()` order), as `width` integers.
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
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => triangles::<_, T>(g))),
        )
    }

    /// Trussness of each edge (per arc, in `csr()` order), by `"pkt"` or `"burkhardt"`, as
    /// `width` integers.
    #[pyo3(signature = (*, algorithm = "pkt", width = "u8"))]
    fn k_truss<'py>(
        &self,
        py: Python<'py>,
        algorithm: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (pkt, w) = (truss_algorithm(algorithm)?, self::width(width)?);
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => k_truss::<_, T>(g, pkt))),
        )
    }

    /// Coreness of each node, by `"bz"` (Batagelj–Zaversnik) or `"liu"` (Liu et al.), as `width`
    /// integers (every node's degree must fit one).
    #[pyo3(signature = (*, algorithm = "bz", width = "u8"))]
    fn k_core<'py>(
        &self,
        py: Python<'py>,
        algorithm: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let w = self::width(width)?;
        let bz = match algorithm {
            "bz" => true,
            "liu" => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown k-core algorithm {algorithm:?}: expected 'bz' or 'liu'"
                )));
            }
        };
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => k_core::<_, T>(g, bz))),
        )
    }

    /// Clustering coefficients: `{"local": per node, "average": ..., "transitivity": ...}`.
    fn clustering<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.run(py, |g| with_graph!(g, g => clustering(g)))
    }

    /// GVE-Louvain communities: `{"communities": per node, "count": ..., "modularity": ...}`.
    fn louvain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.run(py, |g| with_graph!(g, g => louvain(g)))
    }

    /// HyperBall estimates (per node): `distances` and `harmonic_distances` (sums of distances
    /// and of their inverses to reachable nodes), and `closeness`, `harmonic` and `lin`
    /// centralities. `normalized` normalizes closeness and harmonic centralities by each node's
    /// reachable set rather than by `|V|`. `precision` (4 to 16) trades memory (`2^precision`
    /// bytes per node) for accuracy.
    #[pyo3(signature = (*, precision = 8, normalized = true, gpu = false))]
    fn hyperball<'py>(
        &self,
        py: Python<'py>,
        precision: u8,
        normalized: bool,
        gpu: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if gpu {
            return self.hyperball_gpu(py, precision, normalized);
        }
        with_precision!(precision, P => self.inner.run(py, |g| {
            with_graph!(g, g => hyperball::<_, P, _>(g, normalized))
        }))?
    }

    /// Exact closeness centrality of each node.
    #[pyo3(signature = (*, normalized = true))]
    fn closeness<'py>(&self, py: Python<'py>, normalized: bool) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .run(py, |g| with_graph!(g, g => closeness(g, normalized)))
    }

    /// Exact harmonic centrality of each node.
    #[pyo3(signature = (*, normalized = true))]
    fn harmonic<'py>(&self, py: Python<'py>, normalized: bool) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .run(py, |g| with_graph!(g, g => harmonic(g, normalized)))
    }

    /// Exact Lin's centrality of each node.
    fn lin<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.run(py, |g| with_graph!(g, g => lin(g)))
    }

    /// BFS distance of each node from `source` (`-1` for nodes it doesn't reach).
    #[pyo3(signature = (source))]
    fn bfs<'py>(&self, py: Python<'py>, source: usize) -> PyResult<Bound<'py, PyAny>> {
        self.inner.check_node(source)?;
        self.inner.run(py, |g| with_graph!(g, g => bfs(g, source)))
    }

    /// The graph as a `DiGraph`, each edge an arc each way: for directed algorithms. It maps
    /// this graph's cache entry, which this graph owns: it never drops it, and is only usable
    /// while this graph is open (it raises `RuntimeError` after).
    fn to_directed(&self, py: Python<'_>) -> PyResult<DGraph> {
        let inner = self
            .inner
            .convert::<Directed>(py, |g| Ok(with_graph!(g, g => g.cache_filename())))?;
        Ok(DGraph { inner })
    }
});

graph_class!(DGraph, "DiGraph", {
    /// Number of arcs into node `u`.
    fn in_degree(&self, py: Python<'_>, u: usize) -> PyResult<usize> {
        self.inner.check_node(u)?;
        self.inner
            .with_detached(py, |g| Ok(with_graph!(g, g => g.in_degree(u)?)))
    }

    /// In-neighbours of node `u`, sorted.
    fn in_neighbours<'py>(&self, py: Python<'py>, u: usize) -> PyResult<Bound<'py, PyAny>> {
        self.inner.check_node(u)?;
        self.inner.run(py, |g| {
            with_graph!(g, g => {
                let v: Vec<i64> = g.in_neighbours(u)?.map(|v| v as i64).collect();
                Ok(array(v))
            })
        })
    }

    /// Number of directed triangles each arc is in (in `csr()` order), in the roles `kind`
    /// selects: `"cycle"`, `"out"`, `"in"`, `"middleman"`, `"feedforward"`,
    /// `"cyclefeedforward"`, or any other combination of roles (`"cycle_out"`,
    /// `"out_in_middleman"`, ...: see `DIRECTED_TRIANGLE_TYPES`).
    #[pyo3(signature = (kind, *, width = "u8"))]
    fn directed_triangles<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (w, kind) = (self::width(width)?, triangle_type(kind)?);
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => directed_triangles::<_, T>(g, kind))),
        )
    }

    /// Trussness of each arc (in `csr()` order) over its directed triangles of a `kind` (see
    /// `directed_triangles`), by `"pkt"` or `"burkhardt"`, as `width` integers. For the
    /// trussness of the underlying undirected graph, use `to_undirected().k_truss()`.
    #[pyo3(signature = (kind, *, algorithm = "pkt", width = "u8"))]
    fn k_truss<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        algorithm: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (kind, pkt, w) = (
            triangle_type(kind)?,
            truss_algorithm(algorithm)?,
            self::width(width)?,
        );
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => directed_k_truss::<_, T>(g, pkt, kind))),
        )
    }

    /// Coreness of each node, by in-degree (`"in"`), out-degree (`"out"`) or both (`"total"`:
    /// in plus out, counting every arc, networkx's `core_number`, which on a symmetric graph is
    /// twice the usual cores). Self-loops are ignored. As `width` integers (every node's degree
    /// must fit one).
    #[pyo3(signature = (*, degree = "total", width = "u8"))]
    fn k_core<'py>(
        &self,
        py: Python<'py>,
        degree: &str,
        width: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (degree, w) = (core_degree(degree)?, self::width(width)?);
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_width!(w, T => directed_k_core::<_, T>(g, degree))),
        )
    }

    /// Fagiolo's clustering coefficients (networkx's `clustering` on directed graphs):
    /// `{"local": per node, "average": ...}`.
    fn clustering<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .run(py, |g| with_graph!(g, g => directed_clustering(g)))
    }

    /// Louvain communities by Leicht and Newman's directed modularity (the modularity networkx
    /// computes on directed graphs): `{"communities": per node, numbered by smallest node,
    /// "count": ..., "modularity": ...}`.
    fn louvain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .run(py, |g| with_graph!(g, g => directed_louvain(g)))
    }

    /// HyperBall estimates (see `Graph.hyperball`), with distances following arcs in
    /// `direction`: `"out"` (the default) from each node, `"in"` to it (networkx's
    /// convention), `"both"` ignoring directions (on the underlying undirected graph).
    #[pyo3(signature = (*, precision = 8, normalized = true, gpu = false, direction = "out"))]
    fn hyperball<'py>(
        &self,
        py: Python<'py>,
        precision: u8,
        normalized: bool,
        gpu: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = follow(direction)?;
        if gpu {
            return self.hyperball_gpu(py, precision, normalized, d);
        }
        with_precision!(precision, P => self.inner.run(py, |g| {
            with_graph!(g, g => with_view!(g, d, v => hyperball::<_, P, _>(v, normalized)))
        }))?
    }

    /// Exact closeness centrality of each node, with distances following arcs in `direction`
    /// (see `hyperball`).
    #[pyo3(signature = (*, normalized = true, direction = "out"))]
    fn closeness<'py>(
        &self,
        py: Python<'py>,
        normalized: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = follow(direction)?;
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_view!(g, d, v => closeness(v, normalized))),
        )
    }

    /// Exact harmonic centrality of each node, with distances following arcs in `direction`
    /// (see `hyperball`).
    #[pyo3(signature = (*, normalized = true, direction = "out"))]
    fn harmonic<'py>(
        &self,
        py: Python<'py>,
        normalized: bool,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = follow(direction)?;
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_view!(g, d, v => harmonic(v, normalized))),
        )
    }

    /// Exact Lin's centrality of each node, with distances following arcs in `direction` (see
    /// `hyperball`).
    #[pyo3(signature = (*, direction = "out"))]
    fn lin<'py>(&self, py: Python<'py>, direction: &str) -> PyResult<Bound<'py, PyAny>> {
        let d = follow(direction)?;
        self.inner
            .run(py, |g| with_graph!(g, g => with_view!(g, d, v => lin(v))))
    }

    /// BFS distance of each node (`-1` for nodes not reached), following arcs in `direction`:
    /// `"out"` (the default) gives distances from `source`, `"in"` to it, `"both"` ignores
    /// directions.
    #[pyo3(signature = (source, *, direction = "out"))]
    fn bfs<'py>(
        &self,
        py: Python<'py>,
        source: usize,
        direction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.check_node(source)?;
        let d = follow(direction)?;
        self.inner.run(
            py,
            |g| with_graph!(g, g => with_view!(g, d, v => bfs(v, source))),
        )
    }

    /// The underlying undirected graph, as a `Graph`, by `mode`: `"collapse"` (the default) has
    /// one edge `{u, v}` wherever there's an arc `u -> v` or `v -> u` (networkx's
    /// `DiGraph.to_undirected()`), `"each"` one edge per arc, so that reciprocal and parallel
    /// arcs become parallel edges (networkx's `MultiDiGraph.to_undirected()`), and `"mutual"`
    /// one edge wherever there are arcs both ways, a self-loop being its own reverse (networkx's
    /// `to_undirected(reciprocal=True)`). Each is built once and cached alongside this graph,
    /// which owns it: the returned graph never drops it, and is only usable while this graph is
    /// open (it raises `RuntimeError` after).
    #[pyo3(signature = (*, mode = "collapse"))]
    fn to_undirected(&self, py: Python<'_>, mode: &str) -> PyResult<UGraph> {
        let mode = undirected_mode(mode)?;
        let inner = self.inner.convert::<Undirected>(py, move |g| {
            Ok(with_graph!(g, g => g.to_undirected_with(mode)?.cache_filename()))
        })?;
        Ok(UGraph { inner })
    }
});

impl UGraph {
    #[cfg(feature = "gpu")]
    fn triangles_gpu<'py>(&self, py: Python<'py>, w: Width) -> PyResult<Bound<'py, PyAny>> {
        self.inner.run(
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
    ) -> PyResult<Bound<'py, PyAny>> {
        with_precision!(precision, P => self.inner.run(py, |g| {
            with_graph!(g, g => hyperball_gpu::<_, P, _>(g, normalized))
        }))?
    }

    #[cfg(not(feature = "gpu"))]
    fn hyperball_gpu<'py>(
        &self,
        _py: Python<'py>,
        _precision: u8,
        _normalized: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        Err(no_gpu())
    }
}

impl DGraph {
    #[cfg(feature = "gpu")]
    fn hyperball_gpu<'py>(
        &self,
        py: Python<'py>,
        precision: u8,
        normalized: bool,
        d: Follow,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_precision!(precision, P => self.inner.run(py, |g| {
            with_graph!(g, g => with_view!(g, d, v => hyperball_gpu::<_, P, _>(v, normalized)))
        }))?
    }

    #[cfg(not(feature = "gpu"))]
    fn hyperball_gpu<'py>(
        &self,
        _py: Python<'py>,
        _precision: u8,
        _normalized: bool,
        _d: Follow,
    ) -> PyResult<Bound<'py, PyAny>> {
        Err(no_gpu())
    }
}

#[pymodule]
fn _sparx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UGraph>()?;
    m.add_class::<DGraph>()?;
    m.add(
        "DIRECTED_TRIANGLE_TYPES",
        DirectedTriangleType::ALL.map(|t| t.name()).to_vec(),
    )?;
    m.add("GPU", cfg!(feature = "gpu"))?;
    Ok(())
}
