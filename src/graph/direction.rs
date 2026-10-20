//! Graph direction: the [`Directedness`] of a graph type, [`Undirected`] ([`Graph`]) or
//! [`Directed`] ([`DiGraph`]), and views of directed graphs that follow their arcs in a given
//! [`Direction`].
//!
//! Algorithms are implemented for the graph types they make sense on: undirected ones (triangles,
//! k-trusses, k-cores, clustering coefficients, Louvain, heat kernel PageRank) take a [`Graph`],
//! directed ones (directed triangles and clustering, in/out k-cores, directed Louvain) a
//! [`DiGraph`], and those valid on both (traversals, centralities, components) either. Running
//! an undirected algorithm on a directed graph is thus a type error, not a run-time check: to run
//! one on the graph underlying a directed one, use [`DiGraph::to_undirected`].
//!
//! A `Graph` is *symmetric*: every arc `u -> v` has a reverse `v -> u`, counting multiplicity
//! (an undirected edge is stored as an arc each way). Whether a graph's arcs are is recorded once,
//! when its cache entry is built, so loading a `Graph` checks it for free and refuses directed
//! ones. Any graph can be loaded as a `DiGraph` (see [`AnyGraph`] to load whichever fits).
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use sparx::graph::{AnyGraph, DiGraph, Graph};
//! use sparx::k_truss::pkt::AlgoPKT;
//! # let dir = std::env::temp_dir().join(format!("sparx_direction_doc_{}", std::process::id()));
//! # std::fs::create_dir_all(&dir)?;
//! # let path = dir.join("web.mtx");
//! # std::fs::write(&path, "%%MatrixMarket matrix coordinate pattern general\n3 3 3\n1 2\n2 3\n1 3\n")?;
//! // a directed file can't be loaded as an undirected `Graph`...
//! assert!(Graph::<(), (), u32>::from_file(&path, None, Some(2)).is_err());
//! // ... but it can as a `DiGraph`, whose underlying undirected graph is a `Graph`
//! let mut g = DiGraph::<(), (), u32>::from_file(&path, None, Some(2))?;
//! let trussness = AlgoPKT::new(g.to_undirected()?)?.k_trusses().to_vec();
//! assert_eq!(trussness.len(), 6);
//! g.drop_cache()?;
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok(())
//! # }
//! ```
//!
//! Undirected algorithms don't accept directed graphs:
//!
//! ```compile_fail
//! use sparx::graph::DiGraph;
//! use sparx::k_truss::pkt::AlgoPKT;
//! fn trussness(g: &DiGraph<(), (), u32>) {
//!     AlgoPKT::new(g); // expected `&Graph`, found `&DiGraph`
//! }
//! ```
//!
//! ```compile_fail
//! use sparx::graph::DiGraph;
//! use sparx::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
//! fn coreness(g: &DiGraph<(), (), u32>) {
//!     AlgoBatageljZaversnik::new(g);
//! }
//! ```
//!
//! and directed ones don't accept undirected graphs, which [`Graph::into_directed`] turns into
//! directed ones:
//!
//! ```compile_fail
//! use sparx::graph::Graph;
//! use sparx::communities::directed_louvain::DirectedLouvain;
//! fn communities(g: &Graph<(), (), u32>) {
//!     DirectedLouvain::new(g);
//! }
//! ```
//!
//! [`Graph`]: super::Graph
//! [`DiGraph`]: super::DiGraph
//! [`DiGraph::to_undirected`]: super::GraphMemoryMap::to_undirected
//! [`Graph::into_directed`]: super::GraphMemoryMap::into_directed
use super::{DiGraph, Graph, GraphMemoryMap, IndexType};

use crossbeam::thread;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Undirected {}
    impl Sealed for super::Directed {}
}

/// Whether a graph type is directed: [`Undirected`] or [`Directed`].
pub trait Directedness:
    sealed::Sealed + Copy + Default + std::fmt::Debug + Send + Sync + 'static
{
    /// Whether graphs of this type are directed.
    const DIRECTED: bool;
}

/// The [`Directedness`] of undirected graphs ([`Graph`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Undirected;

/// The [`Directedness`] of directed graphs ([`DiGraph`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Directed;

impl Directedness for Undirected {
    const DIRECTED: bool = false;
}

impl Directedness for Directed {
    const DIRECTED: bool = true;
}

/// Which arcs of a directed graph an algorithm follows: out of nodes (the default) or into them.
/// To ignore their directions, run the algorithm on the
/// [underlying undirected graph](super::GraphMemoryMap::to_undirected).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Along arcs: from a node to its out-neighbours. The default.
    #[default]
    Out,
    /// Against arcs: from a node to its in-neighbours (the out-neighbours of the
    /// [transpose](super::GraphMemoryMap::transpose)).
    In,
}

/// How [`DiGraph::to_undirected_with`] makes undirected edges of a directed graph's arcs.
///
/// [`DiGraph::to_undirected_with`]: super::GraphMemoryMap::to_undirected_with
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UndirectedMode {
    /// One edge `u - v` wherever there's an arc `u -> v` or `v -> u` (any number of them): the
    /// simple underlying graph, as networkx's `DiGraph.to_undirected` and igraph's `collapse`
    /// mode. The default.
    #[default]
    Collapse = 0,
    /// One edge per arc: arcs both ways between two nodes, and parallel arcs, become parallel
    /// edges (which triangle based algorithms count separately), as networkx's
    /// `MultiDiGraph.to_undirected` and igraph's `each` mode.
    Each = 1,
    /// One edge `u - v` wherever there are arcs both ways, `u -> v` and `v -> u` (self-loops are
    /// their own reverse), as networkx's `to_undirected(reciprocal=True)` and igraph's `mutual`
    /// mode.
    Mutual = 2,
}

impl UndirectedMode {
    /// The kind of the derived graph it makes, in cache ids.
    pub(crate) fn kind(self) -> &'static str {
        match self {
            UndirectedMode::Collapse => "undirected_collapse",
            UndirectedMode::Each => "undirected_each",
            UndirectedMode::Mutual => "undirected_mutual",
        }
    }
}

/// A graph loaded as whichever type fits it: a [`Graph`] if it's symmetric, a [`DiGraph`] if not.
pub enum AnyGraph<
    N: super::N = super::label::VoidLabel,
    E: super::E = super::label::VoidLabel,
    Ix: IndexType = super::DefaultIx,
> {
    /// A symmetric graph.
    Undirected(Graph<N, E, Ix>),
    /// A graph with an arc without a reverse.
    Directed(DiGraph<N, E, Ix>),
}

impl<N: super::N, E: super::E, Ix: IndexType> AnyGraph<N, E, Ix> {
    /// Parses a file (see [`GraphMemoryMap::from_file`]) into whichever graph type fits it.
    pub fn from_file<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::from(DiGraph::from_file(p, id, thread_count)?))
    }

    /// Opens a cache entry (see [`GraphMemoryMap::open`]) as whichever graph type fits it.
    pub fn open<P: AsRef<Path>>(
        filename: P,
        batch: Option<usize>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::from(DiGraph::open(filename, batch, thread_count)?))
    }
}

impl<N: super::N, E: super::E, Ix: IndexType> From<DiGraph<N, E, Ix>> for AnyGraph<N, E, Ix> {
    fn from(g: DiGraph<N, E, Ix>) -> Self {
        match g.try_into_undirected() {
            Ok(g) => Self::Undirected(g),
            Err(g) => Self::Directed(g),
        }
    }
}

/// Finds an arc `u -> v` without a reverse `v -> u`, with multiplicity: `u -> v` appearing more
/// times than `v -> u` counts as one. Neighbour lists must be sorted: both counts are then runs
/// found by binary search.
///
/// `offsets` has one entry per node plus a final one (`neighbours.len()`).
pub(crate) fn find_asymmetric_arc<Ix: IndexType>(
    offsets: &[Ix],
    neighbours: &[Ix],
    threads: usize,
) -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
    let node_count = offsets.len().saturating_sub(1);
    let threads = threads.max(1);
    let thread_load = node_count.div_ceil(threads).max(1);
    let found = AtomicBool::new(false);

    let neighbours_of = |w: usize| -> &[Ix] {
        neighbours
            .get(offsets[w].index()..offsets[w + 1].index())
            .unwrap_or(&[])
    };

    let per_thread = thread::scope(|scope| {
        let found = &found;
        let neighbours_of = &neighbours_of;
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                scope.spawn(move |_| -> Option<(usize, usize)> {
                    let begin = std::cmp::min(tid * thread_load, node_count);
                    let end = std::cmp::min(begin + thread_load, node_count);
                    for u in begin..end {
                        if found.load(Ordering::Relaxed) {
                            return None;
                        }
                        let n_u = neighbours_of(u);
                        let mut i = 0;
                        while i < n_u.len() {
                            let v = n_u[i].index();
                            let run = n_u[i..].partition_point(|w| w.index() == v);
                            // self-loops are their own reverse
                            if v != u {
                                let n_v = neighbours_of(v);
                                let lo = n_v.partition_point(|w| w.index() < u);
                                let hi = n_v.partition_point(|w| w.index() <= u);
                                if hi - lo < run {
                                    found.store(true, Ordering::Relaxed);
                                    return Some((u, v));
                                }
                            }
                            i += run;
                        }
                    }
                    None
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join()).collect::<Vec<_>>()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

    let mut arc = None;
    for r in per_thread {
        let r = r.map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;
        // the lowest thread's: the smallest such `u` it found
        arc = arc.or(r);
    }
    Ok(arc)
}

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: IndexType, Dir: Directedness> GraphMemoryMap<N, E, Ix, Dir> {
    /// Whether this is a directed graph type ([`DiGraph`]): a property of the type, see
    /// [`is_symmetric`](Self::is_symmetric) for one of the arcs.
    pub const fn is_directed(&self) -> bool {
        Dir::DIRECTED
    }

    /// Whether every arc `u -> v` has a reverse `v -> u`, counting multiplicity (self-loops are
    /// their own reverse): always, for a [`Graph`].
    ///
    /// Recorded when the graph's cache entry is built, so this is free: MatrixMarket
    /// `symmetric` files are symmetric by construction, other inputs are checked once (in
    /// `O(|E| log(max degree))`) while parsing. Entries built by earlier versions are checked on
    /// first call.
    pub fn is_symmetric(&self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(&symmetric) = self.symmetric.get() {
            return Ok(symmetric);
        }
        let symmetric = self.asymmetric_arc()?.is_none();
        let _ = self.symmetric.set(symmetric);
        Ok(symmetric)
    }

    /// An arc `u -> v` without a reverse `v -> u` (with multiplicity), if the graph has any.
    fn asymmetric_arc(&self) -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
        let offsets =
            unsafe { std::slice::from_raw_parts(self.offsets_ptr(), self.offsets_size()) };
        find_asymmetric_arc(offsets, &self[..], self.thread_num())
    }

    /// Errors, naming an arc without a reverse, unless the graph is symmetric: what loading a
    /// directed graph as a [`Graph`] does.
    pub(super) fn require_symmetric(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_symmetric()? {
            return Ok(());
        }
        let arc = match self.asymmetric_arc()? {
            Some((u, v)) => format!(": arc {u} -> {v} has no reverse {v} -> {u}"),
            None => String::new(),
        };
        Err(format!(
            "error the graph is directed{arc}, so it can't be loaded as an undirected `Graph`: \
             load it as a `DiGraph` (whose `to_undirected` is the underlying undirected graph)"
        )
        .into())
    }

    /// The same graph as another type, keeping its cache entry (and not its views).
    fn retype<D2: Directedness>(self) -> GraphMemoryMap<N, E, Ix, D2> {
        GraphMemoryMap {
            graph: self.graph,
            index: self.index,
            metalabels: self.metalabels,
            node_labels: self.node_labels,
            edge_labels: self.edge_labels,
            graph_cache: self.graph_cache,
            edge_size: self.edge_size,
            thread_count: self.thread_count,
            exports: self.exports,
            symmetric: self.symmetric,
            transpose: std::sync::OnceLock::new(),
            undirected: Default::default(),
            derive_lock: self.derive_lock,
            instance: self.instance,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<N: super::N, E: super::E, Ix: IndexType> Graph<N, E, Ix> {
    /// The graph as a [`DiGraph`], each edge an arc each way: for directed algorithms.
    pub fn into_directed(self) -> DiGraph<N, E, Ix> {
        self.retype()
    }
}

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: IndexType> DiGraph<N, E, Ix> {
    /// The graph as a [`Graph`] if it's symmetric (see [`is_symmetric`](Self::is_symmetric)),
    /// or else itself back.
    // the graph itself back on failure (like `Rc::try_unwrap`), so that it isn't lost
    #[allow(clippy::result_large_err)]
    pub fn try_into_undirected(self) -> Result<Graph<N, E, Ix>, Self> {
        match self.is_symmetric() {
            Ok(true) => Ok(self.retype()),
            _ => Err(self),
        }
    }

    /// The graph whose out-neighbours are this graph's neighbours in `direction`: the graph
    /// itself ([`Direction::Out`]) or its [transpose](Self::transpose) ([`Direction::In`]).
    ///
    /// Node ids are the same in both, so per-node results of an algorithm run on a view are
    /// per-node results of this graph. Edge ids are not (see [`transpose`](Self::transpose)).
    pub fn view(&self, direction: Direction) -> Result<&Self, Box<dyn std::error::Error>> {
        match direction {
            Direction::Out => Ok(self),
            Direction::In => self.transpose(),
        }
    }

    /// The transpose of the graph: every arc `u -> v` reversed into `v -> u`, so that a node's
    /// out-neighbours are its in-neighbours here. Each edge keeps the label of the arc it
    /// reverses; edges are ordered by node, and by neighbour within a node, as in every graph.
    ///
    /// Built once (in `O(|V| + |E|)`) and cached alongside the graph, which owns it:
    /// [`drop_cache`](Self::drop_cache) removes it too. A symmetric graph is its own transpose.
    pub fn transpose(&self) -> Result<&Self, Box<dyn std::error::Error>> {
        if self.is_symmetric()? {
            return Ok(self);
        }
        if let Some(transpose) = self.transpose.get() {
            return Ok(transpose);
        }
        let _building = self
            .derive_lock
            .lock()
            .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
        if let Some(transpose) = self.transpose.get() {
            return Ok(transpose);
        }
        let cache = self.graph_cache.derive_transpose()?;
        let transpose = Self::init_from_cache(cache, Some(self.thread_count))?;
        Ok(self.transpose.get_or_init(|| Box::new(transpose)))
    }

    /// The underlying undirected graph, [collapsed](UndirectedMode::Collapse): an edge `u - v`
    /// wherever there's an arc `u -> v` or `v -> u`, once (as networkx's `DiGraph.to_undirected`
    /// and igraph's default). See [`to_undirected_with`](Self::to_undirected_with) for other
    /// modes. Undirected algorithms run on it.
    pub fn to_undirected(&self) -> Result<&Graph<N, E, Ix>, Box<dyn std::error::Error>> {
        self.to_undirected_with(UndirectedMode::Collapse)
    }

    /// The underlying undirected graph in a given [`UndirectedMode`]: each edge stored as `u -> v`
    /// and `v -> u` (self-loops once, as MatrixMarket `symmetric` files store them), keeping the
    /// label of the (first) arc it comes from.
    ///
    /// Built once per mode (in `O(|V| + |E|)`, after the [transpose](Self::transpose)) and
    /// cached alongside the graph, which owns it: [`drop_cache`](Self::drop_cache) removes it
    /// too. A symmetric graph without parallel arcs already is its collapsed (and mutual)
    /// undirected graph: that's the same entry, as a [`Graph`].
    pub fn to_undirected_with(
        &self,
        mode: UndirectedMode,
    ) -> Result<&Graph<N, E, Ix>, Box<dyn std::error::Error>> {
        let slot = &self.undirected[mode as usize];
        if let Some(undirected) = slot.get() {
            return Ok(undirected);
        }
        let symmetric = self.is_symmetric()?;
        let itself = symmetric && mode != UndirectedMode::Each && !self.has_parallel_arcs();
        // (a symmetric graph is its own transpose)
        let transpose = self.transpose()?;
        let _building = self
            .derive_lock
            .lock()
            .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
        if let Some(undirected) = slot.get() {
            return Ok(undirected);
        }
        let cache = if itself {
            self.graph_cache.clone()
        } else {
            self.graph_cache
                .derive_undirected(&transpose.graph_cache, mode)?
        };
        let undirected = Graph::init_from_cache(cache, Some(self.thread_count))?;
        Ok(slot.get_or_init(|| Box::new(undirected)))
    }

    /// Whether some arc appears more than once (a self-loop too).
    fn has_parallel_arcs(&self) -> bool {
        (0..self.size()).any(|u| {
            let neighbours = &self[self.index_node(u)];
            neighbours.windows(2).any(|w| w[0] == w[1])
        })
    }

    /// Number of arcs into a node: its out-degree in the [transpose](Self::transpose).
    pub fn in_degree(&self, node_id: usize) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(self.transpose()?.node_degree(node_id))
    }

    /// A node's in-neighbours, sorted: its out-neighbours in the [transpose](Self::transpose).
    pub fn in_neighbours(
        &self,
        node_id: usize,
    ) -> Result<super::NeighbourIter<N, E, Ix>, Box<dyn std::error::Error>> {
        self.transpose()?.neighbours(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cache::{
        CacheMetadata,
        utils::{CACHE_DIR, id_from_filename},
    };
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn write_mtx(name: &str, header: &str, n: usize, edges: &[(usize, usize)]) -> PathBuf {
        let dir = unique_test_dir("direction_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        path
    }

    fn id(name: &str) -> String {
        format!("direction_{name}_{}", unique_run_tag())
    }

    fn digraph(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
    ) -> TestGraph<(), (), usize, Directed> {
        let path = write_mtx(name, header, n, edges);
        let g = DiGraph::from_file(&path, Some(id(name)), Some(3)).expect("parse should succeed");
        TestGraph::new(g, path.parent().map(Path::to_path_buf))
    }

    pub(super) fn lists<
        N: crate::graph::N,
        E: crate::graph::E,
        Ix: IndexType,
        Dir: Directedness,
    >(
        g: &GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Vec<Vec<usize>> {
        (0..g.size())
            .map(|u| g.neighbours(u).unwrap().collect())
            .collect()
    }

    /// Files of the cache entry whose (hashed) id is `hashed`.
    fn files_of(hashed: &str) -> Vec<PathBuf> {
        glob::glob(&format!("{CACHE_DIR}*{hashed}.*"))
            .unwrap()
            .flatten()
            .collect()
    }

    /// Random multigraph with self-loops, reciprocal and parallel arcs.
    fn random_arcs(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut arcs: Vec<_> = (0..m).map(|_| (next() % n, next() % n)).collect();
        arcs.extend([(0, 1), (1, 0), (2, 3), (2, 3), (4, 4)]);
        arcs
    }

    #[test]
    fn symmetry_is_recorded_in_the_metadata() {
        for (name, header, edges, symmetric) in [
            ("symmetric", "symmetric", vec![(1, 0), (2, 1)], true),
            ("general_directed", "general", vec![(0, 1), (1, 2)], false),
            ("general_both_ways", "general", vec![(0, 1), (1, 0)], true),
            ("general_loops", "general", vec![(0, 0), (2, 2)], true),
            (
                "multiplicity",
                "general",
                vec![(0, 1), (0, 1), (1, 0)],
                false,
            ),
        ] {
            let g = digraph(name, header, 3, &edges);
            assert_eq!(g.metadata().unwrap().symmetric, Some(symmetric), "{name}");
            assert_eq!(g.is_symmetric().unwrap(), symmetric, "{name}");
            assert!(
                g.is_directed(),
                "{name}: a DiGraph is a directed graph type"
            );
            // and reopened from the cache, without parsing
            let reopened =
                DiGraph::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, None)
                    .unwrap();
            assert_eq!(
                reopened.symmetric.get(),
                Some(&symmetric),
                "{name}, reopened"
            );
        }
    }

    #[test]
    fn metadata_without_symmetry_reads_as_unknown() {
        let mut meta = CacheMetadata::now::<VoidLabel, VoidLabel, usize>(
            "d", "id", 3, 2, 8, false, false, false, 0, 0, None, None,
        );
        assert!(!meta.to_toml().contains("symmetric"));
        assert_eq!(
            CacheMetadata::from_toml(&meta.to_toml()).unwrap().symmetric,
            None
        );
        meta.symmetric = Some(false);
        assert_eq!(
            CacheMetadata::from_toml(&meta.to_toml()).unwrap().symmetric,
            Some(false)
        );
    }

    /// Only symmetric graphs load as `Graph`s; a directed one fails, naming an unreciprocated
    /// arc, and a freshly parsed entry it fails on is removed.
    #[test]
    fn graphs_load_only_symmetric_inputs() {
        let symmetric = write_mtx("graph_symmetric", "symmetric", 3, &[(1, 0), (2, 1)]);
        let both_ways = write_mtx("graph_both_ways", "general", 3, &[(0, 1), (1, 0)]);
        let directed = write_mtx("graph_directed", "general", 3, &[(0, 1), (1, 2)]);
        for path in [&symmetric, &both_ways] {
            let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(
                path,
                Some(id("graph_ok")),
                Some(2),
            )
            .unwrap();
            let g = TestGraph::new(g, None);
            assert!(!g.is_directed() && g.is_symmetric().unwrap());
        }
        let refused_id = id("graph_refused");
        let err = Graph::<VoidLabel, VoidLabel, usize>::from_file(
            &directed,
            Some(refused_id.clone()),
            Some(2),
        )
        .err()
        .expect("a directed graph isn't a Graph");
        assert!(
            err.to_string().contains("arc 0 -> 1 has no reverse"),
            "{err}"
        );
        assert!(err.to_string().contains("DiGraph"), "{err}");
        assert!(
            files_of(&id_from_filename(&refused_id).unwrap()).is_empty(),
            "refused entry left behind"
        );

        // an existing directed entry can't be opened as a `Graph`, and isn't removed for it
        let d = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
            &directed,
            Some(id("graph_open")),
            Some(2),
        )
        .unwrap();
        let d = TestGraph::new(d, None);
        assert!(
            Graph::<VoidLabel, VoidLabel, usize>::open(d.cache_filename(), None, None).is_err()
        );
        assert!(Path::new(&d.cache_filename()).exists());
        for path in [symmetric, both_ways, directed] {
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    #[test]
    fn any_graph_and_conversions_pick_the_type_that_fits() {
        let symmetric = write_mtx("any_symmetric", "general", 2, &[(0, 1), (1, 0)]);
        let directed = write_mtx("any_directed", "general", 2, &[(0, 1)]);
        match AnyGraph::<VoidLabel, VoidLabel, usize>::from_file(
            &symmetric,
            Some(id("any_s")),
            Some(2),
        )
        .unwrap()
        {
            AnyGraph::Undirected(g) => drop(TestGraph::new(g, None)),
            AnyGraph::Directed(_) => panic!("a symmetric graph is a Graph"),
        }
        let d = match AnyGraph::<VoidLabel, VoidLabel, usize>::from_file(
            &directed,
            Some(id("any_d")),
            Some(2),
        )
        .unwrap()
        {
            AnyGraph::Directed(d) => d,
            AnyGraph::Undirected(_) => panic!("a directed graph is a DiGraph"),
        };
        // a directed graph doesn't become a Graph, a Graph always becomes a DiGraph and back
        let d = d.try_into_undirected().err().expect("directed");
        let d = TestGraph::new(d, None);
        let g =
            Graph::<VoidLabel, VoidLabel, usize>::from_file(&symmetric, Some(id("any_g")), Some(2))
                .unwrap();
        let as_directed = g.into_directed();
        assert!(as_directed.is_directed());
        let g = as_directed.try_into_undirected().ok().expect("symmetric");
        drop(TestGraph::new(g, None));
        drop(d);
        for path in [symmetric, directed] {
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    /// Each node's undirected neighbours in a mode, by brute force over the arcs.
    fn undirected_by_brute_force(
        n: usize,
        arcs: &[(usize, usize)],
        mode: UndirectedMode,
    ) -> Vec<Vec<usize>> {
        let count = |u: usize, v: usize| arcs.iter().filter(|&&a| a == (u, v)).count();
        (0..n)
            .map(|u| {
                let mut l = vec![];
                for v in 0..n {
                    let (out, into) = (count(u, v), if u == v { 0 } else { count(v, u) });
                    let copies = match mode {
                        UndirectedMode::Each => out + into,
                        UndirectedMode::Collapse => usize::from(out + into > 0),
                        UndirectedMode::Mutual => usize::from(out > 0 && (into > 0 || u == v)),
                    };
                    l.extend(std::iter::repeat_n(v, copies));
                }
                l
            })
            .collect()
    }

    #[test]
    fn transpose_and_undirected_views_match_brute_force() {
        let n = 40;
        let arcs = random_arcs(n, 300, 7);
        let g = digraph("random_multi", "general", n, &arcs);
        assert!(!g.is_symmetric().unwrap());

        let mut expected_in = vec![Vec::new(); n];
        for &(u, v) in &arcs {
            expected_in[v].push(u);
        }
        expected_in.iter_mut().for_each(|l| l.sort());
        let transpose = g.transpose().unwrap();
        assert_eq!(lists(transpose), expected_in);
        assert!(!transpose.is_symmetric().unwrap());
        assert_eq!(transpose.width(), g.width());
        for v in 0..n {
            assert_eq!(g.in_degree(v).unwrap(), expected_in[v].len());
            assert_eq!(
                g.in_neighbours(v).unwrap().collect::<Vec<_>>(),
                expected_in[v]
            );
        }

        for mode in [
            UndirectedMode::Collapse,
            UndirectedMode::Each,
            UndirectedMode::Mutual,
        ] {
            let undirected = g.to_undirected_with(mode).unwrap();
            assert_eq!(
                lists(undirected),
                undirected_by_brute_force(n, &arcs, mode),
                "{mode:?}"
            );
            assert!(
                undirected.is_symmetric().unwrap() && !undirected.is_directed(),
                "{mode:?}"
            );
            // built once
            assert!(
                std::ptr::eq(g.to_undirected_with(mode).unwrap(), undirected),
                "{mode:?}"
            );
        }
        assert!(std::ptr::eq(
            g.to_undirected().unwrap(),
            g.to_undirected_with(UndirectedMode::Collapse).unwrap()
        ));
        assert!(std::ptr::eq(g.view(Direction::Out).unwrap(), &*g));
        assert!(std::ptr::eq(g.view(Direction::In).unwrap(), transpose));
        assert!(std::ptr::eq(g.transpose().unwrap(), transpose));
    }

    /// A symmetric DiGraph's modes: collapsed and mutual are the graph itself when it has no
    /// parallel arcs, and every mode is built when it has.
    #[test]
    fn undirected_modes_of_symmetric_digraphs() {
        for (name, arcs) in [
            ("simple", vec![(1, 0), (2, 1), (3, 3)]),
            ("parallel", vec![(1, 0), (1, 0), (2, 1), (3, 3), (3, 3)]),
        ] {
            let g = digraph(name, "symmetric", 4, &arcs);
            let stored = lists(&*g);
            let all_arcs: Vec<(usize, usize)> = (0..4)
                .flat_map(|u| stored[u].iter().map(move |&v| (u, v)))
                .collect();
            for mode in [
                UndirectedMode::Collapse,
                UndirectedMode::Each,
                UndirectedMode::Mutual,
            ] {
                let undirected = g.to_undirected_with(mode).unwrap();
                assert_eq!(
                    lists(undirected),
                    undirected_by_brute_force(4, &all_arcs, mode),
                    "{name} {mode:?}"
                );
                let same_entry = undirected.cache_filename() == g.cache_filename();
                assert_eq!(
                    same_entry,
                    name == "simple" && mode != UndirectedMode::Each,
                    "{name} {mode:?}"
                );
            }
        }
    }

    /// A symmetric `DiGraph` is its own transpose, and its undirected graph is the same entry.
    #[test]
    fn views_of_symmetric_digraphs_cost_nothing() {
        let g = digraph(
            "symmetric_digraph",
            "symmetric",
            4,
            &[(1, 0), (2, 1), (3, 3)],
        );
        assert!(std::ptr::eq(g.transpose().unwrap(), &*g));
        assert!(std::ptr::eq(g.view(Direction::In).unwrap(), &*g));
        let undirected = g.to_undirected().unwrap();
        assert_eq!(undirected.cache_filename(), g.cache_filename());
        assert_eq!(lists(undirected), lists(&*g));
        assert_eq!(g.in_degree(1).unwrap(), 2);
    }

    #[test]
    fn derived_graphs_are_reused_and_dropped_with_the_graph() {
        let path = write_mtx("reuse", "general", 3, &[(0, 1), (1, 2), (2, 0), (0, 2)]);
        let mut g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id("reuse")), Some(2))
                .unwrap();
        let expected = (
            lists(g.transpose().unwrap()),
            lists(g.to_undirected().unwrap()),
        );
        let derived_files = |g: &DiGraph<VoidLabel, VoidLabel, usize>| -> Vec<PathBuf> {
            ["transpose", "undirected"]
                .iter()
                .flat_map(|kind| {
                    files_of(&id_from_filename(&g.graph_cache.derived_id(kind).unwrap()).unwrap())
                })
                .collect()
        };
        assert!(!derived_files(&g).is_empty());

        // a fresh instance finds them on disk
        let reopened =
            DiGraph::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, Some(2))
                .unwrap();
        assert_eq!(
            (
                lists(reopened.transpose().unwrap()),
                lists(reopened.to_undirected().unwrap())
            ),
            expected
        );

        g.drop_cache().unwrap();
        assert!(
            derived_files(&reopened).is_empty(),
            "derived graphs left behind"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Rebuilding a graph's entry (e.g. parsing a changed dataset under the same id) discards
    /// the graphs derived from the old one.
    #[test]
    fn rebuilt_graphs_do_not_reuse_stale_derived_graphs() {
        let id = id("rebuilt");
        let first = write_mtx("rebuilt_a", "general", 3, &[(0, 1), (1, 2)]);
        let g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&first, Some(id.clone()), Some(2))
                .unwrap();
        assert_eq!(
            lists(g.transpose().unwrap()),
            vec![vec![], vec![0], vec![1]]
        );
        drop(g);

        let second = write_mtx("rebuilt_b", "general", 3, &[(2, 0), (2, 1)]);
        let g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&second, Some(id), Some(2)).unwrap();
        let g = TestGraph::new(g, None);
        assert_eq!(
            lists(g.transpose().unwrap()),
            vec![vec![2], vec![2], vec![]]
        );
        assert_eq!(
            lists(g.to_undirected().unwrap()),
            vec![vec![2], vec![2], vec![0, 1]]
        );
        for path in [first, second] {
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    struct Weight(usize);
    unsafe impl bytemuck::Zeroable for Weight {}
    unsafe impl bytemuck::Pod for Weight {}
    impl crate::graph::E for Weight {
        fn new(v: usize) -> Self {
            Weight(v)
        }
        fn is_labeled() -> bool {
            true
        }
    }

    #[test]
    fn derived_edges_keep_their_arcs_labels() {
        let dir = unique_test_dir("direction_labels");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weighted.mtx");
        // 0 -> 1 (10), 0 -> 2 (20), 1 -> 0 (30), 2 -> 2 (40)
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate integer general\n3 3 4\n1 2 10\n1 3 20\n2 1 30\n3 3 40\n",
        )
        .unwrap();
        let g = DiGraph::<VoidLabel, Weight, usize>::from_file(&path, Some(id("labels")), Some(2))
            .unwrap();
        let g = TestGraph::new(g, Some(dir));

        fn labelled<Dir: Directedness>(
            g: &GraphMemoryMap<VoidLabel, Weight, usize, Dir>,
        ) -> BTreeMap<usize, Vec<(usize, usize)>> {
            (0..g.size())
                .map(|u| {
                    let edges = g
                        .index_node(u)
                        .zip(g.neighbours(u).unwrap())
                        .map(|(e, v)| (v, g.edge_label(e).0))
                        .collect();
                    (u, edges)
                })
                .collect()
        }
        assert_eq!(
            labelled(g.transpose().unwrap()),
            BTreeMap::from([
                (0, vec![(1, 30)]),
                (1, vec![(0, 10)]),
                (2, vec![(0, 20), (2, 40)])
            ])
        );
        // collapsed: the first arc's label, out-arcs first
        assert_eq!(
            labelled(g.to_undirected().unwrap()),
            BTreeMap::from([
                (0, vec![(1, 10), (2, 20)]),
                (1, vec![(0, 30)]),
                (2, vec![(0, 20), (2, 40)])
            ])
        );
        assert_eq!(
            labelled(g.to_undirected_with(UndirectedMode::Each).unwrap()),
            BTreeMap::from([
                (0, vec![(1, 10), (1, 30), (2, 20)]),
                (1, vec![(0, 30), (0, 10)]),
                (2, vec![(0, 20), (2, 40)])
            ])
        );
        // mutual: only 0 - 1 (both ways) and the self-loop
        assert_eq!(
            labelled(g.to_undirected_with(UndirectedMode::Mutual).unwrap()),
            BTreeMap::from([(0, vec![(1, 10)]), (1, vec![(0, 30)]), (2, vec![(2, 40)])])
        );
    }
}

#[cfg(test)]
mod directed_algorithms_tests {
    use super::*;
    use crate::centralities::exact::{ExactClosenessCentrality, ExactHarmonicCentrality};
    use crate::centralities::hyper_ball::HyperBall8;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use crate::trails::bfs::BFSDists;

    fn digraph(
        name: &str,
        n: usize,
        arcs: &[(usize, usize)],
    ) -> TestGraph<(), (), usize, Directed> {
        let dir = unique_test_dir("direction_algorithms");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut text = format!(
            "%%MatrixMarket matrix coordinate pattern general\n{n} {n} {}\n",
            arcs.len()
        );
        for (u, v) in arcs {
            text += &format!("{} {}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("direction_algorithms_{name}_{}", unique_run_tag());
        TestGraph::new(
            DiGraph::from_file(&path, Some(id), Some(2)).unwrap(),
            Some(dir),
        )
    }

    fn close(got: &[f64], want: &[f64], tolerance: f64) -> bool {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() <= tolerance)
    }

    /// On 0 -> 1 -> 2: outgoing distances are from a node, incoming ones to it (networkx's
    /// convention), and the undirected graph's ignore directions.
    #[test]
    fn traversals_and_centralities_follow_the_chosen_direction() {
        let g = digraph("path", 3, &[(0, 1), (1, 2)]);
        let undirected = g.to_undirected().unwrap();
        // unreached nodes are at distance 0, like the source
        assert_eq!(
            BFSDists::new_directed(&g, 0, Direction::Out)
                .unwrap()
                .get_distances(),
            &[0, 1, 2]
        );
        assert_eq!(
            BFSDists::new_directed(&g, 2, Direction::In)
                .unwrap()
                .get_distances(),
            &[2, 1, 0]
        );
        assert_eq!(
            BFSDists::new(undirected, 2).unwrap().get_distances(),
            &[2, 1, 0]
        );

        let closeness = |c: ExactClosenessCentrality<'_, (), (), usize, f64, _>| -> Vec<f64> {
            (0..3).map(|v| c.get_node_centrality(v)).collect()
        };
        // (n - 1) / total distance to the nodes reached
        assert!(close(
            &closeness(
                ExactClosenessCentrality::new_directed(&g, Some(false), Direction::Out).unwrap()
            ),
            &[2. / 3., 2., 0.],
            1e-12
        ));
        assert!(close(
            &closeness(
                ExactClosenessCentrality::new_directed(&g, Some(false), Direction::In).unwrap()
            ),
            &[0., 2., 2. / 3.],
            1e-12
        ));
        let both: Vec<f64> = {
            let c = ExactClosenessCentrality::<_, _, _, f64>::new(undirected, Some(false)).unwrap();
            (0..3).map(|v| c.get_node_centrality(v)).collect()
        };
        assert!(close(&both, &[2. / 3., 1., 2. / 3.], 1e-12));

        // networkx's harmonic_centrality (incoming): [0, 1, 1.5]
        let harmonic =
            ExactHarmonicCentrality::<_, _, _, f64, _>::new_directed(&g, None, Direction::In)
                .unwrap();
        let harmonic: Vec<f64> = (0..3).map(|v| harmonic.get_node_centrality(v)).collect();
        assert!(close(&harmonic, &[0., 1., 1.5], 1e-12));

        // HyperBall's estimates agree (exactly enough, on 3 nodes)
        let mut in_ball = HyperBall8::new_directed(&g, Direction::In).unwrap();
        let estimated = in_ball
            .get_or_compute_closeness_centrality(true)
            .unwrap()
            .to_vec();
        let exact = ExactClosenessCentrality::<_, _, _, f64, _>::new_directed(
            &g,
            Some(true),
            Direction::In,
        )
        .unwrap();
        let exact: Vec<f64> = (0..3).map(|v| exact.get_node_centrality(v)).collect();
        assert!(
            close(&estimated, &exact, 0.05),
            "{estimated:?} vs {exact:?}"
        );
    }

    #[test]
    fn hierholzer_needs_balanced_directed_graphs() {
        use crate::trails::hierholzer::AlgoHierholzer;
        let unbalanced = digraph("path", 3, &[(0, 1), (1, 2)]);
        let err = AlgoHierholzer::new(&unbalanced)
            .err()
            .expect("a path has no closed trail");
        assert!(
            err.to_string().contains("node 0 has 0 in and 1 out"),
            "{err}"
        );

        // two cycles sharing node 1: one closed trail over all 5 arcs
        let eulerian = digraph("cycles", 4, &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 1)]);
        let euler = AlgoHierholzer::new(&eulerian).unwrap();
        assert_eq!(euler.trail_number(), 1);
        let trail: Vec<usize> = euler
            .get_trail(0)
            .unwrap()
            .iter()
            .map(|v| v.index())
            .collect();
        let mut arcs: Vec<(usize, usize)> = (0..trail.len())
            .map(|i| (trail[i], trail[(i + 1) % trail.len()]))
            .collect();
        arcs.sort();
        assert_eq!(arcs, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 1)]);
    }

    /// In a directed graph the arc back to a node's parent closes a cycle; in an undirected one
    /// it's the tree edge walked back.
    #[test]
    fn dfs_visit_reports_back_edges_of_directed_graphs() {
        use crate::trails::dfs_visit::{DFSVisit, DfsVisitor};
        use std::ops::ControlFlow;

        #[derive(Default)]
        struct BackEdges(Vec<(usize, usize)>);
        impl DfsVisitor<usize> for BackEdges {
            fn back_edge(&mut self, u: usize, v: usize) -> ControlFlow<()> {
                self.0.push((u, v));
                ControlFlow::Continue(())
            }
        }
        let g = digraph("two_cycle", 2, &[(0, 1), (1, 0)]);
        let mut visitor = BackEdges::default();
        DFSVisit::new(&*g, 0, &mut visitor).unwrap();
        assert_eq!(visitor.0, vec![(1, 0)]);

        let mut visitor = BackEdges::default();
        DFSVisit::new(g.to_undirected().unwrap(), 0, &mut visitor).unwrap();
        assert_eq!(visitor.0, vec![]);
    }
}
