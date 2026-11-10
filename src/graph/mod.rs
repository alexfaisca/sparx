pub(crate) mod cache;
pub mod direction;
#[cfg(test)]
mod direction_regression_test;
mod edge_weight;
mod export_induced_subgraph;
mod impl_algos;
mod impl_csc;
mod impl_dists;
mod impl_miscelanious;
mod impl_partition;
pub(crate) mod impl_traits;
pub mod label;
mod weight;

#[cfg(feature = "petgraph")]
mod export_petgraph;
#[cfg(feature = "rustworkx")]
mod export_rustworkx_core;
#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

use crate::graph::cache::{AccumulatorAtomic, GraphCache};
use crate::shared_slice::{
    AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedSliceMut,
};
use cache::CacheMetadata;
pub use direction::{
    AnyGraph, Directed, Directedness, Direction, Undirected, UndirectedMode, WeightMerge,
};
pub use edge_weight::{EdgeWeight, ParseWeightError, WeightDomain, WeightParsing, Weighted};
pub use graph_derive::{GenericEdge, GenericEdgeType};
pub use weight::{Weight, WeightAtomic};

use label::VoidLabel;
use portable_atomic::{AtomicUsize, Ordering};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IndexedParallelIterator, ParallelIterator,
    plumbing::{Folder, Producer, UnindexedProducer, bridge, bridge_unindexed},
};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::{fmt::Debug, fs::File, sync::Arc};

pub use graph_derive::E;
pub use graph_derive::N;
use graph_derive::sparkx_label;

/// The default integer type for indexing.
///
/// The default type is `usize` to facilitate usage. As graphs that fit in `u32` or smaller unsizeds
/// won't have performance issues given the nature of our algorithms.
///
/// Used for node indices in memmaped `CSR` representation.
pub type DefaultIx = usize;

/// Trait for the unsigned integer type used for indexing garph elements.
///
/// # Safety
///
/// Marked `unsafe` because: the trait must faithfully preserve
/// and convert index values.
pub unsafe trait IndexType:
    Copy + Default + Hash + Ord + Debug + Send + Sync + 'static
{
    /// Width-matched atomic counterpart (see [`cache::AccumulatorAtomic`]), usable wherever a
    /// lock-free counter the same width as `Self` is needed (e.g. community-detection's
    /// aggregation-phase edge counters).
    ///
    /// [`cache::AccumulatorAtomic`]: ./cache/trait.AccumulatorAtomic.html#
    type Atomic: cache::AccumulatorAtomic<Plain = Self>;
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

unsafe impl IndexType for usize {
    type Atomic = portable_atomic::AtomicUsize;
    #[inline(always)]
    fn new(x: usize) -> Self {
        x
    }
    #[inline(always)]
    fn index(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn max() -> Self {
        usize::MAX
    }
}

unsafe impl IndexType for u32 {
    type Atomic = portable_atomic::AtomicU32;
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u32
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u32::MAX
    }
}

unsafe impl IndexType for u16 {
    type Atomic = portable_atomic::AtomicU16;
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u16
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u16::MAX
    }
}

unsafe impl IndexType for u8 {
    type Atomic = portable_atomic::AtomicU8;
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u8
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u8::MAX
    }
}

/// A directed graph's underlying undirected graphs, once built: one per [`UndirectedMode`] and,
/// for weighted edges, [`WeightMerge`].
type UndirectedSlots<N, E, Ix> =
    [std::sync::OnceLock<Box<GraphMemoryMap<N, E, Ix, Undirected>>>; 3 * WeightMerge::COUNT];

/// Node identifier.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NodeIndex<Ix = DefaultIx>(Ix);

impl<Ix: IndexType> NodeIndex<Ix> {
    #[inline]
    pub fn new(x: usize) -> Self {
        NodeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    #[inline]
    pub fn end() -> Self {
        NodeIndex(IndexType::max())
    }

    fn _into_edge(self) -> EdgeIndex<Ix> {
        EdgeIndex(self.0)
    }
}
/// Edge identifier.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct EdgeIndex<Ix = DefaultIx>(Ix);

impl<Ix: IndexType> EdgeIndex<Ix> {
    #[inline]
    pub fn new(x: usize) -> Self {
        EdgeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    /// An invalid `EdgeIndex` used to denote absence of an edge, for example
    /// to end an adjacency list.
    #[inline]
    pub fn end() -> Self {
        EdgeIndex(IndexType::max())
    }

    fn _into_node(self) -> NodeIndex<Ix> {
        NodeIndex(self.0)
    }
}

impl<Ix: IndexType> From<Ix> for EdgeIndex<Ix> {
    fn from(ix: Ix) -> Self {
        EdgeIndex(ix)
    }
}

/// Short version of `NodeIndex::new`
pub fn node_index<Ix: IndexType>(index: usize) -> NodeIndex<Ix> {
    NodeIndex::new(index)
}

/// Short version of `EdgeIndex::new`
pub fn edge_index<Ix: IndexType>(index: usize) -> EdgeIndex<Ix> {
    EdgeIndex::new(index)
}

/// Describes the behavior node labels must exhibit to be used by the tool.
///
/// Given a repr C/transparent struct the required traits may be derived automatically by
/// annotating the struct with #[sparkx_label].
#[allow(dead_code)]
pub trait N:
    Copy + std::fmt::Debug + Ord + bytemuck::Pod + bytemuck::Zeroable + Send + Sync
{
    fn new(v: usize) -> Self;
    #[inline]
    fn is_labeled() -> bool {
        false
    }
    #[inline]
    fn is_weighted() -> bool {
        false
    }
    #[inline]
    fn is_colored() -> bool {
        false
    }
    #[inline]
    fn label(&self) -> usize {
        0
    }
    #[inline]
    fn color(&self) -> usize {
        0
    }
    /// The node's weight, as an algorithm's `W`: `1` unless [`is_weighted`](Self::is_weighted).
    #[inline]
    fn weight<W: Weight>(&self) -> W {
        W::one()
    }
}

/// Describes the behavior edge labels must exhibit to be used by the tool.
///
/// Given a repr C/transparent struct the required traits may be derived automatically by
/// annotating the struct with #[sparkx_label].
///
/// Being labeled ([`is_labeled`](Self::is_labeled)) and being weighted
/// ([`is_weighted`](Self::is_weighted)) are independent: a labeled but unweighted edge (e.g. a
/// GGCAT edge's direction) keeps its label, yet weighs `1` to every algorithm. The ready-made
/// weighted label is [`Weighted`].
#[allow(dead_code)]
pub trait E:
    Copy + std::fmt::Debug + Ord + bytemuck::Pod + bytemuck::Zeroable + Send + Sync
{
    fn new(v: usize) -> Self;
    #[inline]
    fn is_labeled() -> bool {
        false
    }
    /// Whether the label holds a weight algorithms should use. When `false`, every edge weighs
    /// `1` and algorithms needn't read the edge label file at all.
    #[inline]
    fn is_weighted() -> bool {
        false
    }
    #[inline]
    fn is_colored() -> bool {
        false
    }
    #[inline]
    fn label(&self) -> usize {
        0
    }
    #[inline]
    fn color(&self) -> usize {
        0
    }
    /// The edge's weight, as an algorithm's `W`: `1` unless [`is_weighted`](Self::is_weighted).
    #[inline]
    fn weight<W: Weight>(&self) -> W {
        W::one()
    }
    /// The edge's weight as an exact non-negative integer, if it is one: `1` unless
    /// [`is_weighted`](Self::is_weighted). For algorithms computing exactly in integers.
    #[inline]
    fn integer_weight(&self) -> Option<u64> {
        Some(1)
    }
    /// What values the weight can hold (only meaningful if [`is_weighted`](Self::is_weighted)),
    /// checked once per input file against what the file declares.
    #[inline]
    fn weight_domain() -> WeightDomain {
        WeightDomain::Real
    }
    /// Reads a weight from an input token: validated if `CHECKED` (see [`WeightParsing`]),
    /// otherwise never failing. Only called if [`is_weighted`](Self::is_weighted).
    #[inline]
    fn parse_weight<const CHECKED: bool>(_tok: &[u8]) -> Result<Self, ParseWeightError> {
        Ok(Self::new(1))
    }
    /// The weight of the reverse of an arc in a skew-symmetric input.
    #[inline]
    fn negate_weight(self) -> Self {
        self
    }
    /// Two arcs' labels merged into one edge's (see [`WeightMerge`]), `None` if their weights'
    /// merge doesn't fit the weight type. Unweighted labels keep the first.
    #[inline]
    fn merge_weight(self, _other: Self, _merge: WeightMerge) -> Option<Self> {
        Some(self)
    }
}

/// Describes the behavior edge types must exhibit to be used by the tool.
///
/// Given an struct this trait may be automatically implemented using the provided procedural macro `GenericEdgeType`.
#[allow(dead_code)]
pub trait GenericEdgeType:
    Copy
    + Clone
    + Default
    + Debug
    + std::fmt::Display
    + PartialEq
    + Eq
    + bytemuck::Zeroable
    + From<u64>
    + From<usize>
    + rustworkx_core::petgraph::EdgeType
    + Send
    + Sync
{
    /// Edge label getter.
    fn label(&self) -> usize;
    /// Edge label setter.
    fn set_label(&mut self, label: u64);
}

/// Describes the behavior edges must exhibit to be used by the tool.
///
/// Given an implicitly packed struct this trait may be automatically implemented using the provided procedural macro `GenericEdge`.
#[allow(dead_code)]
pub trait GenericEdge<T: GenericEdgeType>:
    Copy
    + Clone
    + Default
    + Debug
    + std::fmt::Display
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + bytemuck::Pod
    + bytemuck::Zeroable
    + Send
    + Sync
{
    /// Constructor from an <<edge_dest: u64>> and an <<edge_type: u64>>
    fn new(edge_dest: u64, edge_type: u64) -> Self;
    /// Edge destiny node setter from a <<new_edge_dest: u64>>.
    fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self;
    /// Edge type setter from a <<new_edge_type: u64>>.
    fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self;
    /// Edge destiny node getter.
    fn dest(&self) -> usize;
    /// Edge type getter.
    fn e_type(&self) -> T;
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum CacheFile {
    General,
    BFS,
    Dijkstra,
    Landmarks,
    Spectral,
    LabelPropagation,
    DFS,
    DfsVisitor,
    BfsVisitor,
    EulerIndex,
    EulerTrail,
    KCoreBZ,
    KCoreLEA,
    Triangles,
    KTrussBEA,
    KTrussPKT,
    /// Triangle counts of a directed graph, of a given type.
    DirectedTriangles(crate::k_truss::directed::DirectedTriangleType),
    /// *Burkhardt et al.*'s k-truss decomposition of a directed graph, for a given triangle type.
    KTrussBEADirected(crate::k_truss::directed::DirectedTriangleType),
    /// *PKT*'s k-truss decomposition of a directed graph, for a given triangle type.
    KTrussPKTDirected(crate::k_truss::directed::DirectedTriangleType),
    /// A directed graph's in and out neighbours, merged (working memory of directed triangles).
    DirectedAdjacency,
    ClusteringCoefficient,
    EdgeReciprocal,
    EdgeOver,
    HyperBall,
    HyperBallDistances,
    HyperBallInvDistances,
    HyperBallClosenessCentrality,
    HyperBallHarmonicCentrality,
    HyperBallLinCentrality,
    GVELouvain,
    ExactCloseness,
    ExactHarmonic,
    ExactLin,
    /// Weakly connected components (connected components, for undirected graphs).
    ConnectedComponents,
    StronglyConnectedComponents,
    /// Biconnected components, articulation points, bridges and 2-edge-connected components.
    Biconnectivity,
    /// k-core decomposition of a directed graph, by in-, out- or total degree.
    KCoreDirected(crate::k_core::directed::CoreDegree),
    DirectedClusteringCoefficient,
    /// Louvain partition of a directed graph (by Leicht and Newman's directed modularity).
    DirectedLouvain,
    PageRank,
    /// Assortativity coefficients' scratch arrays.
    Assortativity,
}

enum GraphFile {
    Metadata,
    Neighbors,
    Offsets,
    NodeLabels,
    EdgeLabels,
    MetaLabels,
}

/// A memory-mapped graph: [`Graph`] (the default) for undirected graphs, [`DiGraph`] for
/// directed ones (see [`direction`]).
#[derive(Clone)]
pub struct GraphMemoryMap<
    N: crate::graph::N = VoidLabel,
    E: crate::graph::E = VoidLabel,
    Ix: crate::graph::IndexType = DefaultIx,
    Dir: Directedness = Undirected,
> {
    graph: Arc<memmap2::Mmap>,
    index: Arc<memmap2::Mmap>,
    metalabels: Arc<fst::Map<memmap2::Mmap>>,
    node_labels: Arc<memmap2::Mmap>,
    edge_labels: Arc<memmap2::Mmap>,
    graph_cache: GraphCache<N, E, Ix>,
    edge_size: usize,
    thread_count: u8,
    exports: u8,
    /// Whether every arc has a reverse: recorded in its metadata, or else checked on first use.
    symmetric: std::sync::OnceLock<bool>,
    /// The arcs' total weight (see [`total_weight`](Self::total_weight)), once summed.
    total_weight: std::sync::OnceLock<f64>,
    /// A directed graph's transpose and underlying undirected graph, once built.
    transpose: std::sync::OnceLock<Box<Self>>,
    /// (one per [`UndirectedMode`])
    /// (one per [`UndirectedMode`] and, for weighted edges, [`WeightMerge`])
    undirected: UndirectedSlots<N, E, Ix>,
    derive_lock: Arc<std::sync::Mutex<()>>,
    /// Tells this instance's scratch files apart from other instances' of the same entry.
    instance: InstanceTag,
    _marker: PhantomData<(Ix, Dir)>,
}

/// A tag unique to a graph instance (in this process and across processes): its scratch files
/// carry it. A clone is another instance, with another tag.
#[derive(Debug)]
struct InstanceTag(String);

impl InstanceTag {
    fn new() -> Self {
        static INSTANCES: AtomicUsize = AtomicUsize::new(0);
        InstanceTag(format!(
            "p{}i{}",
            std::process::id(),
            INSTANCES.fetch_add(1, Ordering::Relaxed)
        ))
    }
}

impl Clone for InstanceTag {
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// An undirected graph: every edge is stored as an arc each way, so every arc has a reverse
/// (see [`GraphMemoryMap::is_symmetric`]). Loading a directed one as a `Graph` fails.
pub type Graph<N = VoidLabel, E = VoidLabel, Ix = DefaultIx> = GraphMemoryMap<N, E, Ix, Undirected>;

/// A directed graph, whose arcs need not have reverses. Any graph can be loaded as one.
pub type DiGraph<N = VoidLabel, E = VoidLabel, Ix = DefaultIx> = GraphMemoryMap<N, E, Ix, Directed>;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    GraphMemoryMap<N, E, Ix, Dir>
{
    /// Initializes a [`GraphMemoryMap`] instance from a [`GraphCache`] instance[^1][^2].
    ///
    /// [^1]: [`GraphCache`] instance must be reandonly, dynamic graphs are not supported.
    /// [^2]: despite being readonly, the [`GraphCache`] instance's fst may be rebuilt *a posteriori* if proven necessary, checkout [`GraphCache`]'s documentation for more information on how to perform an fst rebuild.
    ///
    /// # Arguments
    ///
    /// * `graph_cache` --- [`GraphCache`] instance to be used.
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub(crate) fn init_from_cache(
        graph_cache: GraphCache<N, E, Ix>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if graph_cache.readonly {
            let graph = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.neighbors_file)? });
            let index = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.offsets_file)? });
            let metalabels = Arc::new(fst::Map::new(unsafe {
                memmap2::MmapOptions::new().map(&File::open(&graph_cache.metalabel_filename)?)?
            })?);
            // always mmapped, regardless of `N`/`E::is_labeled()`: the backing files are
            // guaranteed non-empty (at least 1 byte) even for zero-sized label types, see
            // `AbstractedProceduralMemoryMut::from_file`'s `.max(1)` sizing.
            let node_labels = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.nodelabel_file)? });
            let edge_labels = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.edgelabel_file)? });
            let edge_size = std::mem::size_of::<Ix>();
            let thread_count = thread_count.unwrap_or(1).max(1);
            let exports = 0u8;
            let symmetric = std::sync::OnceLock::new();
            if let Some(s) = graph_cache
                .symmetric
                .or_else(|| graph_cache.metadata().ok().and_then(|m| m.symmetric))
            {
                let _ = symmetric.set(s);
            }

            let graph = Self {
                graph,
                index,
                metalabels,
                node_labels,
                edge_labels,
                graph_cache,
                edge_size,
                thread_count,
                exports,
                symmetric,
                total_weight: std::sync::OnceLock::new(),
                transpose: std::sync::OnceLock::new(),
                undirected: Default::default(),
                derive_lock: Arc::new(std::sync::Mutex::new(())),
                instance: InstanceTag::new(),
                _marker: PhantomData,
            };
            // an undirected graph is symmetric (a property recorded when the cache was built)
            if !Dir::DIRECTED {
                graph.require_symmetric()?;
            }
            return Ok(graph);
        }

        Err("error graph cache must be readonly to be memmapped".into())
    }

    /// Creates a [`GraphMemoryMap`] instance from a given file.
    ///
    /// # Arguments
    ///
    /// * `p` --- path to the file from which the graph is to be parsed.
    /// * `id` --- optionally, the user may provide an id for the graph.
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// Like [`init_from_cache`](Self::init_from_cache), for a cache entry just built: if it
    /// can't be opened as this type (a directed graph as a [`Graph`]), the entry is removed.
    fn init_from_new_cache(
        graph_cache: GraphCache<N, E, Ix>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut entry = graph_cache.clone();
        Self::init_from_cache(graph_cache, thread_count).inspect_err(|_| {
            let _ = entry.drop_cache();
        })
    }

    #[inline(always)]
    pub fn from_file<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<N, E, Ix>::from_file(p, id, None, None)?;
        Self::init_from_new_cache(graph_cache, thread_count)
    }

    /// Same as [`from_file`](Self::from_file), but lets the caller choose how edge weights are
    /// read (see [`WeightParsing`]); [`from_file`](Self::from_file) reads them
    /// [`Fast`](WeightParsing::Fast), unvalidated.
    #[inline(always)]
    pub fn from_file_with_weight_parsing<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        thread_count: Option<u8>,
        parsing: WeightParsing,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache =
            GraphCache::<N, E, Ix>::from_file_with_weight_parsing(p, id, None, None, parsing)?;
        Self::init_from_new_cache(graph_cache, thread_count)
    }

    /// Same as [`from_file`](Self::from_file), but lets the caller choose the width-matched
    /// atomic type (`D`) used for lock-free per-node degree accumulation and edge-position
    /// tracking during parsing (see [`cache::AccumulatorAtomic`]). Defaults to `AtomicUsize`
    /// (always safe); `AtomicU32`/`AtomicU16` trade a (checked, never silently wrong) range
    /// restriction for reduced memory traffic while streaming through these buffers ---
    /// worthwhile for graphs comfortably within that narrower width (e.g. under ~4B edges for
    /// `AtomicU32`).
    ///
    /// [`cache::AccumulatorAtomic`]: ./cache/trait.AccumulatorAtomic.html#
    #[inline(always)]
    pub fn from_file_with_accumulator<P: AsRef<Path>, D: AccumulatorAtomic>(
        p: P,
        id: Option<String>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache =
            GraphCache::<N, E, Ix>::from_file_with_accumulator::<P, D>(p, id, None, None)?;
        Self::init_from_new_cache(graph_cache, thread_count)
    }

    /// Initializes a [`GraphMemoryMap`] instance from a given file, and builds an fst.
    ///
    /// # Arguments
    ///
    /// * `p` --- path to the file from which the graph is to be parsed.
    /// * `id` --- optionally, the user may provide an id for the graph.
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[inline(always)]
    pub fn from_file_with_fst<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<N, E, Ix>::from_file(p, id, batch, in_fst)?;
        Self::init_from_new_cache(graph_cache, thread_count)
    }

    /// Opens a [`GraphMemoryMap`] instance from a cached entry.
    ///
    /// # Arguments
    ///
    /// * `filename` --- path of one of the graph's cached entry's files (may be any of the graph's files).
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[inline(always)]
    pub fn open<P: AsRef<Path>>(
        filename: P,
        batch: Option<usize>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<N, E, Ix>::open(filename, batch)?;
        Self::init_from_cache(graph_cache, thread_count)
    }

    #[inline(always)]
    pub(crate) fn offsets_ptr(&self) -> *const Ix {
        self.index.as_ptr() as *const Ix
    }

    #[inline(always)]
    pub(crate) fn neighbours_ptr(&self) -> *const Ix {
        self.graph.as_ptr() as *const Ix
    }

    /// The edge labels, in flat edge order. Only meaningful (sized [`width`](Self::width)) for
    /// labeled edges (see [`E::is_labeled`]).
    #[inline(always)]
    pub(crate) fn edge_labels_ptr(&self) -> *const E {
        self.edge_labels.as_ptr() as *const E
    }

    /// Returns the (suggested) number of threads being used for computations on the graph.
    #[inline(always)]
    pub fn thread_num(&self) -> usize {
        self.thread_count.max(1) as usize
    }

    /// Returns the graph's cache id.
    #[inline(always)]
    pub fn graph_id(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.cache_id()
    }

    pub fn metadata(&self) -> Result<CacheMetadata, Box<dyn std::error::Error>> {
        self.graph_cache.metadata()
    }

    /// Returns the path of the graph's cache entry's metadata file, which [`open`](Self::open)
    /// accepts to reopen the graph later without parsing its dataset again.
    pub fn cache_filename(&self) -> String {
        self.graph_cache.metadata_filename()
    }

    /// Converts this [`GraphMemoryMap`] instance's underlying cache entry into a brand new cache
    /// entry that uses a different index type, `NewIx`, re-encoding the `CSR` offsets/neighbors
    /// arrays at the new width, and opens it as a new [`GraphMemoryMap`] instance.
    ///
    /// The original cache entry (and this instance) is left untouched; the new entry's
    /// `dataset_name` is the original's with `_<new index size in bytes>` appended, and it is
    /// assigned its own, freshly derived, `cache_id`.
    ///
    /// This is required whenever a cache is to be opened with an `Ix` that does not match the
    /// one it was built with (opening with a mismatched `Ix` is rejected, see [`GraphMemoryMap::open`]).
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn convert_index_type<NewIx: crate::graph::IndexType>(
        &self,
    ) -> Result<GraphMemoryMap<N, E, NewIx, Dir>, Box<dyn std::error::Error>> {
        let new_cache = self.graph_cache.convert_index_type::<NewIx>()?;
        GraphMemoryMap::<N, E, NewIx, Dir>::init_from_cache(new_cache, Some(self.thread_count))
    }

    /// Returns the graph's cache entry metadata filename.
    #[inline(always)]
    pub fn cache_metadata_filename(&self) -> String {
        self.graph_cache.metadata_filename()
    }

    /// Returns the graph's neighbors' file's filename.
    #[inline(always)]
    pub fn cache_edges_filename(&self) -> String {
        self.graph_cache.neighbors_filename()
    }

    /// Returns the graph's offsets' file's filename.
    #[inline(always)]
    pub fn cache_index_filename(&self) -> String {
        self.graph_cache.offsets_filename()
    }

    /// Returns the graph's node labels' file's filename.
    #[inline(always)]
    pub fn cache_node_labels_filename(&self) -> String {
        self.graph_cache.nodelabels_filename()
    }

    /// Returns the graph's edge labels' file's filename.
    #[inline(always)]
    pub fn cache_edge_labels_filename(&self) -> String {
        self.graph_cache.edgelabels_filename()
    }

    /// Returns the graph's fst file's filename.
    #[inline(always)]
    pub fn cache_fst_filename(&self) -> String {
        self.graph_cache.fst_filename()
    }

    /// Returns the given (by id) node's degree: its number of arcs out, on directed graphs
    /// (see [`in_degree`](Self::in_degree) for arcs in).
    #[inline(always)]
    pub fn node_degree(&self, node_id: usize) -> usize {
        assert!(node_id < self.size());
        unsafe {
            let ptr = (self.index.as_ptr() as *const Ix).add(node_id);
            let begin = ptr.read_unaligned().index();
            ptr.add(1).read_unaligned().index() - begin
        }
    }

    /// Returns the given (by id) node's metalabel if it exists and was stored in the graph's fst.
    #[inline(always)]
    pub fn node_id_from_metalabel(
        &self,
        metalabel: &str,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if let Some(val) = self.metalabels.get(metalabel) {
            Ok(val)
        } else {
            Err(format!("error metalabel {metalabel} not found").into())
        }
    }

    /// Returns the given (by id) node's edges' offsets.
    #[inline(always)]
    pub fn index_node(&self, node_id: usize) -> std::ops::Range<usize> {
        assert!(node_id < self.size());
        unsafe {
            let ptr = (self.index.as_ptr() as *const Ix).add(node_id);
            ptr.read_unaligned().index()..ptr.add(1).read_unaligned().index()
        }
    }

    /// Returns a [`NeighbourIter`] iterator over the given (by id) node's neighbours, sorted: its
    /// out-neighbours, on directed graphs (see [`in_neighbours`](Self::in_neighbours) and
    /// [`view`](Self::view) for others).
    ///
    /// [`NeighbourIter`]: ./struct.NeighbourIter.html#
    pub fn neighbours(
        &self,
        node_id: usize,
    ) -> Result<NeighbourIter<N, E, Ix>, Box<dyn std::error::Error>> {
        if node_id >= self.size() {
            return Err(
                format!("error {node_id} must be smaller than |V| = {}", self.size()).into(),
            );
        }

        Ok(NeighbourIter::<N, E, Ix>::new(
            self.graph.as_ptr() as *const Ix,
            self.index.as_ptr() as *const Ix,
            node_id,
        ))
    }

    /// Returns number of entries in the offset file[^1].
    ///
    /// [^1]: this is equivalent to `|V| + 1`, as there is an extra offset file entry to mark the end of edges' offsets.
    #[inline(always)]
    pub fn offsets_size(&self) -> usize {
        self.graph_cache.index_bytes // num nodes
    }

    /// Returns number of nodes in the offset file[^1].
    ///
    /// Performs a saturating subtraction of 1 to the number of entries in the offset file.
    ///
    /// [^1]: this is equivalent to `|V|`.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.graph_cache.index_bytes.saturating_sub(1)
    }

    /// Returns number of entries in edge file[^1].
    ///
    /// [^1]: this is equivalent to `|E|`.
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.graph_cache.graph_bytes // num edges
    }

    fn exports_fetch_increment(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        self.exports = match self.exports.overflowing_add(1) {
            (r, false) => r,
            (_, true) => {
                self.exports = u8::MAX;
                return Err(
                    "error overflowed export count var in graph struct, please provide an identifier for your export".into()
                );
            }
        };
        Ok((self.exports - 1) as usize)
    }

    /// Applies the mask: fn(usize) -> bool function to each node id and returns the resulting subgraph.
    ///
    /// The resulting subgraph is that wherein only the set of nodes, `S ⊂ V`, of the nodes for whose the output of mask is true, as well as, only the set of edges coming from and going to nodes in `S`[^1].
    ///
    /// [^1]: the subgraph's nodes are numbered `0..|S|`, in their order here (so node `u` of `S`
    /// is the subgraph's node `|{v ∈ S : v < u}|`); they keep their labels and metalabels, and
    /// arcs theirs. It's a cache entry of its own (id `identifier`, or a numbered export of this
    /// graph's), of the same type: an undirected graph's subgraphs are undirected.
    pub fn induced_subgraph(
        &mut self,
        mask: impl Fn(usize) -> bool,
        identifier: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        self.apply_mask_to_nodes(mask, identifier)
    }

    #[inline(always)]
    pub fn check_neighbor(&self, u: usize, v: usize) -> Option<usize> {
        self.is_neighbour_impl(u, v)
    }

    #[inline(always)]
    pub fn check_triangle(&self, u: usize, v: usize, w: usize) -> Option<(usize, usize)> {
        self.is_triangle_impl(u, v, w)
    }

    #[inline(always)]
    fn build_helper_filename(&self, seq: usize) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.build_helper_filename(seq)
    }

    #[inline(always)]
    fn cleanup_helpers(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache.cleanup_helpers()
    }

    /// Build a persistent cached file (`.mmap`) of a given [`CacheFile`] type for the [`GraphMemoryMap`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[inline(always)]
    pub fn build_pers_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.build_pers_cache_filename(file_type, seq)
    }

    /// Build a cached file (either `.mmap` or `.tmp`) of a given [`CacheFile`] type for the [`GraphMemoryMap`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[inline(always)]
    pub fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match seq {
            None => self.graph_cache.build_cache_filename(file_type, None),
            // scratch files are this instance's own
            Some(seq) => {
                self.graph_cache
                    .build_instance_cache_filename(file_type, seq, &self.instance.0)
            }
        }
    }

    /// Removes the [`GraphMemoryMap`] instance's cached `.tmp` files of a given [`CacheFile`]
    /// with the given sequence numbers (those that exist) --- unlike [`cleanup_cache`], without
    /// listing the whole cache directory.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`cleanup_cache`]: #method.cleanup_cache
    pub(crate) fn remove_tmp_files(
        &self,
        target: CacheFile,
        seqs: std::ops::Range<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for seq in seqs {
            match std::fs::remove_file(self.build_cache_filename(target.clone(), Some(seq))?) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }
        Ok(())
    }

    /// Remove [`GraphMemoryMap`] instance's cached `.tmp` files for a given [`CacheFile`] in the cache directory.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`CacheFile`]: ./enum.CacheFile.html#
    #[inline(always)]
    pub fn cleanup_cache(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache
            .cleanup_instance_cache(target, &self.instance.0)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// Every stored arc becomes one petgraph edge: an undirected graph's edges become a pair of
    /// opposite edges each.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[cfg(feature = "petgraph")]
    pub fn export_petgraph_stripped(
        &self,
    ) -> Result<petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        self.export_petgraph_stripped_impl()
    }

    // /// Export the [`GraphMemoryMap`] instance to rustworkx_core compatible petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    // ///
    // /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    // ///
    // /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    // /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    // #[cfg(feature = "rustworkx")]
    // pub fn export_rustworkx(
    //     &self,
    // ) -> Result<
    //     rustworkx_core::petgraph::graph::DiGraph<
    //         rustworkx_core::petgraph::graph::NodeIndex<usize>,
    //         EdgeType,
    //     >,
    //     Box<dyn std::error::Error>,
    // > {
    //     self.export_rustworkx_impl()
    // }

    /// Export the [`GraphMemoryMap`] instance to rustworkx_core compatible petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[cfg(feature = "rustworkx")]
    pub fn export_rustworkx_stripped(
        &self,
    ) -> Result<rustworkx_core::petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        self.export_rustworkx_stripped_impl()
    }

    pub fn iter_node_indices(&self) -> NodeIndices<N, E, Ix> {
        NodeIndices::new(0, self.size())
    }

    pub fn iter_edge_indices(&self) -> EdgeIndices<N, E, Ix> {
        EdgeIndices::new(0, self.width())
    }

    pub fn paralelizable_iter_node_indices(&self) -> ParalelizableNodeIndices<N, E, Ix> {
        ParalelizableNodeIndices::new(0, self.size())
    }

    pub fn paralelizable_iter_edge_indices(&self) -> ParalelizableEdgeIndices<N, E, Ix> {
        ParalelizableEdgeIndices::new(0, self.width())
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_node_indices(&self) -> ParNodeIndices<N, E, Ix> {
        ParNodeIndices::new(0, self.size())
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_edge_indices(&self) -> ParEdgeIndices<N, E, Ix> {
        ParEdgeIndices::new(0, self.width())
    }

    /// Iterates over the nodes without neighbours: on directed graphs, without arcs out (sinks
    /// included; nodes without arcs in or out are isolated in [`to_undirected`](Self::to_undirected)).
    pub fn iter_isolated_nodes(&self) -> NodesWithDegree<N, E, Ix> {
        NodesWithDegree::new(self.offsets_ptr(), self.size(), 0)
    }

    pub fn paralelizable_iter_isolated_nodes(&self) -> ParalelizableNodesWithDegree<N, E, Ix> {
        ParalelizableNodesWithDegree::new(self.offsets_ptr(), self.size(), 0)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_isolated_nodes(&self) -> ParNodesWithDegree<N, E, Ix> {
        ParNodesWithDegree::new(self.offsets_ptr(), self.size(), 0)
    }

    /// Iterates over the nodes of a given [degree](Self::node_degree) (out-degree, on directed
    /// graphs).
    pub fn iter_nodes_with_degree(&self, degree: usize) -> NodesWithDegree<N, E, Ix> {
        NodesWithDegree::new(self.offsets_ptr(), self.size(), degree)
    }

    pub fn paralelizable_iter_nodes_with_degree(
        &self,
        degree: usize,
    ) -> ParalelizableNodesWithDegree<N, E, Ix> {
        ParalelizableNodesWithDegree::new(self.offsets_ptr(), self.size(), degree)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_nodes_with_degree(&self, degree: usize) -> ParNodesWithDegree<N, E, Ix> {
        ParNodesWithDegree::new(self.offsets_ptr(), self.size(), degree)
    }

    /// Iterates over the nodes of [degree](Self::node_degree) below a given one (out-degree, on
    /// directed graphs).
    pub fn iter_nodes_with_degree_lt(&self, degree: usize) -> NodesWithDegreeLT<N, E, Ix> {
        NodesWithDegreeLT::new(self.offsets_ptr(), self.size(), degree)
    }

    pub fn paralelizable_iter_nodes_with_degree_lt(
        &self,
        degree: usize,
    ) -> ParalelizableNodesWithDegreeLT<N, E, Ix> {
        ParalelizableNodesWithDegreeLT::new(self.offsets_ptr(), self.size(), degree)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_nodes_with_degree_lt(&self, degree: usize) -> ParNodesWithDegreeLT<N, E, Ix> {
        ParNodesWithDegreeLT::new(self.offsets_ptr(), self.size(), degree)
    }

    /// Iterates over the nodes of [degree](Self::node_degree) at least a given one (out-degree, on
    /// directed graphs).
    pub fn iter_nodes_with_degree_ge(&self, degree: usize) -> NodesWithDegreeGE<N, E, Ix> {
        NodesWithDegreeGE::new(self.offsets_ptr(), self.size(), degree)
    }

    pub fn paralelizable_iter_nodes_with_degree_ge(
        &self,
        degree: usize,
    ) -> ParalelizableNodesWithDegreeGE<N, E, Ix> {
        ParalelizableNodesWithDegreeGE::new(self.offsets_ptr(), self.size(), degree)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_nodes_with_degree_ge(&self, degree: usize) -> ParNodesWithDegreeGE<N, E, Ix> {
        ParNodesWithDegreeGE::new(self.offsets_ptr(), self.size(), degree)
    }

    pub fn iter_neighbors(&self, idx: usize) -> Neighbors<N, E, Ix> {
        Neighbors::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    #[cfg(feature = "iter_with_offset")]
    pub fn iter_neighbors_with_offset(&self, idx: usize) -> NeighborsWithOffset<N, E, Ix> {
        NeighborsWithOffset::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    pub fn paralelizable_iter_neighbors(&self, idx: usize) -> ParalelizableNeighbors<N, E, Ix> {
        ParalelizableNeighbors::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_neighbors(&self, idx: usize) -> ParNeighbors<N, E, Ix> {
        ParNeighbors::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    pub fn walk_neighbors(&self, idx: usize) -> WalkNeighbors<N, E, Ix> {
        WalkNeighbors::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    #[cfg(feature = "iter_with_offset")]
    pub fn walk_neighbors_with_offset(&self, idx: usize) -> WalkNeighborsWithOffset<N, E, Ix> {
        WalkNeighborsWithOffset::new(self.neighbours_ptr(), self.offsets_ptr(), idx)
    }

    /// Returns the given (by id) node's associated label/weight.
    pub fn node_label(&self, node_id: usize) -> N {
        assert!(node_id < self.size());
        NodeLabels::<N, E, Ix>::read(self.node_labels.as_ptr() as *const N, node_id)
    }

    /// Returns the given (by flat edge-array position) edge's associated label/weight.
    pub fn edge_label(&self, edge_pos: usize) -> E {
        assert!(edge_pos < self.width());
        EdgeLabels::<N, E, Ix>::read(self.edge_labels.as_ptr() as *const E, edge_pos)
    }

    /// Returns a [`NodeLabels`] iterator over every node's associated label/weight, in node-id
    /// order.
    pub fn iter_node_labels(&self) -> NodeLabels<N, E, Ix> {
        NodeLabels::new(self.node_labels.as_ptr() as *const N, self.size())
    }

    /// Returns an [`EdgeLabels`] iterator over every edge's associated label/weight, in flat
    /// edge-array order.
    pub fn iter_edge_labels(&self) -> EdgeLabels<N, E, Ix> {
        EdgeLabels::new(self.edge_labels.as_ptr() as *const E, self.width())
    }

    /// Returns a [`WeightedNeighbours`] iterator over the given (by id) node's neighbours,
    /// paired with each corresponding edge's label/weight.
    pub fn weighted_neighbours(
        &self,
        node_id: usize,
    ) -> Result<WeightedNeighbours<N, E, Ix>, Box<dyn std::error::Error>> {
        if node_id >= self.size() {
            return Err(
                format!("error {node_id} must be smaller than |V| = {}", self.size()).into(),
            );
        }
        let range = self.index_node(node_id);
        Ok(WeightedNeighbours::new(
            self.neighbours_ptr(),
            self.edge_labels.as_ptr() as *const E,
            range.start,
            range.end,
        ))
    }

    /// A node's weighted degree: the total weight of its arcs out (its degree, for unweighted
    /// arcs; see [`E::is_weighted`]).
    pub fn weighted_degree(&self, node_id: usize) -> Result<f64, Box<dyn std::error::Error>> {
        if !E::is_weighted() {
            return Ok(self.node_degree(node_id) as f64);
        }
        Ok(self
            .weighted_neighbours(node_id)?
            .map(|(_, label)| label.weight::<f64>())
            .sum())
    }

    /// The arcs' total weight: their number, for unweighted arcs (see [`E::is_weighted`]), or
    /// the sum of their weights, which must be finite and non-negative. Summed once (in `f64`,
    /// in parallel, in a fixed order), then remembered.
    pub fn total_weight(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if !E::is_weighted() {
            return Ok(self.width() as f64);
        }
        if let Some(&total) = self.total_weight.get() {
            return Ok(total);
        }
        let arcs = self.width();
        let labels = crate::shared_slice::SharedSlice::<E>::new(self.edge_labels_ptr(), arcs);
        let chunks = (self.thread_num().max(1)).min(arcs.max(1));
        let load = arcs.div_ceil(chunks);
        let parts: Vec<Result<f64, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..chunks)
                .map(|c| {
                    s.spawn(move || {
                        let mut total = 0f64;
                        for e in (c * load).min(arcs)..((c + 1) * load).min(arcs) {
                            let label = labels.get(e);
                            let w = label.weight::<f64>();
                            if !(w.is_finite() && w >= 0.) {
                                return Err(format!(
                                    "error edge weights must be finite and non-negative here, but one is {label:?}"
                                ));
                            }
                            total += w;
                        }
                        Ok(total)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|e| Err(format!("{e:?}"))))
                .collect()
        });
        let mut total = 0f64;
        for part in parts {
            total += part?;
        }
        Ok(*self.total_weight.get_or_init(|| total))
    }

    /// Returns an [`Edges`] iterator over every edge in the graph, as `(source, target, weight)`
    /// triples, in flat edge-array order.
    pub fn edges(&self) -> Edges<N, E, Ix> {
        Edges::new(
            self.offsets_ptr(),
            self.neighbours_ptr(),
            self.edge_labels.as_ptr() as *const E,
            self.size(),
            self.width(),
        )
    }

    /// Returns a [`Dfs`] iterator yielding every node reachable from `source` (along arcs out of
    /// nodes, on directed graphs: run it on a [`view`](Self::view) for others), in depth-first
    /// discovery order (`source` itself first) --- a lazy, pull-based alternative to
    /// [`DFSVisit`](crate::trails::dfs_visit::DFSVisit)'s full callback-based traversal, for
    /// callers that just want nodes in DFS order (e.g. `graph.dfs(0)?.take(5).collect()`)
    /// without implementing a visitor. Like the rest of this crate, its "visited" marker and
    /// stack are memory-mapped (not heap `Vec`s), so it scales to graphs far larger than RAM.
    pub fn dfs(&self, source: usize) -> Result<Dfs<'_, N, E, Ix, Dir>, Box<dyn std::error::Error>> {
        if source >= self.size() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                self.size()
            )
            .into());
        }
        Dfs::new(self, source)
    }

    /// Returns a [`Bfs`] iterator yielding every node reachable from `source` (along arcs out of
    /// nodes, on directed graphs: run it on a [`view`](Self::view) for others), in breadth-first
    /// discovery order (`source` itself first). See [`GraphMemoryMap::dfs`] for the rationale.
    pub fn bfs(&self, source: usize) -> Result<Bfs<'_, N, E, Ix, Dir>, Box<dyn std::error::Error>> {
        if source >= self.size() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                self.size()
            )
            .into());
        }
        Bfs::new(self, source)
    }

    /// Removes the graph's cache entry, along with every file left in the cache under its id
    /// (algorithms' results that weren't dropped, scratch files of interrupted runs, ...),
    /// which would be of no use without the graph.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut this = ManuallyDrop::new(self);

        let id = this.graph_id()?;
        let cache_dir = Path::new(&this.cache_filename())
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();

        let er_fn = this.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let eo_fn = this.build_cache_filename(CacheFile::EdgeOver, None)?;
        // the graphs derived from it (its transpose, underlying undirected graph, theirs...),
        // whether built in this process or an earlier one, with their algorithms' results
        cache::drop_derived_of(&id)?;

        let _r = crate::shared_slice::remove_if_exists(er_fn);
        let _r = crate::shared_slice::remove_if_exists(eo_fn);

        this.graph_cache.drop_cache()?;

        // cache files are named `<kind>[_<seq>]_<id>.<ext>`
        let pattern = cache_dir.join(format!("*{id}.*"));
        for f in glob::glob(&pattern.to_string_lossy())?.flatten() {
            let _r = crate::shared_slice::remove_if_exists(f);
        }
        Ok(())
    }
}

/// Methods of undirected graphs only.
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Graph<N, E, Ix> {
    /// Performs a sweep cut over a given diffusion vector[^1] by partition conductance.
    ///
    /// # Arguments
    ///
    /// * `diffusion` --- the diffusion vector[^1][^2].
    /// * `target_size` --- the partition's target size[^3].
    /// * `target_volume` --- the partition's target volume[^4].
    ///
    /// [^1]: diffusion vector entries must be of type (node_id: [`usize`], heat: [`f64`]).
    /// [^2]: entries must be descendingly ordered by diffusion.
    /// [^3]: if [`None`] is provided defaults to `|V|`, effectively, the overall best partition by conducatance is returned independent on the number of nodes in it.
    /// [^4]: if [`None`] is provided defaults to `|E|`, effectively, the overall best partition by conducatance is returned independent on the number of edges in it.
    pub fn sweep_cut(
        &self,
        diffusion: &mut [(usize, f64)],
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        self.sweep_cut_over_diffusion_vector_by_conductance(diffusion, target_size, target_volume)
    }

    /// Computes the modularity over a given partition over the [`GraphMemoryMap`] instance's nodes.
    ///
    /// For directed graphs, see
    /// [`directed_modularity`](crate::communities::directed_louvain::directed_modularity) (Leicht
    /// and Newman's directed modularity).
    ///
    /// # Arguments
    ///
    /// * `communities` --- a slice containing each node's community, the [`GraphMemoryMap`] instance's partition.
    /// * `comms_cardinality` --- cardinality of distinct communities in the given partition.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn modularity(
        &self,
        communities: &[Ix],
        comms_cardinality: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.modularity_impl(communities, comms_cardinality)
    }

    #[inline(always)]
    /// Computes or retrieves from a previously cached file, the `Compressed Sparse Column (CSC)`
    /// representation of the graph.
    ///
    /// This is equivalent to each edge's reciprocal edge in the `Compressed Sparse Row (CSR)`
    /// representation of the graph, i.e., if `e_i = (u, v)` then it's reciprocal `e_i' = (v, u)`.
    ///
    pub fn edge_reciprocal(
        &self,
    ) -> Result<AbstractedProceduralMemory<Ix>, Box<dyn std::error::Error>> {
        self.get_edge_reciprocal_impl()
    }

    #[inline(always)]
    /// Computes or retrieves from a previously cached file, the offset of the first neighbor whose
    /// id is bigger than a given node's id, in the latter's neighbor list.
    ///
    /// If node u has a neighbor list (sorted by id, ascendingly) `n_u = {a, b, c, d, e}`, and `id[c]
    /// =< id[u]`, but `id[d] > id[u]` then u's `edge_over` entry, `edge_over[u] = 3`.
    ///
    pub fn edge_over(&self) -> Result<AbstractedProceduralMemory<Ix>, Box<dyn std::error::Error>> {
        self.get_edge_over_impl()
    }
}

#[derive(Debug, Clone)]
pub struct NeighbourIter<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    edge_ptr: *const Ix,
    _orig_edge_ptr: *const Ix,
    _orig_id_ptr: *const Ix,
    id: usize,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> NeighbourIter<N, E, Ix> {
    fn new(edge_mmap: *const Ix, id_mmap: *const Ix, node_id: usize) -> Self {
        let _orig_edge_ptr = edge_mmap;
        let _orig_id_ptr = id_mmap;
        let id_ptr = unsafe { id_mmap.add(node_id) };
        let offset = unsafe { id_ptr.read_unaligned().index() };

        NeighbourIter {
            edge_ptr: unsafe { edge_mmap.add(offset) },
            _orig_edge_ptr,
            _orig_id_ptr,
            id: node_id,
            count: unsafe { id_ptr.add(1).read_unaligned().index() - offset },
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    fn _into_neighbour(&self) -> Self {
        NeighbourIter::new(self._orig_edge_ptr, self._orig_id_ptr, unsafe {
            self.edge_ptr.read_unaligned().index()
        })
    }

    fn _next_back_with_offset(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: (usize, usize);
        unsafe {
            next = (
                self.id,
                self.edge_ptr.add(self.count).read_unaligned().index(),
            );
        };
        Some(next)
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }

    fn _next_with_offset(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: (usize, usize);
        self.edge_ptr = unsafe {
            next = (self.offset, self.edge_ptr.read_unaligned().index());
            self.edge_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NeighbourIter<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: usize;
        unsafe {
            next = self.edge_ptr.add(self.count).read_unaligned().index();
        };
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NeighbourIter<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        self.offset += 1;
        let next: usize;
        self.edge_ptr = unsafe {
            next = self.edge_ptr.read_unaligned().index();
            self.edge_ptr.add(1)
        };
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NeighbourIter<N, E, Ix>
{
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    std::ops::Index<std::ops::RangeFull> for GraphMemoryMap<N, E, Ix, Dir>
{
    type Output = [Ix];
    #[inline]
    fn index(&self, _index: std::ops::RangeFull) -> &[Ix] {
        unsafe { std::slice::from_raw_parts(self.graph.as_ptr() as *const Ix, self.width()) }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    std::ops::Index<std::ops::Range<usize>> for GraphMemoryMap<N, E, Ix, Dir>
{
    type Output = [Ix];
    #[inline]
    fn index(&self, index: std::ops::Range<usize>) -> &[Ix] {
        unsafe {
            std::slice::from_raw_parts(
                self.graph.as_ptr().add(index.start * self.edge_size) as *const Ix,
                index.end - index.start,
            )
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    std::ops::Index<std::ops::Range<u64>> for GraphMemoryMap<N, E, Ix, Dir>
{
    type Output = [Ix];
    #[inline]
    fn index(&self, index: std::ops::Range<u64>) -> &[Ix] {
        let start = index.start as usize;
        let end = index.end as usize;

        unsafe {
            std::slice::from_raw_parts(
                self.graph.as_ptr().add(start * self.edge_size) as *const Ix,
                end - start,
            )
        }
    }
}

/// Iterating over `&GraphMemoryMap` directly yields every node's id, in id order --- equivalent
/// to `graph.iter_node_indices()`, provided for idiomatic `for node in &graph` usage.
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    IntoIterator for &GraphMemoryMap<N, E, Ix, Dir>
{
    type Item = usize;
    type IntoIter = NodeIndices<N, E, Ix>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_node_indices()
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness> Debug
    for GraphMemoryMap<N, E, Ix, Dir>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryMappedData {{\n\t
            neighbors_filename: {},\n\t
            offsets_filename: {},\n\t
            size: {},\n\t
            width: {},\n\t
            }}",
            self.graph_cache.neighbors_filename(),
            self.graph_cache.offsets_filename(),
            self.size(),
            self.width(),
        )
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Community<NodeId: Copy + Debug> {
    /// (s, h(s)), ∀ s ϵ S, where h(s) is the heat kernel diffusion for a given node s
    pub nodes: Vec<(NodeId, f64)>,
    /// |S|
    pub size: usize,
    /// vol(S), in arcs: the number of arcs out of S's nodes
    pub width: usize,
    /// vol(S), weighted: the total weight of the arcs out of S's nodes (`width`, for unweighted
    /// arcs). What `conductance` is computed with.
    pub volume: f64,
    /// ϕ(S)
    pub conductance: f64,
}

// Iterators

pub struct NodeIndices<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    begin: usize,
    end: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> NodeIndices<N, E, Ix> {
    fn new(begin: usize, end: usize) -> Self {
        Self {
            begin,
            end,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    pub fn remaning_edges(&self) -> usize {
        self.end - self.begin
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NodeIndices<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.begin >= self.end {
            return None;
        }
        self.end = self.end.saturating_sub(1);
        Some(self.end)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.begin >= self.end {
            return None;
        }
        let res = self.begin;
        self.begin = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.begin, Some(self.end - self.begin))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NodeIndices<N, E, Ix>
{
}

#[derive(Clone)]
pub struct ParalelizableNodeIndices<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableNodeIndices<N, E, Ix>
{
    fn new(begin_offset: usize, end_offset: usize) -> Self {
        Self {
            stop_offset: end_offset,
            offset: Arc::new(AtomicUsize::new(begin_offset)),
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableNodeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        let curr = self.offset.fetch_add(1, Ordering::Relaxed);
        if curr >= self.stop_offset {
            return None;
        }
        Some(curr)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.stop_offset - self.offset.load(Ordering::Relaxed);
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "rayon")]
#[derive(Clone)]
pub struct ParNodeIndices<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    stop_offset: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParNodeIndices<N, E, Ix> {
    fn new(begin_offset: usize, end_offset: usize) -> Self {
        Self {
            stop_offset: end_offset,
            offset: begin_offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParNodeIndices<N, E, Ix>
{
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.offset..self.stop_offset
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
        (
            Self {
                stop_offset: mid,
                offset: self.offset,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
            Self {
                stop_offset: self.stop_offset,
                offset: mid,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParNodeIndices<N, E, Ix>
{
    type Item = usize;

    // `self` is also `IndexedParallelIterator`, so route the unindexed entry point through the
    // *indexed* `bridge` (reusing the existing, already-correct `Producer`/`with_producer`
    // machinery) rather than `bridge_unindexed` + a redundant, hand-rolled `UnindexedProducer`.
    // This matters, not just for symmetry: rayon's own `.collect()` --- via
    // `ParallelExtend for Vec<T>`'s `fast_collect` --- checks `opt_len()`, and if it's `Some`,
    // drives an internal `CollectConsumer` through *this exact* `drive_unindexed` entry point
    // with a consumer that panics if ever asked to split itself (`split_off_left`). Reusing a
    // hand-rolled `UnindexedProducer::split()` here would, for any input large enough to
    // actually split (> 4096 items), ask that consumer to split and immediately panic. Routing
    // through the indexed `bridge()` avoids ever calling `split_off_left` in the first place,
    // since indexed splitting always provides an explicit index instead.
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParNodeIndices<N, E, Ix>
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.stop_offset - self.offset
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

pub struct EdgeIndices<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    begin: usize,
    end: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> EdgeIndices<N, E, Ix> {
    fn new(begin: usize, end: usize) -> Self {
        Self {
            begin,
            end,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    pub fn remaning_edges(&self) -> usize {
        self.end - self.begin
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for EdgeIndices<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.begin >= self.end {
            return None;
        }
        self.end = self.end.saturating_sub(1);
        Some(self.end)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for EdgeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.begin >= self.end {
            return None;
        }
        let res = self.begin;
        self.begin = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.begin, Some(self.end - self.begin))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for EdgeIndices<N, E, Ix>
{
}

#[derive(Clone)]
pub struct ParalelizableEdgeIndices<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableEdgeIndices<N, E, Ix>
{
    fn new(begin_offset: usize, end_offset: usize) -> Self {
        Self {
            stop_offset: end_offset,
            offset: Arc::new(AtomicUsize::new(begin_offset)),
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableEdgeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        let curr = self.offset.fetch_add(1, Ordering::Relaxed);
        if curr >= self.stop_offset {
            return None;
        }
        Some(curr)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.stop_offset - self.offset.load(Ordering::Relaxed);
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "rayon")]
#[derive(Clone)]
pub struct ParEdgeIndices<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    stop_offset: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParEdgeIndices<N, E, Ix> {
    fn new(begin_offset: usize, end_offset: usize) -> Self {
        Self {
            stop_offset: end_offset,
            offset: begin_offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParEdgeIndices<N, E, Ix>
{
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.offset..self.stop_offset
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
        (
            Self {
                stop_offset: mid,
                offset: self.offset,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
            Self {
                stop_offset: self.stop_offset,
                offset: mid,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParEdgeIndices<N, E, Ix>
{
    type Item = usize;

    // see the identical comment on `ParNodeIndices::drive_unindexed`.
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParEdgeIndices<N, E, Ix>
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.stop_offset - self.offset
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

pub struct EdgesConnecting<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    EdgesConnecting<N, E, Ix>
{
    fn new(
        neigh_mmap: *const Ix,
        off_mmap: *const Ix,
        orig_node: usize,
        target_node: usize,
    ) -> Self {
        let begin_bound = unsafe { off_mmap.add(orig_node).read().index() };
        let end_bound = unsafe { off_mmap.add(orig_node + 1).read().index() };

        // neighbour lists are kept sorted (see `impl_sort::sort_edges`), so every occurrence of
        // `target_node` within this node's range forms a single contiguous run --- scan for its
        // start, count its length, then stop as soon as the run ends (no need to keep scanning
        // the rest of the range).
        let mut offset = begin_bound;
        let mut count = 0usize;
        let mut found = false;
        for i in begin_bound..end_bound {
            let is_target = unsafe { neigh_mmap.add(i).read().index() } == target_node;
            if is_target {
                if !found {
                    found = true;
                    offset = i;
                }
                count += 1;
            } else if found {
                break;
            }
        }

        Self {
            count,
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    pub fn remaning_edges(&self) -> usize {
        self.count
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for EdgesConnecting<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        Some(self.offset + self.count)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for EdgesConnecting<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let res = self.offset;
        self.offset += 1;
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for EdgesConnecting<N, E, Ix>
{
}

pub struct NodesWithDegree<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    off_ptr: *const Ix,
    count: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    NodesWithDegree<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            count: node_count - offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NodesWithDegree<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset + self.count;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg != self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodesWithDegree<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset;
            self.offset += 1;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg != self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.count))
    }
}

pub struct NodesWithDegreeLT<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    off_ptr: *const Ix,
    count: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    NodesWithDegreeLT<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            count: node_count - offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NodesWithDegreeLT<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset + self.count;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg >= self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodesWithDegreeLT<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset;
            self.offset += 1;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg >= self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.count))
    }
}

pub struct NodesWithDegreeGE<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    off_ptr: *const Ix,
    count: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    NodesWithDegreeGE<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            count: node_count - offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NodesWithDegreeGE<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset + self.count;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg < self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodesWithDegreeGE<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.count == 0 {
                return None;
            }
            self.count = self.count.saturating_sub(1);
            let candidate = self.offset;
            self.offset += 1;
            let deg = unsafe {
                self.off_ptr.add(candidate + 1).read().index()
                    - self.off_ptr.add(candidate).read().index()
            };
            if deg < self.target_deg {
                continue;
            }
            return Some(candidate);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.count))
    }
}

#[derive(Clone)]
pub struct ParalelizableNodesWithDegreeLT<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableNodesWithDegreeLT<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset: Arc::new(AtomicUsize::new(offset)),
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableNodesWithDegreeLT<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            let curr = self.offset.fetch_add(1, Ordering::Relaxed);
            if curr >= self.stop_offset {
                return None;
            }
            if unsafe {
                self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index()
            } >= self.target_deg
            {
                continue;
            }
            return Some(curr);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stop_offset - self.offset.load(Ordering::Relaxed), None)
    }
}

#[cfg(feature = "rayon")]
#[derive(Copy, Clone)]
pub struct ParNodesWithDegreeLT<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
{
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Send
    for ParNodesWithDegreeLT<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Sync
    for ParNodesWithDegreeLT<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegreeLT<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParNodesWithDegreeLT<N, E, Ix>
{
    type Item = usize;

    fn split(self) -> (Self, Option<Self>) {
        let len = self.stop_offset - self.offset;
        if len <= 4096 {
            (self, None) // small enough: stop splitting
        } else {
            let mid = self.offset + len / 2;
            (
                Self {
                    off_ptr: self.off_ptr,
                    stop_offset: mid,
                    offset: self.offset,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    off_ptr: self.off_ptr,
                    stop_offset: self.stop_offset,
                    offset: mid,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                }),
            )
        }
    }

    // Sequential fold of a chunk
    fn fold_with<F>(self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        for i in self.offset..self.stop_offset {
            let deg = unsafe {
                self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index()
            };
            if deg < self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParNodesWithDegreeLT<N, E, Ix>
{
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }

    // see the identical comment on `ParNodesWithDegree`'s `opt_len`.
    fn opt_len(&self) -> Option<usize> {
        None
    }
}

#[derive(Clone)]
pub struct ParalelizableNodesWithDegreeGE<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableNodesWithDegreeGE<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset: Arc::new(AtomicUsize::new(offset)),
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableNodesWithDegreeGE<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            let curr = self.offset.fetch_add(1, Ordering::Relaxed);
            if curr >= self.stop_offset {
                return None;
            }
            if unsafe {
                self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index()
            } < self.target_deg
            {
                continue;
            }
            return Some(curr);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stop_offset - self.offset.load(Ordering::Relaxed), None)
    }
}

#[cfg(feature = "rayon")]
#[derive(Copy, Clone)]
pub struct ParNodesWithDegreeGE<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
{
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Send
    for ParNodesWithDegreeGE<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Sync
    for ParNodesWithDegreeGE<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegreeGE<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParNodesWithDegreeGE<N, E, Ix>
{
    type Item = usize;

    fn split(self) -> (Self, Option<Self>) {
        let len = self.stop_offset - self.offset;
        if len <= 4096 {
            (self, None) // small enough: stop splitting
        } else {
            let mid = self.offset + len / 2;
            (
                Self {
                    off_ptr: self.off_ptr,
                    stop_offset: mid,
                    offset: self.offset,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    off_ptr: self.off_ptr,
                    stop_offset: self.stop_offset,
                    offset: mid,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                }),
            )
        }
    }

    // Sequential fold of a chunk
    fn fold_with<F>(self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        for i in self.offset..self.stop_offset {
            let deg = unsafe {
                self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index()
            };
            if deg >= self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParNodesWithDegreeGE<N, E, Ix>
{
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }

    // see the identical comment on `ParNodesWithDegree`'s `opt_len`.
    fn opt_len(&self) -> Option<usize> {
        None
    }
}

#[derive(Clone)]
pub struct ParalelizableNodesWithDegree<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableNodesWithDegree<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset: Arc::new(AtomicUsize::new(offset)),
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableNodesWithDegree<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            let curr = self.offset.fetch_add(1, Ordering::Relaxed);
            if curr >= self.stop_offset {
                return None;
            }
            if unsafe {
                self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index()
            } != self.target_deg
            {
                continue;
            }
            return Some(curr);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stop_offset - self.offset.load(Ordering::Relaxed), None)
    }
}

#[cfg(feature = "rayon")]
#[derive(Copy, Clone)]
pub struct ParNodesWithDegree<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    off_ptr: *const Ix,
    stop_offset: usize,
    offset: usize,
    target_deg: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Send
    for ParNodesWithDegree<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Sync
    for ParNodesWithDegree<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegree<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = node_count;
        let offset = unsafe { off_mmap.read().index() };

        Self {
            off_ptr: off_mmap,
            stop_offset,
            offset,
            target_deg,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParNodesWithDegree<N, E, Ix>
{
    type Item = usize;

    fn split(self) -> (Self, Option<Self>) {
        let len = self.stop_offset - self.offset;
        if len <= 4096 {
            (self, None) // small enough: stop splitting
        } else {
            let mid = self.offset + len / 2;
            (
                Self {
                    off_ptr: self.off_ptr,
                    stop_offset: mid,
                    offset: self.offset,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    off_ptr: self.off_ptr,
                    stop_offset: self.stop_offset,
                    offset: mid,
                    target_deg: self.target_deg,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                }),
            )
        }
    }

    // Sequential fold of a chunk
    fn fold_with<F>(self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        for i in self.offset..self.stop_offset {
            let deg = unsafe {
                self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index()
            };
            if deg == self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParNodesWithDegree<N, E, Ix>
{
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }

    // `self.stop_offset - self.offset` is the number of remaining *candidate* nodes to scan, not
    // the number this filtering iterator will actually yield --- which can't be known without
    // scanning ahead (same reason `std::iter::Filter` reports no exact/optional length either).
    // Returning `None` here is the only honest answer.
    fn opt_len(&self) -> Option<usize> {
        None
    }
}

pub struct Neighbors<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    neigh_ptr: *const Ix,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Neighbors<N, E, Ix> {
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let off_ptr = unsafe { off_mmap.add(node_id) };
        let offset = unsafe { off_ptr.read_unaligned().index() };

        Self {
            neigh_ptr: unsafe { neigh_mmap.add(offset) },
            count: unsafe { off_ptr.add(1).read_unaligned().index() - offset },
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
    #[cfg(feature = "rayon")]
    fn new_by_offset(neigh_mmap: *const Ix, offset: usize, stop_offset: usize) -> Self {
        Self {
            neigh_ptr: unsafe { neigh_mmap.add(offset) },
            count: stop_offset - offset,
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for Neighbors<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        unsafe { Some(self.neigh_ptr.add(self.count).read_unaligned().index()) }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for Neighbors<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: usize;
        self.neigh_ptr = unsafe {
            next = self.neigh_ptr.read().index();
            self.neigh_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

#[derive(Clone)]
pub struct ParalelizableNeighbors<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    neigh_ptr: *const Ix,
    stop_offset: usize,
    offset: Arc<AtomicUsize>,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParalelizableNeighbors<N, E, Ix>
{
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let offset = unsafe { off_mmap.add(node_id).read().index() };
        let stop_offset = unsafe { off_mmap.add(node_id + 1).read().index() };

        Self {
            neigh_ptr: neigh_mmap,
            stop_offset,
            offset: Arc::new(AtomicUsize::new(offset)),
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for ParalelizableNeighbors<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        let curr = self.offset.fetch_add(1, Ordering::Relaxed);
        if curr >= self.stop_offset {
            return None;
        }
        unsafe { Some(self.neigh_ptr.add(curr).read().index()) }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.stop_offset - self.offset.load(Ordering::Relaxed);
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "rayon")]
#[derive(Copy, Clone)]
pub struct ParNeighbors<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    neigh_ptr: *const Ix,
    stop_offset: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Send
    for ParNeighbors<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
unsafe impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Sync
    for ParNeighbors<N, E, Ix>
{
}

// Not actually rayon-specific (unlike the structs around it): `Neighbors` is the crate's base
// sequential neighbour iterator, used unconditionally. `size_hint()` already reports exact
// bounds (see its `Iterator` impl above), so this marker impl must not be feature-gated, or
// `cargo build --tests`/`cargo test` fails to compile on default features (no `rayon`).
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for Neighbors<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParNeighbors<N, E, Ix> {
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let offset = unsafe { off_mmap.add(node_id).read().index() };
        let stop_offset = unsafe { off_mmap.add(node_id + 1).read().index() };

        Self {
            neigh_ptr: neigh_mmap,
            stop_offset,
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParNeighbors<N, E, Ix>
{
    type Item = usize;
    type IntoIter = Neighbors<N, E, Ix>;

    fn into_iter(self) -> Self::IntoIter {
        Neighbors::new_by_offset(self.neigh_ptr, self.offset, self.stop_offset)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
        (
            Self {
                neigh_ptr: self.neigh_ptr,
                stop_offset: mid,
                offset: self.offset,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
            Self {
                neigh_ptr: self.neigh_ptr,
                stop_offset: self.stop_offset,
                offset: mid,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ParallelIterator
    for ParNeighbors<N, E, Ix>
{
    type Item = usize;

    // see the identical comment on `ParNodeIndices::drive_unindexed`.
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParNeighbors<N, E, Ix>
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.stop_offset - self.offset
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

pub struct NeighborsWithOffset<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
{
    neigh_ptr: *const Ix,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    NeighborsWithOffset<N, E, Ix>
{
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let off_ptr = unsafe { off_mmap.add(node_id) };
        let offset = unsafe { off_ptr.read_unaligned().index() };

        Self {
            neigh_ptr: unsafe { neigh_mmap.add(offset) },
            count: unsafe { off_ptr.add(1).read_unaligned().index() - offset },
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NeighborsWithOffset<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        unsafe {
            Some((
                self.offset + self.count,
                self.neigh_ptr.add(self.count).read_unaligned().index(),
            ))
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NeighborsWithOffset<N, E, Ix>
{
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: (usize, usize);
        self.neigh_ptr = unsafe {
            next = (self.offset, self.neigh_ptr.read_unaligned().index());
            self.neigh_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NeighborsWithOffset<N, E, Ix>
{
}

pub struct WalkNeighbors<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    neigh_ptr: *const Ix,
    orig_neigh_ptr: *const Ix,
    orig_off_ptr: *const Ix,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> WalkNeighbors<N, E, Ix> {
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let orig_neigh_ptr = neigh_mmap;
        let orig_off_ptr = off_mmap;
        let off_ptr = unsafe { off_mmap.add(node_id) };
        let offset = unsafe { off_ptr.read_unaligned().index() };

        Self {
            neigh_ptr: unsafe { neigh_mmap.add(offset) },
            orig_neigh_ptr,
            orig_off_ptr,
            count: unsafe { off_ptr.add(1).read_unaligned().index() - offset },
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    pub fn into_neighbour(self) -> Self {
        Self::new(self.orig_neigh_ptr, self.orig_off_ptr, unsafe {
            self.neigh_ptr.read_unaligned().index()
        })
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for WalkNeighbors<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: usize;
        unsafe {
            next = self.neigh_ptr.add(self.count).read_unaligned().index();
        };
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for WalkNeighbors<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        self.offset += 1;
        let next: usize;
        self.neigh_ptr = unsafe {
            next = self.neigh_ptr.read_unaligned().index();
            self.neigh_ptr.add(1)
        };
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for WalkNeighbors<N, E, Ix>
{
}

pub struct WalkNeighborsWithOffset<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
> {
    neigh_ptr: *const Ix,
    orig_neigh_ptr: *const Ix,
    orig_off_ptr: *const Ix,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    WalkNeighborsWithOffset<N, E, Ix>
{
    fn new(neigh_mmap: *const Ix, off_mmap: *const Ix, node_id: usize) -> Self {
        let orig_neigh_ptr = neigh_mmap;
        let orig_off_ptr = off_mmap;
        let off_ptr = unsafe { off_mmap.add(node_id) };
        let offset = unsafe { off_ptr.read_unaligned().index() };

        Self {
            neigh_ptr: unsafe { neigh_mmap.add(offset) },
            orig_neigh_ptr,
            orig_off_ptr,
            count: unsafe { off_ptr.add(1).read_unaligned().index() - offset },
            offset,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    pub fn into_neighbour(self) -> Self {
        Self::new(self.orig_neigh_ptr, self.orig_off_ptr, unsafe {
            self.neigh_ptr.read_unaligned().index()
        })
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for WalkNeighborsWithOffset<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: (usize, usize);
        unsafe {
            next = (
                self.offset + self.count,
                self.neigh_ptr.add(self.count).read_unaligned().index(),
            );
        };
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for WalkNeighborsWithOffset<N, E, Ix>
{
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count = self.count.saturating_sub(1);
        let next: (usize, usize);
        self.neigh_ptr = unsafe {
            next = (self.offset, self.neigh_ptr.read_unaligned().index());
            self.neigh_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for WalkNeighborsWithOffset<N, E, Ix>
{
}

/// Iterator over every node's associated label/weight (`N`), in node-id order.
///
/// Yields `N` by value (`N: Copy`). For a zero-sized `N` (e.g. the default [`VoidLabel`], used
/// whenever a graph carries no node data), no memory is ever read --- every item is simply
/// `N::new(0)`, which is the type's only possible value --- since indexing a zero-sized element
/// past position 0 in the backing mmap is not something `SharedSlice`'s generic ZST handling
/// supports (see `SharedSlice::from_file`).
pub struct NodeLabels<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    ptr: *const N,
    begin: usize,
    end: usize,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> NodeLabels<N, E, Ix> {
    fn new(ptr: *const N, node_count: usize) -> Self {
        Self {
            ptr,
            begin: 0,
            end: node_count,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    fn read(ptr: *const N, idx: usize) -> N {
        if std::mem::size_of::<N>() == 0 {
            N::new(0)
        } else {
            unsafe { *ptr.add(idx) }
        }
    }

    pub fn remaining_labels(&self) -> usize {
        self.end - self.begin
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NodeLabels<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<N> {
        if self.begin >= self.end {
            return None;
        }
        self.end = self.end.saturating_sub(1);
        Some(Self::read(self.ptr, self.end))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodeLabels<N, E, Ix>
{
    type Item = N;

    #[inline(always)]
    fn next(&mut self) -> Option<N> {
        if self.begin >= self.end {
            return None;
        }
        let res = Self::read(self.ptr, self.begin);
        self.begin = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.begin;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NodeLabels<N, E, Ix>
{
}

/// Iterator over every edge's associated label/weight (`E`), in flat edge-array order (i.e. the
/// same order as [`GraphMemoryMap::iter_edge_indices`]/[`GraphMemoryMap::neighbours`]).
///
/// See [`NodeLabels`] for the zero-sized-`E` handling rationale.
pub struct EdgeLabels<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    ptr: *const E,
    begin: usize,
    end: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom3: std::marker::PhantomData<Ix>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> EdgeLabels<N, E, Ix> {
    fn new(ptr: *const E, edge_count: usize) -> Self {
        Self {
            ptr,
            begin: 0,
            end: edge_count,
            _phantom1: std::marker::PhantomData::<N>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    fn read(ptr: *const E, idx: usize) -> E {
        if std::mem::size_of::<E>() == 0 {
            E::new(0)
        } else {
            unsafe { *ptr.add(idx) }
        }
    }

    pub fn remaining_labels(&self) -> usize {
        self.end - self.begin
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for EdgeLabels<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<E> {
        if self.begin >= self.end {
            return None;
        }
        self.end = self.end.saturating_sub(1);
        Some(Self::read(self.ptr, self.end))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for EdgeLabels<N, E, Ix>
{
    type Item = E;

    #[inline(always)]
    fn next(&mut self) -> Option<E> {
        if self.begin >= self.end {
            return None;
        }
        let res = Self::read(self.ptr, self.begin);
        self.begin = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.begin;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for EdgeLabels<N, E, Ix>
{
}

/// Iterator over a single node's neighbours, each paired with its corresponding edge's
/// label/weight. See [`GraphMemoryMap::weighted_neighbours`].
pub struct WeightedNeighbours<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    neigh_ptr: *const Ix,
    label_ptr: *const E,
    begin: usize,
    end: usize,
    _phantom1: std::marker::PhantomData<N>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    WeightedNeighbours<N, E, Ix>
{
    fn new(neigh_ptr: *const Ix, label_ptr: *const E, begin: usize, end: usize) -> Self {
        Self {
            neigh_ptr,
            label_ptr,
            begin,
            end,
            _phantom1: std::marker::PhantomData::<N>,
        }
    }

    #[inline(always)]
    fn read(neigh_ptr: *const Ix, label_ptr: *const E, pos: usize) -> (usize, E) {
        let v = unsafe { (*neigh_ptr.add(pos)).index() };
        let w = EdgeLabels::<N, E, Ix>::read(label_ptr, pos);
        (v, w)
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.end - self.begin
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for WeightedNeighbours<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<(usize, E)> {
        if self.begin >= self.end {
            return None;
        }
        self.end = self.end.saturating_sub(1);
        Some(Self::read(self.neigh_ptr, self.label_ptr, self.end))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for WeightedNeighbours<N, E, Ix>
{
    type Item = (usize, E);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, E)> {
        if self.begin >= self.end {
            return None;
        }
        let res = Self::read(self.neigh_ptr, self.label_ptr, self.begin);
        self.begin = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.begin;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for WeightedNeighbours<N, E, Ix>
{
}

/// Iterator over every edge in the graph, as `(source, target, weight)` triples, in flat
/// edge-array order. See [`GraphMemoryMap::edges`].
///
/// Unlike this file's other edge/neighbour iterators, this one walks *every* node's range in a
/// single pass (merging across node boundaries, transparently skipping isolated/degree-0 nodes),
/// so it intentionally does not implement `DoubleEndedIterator` --- reverse iteration would need
/// a second, independently-converging node cursor, which isn't worth the added complexity for
/// this struct's use case (bulk, forward-only edge enumeration).
pub struct Edges<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    off_ptr: *const Ix,
    neigh_ptr: *const Ix,
    label_ptr: *const E,
    node: usize,
    node_count: usize,
    pos: usize,
    end: usize,
    _phantom1: std::marker::PhantomData<N>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Edges<N, E, Ix> {
    fn new(
        off_ptr: *const Ix,
        neigh_ptr: *const Ix,
        label_ptr: *const E,
        node_count: usize,
        edge_count: usize,
    ) -> Self {
        Self {
            off_ptr,
            neigh_ptr,
            label_ptr,
            node: 0,
            node_count,
            pos: 0,
            end: edge_count,
            _phantom1: std::marker::PhantomData::<N>,
        }
    }

    pub fn remaining_edges(&self) -> usize {
        self.end - self.pos
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for Edges<N, E, Ix>
{
    type Item = (usize, usize, E);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize, E)> {
        if self.pos >= self.end {
            return None;
        }
        // advance past every node whose range ends at or before `pos` --- isolated (degree-0)
        // nodes have a zero-width range and are skipped over automatically, landing on the
        // correct source node for `pos` either way.
        while self.node + 1 < self.node_count
            && unsafe { (*self.off_ptr.add(self.node + 1)).index() } <= self.pos
        {
            self.node = self.node.saturating_add(1);
        }
        let u = self.node;
        let v = unsafe { (*self.neigh_ptr.add(self.pos)).index() };
        let w = EdgeLabels::<N, E, Ix>::read(self.label_ptr, self.pos);
        self.pos = self.pos.saturating_add(1);
        Some((u, v, w))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.pos;
        (remaining, Some(remaining))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for Edges<N, E, Ix>
{
}

/// Globally unique id for a single [`Dfs`]/[`Bfs`] traversal instance, used to namespace its
/// memory-mapped scratch files --- without this, two simultaneously-alive traversals over the
/// same graph (e.g. two threads, or a connected-components loop holding more than one at once)
/// would derive the exact same cache filename and clobber each other's backing mmap.
fn next_traversal_instance_id() -> usize {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Growable, memory-mapped `usize` buffer backing [`Dfs`]'s stack and [`Bfs`]'s queue --- a
/// heap `Vec` growing to O(|V|) is not acceptable for traversals over the 100M-to-billions-of-
/// edges graphs this crate targets, so, consistent with the rest of its disk-backed procedural-
/// memory approach, this stays memory-mapped throughout.
///
/// Starts at 5% of `max_capacity` (clamped to at least 1) and doubles whenever full, capped at
/// `max_capacity` --- the hard upper bound on how many distinct node ids can ever be pushed,
/// since both `Dfs`/`Bfs` only ever push a node once, immediately after marking it visited.
/// Each growth re-mmaps into a freshly, uniquely named cache file (via `target`/`instance_id`/
/// the new generation number) and removes the previous one; the final file is removed on drop.
struct GrowableIdBuffer<
    'a,
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
    Dir: Directedness = Undirected,
> {
    graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
    target: CacheFile,
    instance_id: usize,
    generation: usize,
    mem: AbstractedProceduralMemoryMut<Ix>,
    file_name: String,
    cap: usize,
    len: usize,
    max_capacity: usize,
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    GrowableIdBuffer<'a, N, E, Ix, Dir>
{
    fn new(
        graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
        target: CacheFile,
        instance_id: usize,
        max_capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cap = (max_capacity / 20).clamp(1, max_capacity.max(1));
        // reserve a small, fixed range of sequence numbers per instance (1 for the initial
        // buffer, +1 per growth generation) so concurrently-alive instances never collide;
        // see `grow`'s doc comment for why 16 is comfortably more headroom than ever needed.
        let file_name = graph.build_cache_filename(target.clone(), Some(instance_id * 16 + 1))?;
        let mem = SharedSliceMut::<Ix>::abst_mem_mut(&file_name, cap, true)?;
        Ok(Self {
            graph,
            target,
            instance_id,
            generation: 0,
            mem,
            file_name,
            cap,
            len: 0,
            max_capacity,
        })
    }

    /// 5% -> 100% by doubling is at most ~5 generations regardless of graph size (the ratio is
    /// fixed), so reserving 16 sequence numbers per instance (1 initial + 15 possible growths)
    /// is a comfortable, permanent upper bound.
    fn grow(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let new_cap = self
            .cap
            .saturating_mul(2)
            .min(self.max_capacity)
            .max(self.cap + 1);
        self.generation += 1;
        let new_file_name = self.graph.build_cache_filename(
            self.target.clone(),
            Some(self.instance_id * 16 + 1 + self.generation),
        )?;
        let mut new_mem = SharedSliceMut::<Ix>::abst_mem_mut(&new_file_name, new_cap, true)?;
        new_mem.as_mut_slice()[..self.len].copy_from_slice(&self.mem.as_slice()[..self.len]);
        let old_file_name = std::mem::replace(&mut self.file_name, new_file_name);
        self.mem = new_mem;
        self.cap = new_cap;
        let _ = std::fs::remove_file(&old_file_name);
        Ok(())
    }

    #[inline]
    fn push(&mut self, v: usize) -> Result<(), Box<dyn std::error::Error>> {
        if self.len == self.cap {
            self.grow()?;
        }
        *self.mem.get_mut(self.len) = Ix::new(v);
        self.len += 1;
        Ok(())
    }

    #[inline]
    fn pop(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.mem.get(self.len).index())
    }

    #[inline]
    fn get(&self, idx: usize) -> usize {
        self.mem.get(idx).index()
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Drop for GrowableIdBuffer<'a, N, E, Ix, Dir>
{
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file_name);
    }
}

/// Lazy, pull-based depth-first traversal iterator. See [`GraphMemoryMap::dfs`].
///
/// Yields `source` itself first, then every other node reachable from it, each exactly once, in
/// depth-first discovery order. Its "visited" marker and explicit stack are both memory-mapped
/// (see `GrowableIdBuffer`), not heap-allocated, so traversing a graph far larger than
/// available RAM is sound. Unlike [`DFSVisit`](crate::trails::dfs_visit::DFSVisit), it exposes
/// no hooks beyond the node sequence itself --- for edge classification (tree/back/forward-cross
/// edges), discovery/finish timestamps, or parent tracking, use `DFSVisit`'s visitor API
/// instead.
///
/// Does *not* implement `ExactSizeIterator`: like `NodesWithDegree`/`LT`/`GE`, the number of
/// reachable nodes can't be known ahead of actually traversing the graph. It does implement
/// `FusedIterator`: once the internal stack empties, it stays empty, so `next()` keeps returning
/// `None`.
pub struct Dfs<
    'a,
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
    Dir: Directedness = Undirected,
> {
    graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
    stack: GrowableIdBuffer<'a, N, E, Ix, Dir>,
    visited: AbstractedProceduralMemoryMut<bool>,
    visited_file: String,
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Dfs<'a, N, E, Ix, Dir>
{
    /// `source` must already be known to be `< graph.size()` (checked once by
    /// [`GraphMemoryMap::dfs`]); every `v` reachable through it from there on is itself a valid
    /// node id by the CSR neighbour-list invariant, so the `.expect()` in `next()` can't fail.
    fn new(
        graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = graph.size();
        let instance_id = next_traversal_instance_id();
        let visited_file =
            graph.build_cache_filename(CacheFile::DfsVisitor, Some(instance_id * 16))?;
        let mut visited = SharedSliceMut::<bool>::abst_mem_mut(&visited_file, node_count, true)?;
        let mut stack =
            GrowableIdBuffer::new(graph, CacheFile::DfsVisitor, instance_id, node_count)?;
        *visited.get_mut(source) = true;
        stack.push(source)?;
        Ok(Self {
            graph,
            stack,
            visited,
            visited_file,
        })
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Iterator for Dfs<'a, N, E, Ix, Dir>
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let u = self.stack.pop()?;
        for v in self
            .graph
            .neighbours(u)
            .expect("u is a valid, in-range node id")
        {
            if !*self.visited.get(v) {
                *self.visited.get_mut(v) = true;
                self.stack
                    .push(v)
                    .expect("failed to grow Dfs's memory-mapped stack (disk I/O error)");
            }
        }
        Some(u)
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Drop for Dfs<'a, N, E, Ix, Dir>
{
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.visited_file);
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    std::iter::FusedIterator for Dfs<'a, N, E, Ix, Dir>
{
}

/// Lazy, pull-based breadth-first traversal iterator. See [`GraphMemoryMap::bfs`].
///
/// Yields `source` itself first, then every other node reachable from it, each exactly once, in
/// breadth-first discovery order. See [`Dfs`]'s documentation for the same rationale
/// (memory-mapped bookkeeping, no `ExactSizeIterator`, `BFSVisit` as the richer alternative when
/// edge classification/distances/parent-tracking are needed). Its "queue" is append-only ---
/// dequeuing just advances `read_pos` rather than physically removing the front element, so the
/// same growable buffer design as `Dfs`'s stack applies directly (capacity bounded by the graph's
/// node count either way).
pub struct Bfs<
    'a,
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
    Dir: Directedness = Undirected,
> {
    graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
    queue: GrowableIdBuffer<'a, N, E, Ix, Dir>,
    read_pos: usize,
    visited: AbstractedProceduralMemoryMut<bool>,
    visited_file: String,
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Bfs<'a, N, E, Ix, Dir>
{
    fn new(
        graph: &'a GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = graph.size();
        let instance_id = next_traversal_instance_id();
        let visited_file =
            graph.build_cache_filename(CacheFile::BfsVisitor, Some(instance_id * 16))?;
        let mut visited = SharedSliceMut::<bool>::abst_mem_mut(&visited_file, node_count, true)?;
        let mut queue =
            GrowableIdBuffer::new(graph, CacheFile::BfsVisitor, instance_id, node_count)?;
        *visited.get_mut(source) = true;
        queue.push(source)?;
        Ok(Self {
            graph,
            queue,
            read_pos: 0,
            visited,
            visited_file,
        })
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Iterator for Bfs<'a, N, E, Ix, Dir>
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.read_pos >= self.queue.len() {
            return None;
        }
        let u = self.queue.get(self.read_pos);
        self.read_pos += 1;
        for v in self
            .graph
            .neighbours(u)
            .expect("u is a valid, in-range node id")
        {
            if !*self.visited.get(v) {
                *self.visited.get_mut(v) = true;
                self.queue
                    .push(v)
                    .expect("failed to grow Bfs's memory-mapped queue (disk I/O error)");
            }
        }
        Some(u)
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    Drop for Bfs<'a, N, E, Ix, Dir>
{
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.visited_file);
    }
}

impl<'a, N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: Directedness>
    std::iter::FusedIterator for Bfs<'a, N, E, Ix, Dir>
{
}

// `NodeLabelsMut`/`EdgeLabelsMut` are intentionally not implemented: `GraphMemoryMap` is a
// readonly, memory-mapped view over a [`GraphCache`] (see `init_from_cache`'s `readonly` check)
// --- in-place mutation of label data isn't supported by the current architecture. Label values
// may still be changed by re-parsing/re-building the cache.
pub struct NodeLabelsMut {}
pub struct EdgeLabelsMut {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::unique_test_dir;

    /// Graph with a deliberately varied degree distribution, so that the degree-filtered
    /// (`NodesWithDegree`/`LT`/`GE`), index-range (`NodeIndices`/`EdgeIndices`), and
    /// neighbour-walking (`Neighbors`/`NeighborsWithOffset`/`WalkNeighbors*`) iterator families
    /// all have non-trivial, distinguishable output to check against:
    ///
    /// node 0 -> {1, 2, 3}  (degree 3)
    /// node 1 -> {2}        (degree 1)
    /// node 2 -> {}         (degree 0, isolated)
    /// node 3 -> {0}        (degree 1)
    /// node 4 -> {}         (degree 0, isolated)
    /// node 5 -> {0, 1}     (degree 2)
    fn build_degree_test_graph()
    -> crate::graph::test_utils::TestGraph<VoidLabel, VoidLabel, usize, crate::graph::Directed>
    {
        let dir = unique_test_dir("iter_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("degrees.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern general\n6 6 7\n1 2\n1 3\n1 4\n2 3\n4 1\n6 1\n6 2\n",
        )
        .expect("error writing mtx fixture");

        let cache = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph =
            crate::graph::DiGraph::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        crate::graph::test_utils::TestGraph::new(graph, None)
    }

    #[test]
    fn node_indices_cover_every_node_forward_and_backward() {
        let graph = build_degree_test_graph();

        // historical bug: an inverted bounds check made `next()` return `None` immediately for
        // any non-empty range, so this would previously collect to an empty `Vec`.
        let forward: Vec<usize> = graph.iter_node_indices().collect();
        assert_eq!(forward, (0..graph.size()).collect::<Vec<_>>());

        // historical bug: `next_back()` never actually shrank the range, so `.rev()` would
        // return the same last element forever instead of walking backward.
        let backward: Vec<usize> = graph.iter_node_indices().rev().collect();
        assert_eq!(backward, (0..graph.size()).rev().collect::<Vec<_>>());

        assert_eq!(graph.iter_node_indices().count(), graph.size());
    }

    #[test]
    fn into_iterator_for_ref_graph_matches_iter_node_indices() {
        let graph = build_degree_test_graph();

        let via_into_iter: Vec<usize> = (&graph).into_iter().collect();
        let via_method: Vec<usize> = graph.iter_node_indices().collect();
        assert_eq!(via_into_iter, via_method);

        // the idiomatic `for node in &graph` form itself (`graph` is a test wrapper, `*graph` the
        // `GraphMemoryMap`).
        let mut via_for_loop = Vec::new();
        for node in &*graph {
            via_for_loop.push(node);
        }
        assert_eq!(via_for_loop, via_method);
    }

    /// Independent, textbook-style reference BFS (no shared code with [`Bfs`]/[`Dfs`]), used as
    /// ground truth: returns (reachable set in distance order, distance-from-source for each
    /// node).
    fn reference_bfs(
        graph: &crate::graph::DiGraph<VoidLabel, VoidLabel, usize>,
        source: usize,
    ) -> Vec<usize> {
        let mut dist = vec![usize::MAX; graph.size()];
        let mut order = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        dist[source] = 0;
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for v in graph.neighbours(u).unwrap() {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        order
    }

    #[test]
    fn dfs_visits_exactly_the_reachable_set_once_each_source_first() {
        let graph = build_degree_test_graph();

        for source in 0..graph.size() {
            let mut expected_reachable: Vec<usize> = reference_bfs(&graph, source);
            expected_reachable.sort_unstable();

            let via_dfs: Vec<usize> = graph.dfs(source).unwrap().collect();
            assert_eq!(
                via_dfs[0], source,
                "Dfs must yield source first for source={source}"
            );

            let mut sorted_dfs = via_dfs.clone();
            sorted_dfs.sort_unstable();
            assert_eq!(
                sorted_dfs, expected_reachable,
                "Dfs's reachable set mismatch for source={source}"
            );

            // no duplicates: a sorted, deduped copy must have the same length.
            let mut deduped = sorted_dfs.clone();
            deduped.dedup();
            assert_eq!(
                deduped.len(),
                sorted_dfs.len(),
                "Dfs yielded a duplicate for source={source}"
            );
        }
    }

    #[test]
    fn bfs_visits_exactly_the_reachable_set_once_each_in_nondecreasing_distance_order() {
        let graph = build_degree_test_graph();

        for source in 0..graph.size() {
            let expected_order = reference_bfs(&graph, source);
            let mut expected_sorted = expected_order.clone();
            expected_sorted.sort_unstable();

            let via_bfs: Vec<usize> = graph.bfs(source).unwrap().collect();
            assert_eq!(
                via_bfs[0], source,
                "Bfs must yield source first for source={source}"
            );

            let mut sorted_bfs = via_bfs.clone();
            sorted_bfs.sort_unstable();
            assert_eq!(
                sorted_bfs, expected_sorted,
                "Bfs's reachable set mismatch for source={source}"
            );

            let mut deduped = sorted_bfs.clone();
            deduped.dedup();
            assert_eq!(
                deduped.len(),
                sorted_bfs.len(),
                "Bfs yielded a duplicate for source={source}"
            );

            // stronger order check: recompute each node's true distance from `source`
            // independently, then assert the yielded sequence is itself non-decreasing in that
            // distance --- a property specific to BFS (not required/true for DFS).
            let mut dist = vec![usize::MAX; graph.size()];
            let mut queue = std::collections::VecDeque::new();
            dist[source] = 0;
            queue.push_back(source);
            while let Some(u) = queue.pop_front() {
                for v in graph.neighbours(u).unwrap() {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            for w in via_bfs.windows(2) {
                assert!(
                    dist[w[0]] <= dist[w[1]],
                    "Bfs order not non-decreasing in distance for source={source}: {} (d={}) before {} (d={})",
                    w[0],
                    dist[w[0]],
                    w[1],
                    dist[w[1]]
                );
            }
        }
    }

    #[test]
    fn dfs_and_bfs_reject_out_of_range_source() {
        let graph = build_degree_test_graph();
        assert!(graph.dfs(graph.size()).is_err());
        assert!(graph.bfs(graph.size()).is_err());
    }

    #[test]
    fn dfs_and_bfs_are_fused() {
        let graph = build_degree_test_graph();

        let mut dfs = graph.dfs(2).unwrap(); // node 2 is isolated: reachable set = {2}
        assert_eq!(dfs.next(), Some(2));
        assert_eq!(dfs.next(), None);
        assert_eq!(
            dfs.next(),
            None,
            "Dfs must keep returning None once exhausted"
        );

        let mut bfs = graph.bfs(2).unwrap();
        assert_eq!(bfs.next(), Some(2));
        assert_eq!(bfs.next(), None);
        assert_eq!(
            bfs.next(),
            None,
            "Bfs must keep returning None once exhausted"
        );
    }

    /// Forces `GrowableIdBuffer` to actually grow multiple times (starting at 5% of a
    /// 10,001-node graph's node count --- i.e. ~500 --- doubling repeatedly to cover node 0's
    /// 6000-degree hub), verifying the copy-on-grow logic doesn't corrupt or drop any entries.
    #[cfg(feature = "rayon")]
    #[test]
    fn dfs_and_bfs_remain_correct_when_the_growable_buffer_must_grow() {
        let graph = build_large_test_graph();
        assert!(
            graph.node_degree(0) > graph.size() / 20,
            "hub degree must exceed the initial 5% capacity to force real growth"
        );

        let expected = reference_bfs(&graph, 0);
        let mut expected_sorted = expected.clone();
        expected_sorted.sort_unstable();

        let mut via_dfs: Vec<usize> = graph.dfs(0).unwrap().collect();
        via_dfs.sort_unstable();
        assert_eq!(
            via_dfs, expected_sorted,
            "Dfs must remain correct across buffer growth"
        );

        let mut via_bfs: Vec<usize> = graph.bfs(0).unwrap().collect();
        via_bfs.sort_unstable();
        assert_eq!(
            via_bfs, expected_sorted,
            "Bfs must remain correct across buffer growth"
        );
    }

    /// Two simultaneously-alive traversals over the same graph must not clobber each other's
    /// memory-mapped scratch files (the whole reason for `next_traversal_instance_id`).
    #[test]
    fn concurrently_alive_dfs_and_bfs_instances_do_not_interfere() {
        let graph = build_degree_test_graph();

        let mut dfs_a = graph.dfs(0).unwrap();
        let mut dfs_b = graph.dfs(5).unwrap();
        let mut bfs_a = graph.bfs(0).unwrap();
        let mut bfs_b = graph.bfs(5).unwrap();

        // interleave stepping all four, as a concurrent/interleaved-use smoke test.
        let mut a1 = Vec::new();
        let mut b1 = Vec::new();
        let mut a2 = Vec::new();
        let mut b2 = Vec::new();
        loop {
            let (n1, n2, n3, n4) = (dfs_a.next(), dfs_b.next(), bfs_a.next(), bfs_b.next());
            let done = n1.is_none() && n2.is_none() && n3.is_none() && n4.is_none();
            a1.extend(n1);
            b1.extend(n2);
            a2.extend(n3);
            b2.extend(n4);
            if done {
                break;
            }
        }

        a1.sort_unstable();
        a2.sort_unstable();
        let mut expected_from_0 = reference_bfs(&graph, 0);
        expected_from_0.sort_unstable();
        assert_eq!(
            a1, expected_from_0,
            "interleaved Dfs from 0 corrupted by a concurrent instance"
        );
        assert_eq!(
            a2, expected_from_0,
            "interleaved Bfs from 0 corrupted by a concurrent instance"
        );

        b1.sort_unstable();
        b2.sort_unstable();
        let mut expected_from_5 = reference_bfs(&graph, 5);
        expected_from_5.sort_unstable();
        assert_eq!(
            b1, expected_from_5,
            "interleaved Dfs from 5 corrupted by a concurrent instance"
        );
        assert_eq!(
            b2, expected_from_5,
            "interleaved Bfs from 5 corrupted by a concurrent instance"
        );
    }

    /// `Dfs`/`Bfs`'s memory-mapped scratch files must not leak once the iterator is dropped.
    /// Filters by this test's own graph's cache id (not just the `dfsvisitor`/`bfsvisitor`
    /// suffix) so this remains robust when run in parallel with other tests/instances that share
    /// the same on-disk cache directory.
    #[test]
    fn dfs_and_bfs_clean_up_their_scratch_files_on_drop() {
        let graph = build_degree_test_graph();
        let cache_id = graph.graph_id().unwrap();

        let dfs = graph.dfs(0).unwrap();
        let bfs = graph.bfs(0).unwrap();
        // both still alive here; their scratch files are expected to exist at this point.
        assert!(
            !traversal_scratch_files(&cache_id).is_empty(),
            "expected at least one dfsvisitor/bfsvisitor scratch file while still alive"
        );
        drop(dfs);
        drop(bfs);

        assert!(
            traversal_scratch_files(&cache_id).is_empty(),
            "Dfs/Bfs left scratch files behind after being dropped: {:?}",
            traversal_scratch_files(&cache_id)
        );
    }

    /// Lists every `dfsvisitor`/`bfsvisitor`-suffixed scratch file, for the given graph cache id,
    /// currently in the on-disk cache directory used by [`GraphCache`].
    fn traversal_scratch_files(cache_id: &str) -> Vec<std::path::PathBuf> {
        let dir = if cfg!(any(test, feature = "bench")) {
            "./.test_cache"
        } else {
            "./.cache"
        };
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                            (n.contains("dfsvisitor") || n.contains("bfsvisitor"))
                                && n.contains(cache_id)
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn edge_indices_cover_every_edge_forward_and_backward() {
        let graph = build_degree_test_graph();

        let forward: Vec<usize> = graph.iter_edge_indices().collect();
        assert_eq!(forward, (0..graph.width()).collect::<Vec<_>>());

        let backward: Vec<usize> = graph.iter_edge_indices().rev().collect();
        assert_eq!(backward, (0..graph.width()).rev().collect::<Vec<_>>());

        assert_eq!(graph.iter_edge_indices().count(), graph.width());
    }

    #[test]
    fn isolated_nodes_match_zero_degree_nodes() {
        let graph = build_degree_test_graph();

        let mut via_iter: Vec<usize> = graph.iter_isolated_nodes().collect();
        via_iter.sort_unstable();
        let mut via_scan: Vec<usize> = (0..graph.size())
            .filter(|&u| graph.node_degree(u) == 0)
            .collect();
        via_scan.sort_unstable();
        assert_eq!(via_iter, via_scan);
        assert_eq!(via_iter, vec![2, 4]);
    }

    /// Checks `iter_nodes_with_degree`/`_lt`/`_ge` against a brute-force scan over
    /// [`GraphMemoryMap::node_degree`] for every degree threshold actually present in the graph
    /// (plus one past the max, to exercise the "nothing matches" / "everything matches" edges).
    ///
    /// Historical bug: every one of these iterators' `next()`/`next_back()` computed its
    /// advanced `count`/`offset` via `saturating_add`/`saturating_sub` but discarded the result
    /// instead of writing it back --- so `next()` would either infinite-loop (if the first
    /// candidate didn't match) or return the same node forever (if it did), and `next_back()`
    /// behaved identically in reverse.
    #[test]
    fn nodes_with_degree_family_matches_manual_scan_for_every_threshold() {
        let graph = build_degree_test_graph();
        let max_deg = (0..graph.size())
            .map(|u| graph.node_degree(u))
            .max()
            .unwrap();

        for deg in 0..=(max_deg + 1) {
            let iter = graph.iter_nodes_with_degree(deg);
            let (lower, upper) = iter.size_hint();
            let mut via_iter: Vec<usize> = iter.collect();
            // these are filtering iterators: the exact number of *matches* remaining can't be
            // known without scanning ahead (same reason `std::iter::Filter` doesn't implement
            // `ExactSizeIterator`) --- `size_hint` must therefore report a safe lower bound of 0
            // and an upper bound no smaller than the true count.
            assert_eq!(lower, 0);
            assert!(upper.is_some_and(|u| u >= via_iter.len()));
            via_iter.sort_unstable();
            let mut via_scan: Vec<usize> = (0..graph.size())
                .filter(|&u| graph.node_degree(u) == deg)
                .collect();
            via_scan.sort_unstable();
            assert_eq!(via_iter, via_scan, "iter_nodes_with_degree({deg})");

            let mut via_iter_lt: Vec<usize> = graph.iter_nodes_with_degree_lt(deg).collect();
            via_iter_lt.sort_unstable();
            let mut via_scan_lt: Vec<usize> = (0..graph.size())
                .filter(|&u| graph.node_degree(u) < deg)
                .collect();
            via_scan_lt.sort_unstable();
            assert_eq!(via_iter_lt, via_scan_lt, "iter_nodes_with_degree_lt({deg})");

            let mut via_iter_ge: Vec<usize> = graph.iter_nodes_with_degree_ge(deg).collect();
            via_iter_ge.sort_unstable();
            let mut via_scan_ge: Vec<usize> = (0..graph.size())
                .filter(|&u| graph.node_degree(u) >= deg)
                .collect();
            via_scan_ge.sort_unstable();
            assert_eq!(via_iter_ge, via_scan_ge, "iter_nodes_with_degree_ge({deg})");
        }
    }

    #[test]
    fn nodes_with_degree_family_is_double_ended_and_terminates() {
        let graph = build_degree_test_graph();

        for (mut forward, backward_rev) in [
            (
                graph.iter_nodes_with_degree(1).collect::<Vec<_>>(),
                graph.iter_nodes_with_degree(1).rev().collect::<Vec<_>>(),
            ),
            (
                graph.iter_nodes_with_degree_lt(2).collect::<Vec<_>>(),
                graph.iter_nodes_with_degree_lt(2).rev().collect::<Vec<_>>(),
            ),
            (
                graph.iter_nodes_with_degree_ge(2).collect::<Vec<_>>(),
                graph.iter_nodes_with_degree_ge(2).rev().collect::<Vec<_>>(),
            ),
        ] {
            let mut backward = backward_rev;
            forward.sort_unstable();
            backward.sort_unstable();
            assert_eq!(forward, backward);
        }
    }

    /// Cross-checks `iter_neighbors`/`walk_neighbors` against the already-tested, zero-allocation
    /// [`GraphMemoryMap::neighbours`]/[`NeighbourIter`], in both directions.
    ///
    /// Historical bug: `next()` computed its advanced `count`/`offset` but discarded the result
    /// (only the raw pointer itself was correctly advanced), so termination relying on `count`
    /// never actually triggered; `next_back()` for `Neighbors` didn't decrement `count` at all,
    /// so repeated calls returned the same trailing element forever.
    #[test]
    fn iter_neighbors_and_walk_neighbors_match_trusted_neighbours_iterator() {
        let graph = build_degree_test_graph();

        for u in 0..graph.size() {
            let trusted: Vec<usize> = graph.neighbours(u).unwrap().collect();

            let via_iter: Vec<usize> = graph.iter_neighbors(u).collect();
            assert_eq!(trusted, via_iter, "iter_neighbors mismatch for node {u}");
            assert_eq!(graph.iter_neighbors(u).count(), graph.node_degree(u));

            let via_walk: Vec<usize> = graph.walk_neighbors(u).collect();
            assert_eq!(trusted, via_walk, "walk_neighbors mismatch for node {u}");

            let mut rev_trusted = trusted.clone();
            rev_trusted.reverse();

            let via_iter_rev: Vec<usize> = graph.iter_neighbors(u).rev().collect();
            assert_eq!(
                rev_trusted, via_iter_rev,
                "iter_neighbors reverse mismatch for node {u}"
            );

            let via_walk_rev: Vec<usize> = graph.walk_neighbors(u).rev().collect();
            assert_eq!(
                rev_trusted, via_walk_rev,
                "walk_neighbors reverse mismatch for node {u}"
            );
        }
    }

    /// Cross-checks `iter_neighbors_with_offset`/`walk_neighbors_with_offset` --- which pair each
    /// neighbour with its exact flat edge-array position --- against
    /// [`GraphMemoryMap::index_node`] (the position range) zipped with the trusted neighbour
    /// iterator.
    #[cfg(feature = "iter_with_offset")]
    #[test]
    fn neighbors_with_offset_report_exact_edge_array_positions() {
        let graph = build_degree_test_graph();

        for u in 0..graph.size() {
            let expected: Vec<(usize, usize)> = graph
                .index_node(u)
                .zip(graph.neighbours(u).unwrap())
                .collect();

            let via_iter: Vec<(usize, usize)> = graph.iter_neighbors_with_offset(u).collect();
            assert_eq!(
                via_iter, expected,
                "iter_neighbors_with_offset mismatch for node {u}"
            );

            let via_walk: Vec<(usize, usize)> = graph.walk_neighbors_with_offset(u).collect();
            assert_eq!(
                via_walk, expected,
                "walk_neighbors_with_offset mismatch for node {u}"
            );

            let mut expected_rev = expected.clone();
            expected_rev.reverse();

            let via_iter_rev: Vec<(usize, usize)> =
                graph.iter_neighbors_with_offset(u).rev().collect();
            assert_eq!(
                via_iter_rev, expected_rev,
                "iter_neighbors_with_offset reverse mismatch for node {u}"
            );

            let via_walk_rev: Vec<(usize, usize)> =
                graph.walk_neighbors_with_offset(u).rev().collect();
            assert_eq!(
                via_walk_rev, expected_rev,
                "walk_neighbors_with_offset reverse mismatch for node {u}"
            );
        }
    }

    /// `EdgesConnecting` has no public constructor yet (nothing on `GraphMemoryMap` builds one),
    /// so this test reaches its crate-private `new` directly --- valid since this test module is
    /// a descendant of the `graph` module. Checks both the "found" path (an actual edge run) and
    /// the "not found" path (no edge to that target), in both directions.
    ///
    /// Historical bug: `next()`/`next_back()` computed `count`/`offset` updates via
    /// `saturating_add`/`saturating_sub` but discarded the results, so the iterator never
    /// advanced past its first element.
    #[test]
    fn edges_connecting_finds_exact_contiguous_run() {
        let graph = build_degree_test_graph();
        let neigh_ptr = graph.neighbours_ptr();
        let off_ptr = graph.offsets_ptr();

        for orig in 0..graph.size() {
            let positions: Vec<usize> = graph.index_node(orig).collect();
            let neighbours: Vec<usize> = graph.neighbours(orig).unwrap().collect();

            for target in 0..graph.size() {
                let expected: Vec<usize> = positions
                    .iter()
                    .copied()
                    .zip(neighbours.iter().copied())
                    .filter(|&(_, n)| n == target)
                    .map(|(pos, _)| pos)
                    .collect();

                let conn = EdgesConnecting::<VoidLabel, VoidLabel, usize>::new(
                    neigh_ptr, off_ptr, orig, target,
                );
                assert_eq!(conn.remaning_edges(), expected.len());

                let mut via_iter: Vec<usize> = EdgesConnecting::<VoidLabel, VoidLabel, usize>::new(
                    neigh_ptr, off_ptr, orig, target,
                )
                .collect();
                via_iter.sort_unstable();
                assert_eq!(via_iter, expected, "orig={orig} target={target}");

                let mut via_iter_rev: Vec<usize> =
                    EdgesConnecting::<VoidLabel, VoidLabel, usize>::new(
                        neigh_ptr, off_ptr, orig, target,
                    )
                    .rev()
                    .collect();
                via_iter_rev.sort_unstable();
                assert_eq!(
                    via_iter_rev, expected,
                    "reverse orig={orig} target={target}"
                );
            }
        }
    }

    /// The sequential, manually-chunked `paralelizable_iter_nodes_with_degree_lt` shares no code
    /// with [`NodesWithDegreeLT`] --- it had its own, independently wrong comparison (`>` instead
    /// of `>=`), which silently included nodes whose degree was *equal to* (not just less than)
    /// the threshold.
    #[test]
    fn paralelizable_nodes_with_degree_lt_matches_manual_scan() {
        let graph = build_degree_test_graph();
        let max_deg = (0..graph.size())
            .map(|u| graph.node_degree(u))
            .max()
            .unwrap();

        for deg in 0..=(max_deg + 1) {
            let mut via_iter: Vec<usize> =
                graph.paralelizable_iter_nodes_with_degree_lt(deg).collect();
            via_iter.sort_unstable();
            let mut via_scan: Vec<usize> = (0..graph.size())
                .filter(|&u| graph.node_degree(u) < deg)
                .collect();
            via_scan.sort_unstable();
            assert_eq!(
                via_iter, via_scan,
                "paralelizable_iter_nodes_with_degree_lt({deg})"
            );
        }
    }

    /// Minimal real `E` implementation (unlike [`crate::graph::label::EdgeLabel`], which never
    /// overrides `is_labeled`) used to exercise the edge-label read path end to end: without
    /// `is_labeled() == true`, the mtx parser never writes real weights into the edge-label file
    /// in the first place, so `iter_edge_labels`/`edge_label` would only ever observe zeros.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    struct TestWeight(usize);

    unsafe impl bytemuck::Zeroable for TestWeight {}
    unsafe impl bytemuck::Pod for TestWeight {}

    impl crate::graph::E for TestWeight {
        fn new(v: usize) -> Self {
            TestWeight(v)
        }
        fn is_labeled() -> bool {
            true
        }
    }

    /// Directed, integer-weighted graph:
    /// 0 -> 1 (w=10), 0 -> 2 (w=20), 1 -> 2 (w=30), 3 -> 0 (w=40)
    fn build_weighted_edge_test_graph()
    -> crate::graph::test_utils::TestGraph<VoidLabel, TestWeight, usize, crate::graph::Directed>
    {
        let dir = unique_test_dir("label_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("weighted.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate integer general\n4 4 4\n1 2 10\n1 3 20\n2 3 30\n4 1 40\n",
        )
        .expect("error writing mtx fixture");

        let cache = GraphCache::<VoidLabel, TestWeight, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph =
            crate::graph::DiGraph::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        crate::graph::test_utils::TestGraph::new(graph, None)
    }

    #[test]
    fn edge_labels_match_input_weights_in_flat_edge_array_order() {
        let graph = build_weighted_edge_test_graph();

        // ground truth: (source, target) -> weight, as given in the mtx fixture above.
        let expected_weight = |u: usize, v: usize| -> usize {
            match (u, v) {
                (0, 1) => 10,
                (0, 2) => 20,
                (1, 2) => 30,
                (3, 0) => 40,
                _ => panic!("no edge {u} -> {v} in the fixture"),
            }
        };

        for u in 0..graph.size() {
            for (pos, v) in graph.index_node(u).zip(graph.neighbours(u).unwrap()) {
                assert_eq!(
                    graph.edge_label(pos),
                    TestWeight(expected_weight(u, v)),
                    "edge_label mismatch for edge {u} -> {v} at position {pos}"
                );
            }
        }

        // `iter_edge_labels` must walk the exact same flat order as `edge_label(pos)`.
        let via_iter: Vec<TestWeight> = graph.iter_edge_labels().collect();
        let via_accessor: Vec<TestWeight> =
            (0..graph.width()).map(|p| graph.edge_label(p)).collect();
        assert_eq!(via_iter, via_accessor);
        assert_eq!(graph.iter_edge_labels().count(), graph.width());

        // double-ended: reverse should be the exact mirror.
        let mut via_iter_rev: Vec<TestWeight> = graph.iter_edge_labels().rev().collect();
        via_iter_rev.reverse();
        assert_eq!(via_iter_rev, via_iter);
    }

    /// For the common, unlabeled (`VoidLabel`) case, label iteration must still terminate
    /// correctly (yielding exactly `size()`/`width()` items) without ever touching the 1-byte
    /// placeholder mmap, since `VoidLabel` is zero-sized.
    #[test]
    fn node_and_edge_labels_are_well_defined_for_void_label_graphs() {
        let graph = build_degree_test_graph();

        let node_labels: Vec<VoidLabel> = graph.iter_node_labels().collect();
        assert_eq!(node_labels.len(), graph.size());

        let edge_labels: Vec<VoidLabel> = graph.iter_edge_labels().collect();
        assert_eq!(edge_labels.len(), graph.width());

        for u in 0..graph.size() {
            assert_eq!(graph.node_label(u), VoidLabel {});
        }
        for p in 0..graph.width() {
            assert_eq!(graph.edge_label(p), VoidLabel {});
        }
    }

    /// Regression test for a real pre-existing bug: `GraphCache::open` (the path used when
    /// re-opening an *already built* cache, as opposed to `from_file`'s build-or-reuse) was
    /// opening `metalabel_filename` instead of `nodelabel_filename` for its `nodelabel_file`
    /// handle. Harmless for unlabeled graphs (it only reads garbage never exercised), but would
    /// have made node-label reads on a *reopened* labeled-node graph silently wrong. This test
    /// exercises edge labels (the only label kind this crate currently ever populates from an
    /// mtx file) across a drop-and-reopen cycle.
    #[test]
    fn edge_labels_survive_reopening_the_cache() {
        let dir = unique_test_dir("reopen_label_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("weighted_reopen.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate integer general\n4 4 4\n1 2 10\n1 3 20\n2 3 30\n4 1 40\n",
        )
        .expect("error writing mtx fixture");

        let cache = GraphCache::<VoidLabel, TestWeight, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph =
            crate::graph::DiGraph::<VoidLabel, TestWeight, usize>::init_from_cache(cache, Some(1))
                .expect("load should succeed");
        let before: Vec<TestWeight> = graph.iter_edge_labels().collect();
        let edges_filename = graph.cache_edges_filename();
        drop(graph);

        let reopened = crate::graph::DiGraph::<VoidLabel, TestWeight, usize>::open(
            &edges_filename,
            None,
            Some(1),
        )
        .expect("reopening the just-built cache should succeed");
        let after: Vec<TestWeight> = reopened.iter_edge_labels().collect();

        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(before, after);
    }

    #[test]
    fn weighted_neighbours_matches_neighbours_zipped_with_edge_label() {
        let graph = build_weighted_edge_test_graph();

        for u in 0..graph.size() {
            let expected: Vec<(usize, TestWeight)> = graph
                .neighbours(u)
                .unwrap()
                .zip(graph.index_node(u).map(|pos| graph.edge_label(pos)))
                .collect();
            let via_iter: Vec<(usize, TestWeight)> =
                graph.weighted_neighbours(u).unwrap().collect();
            assert_eq!(
                via_iter, expected,
                "weighted_neighbours mismatch for node {u}"
            );
            assert_eq!(graph.weighted_neighbours(u).unwrap().len(), expected.len());

            let mut expected_rev = expected.clone();
            expected_rev.reverse();
            let via_iter_rev: Vec<(usize, TestWeight)> =
                graph.weighted_neighbours(u).unwrap().rev().collect();
            assert_eq!(
                via_iter_rev, expected_rev,
                "weighted_neighbours reverse mismatch for node {u}"
            );
        }

        assert!(graph.weighted_neighbours(graph.size()).is_err());
    }

    #[test]
    fn edges_visits_every_edge_exactly_once_with_correct_endpoints_and_weight() {
        let graph = build_weighted_edge_test_graph();

        // ground truth: walk every node's weighted neighbours manually, in node-id order ---
        // `edges()` must produce the exact same flattened sequence, since both are defined to
        // walk the flat edge array in order.
        let mut expected: Vec<(usize, usize, TestWeight)> = Vec::new();
        for u in 0..graph.size() {
            for (v, w) in graph.weighted_neighbours(u).unwrap() {
                expected.push((u, v, w));
            }
        }

        let via_iter: Vec<(usize, usize, TestWeight)> = graph.edges().collect();
        assert_eq!(via_iter, expected);
        assert_eq!(graph.edges().len(), graph.width());
        assert_eq!(graph.edges().count(), expected.len());
    }

    #[test]
    fn edges_correctly_skips_isolated_nodes() {
        // reuses the degree-varied fixture, which has two isolated (degree-0) nodes (2 and 4)
        // interleaved with non-isolated ones --- exercising `Edges::next()`'s node-boundary
        // skip loop across more than one consecutive/non-consecutive isolated node.
        let graph = build_degree_test_graph();

        let mut expected: Vec<(usize, usize)> = Vec::new();
        for u in 0..graph.size() {
            for v in graph.neighbours(u).unwrap() {
                expected.push((u, v));
            }
        }

        let via_iter: Vec<(usize, usize)> = graph.edges().map(|(u, v, _)| (u, v)).collect();
        assert_eq!(via_iter, expected);
    }

    /// Exercises `ExactSizeIterator::len()` (whose default implementation debug-asserts that
    /// `size_hint()` reports `(n, Some(n))`) for every struct that now implements it, to guard
    /// against a historical landmine: several of these already had `ExactSizeIterator` declared
    /// but `size_hint()` returning `(n, None)`, which would have panicked in debug builds the
    /// first time anyone actually called `.len()`.
    #[test]
    fn exact_size_iterator_len_matches_count_for_every_implementor() {
        let graph = build_degree_test_graph();

        assert_eq!(
            graph.iter_node_indices().len(),
            graph.iter_node_indices().count()
        );
        assert_eq!(
            graph.iter_edge_indices().len(),
            graph.iter_edge_indices().count()
        );
        for u in 0..graph.size() {
            assert_eq!(
                graph.iter_neighbors(u).len(),
                graph.iter_neighbors(u).count()
            );
            assert_eq!(
                graph.walk_neighbors(u).len(),
                graph.walk_neighbors(u).count()
            );
        }
        assert_eq!(
            graph.iter_node_labels().len(),
            graph.iter_node_labels().count()
        );
        assert_eq!(
            graph.iter_edge_labels().len(),
            graph.iter_edge_labels().count()
        );

        #[cfg(feature = "iter_with_offset")]
        for u in 0..graph.size() {
            assert_eq!(
                graph.iter_neighbors_with_offset(u).len(),
                graph.iter_neighbors_with_offset(u).count()
            );
            assert_eq!(
                graph.walk_neighbors_with_offset(u).len(),
                graph.walk_neighbors_with_offset(u).count()
            );
        }

        let neigh_ptr = graph.neighbours_ptr();
        let off_ptr = graph.offsets_ptr();
        let conn = EdgesConnecting::<VoidLabel, VoidLabel, usize>::new(neigh_ptr, off_ptr, 0, 1);
        assert_eq!(conn.len(), conn.remaning_edges());

        let weighted = build_weighted_edge_test_graph();
        for u in 0..weighted.size() {
            assert_eq!(
                weighted.weighted_neighbours(u).unwrap().len(),
                weighted.weighted_neighbours(u).unwrap().count()
            );
        }
        assert_eq!(weighted.edges().len(), weighted.edges().count());
    }

    /// Verifies that `ParNodesWithDegree`/`LT`/`GE` --- after removing their `IndexedParallelIterator`
    /// impls (which falsely claimed `len()` equal to the candidate count, not the actual filtered
    /// match count, causing `rayon`'s `bridge()` to panic on `.collect()`) --- still produce
    /// correct results through their remaining `ParallelIterator`/`UnindexedProducer` path.
    #[cfg(feature = "rayon")]
    #[test]
    fn par_iter_nodes_with_degree_family_matches_sequential_after_producer_change() {
        let graph = build_degree_test_graph();
        let max_deg = (0..graph.size())
            .map(|u| graph.node_degree(u))
            .max()
            .unwrap();

        for deg in 0..=(max_deg + 1) {
            let mut seq: Vec<usize> = graph.iter_nodes_with_degree(deg).collect();
            seq.sort_unstable();
            let mut par: Vec<usize> = graph.par_iter_nodes_with_degree(deg).collect();
            par.sort_unstable();
            assert_eq!(par, seq, "par_iter_nodes_with_degree({deg})");

            let mut seq_lt: Vec<usize> = graph.iter_nodes_with_degree_lt(deg).collect();
            seq_lt.sort_unstable();
            let mut par_lt: Vec<usize> = graph.par_iter_nodes_with_degree_lt(deg).collect();
            par_lt.sort_unstable();
            assert_eq!(par_lt, seq_lt, "par_iter_nodes_with_degree_lt({deg})");

            let mut seq_ge: Vec<usize> = graph.iter_nodes_with_degree_ge(deg).collect();
            seq_ge.sort_unstable();
            let mut par_ge: Vec<usize> = graph.par_iter_nodes_with_degree_ge(deg).collect();
            par_ge.sort_unstable();
            assert_eq!(par_ge, seq_ge, "par_iter_nodes_with_degree_ge({deg})");
        }
    }

    /// Graph large enough to force *real* splitting in every rayon `Producer`/`UnindexedProducer`
    /// in this file (the split threshold is 4096 items) --- every other test fixture in this
    /// module is far too small to ever trigger `split`/`split_at`, so none of them have actually
    /// exercised that logic before now. Node 0 is a "hub" with degree 6000 (> 4096, forcing
    /// `ParNeighbors`'s own per-node split path specifically); the remaining ~4000 nodes form a
    /// simple chain (degree 1 each), plus the 6000 pure hub targets (degree 0 each) and the
    /// chain's final node (also degree 0) giving a real spread of isolated/degree-1/high-degree
    /// nodes for the degree-filter family too.
    #[cfg(feature = "rayon")]
    fn build_large_test_graph()
    -> crate::graph::test_utils::TestGraph<VoidLabel, VoidLabel, usize, crate::graph::Directed>
    {
        let dir = unique_test_dir("large_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("large.mtx");

        const HUB_DEGREE: usize = 6000;
        const CHAIN_LEN: usize = 4000;
        const NODE_COUNT: usize = 1 + HUB_DEGREE + CHAIN_LEN;
        const EDGE_COUNT: usize = HUB_DEGREE + (CHAIN_LEN - 1);

        let to_mtx = |node: usize| node + 1;
        let mut mtx = String::with_capacity(EDGE_COUNT * 12 + 64);
        mtx.push_str("%%MatrixMarket matrix coordinate pattern general\n");
        mtx.push_str(&format!("{NODE_COUNT} {NODE_COUNT} {EDGE_COUNT}\n"));
        for i in 1..=HUB_DEGREE {
            mtx.push_str(&format!("{} {}\n", to_mtx(0), to_mtx(i)));
        }
        for k in 0..(CHAIN_LEN - 1) {
            let a = HUB_DEGREE + 1 + k;
            let b = a + 1;
            mtx.push_str(&format!("{} {}\n", to_mtx(a), to_mtx(b)));
        }
        std::fs::write(&path, mtx).expect("error writing large mtx fixture");

        let cache = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph =
            crate::graph::DiGraph::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        crate::graph::test_utils::TestGraph::new(graph, None)
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_iter_node_and_edge_indices_match_sequential_at_split_forcing_scale() {
        let graph = build_large_test_graph();
        assert!(
            graph.size() > 4096 && graph.width() > 4096,
            "fixture too small to force a real split"
        );

        let mut seq: Vec<usize> = graph.iter_node_indices().collect();
        seq.sort_unstable();
        let mut par: Vec<usize> = graph.par_iter_node_indices().collect();
        par.sort_unstable();
        assert_eq!(par, seq, "par_iter_node_indices");
        assert_eq!(par.len(), graph.size());

        let mut seq_e: Vec<usize> = graph.iter_edge_indices().collect();
        seq_e.sort_unstable();
        let mut par_e: Vec<usize> = graph.par_iter_edge_indices().collect();
        par_e.sort_unstable();
        assert_eq!(par_e, seq_e, "par_iter_edge_indices");
        assert_eq!(par_e.len(), graph.width());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_iter_neighbors_matches_sequential_at_split_forcing_scale() {
        let graph = build_large_test_graph();
        // node 0 is the hub with degree 6000, comfortably above the 4096 split threshold ---
        // this specifically forces `ParNeighbors::split`/`split_at` to actually run, unlike every
        // other (tiny) fixture in this module.
        assert!(
            graph.node_degree(0) > 4096,
            "hub not large enough to force a real split"
        );

        let mut seq: Vec<usize> = graph.neighbours(0).unwrap().collect();
        seq.sort_unstable();
        let mut par: Vec<usize> = graph.par_iter_neighbors(0).collect();
        par.sort_unstable();
        assert_eq!(par, seq, "par_iter_neighbors(0)");
        assert_eq!(par.len(), graph.node_degree(0));

        // a degree-1 node, well under the split threshold, as a small-input control.
        let small_node = 1 + 6000; // first chain node, per build_large_test_graph's layout
        let mut seq_small: Vec<usize> = graph.neighbours(small_node).unwrap().collect();
        seq_small.sort_unstable();
        let mut par_small: Vec<usize> = graph.par_iter_neighbors(small_node).collect();
        par_small.sort_unstable();
        assert_eq!(par_small, seq_small, "par_iter_neighbors(small_node)");
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_iter_nodes_with_degree_family_matches_sequential_at_split_forcing_scale() {
        let graph = build_large_test_graph();
        assert!(
            graph.size() > 4096,
            "fixture too small to force a real split"
        );

        for deg in [0usize, 1, 2, 6000] {
            let mut seq: Vec<usize> = graph.iter_nodes_with_degree(deg).collect();
            seq.sort_unstable();
            let mut par: Vec<usize> = graph.par_iter_nodes_with_degree(deg).collect();
            par.sort_unstable();
            assert_eq!(par, seq, "par_iter_nodes_with_degree({deg}) at scale");

            let mut seq_lt: Vec<usize> = graph.iter_nodes_with_degree_lt(deg).collect();
            seq_lt.sort_unstable();
            let mut par_lt: Vec<usize> = graph.par_iter_nodes_with_degree_lt(deg).collect();
            par_lt.sort_unstable();
            assert_eq!(
                par_lt, seq_lt,
                "par_iter_nodes_with_degree_lt({deg}) at scale"
            );

            let mut seq_ge: Vec<usize> = graph.iter_nodes_with_degree_ge(deg).collect();
            seq_ge.sort_unstable();
            let mut par_ge: Vec<usize> = graph.par_iter_nodes_with_degree_ge(deg).collect();
            par_ge.sort_unstable();
            assert_eq!(
                par_ge, seq_ge,
                "par_iter_nodes_with_degree_ge({deg}) at scale"
            );
        }

        // isolated-node convenience method shares the same (now-fixed) code path.
        let mut seq_iso: Vec<usize> = graph.iter_isolated_nodes().collect();
        seq_iso.sort_unstable();
        let mut par_iso: Vec<usize> = graph.par_iter_isolated_nodes().collect();
        par_iso.sort_unstable();
        assert_eq!(par_iso, seq_iso, "par_iter_isolated_nodes at scale");
    }

    /// Dropping a graph's cache also removes what algorithms left under its id, but nothing of
    /// other graphs.
    #[test]
    fn drop_cache_removes_every_file_of_the_graph() {
        use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
        use crate::k_truss::triangles::Triangles;
        type G = crate::graph::Graph<VoidLabel, VoidLabel, usize>;

        let dir = unique_test_dir("drop_cache");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("triangle.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 3\n2 1\n3 1\n3 2\n",
        )
        .unwrap();
        let tag = crate::graph::test_utils::unique_run_tag();
        let open =
            |name: &str| G::from_file(&path, Some(format!("{name}_{tag}")), Some(1)).unwrap();
        let files_of = |g: &G| -> Vec<std::path::PathBuf> {
            let id = g.graph_id().unwrap();
            let pattern = format!("{}*{id}.*", crate::graph::cache::utils::CACHE_DIR);
            glob::glob(&pattern).unwrap().flatten().collect()
        };

        let (mut g, other) = (open("dropped"), open("kept"));
        // results that aren't dropped, and a scratch file
        AlgoBatageljZaversnik::new(&g).unwrap();
        Triangles::new(&g).unwrap();
        std::fs::write(
            g.build_cache_filename(CacheFile::General, Some(3)).unwrap(),
            b"",
        )
        .unwrap();
        AlgoBatageljZaversnik::new(&other).unwrap();

        let (files, other_files) = (files_of(&g), files_of(&other));
        assert!(files.len() > 6, "{files:?}");
        g.drop_cache().unwrap();
        assert!(files.iter().all(|f| !f.exists()), "left behind: {files:?}");
        assert_eq!(files_of(&other), other_files);
    }
}
