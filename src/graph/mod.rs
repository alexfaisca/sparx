pub(crate) mod cache;
mod export_induced_subgraph;
mod impl_algos;
mod impl_csc;
mod impl_dists;
mod impl_miscelanious;
mod impl_partition;
pub(crate) mod impl_traits;
pub mod label;

#[cfg(feature = "petgraph")]
mod export_petgraph;
#[cfg(feature = "rustworkx")]
mod export_rustworkx_core;
#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

use crate::graph::cache::GraphCache;
use crate::shared_slice::AbstractedProceduralMemory;
use cache::CacheMetadata;
pub use graph_derive::{GenericEdge, GenericEdgeType};

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
/// won't have performance issues give nthe nature of our algorithms.
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
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

unsafe impl IndexType for usize {
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

unsafe impl<Ix: IndexType> IndexType for NodeIndex<Ix> {
    fn index(&self) -> usize {
        self.0.index()
    }
    fn new(x: usize) -> Self {
        NodeIndex::new(x)
    }
    fn max() -> Self {
        NodeIndex(<Ix as IndexType>::max())
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
    #[inline]
    fn weigth(&self) -> usize {
        1
    }
}

/// Describes the behavior edge labels must exhibit to be used by the tool.
///
/// Given a repr C/transparent struct the required traits may be derived automatically by
/// annotating the struct with #[sparkx_label].
#[allow(dead_code)]
pub trait E:
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
    #[inline]
    fn weigth(&self) -> usize {
        1
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
    DFS,
    EulerIndex,
    EulerTrail,
    KCoreBZ,
    KCoreLEA,
    Triangles,
    KTrussBEA,
    KTrussPKT,
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
}

enum GraphFile {
    Metadata,
    Neighbors,
    Offsets,
    NodeLabels,
    EdgeLabels,
    MetaLabels,
}

#[derive(Clone)]
pub struct GraphMemoryMap<
    N: crate::graph::N = VoidLabel,
    E: crate::graph::E = VoidLabel,
    Ix: crate::graph::IndexType = DefaultIx,
> {
    graph: Arc<memmap2::Mmap>,
    index: Arc<memmap2::Mmap>,
    metalabels: Arc<fst::Map<memmap2::Mmap>>,
    graph_cache: GraphCache<N, E, Ix>,
    edge_size: usize,
    thread_count: u8,
    exports: u8,
    _marker: PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
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
            let edge_size = std::mem::size_of::<Ix>();
            let thread_count = thread_count.unwrap_or(1).max(1);
            let exports = 0u8;

            return Ok(Self {
                graph,
                index,
                metalabels,
                graph_cache,
                edge_size,
                thread_count,
                exports,
                _marker: PhantomData::<Ix>,
            });
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
    #[inline(always)]
    pub fn from_file<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<N, E, Ix>::from_file(p, id, None, None)?;
        Self::init_from_cache(graph_cache, thread_count)
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
        Self::init_from_cache(graph_cache, thread_count)
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
    ) -> Result<GraphMemoryMap<N, E, NewIx>, Box<dyn std::error::Error>> {
        let new_cache = self.graph_cache.convert_index_type::<NewIx>()?;
        GraphMemoryMap::<N, E, NewIx>::init_from_cache(new_cache, Some(self.thread_count))
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

    /// Returns the given (by id) node's degree.
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

    /// Returns a [`NeighbourIter`] iterator over the given (by id) node's neighbours.
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
    /// # Arguments
    ///
    /// * `communities` --- a slice containing each node's community, the [`GraphMemoryMap`] instance's partition.
    /// * `comms_cardinality` --- cardinality of distinct communities in the given partition.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn modularity(
        &self,
        communities: &[usize],
        comms_cardinality: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.modularity_impl(communities, comms_cardinality)
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
    /// [^1]: the node ids of the subgraph may not, and probably will not, correspond to the original node identifiers, efffectively, it will be a whole new graph.
    pub fn induced_subgraph(
        &mut self,
        mask: fn(usize) -> bool,
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
    pub fn edge_over(
        &self,
    ) -> Result<AbstractedProceduralMemory<Ix>, Box<dyn std::error::Error>> {
        self.get_edge_over_impl()
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
        self.graph_cache.build_cache_filename(file_type, seq)
    }

    /// Remove [`GraphMemoryMap`] instance's cached `.tmp` files for a given [`CacheFile`] in the cache directory.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`CacheFile`]: ./enum.CacheFile.html#
    #[inline(always)]
    pub fn cleanup_cache(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache.cleanup_cache(target)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
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

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut this = ManuallyDrop::new(self);

        let er_fn = this.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let eo_fn = this.build_cache_filename(CacheFile::EdgeOver, None)?;

        let _r = std::fs::remove_file(er_fn);
        let _r = std::fs::remove_file(eo_fn);

        this.graph_cache.drop_cache()
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
        self.count -= 1;
        let next: (usize, usize);
        unsafe {
            next = (self.id, self.edge_ptr.add(self.count).read_unaligned().index());
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
        self.count -= 1;
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
        self.count -= 1;
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
        self.count -= 1;
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::RangeFull> for GraphMemoryMap<N, E, Ix>
{
    type Output = [Ix];
    #[inline]
    fn index(&self, _index: std::ops::RangeFull) -> &[Ix] {
        unsafe { std::slice::from_raw_parts(self.graph.as_ptr() as *const Ix, self.width()) }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::Range<usize>> for GraphMemoryMap<N, E, Ix>
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::Range<u64>> for GraphMemoryMap<N, E, Ix>
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Debug
    for GraphMemoryMap<N, E, Ix>
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
    /// vol(S)
    pub width: usize,
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
        Some(self.end.saturating_sub(1))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NodeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.begin < self.end {
            return None;
        }
        let res = self.begin;
        let _ = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.begin, None)
    }
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
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParNodeIndices<N, E, Ix>
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
                    stop_offset: mid,
                    offset: self.offset,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    stop_offset: self.stop_offset,
                    offset: mid,
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
            folder = folder.consume(i);
        }
        folder
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

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
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
        Some(self.end.saturating_sub(1))
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for EdgeIndices<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.begin < self.end {
            return None;
        }
        let res = self.begin;
        let _ = self.begin.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.begin, None)
    }
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
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParEdgeIndices<N, E, Ix>
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
                    stop_offset: mid,
                    offset: self.offset,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    stop_offset: self.stop_offset,
                    offset: mid,
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
            folder = folder.consume(i);
        }
        folder
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

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
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
        let mut offset = unsafe { off_mmap.add(orig_node).read().index() };
        let mut count = unsafe { off_mmap.add(orig_node + 1).read().index() };

        let mut begin = false;
        #[allow(clippy::mut_range_bound)]
        for i in offset..count {
            if !begin && unsafe { neigh_mmap.add(i).read().index() } == target_node {
                begin = true;
                offset = i;
            } else if begin && unsafe { neigh_mmap.add(i).read().index() } != target_node {
                count = i - offset;
            }
        }
        if !begin {
            count = 0;
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
        Some(self.offset + self.count.saturating_sub(1))
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
        let _ = self.count.saturating_sub(1);
        let res = self.offset;
        let _ = self.offset.saturating_add(1);
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, None)
    }
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
            let curr = unsafe { self.off_ptr.add(self.count).read().index() };
            if curr - unsafe { self.off_ptr.add(self.count.saturating_sub(1)).read().index() }
                != self.target_deg
            {
                continue;
            }
            return Some(self.offset + self.count);
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
            let _ = self.count.saturating_sub(1);
            if unsafe {
                self.off_ptr.add(self.offset + 1).read().index() - self.off_ptr.add(self.offset).read().index()
            } != self.target_deg
            {
                let _ = self.offset.saturating_add(1);
                continue;
            }
            let _ = self.offset.saturating_add(1);
            return Some(self.offset);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, None)
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
            let curr = unsafe { self.off_ptr.add(self.count).read().index() };
            if curr - unsafe { self.off_ptr.add(self.count.saturating_sub(1)).read().index() }
                >= self.target_deg
            {
                continue;
            }
            return Some(self.offset + self.count);
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
            let _ = self.count.saturating_sub(1);
            if unsafe {
                self.off_ptr.add(self.offset + 1).read().index() - self.off_ptr.add(self.offset).read().index()
            } >= self.target_deg
            {
                let _ = self.offset.saturating_add(1);
                continue;
            }
            let _ = self.offset.saturating_add(1);
            return Some(self.offset);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, None)
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
            let curr = unsafe { self.off_ptr.add(self.count).read().index() };
            if curr - unsafe { self.off_ptr.add(self.count.saturating_sub(1)).read().index() }
                < self.target_deg
            {
                continue;
            }
            return Some(self.offset + self.count);
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
            let _ = self.count.saturating_sub(1);
            if unsafe {
                self.off_ptr.add(self.offset + 1).read().index() - self.off_ptr.add(self.offset).read().index()
            } < self.target_deg
            {
                let _ = self.offset.saturating_add(1);
                continue;
            }
            let _ = self.offset.saturating_add(1);
            return Some(self.offset);
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, None)
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
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            if unsafe { self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index() }
                > self.target_deg
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NodesWithDegreeLT<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegreeLT<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            let deg = unsafe { self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index() };
            if deg < self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParNodesWithDegreeLT<N, E, Ix>
{
    type Item = usize;
    type IntoIter = NodesWithDegreeLT<N, E, Ix>;

    fn into_iter(self) -> Self::IntoIter {
        NodesWithDegreeLT::new(
            unsafe { self.off_ptr.add(self.offset) },
            self.stop_offset - self.offset,
            self.target_deg,
        )
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
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
            Self {
                off_ptr: self.off_ptr,
                stop_offset: self.stop_offset,
                offset: mid,
                target_deg: self.target_deg,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
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

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParNodesWithDegreeLT<N, E, Ix>
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
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            if unsafe { self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index() }
                < self.target_deg
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NodesWithDegreeGE<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegreeGE<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            let deg = unsafe { self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index() };
            if deg >= self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParNodesWithDegreeGE<N, E, Ix>
{
    type Item = usize;
    type IntoIter = NodesWithDegreeGE<N, E, Ix>;

    fn into_iter(self) -> Self::IntoIter {
        NodesWithDegreeGE::new(
            unsafe { self.off_ptr.add(self.offset) },
            self.stop_offset - self.offset,
            self.target_deg,
        )
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
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
            Self {
                off_ptr: self.off_ptr,
                stop_offset: self.stop_offset,
                offset: mid,
                target_deg: self.target_deg,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
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

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParNodesWithDegreeGE<N, E, Ix>
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
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            if unsafe { self.off_ptr.add(curr + 1).read().index() - self.off_ptr.add(curr).read().index() }
                != self.target_deg
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> ExactSizeIterator
    for NodesWithDegree<N, E, Ix>
{
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    ParNodesWithDegree<N, E, Ix>
{
    fn new(off_mmap: *const Ix, node_count: usize, target_deg: usize) -> Self {
        let stop_offset = unsafe { off_mmap.add(node_count).read().index() };
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
            let deg = unsafe { self.off_ptr.add(i + 1).read().index() - self.off_ptr.add(i).read().index() };
            if deg == self.target_deg {
                folder = folder.consume(i);
            }
        }
        folder
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Producer
    for ParNodesWithDegree<N, E, Ix>
{
    type Item = usize;
    type IntoIter = NodesWithDegree<N, E, Ix>;

    fn into_iter(self) -> Self::IntoIter {
        NodesWithDegree::new(
            unsafe { self.off_ptr.add(self.offset) },
            self.stop_offset - self.offset,
            self.target_deg,
        )
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        debug_assert!(index <= self.stop_offset - self.offset);
        let mid = self.offset + index;
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
            Self {
                off_ptr: self.off_ptr,
                stop_offset: self.stop_offset,
                offset: mid,
                target_deg: self.target_deg,
                _phantom1: std::marker::PhantomData::<N>,
                _phantom2: std::marker::PhantomData::<E>,
                _phantom3: std::marker::PhantomData::<Ix>,
            },
        )
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

    fn opt_len(&self) -> Option<usize> {
        Some(self.stop_offset - self.offset)
    }
}

#[cfg(feature = "rayon")]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> IndexedParallelIterator
    for ParNodesWithDegree<N, E, Ix>
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
        unsafe {
            Some(
                self.neigh_ptr
                    .add(self.count.saturating_sub(1))
                    .read_unaligned().index(),
            )
        }
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
        let _ = self.count.saturating_sub(1);
        let next: usize;
        self.neigh_ptr = unsafe {
            next = self.neigh_ptr.read().index();
            self.neigh_ptr.add(1)
        };
        let _ = self.offset.saturating_add(1);
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

#[cfg(feature = "rayon")]
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
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> UnindexedProducer
    for ParNeighbors<N, E, Ix>
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
                    neigh_ptr: self.neigh_ptr,
                    stop_offset: mid,
                    offset: self.offset,
                    _phantom1: std::marker::PhantomData::<N>,
                    _phantom2: std::marker::PhantomData::<E>,
                    _phantom3: std::marker::PhantomData::<Ix>,
                },
                Some(Self {
                    neigh_ptr: self.neigh_ptr,
                    stop_offset: self.stop_offset,
                    offset: mid,
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
            unsafe { folder = folder.consume(self.neigh_ptr.add(i).read().index()) };
        }
        folder
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

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
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
        let _ = self.count.saturating_sub(1);
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
        let _ = self.count.saturating_sub(1);
        let next: (usize, usize);
        self.neigh_ptr = unsafe {
            next = (self.offset, self.neigh_ptr.read_unaligned().index());
            self.neigh_ptr.add(1)
        };
        let _ = self.offset.saturating_add(1);
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
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
        let _ = self.count.saturating_sub(1);
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
        let _ = self.count.saturating_sub(1);
        let _ = self.offset.saturating_add(1);
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
        let _ = self.count.saturating_sub(1);
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
        let _ = self.count.saturating_sub(1);
        let next: (usize, usize);
        self.neigh_ptr = unsafe {
            next = (self.offset, self.neigh_ptr.read_unaligned().index());
            self.neigh_ptr.add(1)
        };
        let _ = self.offset.saturating_add(1);
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

pub struct NodeLabels {}
pub struct NodeLabelsMut {}
pub struct EdgeLabels {}
pub struct EdgeLabelsMut {}
