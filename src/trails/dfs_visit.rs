use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;
use std::ops::ControlFlow;

type ProceduralMemoryDFS = (
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
);

// ----- Minimal graph trait -----
pub trait Graph {
    /// Width-matched node/edge index type --- threaded through `neigh`'s item type,
    /// [`DfsVisitor`]'s callback parameters, and `DFSVisit`'s own internal scratch arrays, so
    /// nothing in this traversal ever needs to round-trip through `usize`.
    type Ix: crate::graph::IndexType;
    /// |V|
    fn s(&self) -> usize;
    /// |E|
    fn w(&self) -> usize;
    /// Iterator type returned by [`neigh`](Graph::neigh); borrows no allocation --- implementors
    /// are expected to walk the graph's adjacency representation directly, in place.
    type Neigh<'a>: Iterator<Item = Self::Ix>
    where
        Self: 'a;
    /// neighbour function
    fn neigh(&self, u: Self::Ix) -> Self::Neigh<'_>;
    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>>;
    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>>;
    /// Whether the graph is directed: if not, the edge back to a node's parent is the tree edge
    /// walked the other way, rather than a back edge.
    fn directed(&self) -> bool {
        false
    }
}

/// DFS visitor with default no-ops. Return Break(()) to stop early.
pub trait DfsVisitor<Ix: crate::graph::IndexType> {
    #[inline]
    fn start_vertex(&mut self, _s: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn discover_vertex(&mut self, _u: Ix, _time: Ix, _parent: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn examine_edge(&mut self, _u: Ix, _v: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn tree_edge(&mut self, _u: Ix, _v: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn back_edge(&mut self, _u: Ix, _v: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn forward_or_cross_edge(&mut self, _u: Ix, _v: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn finish_vertex(&mut self, _u: Ix, _time: Ix) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

pub struct NoOpDfsVisitor;
impl<Ix: crate::graph::IndexType> DfsVisitor<Ix> for NoOpDfsVisitor {}

macro_rules! cf_chain {
    ($first:expr, $second:expr) => {
        match $first {
            std::ops::ControlFlow::Break(()) => std::ops::ControlFlow::Break(()),
            std::ops::ControlFlow::Continue(()) => $second,
        }
    };
}

/// A bundle of two visitors, applied sequentially.
pub struct BundleVisitors<A, B>(pub A, pub B);

impl<Ix: crate::graph::IndexType, A: DfsVisitor<Ix>, B: DfsVisitor<Ix>> DfsVisitor<Ix>
    for BundleVisitors<A, B>
{
    fn start_vertex(&mut self, s: Ix) -> ControlFlow<()> {
        cf_chain!(self.0.start_vertex(s), self.1.start_vertex(s))
    }
    fn discover_vertex(&mut self, u: Ix, t: Ix, p: Ix) -> ControlFlow<()> {
        cf_chain!(
            self.0.discover_vertex(u, t, p),
            self.1.discover_vertex(u, t, p)
        )
    }
    fn examine_edge(&mut self, u: Ix, v: Ix) -> ControlFlow<()> {
        cf_chain!(self.0.examine_edge(u, v), self.1.examine_edge(u, v))
    }
    fn tree_edge(&mut self, u: Ix, v: Ix) -> ControlFlow<()> {
        cf_chain!(self.0.tree_edge(u, v), self.1.tree_edge(u, v))
    }
    fn back_edge(&mut self, u: Ix, v: Ix) -> ControlFlow<()> {
        cf_chain!(self.0.back_edge(u, v), self.1.back_edge(u, v))
    }
    fn forward_or_cross_edge(&mut self, u: Ix, v: Ix) -> ControlFlow<()> {
        cf_chain!(
            self.0.forward_or_cross_edge(u, v),
            self.1.forward_or_cross_edge(u, v)
        )
    }
    fn finish_vertex(&mut self, u: Ix, t: Ix) -> ControlFlow<()> {
        cf_chain!(self.0.finish_vertex(u, t), self.1.finish_vertex(u, t))
    }
}

/// For the computation of DFS with visitor pattern on a [`GraphMemoryMap`] instance.
///
/// It follows arcs out of nodes, on directed graphs: pass it a [`view`] of the graph to follow
/// them in another [`Direction`].
///
/// [`view`]: ../../graph/struct.GraphMemoryMap.html#method.view
/// [`Direction`]: ../../graph/direction/enum.Direction.html
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct DFSVisit<'a, G: Graph> {
    /// Graph in which the DFS is to be performed.
    g: &'a G,
    /// DFS source node
    source: usize,
    /// Memmapped slice containing the nodes' parents.
    parents: AbstractedProceduralMemoryMut<G::Ix>,
    /// Visit functions.
    visit: &'a mut dyn DfsVisitor<G::Ix>,
}

#[allow(dead_code)]
impl<'a, G: Graph> DFSVisit<'a, G> {
    pub fn new(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn DfsVisitor<G::Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bfs = Self::new_no_compute(g, source, visitors)?;
        let proc_mem = bfs.init_cache_mem()?;

        bfs.compute_with_proc_mem(proc_mem)?;

        Ok(bfs)
    }

    /// Returns all nodes' parents as determined by the DFS with the given source.
    pub fn get_parents(&'a self) -> &'a [G::Ix] {
        self.parents.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.cache_file_name(CacheFile::BFS, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, G: Graph> DFSVisit<'a, G> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn DfsVisitor<G::Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source, visitors)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn DfsVisitor<G::Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source, visitors)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryDFS, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<ProceduralMemoryDFS, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryDFS,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryDFS,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.w()
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.cache_file_name(file_type, seq)
    }

    fn new_no_compute_impl(
        g: &'a G,
        source: usize,
        visit: &'a mut dyn DfsVisitor<G::Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source >= g.s() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                g.s()
            )
            .into());
        }
        let out_fn = g.cache_file_name(CacheFile::DFS, None)?;
        Ok(Self {
            g,
            source,
            parents: SharedSliceMut::<G::Ix>::abst_mem_mut(&out_fn, g.s(), true)?,
            visit,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryDFS, Box<dyn std::error::Error>> {
        let node_count = self.g.s();

        let v_fn = self.build_cache_filename(CacheFile::DFS, Some(0))?;
        let f_fn = self.build_cache_filename(CacheFile::DFS, Some(1))?;

        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;
        let finished = SharedSliceMut::<bool>::abst_mem_mut(&f_fn, node_count, true)?;

        Ok((visited, finished))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryDFS,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut visited, mut finished) = proc_mem;
        let mut par = self.parents.shared_slice();

        let source = G::Ix::new(self.source);
        // in undirected graphs, the edge back to a node's parent is the tree edge reversed
        let skip_parent_edge = !self.g.directed();

        // explicit DFS stack: pairs each pending node with an in-place, resumable neighbour
        // iterator (no per-node allocation/copy of its adjacency list).
        let mut next: Vec<(G::Ix, G::Neigh<'_>)> = Vec::with_capacity(self.g.s());

        next.push((source, self.g.neigh(source)));
        *par.get_mut(self.source) = source;
        // mark the source visited up front --- otherwise a back-edge cycling to the source (a
        // perfectly normal case in any non-trivial/cyclic graph) would be mistaken for an
        // undiscovered node and re-processed as a fresh tree edge.
        *visited.get_mut(self.source) = true;
        let mut time = 1usize;

        'outer: while let Some((u, mut neigh)) = next.pop() {
            while let Some(v) = neigh.next() {
                if let ControlFlow::Break(()) = self.visit.examine_edge(u, v) {
                    return Ok(());
                }

                if !*visited.get(v.index()) {
                    if let ControlFlow::Break(()) = self.visit.tree_edge(u, v) {
                        return Ok(());
                    }
                    *visited.get_mut(v.index()) = true;
                    *par.get_mut(v.index()) = u;
                    if let ControlFlow::Break(()) =
                        self.visit.discover_vertex(v, G::Ix::new(time), u)
                    {
                        return Ok(());
                    }
                    time += 1;
                    // resume u later; descend to v now. `u` is only paused, not finished, so we
                    // must skip straight past the "finish u" block below and let the outer loop
                    // pop (and process) `v` next --- a plain inner-loop `continue`/`break` cannot
                    // express this (it would either re-examine u's remaining neighbours before v,
                    // or wrongly fall through to mark u finished while it still has unexamined
                    // neighbours).
                    next.push((u, neigh));
                    next.push((v, self.g.neigh(v)));
                    continue 'outer;
                } else if !finished.get(v.index()) {
                    // back edge (except, in undirected graphs, the tree edge reversed)
                    if !(skip_parent_edge && u != v && *par.get(u.index()) == v) {
                        if let ControlFlow::Break(()) = self.visit.back_edge(u, v) {
                            return Ok(());
                        }
                    }
                } else if let ControlFlow::Break(()) = self.visit.forward_or_cross_edge(u, v) {
                    return Ok(());
                }
            }

            if !finished.get(u.index()) {
                *finished.get_mut(u.index()) = true;
                time += 1;
                if let ControlFlow::Break(()) = self.visit.finish_vertex(u, G::Ix::new(time)) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}
