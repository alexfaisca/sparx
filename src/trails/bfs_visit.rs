use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;
use std::ops::ControlFlow;

type ProceduralMemoryBFSParents = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<bool>,
);

// ----- Minimal graph trait -----
pub trait Graph {
    fn s(&self) -> usize;
    fn w(&self) -> usize;
    /// Iterator type returned by [`neigh`](Graph::neigh); borrows no allocation --- implementors
    /// are expected to walk the graph's adjacency representation directly, in place.
    type Neigh<'a>: Iterator<Item = usize>
    where
        Self: 'a;
    fn neigh(&self, u: usize) -> Self::Neigh<'_>;
    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>>;
    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>>;
}

/// BFSVisitor methods have defaults (no-ops), so users can override only what they need.
/// Returning `ControlFlow::Break(())` from any hook stops the traversal early.
pub trait BfsVisitor {
    #[inline]
    fn start_vertex(&mut self, _s: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn discover_vertex(&mut self, _u: usize, _dist: usize, _parent: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn examine_edge(&mut self, _u: usize, _v: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn tree_edge(&mut self, _u: usize, _v: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn non_tree_edge(&mut self, _u: usize, _v: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    #[inline]
    fn finish_vertex(&mut self, _u: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A no-op visitor (handy default)
pub struct NoOpBFSVisitor;
impl BfsVisitor for NoOpBFSVisitor {}

macro_rules! cf_chain {
    ($first:expr, $second:expr) => {
        match $first {
            std::ops::ControlFlow::Break(()) => std::ops::ControlFlow::Break(()),
            std::ops::ControlFlow::Continue(()) => $second,
        }
    };
}

/// A composite visitor for infinite customization :)
pub struct BundleVisitor<A, B>(A, B);
impl<A: BfsVisitor, B: BfsVisitor> BfsVisitor for BundleVisitor<A, B> {
    fn start_vertex(&mut self, s: usize) -> ControlFlow<()> {
        cf_chain!(self.0.start_vertex(s), self.1.start_vertex(s))
    }
    fn discover_vertex(&mut self, u: usize, d: usize, p: usize) -> ControlFlow<()> {
        cf_chain!(
            self.0.discover_vertex(u, d, p),
            self.1.discover_vertex(u, d, p)
        )
    }
    fn examine_edge(&mut self, u: usize, v: usize) -> ControlFlow<()> {
        cf_chain!(self.0.examine_edge(u, v), self.1.examine_edge(u, v))
    }
    fn tree_edge(&mut self, u: usize, v: usize) -> ControlFlow<()> {
        cf_chain!(self.0.tree_edge(u, v), self.1.tree_edge(u, v))
    }
    fn non_tree_edge(&mut self, u: usize, v: usize) -> ControlFlow<()> {
        cf_chain!(self.0.non_tree_edge(u, v), self.1.non_tree_edge(u, v))
    }
    fn finish_vertex(&mut self, u: usize) -> ControlFlow<()> {
        cf_chain!(self.0.finish_vertex(u), self.1.finish_vertex(u))
    }
}

/// For the computation of BFS with visitor pattern on a [`GraphMemoryMap`] instance.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct BFSVisit<'a, G: Graph> {
    /// Graph in which the BFS is to be performed.
    g: &'a G,
    /// BFS source node.
    source: usize,
    /// Memmapped slice containing each nodes' distance to the source node.
    distances: AbstractedProceduralMemoryMut<usize>,
    /// Visit functions.
    visit: &'a mut dyn BfsVisitor,
}

#[allow(dead_code)]
impl<'a, G: Graph> BFSVisit<'a, G> {
    /// Performs BFS with visitor pattern on a [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn BfsVisitor,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bfs = Self::new_no_compute(g, source, visitors)?;
        let proc_mem = bfs.init_cache_mem()?;

        bfs.compute_with_proc_mem(proc_mem)?;

        Ok(bfs)
    }

    /// Returns all nodes' distance to the source node as determined by the BFS.
    pub fn get_distances(&'a self) -> &'a [usize] {
        self.distances.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::BFS, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, G: Graph> BFSVisit<'a, G> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn BfsVisitor,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source, visitors)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a G,
        source: usize,
        visitors: &'a mut dyn BfsVisitor,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source, visitors)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryBFSParents, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBFSParents, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSParents,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSParents,
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
        visit: &'a mut dyn BfsVisitor,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source >= g.s() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                g.s()
            )
            .into());
        }
        let out_fn = g.cache_file_name(CacheFile::BFS, None)?;
        Ok(Self {
            g,
            source,
            distances: SharedSliceMut::<usize>::abst_mem_mut(&out_fn, g.s(), true)?,
            visit,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<ProceduralMemoryBFSParents, Box<dyn std::error::Error>> {
        let node_count = self.g.s();

        let q_fn = self.build_cache_filename(CacheFile::BFS, Some(0))?;
        let v_fn = self.build_cache_filename(CacheFile::BFS, Some(1))?;

        let queue = SharedSliceMut::<usize>::abst_mem_mut(&q_fn, node_count, true)?;
        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;

        Ok((queue, visited))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryBFSParents,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (queue, mut visited) = proc_mem;
        let mut next = SharedQueueMut::from_shared_slice(queue.shared_slice());

        // Start vertex
        if let ControlFlow::Break(()) = self.visit.start_vertex(self.source) {
            return Ok(());
        }
        next.push(self.source);
        // mark the source visited up front --- otherwise a back-edge cycling to the source (a
        // perfectly normal case in any non-trivial/cyclic graph) would be mistaken for an
        // undiscovered node and re-processed as a fresh tree edge.
        *visited.get_mut(self.source) = true;

        while let Some(u) = next.pop() {
            let dist_u = *self.distances.get(u);
            for v in self.g.neigh(u) {
                if let ControlFlow::Break(()) = self.visit.examine_edge(u, v) {
                    return Ok(());
                }

                if !*visited.get(v) {
                    if let ControlFlow::Break(()) = self.visit.tree_edge(u, v) {
                        return Ok(());
                    }
                    *visited.get_mut(v) = true;
                    *self.distances.get_mut(v) = dist_u + 1;
                    if let ControlFlow::Break(()) =
                        self.visit.discover_vertex(v, *self.distances.get(v), u)
                    {
                        return Ok(());
                    }
                    next.push(v);
                } else if let ControlFlow::Break(()) = self.visit.non_tree_edge(u, v) {
                    return Ok(());
                }
            }
            if let ControlFlow::Break(()) = self.visit.finish_vertex(u) {
                return Ok(());
            }
        }

        // cleanup cache
        self.g.cleanup(CacheFile::BFS)?;

        Ok(())
    }
}
