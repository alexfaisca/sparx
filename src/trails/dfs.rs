use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;

type ProceduralMemoryDFS = (AbstractedProceduralMemoryMut<bool>,);

/// For the computation of DFS in a [`GraphMemoryMap`] instance, recording the order at
/// which each node is found and the total number of reacheable nodes.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct DFS<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph in which the DFS is to be performed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Source node for DFS.
    source: usize,
    /// Memmapped slice containing the nodes in discovery order.
    order: AbstractedProceduralMemoryMut<Ix>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DFS<'a, N, E, Ix> {
    /// Performs DFS in a [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bfs = Self::new_no_compute(g, source)?;
        let proc_mem = bfs.init_cache_mem()?;

        bfs.compute_with_proc_mem(proc_mem)?;

        Ok(bfs)
    }

    /// Returns all nodes' discovery order as determined by the DFS with the given source.
    pub fn get_order(&'a self) -> &'a [Ix] {
        self.order.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::BFS, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DFS<'a, N, E, Ix> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source)
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
        self.g.width()
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source >= g.size() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                g.size()
            )
            .into());
        }
        let out_fn = g.build_cache_filename(CacheFile::DFS, None)?;
        Ok(Self {
            g,
            source,
            order: SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryDFS, Box<dyn std::error::Error>> {
        let node_count = self.g.size();

        let v_fn = self.build_cache_filename(CacheFile::DFS, Some(1))?;

        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;

        Ok((visited,))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryDFS,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut visited,) = proc_mem;
        let mut ord = self.order.shared_slice();

        let mut next = Vec::with_capacity(self.g.size());

        next.push(self.source);
        // the loop below marks a node visited and writes it into `ord` the first time it's
        // popped; pre-seeding `ord[0]`/pre-pushing the source here while *also* letting the loop
        // process it as its first iteration would write the source twice (once here, once as the
        // loop's first pop, since it isn't marked visited yet) and overflow `ord` by one slot for
        // any graph where every node is reachable from `source` --- so the source must only be
        // pushed onto the stack, not separately pre-written.
        let mut idx = 0;

        // find cycles until no unused edges remain
        while let Some(u) = next.pop() {
            if *visited.get(u) {
                continue;
            }
            *visited.get_mut(u) = true;
            *ord.get_mut(idx) = Ix::new(u);
            idx += 1;
            for v in self.g.neighbours(u)?.rev() {
                if !*visited.get(v) {
                    next.push(v);
                }
            }
        }

        // cleanup cache
        self.g.cleanup_cache(CacheFile::DFS)?;

        Ok(())
    }
}
