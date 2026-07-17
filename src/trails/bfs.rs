use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;

type ProceduralMemoryBFSDists<Ix> = (
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<bool>,
);

/// For the computation of a simple BFS in a [`GraphMemoryMap`] instance, recording the distance at
/// which each node is found and the total number of reacheable nodes.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct BFSDists<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph in which the BFS is to be performed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    source: usize,
    /// Memmapped slice containing each nodes' distance to the source node.
    distances: AbstractedProceduralMemoryMut<Ix>,
    /// Number of nodes reachable from the source node, the source node itself included.
    reacheable: usize,
    /// Sum of all nodes' distance to the source node.
    total_distances: f64,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> BFSDists<'a, N, E, Ix> {
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bfs = Self::new_no_compute(g, source)?;
        let proc_mem = bfs.init_cache_mem()?;

        bfs.compute_with_proc_mem(proc_mem)?;

        Ok(bfs)
    }
    pub fn new_t(
        g: &'a GraphMemoryMap<N, E, Ix>,
        source: usize,
        thread: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source >= g.size() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                g.size()
            )
            .into());
        }
        let out_fn = g.build_pers_cache_filename(CacheFile::BFS, Some(thread))?;
        let mut bfs = Self {
            g,
            source,
            distances: SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?,
            reacheable: 0,
            total_distances: 0.,
        };
        let node_count = bfs.g.size();

        let q_fn = bfs.build_cache_filename(CacheFile::BFS, Some(3 * thread))?;
        let v_fn = bfs.build_cache_filename(CacheFile::BFS, Some(3 * thread + 1))?;

        let queue = SharedSliceMut::<Ix>::abst_mem_mut(&q_fn, node_count, true)?;
        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;
        bfs.compute_with_proc_mem((queue, visited))?;
        std::fs::remove_file(q_fn)?;
        std::fs::remove_file(v_fn)?;

        Ok(bfs)
    }

    /// Returns the sum of all nodes' distances to the source node as determined by the BFS.
    pub fn total_distances(&self) -> f64 {
        self.total_distances
    }

    /// Returns the number of reacheable nodes from the source node as determined by the BFS, the
    /// source node itself included.
    pub fn recheable(&self) -> usize {
        self.reacheable
    }

    /// Returns all nodes' distance to the source node as determined by the BFS.
    pub fn get_distances(&self) -> &[Ix] {
        self.distances.as_slice()
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
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> BFSDists<'a, N, E, Ix> {
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
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryBFSDists<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBFSDists<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists<Ix>,
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
        let out_fn = g.build_cache_filename(CacheFile::BFS, None)?;
        Ok(Self {
            g,
            source,
            distances: SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?,
            reacheable: 0,
            total_distances: 0.,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryBFSDists<Ix>, Box<dyn std::error::Error>> {
        let node_count = self.g.size();

        let q_fn = self.build_cache_filename(CacheFile::BFS, Some(0))?;
        let v_fn = self.build_cache_filename(CacheFile::BFS, Some(1))?;

        let queue = SharedSliceMut::<Ix>::abst_mem_mut(&q_fn, node_count, true)?;
        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;

        Ok((queue, visited))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (queue, mut visited) = proc_mem;
        let mut dist = self.distances.shared_slice();

        let mut next = SharedQueueMut::from_shared_slice(queue.shared_slice());
        // the source reaches itself (at distance 0)
        let mut reacheable = 1;
        let mut sum: f64 = 0.;

        next.push(Ix::new(self.source));
        // mark the source visited up front --- otherwise a back-edge cycling to the source (any
        // undirected edge, or cycle, through it) rediscovers it as a fresh node at the cycle's
        // length, corrupting its distance, the distance sum and the reachable count
        *visited.get_mut(self.source) = true;
        *dist.get_mut(self.source) = Ix::new(0);

        // find cycles until no unused edges remain
        while let Some(u) = next.pop() {
            let u = u.index();
            let du = dist.get(u).index();
            let dup1 = (du + 1) as f64;
            for v in self.g.neighbours(u)? {
                if !*visited.get(v) {
                    *visited.get_mut(v) = true;
                    *dist.get_mut(v) = Ix::new(du + 1);
                    sum += dup1;
                    reacheable += 1;
                    next.push(Ix::new(v));
                }
            }
        }

        self.reacheable = reacheable;
        self.total_distances = sum;
        // cleanup cache
        // self.g.cleanup_cache(CacheFile::BFS)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use std::io::Write;

    fn graph_from_mtx(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
    ) -> crate::graph::test_utils::TestGraph<VoidLabel, VoidLabel, usize> {
        let dir = unique_test_dir("bfs_dists_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let graph =
            GraphMemoryMap::from_file(&path, Some(format!("bfs_{name}_{}", unique_run_tag())), None)
                .expect("graph parse should succeed");
        crate::graph::test_utils::TestGraph::new(graph, Some(dir))
    }

    /// On an undirected path every edge leads back to the source: it must keep distance 0 and be
    /// counted (once) as reachable, without inflating the distance sum.
    #[test]
    fn bfs_dists_undirected_path() {
        let g = graph_from_mtx("undirected_path", "symmetric", 4, &[(0, 1), (1, 2), (2, 3)]);
        for (source, distances, total) in [(0, [0, 1, 2, 3], 6.), (1, [1, 0, 1, 2], 4.)] {
            let bfs = BFSDists::new(&g, source).unwrap();
            let got: Vec<usize> = bfs.get_distances().iter().map(|d| d.index()).collect();
            assert_eq!(got, distances, "distances from {source}");
            assert_eq!(bfs.total_distances(), total, "distance sum from {source}");
            assert_eq!(bfs.recheable(), 4, "reachable nodes from {source}");
        }
    }

    /// On a directed path (no way back to the source) only downstream nodes are reachable.
    #[test]
    fn bfs_dists_directed_path() {
        let g = graph_from_mtx("directed_path", "general", 3, &[(0, 1), (1, 2)]);
        let bfs = BFSDists::new(&g, 1).unwrap();
        assert_eq!(bfs.total_distances(), 1.);
        assert_eq!(bfs.recheable(), 2, "node 1 reaches itself and node 2");
        let bfs = BFSDists::new(&g, 2).unwrap();
        assert_eq!(bfs.total_distances(), 0.);
        assert_eq!(bfs.recheable(), 1, "node 2 only reaches itself");
    }
}
