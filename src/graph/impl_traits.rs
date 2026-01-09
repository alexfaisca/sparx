use crate::trails::{bfs_visit, dfs_visit};

use super::*;

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> dfs_visit::Graph
    for GraphMemoryMap<N, E, Ix>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    type Neigh<'a>
        = NeighbourIter<N, E, Ix>
    where
        Self: 'a;

    /// Correctness note: this *must* go through [`GraphMemoryMap::neighbours`] (the per-node
    /// offset-range lookup), not a raw `self[u..u + 1]` edge-array slice --- `u` here is a node
    /// id, not a raw position in the flat edges array, and the two only coincide by chance for a
    /// node whose neighbour range happens to start at edge-array position `u`.
    fn neigh(&self, u: usize) -> Self::Neigh<'_> {
        self.neighbours(u)
            .unwrap_or_else(|e| panic!("error getting neighbours of node {u}: {e}"))
    }

    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.build_cache_filename(target, seq)
    }

    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.cleanup_cache(target)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> bfs_visit::Graph
    for GraphMemoryMap<N, E, Ix>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    type Neigh<'a>
        = NeighbourIter<N, E, Ix>
    where
        Self: 'a;

    /// Correctness note: this *must* go through [`GraphMemoryMap::neighbours`] (the per-node
    /// offset-range lookup), not a raw `self[u..u + 1]` edge-array slice --- `u` here is a node
    /// id, not a raw position in the flat edges array, and the two only coincide by chance for a
    /// node whose neighbour range happens to start at edge-array position `u`.
    fn neigh(&self, u: usize) -> Self::Neigh<'_> {
        self.neighbours(u)
            .unwrap_or_else(|e| panic!("error getting neighbours of node {u}: {e}"))
    }

    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.build_cache_filename(target, seq)
    }

    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.cleanup_cache(target)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::cache::GraphCache;
    use crate::graph::label::VoidLabel;
    use crate::trails::{
        bfs_visit::{BFSVisit, BfsVisitor},
        dfs_visit::{DFSVisit, DfsVisitor},
    };
    use std::ops::ControlFlow;
    use std::sync::{Arc, Mutex};

    /// Tiny directed graph with deliberately irregular degrees, so that a node's id never
    /// coincides with its neighbour range's starting position in the flat edges array --- this
    /// is exactly the condition under which the historical `self[u..u + 1]` bug (reading a raw
    /// edge-array slot instead of looking up node `u`'s real offset range) would misbehave:
    ///
    /// 0 -> 1, 0 -> 2, 0 -> 3   (node 0 has degree 3)
    /// 1 -> 2                  (node 1 has degree 1)
    /// 2 -> (none)             (node 2 has degree 0, an isolated sink)
    /// 3 -> 0                  (node 3 has degree 1)
    fn build_tiny_graph() -> GraphMemoryMap<VoidLabel, VoidLabel, usize> {
        let dir = std::env::temp_dir().join(format!(
            "sparx_impl_traits_neigh_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("tiny.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern general\n4 4 5\n1 2\n1 3\n1 4\n2 3\n4 1\n",
        )
        .expect("error writing tiny mtx fixture");

        let cache = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph = GraphMemoryMap::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        graph
    }

    #[test]
    fn neigh_matches_real_neighbours_for_every_node() {
        let graph = build_tiny_graph();
        for u in 0..graph.size() {
            let real: Vec<usize> = graph.neighbours(u).unwrap().collect();
            let via_dfs_trait: Vec<usize> = dfs_visit::Graph::neigh(&graph, u).collect();
            let via_bfs_trait: Vec<usize> = bfs_visit::Graph::neigh(&graph, u).collect();
            assert_eq!(real, via_dfs_trait, "dfs_visit::Graph::neigh mismatch for node {u}");
            assert_eq!(real, via_bfs_trait, "bfs_visit::Graph::neigh mismatch for node {u}");
        }
    }

    /// Records discovered-vertex order into a shared, independently-owned log, so test code can
    /// inspect it after the traversal without fighting `DFSVisit`/`BFSVisit`'s (unrelated,
    /// pre-existing) `&'a mut dyn Visitor` borrow, which otherwise stays alive for as long as the
    /// `DFSVisit`/`BFSVisit` value itself does.
    struct RecordingVisitor {
        discovered: Arc<Mutex<Vec<usize>>>,
    }

    impl DfsVisitor for RecordingVisitor {
        fn discover_vertex(&mut self, u: usize, _time: usize, _parent: usize) -> ControlFlow<()> {
            self.discovered.lock().unwrap().push(u);
            ControlFlow::Continue(())
        }
    }

    impl BfsVisitor for RecordingVisitor {
        fn discover_vertex(&mut self, u: usize, _dist: usize, _parent: usize) -> ControlFlow<()> {
            self.discovered.lock().unwrap().push(u);
            ControlFlow::Continue(())
        }
    }

    #[test]
    fn dfs_visit_reaches_every_node_from_source_0() {
        let graph = build_tiny_graph();
        let discovered_log = Arc::new(Mutex::new(vec![0]));
        let mut visitor = RecordingVisitor {
            discovered: discovered_log.clone(),
        };
        let dfs = DFSVisit::new(&graph, 0, &mut visitor).expect("dfs should succeed");

        let parents = dfs.get_parents();
        assert_eq!(parents[0], 0, "source's parent should be itself");
        for v in [1, 2, 3] {
            assert!(
                graph.neighbours(parents[v]).unwrap().any(|n| n == v),
                "node {v}'s recorded parent {} must actually be adjacent to it",
                parents[v]
            );
        }

        let mut discovered = discovered_log.lock().unwrap().clone();
        discovered.sort_unstable();
        assert_eq!(discovered, vec![0, 1, 2, 3], "DFS should discover every node reachable from 0");
    }

    #[test]
    fn bfs_visit_reaches_every_node_from_source_0() {
        let graph = build_tiny_graph();
        let discovered_log = Arc::new(Mutex::new(vec![0]));
        let mut visitor = RecordingVisitor {
            discovered: discovered_log.clone(),
        };
        BFSVisit::new(&graph, 0, &mut visitor).expect("bfs should succeed");

        let mut discovered = discovered_log.lock().unwrap().clone();
        discovered.sort_unstable();
        assert_eq!(discovered, vec![0, 1, 2, 3], "BFS should discover every node reachable from 0");
    }
}
