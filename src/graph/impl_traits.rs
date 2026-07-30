use crate::trails::{bfs_visit, dfs_visit};

use super::*;

/// Widens a `NeighbourIter<N, E, Ix>`'s native `usize` output back into `Ix` --- `NeighbourIter`
/// itself stays `usize`-yielding (it's `GraphMemoryMap::neighbours`'s pervasively-used public
/// return type, far too widely relied upon crate-wide to change), so `bfs_visit`/`dfs_visit`'s
/// `Graph::neigh` implementations adapt it locally instead, letting `BFSVisit`/`DFSVisit` and
/// their visitor callbacks stay natively `Ix`-typed throughout.
#[inline(always)]
fn widen_to_ix<Ix: crate::graph::IndexType>(x: usize) -> Ix {
    Ix::new(x)
}

type IxNeighbours<N, E, Ix> = std::iter::Map<NeighbourIter<N, E, Ix>, fn(usize) -> Ix>;

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: crate::graph::Directedness> dfs_visit::Graph
    for GraphMemoryMap<N, E, Ix, Dir>
{
    type Ix = Ix;

    fn directed(&self) -> bool {
        // in a directed graph an arc back to a node's parent closes a cycle
        Dir::DIRECTED
    }

    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    type Neigh<'a>
        = IxNeighbours<N, E, Ix>
    where
        Self: 'a;

    /// Correctness note: this *must* go through [`GraphMemoryMap::neighbours`] (the per-node
    /// offset-range lookup), not a raw `self[u..u + 1]` edge-array slice --- `u` here is a node
    /// id, not a raw position in the flat edges array, and the two only coincide by chance for a
    /// node whose neighbour range happens to start at edge-array position `u`.
    fn neigh(&self, u: Ix) -> Self::Neigh<'_> {
        let u = u.index();
        self.neighbours(u)
            .unwrap_or_else(|e| panic!("error getting neighbours of node {u}: {e}"))
            .map(widen_to_ix::<Ix>)
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: crate::graph::Directedness> bfs_visit::Graph
    for GraphMemoryMap<N, E, Ix, Dir>
{
    type Ix = Ix;

    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    type Neigh<'a>
        = IxNeighbours<N, E, Ix>
    where
        Self: 'a;

    /// Correctness note: this *must* go through [`GraphMemoryMap::neighbours`] (the per-node
    /// offset-range lookup), not a raw `self[u..u + 1]` edge-array slice --- `u` here is a node
    /// id, not a raw position in the flat edges array, and the two only coincide by chance for a
    /// node whose neighbour range happens to start at edge-array position `u`.
    fn neigh(&self, u: Ix) -> Self::Neigh<'_> {
        let u = u.index();
        self.neighbours(u)
            .unwrap_or_else(|e| panic!("error getting neighbours of node {u}: {e}"))
            .map(widen_to_ix::<Ix>)
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
    use crate::graph::test_utils::unique_test_dir;
    use crate::trails::{
        bfs_visit::{BFSVisit, BfsVisitor},
        dfs_visit::{DFSVisit, DfsVisitor, NoOpDfsVisitor},
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
    /// (with a guard removing its cached files)
    fn build_tiny_graph() -> (
        crate::graph::DiGraph<VoidLabel, VoidLabel, usize>,
        crate::graph::test_utils::CacheGuard,
    ) {
        let dir = unique_test_dir("impl_traits_neigh_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("tiny.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern general\n4 4 5\n1 2\n1 3\n1 4\n2 3\n4 1\n",
        )
        .expect("error writing tiny mtx fixture");

        let cache = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph = crate::graph::DiGraph::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        let guard = crate::graph::test_utils::CacheGuard::new(&graph, None);
        (graph, guard)
    }

    #[test]
    fn neigh_matches_real_neighbours_for_every_node() {
        let (graph, _cache) = build_tiny_graph();
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

    impl<Ix: IndexType> DfsVisitor<Ix> for RecordingVisitor {
        fn discover_vertex(&mut self, u: Ix, _time: Ix, _parent: Ix) -> ControlFlow<()> {
            self.discovered.lock().unwrap().push(u.index());
            ControlFlow::Continue(())
        }
    }

    impl<Ix: IndexType> BfsVisitor<Ix> for RecordingVisitor {
        fn discover_vertex(&mut self, u: Ix, _dist: Ix, _parent: Ix) -> ControlFlow<()> {
            self.discovered.lock().unwrap().push(u.index());
            ControlFlow::Continue(())
        }
    }

    #[test]
    fn dfs_visit_reaches_every_node_from_source_0() {
        let (graph, _cache) = build_tiny_graph();
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
        let (graph, _cache) = build_tiny_graph();
        let discovered_log = Arc::new(Mutex::new(vec![0]));
        let mut visitor = RecordingVisitor {
            discovered: discovered_log.clone(),
        };
        BFSVisit::new(&graph, 0, &mut visitor).expect("bfs should succeed");

        let mut discovered = discovered_log.lock().unwrap().clone();
        discovered.sort_unstable();
        assert_eq!(discovered, vec![0, 1, 2, 3], "BFS should discover every node reachable from 0");
    }

    /// A strict chain (0 -> 1 -> 2 -> 3), where node 0 is *not* directly adjacent to 2 or 3.
    /// Unlike `build_tiny_graph` (where source 0 happens to be adjacent to every other node,
    /// making a "parent is always the source" bug indistinguishable from a correct result by a
    /// mere adjacency check), this fixture forces a unique, decisive expected parent chain:
    /// `parents == [0, 0, 1, 2]`.
    /// (with a guard removing its cached files)
    fn build_chain_graph() -> (
        crate::graph::DiGraph<VoidLabel, VoidLabel, usize>,
        crate::graph::test_utils::CacheGuard,
    ) {
        let dir = unique_test_dir("impl_traits_chain_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("chain.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern general\n4 4 3\n1 2\n2 3\n3 4\n",
        )
        .expect("error writing chain mtx fixture");

        let cache = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(&path, None, None, None)
            .expect("parse should succeed");
        let graph = crate::graph::DiGraph::init_from_cache(cache, Some(1)).expect("load should succeed");
        std::fs::remove_dir_all(&dir).ok();
        let guard = crate::graph::test_utils::CacheGuard::new(&graph, None);
        (graph, guard)
    }

    /// Regression test for a real bug found while generalizing `DFSVisit` to be `Ix`-generic:
    /// `parents[v]` was never actually written for any node but the source, so `get_parents()`
    /// silently returned all-zeroes for everyone else --- undetected by the existing
    /// `dfs_visit_reaches_every_node_from_source_0` test purely because that test's fixture
    /// happens to have the source adjacent to every other node.
    #[test]
    fn dfs_visit_records_the_exact_parent_chain_on_a_graph_where_source_is_not_adjacent_to_all() {
        let (graph, _cache) = build_chain_graph();
        let mut visitor = NoOpDfsVisitor;
        let dfs = DFSVisit::new(&graph, 0, &mut visitor).expect("dfs should succeed");

        assert_eq!(
            dfs.get_parents(),
            &[0, 0, 1, 2],
            "expected the exact chain parent assignment --- a regression here would mean \
             parents[v] is only ever written for the source node again"
        );
    }
}
