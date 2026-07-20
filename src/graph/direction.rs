//! Direction: whether a graph is directed, and views of a graph that follow its arcs in a given
//! [`Direction`].
//!
//! A graph is *directed* when some arc `u -> v` has no reverse `v -> u` (counting multiplicity:
//! `u -> v` appearing more times than `v -> u` counts). This is recorded once, when the graph's
//! cache entry is built (see [`GraphMemoryMap::is_directed`]), so that algorithms can check it
//! for free.
//!
//! Algorithms on directed graphs choose a [`Direction`] (outgoing by default) and run on the
//! matching view of the graph: [`GraphMemoryMap::view`]. Views of an undirected graph are the
//! graph itself, so undirected graphs never pay for them.
//!
//! [`GraphMemoryMap::is_directed`]: ../struct.GraphMemoryMap.html#method.is_directed
//! [`GraphMemoryMap::view`]: ../struct.GraphMemoryMap.html#method.view
use super::{GraphMemoryMap, IndexType};

use crossbeam::thread;
use std::sync::atomic::{AtomicBool, Ordering};

/// Which arcs of a directed graph an algorithm follows. On undirected graphs all three are the
/// same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Along arcs: from a node to its out-neighbours. The default.
    #[default]
    Out,
    /// Against arcs: from a node to its in-neighbours (the out-neighbours of the
    /// [transpose](../struct.GraphMemoryMap.html#method.transpose)).
    In,
    /// Both ways: the neighbours of the
    /// [underlying undirected graph](../struct.GraphMemoryMap.html#method.to_undirected).
    Both,
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
impl<N: super::N, E: super::E, Ix: IndexType> GraphMemoryMap<N, E, Ix> {
    /// Whether the graph is directed: whether some arc `u -> v` has no reverse `v -> u`,
    /// counting multiplicity (self-loops are their own reverse).
    ///
    /// Recorded when the graph's cache entry is built, so this is free: MatrixMarket
    /// `symmetric` files are undirected by construction, other inputs are checked once (in
    /// `O(|E| log(max degree))`) while parsing. A `general` file that lists every arc both ways
    /// is undirected. Entries built by earlier versions are checked on first call.
    ///
    /// Algorithms on undirected graphs (triangles, k-trusses, k-cores, clustering coefficients,
    /// Louvain, heat kernel PageRank) refuse directed graphs; run them on
    /// [`to_undirected`](Self::to_undirected) to use them on the underlying undirected graph.
    pub fn is_directed(&self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(&directed) = self.directed.get() {
            return Ok(directed);
        }
        let directed = self.asymmetric_arc()?.is_some();
        let _ = self.directed.set(directed);
        Ok(directed)
    }

    /// An arc `u -> v` without a reverse `v -> u` (with multiplicity), if the graph has any.
    fn asymmetric_arc(&self) -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
        let offsets = unsafe { std::slice::from_raw_parts(self.offsets_ptr(), self.offsets_size()) };
        find_asymmetric_arc(offsets, &self[..], self.thread_num())
    }

    /// Errors (naming an offending arc) if the graph is directed, see [`is_directed`]: `what`
    /// needs an undirected graph, `directed` is what to use instead for directed ones, if
    /// anything. Free for graphs whose direction was recorded: the arc is only looked for when
    /// erroring.
    ///
    /// [`is_directed`]: ./struct.GraphMemoryMap.html#method.is_directed
    pub(crate) fn require_symmetric(
        &self,
        what: &str,
        directed: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_directed()? {
            return Ok(());
        }
        let arc = match self.asymmetric_arc()? {
            Some((u, v)) => format!(": arc {u} -> {v} has no reverse {v} -> {u}"),
            None => String::new(),
        };
        let hint = match directed {
            Some(alternative) => format!("for directed graphs use {alternative}, or run it on"),
            None => "run it on".to_string(),
        };
        Err(format!(
            "error {what} needs an undirected graph, but this one is directed{arc} ({hint} \
             `GraphMemoryMap::to_undirected`, the underlying undirected graph)"
        )
        .into())
    }

    /// The graph whose out-neighbours are this graph's neighbours in `direction`: the graph
    /// itself ([`Direction::Out`]), its [transpose](Self::transpose) ([`Direction::In`]), or
    /// the [underlying undirected graph](Self::to_undirected) ([`Direction::Both`]).
    ///
    /// Node ids are the same in every view, so per-node results of an algorithm run on a view
    /// are per-node results of this graph. Edge ids are not (see the views' documentation).
    /// For undirected graphs every view is the graph itself.
    pub fn view(&self, direction: Direction) -> Result<&Self, Box<dyn std::error::Error>> {
        match direction {
            Direction::Out => Ok(self),
            Direction::In => self.transpose(),
            Direction::Both => self.to_undirected(),
        }
    }

    /// The transpose of the graph: every arc `u -> v` reversed into `v -> u`, so that a node's
    /// out-neighbours are its in-neighbours here. Each edge keeps the label of the arc it
    /// reverses; edges are ordered by node, and by neighbour within a node, as in every graph.
    ///
    /// Built once (in `O(|V| + |E|)`) and cached alongside the graph, which owns it:
    /// [`drop_cache`](Self::drop_cache) removes it too. An undirected graph is its own
    /// transpose.
    pub fn transpose(&self) -> Result<&Self, Box<dyn std::error::Error>> {
        if !self.is_directed()? {
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

    /// The underlying undirected graph: every arc `u -> v` becomes an undirected edge, stored
    /// as `u -> v` and `v -> u` (self-loops once, as MatrixMarket `symmetric` files store
    /// them). Multiplicity is kept: arcs both ways between two nodes, or parallel arcs, become
    /// parallel edges --- which triangle based algorithms count separately. Each edge keeps the
    /// label of the arc it comes from.
    ///
    /// Built once (in `O(|V| + |E|)`, after the [transpose](Self::transpose)) and cached
    /// alongside the graph, which owns it: [`drop_cache`](Self::drop_cache) removes it too. An
    /// undirected graph is its own underlying undirected graph.
    pub fn to_undirected(&self) -> Result<&Self, Box<dyn std::error::Error>> {
        if !self.is_directed()? {
            return Ok(self);
        }
        if let Some(undirected) = self.undirected.get() {
            return Ok(undirected);
        }
        let transpose = self.transpose()?;
        let _building = self
            .derive_lock
            .lock()
            .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
        if let Some(undirected) = self.undirected.get() {
            return Ok(undirected);
        }
        let cache = self.graph_cache.derive_undirected(&transpose.graph_cache)?;
        let undirected = Self::init_from_cache(cache, Some(self.thread_count))?;
        Ok(self.undirected.get_or_init(|| Box::new(undirected)))
    }

    /// Number of arcs into a node: its out-degree in the [transpose](Self::transpose). For
    /// undirected graphs, its degree.
    pub fn in_degree(&self, node_id: usize) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(self.transpose()?.node_degree(node_id))
    }

    /// A node's in-neighbours, sorted: its out-neighbours in the [transpose](Self::transpose).
    /// For undirected graphs, its neighbours.
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

    fn graph(name: &str, header: &str, n: usize, edges: &[(usize, usize)]) -> TestGraph {
        let path = write_mtx(name, header, n, edges);
        let id = format!("direction_{name}_{}", unique_run_tag());
        let graph = GraphMemoryMap::from_file(&path, Some(id), Some(3)).expect("parse should succeed");
        TestGraph::new(graph, path.parent().map(Path::to_path_buf))
    }

    fn lists<N: crate::graph::N, E: crate::graph::E, Ix: IndexType>(
        g: &GraphMemoryMap<N, E, Ix>,
    ) -> Vec<Vec<usize>> {
        (0..g.size()).map(|u| g.neighbours(u).unwrap().collect()).collect()
    }

    /// Random multigraph with self-loops, reciprocal and parallel arcs.
    fn random_arcs(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut arcs: Vec<_> = (0..m).map(|_| (next() % n, next() % n)).collect();
        // some reciprocal pairs, and a duplicated arc
        arcs.extend([(0, 1), (1, 0), (2, 3), (2, 3), (4, 4)]);
        arcs
    }

    #[test]
    fn direction_is_recorded_in_the_metadata() {
        for (name, header, edges, directed) in [
            ("symmetric", "symmetric", vec![(1, 0), (2, 1)], false),
            ("general_directed", "general", vec![(0, 1), (1, 2)], true),
            ("general_both_ways", "general", vec![(0, 1), (1, 0)], false),
            ("general_loops", "general", vec![(0, 0), (2, 2)], false),
        ] {
            let g = graph(name, header, 3, &edges);
            assert_eq!(g.metadata().unwrap().directed, Some(directed), "{name}");
            assert_eq!(g.is_directed().unwrap(), directed, "{name}");
            // and reopened from the cache, without parsing
            let reopened = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, None)
                .unwrap();
            assert_eq!(reopened.directed.get(), Some(&directed), "{name}, reopened");
        }
    }

    #[test]
    fn metadata_without_direction_reads_as_unknown() {
        let mut meta = crate::graph::cache::CacheMetadata::now::<VoidLabel, VoidLabel, usize>(
            "d", "id", 3, 2, 8, false, false, false, 0, 0, None, None,
        );
        assert!(!meta.to_toml().contains("directed"));
        assert_eq!(crate::graph::cache::CacheMetadata::from_toml(&meta.to_toml()).unwrap().directed, None);
        meta.directed = Some(true);
        let parsed = crate::graph::cache::CacheMetadata::from_toml(&meta.to_toml()).unwrap();
        assert_eq!(parsed.directed, Some(true));
    }

    #[test]
    fn transpose_and_undirected_views_match_brute_force() {
        let n = 40;
        let arcs = random_arcs(n, 300, 7);
        let g = graph("random_multi", "general", n, &arcs);
        assert!(g.is_directed().unwrap());

        let mut expected_in = vec![Vec::new(); n];
        let mut expected_undirected = vec![Vec::new(); n];
        for &(u, v) in &arcs {
            expected_in[v].push(u);
            expected_undirected[u].push(v);
            if u != v {
                expected_undirected[v].push(u);
            }
        }
        expected_in.iter_mut().for_each(|l| l.sort());
        expected_undirected.iter_mut().for_each(|l| l.sort());

        let transpose = g.transpose().unwrap();
        assert_eq!(lists(transpose), expected_in);
        assert!(transpose.is_directed().unwrap());
        assert_eq!(transpose.width(), g.width());
        for v in 0..n {
            assert_eq!(g.in_degree(v).unwrap(), expected_in[v].len());
            assert_eq!(g.in_neighbours(v).unwrap().collect::<Vec<_>>(), expected_in[v]);
        }

        let undirected = g.to_undirected().unwrap();
        assert_eq!(lists(undirected), expected_undirected);
        assert!(!undirected.is_directed().unwrap());

        // views
        assert!(std::ptr::eq(g.view(Direction::Out).unwrap(), &*g));
        assert!(std::ptr::eq(g.view(Direction::In).unwrap(), transpose));
        assert!(std::ptr::eq(g.view(Direction::Both).unwrap(), undirected));
        // built once
        assert!(std::ptr::eq(g.transpose().unwrap(), transpose));
    }

    #[test]
    fn views_of_undirected_graphs_are_the_graph_itself() {
        let g = graph("undirected", "symmetric", 4, &[(1, 0), (2, 1), (3, 3)]);
        for direction in [Direction::Out, Direction::In, Direction::Both] {
            assert!(std::ptr::eq(g.view(direction).unwrap(), &*g), "{direction:?}");
        }
        assert_eq!(g.in_degree(1).unwrap(), 2);
    }

    #[test]
    fn derived_graphs_are_reused_and_dropped_with_the_graph() {
        let path = write_mtx("reuse", "general", 3, &[(0, 1), (1, 2), (2, 0), (0, 2)]);
        let id = format!("direction_reuse_{}", unique_run_tag());
        let mut g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2))
            .unwrap();
        let expected = (lists(g.transpose().unwrap()), lists(g.to_undirected().unwrap()));
        let derived_files = |g: &GraphMemoryMap<VoidLabel, VoidLabel, usize>| -> Vec<PathBuf> {
            ["transpose", "undirected"]
                .iter()
                .flat_map(|kind| {
                    let hashed = crate::graph::cache::utils::id_from_filename(
                        &g.graph_cache.derived_id(kind).unwrap(),
                    )
                    .unwrap();
                    glob::glob(&format!("{}*{hashed}.*", crate::graph::cache::utils::CACHE_DIR))
                        .unwrap()
                        .flatten()
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        assert!(!derived_files(&g).is_empty());

        // a fresh instance finds them on disk
        let reopened = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, Some(2))
            .unwrap();
        assert_eq!(
            (lists(reopened.transpose().unwrap()), lists(reopened.to_undirected().unwrap())),
            expected
        );

        g.drop_cache().unwrap();
        let g = reopened;
        assert!(derived_files(&g).is_empty(), "derived graphs left behind");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Rebuilding a graph's entry (e.g. parsing a changed dataset under the same id) discards
    /// the graphs derived from the old one.
    #[test]
    fn rebuilt_graphs_do_not_reuse_stale_derived_graphs() {
        let id = format!("direction_rebuilt_{}", unique_run_tag());
        let first = write_mtx("rebuilt_a", "general", 3, &[(0, 1), (1, 2)]);
        let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&first, Some(id.clone()), Some(2))
            .unwrap();
        assert_eq!(lists(g.transpose().unwrap()), vec![vec![], vec![0], vec![1]]);
        drop(g);

        let second = write_mtx("rebuilt_b", "general", 3, &[(2, 0), (2, 1)]);
        let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&second, Some(id), Some(2))
            .unwrap();
        assert_eq!(lists(g.transpose().unwrap()), vec![vec![2], vec![2], vec![]]);
        assert_eq!(lists(g.to_undirected().unwrap()), vec![vec![2], vec![2], vec![0, 1]]);
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
        let id = format!("direction_labels_{}", unique_run_tag());
        let g = GraphMemoryMap::<VoidLabel, Weight, usize>::from_file(&path, Some(id), Some(2)).unwrap();
        let g = TestGraph::new(g, Some(dir));

        let labelled = |g: &GraphMemoryMap<VoidLabel, Weight, usize>| -> BTreeMap<usize, Vec<(usize, usize)>> {
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
        };
        assert_eq!(
            labelled(g.transpose().unwrap()),
            BTreeMap::from([
                (0, vec![(1, 30)]),
                (1, vec![(0, 10)]),
                (2, vec![(0, 20), (2, 40)]),
            ])
        );
        assert_eq!(
            labelled(g.to_undirected().unwrap()),
            BTreeMap::from([
                (0, vec![(1, 10), (1, 30), (2, 20)]),
                (1, vec![(0, 30), (0, 10)]),
                (2, vec![(0, 20), (2, 40)]),
            ])
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

    fn path_graph() -> TestGraph {
        let dir = unique_test_dir("direction_algorithms");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("path.mtx");
        std::fs::write(&path, "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n").unwrap();
        let id = format!("direction_algorithms_{}", unique_run_tag());
        TestGraph::new(GraphMemoryMap::from_file(&path, Some(id), Some(2)).unwrap(), Some(dir))
    }

    fn close(got: &[f64], want: &[f64], tolerance: f64) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tolerance)
    }

    /// On 0 -> 1 -> 2: outgoing distances are from a node, incoming ones to it (networkx's
    /// convention), both ignore directions.
    #[test]
    fn traversals_and_centralities_follow_the_chosen_direction() {
        let g = path_graph();
        let distances = |source, direction| -> Vec<usize> {
            let bfs = BFSDists::new_directed(&g, source, direction).unwrap();
            bfs.get_distances().to_vec()
        };
        // unreached nodes are at distance 0, like the source
        assert_eq!(distances(0, Direction::Out), vec![0, 1, 2]);
        assert_eq!(distances(2, Direction::In), vec![2, 1, 0]);
        assert_eq!(distances(2, Direction::Both), vec![2, 1, 0]);

        let closeness = |direction| -> Vec<f64> {
            let c = ExactClosenessCentrality::<_, _, _, f64>::new_directed(&g, Some(false), direction).unwrap();
            (0..3).map(|v| c.get_node_centrality(v)).collect()
        };
        // (n - 1) / total distance to the nodes reached
        assert!(close(&closeness(Direction::Out), &[2. / 3., 2., 0.], 1e-12));
        assert!(close(&closeness(Direction::In), &[0., 2., 2. / 3.], 1e-12));
        assert!(close(&closeness(Direction::Both), &[2. / 3., 1., 2. / 3.], 1e-12));

        let harmonic = |direction| -> Vec<f64> {
            let c = ExactHarmonicCentrality::<_, _, _, f64>::new_directed(&g, None, direction).unwrap();
            (0..3).map(|v| c.get_node_centrality(v)).collect()
        };
        // networkx's harmonic_centrality (incoming): [0, 1, 1.5]
        assert!(close(&harmonic(Direction::In), &[0., 1., 1.5], 1e-12));
        assert!(close(&harmonic(Direction::Out), &[1.5, 1., 0.], 1e-12));

        // HyperBall's estimates agree (exactly enough, on 3 nodes)
        let mut in_ball = HyperBall8::new_directed(&g, Direction::In).unwrap();
        let estimated = in_ball.get_or_compute_closeness_centrality(true).unwrap().to_vec();
        let exact = ExactClosenessCentrality::<_, _, _, f64>::new_directed(&g, Some(true), Direction::In).unwrap();
        let exact: Vec<f64> = (0..3).map(|v| exact.get_node_centrality(v)).collect();
        assert!(close(&estimated, &exact, 0.05), "{estimated:?} vs {exact:?}");
    }

    #[test]
    fn hierholzer_needs_balanced_directed_graphs() {
        use crate::trails::hierholzer::AlgoHierholzer;
        let graph = |name: &str, n: usize, arcs: &[(usize, usize)]| -> TestGraph {
            let dir = unique_test_dir("direction_hierholzer");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.mtx"));
            let mut text = format!("%%MatrixMarket matrix coordinate pattern general\n{n} {n} {}\n", arcs.len());
            for (u, v) in arcs {
                text += &format!("{} {}\n", u + 1, v + 1);
            }
            std::fs::write(&path, text).unwrap();
            let id = format!("direction_hierholzer_{name}_{}", unique_run_tag());
            TestGraph::new(GraphMemoryMap::from_file(&path, Some(id), Some(2)).unwrap(), Some(dir))
        };

        let unbalanced = graph("path", 3, &[(0, 1), (1, 2)]);
        let err = AlgoHierholzer::new(&unbalanced).err().expect("a path has no closed trail");
        assert!(err.to_string().contains("node 0 has 0 in and 1 out"), "{err}");

        // two cycles sharing node 1: one closed trail over all 5 arcs
        let eulerian = graph("cycles", 4, &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 1)]);
        let euler = AlgoHierholzer::new(&eulerian).unwrap();
        assert_eq!(euler.trail_number(), 1);
        let trail: Vec<usize> = euler.get_trail(0).unwrap().iter().map(|v| v.index()).collect();
        // closed: each node is followed by an out-neighbour, the last by the first
        let mut arcs: Vec<(usize, usize)> = (0..trail.len()).map(|i| (trail[i], trail[(i + 1) % trail.len()])).collect();
        arcs.sort();
        assert_eq!(arcs, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 1)]);
    }

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
        let back_edges = |header: &str, arcs: &str| -> Vec<(usize, usize)> {
            let dir = unique_test_dir("direction_dfs_visit");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("g.mtx");
            std::fs::write(&path, format!("%%MatrixMarket matrix coordinate pattern {header}\n{arcs}")).unwrap();
            let id = format!("direction_dfs_visit_{}", unique_run_tag());
            let g: TestGraph = TestGraph::new(GraphMemoryMap::from_file(&path, Some(id), Some(2)).unwrap(), Some(dir));
            let mut visitor = BackEdges::default();
            DFSVisit::new(&*g, 0, &mut visitor).unwrap();
            visitor.0
        };
        // 0 <-> 1 in a directed graph (1 -> 2 has no reverse): 1 -> 0 closes a cycle
        assert_eq!(back_edges("general", "3 3 3\n1 2\n2 1\n2 3\n"), vec![(1, 0)]);
        // the same edge undirected is just the tree edge
        assert_eq!(back_edges("symmetric", "2 2 1\n2 1\n"), vec![]);
    }
}
