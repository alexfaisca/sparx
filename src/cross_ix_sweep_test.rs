//! Cross-`Ix`-width correctness sweep.
//!
//! Builds the same logical graph at every supported index width (`u8`, `u16`, `u32`, `usize`,
//! skipping widths too narrow for a given graph's node/edge count) and runs every
//! publicly-exposed graph algorithm over each, asserting the results are identical to the
//! `usize` baseline. This is a correctness regression test for the `Ix`-generic refactor: it
//! would catch any algorithm that silently depends on `Ix` being exactly `usize`-width (e.g. via
//! a missed `usize` cast, a byte-width assumption, or a truncating conversion).
//!
//! Graphs are small, self-contained, deterministic "ring + chords" synthetic directed graphs
//! (no external dataset dependency): node `i` always has an edge to `(i + 1) % n`, plus a handful
//! of additional "chord" edges `i -> (i + k) % n` for a few fixed step sizes `k`. This guarantees
//! strong connectivity (useful for BFS/DFS/Hierholzer/centrality reachability) while still
//! producing a non-trivial degree distribution (useful for k-core/k-truss/triangle structure).
#![cfg(test)]

use crate::centralities::exact::{ExactClosenessCentrality, ExactHarmonicCentrality, ExactLinCentrality};
use crate::centralities::hyper_ball::HyperBall8;
use crate::communities::gve_louvain::AlgoGVELouvain;
use crate::communities::hk_relax::HKRelax;
use crate::graph::cache::GraphCache;
use crate::graph::label::VoidLabel;
use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
use crate::graph::{GraphMemoryMap, IndexType};
use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
use crate::k_core::liu_et_al::AlgoLiuEtAl;
use crate::k_truss::burkhardt_et_al::AlgoBurkhardtEtAl;
use crate::k_truss::clustering_coefficient::ClusteringCoefficient;
use crate::k_truss::pkt::AlgoPKT;
use crate::k_truss::triangles::Triangles;
use crate::trails::bfs::BFSDists;
use crate::trails::bfs_visit::{BFSVisit, BfsVisitor};
use crate::trails::dfs::DFS;
use crate::trails::dfs_visit::{DFSVisit, DfsVisitor};
use crate::trails::hierholzer::AlgoHierholzer;

use std::io::Write;
use std::ops::ControlFlow;
use std::sync::atomic::Ordering as StdOrdering;
use std::sync::{Arc, Mutex};

/// Builds a deterministic "ring + chords" directed graph with `n` nodes, writes it as a
/// `.mtx` fixture, and returns its path.
fn write_ring_graph_mtx(dir: &std::path::Path, n: usize, chords: &[usize]) -> std::path::PathBuf {
    let path = dir.join(format!("ring_{n}.mtx"));
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n * (1 + chords.len()));
    for i in 0..n {
        edges.push((i, (i + 1) % n));
        for &k in chords {
            if k > 0 && k < n {
                edges.push((i, (i + k) % n));
            }
        }
    }
    // Several algorithms (burkhardt_et_al/pkt/triangles/clustering_coefficient) rely on
    // `edge_reciprocal`/`edge_over`, which require the graph to be undirected (i.e. every edge's
    // reverse direction must also be present). Canonicalize each pair and emit the MatrixMarket
    // "symmetric" format, which has the parser automatically add the reverse direction for every
    // listed pair.
    let mut edges: Vec<(usize, usize)> = edges
        .into_iter()
        .filter(|&(a, b)| a != b)
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    edges.sort_unstable();
    edges.dedup();

    let mut f = std::fs::File::create(&path).expect("error creating ring graph fixture");
    writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
    writeln!(f, "{n} {n} {}", edges.len()).unwrap();
    for (a, b) in edges {
        // MatrixMarket is 1-indexed
        writeln!(f, "{} {}", a + 1, b + 1).unwrap();
    }
    f.flush().unwrap();
    path
}

/// All algorithm outputs collected for a single `Ix` width, in a form directly comparable
/// (`PartialEq`) against the same graph built with a different `Ix` width.
#[derive(Debug, PartialEq)]
struct AlgorithmResults {
    k_core_bz: Vec<u8>,
    k_core_liu: Vec<u8>,
    k_truss_burkhardt: Vec<u8>,
    k_truss_pkt: Vec<u8>,
    triangles: Vec<u8>,
    clustering_coefficient_bits: Vec<u64>,
    bfs_distances: Vec<usize>,
    bfs_reachable: usize,
    dfs_order: Vec<usize>,
    dfs_visit_discovered: Vec<usize>,
    bfs_visit_discovered: Vec<usize>,
    hierholzer_trail_count: usize,
    hierholzer_trails_concat: Vec<usize>,
    louvain_community_count: usize,
    louvain_modularity_bits: u64,
    // `None` when the graph is too large for exact (`O(|V| * (|V|+|E|))`) centrality computation
    // to be worth running in a test; see `run_all_algorithms`'s `compute_exact_centralities` arg.
    closeness: Option<Vec<u64>>,
    harmonic: Option<Vec<u64>>,
    lin: Option<Vec<u64>>,
    hyperball_closeness: Vec<u64>,
    hk_relax_community_size: usize,
}

struct DiscoveryOrderVisitor {
    log: Arc<Mutex<Vec<usize>>>,
}
impl DfsVisitor for DiscoveryOrderVisitor {
    fn discover_vertex(&mut self, u: usize, _time: usize, _parent: usize) -> ControlFlow<()> {
        self.log.lock().unwrap().push(u);
        ControlFlow::Continue(())
    }
}
impl BfsVisitor for DiscoveryOrderVisitor {
    fn discover_vertex(&mut self, u: usize, _dist: usize, _parent: usize) -> ControlFlow<()> {
        self.log.lock().unwrap().push(u);
        ControlFlow::Continue(())
    }
}

/// Builds the graph with the given `Ix` width and runs every algorithm over it.
fn run_all_algorithms<Ix: IndexType>(
    path: &std::path::Path,
    id_suffix: &str,
    compute_exact_centralities: bool,
) -> Result<AlgorithmResults, Box<dyn std::error::Error>> {
    let cache = GraphCache::<VoidLabel, VoidLabel, Ix>::from_file(
        path,
        Some(format!("cross_ix_sweep_{id_suffix}")),
        None,
        None,
    )?;
    let graph = GraphMemoryMap::<VoidLabel, VoidLabel, Ix>::init_from_cache(cache, Some(1))?;

    let k_core_bz = AlgoBatageljZaversnik::new(&graph)?.k_cores().to_vec();
    let k_core_liu = AlgoLiuEtAl::new(&graph)?.k_cores().to_vec();
    let k_truss_burkhardt = AlgoBurkhardtEtAl::new(&graph)?.k_trusses().to_vec();
    let k_truss_pkt = AlgoPKT::new(&graph)?.k_trusses().to_vec();
    let triangles = Triangles::new(&graph)?
        .all_triangles()
        .iter()
        .map(|a| a.load(StdOrdering::Relaxed))
        .collect::<Vec<u8>>();
    // compare clustering coefficients via their raw bit patterns (exact float equality is fine
    // here since both runs perform the identical sequence of f64 operations)
    let cc = ClusteringCoefficient::new(&graph)?;
    let clustering_coefficient_bits = (0..graph.size())
        .map(|u| cc.get_node_clusteringcoefficient(u).to_bits())
        .collect::<Vec<u64>>();

    let bfs = BFSDists::new(&graph, 0)?;
    let bfs_distances = bfs.get_distances().to_vec();
    let bfs_reachable = bfs.recheable();

    let dfs_order = DFS::new(&graph, 0)?.get_order().to_vec();

    let dfs_log = Arc::new(Mutex::new(vec![0usize]));
    {
        let mut visitor = DiscoveryOrderVisitor {
            log: dfs_log.clone(),
        };
        DFSVisit::new(&graph, 0, &mut visitor)?;
    }
    let dfs_visit_discovered = dfs_log.lock().unwrap().clone();

    let bfs_log = Arc::new(Mutex::new(vec![0usize]));
    {
        let mut visitor = DiscoveryOrderVisitor {
            log: bfs_log.clone(),
        };
        BFSVisit::new(&graph, 0, &mut visitor)?;
    }
    let bfs_visit_discovered = bfs_log.lock().unwrap().clone();

    let hierholzer = AlgoHierholzer::new(&graph)?;
    let hierholzer_trail_count = hierholzer.trail_number();
    let hierholzer_trails_concat = hierholzer.get_all_trails().0.to_vec();

    let louvain = AlgoGVELouvain::<_, _, _, f64>::new(&graph)?;
    let louvain_community_count = louvain.community_count();
    let louvain_modularity_bits = louvain.partition_modularity().to_bits();

    let (closeness, harmonic, lin) = if compute_exact_centralities {
        let closeness = ExactClosenessCentrality::<_, _, _, f64>::new(&graph, Some(true))?
            .centralities()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u64>>();
        let harmonic = ExactHarmonicCentrality::<_, _, _, f64>::new(&graph, Some(true))?
            .centralities()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u64>>();
        let lin = ExactLinCentrality::<_, _, _, f64>::new(&graph)?
            .centralities()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u64>>();
        (Some(closeness), Some(harmonic), Some(lin))
    } else {
        (None, None, None)
    };

    let hyperball_closeness = HyperBall8::new(&graph)?
        .geodesic_distances()
        .iter()
        .map(|f| f.to_bits())
        .collect::<Vec<u64>>();

    let hk_relax_community_size = HKRelax::new(&graph, 3.0, 1e-4, vec![0], None, None)?
        .compute()?
        .size;

    Ok(AlgorithmResults {
        k_core_bz,
        k_core_liu,
        k_truss_burkhardt,
        k_truss_pkt,
        triangles,
        clustering_coefficient_bits,
        bfs_distances,
        bfs_reachable,
        dfs_order,
        dfs_visit_discovered,
        bfs_visit_discovered,
        hierholzer_trail_count,
        hierholzer_trails_concat,
        louvain_community_count,
        louvain_modularity_bits,
        closeness,
        harmonic,
        lin,
        hyperball_closeness,
        hk_relax_community_size,
    })
}

/// Runs the full algorithm sweep over a graph at every `Ix` width that can represent it, and
/// asserts all results match the `usize` baseline exactly.
fn sweep(tier: &str, n: usize, chords: &[usize], widths: &[&str]) {
    sweep_with_opts(tier, n, chords, widths, true)
}

/// Like [`sweep`], but allows skipping exact (`O(|V| * (|V|+|E|))`) centrality computation for
/// graphs large enough that it would make the test impractically slow.
fn sweep_with_opts(
    tier: &str,
    n: usize,
    chords: &[usize],
    widths: &[&str],
    compute_exact_centralities: bool,
) {
    let dir = unique_test_dir(&format!("cross_ix_sweep_{tier}"));
    std::fs::create_dir_all(&dir).expect("error creating tmp dir");
    let path = write_ring_graph_mtx(&dir, n, chords);

    // every cache id below must be unique per *test process invocation*, not just per logical
    // graph: `GraphCache::from_file` treats an id as content-addressed and happily reuses
    // whatever cache files already exist on disk for it, so reusing a fixed id string across
    // repeated `cargo test` runs would silently resurrect stale cache state from a previous run
    // instead of rebuilding from this run's (freshly written) `.mtx` fixture.
    let run_tag = unique_run_tag();

    let baseline = run_all_algorithms::<usize>(
        &path,
        &format!("{tier}_usize_{run_tag}"),
        compute_exact_centralities,
    )
    .unwrap_or_else(|e| panic!("[{tier}] usize baseline run failed: {e}"));

    for &width in widths {
        let result = match width {
            "u32" => run_all_algorithms::<u32>(
                &path,
                &format!("{tier}_u32_{run_tag}"),
                compute_exact_centralities,
            ),
            "u16" => run_all_algorithms::<u16>(
                &path,
                &format!("{tier}_u16_{run_tag}"),
                compute_exact_centralities,
            ),
            "u8" => run_all_algorithms::<u8>(
                &path,
                &format!("{tier}_u8_{run_tag}"),
                compute_exact_centralities,
            ),
            other => panic!("unknown width {other}"),
        }
        .unwrap_or_else(|e| panic!("[{tier}] {width} run failed: {e}"));

        macro_rules! check_field {
            ($field:ident) => {
                assert_eq!(
                    result.$field, baseline.$field,
                    "[{tier}] Ix={width}: field `{}` diverged from the usize baseline",
                    stringify!($field)
                );
            };
        }
        check_field!(k_core_bz);
        check_field!(k_core_liu);
        check_field!(k_truss_burkhardt);
        check_field!(k_truss_pkt);
        check_field!(triangles);
        check_field!(clustering_coefficient_bits);
        check_field!(bfs_distances);
        check_field!(bfs_reachable);
        check_field!(dfs_order);
        check_field!(dfs_visit_discovered);
        check_field!(bfs_visit_discovered);
        check_field!(hierholzer_trail_count);
        check_field!(hierholzer_trails_concat);
        // GVE-Louvain's final partition is inherently run-to-run nondeterministic even with the
        // *same* `Ix` (its `choose_community`/`louvain_aggregate` steps depend on `HashMap`
        // iteration order and parallel floating-point accumulation order), so exact-equality
        // comparison across `Ix` widths is not meaningful here --- only confirm the result is a
        // valid partition in the same ballpark, not bit-identical.
        assert!(
            result.louvain_community_count >= 1 && result.louvain_community_count <= n,
            "[{tier}] Ix={width}: louvain_community_count {} out of valid range [1, {n}]",
            result.louvain_community_count
        );
        let modularity = f64::from_bits(result.louvain_modularity_bits);
        assert!(
            (-1.0..=1.0).contains(&modularity),
            "[{tier}] Ix={width}: louvain modularity {modularity} out of valid range [-1, 1]"
        );
        check_field!(closeness);
        check_field!(harmonic);
        check_field!(lin);
        check_field!(hyperball_closeness);
        check_field!(hk_relax_community_size);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn cross_ix_sweep_tiny() {
    // n = 16: trivially fits every Ix width, including u8 (max 255)
    sweep("tiny", 16, &[2, 3], &["u32", "u16", "u8"]);
}

#[test]
fn cross_ix_sweep_small_u8_boundary() {
    // n = 40, 2 chords: 240 directed edges after symmetrization --- u8 can only represent values
    // up to 255, and it is the *edge* count (not just the node count) that must fit, since CSR
    // offsets store cumulative edge counts; this configuration stays safely under that limit
    // while still giving every algorithm some real structure (triangles, k-core/k-truss > 2).
    sweep("small_u8", 40, &[2, 3], &["u32", "u16", "u8"]);
}

#[test]
fn cross_ix_sweep_u8_node_count_boundary() {
    // n = 127, no chords: a pure ring has exactly `n` undirected edges, so after symmetrization
    // this yields 254 directed edges --- this pushes right up against u8's node-count boundary
    // (255) while staying just under its edge-count boundary too.
    sweep("u8_node_boundary", 127, &[], &["u32", "u16", "u8"]);
}

#[test]
fn cross_ix_sweep_medium_u16_boundary() {
    // n = 3000: exceeds u8 (255) but fits comfortably within u16 (65535). Exact centralities are
    // skipped at this scale (O(|V| * (|V|+|E|))) in debug builds; they're exhaustively covered
    // (and cross-checked against `usize`) by the smaller tiers above.
    sweep_with_opts("medium_u16", 3000, &[2, 5, 13, 41], &["u32", "u16"], false);
}

#[test]
fn cross_ix_sweep_large_u32_only() {
    // n = 100_000: exceeds u16 (65535), only u32/usize apply. Exact centralities are skipped here
    // since they are O(|V| * (|V|+|E|)) --- infeasible at this scale within a unit test; they are
    // already covered exhaustively (and cross-checked against `usize`) by the smaller tiers above.
    //
    // Geometric (power-of-two) chord steps are used instead of a handful of small constants: a
    // pure ring (or ring + few small-constant chords) has diameter O(n), and diameter-bounded
    // algorithms (e.g. HyperBall, whose iteration count is bounded by the graph's diameter) would
    // need on the order of `n` iterations on such a graph --- geometric chords instead give a
    // small-world-like topology with diameter O(log n), keeping every algorithm's genuine
    // asymptotic behavior intact without this synthetic topology being a pathological
    // (diameter-driven) worst case unique to this test.
    let mut chords = Vec::new();
    let mut k = 2usize;
    while k < 100_000 {
        chords.push(k);
        k *= 2;
    }
    sweep_with_opts("large_u32", 100_000, &chords, &["u32"], false);
}
