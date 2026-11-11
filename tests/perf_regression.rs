//! Performance regression checks, for regressions of an order of magnitude: each compares an
//! operation with a reference timed in the same process, on the same machine, so that the checks
//! hold on machines of any speed (CI runners included), and fails only well past run-to-run noise.
//!
//! They're ignored by `cargo test`, as timings are only meaningful in release builds on an
//! otherwise idle machine: run them with
//!
//! ```sh
//! cargo test --release --test perf_regression -- --ignored --nocapture --test-threads=1
//! ```
use sparx::assortativity::{Correlation, degree_assortativity};
use sparx::centralities::betweenness::{Betweenness, BetweennessEngine, BetweennessOptions};
use sparx::centralities::exact::ExactClosenessCentrality;
use sparx::centralities::hyper_log_log::{HyperLogLog, Precision, Precision4, Precision8};
use sparx::communities::directed_louvain::DirectedLouvain;
use sparx::communities::label_propagation::LabelPropagation;
use sparx::communities::spectral::{EmbeddingOptions, SpectralEmbedding, SpectralMatrix};
use sparx::components::biconnected::Biconnectivity;
use sparx::components::weak::ConnectedComponents;
use sparx::graph::{DiGraph, Graph};
use sparx::k_core::directed::{CoreDegree, DirectedKCores};
use sparx::k_truss::clustering_coefficient::{
    ClusteringCoefficient, DirectedClusteringCoefficient,
};
use sparx::k_truss::triangles::Triangles;
use sparx::trails::dijkstra::{DijkstraDists, ShortestPathEngine};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The fastest of `runs` runs of `f`.
fn fastest(runs: usize, mut f: impl FnMut()) -> Duration {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .min()
        .unwrap()
}

/// Seeded pseudo-random numbers.
fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 17
    }
}

/// A random undirected graph of `n` nodes and about `m` edges, as a MatrixMarket file in a fresh
/// directory, loaded with `threads` threads.
fn random_graph(name: &str, n: usize, m: usize, threads: u8) -> (Graph<(), (), u32>, PathBuf) {
    let dir = std::env::temp_dir().join(format!("sparx_perf_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut next = rng(n as u64 ^ m as u64);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    writeln!(
        f,
        "%%MatrixMarket matrix coordinate pattern symmetric\n{n} {n} {m}"
    )
    .unwrap();
    for _ in 0..m {
        let (u, v) = (next() as usize % n, next() as usize % n);
        writeln!(f, "{} {}", u.max(v) + 1, u.min(v) + 1).unwrap();
    }
    drop(f);
    let id = format!("perf_{name}_{}", std::process::id());
    (
        Graph::from_file(&path, Some(id), Some(threads)).unwrap(),
        dir,
    )
}

/// A random directed graph of `n` nodes and about `out_degree` arcs out of each, as a MatrixMarket
/// file in a fresh directory, loaded with `threads` threads.
fn random_digraph(
    name: &str,
    n: usize,
    out_degree: usize,
    threads: u8,
) -> (DiGraph<(), (), u32>, PathBuf) {
    let dir = std::env::temp_dir().join(format!("sparx_perf_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut next = rng(n as u64 ^ out_degree as u64);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let m = n * out_degree;
    writeln!(
        f,
        "%%MatrixMarket matrix coordinate pattern general\n{n} {n} {m}"
    )
    .unwrap();
    for u in 0..n {
        for _ in 0..out_degree {
            writeln!(f, "{} {}", u + 1, next() as usize % n + 1).unwrap();
        }
    }
    drop(f);
    let id = format!("perf_{name}_{}", std::process::id());
    (
        DiGraph::from_file(&path, Some(id), Some(threads)).unwrap(),
        dir,
    )
}

/// A planted partition of `n` nodes into `k` equal blocks, with `n * degree / 2` edges, each
/// between blocks with probability `mu` (within one otherwise), as a MatrixMarket file in a fresh
/// directory: the file's path, to be loaded with as many threads as each check needs.
fn planted_partition(name: &str, n: usize, k: usize, degree: usize, mu: f64) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("sparx_perf_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut next = rng((n ^ k ^ degree) as u64);
    let size = n / k;
    let m = n * degree / 2;
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    writeln!(
        f,
        "%%MatrixMarket matrix coordinate pattern symmetric\n{n} {n} {m}"
    )
    .unwrap();
    let mut written = 0;
    while written < m {
        let u = next() as usize % n;
        let v = if ((next() % (1 << 30)) as f64) < mu * (1u64 << 30) as f64 {
            next() as usize % n
        } else {
            ((u / size) * size + next() as usize % size) % n
        };
        if u != v {
            writeln!(f, "{} {}", u.max(v) + 1, u.min(v) + 1).unwrap();
            written += 1;
        }
    }
    (path, dir)
}

/// The threads checks timing parallel algorithms run on: the machine's, up to `max`.
fn threads(max: usize) -> u8 {
    std::thread::available_parallelism()
        .map_or(4, |t| t.get())
        .min(max) as u8
}

/// The fastest of 3 runs of connected components on `g`: one union-find pass over its edges,
/// the reference parallel algorithms on the same graph are timed against.
fn connected_components<Dir: sparx::graph::Directedness>(
    g: &sparx::graph::GraphMemoryMap<(), (), u32, Dir>,
) -> Duration {
    fastest(3, || {
        let mut cc = ConnectedComponents::new(g).unwrap();
        black_box(cc.count());
        cc.drop_cache().unwrap();
    })
}

/// How many times faster [`HyperLogLog::union`] merges neighbours' counters, in *HyperBall*'s
/// loop, than a register at a time (`black_box` keeps the latter from being vectorized).
fn union_speedup<P: Precision>() -> f64 {
    const COUNTERS: usize = 1 << 12;
    const DEGREE: usize = 8;
    let mut next = rng(1);
    let counters: Vec<HyperLogLog<P>> = (0..COUNTERS)
        .map(|_| {
            let mut c = HyperLogLog::default();
            for _ in 0..next() % 2048 {
                c.insert(next());
            }
            c
        })
        .collect();
    let neighbours: Vec<usize> = (0..COUNTERS * DEGREE)
        .map(|_| next() as usize % COUNTERS)
        .collect();

    // as in HyperBall: a node's counter, merged with its neighbours'
    let union = fastest(9, || {
        let mut changed = 0;
        for u in 0..COUNTERS {
            let mut a = counters[u];
            for &v in &neighbours[u * DEGREE..(u + 1) * DEGREE] {
                changed += a.union(&counters[v]) as usize;
            }
            black_box(&a);
        }
        black_box(changed);
    });
    let scalar = fastest(9, || {
        let mut changed = 0;
        for u in 0..COUNTERS {
            let mut a = counters[u].registers().to_vec();
            for &v in &neighbours[u * DEGREE..(u + 1) * DEGREE] {
                let mut c = false;
                for (x, &y) in a.iter_mut().zip(counters[v].registers()) {
                    let x_now = black_box(*x);
                    if y > x_now {
                        *x = y;
                        c = true;
                    }
                }
                changed += c as usize;
            }
            black_box(&a);
        }
        black_box(changed);
    });
    let speedup = scalar.as_secs_f64() / union.as_secs_f64();
    println!(
        "precision {}: union {union:?}, a register at a time {scalar:?} ({speedup:.1}x)",
        P::EXPONENT
    );
    speedup
}

/// *HyperBall*'s loop merges a node's neighbours' counters: the union must be vectorized there.
/// Written as a plain loop, it wasn't at precision 4 (16 registers), and ran 14 times slower.
#[test]
#[ignore]
fn hyperloglog_union_is_vectorized() {
    let (p4, p8) = (union_speedup::<Precision4>(), union_speedup::<Precision8>());
    for (precision, speedup) in [(4, p4), (8, p8)] {
        assert!(
            speedup >= 4.,
            "precision {precision}: HyperLogLog::union is only {speedup:.1}x faster than a register at a time: is it vectorized?"
        );
    }
}

/// The exact centralities' BFSs from every node share their work (multi-source BFSs): faster
/// than separate, efficient, BFSs from each node.
#[test]
#[ignore]
fn exact_centralities_share_their_bfss() {
    let (mut g, dir) = random_graph("exact", 8192, 32768, 1);
    let n = g.size();
    // a BFS from each node, with memory reused across them
    let mut stamp = vec![0u32; n];
    let mut queue = vec![0usize; n];
    let mut reference_sums = vec![0u64; n];
    let separate = fastest(3, || {
        for source in 0..n {
            let mark = source as u32 + 1;
            stamp[source] = mark;
            queue[0] = source;
            let (mut head, mut tail, mut level_end, mut level, mut sum) = (0, 1, 1, 1u64, 0u64);
            while head < tail {
                for v in g.neighbours(queue[head]).unwrap() {
                    if stamp[v] != mark {
                        stamp[v] = mark;
                        queue[tail] = v;
                        tail += 1;
                        sum += level;
                    }
                }
                head += 1;
                if head == level_end {
                    level_end = tail;
                    level += 1;
                }
            }
            reference_sums[source] = sum;
        }
    });
    let mut sums = vec![];
    let shared = fastest(3, || {
        let mut c = ExactClosenessCentrality::<_, _, _, f64, _>::new(&g, None).unwrap();
        sums = c.centralities().to_vec();
        c.drop_cache().unwrap();
    });
    // the same distances (closeness without normalization is their sum)
    for (u, (&got, &want)) in sums.iter().zip(&reference_sums).enumerate() {
        assert!(
            got == want as f64 || (got == 0. && want == 0),
            "node {u}: {got} vs {want}"
        );
    }
    let speedup = separate.as_secs_f64() / shared.as_secs_f64();
    println!("exact closeness: {shared:?}, a BFS from each node: {separate:?} ({speedup:.1}x)");
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        speedup >= 1.5,
        "exact closeness is only {speedup:.1}x as fast as separate BFSs from each node: do they still share their work?"
    );
}

/// The clustering coefficients cost not much more than counting the triangles they're computed
/// from. On a graph of many nodes and few edges, where their per node work weighs most, they took
/// 4 to 6 times as long, but 20 to 40 times when they were summed in shared atomics (every thread
/// contending for them at every node).
#[test]
#[ignore]
fn clustering_costs_little_more_than_triangles() {
    let threads = std::thread::available_parallelism()
        .map_or(4, |t| t.get())
        .min(64) as u8;
    let (mut g, dir) = random_graph("clustering", 1 << 22, 1 << 20, threads);
    // the triangles' pre-processing, shared by both
    g.edge_reciprocal().unwrap();
    let triangles = fastest(3, || {
        let mut t = Triangles::new(&g).unwrap();
        t.drop_cache().unwrap();
    });
    let clustering = fastest(3, || {
        let mut c = ClusteringCoefficient::new(&g).unwrap();
        c.drop_cache().unwrap();
    });
    let ratio = clustering.as_secs_f64() / triangles.as_secs_f64();
    println!(
        "clustering: {clustering:?}, triangles: {triangles:?} ({ratio:.1}x, {threads} threads)"
    );
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        ratio <= 12.,
        "the clustering coefficients take {ratio:.1}x as long as counting triangles"
    );
}

/// Directed Louvain stops on graphs without structure: asynchronous moves keep finding tiny
/// gains there, and phases ending only once no node moves made it run for over 40 minutes on a
/// random graph of 4 million nodes. Its stopping rules (sweeps capped, gain thresholds, refining
/// passes ending once they gain little) keep it to about 100 sweeps over the nodes on this one,
/// whatever the threads, where phases ending only once no node moves took 500 to 800: checked
/// by counting the sweeps, which doesn't depend on the machine (timings, 4 times apart on a graph
/// this small, would hardly tell the two apart). Its time is checked too, for larger
/// regressions (about 90 to 150 connected components' time; the check fails past 600), and its
/// modularity, as stopping too early would pass otherwise.
#[test]
#[ignore]
fn directed_louvain_stops_on_graphs_without_structure() {
    let threads = threads(8);
    let (mut g, dir) = random_digraph("directed_louvain", 1 << 17, 8, threads);
    g.transpose().unwrap();
    let reference = connected_components(&g);
    let (mut modularity, mut sweeps) = (0.0, 0);
    let louvain = fastest(2, || {
        let mut l = DirectedLouvain::new(&g).unwrap();
        modularity = l.partition_modularity();
        sweeps = sweeps.max(l.sweeps());
        l.drop_cache().unwrap();
    });
    let ratio = louvain.as_secs_f64() / reference.as_secs_f64();
    println!(
        "directed louvain: {louvain:?} ({sweeps} sweeps, modularity {modularity:.4}), connected components: {reference:?} ({ratio:.0}x, {threads} threads)"
    );
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        modularity > 0.15,
        "directed louvain's modularity on a random graph is only {modularity:.4} (about 0.2 expected): does it stop too early?"
    );
    assert!(
        sweeps <= 250,
        "directed louvain took {sweeps} sweeps over the nodes (about 100 expected): do its phases still stop?"
    );
    assert!(
        ratio <= 600.,
        "directed louvain takes {ratio:.0}x as long as connected components"
    );
}

/// Biconnectivity is parallel throughout (spanning forest, rooting the trees by their Euler
/// tours, subtrees' edges going out by range queries, components by union-find): about 9 to 12
/// times connected components' time on 4 to 8 threads. With its trees rooted, or its components
/// numbered, sequentially, it took 20 to 30 times on 32 threads; the check fails past 30.
#[test]
#[ignore]
fn biconnectivity_is_parallel() {
    let threads = threads(8);
    // sparse, for many biconnected components, articulation points and bridges
    let (mut g, dir) = random_graph("biconnectivity", 1 << 21, 3 << 19, threads);
    let reference = connected_components(&g);
    let biconnectivity = fastest(3, || {
        let mut b = Biconnectivity::new(&g).unwrap();
        black_box(b.count());
        b.drop_cache().unwrap();
    });
    let ratio = biconnectivity.as_secs_f64() / reference.as_secs_f64();
    println!(
        "biconnectivity: {biconnectivity:?}, connected components: {reference:?} ({ratio:.1}x, {threads} threads)"
    );
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        ratio <= 30.,
        "biconnectivity takes {ratio:.1}x as long as connected components: is it still parallel?"
    );
}

/// Degree assortativity is a single parallel pass over the edges, reading each head's degree
/// from an array of one byte per node: about a tenth to a fifth of connected components' time
/// on 4 to 8 threads. It would take several times as long sequentially, or with a pass more
/// over the edges; the check fails past connected components' time.
#[test]
#[ignore]
fn degree_assortativity_is_one_parallel_pass() {
    let threads = threads(8);
    let (mut g, dir) = random_graph("assortativity", 1 << 21, 1 << 23, threads);
    let reference = connected_components(&g);
    let assortativity = fastest(5, || {
        black_box(degree_assortativity(&g, Correlation::Pearson).unwrap());
    });
    let ratio = assortativity.as_secs_f64() / reference.as_secs_f64();
    println!(
        "degree assortativity: {assortativity:?}, connected components: {reference:?} ({ratio:.2}x, {threads} threads)"
    );
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        ratio <= 1.,
        "degree assortativity takes {ratio:.2}x as long as connected components: is it still one parallel pass?"
    );
}

/// Δ-stepping spreads one source's shortest paths over threads: on a random graph, with 4
/// threads or more, at least twice as fast as Dijkstra's algorithm on one (6 to 12 times, on
/// 16 to 32 threads).
#[test]
#[ignore]
fn delta_stepping_is_parallel() {
    let threads = threads(16);
    if threads < 4 {
        println!("delta stepping: skipped, {threads} threads");
        return;
    }
    let (mut g, dir) = random_digraph("delta", 1 << 19, 8, threads);
    let time = |engine| {
        fastest(3, || {
            let mut d = DijkstraDists::with_engine(&g, 0, engine).unwrap();
            black_box(d.total_distances());
            d.drop_cache().unwrap();
        })
    };
    let sequential = time(ShortestPathEngine::Sequential);
    let parallel = time(ShortestPathEngine::DeltaStepping { delta: None });
    let speedup = sequential.as_secs_f64() / parallel.as_secs_f64();
    println!(
        "delta stepping: {parallel:?}, dijkstra: {sequential:?} ({speedup:.1}x on {threads} threads)"
    );
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        speedup >= 2.,
        "Δ-stepping is only {speedup:.1}x as fast as Dijkstra's algorithm on {threads} threads"
    );
}

/// Label propagation's semi-synchronous rounds run on all threads: on a planted partition of 2^18
/// nodes, many threads must be at least twice as fast as one (about 5x measured, 1 to 16
/// threads).
#[test]
#[ignore]
fn label_propagation_is_parallel() {
    let threads = threads(16);
    if threads < 4 {
        println!("label propagation: skipped, {threads} threads");
        return;
    }
    let (path, dir) = planted_partition("lpa", 1 << 18, 16, 16, 0.3);
    let time = |t: u8| {
        let id = format!("perf_lpa_{t}_{}", std::process::id());
        let mut g = Graph::<(), (), u32>::from_file(&path, Some(id), Some(t)).unwrap();
        let time = fastest(3, || {
            let mut l = LabelPropagation::new(&g).unwrap();
            black_box(l.community_count());
            l.drop_cache().unwrap();
        });
        g.drop_cache().unwrap();
        time
    };
    let sequential = time(1);
    let parallel = time(threads);
    let speedup = sequential.as_secs_f64() / parallel.as_secs_f64();
    println!(
        "label propagation: {parallel:?} on {threads} threads, {sequential:?} on 1 ({speedup:.1}x)"
    );
    let _ = std::fs::remove_dir_all(dir);
    assert!(
        speedup >= 2.,
        "label propagation is only {speedup:.1}x as fast on {threads} threads as on 1"
    );
}

/// The exact spectral engine's filtered subspace iteration converges in a few iterations when
/// the graph has the structure it looks for: on a planted partition of 2^18 nodes into 16 blocks,
/// both matrices' 16 eigenvectors must converge within 10 iterations (2 measured), so that a
/// regression of the filter or the locking shows as many more.
#[test]
#[ignore]
fn exact_spectral_engine_converges_quickly() {
    let (path, dir) = planted_partition("spectral", 1 << 18, 16, 16, 0.3);
    let id = format!("perf_spectral_{}", std::process::id());
    let mut g = Graph::<(), (), u32>::from_file(&path, Some(id), Some(threads(16))).unwrap();
    let mut results = vec![];
    for matrix in [
        SpectralMatrix::RegularizedLaplacian,
        SpectralMatrix::BetheHessian,
    ] {
        let t = Instant::now();
        let mut e = SpectralEmbedding::<_, _, _, f64>::new(
            &g,
            EmbeddingOptions {
                matrix,
                ..EmbeddingOptions::new(16)
            },
        )
        .unwrap();
        println!(
            "exact spectral engine, {matrix:?}: {:?}, {} iterations, converged: {}",
            t.elapsed(),
            e.iterations(),
            e.converged()
        );
        results.push((matrix, e.iterations(), e.converged()));
        e.drop_cache().unwrap();
    }
    g.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    for (matrix, iterations, converged) in results {
        assert!(
            converged && iterations <= 10,
            "the exact engine took {iterations} iterations on {matrix:?} (converged: {converged})"
        );
    }
}

/// KADABRA's samples are drawn on all threads: on a random graph of 2^19 nodes, many threads
/// must be at least twice as fast as one (about 7x measured, 1 to 16 threads), drawing the same
/// samples.
#[test]
#[ignore]
fn kadabra_is_parallel() {
    let threads = threads(16);
    if threads < 4 {
        println!("kadabra: skipped, {threads} threads");
        return;
    }
    let (mut g, dir) = random_graph("kadabra", 1 << 19, 1 << 22, 1);
    let path = dir.join("kadabra.mtx");
    let options = BetweennessOptions {
        engine: BetweennessEngine::Kadabra,
        ..BetweennessOptions::default()
    };
    let time = |g: &Graph<(), (), u32>| {
        let mut samples = 0;
        let time = fastest(3, || {
            let mut b = Betweenness::with_options(g, options).unwrap();
            samples = b.samples();
            black_box(b.centralities());
            b.drop_cache().unwrap();
        });
        (time, samples)
    };
    let (sequential, samples) = time(&g);
    g.drop_cache().unwrap();
    let id = format!("perf_kadabra_many_{}", std::process::id());
    let mut many = Graph::<(), (), u32>::from_file(&path, Some(id), Some(threads)).unwrap();
    let (parallel, parallel_samples) = time(&many);
    many.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    let speedup = sequential.as_secs_f64() / parallel.as_secs_f64();
    println!(
        "kadabra: {parallel:?} on {threads} threads, {sequential:?} on 1 ({speedup:.1}x, {samples} samples)"
    );
    assert_eq!(samples, parallel_samples);
    assert!(
        speedup >= 2.,
        "KADABRA is only {speedup:.1}x as fast on {threads} threads as on 1"
    );
}

/// Directed k-cores and clustering coefficients run on all threads: on a random directed graph of
/// 2^20 nodes, many threads must be at least twice as fast as one at both (measured on a de
/// Bruijn graph of 21.9M nodes: k-cores 3.4x, clustering 9.2x, 1 to 32 threads).
#[test]
#[ignore]
fn directed_cores_and_clustering_are_parallel() {
    let threads = threads(16);
    if threads < 4 {
        println!("directed cores and clustering: skipped, {threads} threads");
        return;
    }
    let (mut one, dir) = random_digraph("directed_parallel", 1 << 20, 8, 1);
    let path = dir.join("directed_parallel.mtx");
    let id = format!("perf_directed_parallel_many_{}", std::process::id());
    let mut many = DiGraph::<(), (), u32>::from_file(&path, Some(id), Some(threads)).unwrap();
    let time = |g: &DiGraph<(), (), u32>| {
        g.transpose().unwrap();
        let cores = fastest(3, || {
            let mut c = DirectedKCores::new(g, CoreDegree::In).unwrap();
            black_box(c.k_cores());
            c.drop_cache().unwrap();
        });
        let clustering = fastest(3, || {
            let mut c = DirectedClusteringCoefficient::new(g).unwrap();
            black_box(c.get_average_clustering_coefficient());
            c.drop_cache().unwrap();
        });
        (cores, clustering)
    };
    let (cores_one, clustering_one) = time(&one);
    let (cores_many, clustering_many) = time(&many);
    one.drop_cache().unwrap();
    many.drop_cache().unwrap();
    let _ = std::fs::remove_dir_all(dir);
    let (cores, clustering) = (
        cores_one.as_secs_f64() / cores_many.as_secs_f64(),
        clustering_one.as_secs_f64() / clustering_many.as_secs_f64(),
    );
    println!(
        "directed k-cores: {cores_many:?} on {threads} threads ({cores:.1}x); clustering: \
         {clustering_many:?} ({clustering:.1}x)"
    );
    assert!(
        cores >= 2.,
        "directed k-cores are only {cores:.1}x as fast on {threads} threads"
    );
    assert!(
        clustering >= 2.,
        "directed clustering is only {clustering:.1}x as fast on {threads} threads"
    );
}
