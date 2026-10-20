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
use sparx::centralities::exact::ExactClosenessCentrality;
use sparx::centralities::hyper_log_log::{HyperLogLog, Precision, Precision4, Precision8};
use sparx::graph::Graph;
use sparx::k_truss::clustering_coefficient::ClusteringCoefficient;
use sparx::k_truss::triangles::Triangles;
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
