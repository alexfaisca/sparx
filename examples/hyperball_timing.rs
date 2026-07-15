//! Per-iteration wall-time breakdown of `HyperBall`.
//!
//! cargo run --release --features bench --example hyperball_timing -- <graph> [threads] [precision]
//!
//! With the `gpu` feature and a 4th argument `gpu`, instead runs *HyperBall* on the CPU and on the
//! GPU, timing both and checking they agree (`gpu-triangles` does the same for triangle counting).
//!
//! With a 4th argument `ktruss`, instead times triangle counting, both k-truss decompositions and
//! the clustering coefficients on the CPU (printing checksums of their results); `ktruss-directed`
//! does the same for directed triangles and trusses of a few types.
//!
//! With `<graph>` = `micro`, instead times the two per-edge/per-node counter operations of an
//! iteration (union and cardinality estimation) in isolation, single-threaded.
use sparx::centralities::hyper_ball::*;
use sparx::centralities::hyper_log_log::{HyperLogLog, Precision};
use sparx::graph::GraphMemoryMap;
use std::time::{Duration, Instant};

fn run<P: Precision>(
    graph: &GraphMemoryMap<(), (), u32>,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let time = Instant::now();
    let mut hb = HyperBallInner::<(), (), u32, P, f64>::new_no_compute(graph, None, threads)?;
    let init = time.elapsed();
    let time = Instant::now();
    let swap = hb.init_cache_mem()?;
    let swap_init = time.elapsed();
    let time = Instant::now();
    hb.compute_with_proc_mem(swap)?;
    let compute = time.elapsed();
    let total = init + swap_init + compute;
    println!("init {init:?}, swap buffer {swap_init:?}, compute {compute:?}");

    let iter_times = hb.get_iter_times();
    let iter_sum: Duration = iter_times.iter().sum();
    println!(
        "counter size {} B, threads {threads}, iterations {}",
        std::mem::size_of::<HyperLogLog<P>>(),
        iter_times.len()
    );
    println!(
        "total {total:?} (iterations {iter_sum:?}, rest of compute {:?})",
        compute - iter_sum
    );
    for (i, t) in iter_times.iter().enumerate() {
        println!("  iter {i:>4}: {t:?}");
    }
    hb.drop_cache()?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn compare_gpu<P: Precision>(
    graph: &GraphMemoryMap<(), (), u32>,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use sparx::gpu::{GpuBackend, WgpuBackend};

    let time = Instant::now();
    let mut cpu = HyperBallInner::<(), (), u32, P, f64>::new_with_conf(graph, None, threads)?;
    println!("CPU: {:?}, {} iterations", time.elapsed(), cpu.get_iters());
    // both runs use the same cache files, so keep the CPU results in memory
    let cpu_counters = cpu.counters().to_vec();
    let cpu_distances = cpu.geodesic_distances().to_vec();
    let cpu_iters = cpu.get_iters();
    cpu.drop_cache()?;

    let time = Instant::now();
    let backend = WgpuBackend::new()?;
    println!(
        "GPU: adapter {:?} in {:?}",
        backend.adapter_info().name,
        time.elapsed()
    );
    let time = Instant::now();
    let device_graph = backend.upload_graph(graph)?;
    println!("GPU: graph uploaded in {:?}", time.elapsed());
    let time = Instant::now();
    let mut gpu =
        HyperBallInner::<(), (), u32, P, f64>::new_gpu(graph, &backend, &device_graph, None)?;
    println!("GPU: {:?}, {} iterations", time.elapsed(), gpu.get_iters());

    let differing = cpu_counters
        .iter()
        .zip(gpu.counters())
        .filter(|(c, g)| c != g)
        .count();
    let max_relative = cpu_distances
        .iter()
        .zip(gpu.geodesic_distances())
        .map(|(&c, &g)| (g - c).abs() / c.abs().max(1.))
        .fold(0f64, f64::max);
    println!(
        "same iterations: {}, differing counters: {differing}, max relative distance difference: {max_relative:e}",
        cpu_iters == gpu.get_iters()
    );
    gpu.drop_cache()?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn compare_gpu_triangles(
    graph: &GraphMemoryMap<(), (), u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use portable_atomic::Ordering;
    use sparx::gpu::{GpuBackend, WgpuBackend};
    use sparx::k_truss::triangles::Triangles;

    // computed (and cached) once, used by both
    let time = Instant::now();
    graph.edge_over()?;
    graph.edge_reciprocal()?;
    println!("edge over and reciprocals: {:?}", time.elapsed());

    let time = Instant::now();
    let mut cpu = Triangles::new(graph)?;
    println!("CPU: {:?}", time.elapsed());
    let cpu_counts: Vec<u8> = cpu
        .all_triangles()
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();
    cpu.drop_cache()?;

    let backend = WgpuBackend::new()?;
    let time = Instant::now();
    let device_graph = backend.upload_graph(graph)?;
    println!(
        "GPU ({:?}): graph uploaded in {:?}",
        backend.adapter_info().name,
        time.elapsed()
    );
    let time = Instant::now();
    let mut gpu = Triangles::new_gpu(graph, &backend, &device_graph)?;
    println!("GPU: {:?}", time.elapsed());

    let differing = cpu_counts
        .iter()
        .zip(gpu.all_triangles())
        .filter(|(c, g)| **c != g.load(Ordering::Relaxed))
        .count();
    let triangles: u64 = cpu_counts.iter().map(|&c| c as u64).sum::<u64>() / 6;
    println!("triangles: {triangles}, differing edge counts: {differing}");
    gpu.drop_cache()?;
    Ok(())
}

fn time_ktruss(
    graph: &GraphMemoryMap<(), (), u32>,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use portable_atomic::Ordering;
    use sparx::k_truss::{
        burkhardt_et_al::AlgoBurkhardtEtAl, clustering_coefficient::ClusteringCoefficient,
        pkt::AlgoPKT, triangles::Triangles,
    };
    let checksum = |values: &mut dyn Iterator<Item = u64>| {
        values.fold(0u64, |h, v| {
            h.wrapping_mul(0x100_0000_01B3).wrapping_add(v + 1)
        })
    };

    // computed (and cached) once, used by all
    let time = Instant::now();
    graph.edge_over()?;
    graph.edge_reciprocal()?;
    println!("edge over and reciprocals: {:?}", time.elapsed());

    let time = Instant::now();
    let mut triangles = Triangles::new(graph)?;
    let elapsed = time.elapsed();
    let counts = triangles.all_triangles();
    println!(
        "triangles: {elapsed:?} (max per edge {}, checksum {:x})",
        counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0),
        checksum(&mut counts.iter().map(|c| c.load(Ordering::Relaxed) as u64))
    );
    triangles.drop_cache()?;

    let time = Instant::now();
    let mut pkt = AlgoPKT::new_with_conf(graph, threads)?;
    let elapsed = time.elapsed();
    println!(
        "pkt: {elapsed:?} (max trussness {}, checksum {:x})",
        pkt.k_trusses().iter().max().unwrap_or(&0),
        checksum(&mut pkt.k_trusses().iter().map(|&k| k as u64))
    );
    let pkt_trussness = pkt.k_trusses().to_vec();
    pkt.drop_cache()?;

    let time = Instant::now();
    let mut bea = AlgoBurkhardtEtAl::new(graph)?;
    let elapsed = time.elapsed();
    println!(
        "burkhardt et al.: {elapsed:?} (agrees with pkt: {})",
        bea.k_trusses() == pkt_trussness.as_slice()
    );
    bea.drop_cache()?;

    let time = Instant::now();
    let cc = ClusteringCoefficient::new(graph)?;
    println!(
        "clustering coefficient: {:?} (average {}, transitivity {})",
        time.elapsed(),
        cc.get_average_clustering_coefficient(),
        cc.get_graph_transitivity()
    );
    Ok(())
}

fn time_ktruss_directed(
    graph: &GraphMemoryMap<(), (), u32>,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use sparx::k_truss::count::TriangleCounter;
    use sparx::k_truss::directed::{DirectedTriangleType, DirectedTriangles};
    use sparx::k_truss::{burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT};
    let checksum = |values: &mut dyn Iterator<Item = u64>| {
        values.fold(0u64, |h, v| {
            h.wrapping_mul(0x100_0000_01B3).wrapping_add(v + 1)
        })
    };
    for t in [
        DirectedTriangleType::Cycle,
        DirectedTriangleType::Out,
        DirectedTriangleType::FeedForward,
        DirectedTriangleType::CycleFeedForward,
    ] {
        let time = Instant::now();
        let mut triangles = DirectedTriangles::<_, _, _, u16>::new_tx(graph, t)?;
        let elapsed = time.elapsed();
        let counts = triangles.all_triangles();
        println!(
            "{t:?} triangles: {elapsed:?} (max per edge {}, checksum {:x})",
            counts.iter().map(|c| c.count()).max().unwrap_or(0),
            checksum(&mut counts.iter().map(|c| c.count() as u64))
        );
        triangles.drop_cache()?;

        let time = Instant::now();
        let mut pkt = AlgoPKT::<_, _, _, u16>::new_directed_tx(graph, t, threads)?;
        let elapsed = time.elapsed();
        println!(
            "{t:?} pkt: {elapsed:?} (max trussness {}, checksum {:x})",
            pkt.k_trusses().iter().max().unwrap_or(&0),
            checksum(&mut pkt.k_trusses().iter().map(|&k| k as u64))
        );
        let pkt_trussness = pkt.k_trusses().to_vec();
        pkt.drop_cache()?;

        let time = Instant::now();
        let mut bea = AlgoBurkhardtEtAl::<_, _, _, u16>::new_directed_tx(graph, t)?;
        println!(
            "{t:?} burkhardt et al.: {:?} (agrees with pkt: {})",
            time.elapsed(),
            bea.k_trusses() == pkt_trussness.as_slice()
        );
        bea.drop_cache()?;
    }
    Ok(())
}

fn micro<P: Precision>() {
    const COUNTERS: usize = 1 << 14;
    const OPS: usize = 1 << 22;
    // pseudo-random access pattern, like neighbour lookups
    let idx = |i: usize| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7) % COUNTERS;

    // small balls (most registers zero, like the first iterations) and large ones (no zero
    // registers, like the last iterations)
    for (label, max_elements) in [("small", 64), ("large", 1 << 14)] {
        let counters: Vec<HyperLogLog<P>> = (0..COUNTERS)
            .map(|i| {
                let mut c = HyperLogLog::<P>::default();
                let elements = if max_elements == 64 {
                    i % 64 + 1
                } else {
                    max_elements
                };
                for j in 0..elements {
                    c.insert((i * max_elements + j) as u64);
                }
                c
            })
            .collect();

        let time = Instant::now();
        let mut acc = HyperLogLog::<P>::default();
        for i in 0..OPS {
            acc.union(&counters[idx(i)]);
        }
        let union = time.elapsed();
        std::hint::black_box(&acc);

        let time = Instant::now();
        let mut sum = 0f64;
        for i in 0..OPS {
            sum += counters[idx(i)].estimate_cardinality();
        }
        let estimate = time.elapsed();
        std::hint::black_box(sum);

        println!(
            "counter size {} B, {label} balls: union {:.1} ns/op, estimate {:.1} ns/op",
            std::mem::size_of::<HyperLogLog<P>>(),
            union.as_nanos() as f64 / OPS as f64,
            estimate.as_nanos() as f64 / OPS as f64
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: hyperball_timing <graph|micro> [threads] [precision]")?;
    if path == "micro" {
        micro::<Precision4>();
        micro::<Precision6>();
        micro::<Precision8>();
        return Ok(());
    }
    let threads: u8 = args.get(2).map_or(Ok(16), |t| t.parse())?;
    let precision: u8 = args.get(3).map_or(Ok(6), |p| p.parse())?;

    let time = Instant::now();
    let graph = GraphMemoryMap::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!(
        "graph |V| = {}, |E| = {} loaded in {:?}",
        graph.size(),
        graph.width(),
        time.elapsed()
    );

    if args.get(4).is_some_and(|a| a == "ktruss-directed") {
        return time_ktruss_directed(&graph, threads as usize);
    }
    if args.get(4).is_some_and(|a| a == "ktruss") {
        return time_ktruss(&graph, threads as usize);
    }
    if args.get(4).is_some_and(|a| a == "gpu-triangles") {
        #[cfg(feature = "gpu")]
        return compare_gpu_triangles(&graph);
        #[cfg(not(feature = "gpu"))]
        return Err("the gpu comparison needs the `gpu` feature".into());
    }
    if args.get(4).is_some_and(|a| a == "gpu") {
        #[cfg(feature = "gpu")]
        return match precision {
            4 => compare_gpu::<Precision4>(&graph, threads as usize),
            6 => compare_gpu::<Precision6>(&graph, threads as usize),
            8 => compare_gpu::<Precision8>(&graph, threads as usize),
            p => Err(format!("unsupported precision {p} (use 4, 6 or 8)").into()),
        };
        #[cfg(not(feature = "gpu"))]
        return Err("the gpu comparison needs the `gpu` feature".into());
    }

    match precision {
        4 => run::<Precision4>(&graph, threads as usize),
        6 => run::<Precision6>(&graph, threads as usize),
        8 => run::<Precision8>(&graph, threads as usize),
        p => Err(format!("unsupported precision {p} (use 4, 6 or 8)").into()),
    }
}
