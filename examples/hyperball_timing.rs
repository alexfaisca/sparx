//! Per-iteration wall-time breakdown of `HyperBall`.
//!
//! cargo run --release --features bench --example hyperball_timing -- <graph> [threads] [precision]
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

    match precision {
        4 => run::<Precision4>(&graph, threads as usize),
        6 => run::<Precision6>(&graph, threads as usize),
        8 => run::<Precision8>(&graph, threads as usize),
        p => Err(format!("unsupported precision {p} (use 4, 6 or 8)").into()),
    }
}
