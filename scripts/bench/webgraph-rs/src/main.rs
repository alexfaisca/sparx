//! Boldi and Vigna's Rust HyperBall (`webgraph-algo`), for scripts/bench/hyperball_webgraph.py:
//!
//!     hyperball-webgraph-rs <basename> <log2m> <threads> <seed> <out prefix> [hll|hll8]
//!
//! Loads the (symmetric) BVGraph `<basename>` (its `.ef` built beforehand, by `webgraph build ef`),
//! runs HyperBall with `2^log2m` registers per counter on a pool of `<threads>` threads until no
//! counter changes, writes each node's estimated sums of distances and of inverse distances to
//! `<out prefix>.webgraph-rs.{dist,harm}` (little-endian `f32`s), and prints a line of JSON: the
//! run's seconds (from building the counters to the last iteration, as the Java benchmark's
//! `init()` + `run()` and sparx's) and its iterations. Its counters are `hll` (the default:
//! `HyperLogLog`, registers packed as WebGraph's Java ones) or `hll8` (`HyperLogLog8`: a byte per
//! register, merged with SIMD instructions, as sparx's).
use dsi_progress_logger::no_logs;
use rand::SeedableRng;
use std::io::Write;
use std::time::Instant;
use webgraph::prelude::BvGraph;
use webgraph::traits::{RandomAccessLabeling, SequentialLabeling};
use webgraph_algo::distances::hyperball::HyperBallBuilder;

fn write_f32s(path: &str, values: &[f32]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 6 || args.len() == 7,
        "usage: {} <basename> <log2m> <threads> <seed> <out prefix> [hll|hll8]",
        args[0]
    );
    let (basename, out) = (&args[1], &args[5]);
    let log2m: u32 = args[2].parse()?;
    let threads: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;

    let graph = BvGraph::with_basename(basename).load()?;
    // the degrees' cumulative function, which balances the iterations' work (out of the timing,
    // as it's a property of the graph, kept with it by webgraph-rs's `webgraph build dcf`)
    let dcf = graph.build_dcf();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;

    let variant = args.get(6).map_or("hll", String::as_str);
    // (the builders' counters are different types: the same steps for each)
    macro_rules! run {
        ($builder:ident) => {
            pool.install(|| -> anyhow::Result<_> {
                let rng = rand::rngs::SmallRng::seed_from_u64(seed);
                let start = Instant::now();
                // a symmetric graph is its own transpose (for the systolic iterations)
                let mut hyperball =
                    HyperBallBuilder::$builder(&graph, Some(&graph), &dcf, log2m, None)?
                        .sum_of_distances(true)
                        .sum_of_inverse_distances(true)
                        .build(no_logs![]);
                hyperball.run_until_done(rng, no_logs![])?;
                let seconds = start.elapsed().as_secs_f64();
                // the neighbourhood function's first value is |V|, then one per iteration
                let iterations = hyperball.neighborhood_function()?.len() - 1;
                Ok((
                    seconds,
                    iterations,
                    hyperball.sum_of_distances()?.to_vec(),
                    hyperball.harmonic_centralities()?.to_vec(),
                ))
            })?
        };
    }
    let (seconds, iterations, distances, harmonic) = match variant {
        "hll" => run!(with_hyper_log_log),
        "hll8" => run!(with_hyper_log_log8),
        other => anyhow::bail!("counters must be hll or hll8, not {other}"),
    };
    let name = if variant == "hll" { "webgraph-rs" } else { "webgraph-rs8" };
    write_f32s(&format!("{out}.{name}.dist"), &distances)?;
    write_f32s(&format!("{out}.{name}.harm"), &harmonic)?;
    println!(
        "{{\"impl\": \"{name}\", \"seconds\": {seconds}, \"iterations\": {iterations}, \"nodes\": {}, \"arcs\": {}}}",
        graph.num_nodes(),
        graph.num_arcs()
    );
    Ok(())
}
