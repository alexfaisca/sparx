//! sparx's *HyperBall* for `scripts/bench/hyperball_webgraph.py`, which compares it with Boldi and
//! Vigna's reference implementation (WebGraph's `it.unimi.dsi.webgraph.algo.HyperBall`).
//!
//! cargo run --release --features bench --example hyperball_bench -- \
//!     <graph> <threads> <log2m> <out prefix> [auto|ram|disk] [write]
//!
//! Runs *HyperBall* with `2^log2m` registers per counter (sparx's `Precision<log2m>`, WebGraph's
//! `log2m`) until no counter changes (the iteration cap raised to its largest, 1024, so that runs
//! end as WebGraph's do, by stabilisation), and writes each node's estimated sum of distances and
//! sum of inverse distances, and the estimated number of nodes it reaches (itself included), to
//! `<out prefix>.sparx.{dist,harm,reach}` (`f64`s, native endianness).
//! The working memory is kept where the fifth argument says (see `MemoryPolicy`; `auto` by
//! default). With `write`, the results are also written to the graph's cache, as
//! `HyperBallInner::new` does (`2^log2m` bytes per node: 13 GB at `log2m` 8 on 50M nodes), and
//! that's timed apart. Prints one line of JSON: the run's seconds (counters' initialisation
//! included, as WebGraph's `init()` + `run()`), the seconds taken to write its results to the
//! graph's cache (which WebGraph's timing has no counterpart of; 0 without `write`), whether its
//! working memory was in RAM, its iterations, and the graph's size.
use sparx::centralities::hyper_ball::*;
use sparx::centralities::hyper_log_log::Precision;
use sparx::graph::Graph;
use sparx::shared_slice::MemoryPolicy;
use std::io::Write;
use std::time::Instant;

type G = Graph<(), (), u32>;

fn write_f64s(path: &str, values: impl Iterator<Item = f64>) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        out.write_all(&v.to_ne_bytes())?;
    }
    out.flush()
}

/// A run's seconds (computing), seconds writing its results to the cache, whether its working
/// memory was in RAM, and iterations.
type Run = (f64, f64, bool, usize);

/// Times *HyperBall* at precision `P`, writes its estimates, and returns its [`Run`].
fn hyperball<P: Precision>(
    g: &G,
    threads: usize,
    memory: MemoryPolicy,
    write: bool,
    out: &str,
) -> Result<Run, Box<dyn std::error::Error>> {
    let time = Instant::now();
    let mut hb = HyperBallInner::<(), (), u32, P, f64, _>::new_no_compute_with_memory(
        g,
        Some(1024),
        threads,
        memory,
    )?;
    let swap = hb.init_cache_mem()?;
    hb.compute_with_proc_mem(swap)?;
    let seconds = time.elapsed().as_secs_f64();
    let time = Instant::now();
    if write {
        hb.write_results()?;
    }
    let write_seconds = time.elapsed().as_secs_f64();
    let iterations = hb.get_iters();
    write_f64s(
        &format!("{out}.sparx.dist"),
        hb.geodesic_distances().iter().copied(),
    )?;
    write_f64s(
        &format!("{out}.sparx.harm"),
        hb.harmonic_geodesic_distances().iter().copied(),
    )?;
    write_f64s(
        &format!("{out}.sparx.reach"),
        hb.counters().iter().map(|c| c.estimate_cardinality()),
    )?;
    let in_ram = hb.in_ram();
    hb.drop_cache()?;
    Ok((seconds, write_seconds, in_ram, iterations))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <graph> <threads> <log2m> <out prefix> [auto|ram|disk] [write]",
            args[0]
        );
        std::process::exit(2);
    }
    let memory = match args.get(5).map(String::as_str) {
        None | Some("auto") => MemoryPolicy::Auto,
        Some("ram") => MemoryPolicy::Ram,
        Some("disk") => MemoryPolicy::Disk,
        Some(other) => return Err(format!("memory must be auto, ram or disk, not {other}").into()),
    };
    let write = match args.get(6).map(String::as_str) {
        None => false,
        Some("write") => true,
        Some(other) => {
            return Err(format!("the sixth argument can only be write, not {other}").into());
        }
    };
    let (path, out) = (&args[1], &args[4]);
    let threads: usize = args[2].parse()?;
    let log2m: u32 = args[3].parse()?;
    let threads_u8 = u8::try_from(threads).map_err(|_| "threads must be at most 255")?;

    let time = Instant::now();
    let mut g = G::from_file(path, None, Some(threads_u8))?;
    let load = time.elapsed().as_secs_f64();
    let (nodes, arcs) = (g.size(), g.width());

    let result: Result<Run, Box<dyn std::error::Error>> = {
        macro_rules! by_precision {
            ($($p:literal => $t:ident),*) => {
                match log2m {
                    $($p => hyperball::<sparx::centralities::hyper_log_log::$t>(&g, threads, memory, write, out),)*
                    _ => Err(format!("log2m must be 4 to 16, not {log2m}").into()),
                }
            };
        }
        by_precision!(
            4 => Precision4, 5 => Precision5, 6 => Precision6, 7 => Precision7,
            8 => Precision8, 9 => Precision9, 10 => Precision10, 11 => Precision11,
            12 => Precision12, 13 => Precision13, 14 => Precision14, 15 => Precision15,
            16 => Precision16
        )
    };
    // the graph's cache entry is dropped whatever happened
    let dropped = g.drop_cache();
    let (seconds, write_seconds, in_ram, iterations) = result?;
    dropped?;
    println!(
        "{{\"impl\": \"sparx\", \"seconds\": {seconds}, \"write_seconds\": {write_seconds}, \"in_ram\": {in_ram}, \"iterations\": {iterations}, \"nodes\": {nodes}, \"arcs\": {arcs}, \"load_seconds\": {load}}}"
    );
    Ok(())
}
