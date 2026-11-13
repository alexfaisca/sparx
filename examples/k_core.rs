//! k-core decomposition, sequential (Batagelj–Zaversnik) and parallel (Liu et al.).
//!
//! ```sh
//! cargo run --release --example k_core -- [graph] [threads]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4. Coreness is stored as `u32`s
//! here, so that any graph's degrees fit (`new` stores `u8`s, erroring on degrees above 255).
use sparx::graph::Graph;
use sparx::k_core::{batagelj_zaversnik::AlgoBatageljZaversnik, liu_et_al::AlgoLiuEtAl};

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;

    let mut graph = Graph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph);
    // the graph's cache entry is dropped, also when the run failed
    graph.drop_cache()?;
    result
}

fn run(graph: &Graph<(), (), u32>) -> Result<(), Box<dyn std::error::Error>> {
    let mut bz = AlgoBatageljZaversnik::<_, _, _, u32>::new_cx(graph)?;
    let cores = bz.k_cores().to_vec();
    bz.drop_cache()?;

    let mut liu = AlgoLiuEtAl::<_, _, _, u32>::new_cx(graph, graph.thread_num())?;
    println!(
        "both algorithms agree: {}",
        liu.k_cores() == cores.as_slice()
    );
    liu.drop_cache()?;

    let degeneracy = cores.iter().copied().max().unwrap_or(0);
    println!("degeneracy (max coreness): {degeneracy}");
    let mut histogram = vec![0usize; degeneracy as usize + 1];
    for &k in &cores {
        histogram[k as usize] += 1;
    }
    for (k, count) in histogram.iter().enumerate().filter(|(_, c)| **c > 0) {
        println!("  coreness {k:>3}: {count} nodes");
    }
    Ok(())
}
