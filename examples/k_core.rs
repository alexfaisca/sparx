//! k-core decomposition, sequential (Batagelj–Zaversnik) and parallel (Liu et al.).
//!
//! cargo run --release --example k_core -- [graph] [threads]
//!
//! `[graph]` defaults to a small bundled fixture.
use sparx::graph::GraphMemoryMap;
use sparx::k_core::{batagelj_zaversnik::AlgoBatageljZaversnik, liu_et_al::AlgoLiuEtAl};

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;

    let mut graph = GraphMemoryMap::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph |V| = {}, |E| = {}", graph.size(), graph.width());

    let result = run(&graph);
    // also when a run fails
    graph.drop_cache()?;
    result
}

fn run(graph: &GraphMemoryMap<(), (), u32>) -> Result<(), Box<dyn std::error::Error>> {
    let mut bz = AlgoBatageljZaversnik::new(graph)?;
    let cores = bz.k_cores().to_vec();
    bz.drop_cache()?;

    let mut liu = AlgoLiuEtAl::new(graph)?;
    println!(
        "both algorithms agree: {}",
        liu.k_cores() == cores.as_slice()
    );
    liu.drop_cache()?;

    let degeneracy = cores.iter().copied().max().unwrap_or(0);
    println!("degeneracy (max coreness): {degeneracy}");
    // edges are indexed by their position in the CSR, so look an edge's coreness up by index
    println!(
        "coreness of edge 0: {}",
        cores.first().copied().unwrap_or(0)
    );
    let mut histogram = vec![0usize; degeneracy as usize + 1];
    for &k in &cores {
        histogram[k as usize] += 1;
    }
    for (k, count) in histogram.iter().enumerate().filter(|(_, c)| **c > 0) {
        println!("  coreness {k:>3}: {count} edges");
    }
    Ok(())
}
