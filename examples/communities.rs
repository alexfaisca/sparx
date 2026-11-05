//! Community detection: a global partition (GVE-Louvain) and a local community around a seed node
//! (heat kernel PageRank, HK-relax).
//!
//! cargo run --release --example communities -- [graph] [threads] [seed node]
//!
//! `[graph]` defaults to a small bundled fixture.
use sparx::communities::{gve_louvain::AlgoGVELouvain, hk_relax::HKRelax};
use sparx::graph::GraphMemoryMap;

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;
    let seed: usize = args.get(3).map_or(Ok(0), |s| s.parse())?;

    let mut graph = GraphMemoryMap::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph |V| = {}, |E| = {}", graph.size(), graph.width());

    let result = run(&graph, seed);
    // also when a run fails
    graph.drop_cache()?;
    result
}

fn run(graph: &GraphMemoryMap<(), (), u32>, seed: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut louvain = AlgoGVELouvain::<_, _, _, f64>::new(graph)?;
    println!(
        "louvain: {} communities, modularity {:.4}",
        louvain.community_count(),
        louvain.partition_modularity()
    );
    println!(
        "node {seed} is in community {}",
        louvain.communities()[seed]
    );
    louvain.drop_cache()?;

    // heat kernel diffusion time `t` and error tolerance `eps`
    let hk = HKRelax::new(graph, 45., 0.01, vec![seed], None, None)?;
    let community = hk.compute()?;
    println!(
        "hk-relax around node {seed}: {} nodes, volume {}, conductance {:.4}",
        community.size, community.width, community.conductance
    );
    Ok(())
}
