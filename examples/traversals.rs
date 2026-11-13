//! Traversals: BFS distances from a source node, and Euler trails (Hierholzer's algorithm), e.g. to
//! walk a de Bruijn graph's edges.
//!
//! ```sh
//! cargo run --release --example traversals -- [graph] [threads] [source]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4, `[source]` to node 0.
use sparx::graph::{DiGraph, IndexType};
use sparx::trails::{bfs::BFSDists, hierholzer::AlgoHierholzer};

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;
    let source: usize = args.get(3).map_or(Ok(0), |s| s.parse())?;

    let mut graph = DiGraph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph, source);
    // the graph's cache entry is dropped, also when the run failed
    graph.drop_cache()?;
    result
}

fn run(graph: &DiGraph<(), (), u32>, source: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut bfs = BFSDists::new(graph, source)?;
    let eccentricity = bfs
        .get_distances()
        .iter()
        .map(|d| d.index())
        .max()
        .unwrap_or(0);
    println!(
        "BFS from node {source}: {} nodes reached, eccentricity {eccentricity}",
        bfs.reachable()
    );
    bfs.drop_cache()?;

    let mut euler = AlgoHierholzer::new(graph)?;
    let trails = euler.trail_number();
    let longest = (0..trails)
        .filter_map(|i| euler.get_trail(i).map(<[u32]>::len))
        .max()
        .unwrap_or(0);
    println!("{trails} Euler trail(s) cover every arc, the longest has length {longest}");
    euler.drop_cache()?;
    Ok(())
}
