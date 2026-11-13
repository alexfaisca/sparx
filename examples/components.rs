//! Structure: connected components, biconnected components (articulation points and bridges),
//! strongly connected components, and degree assortativity.
//!
//! ```sh
//! cargo run --release --example components -- [graph] [threads]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4.
use sparx::assortativity::{Correlation, degree_assortativity};
use sparx::components::biconnected::Biconnectivity;
use sparx::components::strong::StronglyConnectedComponents;
use sparx::components::weak::ConnectedComponents;
use sparx::graph::{DiGraph, Graph};

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;

    // the same file, as an undirected and as a directed graph
    let mut graph = Graph::<(), (), u32>::from_file(path, None, Some(threads))?;
    let mut directed = DiGraph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph, &directed);
    // the graphs' cache entries are dropped, also when the run failed
    graph.drop_cache()?;
    directed.drop_cache()?;
    result
}

fn run(
    graph: &Graph<(), (), u32>,
    directed: &DiGraph<(), (), u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut components = ConnectedComponents::new(graph)?;
    let largest = components.sizes().iter().copied().max().unwrap_or(0);
    println!(
        "connected components: {}, the largest has {largest} nodes",
        components.count()
    );
    components.drop_cache()?;

    let mut bcc = Biconnectivity::new(graph)?;
    println!(
        "biconnected components: {}, articulation points: {}, bridges: {}",
        bcc.count(),
        bcc.articulation_points().len(),
        bcc.bridges().len()
    );
    bcc.drop_cache()?;

    let mut strong = StronglyConnectedComponents::new(directed)?;
    println!(
        "strongly connected components (directed): {}",
        strong.count()
    );
    strong.drop_cache()?;

    // Newman's degree assortativity: positive when nodes link to nodes of similar degrees
    println!(
        "degree assortativity: {:.4} (Pearson), {:.4} (Spearman, of the degrees' ranks)",
        degree_assortativity(graph, Correlation::Pearson)?,
        degree_assortativity(graph, Correlation::Spearman)?
    );
    Ok(())
}
