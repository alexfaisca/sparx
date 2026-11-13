//! Triangle counting, k-truss decomposition (PKT and Burkhardt et al.) and clustering coefficients.
//!
//! ```sh
//! cargo run --release --example k_truss -- [graph] [threads]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4.
use sparx::graph::Graph;
use sparx::k_truss::{
    burkhardt_et_al::AlgoBurkhardtEtAl, clustering_coefficient::ClusteringCoefficient,
    count::TriangleCounter, pkt::AlgoPKT, triangles::Triangles,
};

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
    // per-edge triangle counts: each triangle is counted once on each of its edges, in both
    // directions
    let mut triangles = Triangles::new(graph)?;
    let total: usize = triangles
        .all_triangles()
        .iter()
        .map(|c| c.count())
        .sum::<usize>()
        / 6;
    println!("triangles: {total}");
    triangles.drop_cache()?;

    let mut pkt = AlgoPKT::new_with_conf(graph, graph.thread_num())?;
    let trussness = pkt.k_trusses().to_vec();
    pkt.drop_cache()?;
    println!(
        "max trussness: {}",
        trussness.iter().copied().max().unwrap_or(0)
    );

    let mut bea = AlgoBurkhardtEtAl::new(graph)?;
    println!(
        "PKT and Burkhardt et al. agree: {}",
        bea.k_trusses() == trussness.as_slice()
    );
    bea.drop_cache()?;

    let mut cc = ClusteringCoefficient::new(graph)?;
    println!(
        "average clustering coefficient: {:.4}, transitivity: {:.4}",
        cc.get_average_clustering_coefficient(),
        cc.get_graph_transitivity()
    );
    cc.drop_cache()?;
    Ok(())
}
