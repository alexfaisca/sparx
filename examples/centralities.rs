//! Centralities: closeness and harmonic estimated with HyperBall (checked against the exact
//! values), PageRank, and the most central nodes by betweenness (KADABRA's sampling).
//!
//! ```sh
//! cargo run --release --example centralities -- [graph] [threads]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4. The exact centralities run a
//! BFS from every node: only use them on small graphs.
use sparx::centralities::betweenness::{Betweenness, BetweennessEngine, BetweennessOptions};
use sparx::centralities::exact::ExactClosenessCentrality;
use sparx::centralities::hyper_ball::HyperBall8;
use sparx::centralities::pagerank::PageRank;
use sparx::graph::DiGraph;

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;

    let mut graph = DiGraph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph);
    // the graph's cache entry is dropped, also when the run failed
    graph.drop_cache()?;
    result
}

fn run(graph: &DiGraph<(), (), u32>) -> Result<(), Box<dyn std::error::Error>> {
    let n = graph.size();
    // 2^8 registers per counter: a relative standard error of about 1.04 / sqrt(2^8) = 6.5%
    let mut hyperball = HyperBall8::new(graph)?;
    // normalized by each node's reachable set
    let closeness = hyperball
        .get_or_compute_closeness_centrality(true)?
        .to_vec();
    let harmonic = hyperball.get_or_compute_harmonic_centrality(true)?.to_vec();
    hyperball.drop_cache()?;

    let mut exact = ExactClosenessCentrality::<_, _, _, f64, _>::new(graph, Some(true))?;
    let mean_relative_error = (0..n)
        .map(|v| {
            let e = exact.get_node_centrality(v);
            if e == 0. {
                0.
            } else {
                (closeness[v] - e).abs() / e
            }
        })
        .sum::<f64>()
        / n as f64;
    exact.drop_cache()?;
    println!(
        "HyperBall closeness: mean relative error {:.2}% against the exact values",
        100. * mean_relative_error
    );
    let top = (0..n)
        .max_by(|&a, &b| harmonic[a].total_cmp(&harmonic[b]))
        .ok_or("empty graph")?;
    println!(
        "most central node (harmonic): {top}, centrality {:.4}",
        harmonic[top]
    );

    let mut pagerank = PageRank::<_, _, _, f64, _>::new(graph)?;
    let ranks = pagerank.ranks();
    let top = (0..n)
        .max_by(|&a, &b| ranks[a].total_cmp(&ranks[b]))
        .ok_or("empty graph")?;
    println!(
        "PageRank: converged in {} iterations, highest rank {:.4e} (node {top})",
        pagerank.iterations(),
        ranks[top]
    );
    pagerank.drop_cache()?;

    // the 5 most central nodes by betweenness, by sampling shortest paths until they're told apart
    let options = BetweennessOptions {
        engine: BetweennessEngine::Kadabra,
        top_k: Some(5),
        ..BetweennessOptions::default()
    };
    let mut betweenness = Betweenness::with_options(graph, options)?;
    println!(
        "betweenness (KADABRA, {} samples): top 5 nodes {:?}",
        betweenness.samples(),
        betweenness.top()
    );
    betweenness.drop_cache()?;
    Ok(())
}
