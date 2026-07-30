//! Closeness and harmonic centralities: estimated with HyperBall, checked against the exact values.
//!
//! cargo run --release --example centralities -- [graph] [threads]
//!
//! `[graph]` defaults to a small bundled fixture. The exact centralities run a BFS from every
//! node, so only use them on small graphs.
use sparx::centralities::{exact::ExactClosenessCentrality, hyper_ball::HyperBall8};
use sparx::graph::DiGraph;

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;

    let mut graph = DiGraph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph |V| = {}, |E| = {}", graph.size(), graph.width());

    let result = run(&graph);
    // also when a run fails
    graph.drop_cache()?;
    result
}

fn run(graph: &DiGraph<(), (), u32>) -> Result<(), Box<dyn std::error::Error>> {
    // 2^8 registers per counter: a relative standard error of about 1.04 / sqrt(2^8) = 6.5%
    let mut hyperball = HyperBall8::new(graph)?;
    // normalized by each node's reachable set
    let closeness = hyperball
        .get_or_compute_closeness_centrality(true)?
        .to_vec();
    let harmonic = hyperball.get_or_compute_harmonic_centrality(true)?.to_vec();
    hyperball.drop_cache()?;

    let mut exact = ExactClosenessCentrality::<_, _, _, f64, _>::new(graph, Some(true))?;
    let n = graph.size();
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

    let top = (0..n)
        .max_by(|&a, &b| harmonic[a].total_cmp(&harmonic[b]))
        .ok_or("empty graph")?;
    println!(
        "most central node (harmonic): {top}, centrality {:.4}",
        harmonic[top]
    );
    println!(
        "hyperball closeness mean relative error vs exact: {:.2}%",
        100. * mean_relative_error
    );
    Ok(())
}
