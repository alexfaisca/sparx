//! Community detection: global partitions (GVE-Louvain, GVE-Leiden, label propagation and
//! spectral clustering) and a local community around a seed node (heat kernel diffusion,
//! HK-Relax).
//!
//! ```sh
//! cargo run --release --example communities -- [graph] [threads] [seed]
//! ```
//!
//! `[graph]` defaults to a small bundled fixture, `[threads]` to 4, `[seed]` to node 0.
use sparx::communities::gve_louvain::AlgoGVELouvain;
use sparx::communities::hk_relax::HKRelax;
use sparx::communities::label_propagation::{LabelPropagation, LpaEngine, LpaOptions};
use sparx::communities::spectral::{SpectralClustering, SpectralOptions};
use sparx::graph::Graph;

const FIXTURE: &str = "tests/fixtures/ggcat/random_graph_1_5.lz4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or(FIXTURE, String::as_str);
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;
    let seed: usize = args.get(3).map_or(Ok(0), |s| s.parse())?;

    let mut graph = Graph::<(), (), u32>::from_file(path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph, seed);
    // the graph's cache entry is dropped, also when the run failed
    graph.drop_cache()?;
    result
}

fn run(graph: &Graph<(), (), u32>, seed: usize) -> Result<(), Box<dyn std::error::Error>> {
    // `f64` community weights (`f32` halves their memory)
    let mut louvain = AlgoGVELouvain::<_, _, _, f64>::new(graph)?;
    println!(
        "Louvain: {} communities, modularity {:.4}; node {seed} is in community {}",
        louvain.community_count(),
        louvain.partition_modularity(),
        louvain.node_community(seed)
    );
    louvain.drop_cache()?;

    let mut leiden = AlgoGVELouvain::<_, _, _, f64>::new_leiden(graph)?;
    println!(
        "Leiden: {} communities, modularity {:.4}",
        leiden.community_count(),
        leiden.partition_modularity()
    );
    leiden.drop_cache()?;

    // semi-synchronous (the same communities whatever the threads), then asynchronous (faster)
    for engine in [LpaEngine::SemiSynchronous, LpaEngine::Asynchronous] {
        let options = LpaOptions {
            engine,
            ..LpaOptions::default()
        };
        let mut lpa = LabelPropagation::with_options(graph, options)?;
        println!(
            "label propagation ({engine:?}): {} communities, modularity {:.4}",
            lpa.community_count(),
            lpa.partition_modularity()
        );
        lpa.drop_cache()?;
    }

    // 8 clusters, by the compressive engine (filtered random signals)
    let mut spectral = SpectralClustering::<_, _, _, f64>::new(graph, SpectralOptions::new(8))?;
    println!(
        "spectral clustering: {} clusters, modularity {:.4}",
        spectral.community_count(),
        spectral.partition_modularity()
    );
    spectral.drop_cache()?;

    // heat kernel diffusion time `t` and error tolerance `eps`
    let hk = HKRelax::new(graph, 45., 0.01, vec![seed], None, None)?;
    let community = hk.compute()?;
    println!(
        "HK-Relax around node {seed}: {} nodes, volume {}, conductance {:.4}",
        community.size, community.width, community.conductance
    );
    Ok(())
}
