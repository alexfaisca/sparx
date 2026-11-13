//! Weighted shortest paths: every node's distance from a source (Dijkstra's algorithm, or
//! Δ-stepping on many threads), a shortest path tree, and point-to-point A* guided by landmarks.
//!
//! ```sh
//! cargo run --release --example shortest_paths -- [graph.mtx] [threads] [source]
//! ```
//!
//! `[graph.mtx]` is a MatrixMarket file whose values are the edges' weights; by default, a
//! weighted 300×300 grid is written to the system's temporary directory (and removed after).
//! `[threads]` defaults to 4, `[source]` to node 0.
use sparx::graph::{Graph, Weighted};
use sparx::trails::astar::{Landmarks, astar};
use sparx::trails::dijkstra::{DijkstraDists, ShortestPathEngine};
use std::io::Write;

type WeightedGraph = Graph<(), Weighted<f64>, u32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let threads: u8 = args.get(2).map_or(Ok(4), |t| t.parse())?;
    let source: usize = args.get(3).map_or(Ok(0), |s| s.parse())?;
    let grid = std::env::temp_dir().join(format!("sparx_grid_{}.mtx", std::process::id()));
    let path = match args.get(1) {
        Some(path) => path.into(),
        None => {
            write_grid(&grid, 300)?;
            grid.clone()
        }
    };

    let mut graph = WeightedGraph::from_file(&path, None, Some(threads))?;
    println!("graph: {} nodes, {} arcs", graph.size(), graph.width());

    let result = run(&graph, source);
    // the graph's cache entry is dropped (and the generated grid removed), also when the run
    // failed
    graph.drop_cache()?;
    let _ = std::fs::remove_file(&grid);
    result
}

fn run(graph: &WeightedGraph, source: usize) -> Result<(), Box<dyn std::error::Error>> {
    // `Auto` picks Δ-stepping on large graphs and several threads, Dijkstra's algorithm otherwise
    // (the same distances, bit for bit)
    let mut dists = DijkstraDists::with_engine(graph, source, ShortestPathEngine::Auto)?;
    let distances = dists.get_distances().to_vec();
    let farthest = (0..graph.size())
        .filter(|&v| distances[v].is_finite())
        .max_by(|&a, &b| distances[a].total_cmp(&distances[b]))
        .ok_or("empty graph")?;
    println!(
        "from node {source}: {} nodes reached, the farthest is node {farthest}, at {:.2}",
        dists.reachable(),
        distances[farthest]
    );
    // a shortest path, read off the shortest path tree (found when first asked for)
    let path = dists.path_to(farthest)?.ok_or("unreachable")?;
    println!("  a shortest path to it has {} nodes", path.len());
    dists.drop_cache()?;

    // A*, guided by the distances to and from 8 landmarks (kept in the graph's cache)
    let mut landmarks = Landmarks::new(graph, 8)?;
    let found = astar(graph, source, farthest, &landmarks)?.ok_or("unreachable")?;
    println!(
        "A* to node {farthest}: distance {:.2} (the same), {} nodes expanded of {}",
        found.distance,
        found.expanded,
        graph.size()
    );
    landmarks.drop_cache()?;
    Ok(())
}

/// A `side`×`side` grid, each node linked to its right and lower neighbours by an edge of a
/// (pseudo-random, the same every run) weight in `[1, 10)`, as a MatrixMarket file.
fn write_grid(path: &std::path::Path, side: usize) -> std::io::Result<()> {
    let mut state = 0x5eed_u64;
    let mut weight = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        1. + (state >> 11) as f64 / (1u64 << 53) as f64 * 9.
    };
    let n = side * side;
    let mut edges = vec![];
    for r in 0..side {
        for c in 0..side {
            let u = r * side + c;
            if c + 1 < side {
                edges.push((u + 1, u, weight()));
            }
            if r + 1 < side {
                edges.push((u + side, u, weight()));
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "%%MatrixMarket matrix coordinate real symmetric")?;
    writeln!(f, "{n} {n} {}", edges.len())?;
    for (u, v, w) in edges {
        // 1-based, the larger node first (the lower triangle, as symmetric files are)
        writeln!(f, "{} {} {w:.3}", u + 1, v + 1)?;
    }
    f.flush()
}
