//! Algorithms whose results don't depend on the number of threads (each node's value computed on
//! its own, or sums taken in a fixed order) are checked to return the same result bit for bit at
//! every thread count; exact betweenness, whose threads' sums are added separately, to within
//! rounding.
use crate::centralities::exact::{
    ExactBetweennessCentrality, ExactClosenessCentrality, ExactHarmonicCentrality,
    ExactLinCentrality,
};
use crate::centralities::hyper_ball::HyperBall8;
use crate::communities::hk_relax::HKRelax;
use crate::components::{strong::StronglyConnectedComponents, weak::ConnectedComponents};
use crate::graph::label::VoidLabel;
use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
use crate::k_truss::triangles::test::{random_edges, undirected_graph_with_threads};

use std::io::Write;

const THREADS: [u8; 4] = [1, 3, 8, 16];

/// A sparse random graph of several components, with isolated nodes.
fn edges() -> Vec<(usize, usize)> {
    random_edges(400, 0.006, 31)
}

/// `edges` as arcs (each once, `u -> v` for `u < v`), plus some reversed arcs to close cycles.
fn arcs() -> Vec<(usize, usize)> {
    edges()
        .into_iter()
        .enumerate()
        .map(|(i, (u, v))| if i % 3 == 0 { (v, u) } else { (u, v) })
        .collect()
}

fn directed_graph_with_threads(
    n: usize,
    arcs: &[(usize, usize)],
    threads: u8,
) -> TestGraph<VoidLabel, VoidLabel, usize, crate::graph::Directed> {
    let dir = unique_test_dir("threads_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("d.mtx");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
    writeln!(f, "{n} {n} {}", arcs.len()).unwrap();
    for &(u, v) in arcs {
        writeln!(f, "{} {}", u + 1, v + 1).unwrap();
    }
    drop(f);
    let graph = crate::graph::DiGraph::from_file(
        &path,
        Some(format!("threads_d_{}", unique_run_tag())),
        Some(threads),
    )
    .unwrap();
    TestGraph::new(graph, Some(dir))
}

/// `f`'s result on the undirected graph at every thread count, each checked equal to the first.
fn same_on_every_thread_count<T: PartialEq + std::fmt::Debug>(
    what: &str,
    f: impl Fn(&crate::graph::GraphMemoryMap<VoidLabel, VoidLabel, usize>) -> T,
) {
    let edges = edges();
    let mut first = None;
    for threads in THREADS {
        let g = undirected_graph_with_threads(&format!("{what}_{threads}"), 400, &edges, threads);
        let got = f(&g);
        match &first {
            None => first = Some(got),
            Some(expected) => assert_eq!(&got, expected, "{what}: {threads} threads vs 1"),
        }
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn exact_centralities_are_the_same_whatever_the_threads() {
    same_on_every_thread_count("closeness", |g| {
        let c = ExactClosenessCentrality::<_, _, _, f64>::new(g, Some(true)).unwrap();
        bits(c.centralities())
    });
    same_on_every_thread_count("harmonic", |g| {
        let c = ExactHarmonicCentrality::<_, _, _, f64>::new(g, Some(true)).unwrap();
        bits(c.centralities())
    });
    same_on_every_thread_count("lin", |g| {
        let c = ExactLinCentrality::<_, _, _, f64>::new(g).unwrap();
        bits(c.centralities())
    });
}

#[test]
fn hyperball_is_the_same_whatever_the_threads() {
    same_on_every_thread_count("hyperball", |g| {
        let hb = HyperBall8::new(g).unwrap();
        (
            bits(hb.geodesic_distances()),
            bits(hb.harmonic_geodesic_distances()),
        )
    });
}

#[test]
fn hk_relax_is_the_same_whatever_the_threads() {
    same_on_every_thread_count("hk_relax", |g| {
        // the seed: a node of the largest component, of positive degree
        let seed = (0..g.size()).max_by_key(|&u| g.node_degree(u)).unwrap();
        let c = HKRelax::new(g, 5., 1e-4, vec![seed], None, None)
            .unwrap()
            .compute()
            .unwrap();
        (
            c.nodes
                .iter()
                .map(|&(u, h)| (u, h.to_bits()))
                .collect::<Vec<_>>(),
            c.size,
            c.width,
        )
    });
}

#[test]
fn components_are_the_same_whatever_the_threads() {
    same_on_every_thread_count("weak", |g| {
        let c = ConnectedComponents::new(g).unwrap();
        (c.components().to_vec(), c.sizes().to_vec())
    });
    let arcs = arcs();
    let mut first = None;
    for threads in THREADS {
        let d = directed_graph_with_threads(400, &arcs, threads);
        let weak = ConnectedComponents::new(&*d).unwrap();
        let strong = StronglyConnectedComponents::new(&*d).unwrap();
        let got = (
            weak.components().to_vec(),
            strong.components().to_vec(),
            strong.sizes().to_vec(),
        );
        match &first {
            None => first = Some(got),
            Some(expected) => assert_eq!(&got, expected, "directed: {threads} threads vs 1"),
        }
    }
}

/// Documented to differ at most in the last bits across thread counts (and to repeat exactly on
/// as many threads, which `centralities::betweenness`'s tests check).
#[test]
fn exact_betweenness_is_the_same_whatever_the_threads_to_within_rounding() {
    let edges = edges();
    let mut first: Option<Vec<f64>> = None;
    for threads in THREADS {
        let g =
            undirected_graph_with_threads(&format!("betweenness_{threads}"), 400, &edges, threads);
        let b = ExactBetweennessCentrality::new(&*g, false).unwrap();
        let got = b.centralities().to_vec();
        match &first {
            None => first = Some(got),
            Some(expected) => {
                for (u, (x, y)) in got.iter().zip(expected).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-9 * y.abs().max(1.),
                        "node {u}: {x} on {threads} threads, {y} on 1"
                    );
                }
            }
        }
    }
}
