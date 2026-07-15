//! Test matrix of the triangle based algorithms: varied (multi)graphs, directed and undirected,
//! every index width (`Ix`), count width (`Tx`) and triangle type, against brute force references
//! and each other.
use super::burkhardt_et_al::AlgoBurkhardtEtAl;
use super::clustering_coefficient::ClusteringCoefficient;
use super::count::{TriangleCount, TriangleCounter};
use super::directed::test::{brute_force_directed, directed_multigraph_edges};
use super::directed::{DirectedTriangleType, DirectedTriangles};
use super::oracle_test::{brute_force_directed_trussness, brute_force_trussness};
use super::pkt::AlgoPKT;
use super::triangles::Triangles;
use super::triangles::test::{brute_force_triangles, clique_edges, random_edges};
use crate::graph::label::VoidLabel;
use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
use crate::graph::{GraphMemoryMap, IndexType};

use std::io::Write;
use std::path::PathBuf;

/// Writes `edges` (0-based) as a MatrixMarket file over `n` nodes, `symmetric` (undirected) or
/// `general` (directed).
fn write_mtx(name: &str, symmetric: bool, n: usize, edges: &[(usize, usize)]) -> PathBuf {
    let dir = unique_test_dir("k_truss_matrix_test");
    std::fs::create_dir_all(&dir).expect("error creating tmp dir");
    let path = dir.join(format!("{name}.mtx"));
    let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
    let kind = if symmetric { "symmetric" } else { "general" };
    writeln!(f, "%%MatrixMarket matrix coordinate pattern {kind}").unwrap();
    writeln!(f, "{n} {n} {}", edges.len()).unwrap();
    for &(u, v) in edges {
        writeln!(f, "{} {}", u + 1, v + 1).unwrap();
    }
    path
}

fn open<Ix: IndexType>(path: &PathBuf) -> GraphMemoryMap<VoidLabel, VoidLabel, Ix> {
    let id = format!(
        "matrix_{}_{}_{}",
        path.file_stem().unwrap().to_string_lossy(),
        std::any::type_name::<Ix>(),
        unique_run_tag()
    );
    GraphMemoryMap::from_file(path, Some(id), Some(4)).expect("graph parse should succeed")
}

fn counts<C: TriangleCounter>(counts: &[C]) -> Vec<usize> {
    counts.iter().map(|c| c.count()).collect()
}

fn widen<Tx: TriangleCount>(values: &[Tx]) -> Vec<usize> {
    values.iter().map(|v| v.to_usize()).collect()
}

/// Undirected graphs: shapes without triangles, sparse and dense ones, multigraphs.
fn undirected_graphs() -> Vec<(&'static str, usize, Vec<(usize, usize)>)> {
    let star: Vec<_> = (1..20).map(|v| (0, v)).collect();
    let tree: Vec<_> = (1..31).map(|v| ((v - 1) / 2, v)).collect();
    // two disjoint cliques of different sizes, joined by a path
    let mut cliques = clique_edges(5);
    cliques.extend(clique_edges(7).into_iter().map(|(u, v)| (u + 8, v + 8)));
    cliques.extend([(4, 5), (5, 6), (6, 7), (7, 8)]);
    // a dense multigraph: a random graph with every 3rd edge doubled, plus self-loops
    let mut dense_multi = random_edges(25, 0.5, 13);
    let doubled: Vec<_> = dense_multi.iter().step_by(3).copied().collect();
    dense_multi.extend(doubled);
    dense_multi.extend([(0, 0), (7, 7)]);
    vec![
        ("star", 20, star),
        ("tree", 31, tree),
        ("cliques", 15, cliques),
        ("k10", 10, clique_edges(10)),
        ("dense_multi", 25, dense_multi),
    ]
}

/// Directed graphs: a DAG (no cycles), a complete digraph (every reciprocal pair), a ring with
/// chords both ways, a star, and multigraphs with reciprocal and parallel edges and self-loops.
fn directed_graphs() -> Vec<(&'static str, usize, Vec<(usize, usize)>)> {
    let dag: Vec<_> = random_edges(20, 0.35, 21).into_iter().collect();
    let complete: Vec<_> = (0..7)
        .flat_map(|u| (0..7).filter(move |&v| v != u).map(move |v| (u, v)))
        .collect();
    let mut ring: Vec<_> = (0..16).map(|u| (u, (u + 1) % 16)).collect();
    ring.extend((0..16).step_by(2).map(|u| ((u + 2) % 16, u)));
    ring.extend((0..16).step_by(3).map(|u| (u, (u + 5) % 16)));
    let star: Vec<_> = (1..12).flat_map(|v| [(0, v), (v, 0)]).collect();
    vec![
        ("dag", 20, dag),
        ("complete", 7, complete),
        ("ring", 16, ring),
        ("directed_star", 12, star),
        ("multi_a", 24, directed_multigraph_edges(24, 31)),
        ("multi_b", 30, directed_multigraph_edges(30, 47)),
    ]
}

/// Every undirected algorithm's results at index width `Ix` (counts and trussness as `u32`).
fn undirected_results<Ix: IndexType>(
    path: &PathBuf,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<u64>) {
    let g = open::<Ix>(path);
    let mut t = Triangles::<_, _, _, u32>::new_tx(&g).unwrap();
    let triangles = counts(t.all_triangles());
    t.drop_cache().unwrap();
    let mut pkt = AlgoPKT::<_, _, _, u32>::new_tx(&g, 4).unwrap();
    let pkt_k = widen(pkt.k_trusses());
    pkt.drop_cache().unwrap();
    let mut bea = AlgoBurkhardtEtAl::<_, _, _, u32>::new_tx(&g).unwrap();
    let bea_k = widen(bea.k_trusses());
    bea.drop_cache().unwrap();
    let cc = ClusteringCoefficient::new_with_triangles(
        &g,
        Triangles::<_, _, _, u32>::new_tx(&g).unwrap(),
    )
    .unwrap();
    let cc: Vec<u64> = (0..g.size())
        .map(|u| cc.get_node_clusteringcoefficient(u).to_bits())
        .collect();
    (triangles, pkt_k, bea_k, cc)
}

#[test]
fn undirected_graphs_match_brute_force_at_every_index_width() {
    for (name, n, edges) in undirected_graphs() {
        let path = write_mtx(name, true, n, &edges);
        let g = open::<usize>(&path);
        let simple = !name.contains("multi");

        let results = undirected_results::<usize>(&path);
        assert_eq!(results.0, brute_force_triangles(&g), "{name}: triangles");
        assert_eq!(results.1, results.2, "{name}: pkt and burkhardt et al.");
        if simple {
            let expected: Vec<usize> = brute_force_trussness(&g)
                .iter()
                .map(|&k| k as usize)
                .collect();
            assert_eq!(results.1, expected, "{name}: trussness");
        }
        assert_eq!(
            undirected_results::<u32>(&path),
            results,
            "{name}: Ix = u32"
        );
        assert_eq!(
            undirected_results::<u16>(&path),
            results,
            "{name}: Ix = u16"
        );
        if g.width() < u8::MAX as usize {
            assert_eq!(undirected_results::<u8>(&path), results, "{name}: Ix = u8");
        }
    }
}

/// Every directed algorithm's results at index width `Ix`, for `triangle_type`.
fn directed_results<Ix: IndexType>(
    path: &PathBuf,
    triangle_type: DirectedTriangleType,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let g = open::<Ix>(path);
    let mut t = DirectedTriangles::<_, _, _, u32>::new_tx(&g, triangle_type).unwrap();
    let triangles = counts(t.all_triangles());
    t.drop_cache().unwrap();
    let mut pkt = AlgoPKT::<_, _, _, u32>::new_directed_tx(&g, triangle_type, 4).unwrap();
    let pkt_k = widen(pkt.k_trusses());
    pkt.drop_cache().unwrap();
    let mut bea = AlgoBurkhardtEtAl::<_, _, _, u32>::new_directed_tx(&g, triangle_type).unwrap();
    let bea_k = widen(bea.k_trusses());
    bea.drop_cache().unwrap();
    (triangles, pkt_k, bea_k)
}

#[test]
fn directed_graphs_match_brute_force_at_every_index_width() {
    for (name, n, edges) in directed_graphs() {
        let path = write_mtx(name, false, n, &edges);
        let g = open::<usize>(&path);
        for triangle_type in DirectedTriangleType::ALL {
            let roles = triangle_type.roles();
            let results = directed_results::<usize>(&path, triangle_type);
            let what = format!("{name} {triangle_type:?}");
            assert_eq!(
                results.0,
                brute_force_directed(&g, roles),
                "{what}: triangles"
            );
            let expected = brute_force_directed_trussness(&g, roles);
            assert_eq!(results.1, expected, "{what}: pkt");
            assert_eq!(results.2, expected, "{what}: burkhardt et al.");
            // the other widths on a few types (each run builds a graph)
            if matches!(
                triangle_type,
                DirectedTriangleType::Cycle
                    | DirectedTriangleType::Out
                    | DirectedTriangleType::CycleFeedForward
            ) {
                assert_eq!(
                    directed_results::<u32>(&path, triangle_type),
                    results,
                    "{what}: Ix = u32"
                );
                assert_eq!(
                    directed_results::<u16>(&path, triangle_type),
                    results,
                    "{what}: Ix = u16"
                );
                if g.width() < u8::MAX as usize {
                    assert_eq!(
                        directed_results::<u8>(&path, triangle_type),
                        results,
                        "{what}: Ix = u8"
                    );
                }
            }
        }
        // a DAG has no directed cycles
        if name == "dag" {
            let t = DirectedTriangles::new(&g, DirectedTriangleType::Cycle).unwrap();
            assert!(counts(t.all_triangles()).iter().all(|&c| c == 0));
        }
    }
}

/// The same results at every count width (`Tx`) that holds them.
#[test]
fn every_count_width_agrees() {
    fn undirected<Tx: TriangleCount>(path: &PathBuf) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let g = open::<usize>(path);
        let mut t = Triangles::<_, _, _, Tx>::new_tx(&g).unwrap();
        let triangles = counts(t.all_triangles());
        t.drop_cache().unwrap();
        let mut pkt = AlgoPKT::<_, _, _, Tx>::new_tx(&g, 4).unwrap();
        let pkt_k = widen(pkt.k_trusses());
        pkt.drop_cache().unwrap();
        let mut bea = AlgoBurkhardtEtAl::<_, _, _, Tx>::new_tx(&g).unwrap();
        let bea_k = widen(bea.k_trusses());
        bea.drop_cache().unwrap();
        (triangles, pkt_k, bea_k)
    }
    fn directed<Tx: TriangleCount>(
        path: &PathBuf,
        t: DirectedTriangleType,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let g = open::<usize>(path);
        let mut tri = DirectedTriangles::<_, _, _, Tx>::new_tx(&g, t).unwrap();
        let triangles = counts(tri.all_triangles());
        tri.drop_cache().unwrap();
        let mut pkt = AlgoPKT::<_, _, _, Tx>::new_directed_tx(&g, t, 4).unwrap();
        let pkt_k = widen(pkt.k_trusses());
        pkt.drop_cache().unwrap();
        let mut bea = AlgoBurkhardtEtAl::<_, _, _, Tx>::new_directed_tx(&g, t).unwrap();
        let bea_k = widen(bea.k_trusses());
        bea.drop_cache().unwrap();
        (triangles, pkt_k, bea_k)
    }
    let path = write_mtx("widths", true, 25, &random_edges(25, 0.4, 5));
    let expected = undirected::<u8>(&path);
    assert_eq!(undirected::<u16>(&path), expected);
    assert_eq!(undirected::<u32>(&path), expected);
    assert_eq!(undirected::<u64>(&path), expected);
    let path = write_mtx(
        "widths_directed",
        false,
        24,
        &directed_multigraph_edges(24, 8),
    );
    for t in [
        DirectedTriangleType::Cycle,
        DirectedTriangleType::InMiddleman,
        DirectedTriangleType::CycleFeedForward,
    ] {
        let expected = directed::<u8>(&path, t);
        assert_eq!(directed::<u16>(&path, t), expected, "{t:?}");
        assert_eq!(directed::<u32>(&path, t), expected, "{t:?}");
        assert_eq!(directed::<u64>(&path, t), expected, "{t:?}");
    }
}

/// Directed counts beyond `u8`: arcs of a 3-node digraph each repeated 20 times put every edge in
/// hundreds of triangles.
#[test]
fn directed_wide_counts() {
    let mut edges = vec![];
    for _ in 0..20 {
        edges.extend([(0, 1), (1, 2), (2, 0), (0, 2)]);
    }
    let path = write_mtx("directed_parallel20", false, 3, &edges);
    let g = open::<usize>(&path);
    let t = DirectedTriangleType::CycleFeedForward;
    let error = DirectedTriangles::new(&g, t)
        .err()
        .expect("u8 counts should overflow")
        .to_string();
    assert!(
        error.contains("more than can be counted (u8)"),
        "unexpected error: {error}"
    );
    let mut tri = DirectedTriangles::<_, _, _, u16>::new_tx(&g, t).unwrap();
    assert_eq!(
        counts(tri.all_triangles()),
        brute_force_directed(&g, t.roles())
    );
    tri.drop_cache().unwrap();
    let expected = brute_force_directed_trussness(&g, t.roles());
    let mut pkt = AlgoPKT::<_, _, _, u16>::new_directed_tx(&g, t, 4).unwrap();
    assert_eq!(widen(pkt.k_trusses()), expected);
    pkt.drop_cache().unwrap();
    let error = AlgoPKT::new_directed(&g, t)
        .err()
        .expect("u8 trussness should overflow")
        .to_string();
    assert!(error.contains("(u8)"), "unexpected error: {error}");
}

/// Directed PKT (with partial roles, where triangles count for some of their edges only) gives
/// the same results run after run, on any number of threads.
#[test]
fn directed_pkt_is_deterministic_across_threads() {
    let path = write_mtx(
        "directed_race",
        false,
        60,
        &directed_multigraph_edges(60, 77),
    );
    let g = open::<u32>(&path);
    for t in [
        DirectedTriangleType::Out,
        DirectedTriangleType::CycleFeedForward,
    ] {
        let expected = brute_force_directed_trussness(&open::<usize>(&path), t.roles());
        for threads in [8, 16] {
            for run in 0..15 {
                let mut pkt = AlgoPKT::<_, _, _, u16>::new_directed_tx(&g, t, threads).unwrap();
                assert_eq!(
                    widen(pkt.k_trusses()),
                    expected,
                    "{t:?}, {threads} threads, run {run}"
                );
                pkt.drop_cache().unwrap();
            }
        }
    }
}

/// The verifier rejects decompositions that aren't.
#[test]
fn verifier_rejects_wrong_trussness() {
    let g = open::<usize>(&write_mtx("verifier", true, 10, &clique_edges(10)));
    let mut pkt = AlgoPKT::new(&g).unwrap();
    let mut wrong = pkt.k_trusses().to_vec();
    super::verify_k_trusses(&g, &wrong).unwrap();
    // a whole clique one higher: not a member of that truss
    wrong.iter_mut().for_each(|k| *k += 1);
    assert!(super::verify_k_trusses(&g, &wrong).is_err());
    // one edge one lower (on both sides): not maximal
    let mut wrong = pkt.k_trusses().to_vec();
    wrong[0] -= 1;
    assert!(super::verify_k_trusses(&g, &wrong).is_err());
    pkt.drop_cache().unwrap();
}

/// GPU counts at any width match the CPU's, and overflow like them.
#[cfg(feature = "gpu")]
#[test]
fn gpu_triangles_at_every_count_width() {
    use crate::gpu::GpuBackend;
    let Ok(backend) = crate::gpu::WgpuBackend::new() else {
        eprintln!("skipping GPU test, no GPU adapter");
        return;
    };
    let mut edges = vec![];
    for _ in 0..20 {
        edges.extend([(0, 1), (1, 2), (0, 2)]);
    }
    let path = write_mtx("gpu_parallel20", true, 3, &edges);
    let g = open::<u32>(&path);
    let device_graph = backend.upload_graph(&g).unwrap();
    let error = Triangles::new_gpu(&g, &backend, &device_graph)
        .err()
        .expect("u8 should overflow")
        .to_string();
    assert!(
        error.contains("more than can be counted (u8)"),
        "unexpected error: {error}"
    );
    for result in [
        counts(
            Triangles::<_, _, _, u16>::new_gpu_tx(&g, &backend, &device_graph)
                .unwrap()
                .all_triangles(),
        ),
        counts(
            Triangles::<_, _, _, u64>::new_gpu_tx(&g, &backend, &device_graph)
                .unwrap()
                .all_triangles(),
        ),
    ] {
        assert_eq!(
            result,
            counts(
                Triangles::<_, _, _, u32>::new_tx(&g)
                    .unwrap()
                    .all_triangles()
            )
        );
    }
}

/// From precomputed triangles (without the internal "edges in some triangle" flags, so every
/// edge without support is peeled), both decompositions give the same results.
#[test]
fn directed_k_truss_from_precomputed_triangles() {
    let path = write_mtx(
        "directed_precomputed",
        false,
        30,
        &directed_multigraph_edges(30, 61),
    );
    let g = open::<u32>(&path);
    let reference = open::<usize>(&path);
    for t in DirectedTriangleType::ALL {
        let expected = brute_force_directed_trussness(&reference, t.roles());
        let triangles = DirectedTriangles::<_, _, _, u16>::new_tx(&g, t).unwrap();
        let mut pkt = AlgoPKT::new_directed_with_triangles(&g, triangles).unwrap();
        assert_eq!(widen(pkt.k_trusses()), expected, "pkt {t:?}");
        pkt.drop_cache().unwrap();
        let triangles = DirectedTriangles::<_, _, _, u16>::new_tx(&g, t).unwrap();
        let mut bea = AlgoBurkhardtEtAl::new_directed_with_triangles(&g, triangles).unwrap();
        assert_eq!(widen(bea.k_trusses()), expected, "burkhardt et al. {t:?}");
        bea.drop_cache().unwrap();
    }
}
