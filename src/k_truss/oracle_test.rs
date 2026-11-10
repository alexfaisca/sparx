//! k-truss decompositions and clustering coefficients checked against brute force references on
//! generated graphs (the datasets based tests need external fixtures).
use super::burkhardt_et_al::AlgoBurkhardtEtAl;
use super::clustering_coefficient::ClusteringCoefficient;
use super::pkt::AlgoPKT;
use super::triangles::test::*;

use std::collections::{BTreeMap, BTreeSet};

use super::count::TriangleCounter;

/// Each CSR edge's trussness (the largest `k` such that the edge is in the `k`-truss, the
/// subgraph whose edges are each in at least `k - 2` of its triangles), by peeling the
/// undirected simple graph (self-loops have trussness 2).
pub(crate) fn brute_force_trussness(g: &Graph) -> Vec<u8> {
    let key = |u: usize, v: usize| (u.min(v), u.max(v));
    let mut adjacency: Vec<BTreeSet<usize>> = (0..g.size())
        .map(|u| g.neighbours(u).unwrap().filter(|&v| v != u).collect())
        .collect();
    let mut support: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for u in 0..g.size() {
        for &v in &adjacency[u] {
            if u < v {
                support.insert((u, v), adjacency[u].intersection(&adjacency[v]).count());
            }
        }
    }

    let mut trussness = BTreeMap::new();
    let mut k = 2;
    while !support.is_empty() {
        // peel every edge whose support (in what's left) is at most k - 2
        while let Some((&(u, v), _)) = support.iter().find(|&(_, &s)| s + 2 <= k) {
            support.remove(&(u, v));
            trussness.insert((u, v), k as u8);
            adjacency[u].remove(&v);
            adjacency[v].remove(&u);
            let common: Vec<usize> = adjacency[u].intersection(&adjacency[v]).copied().collect();
            for w in common {
                *support.get_mut(&key(u, w)).unwrap() -= 1;
                *support.get_mut(&key(v, w)).unwrap() -= 1;
            }
        }
        k += 1;
    }

    let mut result = vec![];
    for u in 0..g.size() {
        for v in g.neighbours(u).unwrap() {
            result.push(if u == v { 2 } else { trussness[&key(u, v)] });
        }
    }
    result
}

/// Simple graphs (no parallel edges) the references apply to.
fn simple_fixtures() -> Vec<(&'static str, TestGraph)> {
    fixtures()
        .into_iter()
        .filter(|(name, _)| *name != "multi")
        .collect()
}

#[test]
fn pkt_matches_brute_force() {
    for (name, g) in simple_fixtures() {
        for threads in [1, 4, 16] {
            let mut pkt = AlgoPKT::new_with_conf(&g, threads).unwrap();
            assert_eq!(
                pkt.k_trusses(),
                brute_force_trussness(&g),
                "{name}, {threads} threads"
            );
            // the (independent) verifier the dataset based tests use agrees
            super::verify_k_trusses(&g, pkt.k_trusses()).unwrap();
            pkt.drop_cache().unwrap();
        }
    }
}

#[test]
fn burkhardt_et_al_matches_brute_force() {
    for (name, g) in simple_fixtures() {
        let mut bea = AlgoBurkhardtEtAl::new(&g).unwrap();
        assert_eq!(bea.k_trusses(), brute_force_trussness(&g), "{name}");
        bea.drop_cache().unwrap();
    }
}

/// On graphs with parallel edges both algorithms count triangles per combination of parallel
/// edges, so they must agree.
#[test]
fn k_truss_algorithms_agree_on_multigraphs() {
    for (name, g) in fixtures() {
        let mut bea = AlgoBurkhardtEtAl::new(&g).unwrap();
        for threads in [1, 4, 16] {
            let mut pkt = AlgoPKT::new_with_conf(&g, threads).unwrap();
            assert_eq!(
                pkt.k_trusses(),
                bea.k_trusses(),
                "{name}, {threads} threads"
            );
            pkt.drop_cache().unwrap();
        }
        bea.drop_cache().unwrap();
    }
}

/// Local clustering coefficients (edges among a node's distinct neighbours over the possible
/// ones), their average and the transitivity (`3 * triangles / connected triples`), as defined
/// by networkx, by brute force.
fn brute_force_clustering(g: &Graph) -> (Vec<f64>, f64, f64) {
    let adjacency: Vec<BTreeSet<usize>> = (0..g.size())
        .map(|u| g.neighbours(u).unwrap().filter(|&v| v != u).collect())
        .collect();
    let (mut closed, mut triples) = (0., 0.);
    let local: Vec<f64> = (0..g.size())
        .map(|u| {
            let k = adjacency[u].len() as f64;
            let links = adjacency[u]
                .iter()
                .map(|&v| adjacency[u].intersection(&adjacency[v]).count())
                .sum::<usize>() as f64
                / 2.;
            closed += links;
            triples += k * (k - 1.) / 2.;
            if k > 1. {
                links / (k * (k - 1.) / 2.)
            } else {
                0.
            }
        })
        .collect();
    let average = local.iter().sum::<f64>() / g.size() as f64;
    (local, average, closed / triples)
}

#[test]
fn clustering_coefficient_matches_brute_force() {
    for (name, g) in simple_fixtures() {
        let cc = ClusteringCoefficient::new(&g).unwrap();
        let (local, average, transitivity) = brute_force_clustering(&g);
        for (u, expected) in local.iter().enumerate() {
            let got = cc.get_node_clusteringcoefficient(u);
            assert!(
                (got - expected).abs() < 1e-12,
                "{name}: node {u}'s coefficient {got}, expected {expected}"
            );
        }
        let got = cc.get_average_clustering_coefficient();
        assert!(
            (got - average).abs() < 1e-12,
            "{name}: average {got}, expected {average}"
        );
        let got = cc.get_graph_transitivity();
        assert!(
            (got - transitivity).abs() < 1e-12,
            "{name}: transitivity {got}, expected {transitivity}"
        );
    }
}

/// Edge `0 -- 1` plus `pages` nodes adjacent to both: `0 -- 1` is in `pages` triangles, every other
/// edge in 1.
fn book(pages: usize) -> TestGraph {
    let mut edges = vec![(0, 1)];
    for w in 2..pages + 2 {
        edges.extend([(0, w), (1, w)]);
    }
    undirected_graph(&format!("book{pages}"), pages + 2, &edges)
}

/// Counts and trussness are `u8`s: up to 253 triangles per edge (trussness 255) everything is
/// computed, beyond that each algorithm fails with an error rather than wrapping around.
#[test]
fn k_truss_support_range() {
    use super::triangles::Triangles;

    let g = book(253);
    let expected = brute_force_trussness(&g);
    assert!(expected.iter().all(|&k| k == 3));
    let mut pkt = AlgoPKT::new(&g).unwrap();
    assert_eq!(pkt.k_trusses(), expected);
    pkt.drop_cache().unwrap();
    let mut bea = AlgoBurkhardtEtAl::new(&g).unwrap();
    assert_eq!(bea.k_trusses(), expected);
    bea.drop_cache().unwrap();

    // 254 triangles can be counted, but not turned into a trussness
    let g = book(254);
    let mut triangles = Triangles::new(&g).unwrap();
    assert_eq!(triangles.triangles(0), 254);
    triangles.drop_cache().unwrap();
    for error in [AlgoPKT::new(&g).err(), AlgoBurkhardtEtAl::new(&g).err()] {
        let error = error.expect("trussness above 255 should fail").to_string();
        assert!(
            error.contains("can't be represented"),
            "unexpected error: {error}"
        );
    }

    // 300 triangles can't even be counted
    let g = book(300);
    let error = Triangles::new(&g)
        .expect_err("300 triangles should fail")
        .to_string();
    assert!(
        error.contains("more than can be counted"),
        "unexpected error: {error}"
    );
    #[cfg(feature = "gpu")]
    if let Ok(backend) = crate::gpu::WgpuBackend::new() {
        use crate::gpu::GpuBackend;
        let device_graph = backend.upload_graph(&g).unwrap();
        let error = Triangles::new_gpu(&g, &backend, &device_graph)
            .expect_err("300 triangles should fail on the GPU")
            .to_string();
        assert!(
            error.contains("more than can be counted"),
            "unexpected error: {error}"
        );
    }
}

/// Fed triangles counted on a GPU, every triangle based algorithm gives the CPU's results.
#[cfg(feature = "gpu")]
#[test]
fn k_truss_from_gpu_triangles() {
    use super::triangles::Triangles;
    use crate::gpu::GpuBackend;
    let backend = match crate::gpu::WgpuBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("skipping GPU test, no GPU adapter: {e}");
            return;
        }
    };
    for (name, g) in fixtures() {
        let device_graph = backend.upload_graph(&g).unwrap();
        let gpu_triangles = || Triangles::new_gpu(&g, &backend, &device_graph).unwrap();

        let mut cpu = AlgoPKT::new(&g).unwrap();
        let expected = cpu.k_trusses().to_vec();
        cpu.drop_cache().unwrap();
        let mut pkt = AlgoPKT::new_with_triangles(&g, gpu_triangles()).unwrap();
        assert_eq!(pkt.k_trusses(), expected, "{name}: pkt");
        pkt.drop_cache().unwrap();
        let mut bea = AlgoBurkhardtEtAl::new_with_triangles(&g, gpu_triangles()).unwrap();
        assert_eq!(bea.k_trusses(), expected, "{name}: burkhardt et al.");
        bea.drop_cache().unwrap();

        let cpu = ClusteringCoefficient::new(&g).unwrap();
        let gpu = ClusteringCoefficient::new_with_triangles(&g, gpu_triangles()).unwrap();
        assert_eq!(
            cpu.get_graph_transitivity(),
            gpu.get_graph_transitivity(),
            "{name}"
        );
        for u in 0..g.size() {
            assert_eq!(
                cpu.get_node_clusteringcoefficient(u),
                gpu.get_node_clusteringcoefficient(u),
                "{name}: node {u}"
            );
        }
    }
}

/// PKT used to mark a sub-level's edges processed while other threads still processed them,
/// losing triangles (wrong trussness in ~10% of multi-threaded runs): repeated runs must all be
/// right.
#[test]
fn pkt_is_deterministic_across_threads() {
    let g = undirected_graph("pkt_race", 300, &random_edges(300, 0.08, 7));
    let expected = brute_force_trussness(&g);
    for threads in [8, 16] {
        for run in 0..50 {
            let mut pkt = AlgoPKT::new_with_conf(&g, threads).unwrap();
            assert_eq!(pkt.k_trusses(), expected, "{threads} threads, run {run}");
            pkt.drop_cache().unwrap();
        }
    }
}

/// Each CSR edge's trussness for directed triangles of the given roles, by brute force: an edge's
/// support is the number of combinations of edges (still in the graph) on the other two sides of
/// a triangle which put it in a selected role, and the `k`-truss is what's left after repeatedly
/// removing edges with support below `k - 2`.
pub(crate) fn brute_force_directed_trussness(g: &DiGraph, roles: u8) -> Vec<usize> {
    use super::directed::role;
    let edges: Vec<(usize, usize)> = (0..g.size())
        .flat_map(|u| g.neighbours(u).unwrap().map(move |v| (u, v)))
        .collect();
    let mut by_pair = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (e, &(u, v)) in edges.iter().enumerate() {
        by_pair.entry((u, v)).or_default().push(e);
    }
    let mut alive = vec![true; edges.len()];
    let support = |e: usize, alive: &[bool]| -> usize {
        let (u, v) = edges[e];
        if u == v {
            return 0;
        }
        let alive_count = |a: usize, b: usize| {
            by_pair
                .get(&(a, b))
                .map_or(0, |es| es.iter().filter(|&&f| alive[f]).count())
        };
        (0..g.size())
            .filter(|&w| w != u && w != v)
            .map(|w| {
                let mut count = 0;
                for f_u_to_w in [true, false] {
                    for g_v_to_w in [true, false] {
                        if roles & role(f_u_to_w, g_v_to_w) != 0 {
                            let f = if f_u_to_w {
                                alive_count(u, w)
                            } else {
                                alive_count(w, u)
                            };
                            let g = if g_v_to_w {
                                alive_count(v, w)
                            } else {
                                alive_count(w, v)
                            };
                            count += f * g;
                        }
                    }
                }
                count
            })
            .sum()
    };

    let mut trussness = vec![0; edges.len()];
    let mut supports: Vec<usize> = (0..edges.len()).map(|e| support(e, &alive)).collect();
    let mut k = 2;
    while alive.iter().any(|&a| a) {
        // edges left in the k-truss but not the (k + 1)-truss
        while let Some(e) = (0..edges.len()).find(|&e| alive[e] && supports[e] + 2 <= k) {
            alive[e] = false;
            trussness[e] = k;
            // only edges touching the removed one's endpoints shared triangles with it
            let (u, v) = edges[e];
            for f in 0..edges.len() {
                let (a, b) = edges[f];
                if alive[f] && (a == u || a == v || b == u || b == v) {
                    supports[f] = support(f, &alive);
                }
            }
        }
        k += 1;
    }
    trussness
}

/// Directed k-truss decompositions (both algorithms, every triangle type) on a directed
/// multigraph with reciprocal edges, parallel edges and self-loops.
#[test]
fn directed_k_truss_matches_brute_force() {
    use super::directed::DirectedTriangleType;
    use super::directed::test::{directed_graph, directed_multigraph_edges};
    let g = directed_graph("truss", 40, &directed_multigraph_edges(40, 9));
    for triangle_type in DirectedTriangleType::ALL {
        let expected = brute_force_directed_trussness(&g, triangle_type.roles());
        let as_usize = |k: &[u16]| k.iter().map(|&k| k as usize).collect::<Vec<_>>();
        for threads in [1, 4, 16] {
            let mut pkt =
                AlgoPKT::<_, _, _, u16, _>::new_directed_tx(&g, triangle_type, threads).unwrap();
            assert_eq!(
                as_usize(pkt.k_trusses()),
                expected,
                "pkt {triangle_type:?}, {threads} threads"
            );
            pkt.drop_cache().unwrap();
        }
        let mut bea =
            AlgoBurkhardtEtAl::<_, _, _, u16, _>::new_directed_tx(&g, triangle_type).unwrap();
        assert_eq!(
            as_usize(bea.k_trusses()),
            expected,
            "burkhardt et al. {triangle_type:?}"
        );
        bea.drop_cache().unwrap();
    }
}

/// Each type's decomposition is cached in its own file, apart from the undirected one.
#[test]
fn directed_k_truss_cache_per_type() {
    use super::directed::DirectedTriangleType;
    use super::directed::test::{directed_graph, directed_multigraph_edges};
    use crate::graph::CacheFile;
    let g = directed_graph("truss_cache", 30, &directed_multigraph_edges(30, 2));
    let mut cycle = AlgoPKT::new_directed(&g, DirectedTriangleType::Cycle).unwrap();
    let mut out = AlgoBurkhardtEtAl::new_directed(&g, DirectedTriangleType::Out).unwrap();
    assert_eq!(cycle.triangle_type(), Some(DirectedTriangleType::Cycle));
    let pkt_file = g
        .build_cache_filename(
            CacheFile::KTrussPKTDirected(DirectedTriangleType::Cycle),
            None,
        )
        .unwrap();
    let bea_file = g
        .build_cache_filename(
            CacheFile::KTrussBEADirected(DirectedTriangleType::Out),
            None,
        )
        .unwrap();
    assert!(pkt_file.contains("cycle_ktrusspkt") && std::path::Path::new(&pkt_file).exists());
    assert!(bea_file.contains("out_ktrussbea") && std::path::Path::new(&bea_file).exists());
    cycle.drop_cache().unwrap();
    out.drop_cache().unwrap();
    assert!(!std::path::Path::new(&pkt_file).exists() && !std::path::Path::new(&bea_file).exists());
}

/// Wider counts (`Tx`) handle what `u8`s can't: with each side of a triangle twenty-fold, every
/// edge is in 400 triangles.
#[test]
fn wide_triangle_counts() {
    use super::triangles::Triangles;
    let mut edges = vec![];
    for _ in 0..20 {
        edges.extend([(0, 1), (1, 2), (0, 2)]);
    }
    let g = undirected_graph("parallel20", 3, &edges);

    let error = Triangles::new(&g)
        .expect_err("u8 counts should overflow")
        .to_string();
    assert!(
        error.contains("more than can be counted (u8)"),
        "unexpected error: {error}"
    );

    let mut t = Triangles::<_, _, _, u16>::new_tx(&g).unwrap();
    let counts: Vec<usize> = t.all_triangles().iter().map(|c| c.count()).collect();
    assert_eq!(counts, brute_force_triangles(&g));
    assert!(counts.iter().all(|&c| c == 400));
    t.drop_cache().unwrap();

    // all edges are in a single (multi-)triangle: trussness 402
    let mut pkt = AlgoPKT::<_, _, _, u16>::new_tx(&g, 4).unwrap();
    assert!(pkt.k_trusses().iter().all(|&k| k == 402));
    pkt.drop_cache().unwrap();
    let mut bea = AlgoBurkhardtEtAl::<_, _, _, u32>::new_tx(&g).unwrap();
    assert!(bea.k_trusses().iter().all(|&k| k == 402));
    bea.drop_cache().unwrap();
    let cc = ClusteringCoefficient::new_with_triangles(
        &g,
        Triangles::<_, _, _, u64>::new_tx(&g).unwrap(),
    )
    .unwrap();
    assert!(cc.get_graph_transitivity().is_finite());
}
