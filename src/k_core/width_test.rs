//! The k-core decompositions' coreness widths (`Cx`): every width computes the same
//! decomposition, degrees right at a width's maximum fit it, and degrees past it are an error
//! (never a wrap around, nor a result cached for later).
use super::_verify::reference_coreness;
use super::batagelj_zaversnik::AlgoBatageljZaversnik;
use super::count::CoreCount;
use super::directed::{CoreDegree, DirectedKCores};
use super::liu_et_al::AlgoLiuEtAl;
use crate::components::test_utils::{graph, random_arcs};
use crate::graph::{CacheFile, Graph, UndirectedMode};

use std::path::Path;

fn as_usize<Cx: CoreCount>(cores: &[Cx]) -> Vec<usize> {
    cores.iter().map(|c| c.to_usize()).collect()
}

/// Batagelj and Zaversnik's and Liu et al.'s (with several thread counts) coreness of `g`, as
/// `Cx`s.
fn all_cores<Cx: CoreCount>(g: &Graph<(), ()>) -> Vec<(String, Vec<usize>)> {
    let mut all = vec![];
    let bz = AlgoBatageljZaversnik::<_, _, _, Cx>::new_cx(g).expect("batagelj-zaversnik");
    for v in 0..g.size() {
        assert_eq!(bz.coreness(v), bz.k_cores()[v]);
    }
    all.push(("bz".to_string(), as_usize(bz.k_cores())));
    drop(bz);
    for threads in [1, 3, 8, 64] {
        let liu = AlgoLiuEtAl::<_, _, _, Cx>::new_cx(g, threads).expect("liu et al.");
        for v in 0..g.size() {
            assert_eq!(liu.coreness(v), liu.k_cores()[v]);
        }
        all.push((format!("liu ({threads} threads)"), as_usize(liu.k_cores())));
    }
    all
}

fn assert_all_widths<F: Fn(&[(String, Vec<usize>)], &str)>(g: &Graph<(), ()>, check: F) {
    check(&all_cores::<u8>(g), "u8");
    check(&all_cores::<u16>(g), "u16");
    check(&all_cores::<u32>(g), "u32");
    check(&all_cores::<u64>(g), "u64");
}

/// Edges of the complete graph on `n` nodes.
fn clique(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|u| (u + 1..n).map(move |v| (u, v))).collect()
}

/// Edges of a star: node 0 and `leaves` others.
fn star(leaves: usize) -> Vec<(usize, usize)> {
    (1..=leaves).map(|v| (0, v)).collect()
}

#[test]
fn every_width_computes_the_reference_decomposition() {
    for (name, n, m, seed) in [("sparse", 80, 120, 1), ("dense", 60, 1500, 2), ("tiny", 3, 4, 3)] {
        let arcs = random_arcs(n, m, seed);
        let g = graph(name, "symmetric", n, &arcs);
        // simple (one edge per pair) and multigraph (parallel edges, self-loops) views
        for mode in [UndirectedMode::Collapse, UndirectedMode::Each] {
            let u = g.to_undirected_with(mode).unwrap();
            let expected = reference_coreness(u).unwrap();
            assert_all_widths(u, |all, width| {
                for (algo, got) in all {
                    assert_eq!(got, &expected, "{name}, {mode:?}, {algo}, {width}");
                }
            });
        }
    }
}

#[test]
fn degrees_of_a_widths_maximum_fit_it() {
    // every node of degree 255 (and coreness 255): Liu et al.'s k reaches u8::MAX
    let g = graph("k256", "symmetric", 256, &clique(256));
    let u = g.to_undirected().unwrap();
    assert_all_widths(u, |all, width| {
        for (algo, got) in all {
            assert!(got.iter().all(|&c| c == 255), "{algo}, {width}");
        }
    });
    // a node of degree 255 whose coreness is 1
    let g = graph("star255", "symmetric", 256, &star(255));
    let u = g.to_undirected().unwrap();
    assert_all_widths(u, |all, width| {
        for (algo, got) in all {
            assert_eq!(got, &vec![1; 256], "{algo}, {width}");
        }
    });
    // u16's maximum
    let g = graph("star65535", "symmetric", 65536, &star(65535));
    let u = g.to_undirected().unwrap();
    let bz = AlgoBatageljZaversnik::<_, _, _, u16>::new_cx(u).unwrap();
    assert!(bz.k_cores().iter().all(|&c| c == 1));
    let liu = AlgoLiuEtAl::<_, _, _, u16>::new_cx(u, 4).unwrap();
    assert!(liu.k_cores().iter().all(|&c| c == 1));
}

/// Asserts `result` is the error of a degree that doesn't fit a `Cx`.
fn assert_too_wide<T>(result: Result<T, Box<dyn std::error::Error>>, what: &str) {
    match result {
        Ok(_) => panic!("{what} should fail: a degree doesn't fit the coreness type"),
        Err(e) => assert!(e.to_string().contains("more than the coreness type"), "{what}: {e}"),
    }
}

#[test]
fn degrees_past_a_widths_maximum_are_an_error() {
    let g = graph("star256", "symmetric", 257, &star(256));
    let u = g.to_undirected().unwrap();
    assert_too_wide(AlgoBatageljZaversnik::new(u), "batagelj-zaversnik (u8)");
    for threads in [1, 4] {
        assert_too_wide(AlgoLiuEtAl::new_with_conf(u, threads), "liu et al. (u8)");
    }
    // neither leaves a result behind, to be taken for a cached one
    for file in [CacheFile::KCoreBZ, CacheFile::KCoreLEA] {
        let path = u.build_cache_filename(file, None).unwrap();
        assert!(!Path::new(&path).exists(), "{path} left behind");
    }
    assert_too_wide(AlgoBatageljZaversnik::get_or_compute(u), "cached batagelj-zaversnik (u8)");
    assert_too_wide(AlgoLiuEtAl::get_or_compute(u), "cached liu et al. (u8)");
    // a wider type holds them
    let bz = AlgoBatageljZaversnik::<_, _, _, u16>::new_cx(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), vec![1; 257]);
    let liu = AlgoLiuEtAl::<_, _, _, u16>::new_cx(u, 4).unwrap();
    assert_eq!(as_usize(liu.k_cores()), vec![1; 257]);

    // the clique on 257 nodes: coreness 256
    let g = graph("k257", "symmetric", 257, &clique(257));
    let u = g.to_undirected().unwrap();
    assert_too_wide(AlgoBatageljZaversnik::new(u), "batagelj-zaversnik (u8)");
    assert_too_wide(AlgoLiuEtAl::new(u), "liu et al. (u8)");
    let bz = AlgoBatageljZaversnik::<_, _, _, u16>::new_cx(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), vec![256; 257]);
    let liu = AlgoLiuEtAl::<_, _, _, u16>::new_cx(u, 4).unwrap();
    assert_eq!(as_usize(liu.k_cores()), vec![256; 257]);

    // past u16's maximum
    let g = graph("star65536", "symmetric", 65537, &star(65536));
    let u = g.to_undirected().unwrap();
    assert_too_wide(AlgoBatageljZaversnik::<_, _, _, u16>::new_cx(u), "batagelj-zaversnik (u16)");
    assert_too_wide(AlgoLiuEtAl::<_, _, _, u16>::new_cx(u, 4), "liu et al. (u16)");
    let bz = AlgoBatageljZaversnik::<_, _, _, u32>::new_cx(u).unwrap();
    assert!(bz.k_cores().iter().all(|&c| c == 1));
    let liu = AlgoLiuEtAl::<_, _, _, u64>::new_cx(u, 4).unwrap();
    assert!(liu.k_cores().iter().all(|&c| c == 1));
}

#[test]
fn cached_results_of_another_width_are_recomputed() {
    let arcs = random_arcs(41, 300, 4);
    let g = graph("cached", "symmetric", 41, &arcs);
    let u = g.to_undirected().unwrap();
    let expected = reference_coreness(u).unwrap();

    let bz = AlgoBatageljZaversnik::<_, _, _, u16>::new_cx(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), expected);
    drop(bz);
    let bz = AlgoBatageljZaversnik::get_or_compute(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), expected, "u16 cache read as u8");
    drop(bz);
    let bz = AlgoBatageljZaversnik::<_, _, _, u64>::get_or_compute_cx(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), expected, "u8 cache read as u64");
    drop(bz);
    // and the same width's cache is reused
    let bz = AlgoBatageljZaversnik::<_, _, _, u64>::get_or_compute_cx(u).unwrap();
    assert_eq!(as_usize(bz.k_cores()), expected, "u64 cache read as u64");
    drop(bz);

    let liu = AlgoLiuEtAl::<_, _, _, u32>::new_cx(u, 3).unwrap();
    assert_eq!(as_usize(liu.k_cores()), expected);
    drop(liu);
    let liu = AlgoLiuEtAl::get_or_compute(u).unwrap();
    assert_eq!(as_usize(liu.k_cores()), expected, "u32 cache read as u8");
    drop(liu);
    let liu = AlgoLiuEtAl::<_, _, _, u16>::get_or_compute_cx(u).unwrap();
    assert_eq!(as_usize(liu.k_cores()), expected, "u8 cache read as u16");
}

#[test]
fn directed_k_cores_of_every_width_agree() {
    for (name, n, m, seed) in [("sparse", 50, 150, 5), ("dense", 40, 900, 6)] {
        let arcs = random_arcs(n, m, seed);
        let g = graph(name, "general", n, &arcs);
        for degree in [CoreDegree::In, CoreDegree::Out, CoreDegree::Total] {
            let c8 = DirectedKCores::new(&g, degree).unwrap();
            let want = as_usize(c8.k_cores());
            drop(c8);
            let c16 = DirectedKCores::<_, _, _, u16>::new_cx(&g, degree).unwrap();
            assert_eq!(as_usize(c16.k_cores()), want, "{name}, {degree:?}, u16");
            drop(c16);
            let c32 = DirectedKCores::<_, _, _, u32>::new_cx(&g, degree).unwrap();
            assert_eq!(as_usize(c32.k_cores()), want, "{name}, {degree:?}, u32");
            drop(c32);
            let c64 = DirectedKCores::<_, _, _, u64>::new_cx(&g, degree).unwrap();
            for v in 0..n {
                assert_eq!(c64.coreness(v) as usize, want[v]);
            }
        }
    }
}

#[test]
fn directed_degrees_past_a_widths_maximum_are_an_error() {
    // the complete digraph on 256 nodes: in- and out-degrees 255, total degrees 510
    let arcs: Vec<_> = (0..256).flat_map(|u| (0..256).filter(move |&v| v != u).map(move |v| (u, v))).collect();
    let g = graph("complete256", "general", 256, &arcs);
    for degree in [CoreDegree::In, CoreDegree::Out] {
        let c = DirectedKCores::new(&g, degree).unwrap();
        assert!(c.k_cores().iter().all(|&c| c == 255), "{degree:?}");
    }
    assert_too_wide(DirectedKCores::new(&g, CoreDegree::Total), "total (u8)");
    let path = g.build_cache_filename(CacheFile::KCoreDirected(CoreDegree::Total), None).unwrap();
    assert!(!Path::new(&path).exists(), "{path} left behind");
    let c = DirectedKCores::<_, _, _, u16>::new_cx(&g, CoreDegree::Total).unwrap();
    assert!(c.k_cores().iter().all(|&c| c == 510));

    // one node of in-degree 256 (out-degree 0): only its in-cores don't fit a u8
    let arcs: Vec<_> = (1..=256).map(|v| (v, 0)).collect();
    let g = graph("in_star256", "general", 257, &arcs);
    assert_too_wide(DirectedKCores::new(&g, CoreDegree::In), "in (u8)");
    assert_too_wide(DirectedKCores::new(&g, CoreDegree::Total), "total (u8)");
    // the hub has no arcs out, so neither do the leaves once it's peeled
    let out = DirectedKCores::new(&g, CoreDegree::Out).unwrap();
    assert!(out.k_cores().iter().all(|&c| c == 0), "out-cores of a star into its hub");
    let in_cores = DirectedKCores::<_, _, _, u16>::new_cx(&g, CoreDegree::In).unwrap();
    assert!(in_cores.k_cores().iter().all(|&c| c == 0), "in-cores of a star into its hub");
}
