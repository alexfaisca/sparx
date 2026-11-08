use super::*;
use crate::components::test_utils::random_arcs;
use crate::directed_algorithms_test::graph_ix;
use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
use std::io::Write;

/// An undirected graph of `n` nodes and `edges` (parallel edges and self-loops kept).
fn undirected(
    name: &str,
    n: usize,
    edges: &[(usize, usize)],
    threads: u8,
) -> TestGraph<(), (), u32> {
    let dir = unique_test_dir("assortativity_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate pattern symmetric").unwrap();
    writeln!(f, "{n} {n} {}", edges.len()).unwrap();
    for &(u, v) in edges {
        writeln!(f, "{} {}", u.max(v) + 1, u.min(v) + 1).unwrap();
    }
    drop(f);
    let id = format!("assortativity_{name}_{}", unique_run_tag());
    TestGraph::new(
        Graph::from_file(&path, Some(id), Some(threads)).unwrap(),
        Some(dir),
    )
}

/// The arcs networkx pairs values over: each undirected edge both ways, a self-loop once.
fn undirected_arcs(edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut arcs = vec![];
    for &(u, v) in edges {
        arcs.push((u, v));
        if u != v {
            arcs.push((v, u));
        }
    }
    arcs
}

/// Pearson's correlation of pairs, in floats (two passes).
fn pearson_of(pairs: &[(f64, f64)]) -> f64 {
    let m = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / m;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / m;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        xy += (x - mx) * (y - my);
        xx += (x - mx) * (x - mx);
        yy += (y - my) * (y - my);
    }
    if pairs.is_empty() || xx == 0.0 || yy == 0.0 {
        return f64::NAN;
    }
    xy / (xx.sqrt() * yy.sqrt())
}

/// Average ranks (from 1) of `values`.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    let mut r = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let average = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[order[k]] = average;
        }
        i = j + 1;
    }
    r
}

fn correlation_of(pairs: &[(f64, f64)], correlation: Correlation) -> f64 {
    match correlation {
        Correlation::Pearson => pearson_of(pairs),
        Correlation::Spearman => {
            let rx = ranks(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
            let ry = ranks(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
            pearson_of(&rx.into_iter().zip(ry).collect::<Vec<_>>())
        }
    }
}

/// The attribute assortativity of the pairs of labels, from the mixing matrix (networkx's way).
fn attribute_of(pairs: &[(usize, usize)]) -> f64 {
    use std::collections::HashMap;
    let m = pairs.len() as f64;
    let (mut trace, mut a, mut b) = (0.0, HashMap::new(), HashMap::new());
    for &(x, y) in pairs {
        if x == y {
            trace += 1.0 / m;
        }
        *a.entry(x).or_insert(0.0) += 1.0 / m;
        *b.entry(y).or_insert(0.0) += 1.0 / m;
    }
    let ab: f64 = a.iter().map(|(k, v)| v * b.get(k).unwrap_or(&0.0)).sum();
    if pairs.is_empty() || ab == 1.0 {
        return f64::NAN;
    }
    (trace - ab) / (1.0 - ab)
}

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

/// Undirected degree pairs, networkx's way: degrees count self-loops twice.
fn undirected_degree_pairs(n: usize, edges: &[(usize, usize)]) -> Vec<(f64, f64)> {
    let mut degree = vec![0usize; n];
    for &(u, v) in edges {
        degree[u] += 1;
        degree[v] += 1;
    }
    undirected_arcs(edges)
        .into_iter()
        .map(|(u, v)| (degree[u] as f64, degree[v] as f64))
        .collect()
}

#[test]
fn degree_assortativity_of_small_shapes() {
    let cases: Vec<(&str, usize, Vec<(usize, usize)>)> = vec![
        ("path", 4, vec![(0, 1), (1, 2), (2, 3)]),
        ("star", 5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]),
        (
            "triangle_tail",
            5,
            vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)],
        ),
        ("loops", 4, vec![(0, 0), (0, 1), (1, 2), (2, 2), (2, 3)]),
        ("parallel", 4, vec![(0, 1), (0, 1), (1, 2), (2, 3), (1, 3)]),
        // self-loops on linked nodes, doubled ones, and parallel edges between them
        (
            "linked_loops",
            6,
            vec![
                (0, 0),
                (1, 1),
                (1, 1),
                (0, 1),
                (0, 1),
                (1, 2),
                (2, 2),
                (2, 3),
                (3, 4),
                (4, 4),
                (5, 5),
            ],
        ),
        // regular: every degree the same, undefined
        ("cycle", 5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        ("empty", 3, vec![]),
    ];
    for (name, n, edges) in cases {
        let g = undirected(name, n, &edges, 2);
        let pairs = undirected_degree_pairs(n, &edges);
        for c in [Correlation::Pearson, Correlation::Spearman] {
            let r = degree_assortativity(&g, c).unwrap();
            let expected = correlation_of(&pairs, c);
            assert!(close(r, expected), "{name} {c:?}: {r} != {expected}");
        }
    }
}

#[test]
fn degree_assortativity_of_random_graphs() {
    for seed in 0..30u64 {
        let n = 5 + (seed as usize * 7) % 60;
        let m = (seed as usize * 13) % (3 * n) + 1;
        let edges = random_arcs(n, m, seed);
        let pairs = undirected_degree_pairs(n, &edges);
        let mut seen = vec![];
        for threads in [1, 3, 8] {
            let g = undirected(&format!("random_{seed}_{threads}"), n, &edges, threads);
            for c in [Correlation::Pearson, Correlation::Spearman] {
                let r = degree_assortativity(&g, c).unwrap();
                let expected = correlation_of(&pairs, c);
                assert!(close(r, expected), "seed {seed} {c:?}: {r} != {expected}");
                seen.push((threads, c, r));
            }
        }
        // exact: the same, bit for bit, whatever the threads
        for &(_, c, r) in &seen {
            let first = seen.iter().find(|s| s.1 == c).unwrap().2;
            assert!(r.to_bits() == first.to_bits() || (r.is_nan() && first.is_nan()));
        }
    }
}

#[test]
fn degree_assortativity_in_every_width() {
    // a star with a few extra edges, its centre's degree past one byte, two, ...
    for leaves in [300usize, 70_000] {
        let n = leaves + 1;
        let mut edges: Vec<(usize, usize)> = (1..n).map(|v| (0, v)).collect();
        edges.extend((1..50).map(|v| (v, v + 1)));
        let g = undirected(&format!("star_{leaves}"), n, &edges, 4);
        let pairs = undirected_degree_pairs(n, &edges);
        for c in [Correlation::Pearson, Correlation::Spearman] {
            let r = degree_assortativity(&g, c).unwrap();
            let expected = correlation_of(&pairs, c);
            assert!(close(r, expected), "{leaves} {c:?}: {r} != {expected}");
        }
    }
}

#[test]
fn directed_degree_assortativity_of_random_graphs() {
    use DegreeType::*;
    for seed in 0..25u64 {
        let n = 4 + (seed as usize * 5) % 50;
        let m = (seed as usize * 11) % (4 * n) + 1;
        let arcs = random_arcs(n, m, seed + 100);
        let g = graph_ix::<u32>(&format!("assortativity_{seed}"), "general", n, &arcs);
        let (mut out, mut into) = (vec![0usize; n], vec![0usize; n]);
        for &(u, v) in &arcs {
            out[u] += 1;
            into[v] += 1;
        }
        for (x, y) in [(Out, In), (Out, Out), (In, In), (In, Out)] {
            let degree = |d: DegreeType, u: usize| (if d == Out { out[u] } else { into[u] }) as f64;
            let pairs: Vec<(f64, f64)> = arcs
                .iter()
                .map(|&(u, v)| (degree(x, u), degree(y, v)))
                .collect();
            for c in [Correlation::Pearson, Correlation::Spearman] {
                let r = directed_degree_assortativity(&g, x, y, c).unwrap();
                let expected = correlation_of(&pairs, c);
                assert!(
                    close(r, expected),
                    "seed {seed} {x:?}/{y:?} {c:?}: {r} != {expected}"
                );
            }
        }
    }
}

#[test]
fn numeric_and_attribute_assortativity_of_random_graphs() {
    for seed in 0..25u64 {
        let n = 4 + (seed as usize * 5) % 50;
        let m = (seed as usize * 11) % (4 * n) + 1;
        let edges = random_arcs(n, m, seed + 200);
        let values: Vec<f64> = (0..n)
            .map(|u| ((u * 37 + seed as usize) % 11) as f64 * 0.5 - 1.0)
            .collect();
        // small labels, and large ones (made dense by rank)
        let small: Vec<usize> = (0..n).map(|u| (u * 7 + seed as usize) % 4).collect();
        let large: Vec<usize> = small.iter().map(|&l| usize::MAX - 3 * l).collect();

        let g = undirected(&format!("attributes_{seed}"), n, &edges, 3);
        let arcs = undirected_arcs(&edges);
        let numeric: Vec<(f64, f64)> = arcs.iter().map(|&(u, v)| (values[u], values[v])).collect();
        let labelled: Vec<(usize, usize)> =
            arcs.iter().map(|&(u, v)| (small[u], small[v])).collect();
        let r = numeric_assortativity(&g, &values).unwrap();
        assert!(
            close(r, pearson_of(&numeric)),
            "seed {seed}: {r} != {}",
            pearson_of(&numeric)
        );
        for labels in [&small, &large] {
            let r = attribute_assortativity(&g, labels).unwrap();
            assert!(
                close(r, attribute_of(&labelled)),
                "seed {seed}: {r} != {}",
                attribute_of(&labelled)
            );
        }

        let d = graph_ix::<u32>(
            &format!("assortativity_attributes_{seed}"),
            "general",
            n,
            &edges,
        );
        let numeric: Vec<(f64, f64)> = edges.iter().map(|&(u, v)| (values[u], values[v])).collect();
        let labelled: Vec<(usize, usize)> =
            edges.iter().map(|&(u, v)| (small[u], small[v])).collect();
        let r = numeric_assortativity(&d, &values).unwrap();
        assert!(
            close(r, pearson_of(&numeric)),
            "seed {seed}: {r} != {}",
            pearson_of(&numeric)
        );
        for labels in [&small, &large] {
            let r = attribute_assortativity(&d, labels).unwrap();
            assert!(
                close(r, attribute_of(&labelled)),
                "seed {seed}: {r} != {}",
                attribute_of(&labelled)
            );
        }
    }
}

#[test]
fn numeric_assortativity_is_the_same_whatever_the_threads() {
    let n = 30_000;
    let edges = random_arcs(n, 120_000, 7);
    let values: Vec<f64> = (0..n).map(|u| (u as f64 * 0.618).sin() * 1e3).collect();
    let mut seen = vec![];
    for threads in [1, 3, 8] {
        let g = undirected(&format!("threads_{threads}"), n, &edges, threads);
        seen.push(numeric_assortativity(&g, &values).unwrap().to_bits());
    }
    assert!(seen.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn wrong_lengths_are_errors() {
    let g = undirected("lengths", 3, &[(0, 1), (1, 2)], 1);
    assert!(numeric_assortativity(&g, &[1.0]).is_err());
    assert!(attribute_assortativity(&g, &[1, 2]).is_err());
}

#[test]
fn wide_products_are_exact() {
    let m = u128::MAX;
    assert_eq!(mul_wide(m, m), (u128::MAX - 1, 1));
    assert_eq!(mul_wide(1 << 100, 1 << 100), (1 << 72, 0));
    assert_eq!(sub_to_f64((0, 5), (0, 7)), -2.0);
    assert_eq!(sub_to_f64((1, 0), (0, 1)), 2f64.powi(128) - 1.0);
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::gpu::{GpuBackend, WgpuBackend};

    fn backend() -> Option<WgpuBackend> {
        match WgpuBackend::new() {
            Ok(backend) => Some(backend),
            Err(e) => {
                eprintln!("skipping GPU test, no GPU adapter: {e}");
                None
            }
        }
    }

    #[test]
    fn gpu_degree_assortativity_is_the_cpus() {
        let Some(backend) = backend() else { return };
        for seed in 0..12u64 {
            let n = 5 + (seed as usize * 37) % 400;
            let m = (seed as usize * 101) % (4 * n) + 1;
            let edges = random_arcs(n, m, seed + 300);
            let g = undirected(&format!("gpu_{seed}"), n, &edges, 4);
            let device = backend.upload_graph(&*g).unwrap();
            for c in [Correlation::Pearson, Correlation::Spearman] {
                let cpu = degree_assortativity(&g, c).unwrap();
                let gpu = degree_assortativity_gpu(&g, c, &backend, &device).unwrap();
                assert!(
                    cpu.to_bits() == gpu.to_bits() || (cpu.is_nan() && gpu.is_nan()),
                    "{seed} {c:?}"
                );
            }
            let d = graph_ix::<u32>(&format!("assortativity_gpu_{seed}"), "general", n, &edges);
            let device = backend.upload_graph(&*d).unwrap();
            for (x, y) in [
                (DegreeType::Out, DegreeType::In),
                (DegreeType::In, DegreeType::Out),
            ] {
                for c in [Correlation::Pearson, Correlation::Spearman] {
                    let cpu = directed_degree_assortativity(&d, x, y, c).unwrap();
                    let gpu =
                        directed_degree_assortativity_gpu(&d, x, y, c, &backend, &device).unwrap();
                    assert!(
                        cpu.to_bits() == gpu.to_bits() || (cpu.is_nan() && gpu.is_nan()),
                        "{seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_arc_products_are_exact_with_large_values() {
        let Some(backend) = backend() else { return };
        // values near 2^32, every node linked to many: products past 64 bits, sums past 96
        let n = 3000;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| (1..40).map(move |k| (u, (u * 7 + k * 13) % n)))
            .collect();
        let g = undirected("gpu_large", n, &edges, 4);
        let device = backend.upload_graph(&*g).unwrap();
        let x: Vec<u32> = (0..n)
            .map(|u| u32::MAX - (u as u32 * 2_654_435) % 1000)
            .collect();
        let y: Vec<u32> = (0..n)
            .map(|u| u32::MAX - (u as u32 * 40_503) % 777)
            .collect();
        let mut expected = 0u128;
        for u in 0..n {
            for v in g.neighbours(u).unwrap() {
                expected += x[u] as u128 * y[v] as u128;
            }
        }
        assert_eq!(backend.arc_products(&device, &x, &y).unwrap(), expected);
    }
}
