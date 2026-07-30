//! Regression and edge-case tests of the directed graph algorithms: components, directed k-cores,
//! directed clustering coefficients, Hierholzer on directed graphs, and cached results.
#![allow(unused)]
use crate::components::strong::StronglyConnectedComponents;
use crate::components::test_utils::{graph, label, random_arcs, reachability};
use crate::components::weak::ConnectedComponents;
use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
use crate::graph::{Direction, GraphMemoryMap, IndexType};
use crate::k_core::directed::{CoreDegree, DirectedKCores};
use crate::k_truss::clustering_coefficient::{ClusteringCoefficient, DirectedClusteringCoefficient};
use std::io::Write;

pub(crate) fn graph_ix<Ix: IndexType>(
    name: &str,
    header: &str,
    n: usize,
    edges: &[(usize, usize)],
) -> TestGraph<(), (), Ix, crate::graph::Directed> {
    let dir = unique_test_dir("review_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
    writeln!(f, "{n} {n} {}", edges.len()).unwrap();
    for &(u, v) in edges {
        writeln!(f, "{} {}", u + 1, v + 1).unwrap();
    }
    drop(f);
    let id = format!("review_{name}_{}", unique_run_tag());
    let g = crate::graph::DiGraph::<(), (), Ix>::from_file(&path, Some(id), Some(4)).expect("parse");
    TestGraph::new(g, Some(dir))
}

fn cc_brute(n: usize, arcs: &[(usize, usize)]) -> Vec<f64> {
    let mut a = vec![vec![0f64; n]; n];
    for &(u, v) in arcs { if u != v { a[u][v] += 1.; } }
    let s = |i: usize, j: usize| a[i][j] + a[j][i];
    (0..n).map(|i| {
        let mut t = 0.;
        for j in 0..n { for h in 0..n { t += s(i, j) * s(i, h) * s(j, h); } }
        t /= 2.;
        let degree: f64 = (0..n).map(|j| s(i, j)).sum();
        let rec: f64 = (0..n).map(|j| a[i][j] * a[j][i]).sum();
        let p = degree * (degree - 1.) - 2. * rec;
        if p > 0. { t / p } else { 0. }
    }).collect()
}

#[test]
fn dcc_self_loop_on_triangle() {
    let arcs = vec![(0, 1), (1, 2), (2, 0), (0, 0), (1, 1), (1, 1)];
    let g = graph("rv_dcc_loop", "general", 3, &arcs);
    let cc = DirectedClusteringCoefficient::new(&g).unwrap();
    let want = cc_brute(3, &arcs);
    assert_eq!(cc.local_clustering_coefficients(), &want[..]);
}

#[test]
fn dcc_random_with_loops() {
    let mut arcs = random_arcs(20, 120, 77);
    for i in 0..20 { arcs.push((i, i)); }
    let g = graph("rv_dcc_rand_loops", "general", 20, &arcs);
    let cc = DirectedClusteringCoefficient::new(&g).unwrap();
    let want = cc_brute(20, &arcs);
    for (i, (a, b)) in cc.local_clustering_coefficients().iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-12, "node {i}: {a} vs {b}");
    }
}

#[test]
fn dcc_undirected_with_loops_matches_usual() {
    // simple undirected triangle plus a self-loop at 0
    let g = graph("rv_dcc_und_loop", "symmetric", 4, &[(1, 0), (2, 1), (2, 0), (0, 0), (3, 0)]);
    let d = DirectedClusteringCoefficient::new(&g).unwrap();
    let u = ClusteringCoefficient::new(g.to_undirected().unwrap()).unwrap();
    let dv: Vec<f64> = d.local_clustering_coefficients().to_vec();
    let uv: Vec<f64> = (0..4).map(|i| u.get_node_clusteringcoefficient(i)).collect();
    assert_eq!(dv, uv);
}

#[test]
fn weak_u8_at_limit() {
    // 256 nodes (max for u8), a path over 0..=127 and pairs
    let mut arcs: Vec<(usize, usize)> = (0..120).map(|u| (u + 1, u)).collect();
    arcs.extend((0..60).map(|i| (255 - i, 200 - i)));
    let g = graph_ix::<u8>("rv_weak_u8", "general", 256, &arcs);
    let cc = ConnectedComponents::new(&g).unwrap();
    let both: Vec<_> = arcs.iter().flat_map(|&(u, v)| [(u, v), (v, u)]).collect();
    let reach = reachability(256, &both);
    let want = label(256, |u, v| reach[u][v]);
    let got: Vec<usize> = cc.components().iter().map(|c| c.index()).collect();
    assert_eq!(got, want);
    let c2 = ConnectedComponents::get_or_compute(&g).unwrap();
    assert_eq!(c2.count(), cc.count());
}

#[test]
fn strong_u8_at_limit() {
    let mut arcs = random_arcs(256, 250, 5);
    let g = graph_ix::<u8>("rv_strong_u8", "general", 256, &arcs);
    let scc = StronglyConnectedComponents::new(&g).unwrap();
    let reach = reachability(256, &arcs);
    let want = label(256, |u, v| reach[u][v] && reach[v][u]);
    let got: Vec<usize> = scc.components().iter().map(|c| c.index()).collect();
    assert_eq!(got, want);
}

#[test]
fn strong_u8_full_edges() {
    // 255 edges: max offsets for u8; a big cycle with a chord
    let mut arcs: Vec<(usize, usize)> = (0..200).map(|u| (u, (u + 1) % 200)).collect();
    arcs.extend((0..55).map(|i| (i * 3 % 200, 255 - i)));
    let g = graph_ix::<u8>("rv_strong_u8_full", "general", 256, &arcs);
    let scc = StronglyConnectedComponents::new(&g).unwrap();
    let reach = reachability(256, &arcs);
    let want = label(256, |u, v| reach[u][v] && reach[v][u]);
    let got: Vec<usize> = scc.components().iter().map(|c| c.index()).collect();
    assert_eq!(got, want);
}

#[test]
fn empty_graphs() {
    let g = graph("rv_empty", "general", 0, &[]);
    let cc = ConnectedComponents::new(&g);
    println!("weak empty: {:?}", cc.as_ref().map(|c| c.count()).map_err(|e| e.to_string()));
    let scc = StronglyConnectedComponents::new(&g);
    println!("strong empty: {:?}", scc.as_ref().map(|c| c.count()).map_err(|e| e.to_string()));
    let k = DirectedKCores::new(&g, CoreDegree::Total);
    println!("kcore empty: {:?}", k.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    let d = DirectedClusteringCoefficient::new(&g);
    println!("dcc empty: {:?}", d.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    assert!(cc.is_ok() && scc.is_ok() && k.is_ok() && d.is_ok());
}

#[test]
fn kcores_u8_both() {
    // directed graph with 200 arcs on u8: undirected view has 400 > 255
    let arcs = random_arcs(30, 200, 11).into_iter().filter(|(u, v)| u != v).collect::<Vec<_>>();
    let g = graph_ix::<u8>("rv_kcore_u8", "general", 30, &arcs);
    let r = DirectedKCores::new(&g, CoreDegree::Total);
    println!("{:?}", r.as_ref().map(|c| c.k_cores().to_vec()).map_err(|e| e.to_string()));
}

#[test]
fn weak_get_or_compute_many_threads() {
    for seed in 0..20 {
        let arcs = random_arcs(3000, 2500, seed);
        let g = crate::components::test_utils::graph_with_threads(&format!("rv_wc_{seed}"), "general", 3000, &arcs, 32);
        let cc = ConnectedComponents::new(&g).unwrap();
        let both: Vec<_> = arcs.iter().flat_map(|&(u, v)| [(u, v), (v, u)]).collect();
        // sequential uf
        let mut p: Vec<usize> = (0..3000).collect();
        fn f(p: &mut [usize], mut x: usize) -> usize { while p[x] != x { x = p[x]; } x }
        for &(u, v) in &arcs { let (a, b) = (f(&mut p, u), f(&mut p, v)); p[a.max(b)] = a.min(b); }
        let mut num = vec![usize::MAX; 3000]; let mut c = 0;
        let want: Vec<usize> = (0..3000).map(|u| { let r = f(&mut p, u); if num[r] == usize::MAX { num[r] = c; c += 1; } num[r] }).collect();
        let got: Vec<usize> = cc.components().iter().map(|c| c.index()).collect();
        assert_eq!(got, want, "seed {seed}");
        assert_eq!(ConnectedComponents::get_or_compute(&g).unwrap().count(), c);
    }
}

fn write_mtx(dir: &std::path::Path, name: &str, header: &str, n: usize, edges: &[(usize, usize)]) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
    writeln!(f, "{n} {n} {}", edges.len()).unwrap();
    for &(u, v) in edges { writeln!(f, "{} {}", u + 1, v + 1).unwrap(); }
    path
}

/// The same cache id (e.g. the default one: the input file's path) rebuilt from another graph:
/// get_or_compute must not hand back the old graph's cached results.
#[test]
fn stale_cached_results_are_not_reused() {
    use crate::graph::Graph;
    use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
    let dir = unique_test_dir("rv_stale");
    let id = format!("rv_stale_{}", unique_run_tag());
    // a triangle with a pendant node, then a path
    let p1 = write_mtx(&dir, "a", "symmetric", 4, &[(1, 0), (2, 1), (2, 0), (3, 2)]);
    {
        let g1 = Graph::<(), (), usize>::from_file(&p1, Some(id.clone()), Some(2)).unwrap();
        assert_eq!(AlgoBatageljZaversnik::new(&g1).unwrap().k_cores(), &[2, 2, 2, 1]);
    }
    let p2 = write_mtx(&dir, "b", "symmetric", 4, &[(1, 0), (2, 1), (3, 2)]);
    let g2 = TestGraph::new(Graph::<(), (), usize>::from_file(&p2, Some(id), Some(2)).unwrap(), Some(dir));
    let cached = AlgoBatageljZaversnik::get_or_compute(&g2).unwrap();
    assert_eq!(cached.k_cores(), &[1, 1, 1, 1], "stale cores of the old graph");
}

/// Components read back from a stale cache entry of the same id, with another node count.
#[test]
fn cached_components_of_another_graph_are_not_reused() {
    let dir = unique_test_dir("rv_stale_cc");
    let id = format!("rv_stale_cc_{}", unique_run_tag());
    let p1 = write_mtx(&dir, "a", "general", 4, &[(0, 1), (2, 3)]);
    {
        let g1 = crate::graph::DiGraph::<(), (), usize>::from_file(&p1, Some(id.clone()), Some(2)).unwrap();
        let _ = ConnectedComponents::new(&g1).unwrap();
        let _ = StronglyConnectedComponents::new(&g1).unwrap();
    }
    let p2 = write_mtx(&dir, "b", "general", 7, &[(0, 1), (1, 2), (2, 0), (4, 5)]);
    let g2 = TestGraph::new(crate::graph::DiGraph::<(), (), usize>::from_file(&p2, Some(id), Some(2)).unwrap(), Some(dir));
    let cc = ConnectedComponents::get_or_compute(&g2).unwrap();
    let scc = StronglyConnectedComponents::get_or_compute(&g2).unwrap();
    assert_eq!((cc.components().len(), scc.components().len()), (7, 7), "stale results of the old 4-node graph");
}

#[test]
fn scc_adversarial() {
    // nested cycles sharing nodes, back arcs deep in a path, self-loops, parallel arcs
    let n = 400;
    let mut arcs: Vec<(usize, usize)> = (0..n - 1).map(|u| (u, u + 1)).collect();
    for k in (10..n).step_by(37) { arcs.push((k, k - 9)); arcs.push((k, k - 9)); }
    arcs.push((n - 1, 200));
    for u in (0..n).step_by(5) { arcs.push((u, u)); }
    arcs.extend(random_arcs(n, 30, 99));
    let g = graph("rv_scc_adv", "general", n, &arcs);
    let scc = StronglyConnectedComponents::new(&g).unwrap();
    let reach = reachability(n, &arcs);
    let want = label(n, |u, v| reach[u][v] && reach[v][u]);
    let got: Vec<usize> = scc.components().iter().map(|c| c.index()).collect();
    assert_eq!(got, want);
    assert_eq!(StronglyConnectedComponents::get_or_compute(&g).unwrap().count(), want.iter().max().unwrap() + 1);
}

#[test]
fn scc_u16_random() {
    for seed in 0..5u64 {
        let n = 300;
        let arcs = random_arcs(n, 450, seed + 100);
        let g = graph_ix::<u16>(&format!("rv_scc16_{seed}"), "general", n, &arcs);
        let scc = StronglyConnectedComponents::new(&g).unwrap();
        let reach = reachability(n, &arcs);
        let want = label(n, |u, v| reach[u][v] && reach[v][u]);
        let got: Vec<usize> = scc.components().iter().map(|c| c.index()).collect();
        assert_eq!(got, want, "seed {seed}");
    }
}

#[test]
fn dcc_parallel_tx_u8_overflow() {
    // heavy multiplicities: an arc's count Σ_w s(u,w) s(v,w) with s = 20 → 400 > u8
    let mut arcs = vec![];
    for _ in 0..20 { arcs.extend([(0, 1), (1, 2), (2, 0)]); }
    let g = graph("rv_dcc_u8", "general", 3, &arcs);
    let r = DirectedClusteringCoefficient::<_, _, _>::new_tx::<u8>(&g);
    match r {
        Err(e) => println!("u8 overflow error: {e}"),
        Ok(cc) => {
            let want = cc_brute(3, &arcs);
            assert_eq!(cc.local_clustering_coefficients(), &want[..], "silently wrong with u8 counts");
        }
    }
}

#[test]
fn kcores_parallel_reciprocal() {
    // reciprocal pairs and parallel arcs, Both = total degree with multiplicity
    let arcs = vec![(0, 1), (1, 0), (0, 1), (1, 2), (2, 1), (2, 0), (0, 2), (3, 0), (3, 3), (0, 3), (0, 3)];
    let g = graph("rv_kc_rec", "general", 4, &arcs);
    for d in [CoreDegree::In, CoreDegree::Out, CoreDegree::Total] {
        let mut c = DirectedKCores::new(&g, d).unwrap();
        println!("{d:?}: {:?}", c.k_cores());
    }
    // drop caches of one direction leaves the others intact
    let a = DirectedKCores::new(&g, CoreDegree::In).unwrap();
    let mut b = DirectedKCores::new(&g, CoreDegree::Out).unwrap();
    b.drop_cache().unwrap();
    assert_eq!(a.k_cores().len(), 4);
}

#[test]
fn hierholzer_directed() {
    use crate::trails::hierholzer::AlgoHierholzer;
    // union of random directed cycles: balanced
    let n = 50;
    let mut arcs = vec![];
    let r = random_arcs(n, 400, 5);
    let mut i = 0;
    for len in [3usize, 7, 11, 4, 9, 2, 1, 13] {
        let cyc: Vec<usize> = (0..len).map(|k| r[i + k].0).collect();
        i += len;
        for k in 0..len { arcs.push((cyc[k], cyc[(k + 1) % len])); }
    }
    let g = graph("rv_hier", "general", n, &arcs);
    let h = AlgoHierholzer::new(&g).unwrap();
    let mut used: Vec<(usize, usize)> = vec![];
    for t in 0..h.trail_number() {
        let tr: Vec<usize> = h.get_trail(t).unwrap().iter().map(|x| x.index()).collect();
        // closing arc omitted
        for w in tr.windows(2) { used.push((w[0], w[1])); }
        used.push((*tr.last().unwrap(), tr[0]));
    }
    let mut a = arcs.clone(); a.sort(); used.sort();
    assert_eq!(used, a);
}

#[test]
fn error_paths_leave_no_files() {
    use crate::graph::CacheFile;
    let mut arcs = vec![];
    for _ in 0..20 { arcs.extend([(0, 1), (1, 2), (2, 0)]); }
    let g = graph("rv_leak_dcc", "general", 3, &arcs);
    assert!(DirectedClusteringCoefficient::<_, _, _>::new_tx::<u8>(&g).is_err());
    let f = g.build_cache_filename(CacheFile::DirectedClusteringCoefficient, None).unwrap();
    let leaked_dcc = std::path::Path::new(&f).exists();

    let arcs = random_arcs(30, 200, 11).into_iter().filter(|(u, v)| u != v).collect::<Vec<_>>();
    let g8 = graph_ix::<u8>("rv_leak_kc", "general", 30, &arcs);
    assert!(DirectedKCores::new(&g8, CoreDegree::Total).is_err());
    let f = g8.build_cache_filename(CacheFile::KCoreDirected(CoreDegree::Total), None).unwrap();
    let leaked_kc = std::path::Path::new(&f).exists();
    assert!(!leaked_dcc && !leaked_kc, "leaked: dcc {leaked_dcc}, kcore {leaked_kc}");
}

#[test]
fn dfs_visit_directed_back_edge_to_parent() {
    use crate::trails::dfs_visit::{DFSVisit, DfsVisitor};
    use std::ops::ControlFlow;
    struct Rec(Vec<(usize, usize)>);
    impl DfsVisitor<usize> for Rec {
        fn back_edge(&mut self, u: usize, v: usize) -> ControlFlow<()> { self.0.push((u, v)); ControlFlow::Continue(()) }
    }
    let g = graph("rv_dfsv", "general", 3, &[(0, 1), (1, 0), (1, 2)]);
    let mut rec = Rec(vec![]);
    { let _ = DFSVisit::new(&*g, 0, &mut rec).unwrap(); }
    assert_eq!(rec.0, vec![(1, 0)]);
    let u = graph("rv_dfsv_u", "symmetric", 3, &[(1, 0), (2, 1)]);
    let mut rec = Rec(vec![]);
    // as an undirected graph: the arc back to the parent is the tree edge reversed
    { let _ = DFSVisit::new(u.to_undirected().unwrap(), 0, &mut rec).unwrap(); }
    assert_eq!(rec.0, vec![]);
}

/// Hierholzer used to keep each node's remaining out-degree as a u8: degree 256 walked nothing.
#[test]
fn hierholzer_degree_256() {
    use crate::trails::hierholzer::AlgoHierholzer;
    let mut arcs = vec![];
    for _ in 0..256 { arcs.extend([(0, 1), (1, 0)]); }
    let g = graph("rv_hier256", "general", 2, &arcs);
    let h = AlgoHierholzer::new(&g).unwrap();
    let total: usize = (0..h.trail_number()).map(|t| h.get_trail(t).unwrap().len()).sum();
    // a closed trail over k arcs is stored as its k nodes (the closing one dropped)
    assert_eq!(h.trail_number(), 1);
    assert_eq!(total, 512, "the trail covers {total} of 512 arcs");
}

/// `DFS::drop_cache` used to remove the BFS result's file rather than its own.
#[test]
fn dfs_and_bfs_drop_their_own_results() {
    use crate::graph::CacheFile;
    use crate::trails::{bfs::BFSDists, dfs::DFS};
    let g = graph("dfs_bfs_drop", "general", 4, &[(0, 1), (1, 2), (2, 3)]);
    let mut bfs = BFSDists::new(&g, 0).unwrap();
    let mut dfs = DFS::new(&g, 0).unwrap();
    dfs.drop_cache().unwrap();
    bfs.drop_cache().unwrap();
    for kind in [CacheFile::BFS, CacheFile::DFS] {
        let f = g.build_cache_filename(kind.clone(), None).unwrap();
        assert!(!std::path::Path::new(&f).exists(), "{f} left behind");
    }
}
