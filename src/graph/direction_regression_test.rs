//! Regression and edge-case tests of graph direction: recorded direction, labelled graphs, derived
//! graphs (reuse, concurrent builds, cleanup) and the reciprocal edge index under concurrency.
use super::{DiGraph, Directedness, Graph, GraphMemoryMap, IndexType, direction::Direction};
use crate::graph::label::VoidLabel;
use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Weight(usize);
unsafe impl bytemuck::Zeroable for Weight {}
unsafe impl bytemuck::Pod for Weight {}
impl crate::graph::E for Weight {
    fn new(v: usize) -> Self {
        Weight(v)
    }
    fn is_labeled() -> bool {
        true
    }
}

fn write(name: &str, text: &str) -> PathBuf {
    let dir = unique_test_dir("core");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mtx"));
    std::fs::write(&path, text).unwrap();
    path
}

fn mtx(header: &str, n: usize, arcs: &[(usize, usize)]) -> String {
    let mut s = format!(
        "%%MatrixMarket matrix coordinate pattern {header}\n{n} {n} {}\n",
        arcs.len()
    );
    for (u, v) in arcs {
        s += &format!("{} {}\n", u + 1, v + 1);
    }
    s
}

fn lists<N: crate::graph::N, E: crate::graph::E, Ix: IndexType, Dir: Directedness>(
    g: &GraphMemoryMap<N, E, Ix, Dir>,
) -> Vec<Vec<usize>> {
    (0..g.size())
        .map(|u| g.neighbours(u).unwrap().collect())
        .collect()
}

fn files_of(hash: &str) -> Vec<PathBuf> {
    glob::glob(&format!(
        "{}*{hash}.*",
        crate::graph::cache::utils::CACHE_DIR
    ))
    .unwrap()
    .flatten()
    .collect()
}

/// An entry opened with other label types derives graphs of its own: a labelled instance
/// isn't handed the unlabelled one's.
#[test]
fn labelled_views_after_unlabelled_ones() {
    let path = write(
        "labels",
        "%%MatrixMarket matrix coordinate integer general\n3 3 3\n1 2 10\n1 3 20\n2 1 30\n",
    );
    let id = format!("labels_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, Weight, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    {
        let unlabelled =
            DiGraph::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, Some(2))
                .unwrap();
        unlabelled.transpose().unwrap();
    }
    let t = g.transpose();
    assert!(t.is_ok(), "labelled transpose failed: {}", t.err().unwrap());
}

/// Views of views are cache entries too, which `drop_cache` removes.
#[test]
fn drop_cache_removes_views_of_views() {
    let path = write("nested", &mtx("general", 3, &[(0, 1), (1, 2)]));
    let id = format!("nested_{}", unique_run_tag());
    let mut g =
        DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let t = g.transpose().unwrap();
    let tt_hash = crate::graph::cache::utils::id_from_filename(
        &t.graph_cache.derived_id("transpose").unwrap(),
    )
    .unwrap();
    let tu_hash = crate::graph::cache::utils::id_from_filename(
        &t.graph_cache.derived_id("undirected").unwrap(),
    )
    .unwrap();
    assert_eq!(lists(t.transpose().unwrap()), lists(&g));
    t.to_undirected_with(crate::graph::UndirectedMode::Each)
        .unwrap();
    assert!(!files_of(&tt_hash).is_empty());
    g.drop_cache().unwrap();
    let left: Vec<_> = files_of(&tt_hash)
        .into_iter()
        .chain(files_of(&tu_hash))
        .collect();
    // cleanup before asserting
    for f in &left {
        let _ = std::fs::remove_file(f);
    }
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
    assert!(left.is_empty(), "left behind: {left:?}");
}

/// u8 index: the undirected view overflows; it must error and leave nothing behind, twice.
#[test]
fn u8_undirected_overflow_errors_cleanly() {
    // 150 arcs 0->1..150 (wait: nodes up to 151 fits u8), undirected needs 300 > 255
    let arcs: Vec<_> = (1..=150).map(|v| (0, v)).collect();
    let path = write("u8", &mtx("general", 151, &arcs));
    let id = format!("u8_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, VoidLabel, u8>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let t = g.transpose().unwrap();
    for v in 1..=150 {
        assert_eq!(t.neighbours(v).unwrap().collect::<Vec<_>>(), vec![0]);
    }
    let h = crate::graph::cache::utils::id_from_filename(
        &g.graph_cache.derived_id("undirected").unwrap(),
    )
    .unwrap();
    for _ in 0..2 {
        assert!(
            g.to_undirected_with(crate::graph::UndirectedMode::Each)
                .is_err()
        );
        assert!(files_of(&h).is_empty(), "{:?}", files_of(&h));
    }
}

/// Edge cases: |V| = 1, only self loops, isolated head/tail nodes, max node id.
#[test]
fn edge_cases() {
    for (name, n, arcs, directed) in [
        ("single_loop", 1, vec![(0, 0)], false),
        ("loops_multi", 3, vec![(1, 1), (1, 1), (2, 2)], false),
        ("isolated", 6, vec![(2, 3), (3, 3), (3, 2), (2, 3)], true),
        ("maxid", 5, vec![(4, 0), (4, 4), (0, 4), (4, 0)], true),
        ("only_last", 4, vec![(3, 2)], true),
    ] {
        let path = write(name, &mtx("general", n, &arcs));
        let id = format!("edge_{name}_{}", unique_run_tag());
        let g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(4)).unwrap();
        let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
        assert_eq!(!g.is_symmetric().unwrap(), directed, "{name}");
        let mut ein = vec![Vec::new(); n];
        let mut eun = vec![Vec::new(); n];
        for &(u, v) in &arcs {
            ein[v].push(u);
            eun[u].push(v);
            if u != v {
                eun[v].push(u);
            }
        }
        ein.iter_mut().for_each(|l| l.sort());
        eun.iter_mut().for_each(|l| l.sort());
        if directed {
            assert_eq!(lists(g.view(Direction::In).unwrap()), ein, "{name}");
            assert_eq!(
                lists(
                    g.to_undirected_with(crate::graph::UndirectedMode::Each)
                        .unwrap()
                ),
                eun,
                "{name}"
            );
            assert_eq!(g.transpose().unwrap().width(), arcs.len());
        } else {
            // a symmetric graph is its own transpose, and (without parallel arcs) its own collapsed
            // undirected graph's entry
            assert!(std::ptr::eq(g.view(Direction::In).unwrap(), &*g));
            let simple = lists(&*g)
                .iter()
                .all(|l| l.windows(2).all(|w| w[0] != w[1]));
            let same = g.to_undirected().unwrap().cache_filename() == g.cache_filename();
            assert_eq!(same, simple, "{name}");
        }
    }
}

/// Two instances of one entry deriving the transpose at once.
#[test]
fn two_instances_transpose_concurrently() {
    let n = 3000;
    let mut arcs = Vec::new();
    let mut s: u64 = 1;
    for _ in 0..200_000 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (s >> 33) as usize % n;
        arcs.push((u, v));
    }
    let path = write("race", &mtx("general", n, &arcs));
    let id = format!("race_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let mut expected = vec![Vec::new(); n];
    for &(u, v) in &arcs {
        expected[v].push(u);
    }
    expected.iter_mut().for_each(|l| l.sort());
    let cache = g.cache_filename();
    let results: Vec<Result<Vec<Vec<usize>>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..std::env::var("REVIEW_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize))
            .map(|_| {
                let cache = cache.clone();
                sc.spawn(move || {
                    let h = DiGraph::<VoidLabel, VoidLabel, usize>::open(&cache, None, Some(2))
                        .map_err(|e| e.to_string())?;
                    let t = h.transpose().map_err(|e| e.to_string())?;
                    Ok(lists(t))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Ok(l) => assert!(l == expected, "instance {i}: wrong transpose"),
            Err(e) => panic!("instance {i}: {e}"),
        }
    }
}

/// One instance, many threads building the reciprocal edge index at once.
#[test]
fn edge_reciprocal_concurrently() {
    let n = 3000;
    let mut arcs = Vec::new();
    let mut s: u64 = 3;
    for _ in 0..200_000 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (s >> 33) as usize % n;
        arcs.push((u, v));
    }
    let path = write("er_race", &mtx("symmetric", n, &arcs));
    let id = format!("er_race_{}", unique_run_tag());
    let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let errs: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..std::env::var("REVIEW_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize))
            .map(|_| sc.spawn(|| g.edge_reciprocal().map(|_| ()).map_err(|e| e.to_string())))
            .collect();
        hs.into_iter()
            .filter_map(|h| h.join().unwrap().err())
            .collect()
    });
    assert!(errs.is_empty(), "{errs:?}");
}

#[test]
fn metadata_round_trip() {
    for d in [None, Some(true), Some(false)] {
        let mut meta = crate::graph::cache::CacheMetadata::now::<VoidLabel, VoidLabel, usize>(
            "d", "id", 3, 2, 8, false, false, false, 0, 0, None, None,
        );
        meta.symmetric = d.map(|d| !d);
        let p = crate::graph::cache::CacheMetadata::from_toml(&meta.to_toml()).unwrap();
        assert_eq!(p.symmetric, d.map(|d| !d));
        assert_eq!(p.to_toml(), meta.to_toml());
    }
}

/// Header shortcut variants.
#[test]
fn header_variants() {
    for (name, header, arcs, directed) in [
        (
            "upper",
            "%%MatrixMarket MATRIX Coordinate PATTERN SYMMETRIC",
            vec![(1, 0)],
            false,
        ),
        (
            "skew",
            "%%MatrixMarket matrix coordinate pattern skew-symmetric",
            vec![(1, 0)],
            false,
        ),
        (
            "spaces",
            "%%MatrixMarket   matrix\tcoordinate pattern   General  ",
            vec![(1, 0)],
            true,
        ),
    ] {
        let mut s = format!("{header}\n% comment\n3 3 {}\n", arcs.len());
        for (u, v) in &arcs {
            s += &format!("{} {}\n", u + 1, v + 1);
        }
        let path = write(name, &s);
        let id = format!("hdr_{name}_{}", unique_run_tag());
        let g =
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
        let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
        assert_eq!(!g.is_symmetric().unwrap(), directed, "{name}");
        assert_eq!(g.metadata().unwrap().symmetric, Some(!directed), "{name}");
    }
}

fn labelled_check<Ix: IndexType>(tag: &str, n: usize, arcs: &[(usize, usize, usize)]) {
    let mut s = format!(
        "%%MatrixMarket matrix coordinate integer general\n{n} {n} {}\n",
        arcs.len()
    );
    for (u, v, w) in arcs {
        s += &format!("{} {} {}\n", u + 1, v + 1, w);
    }
    let path = write(tag, &s);
    let id = format!("lab_{tag}_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, Weight, Ix>::from_file(&path, Some(id), Some(3)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    fn got<Ix: IndexType, Dir: Directedness>(
        g: &GraphMemoryMap<VoidLabel, Weight, Ix, Dir>,
    ) -> Vec<Vec<(usize, usize)>> {
        (0..g.size())
            .map(|u| {
                let mut l: Vec<_> = g
                    .index_node(u)
                    .zip(g.neighbours(u).unwrap())
                    .map(|(e, v)| (v, g.edge_label(e).0))
                    .collect();
                l.sort();
                l
            })
            .collect()
    }
    let mut ein = vec![Vec::new(); n];
    let mut eun = vec![Vec::new(); n];
    for &(u, v, w) in arcs {
        ein[v].push((u, w));
        eun[u].push((v, w));
        if u != v {
            eun[v].push((u, w));
        }
    }
    ein.iter_mut().for_each(|l| l.sort());
    eun.iter_mut().for_each(|l| l.sort());
    for u in 0..n {
        let l: Vec<usize> = g.neighbours(u).unwrap().collect();
        assert!(
            l.windows(2).all(|w| w[0] <= w[1]),
            "{tag} original node {u} unsorted: {l:?}"
        );
        let l: Vec<usize> = g.transpose().unwrap().neighbours(u).unwrap().collect();
        assert!(
            l.windows(2).all(|w| w[0] <= w[1]),
            "{tag} transpose node {u} unsorted: {l:?}"
        );
    }
    assert!(!g.is_symmetric().unwrap());
    assert_eq!(got(g.transpose().unwrap()), ein, "{tag} transpose");
    assert_eq!(
        got(g
            .to_undirected_with(crate::graph::UndirectedMode::Each)
            .unwrap()),
        eun,
        "{tag} undirected"
    );
    // sorted lists
    for u in 0..n {
        let l: Vec<usize> = g
            .to_undirected_with(crate::graph::UndirectedMode::Each)
            .unwrap()
            .neighbours(u)
            .unwrap()
            .collect();
        assert!(
            l.windows(2).all(|w| w[0] <= w[1]),
            "{tag} undirected node {u} unsorted: {l:?}"
        );
    }
}

#[test]
fn labelled_views_multiplicity() {
    let mut arcs = Vec::new();
    let mut s: u64 = 9;
    for i in 0..2000 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % 50;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = if i % 7 == 0 {
            u
        } else {
            (s >> 33) as usize % 50
        };
        arcs.push((u, v, i + 1));
    }
    arcs.extend([(49, 49, 7777), (49, 0, 8888), (49, 0, 9999)]);
    labelled_check::<usize>("usize", 60, &arcs);
    labelled_check::<u16>("u16", 60, &arcs);
}

/// A labelled `general` file listing every arc both ways is undirected.
#[test]
fn labelled_both_ways_is_undirected() {
    let path = write(
        "lab_both",
        "%%MatrixMarket matrix coordinate integer general\n4 4 6\n1 4 1\n1 2 2\n1 3 3\n4 1 1\n2 1 2\n3 1 3\n",
    );
    let id = format!("lab_both_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, Weight, usize>::from_file(&path, Some(id), Some(1)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let l0: Vec<usize> = g.neighbours(0).unwrap().collect();
    let labels: Vec<usize> = g.index_node(0).map(|e| g.edge_label(e).0).collect();
    eprintln!("l0 {l0:?} labels {labels:?}");
    assert!(g.is_symmetric().unwrap());
}

#[test]
fn same_instance_concurrent_views() {
    let n = 2000;
    let mut arcs = Vec::new();
    let mut s: u64 = 5;
    for _ in 0..100_000 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        arcs.push((u, (s >> 33) as usize % n));
    }
    let path = write("same_inst", &mtx("general", n, &arcs));
    let id = format!("same_inst_{}", unique_run_tag());
    let g = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let ptrs: Vec<(usize, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let g = &g;
                sc.spawn(move || {
                    let (a, b) = if i % 2 == 0 {
                        (
                            g.to_undirected_with(crate::graph::UndirectedMode::Each)
                                .unwrap(),
                            g.transpose().unwrap(),
                        )
                    } else {
                        (
                            g.to_undirected_with(crate::graph::UndirectedMode::Each)
                                .unwrap(),
                            g.transpose().unwrap(),
                        )
                    };
                    (a as *const _ as usize, b as *const _ as usize)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(ptrs.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(
        g.to_undirected_with(crate::graph::UndirectedMode::Each)
            .unwrap()
            .width(),
        arcs.iter()
            .map(|&(u, v)| if u == v { 1 } else { 2 })
            .sum::<usize>()
    );
}

/// Graphs with nodes but no edges load, as both types, and work with the usual operations.
#[test]
fn graphs_without_edges() {
    for header in ["general", "symmetric"] {
        let path = write(
            &format!("no_edges_{header}"),
            &format!("%%MatrixMarket matrix coordinate pattern {header}\n3 3 0\n"),
        );
        let id = format!("no_edges_{header}_{}", unique_run_tag());
        let d = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2))
            .unwrap_or_else(|e| panic!("{header}: {e}"));
        assert_eq!((d.size(), d.width()), (3, 0));
        assert!(d.is_symmetric().unwrap());
        assert_eq!(lists(&d), vec![Vec::<usize>::new(); 3]);
        assert_eq!(d.in_degree(2).unwrap(), 0);
        assert_eq!(
            lists(d.to_undirected().unwrap()),
            vec![Vec::<usize>::new(); 3]
        );
        let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("no_edges_g_{header}_{}", unique_run_tag())),
            Some(2),
        )
        .unwrap_or_else(|e| panic!("{header}, as a Graph: {e}"));
        assert_eq!((g.size(), g.width()), (3, 0));
        // algorithms run on them
        {
            use crate::components::{
                strong::StronglyConnectedComponents, weak::ConnectedComponents,
            };
            use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
            use crate::k_truss::{pkt::AlgoPKT, triangles::Triangles};
            use crate::trails::bfs::BFSDists;
            assert_eq!(ConnectedComponents::new(&d).unwrap().count(), 3);
            assert_eq!(StronglyConnectedComponents::new(&d).unwrap().count(), 3);
            assert_eq!(BFSDists::new(&d, 1).unwrap().get_distances(), &[0, 0, 0]);
            assert_eq!(
                AlgoBatageljZaversnik::new(&g).unwrap().k_cores(),
                &[0, 0, 0]
            );
            assert!(Triangles::new(&g).unwrap().all_triangles().is_empty());
            assert!(AlgoPKT::new(&g).unwrap().k_trusses().is_empty());
        }
        // reopened from the cache
        let reopened =
            DiGraph::<VoidLabel, VoidLabel, usize>::open(d.cache_filename(), None, Some(2))
                .unwrap();
        assert_eq!((reopened.size(), reopened.width()), (3, 0));
        let _ = TestGraph::new(g, None);
        let _ = TestGraph::new(d, path.parent().map(Path::to_path_buf));
    }
}

/// Two instances of one cache entry running the same algorithms at once each get their own
/// result files (a new run replaces the files, rather than truncating those another instance
/// maps): their results are right, and both drop them.
#[test]
fn instances_of_one_entry_run_algorithms_concurrently() {
    use crate::components::weak::ConnectedComponents;
    use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
    use crate::k_truss::triangles::Triangles;
    use std::sync::Barrier;
    // a symmetric random graph, big enough for the runs to overlap
    let mut state = 11u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let n = 3000;
    let arcs: Vec<(usize, usize)> = (0..60_000)
        .map(|_| (next() % n, next() % n))
        .filter(|(u, v)| u > v)
        .collect();
    let path = write("concurrent_instances", &mtx("symmetric", n, &arcs));
    let id = format!("concurrent_instances_{}", unique_run_tag());
    let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let cores = AlgoBatageljZaversnik::new(&g).unwrap().k_cores().to_vec();
    let triangles: Vec<u8> = Triangles::<_, _, _, u8>::new_tx(&g)
        .unwrap()
        .all_triangles()
        .iter()
        .map(|c| c.load(portable_atomic::Ordering::Relaxed))
        .collect();
    let components = ConnectedComponents::new(&g).unwrap().components().to_vec();

    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let h =
                    Graph::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, Some(2))
                        .unwrap();
                for _ in 0..10 {
                    barrier.wait();
                    let mut bz = AlgoBatageljZaversnik::new(&h).unwrap();
                    let mut t = Triangles::<_, _, _, u8>::new_tx(&h).unwrap();
                    let mut cc = ConnectedComponents::new(&h).unwrap();
                    assert_eq!(bz.k_cores(), cores.as_slice());
                    let got: Vec<u8> = t
                        .all_triangles()
                        .iter()
                        .map(|c| c.load(portable_atomic::Ordering::Relaxed))
                        .collect();
                    assert_eq!(got, triangles);
                    assert_eq!(cc.components(), components.as_slice());
                    bz.drop_cache().unwrap();
                    t.drop_cache().unwrap();
                    cc.drop_cache().unwrap();
                }
            });
        }
    });
}

/// Rebuilding an entry (the same id, another dataset) while an instance maps it: the instance
/// keeps reading its graph, intact, and the rebuilt entry is the new one.
#[test]
fn rebuilding_an_entry_leaves_instances_mapping_it_intact() {
    let id = format!("rebuild_mapped_{}", unique_run_tag());
    let first = write(
        "rebuild_mapped_a",
        &mtx("general", 4, &[(0, 1), (1, 2), (2, 3)]),
    );
    let old = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&first, Some(id.clone()), Some(2))
        .unwrap();
    let old_transpose = lists(old.transpose().unwrap());

    let second = write(
        "rebuild_mapped_b",
        &mtx("general", 6, &[(5, 0), (4, 1), (3, 2), (0, 5)]),
    );
    let new =
        DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&second, Some(id), Some(2)).unwrap();
    let new = TestGraph::new(new, None);
    assert_eq!(lists(&*new)[5], vec![0]);
    assert_eq!(new.size(), 6);

    // the old instance (and its transpose) are still the old graph
    assert_eq!(old.size(), 4);
    assert_eq!(lists(&old), vec![vec![1], vec![2], vec![3], vec![]]);
    assert_eq!(lists(old.transpose().unwrap()), old_transpose);
    drop(old);
    for path in [first, second] {
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
