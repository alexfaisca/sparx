//! Regression and edge-case tests of graph direction: recorded direction, labelled graphs, derived
//! graphs (reuse, concurrent builds, cleanup) and the reciprocal edge index under concurrency.
use super::{GraphMemoryMap, IndexType, direction::Direction};
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
    let mut s = format!("%%MatrixMarket matrix coordinate pattern {header}\n{n} {n} {}\n", arcs.len());
    for (u, v) in arcs {
        s += &format!("{} {}\n", u + 1, v + 1);
    }
    s
}

fn lists<N: crate::graph::N, E: crate::graph::E, Ix: IndexType>(g: &GraphMemoryMap<N, E, Ix>) -> Vec<Vec<usize>> {
    (0..g.size()).map(|u| g.neighbours(u).unwrap().collect()).collect()
}

fn files_of(hash: &str) -> Vec<PathBuf> {
    glob::glob(&format!("{}*{hash}.*", crate::graph::cache::utils::CACHE_DIR))
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
    let g = GraphMemoryMap::<VoidLabel, Weight, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    {
        let unlabelled =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::open(g.cache_filename(), None, Some(2)).unwrap();
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
    let mut g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
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
    t.to_undirected().unwrap();
    assert!(!files_of(&tt_hash).is_empty());
    g.drop_cache().unwrap();
    let left: Vec<_> = files_of(&tt_hash).into_iter().chain(files_of(&tu_hash)).collect();
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
    let g = GraphMemoryMap::<VoidLabel, VoidLabel, u8>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let t = g.transpose().unwrap();
    for v in 1..=150 {
        assert_eq!(t.neighbours(v).unwrap().collect::<Vec<_>>(), vec![0]);
    }
    let h = crate::graph::cache::utils::id_from_filename(&g.graph_cache.derived_id("undirected").unwrap()).unwrap();
    for _ in 0..2 {
        assert!(g.to_undirected().is_err());
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
        let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(4)).unwrap();
        let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
        assert_eq!(g.is_directed().unwrap(), directed, "{name}");
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
            assert_eq!(lists(g.view(Direction::Both).unwrap()), eun, "{name}");
            assert_eq!(g.transpose().unwrap().width(), arcs.len());
        } else {
            assert!(std::ptr::eq(g.view(Direction::Both).unwrap(), &*g));
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
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s >> 33) as usize % n;
        arcs.push((u, v));
    }
    let path = write("race", &mtx("general", n, &arcs));
    let id = format!("race_{}", unique_run_tag());
    let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let mut expected = vec![Vec::new(); n];
    for &(u, v) in &arcs {
        expected[v].push(u);
    }
    expected.iter_mut().for_each(|l| l.sort());
    let cache = g.cache_filename();
    let results: Vec<Result<Vec<Vec<usize>>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..std::env::var("REVIEW_N").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize))
            .map(|_| {
                let cache = cache.clone();
                sc.spawn(move || {
                    let h = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::open(&cache, None, Some(2))
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
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s >> 33) as usize % n;
        arcs.push((u, v));
    }
    let path = write("er_race", &mtx("symmetric", n, &arcs));
    let id = format!("er_race_{}", unique_run_tag());
    let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let errs: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..std::env::var("REVIEW_N").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize))
            .map(|_| sc.spawn(|| g.edge_reciprocal().map(|_| ()).map_err(|e| e.to_string())))
            .collect();
        hs.into_iter().filter_map(|h| h.join().unwrap().err()).collect()
    });
    assert!(errs.is_empty(), "{errs:?}");
}

#[test]
fn metadata_round_trip() {
    for d in [None, Some(true), Some(false)] {
        let mut meta = crate::graph::cache::CacheMetadata::now::<VoidLabel, VoidLabel, usize>(
            "d", "id", 3, 2, 8, false, false, false, 0, 0, None, None,
        );
        meta.directed = d;
        let p = crate::graph::cache::CacheMetadata::from_toml(&meta.to_toml()).unwrap();
        assert_eq!(p.directed, d);
        assert_eq!(p.to_toml(), meta.to_toml());
    }
}

/// Header shortcut variants.
#[test]
fn header_variants() {
    for (name, header, arcs, directed) in [
        ("upper", "%%MatrixMarket MATRIX Coordinate PATTERN SYMMETRIC", vec![(1, 0)], false),
        ("skew", "%%MatrixMarket matrix coordinate pattern skew-symmetric", vec![(1, 0)], false),
        ("spaces", "%%MatrixMarket   matrix\tcoordinate pattern   General  ", vec![(1, 0)], true),
    ] {
        let mut s = format!("{header}\n% comment\n3 3 {}\n", arcs.len());
        for (u, v) in &arcs {
            s += &format!("{} {}\n", u + 1, v + 1);
        }
        let path = write(name, &s);
        let id = format!("hdr_{name}_{}", unique_run_tag());
        let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
        let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
        assert_eq!(g.is_directed().unwrap(), directed, "{name}");
        assert_eq!(g.metadata().unwrap().directed, Some(directed), "{name}");
    }
}

fn labelled_check<Ix: IndexType>(tag: &str, n: usize, arcs: &[(usize, usize, usize)]) {
    let mut s = format!("%%MatrixMarket matrix coordinate integer general\n{n} {n} {}\n", arcs.len());
    for (u, v, w) in arcs {
        s += &format!("{} {} {}\n", u + 1, v + 1, w);
    }
    let path = write(tag, &s);
    let id = format!("lab_{tag}_{}", unique_run_tag());
    let g = GraphMemoryMap::<VoidLabel, Weight, Ix>::from_file(&path, Some(id), Some(3)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let got = |g: &GraphMemoryMap<VoidLabel, Weight, Ix>| -> Vec<Vec<(usize, usize)>> {
        (0..g.size())
            .map(|u| {
                let mut l: Vec<_> = g.index_node(u).zip(g.neighbours(u).unwrap()).map(|(e, v)| (v, g.edge_label(e).0)).collect();
                l.sort();
                l
            })
            .collect()
    };
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
        assert!(l.windows(2).all(|w| w[0] <= w[1]), "{tag} original node {u} unsorted: {l:?}");
        let l: Vec<usize> = g.transpose().unwrap().neighbours(u).unwrap().collect();
        assert!(l.windows(2).all(|w| w[0] <= w[1]), "{tag} transpose node {u} unsorted: {l:?}");
    }
    assert!(g.is_directed().unwrap());
    assert_eq!(got(g.transpose().unwrap()), ein, "{tag} transpose");
    assert_eq!(got(g.to_undirected().unwrap()), eun, "{tag} undirected");
    // sorted lists
    for u in 0..n {
        let l: Vec<usize> = g.to_undirected().unwrap().neighbours(u).unwrap().collect();
        assert!(l.windows(2).all(|w| w[0] <= w[1]), "{tag} undirected node {u} unsorted: {l:?}");
    }
}

#[test]
fn labelled_views_multiplicity() {
    let mut arcs = Vec::new();
    let mut s: u64 = 9;
    for i in 0..2000 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % 50;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = if i % 7 == 0 { u } else { (s >> 33) as usize % 50 };
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
    let g = GraphMemoryMap::<VoidLabel, Weight, usize>::from_file(&path, Some(id), Some(1)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let l0: Vec<usize> = g.neighbours(0).unwrap().collect();
    let labels: Vec<usize> = g.index_node(0).map(|e| g.edge_label(e).0).collect();
    eprintln!("l0 {l0:?} labels {labels:?}");
    assert!(!g.is_directed().unwrap());
}

#[test]
fn same_instance_concurrent_views() {
    let n = 2000;
    let mut arcs = Vec::new();
    let mut s: u64 = 5;
    for _ in 0..100_000 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as usize % n;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        arcs.push((u, (s >> 33) as usize % n));
    }
    let path = write("same_inst", &mtx("general", n, &arcs));
    let id = format!("same_inst_{}", unique_run_tag());
    let g = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(2)).unwrap();
    let g = TestGraph::new(g, path.parent().map(Path::to_path_buf));
    let ptrs: Vec<(usize, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let g = &g;
                sc.spawn(move || {
                    let (a, b) = if i % 2 == 0 {
                        (g.to_undirected().unwrap(), g.transpose().unwrap())
                    } else {
                        (g.to_undirected().unwrap(), g.transpose().unwrap())
                    };
                    (a as *const _ as usize, b as *const _ as usize)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(ptrs.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(g.to_undirected().unwrap().width(), arcs.iter().map(|&(u, v)| if u == v { 1 } else { 2 }).sum::<usize>());
}
