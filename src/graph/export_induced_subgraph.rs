use super::GraphMemoryMap;

#[allow(dead_code)]
impl<
    N: crate::graph::N,
    E: crate::graph::E,
    Ix: crate::graph::IndexType,
    Dir: crate::graph::Directedness,
> GraphMemoryMap<N, E, Ix, Dir>
{
    /// The subgraph induced by the nodes `mask` holds for, as a new cache entry of id
    /// `identifier` (or a numbered export of this graph's).
    pub(super) fn apply_mask_to_nodes(
        &mut self,
        mask: impl Fn(usize) -> bool,
        identifier: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let id = match identifier {
            Some(id) => id.to_string(),
            None => {
                let seq = self.exports_fetch_increment()?;
                self.graph_cache.build_subgraph_cache_id(seq)?
            }
        };
        let cache = self.graph_cache.derive_induced(&id, &mask)?;
        GraphMemoryMap::init_from_cache(cache, Some(self.thread_count))
    }
}

#[cfg(test)]
mod test {
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use crate::graph::{DiGraph, Graph, GraphMemoryMap};

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

    /// Each node's (neighbour, edge label) list.
    fn labelled<Dir: crate::graph::Directedness>(
        g: &GraphMemoryMap<VoidLabel, Weight, usize, Dir>,
    ) -> Vec<Vec<(usize, usize)>> {
        (0..g.size())
            .map(|u| {
                g.index_node(u)
                    .zip(g.neighbours(u).unwrap())
                    .map(|(e, v)| (v, g.edge_label(e).0))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn induced_subgraphs_match_brute_force() {
        let dir = unique_test_dir("induced");
        std::fs::create_dir_all(&dir).unwrap();
        let mut state = 5u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let n = 30;
        // labelled arcs, with parallel arcs and self-loops
        let arcs: Vec<(usize, usize, usize)> = (0..150)
            .map(|_| (next() % n, next() % n, 1 + next() % 1000))
            .collect();
        let path = dir.join("g.mtx");
        let mut text = format!(
            "%%MatrixMarket matrix coordinate integer general\n{n} {n} {}\n",
            arcs.len()
        );
        for (u, v, w) in &arcs {
            text += &format!("{} {} {w}\n", u + 1, v + 1);
        }
        std::fs::write(&path, text).unwrap();
        let mut g = DiGraph::<VoidLabel, Weight, usize>::from_file(
            &path,
            Some(format!("induced_{}", unique_run_tag())),
            Some(2),
        )
        .unwrap();

        for (name, keep) in [
            (
                "thirds",
                (|u: usize| !u.is_multiple_of(3)) as fn(usize) -> bool,
            ),
            ("all", |_| true),
            ("none", |_| false),
            ("one", |u| u == 7),
        ] {
            // brute force: kept nodes renumbered in order, the arcs between them, sorted
            let new: Vec<Option<usize>> = (0..n)
                .scan(0, |next, u| {
                    Some(keep(u).then(|| {
                        *next += 1;
                        *next - 1
                    }))
                })
                .collect();
            let kept = new.iter().flatten().count();
            let mut want = vec![Vec::new(); kept];
            for &(u, v, w) in &arcs {
                if let (Some(nu), Some(nv)) = (new[u], new[v]) {
                    want[nu].push((nv, w));
                }
            }
            let sub = g
                .induced_subgraph(keep, Some(&format!("induced_{name}_{}", unique_run_tag())))
                .unwrap();
            let mut got = labelled(&sub);
            got.iter_mut().for_each(|l| l.sort());
            want.iter_mut().for_each(|l| l.sort());
            assert_eq!(got, want, "{name}");
            // a complete entry: it reopens
            let reopened =
                DiGraph::<VoidLabel, Weight, usize>::open(sub.cache_filename(), None, Some(2))
                    .unwrap();
            let mut again = labelled(&reopened);
            again.iter_mut().for_each(|l| l.sort());
            assert_eq!(again, want, "{name}, reopened");
            let _ = TestGraph::new(sub, None);
        }
        // numbered exports, without identifiers, are distinct entries
        let (a, b) = (
            g.induced_subgraph(|u| u < 5, None).unwrap(),
            g.induced_subgraph(|u| u < 9, None).unwrap(),
        );
        assert_ne!(a.cache_filename(), b.cache_filename());
        assert_eq!((a.size(), b.size()), (5, 9));
        let _ = (TestGraph::new(a, None), TestGraph::new(b, None));
        g.drop_cache().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An undirected graph's induced subgraphs are undirected graphs, recorded symmetric; node
    /// metalabels follow their nodes to their new ids.
    #[test]
    fn induced_subgraphs_of_graphs_keep_symmetry_and_metalabels() {
        use fst::Streamer;
        let fixture = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ggcat/random_graph_1_10.lz4"
        ));
        let mut g = Graph::<VoidLabel, VoidLabel, usize>::from_file_with_fst(
            fixture,
            Some(format!("induced_ggcat_{}", unique_run_tag())),
            None,
            Some(|_| true),
            Some(2),
        )
        .unwrap();
        let keep = |u: usize| u.is_multiple_of(2);
        let sub = g
            .induced_subgraph(
                keep,
                Some(&format!("induced_ggcat_sub_{}", unique_run_tag())),
            )
            .unwrap();
        assert!(sub.is_symmetric().unwrap());
        assert_eq!(sub.metadata().unwrap().symmetric, Some(true));
        assert_eq!(sub.size(), g.size().div_ceil(2));
        // every arc between kept nodes, renumbered (new id = old id / 2)
        for u in (0..g.size()).filter(|&u| keep(u)) {
            let want: Vec<usize> = g
                .neighbours(u)
                .unwrap()
                .filter(|&v| keep(v))
                .map(|v| v / 2)
                .collect();
            assert_eq!(
                sub.neighbours(u / 2).unwrap().collect::<Vec<_>>(),
                want,
                "node {u}"
            );
        }
        let mut stream = g.metalabels.stream();
        let mut checked = 0;
        while let Some((metalabel, node)) = stream.next() {
            let metalabel = std::str::from_utf8(metalabel).unwrap().to_string();
            let node = node as usize;
            match sub.node_id_from_metalabel(&metalabel) {
                Ok(new) if keep(node) => assert_eq!(new as usize, node / 2, "{metalabel}"),
                Err(_) if !keep(node) => {}
                other => panic!("{metalabel} (node {node}): {other:?}"),
            }
            checked += 1;
        }
        assert!(checked > 0, "the fixture has metalabels");
        let _ = TestGraph::new(sub, None);
        g.drop_cache().unwrap();
    }
}
