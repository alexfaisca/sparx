use super::{CacheFile, GraphMemoryMap};
use crate::{
    shared_slice::{
        AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut,
    },
    utils::OneOrMany,
};

use crossbeam::thread;
use smallvec::SmallVec;
use std::{
    fs::OpenOptions,
    sync::{Arc, Barrier},
};

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    pub(super) fn get_edge_reciprocal_impl(
        &self,
    ) -> Result<AbstractedProceduralMemory<Ix>, Box<dyn std::error::Error>> {
        let er_fn = self.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let dud = Vec::new();
        match OpenOptions::new().read(true).open(er_fn.as_str()) {
            Ok(i) => {
                let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
                SharedSlice::<Ix>::abstract_mem(&er_fn, dud, len, true)
            }
            Err(_) => {
                self.build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(er_fn.as_str()) {
                    Ok(i) => {
                        let len =
                            i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
                        SharedSlice::<Ix>::abstract_mem(&er_fn, dud, len, true)
                    }
                    Err(e) => Err(format!("error can't abst mem for edge_reciprocal {e}").into()),
                }
            }
        }
    }

    pub(super) fn get_edge_over_impl(
        &self,
    ) -> Result<AbstractedProceduralMemory<Ix>, Box<dyn std::error::Error>> {
        let eo_fn = self.build_cache_filename(CacheFile::EdgeOver, None)?;
        let dud = Vec::new();
        match OpenOptions::new().read(true).open(eo_fn.as_str()) {
            Ok(i) => {
                let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
                SharedSlice::<Ix>::abstract_mem(&eo_fn, dud, len, true)
            }
            Err(_) => {
                self.build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(eo_fn.as_str()) {
                    Ok(i) => {
                        let len =
                            i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
                        SharedSlice::<Ix>::abstract_mem(&eo_fn, dud, len, true)
                    }
                    Err(e) => Err(format!("error can't abst mem for edge_over {e}").into()),
                }
            }
        }
    }

    fn init_procedural_memory_build_reciprocal(
        &self,
        er_fn: &str,
        eo_fn: &str,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<Ix>,
            AbstractedProceduralMemoryMut<Ix>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.size();
        let edge_count = self.width();

        let edge_reciprocal = SharedSliceMut::<Ix>::abst_mem_mut(er_fn, edge_count, true)?;
        let edge_out = SharedSliceMut::<Ix>::abst_mem_mut(eo_fn, node_count, true)?;

        Ok((edge_reciprocal, edge_out))
    }

    /// Builds the reciprocal edge and edge over indices.
    ///
    /// They're built under temporary names of their own and only renamed into place once
    /// complete and checked: a failed or interrupted build never leaves behind files that later
    /// calls would take for a finished index, and concurrent builds (from other threads, under
    /// the graph's lock, or other processes) never write files another one reads. Builds are
    /// deterministic, so whichever renames last leaves the same index.
    fn build_reciprocal_edge_index(&self) -> Result<(), Box<dyn std::error::Error>> {
        // edges of an asymmetric graph lack reciprocals
        self.require_symmetric("the reciprocal edge index", None)?;

        let er_fn = self.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let eo_fn = self.build_cache_filename(CacheFile::EdgeOver, None)?;
        let _building = self
            .derive_lock
            .lock()
            .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
        // built meanwhile, by another thread
        if std::path::Path::new(&er_fn).exists() && std::path::Path::new(&eo_fn).exists() {
            return Ok(());
        }
        static BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let build = format!(
            "{}.{}",
            std::process::id(),
            BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let er_part = format!("{er_fn}.{build}.part");
        let eo_part = format!("{eo_fn}.{build}.part");

        let built = self
            .build_reciprocal_edge_index_into(&er_part, &eo_part)
            .and_then(|()| -> Result<(), Box<dyn std::error::Error>> {
                std::fs::rename(&eo_part, &eo_fn)?;
                std::fs::rename(&er_part, &er_fn)?;
                Ok(())
            });
        if built.is_err() {
            // (a finished index, from another build, stays)
            for f in [&er_part, &eo_part] {
                let _ = std::fs::remove_file(f);
            }
        }
        built
    }

    fn build_reciprocal_edge_index_into(
        &self,
        er_fn: &str,
        eo_fn: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.size();
        let edge_count = self.width();

        let threads = self.thread_count.max(1) as usize;
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<Ix>::new(self.offsets_ptr(), self.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.neighbours_ptr(), edge_count);

        let (er, eo) = self.init_procedural_memory_build_reciprocal(er_fn, eo_fn)?;

        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|scope|-> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let mut er = er.shared_slice();
                let mut eo = eo.shared_slice();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                threads_res.push(scope.spawn(move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

                    let mut edges_start = index_ptr.get(begin).index();

                    for u in begin..end {
                        let mut eo_at_end = true;
                        let edges_stop = index_ptr.get(u + 1).index();
                        for edge_offset in edges_start..edges_stop {
                            let v = neighbours_ptr.get(edge_offset).index();
                            if v > u {
                                eo_at_end = false;
                                *eo.get_mut(u) = Ix::new(edge_offset);
                                break;
                            }
                        }
                        if eo_at_end {
                            *eo.get_mut(u) = Ix::new(edges_stop);
                        }
                        edges_start = edges_stop;
                    }

                    synchronize.wait();

                    // a closure, so that every thread reaches the next barrier even on errors
                    let mut matched = || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let mut prev = OneOrMany::<(usize, usize)>::One((node_count, 0usize));
                    for u in begin..end {
                        for edge_offset in index_ptr.get(u).index()..eo.get(u).index() {
                            let v = neighbours_ptr.get(edge_offset).index();
                            // binary search on neighbours w, where w < v
                            if u == v {
                                // self loops are their own reciprocals
                                *er.get_mut(edge_offset) = Ix::new(edge_offset);
                                // reset
                                prev = OneOrMany::One((v, edge_offset));
                                continue;
                            } else if let OneOrMany::One((prev_v, prev_offset)) = prev {
                                if prev_v == v {
                                    // find reciprocal, dest is the same, offset is different
                                    // make entry into list and record (never actually loops)
                                    let reciprocal = if prev_offset < edge_count - 1 && neighbours_ptr.get(prev_offset + 1).index() == u {
                                        prev_offset + 1
                                    } /*else if prev_offset < edge_count - 1 {
                                            println!("after {prev_offset}->{u} comes {}", graph_ptr.get(prev_offset + 1).dest());
                                        }*/
                                        // search backwards
                                    else if prev_offset > 0 && neighbours_ptr.get(prev_offset - 1).index() == u {
                                        prev_offset - 1
                                    } /*else if prev_offset > 0 {
                                            println!("before {prev_offset}->{u} comes {}", graph_ptr.get(prev_offset - 1).dest());
                                        }*/
                                    else {
                                        return Err(format!("error couldn't find reciprocal for 2nd multi edge {edge_offset}, u: ({u}) -> v: ({v}), all options were already taken").into());
                                    };
                                    let mut vs: SmallVec<[(usize, usize); 4]> = SmallVec::new();
                                    vs.push((prev_v, prev_offset));
                                    vs.push((v, reciprocal));
                                    prev = OneOrMany::Many(vs);
                                    *er.get_mut(edge_offset) = Ix::new(reciprocal);
                                    *er.get_mut(reciprocal) = Ix::new(edge_offset);
                                    continue;
                                }
                            } else if let OneOrMany::Many(ref mut vs) = prev {
                                if vs[0].0 == v {
                                    // find reciprocal, dest is the same, offset is different from
                                    // any of the recorded offsets: greedy lookup
                                    let reciprocal = 'search_adjacents: {
                                        // search forwards
                                        let mut res = vs[0].1;
                                        while res < edge_count - 1 && neighbours_ptr.get(res + 1).index() == u {
                                            res += 1;
                                            if vs.iter().any(|&(_, b)| b == res) {
                                                continue;
                                            } else {
                                                break 'search_adjacents res;
                                            }
                                        }
                                        // reset and search backwards
                                        res = vs[0].1;
                                        while res > 0 && neighbours_ptr.get(res - 1).index() == u {
                                            res -= 1;
                                            if vs.iter().any(|&(_, b)| b == res) {
                                                continue;
                                            } else {
                                                break 'search_adjacents res;
                                            }
                                        }

                                        return Err(format!("error couldn't find reciprocal for nth multi edge {edge_offset}, u: ({u}) -> v: ({v}), all options were already taken").into());
                                    };

                                    // extend entry's list
                                    vs.push((v, reciprocal));
                                    *er.get_mut(edge_offset) = Ix::new(reciprocal);
                                    *er.get_mut(reciprocal) = Ix::new(edge_offset);
                                    continue;
                                }
                            }

                                // binary search over v's neighbours above v, [floor, ceil)
                                let mut floor = eo.get(v).index();
                                let mut ceil = index_ptr.get(v + 1).index();
                                let reciprocal =  loop {
                                    if floor >= ceil {
                                        return Err(format!("error couldn't find reciprocal for edge {edge_offset}, u: ({u}) -> v: ({v})").into());
                                    }

                                    let m = floor + (ceil - floor) / 2;
                                    let dest = neighbours_ptr.get(m).index();

                                    match dest.cmp(&u) {
                                        std::cmp::Ordering::Equal => break m,
                                        std::cmp::Ordering::Greater => ceil = m,
                                        std::cmp::Ordering::Less => floor = m + 1,
                                    }
                                };

                                *er.get_mut(edge_offset) = Ix::new(reciprocal);
                                *er.get_mut(reciprocal) = Ix::new(edge_offset);
                                // reset
                                prev = OneOrMany::One((v, reciprocal));
                        }

                        prev = OneOrMany::One((node_count, 0));
                    }
                    Ok(())
                    };
                    let matched = matched();

                    synchronize.wait();
                    matched?;

                    // check: every edge u -> v is paired with an edge v -> u, paired back with it
                    for u in begin..end {
                        for edge_offset in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                            let v = neighbours_ptr.get(edge_offset).index();
                            let reciprocal = er.get(edge_offset).index();
                            let paired = reciprocal < edge_count
                                && (index_ptr.get(v).index()..index_ptr.get(v + 1).index()).contains(&reciprocal)
                                && neighbours_ptr.get(reciprocal).index() == u
                                && er.get(reciprocal).index() == edge_offset;
                            if !paired {
                                return Err(format!("error edge {edge_offset}, u: ({u}) -> v: ({v}), was paired with edge {reciprocal}, which isn't its reciprocal").into());
                            }
                        }
                    }

                    Ok(())
                }));
            }

            // check for errors
            for (tid, r) in threads_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        er.flush()?;
        eo.flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use std::io::Write;
    use std::path::Path;

    fn graph_from_mtx(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
    ) -> TestGraph<VoidLabel, VoidLabel, usize> {
        let dir = unique_test_dir("impl_csc_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let graph = GraphMemoryMap::from_file(
            &path,
            Some(format!("impl_csc_{name}_{}", unique_run_tag())),
            Some(2),
        )
        .expect("graph parse should succeed");
        TestGraph::new(graph, Some(dir))
    }

    #[test]
    fn is_directed_counts_multiplicity() {
        for (name, header, n, edges, symmetric) in [
            ("sym_path", "symmetric", 3, vec![(1, 0), (2, 1)], true),
            ("dir_path", "general", 3, vec![(0, 1), (1, 2)], false),
            ("both_ways", "general", 3, vec![(0, 1), (1, 0), (1, 2), (2, 1)], true),
            ("self_loop", "general", 2, vec![(0, 0), (1, 1)], true),
            ("multi_unbalanced", "general", 2, vec![(0, 1), (0, 1), (1, 0)], false),
            ("multi_balanced", "general", 2, vec![(0, 1), (0, 1), (1, 0), (1, 0)], true),
        ] {
            let g = graph_from_mtx(name, header, n, &edges);
            assert_eq!(g.is_directed().unwrap(), !symmetric, "{name}");
            // remembered
            assert_eq!(g.is_directed().unwrap(), !symmetric, "{name}, again");
        }
    }

    /// Asymmetric graphs used to get reciprocal indices of zeros (arcs to higher nodes were never
    /// checked), mismatched edges (the binary search read past a node's neighbours), or a
    /// subtraction overflow; and a failed build left files behind that later calls trusted.
    #[test]
    fn edge_reciprocal_refuses_asymmetric_graphs_and_leaves_no_files() {
        for (name, n, edges) in [
            ("upward_dag", 3, vec![(0, 1), (0, 2), (1, 2)]),
            ("downward", 3, vec![(1, 0), (2, 1)]),
            ("search_overrun", 3, vec![(1, 2), (2, 0)]),
            ("search_underflow", 3, vec![(0, 2), (1, 0)]),
            ("one_upward", 3, vec![(0, 1), (1, 0), (1, 2)]),
        ] {
            let g = graph_from_mtx(name, "general", n, &edges);
            let er_fn = g.build_cache_filename(CacheFile::EdgeReciprocal, None).unwrap();
            let eo_fn = g.build_cache_filename(CacheFile::EdgeOver, None).unwrap();
            for attempt in 0..2 {
                let err = g.edge_reciprocal().err().unwrap_or_else(|| {
                    panic!("{name}: attempt {attempt} should fail")
                });
                assert!(err.to_string().contains("needs an undirected graph"), "{name}: {err}");
                assert!(g.edge_over().is_err(), "{name}: edge over, attempt {attempt}");
                for f in [&er_fn, &eo_fn] {
                    assert!(!Path::new(f).exists(), "{name}: {f} left behind");
                    assert!(!Path::new(&format!("{f}.part")).exists(), "{name}: {f}.part left behind");
                }
            }
        }
    }

    #[test]
    fn edge_reciprocal_pairs_every_edge_of_a_symmetric_multigraph() {
        // parallel edges, self-loops, and a node (3) with no neighbours above it
        let edges = [(0, 1), (0, 1), (1, 2), (2, 2), (0, 2), (3, 0), (3, 1)];
        let g = graph_from_mtx("sym_multi", "symmetric", 4, &edges);
        let er = g.edge_reciprocal().unwrap();
        let er = er.as_slice();
        let index = SharedSlice::<usize>::new(g.offsets_ptr(), g.offsets_size());
        let offsets = |w: usize| *index.get(w)..*index.get(w + 1);
        for u in 0..g.size() {
            for e in offsets(u) {
                let v = g[..][e];
                let r = er[e];
                assert!(offsets(v).contains(&r), "edge {e} {u}->{v} paired outside {v}'s edges");
                assert_eq!(g[..][r], u, "edge {e} {u}->{v} paired with a wrong edge");
                assert_eq!(er[r], e, "edge {e} {u}->{v} not paired back");
            }
        }
    }

    #[test]
    fn undirected_algorithms_refuse_directed_graphs() {
        use crate::communities::{
            approx_dirichlet_hkpr::ApproxDirHKPR, gve_louvain::AlgoGVELouvain, hk_relax::HKRelax,
        };
        use crate::k_core::{batagelj_zaversnik::AlgoBatageljZaversnik, liu_et_al::AlgoLiuEtAl};
        use crate::k_truss::directed::{DirectedTriangleType, DirectedTriangles};
        use crate::k_truss::{
            burkhardt_et_al::AlgoBurkhardtEtAl, clustering_coefficient::ClusteringCoefficient,
            pkt::AlgoPKT, triangles::Triangles,
        };

        let g = graph_from_mtx("dag", "general", 4, &[(0, 1), (0, 2), (1, 2), (2, 3)]);
        let refused = |what: &str, err: Option<Box<dyn std::error::Error>>| {
            let err = err.unwrap_or_else(|| panic!("{what} accepted a directed graph"));
            assert!(err.to_string().contains("needs an undirected graph"), "{what}: {err}");
        };
        refused("triangles", Triangles::<_, _, _, u8>::new_tx(&g).err());
        refused("pkt", AlgoPKT::<_, _, _, u8>::new_tx(&g, 2).err());
        refused("burkhardt et al.", AlgoBurkhardtEtAl::<_, _, _, u8>::new_tx(&g).err());
        refused("clustering coefficient", ClusteringCoefficient::new(&g).err());
        refused("batagelj zaversnik", AlgoBatageljZaversnik::new(&g).err());
        refused("liu et al.", AlgoLiuEtAl::new(&g).err());
        refused("louvain", AlgoGVELouvain::<_, _, _, f64>::new(&g).err());
        refused("modularity", g.modularity(&[0, 0, 1, 1], 2).err());
        refused("sweep cut", g.sweep_cut(&mut [(0, 1.), (1, 0.5)], None, None).err());
        refused("hk-relax", HKRelax::new(&g, 1., 1e-3, vec![0], None, None).err());
        refused("approx. dirichlet hkpr", ApproxDirHKPR::new(&g, 0.05, 0, 2, 2, 0.5).err());

        // the directed algorithms still accept it
        let t = DirectedTriangleType::FeedForward;
        assert!(DirectedTriangles::<_, _, _, u8>::new_tx(&g, t).is_ok());
        assert!(AlgoPKT::<_, _, _, u8>::new_directed(&g, t).is_ok());
        assert!(AlgoBurkhardtEtAl::<_, _, _, u8>::new_directed(&g, t).is_ok());
    }
}
