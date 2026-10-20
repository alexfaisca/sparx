use super::{CacheFile, GraphMemoryMap};
use crate::shared_slice::{
    AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut,
};

use crossbeam::thread;
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
                        let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
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
                        let len = i.metadata().unwrap().len() as usize / std::mem::size_of::<Ix>();
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

        // edges `u -> v` below (`v < u`) and above (`v > u`) their node, all threads'
        let (below, above) = thread::scope(|scope| -> Result<(usize, usize), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let mut er = er.shared_slice();
                let mut eo = eo.shared_slice();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                threads_res.push(scope.spawn(move |_| -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {

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
                    let mut matched = || -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
                        // every run of parallel edges u -> v (v <= u) is paired, in order, with the
                        // run of edges v -> u in v's own neighbours, as long (the graph being
                        // symmetric), and found by binary search within v's neighbours above v
                        let (mut below, mut above) = (0, 0);
                        for u in begin..end {
                            let stop = eo.get(u).index();
                            above += index_ptr.get(u + 1).index() - stop;
                            let mut e = index_ptr.get(u).index();
                            while e < stop {
                                let v = neighbours_ptr.get(e).index();
                                let mut run = 1;
                                while e + run < stop && neighbours_ptr.get(e + run).index() == v {
                                    run += 1;
                                }
                                if v == u {
                                    // self-loops are their own reciprocals
                                    for k in e..e + run {
                                        *er.get_mut(k) = Ix::new(k);
                                    }
                                } else {
                                    let end_v = index_ptr.get(v + 1).index();
                                    let (mut lo, mut hi) = (eo.get(v).index(), end_v);
                                    while lo < hi {
                                        let m = lo + (hi - lo) / 2;
                                        if neighbours_ptr.get(m).index() < u {
                                            lo = m + 1;
                                        } else {
                                            hi = m;
                                        }
                                    }
                                    if lo + run > end_v || (0..run).any(|k| neighbours_ptr.get(lo + k).index() != u) {
                                        return Err(format!(
                                            "error couldn't find the reciprocals of edges {e}..{}, u: ({u}) -> v: ({v})",
                                            e + run
                                        )
                                        .into());
                                    }
                                    for k in 0..run {
                                        *er.get_mut(e + k) = Ix::new(lo + k);
                                        *er.get_mut(lo + k) = Ix::new(e + k);
                                    }
                                    below += run;
                                }
                                e += run;
                            }
                        }
                        Ok((below, above))
                    };
                    let matched = matched();

                    synchronize.wait();
                    let counts = matched?;

                    // every edge below its node was paired with one above its own (an edge
                    // v -> u for each u -> v, in a run as long), and no two with the same one
                    // (each run with v's run of edges to u): so every edge above its node was
                    // paired too if there are as many as below, which is checked once all threads
                    // are done. Debug builds also check every pair (two random reads per edge)
                    if !cfg!(debug_assertions) {
                        return Ok(counts);
                    }
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

                    Ok(counts)
                }));
            }

            // check for errors
            let (mut below, mut above) = (0, 0);
            for (tid, r) in threads_res.into_iter().enumerate() {
                let (b, a) = r
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
                below += b;
                above += a;
            }
            Ok((below, above))
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        if below != above {
            return Err(format!(
                "error the graph isn't symmetric: {above} edges u -> v (v > u) but {below} with an edge v -> u"
            )
            .into());
        }

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

    fn digraph_from_mtx(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
    ) -> TestGraph<VoidLabel, VoidLabel, usize, crate::graph::Directed> {
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
        let graph = crate::graph::DiGraph::from_file(
            &path,
            Some(format!("impl_csc_{name}_{}", unique_run_tag())),
            Some(2),
        )
        .expect("graph parse should succeed");
        TestGraph::new(graph, Some(dir))
    }

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
    fn symmetry_counts_multiplicity() {
        for (name, header, n, edges, symmetric) in [
            ("sym_path", "symmetric", 3, vec![(1, 0), (2, 1)], true),
            ("dir_path", "general", 3, vec![(0, 1), (1, 2)], false),
            (
                "both_ways",
                "general",
                3,
                vec![(0, 1), (1, 0), (1, 2), (2, 1)],
                true,
            ),
            ("self_loop", "general", 2, vec![(0, 0), (1, 1)], true),
            (
                "multi_unbalanced",
                "general",
                2,
                vec![(0, 1), (0, 1), (1, 0)],
                false,
            ),
            (
                "multi_balanced",
                "general",
                2,
                vec![(0, 1), (0, 1), (1, 0), (1, 0)],
                true,
            ),
        ] {
            let g = digraph_from_mtx(name, header, n, &edges);
            assert_eq!(g.is_symmetric().unwrap(), symmetric, "{name}");
            // remembered
            assert_eq!(g.is_symmetric().unwrap(), symmetric, "{name}, again");
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
                assert!(
                    offsets(v).contains(&r),
                    "edge {e} {u}->{v} paired outside {v}'s edges"
                );
                assert_eq!(g[..][r], u, "edge {e} {u}->{v} paired with a wrong edge");
                assert_eq!(er[r], e, "edge {e} {u}->{v} not paired back");
            }
        }
    }

    /// Parallel edges next to another node's neighbours: the reciprocals of the 2nd and later
    /// copies used to be looked for past the end of the node's own neighbours.
    #[test]
    fn edge_reciprocal_pairs_parallel_edges_of_random_multigraphs() {
        use crate::k_truss::{
            burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT, triangles::Triangles,
        };
        let mut cases = vec![("reviewed", 3, vec![(2, 0), (2, 0), (2, 1)])];
        for seed in 0..12u64 {
            let mut state = seed + 1;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as usize
            };
            let n = 3 + next() % 20;
            let mut edges = vec![];
            for _ in 0..(n * 3) {
                let (u, v) = (next() % n, next() % n);
                // heavy parallelism and self-loops
                for _ in 0..1 + next() % 3 {
                    edges.push((u.max(v), u.min(v)));
                }
            }
            cases.push(("random", n, edges));
        }
        for (case, (name, n, edges)) in cases.into_iter().enumerate() {
            let g = graph_from_mtx(&format!("{name}_{case}"), "symmetric", n, &edges);
            let er = g
                .edge_reciprocal()
                .unwrap_or_else(|e| panic!("{name} {case}: {e}"));
            let er = er.as_slice();
            let index = SharedSlice::<usize>::new(g.offsets_ptr(), g.offsets_size());
            for u in 0..g.size() {
                for e in *index.get(u)..*index.get(u + 1) {
                    let (v, r) = (g[..][e], er[e]);
                    assert!(
                        (*index.get(v)..*index.get(v + 1)).contains(&r),
                        "{name} {case}: edge {e}"
                    );
                    assert_eq!(
                        (g[..][r], er[r]),
                        (u, e),
                        "{name} {case}: edge {e} {u}->{v}"
                    );
                }
            }
            // the triangle based algorithms run, and agree
            Triangles::<_, _, _, u32>::new_tx(&g).unwrap_or_else(|e| panic!("{name} {case}: {e}"));
            let pkt = AlgoPKT::<_, _, _, u32>::new_tx(&g, 3)
                .unwrap()
                .k_trusses()
                .to_vec();
            let bea = AlgoBurkhardtEtAl::<_, _, _, u32>::new_tx(&g)
                .unwrap()
                .k_trusses()
                .to_vec();
            assert_eq!(pkt, bea, "{name} {case}");
        }
    }
}
