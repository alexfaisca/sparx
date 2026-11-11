use crate::graph;
use crate::graph::csr::Csr;
use crate::graph::*;
use crate::k_core::count::CoreCount;
use crate::k_core::liu_et_al::liu_peel;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;

/// The degree a directed graph's k-cores are by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoreDegree {
    /// Arcs into a node.
    In,
    /// Arcs out of a node.
    Out,
    /// Arcs in and out of a node: what networkx's `core_number` uses on directed graphs. The
    /// default.
    #[default]
    Total,
}

/// For the computation of a [`DiGraph`]'s k-core decomposition, by in-, out- or total degree
/// (see [`CoreDegree`]): a node's coreness is the largest `k` such that it's in a subgraph whose
/// every node has at least `k` arcs in (out, or in and out together) from within it. Self-loops
/// are ignored.
///
/// [`CoreDegree::Total`] is what networkx's `core_number` computes on directed graphs. On a
/// symmetric graph, in- and out-cores are its usual k-cores (see [`AlgoBatageljZaversnik`], for
/// [`Graph`]s), and total degree ones twice those (every edge being two arcs).
///
/// Computed in parallel by Liu et al.'s peeling (["Parallel k-Core Decomposition: Theory and
/// Practice"](https://doi.org/10.48550/arXiv.2502.08042), as [`AlgoLiuEtAl`] for [`Graph`]s),
/// peeling by in-degree a view of the graph: peeling a node lowers the in-degree of its
/// out-neighbours, so in-cores peel the graph, out-cores its transpose and total degree cores
/// its underlying undirected graph with an edge per arc ([`UndirectedMode::Each`]). The result is
/// the same whatever the threads.
///
/// Coreness is stored as `Cx` integers (`u8` by default, see [`CoreCount`]): every node's degree
/// (of the kind cores are by) must fit a `Cx`, or the decomposition fails with an error.
///
/// [`CoreCount`]: ../count/trait.CoreCount.html#
/// [`UndirectedMode::Each`]: ../../graph/direction/enum.UndirectedMode.html#variant.Each
///
/// [`DiGraph`]: ../../graph/type.DiGraph.html
/// [`Graph`]: ../../graph/type.Graph.html
/// [`AlgoBatageljZaversnik`]: ../batagelj_zaversnik/struct.AlgoBatageljZaversnik.html
/// [`AlgoLiuEtAl`]: ../liu_et_al/struct.AlgoLiuEtAl.html
#[allow(dead_code)]
#[derive(Debug)]
pub struct DirectedKCores<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Cx: CoreCount = u8> {
    /// Graph for which node coreness is computed.
    g: &'a DiGraph<N, E, Ix>,
    /// The degree cores are by.
    degree: CoreDegree,
    /// Memmapped slice containing the coreness of each node.
    k_cores: AbstractedProceduralMemoryMut<Cx>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedKCores<'a, N, E, Ix> {
    /// Performs the k-core decomposition of the graph by a given degree.
    ///
    /// The coreness of each node is stored in memory (in a memmapped file), as `u8`s: see
    /// [`new_cx`](Self::new_cx) for other widths.
    pub fn new(
        g: &'a DiGraph<N, E, Ix>,
        degree: CoreDegree,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_cx(g, degree)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Cx: CoreCount>
    DirectedKCores<'a, N, E, Ix, Cx>
{
    /// Performs the k-core decomposition of the graph by a given degree, storing coreness as `Cx`
    /// integers.
    pub fn new_cx(
        g: &'a DiGraph<N, E, Ix>,
        degree: CoreDegree,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // the views first, so that failing to build them leaves no result file behind: the
        // graph peeled by in-degree (peeling a node lowers the in-degree of its out-neighbours),
        // and its transpose, whose rows are its in-arcs
        let (peeled, reverse) = match degree {
            CoreDegree::In => (Csr::of(g), Csr::of(g.transpose()?)),
            CoreDegree::Out => (Csr::of(g.transpose()?), Csr::of(g)),
            // in plus out degree: every arc counts, both ways too
            CoreDegree::Total => {
                let both = Csr::of(g.to_undirected_with(UndirectedMode::Each)?);
                (both, both)
            }
        };
        let out_fn = g.build_cache_filename(CacheFile::KCoreDirected(degree), None)?;
        let k_cores = SharedSliceMut::<Cx>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut cores = Self { g, degree, k_cores };
        if let Err(e) = cores.peel(peeled, reverse) {
            let _ = std::fs::remove_file(&out_fn);
            return Err(e);
        }
        Ok(cores)
    }

    /// Returns the degree the cores are by.
    pub fn degree(&self) -> CoreDegree {
        self.degree
    }

    /// Returns the coreness of a given node.
    pub fn coreness(&self, node: usize) -> Cx {
        assert!(node < self.g.size());
        *self.k_cores.get(node)
    }

    /// Returns a slice containing the coreness of each node.
    pub fn k_cores(&self) -> &[Cx] {
        self.k_cores.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let target = || CacheFile::KCoreDirected(this.degree);
        this.g.cleanup_cache(target())?;
        crate::shared_slice::remove_if_exists(this.g.build_cache_filename(target(), None)?)?;
        Ok(())
    }

    /// Peels the graph whose arcs are `peeled` by in-degree (each node's in-arcs being its row
    /// of `reverse`, self-loops not counted), in parallel: Liu et al.'s peeling (see
    /// [`AlgoLiuEtAl`]).
    ///
    /// [`AlgoLiuEtAl`]: ../liu_et_al/struct.AlgoLiuEtAl.html
    fn peel(
        &mut self,
        peeled: Csr<Ix, E>,
        reverse: Csr<Ix, E>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        if node_count == 0 {
            return Ok(());
        }
        let target = || CacheFile::KCoreDirected(self.degree);
        let files: Vec<String> = (0..5)
            .map(|seq| self.g.build_cache_filename(target(), Some(seq)))
            .collect::<Result<_, _>>()?;
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            // each node is in a frontier once
            let proc_mem = (
                SharedSliceMut::<Cx::Atomic>::abst_mem_mut(&files[0], node_count, true)?,
                SharedSliceMut::<Ix>::abst_mem_mut(&files[1], node_count, true)?,
                SharedSliceMut::<portable_atomic::AtomicBool>::abst_mem_mut(
                    &files[2], node_count, true,
                )?,
                SharedSliceMut::<Ix>::abst_mem_mut(&files[3], node_count, true)?,
                SharedSliceMut::<Ix>::abst_mem_mut(&files[4], node_count, true)?,
            );
            // in-degrees, without self-loops (rows are sorted)
            let in_degree = |v: usize| {
                let row = &reverse.targets[reverse.range(v)];
                let loops = row.partition_point(|x| x.index() < v);
                let loops = row[loops..].iter().take_while(|x| x.index() == v).count();
                row.len() - loops
            };
            let slice = |s: &[Ix]| SharedSlice::<Ix>::new(s.as_ptr(), s.len());
            liu_peel(
                slice(peeled.offsets),
                slice(peeled.targets),
                node_count,
                self.g.thread_num(),
                &in_degree,
                proc_mem,
                self.k_cores.shared_slice(),
            )?;
            self.k_cores.flush_async()?;
            Ok(())
        })();
        for f in &files {
            let _ = std::fs::remove_file(f);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, random_arcs};

    /// Peels by repeatedly removing the nodes whose degree (counted by `degree` over the
    /// remaining arcs, without self-loops) is below k.
    fn brute_force(n: usize, arcs: &[(usize, usize)], degree: CoreDegree) -> Vec<usize> {
        let mut core = vec![0; n];
        let mut alive = vec![true; n];
        let mut k = 0;
        while alive.iter().any(|&a| a) {
            loop {
                let mut count = vec![0; n];
                for &(u, v) in arcs {
                    if u != v && alive[u] && alive[v] {
                        match degree {
                            CoreDegree::In => count[v] += 1,
                            CoreDegree::Out => count[u] += 1,
                            CoreDegree::Total => {
                                count[u] += 1;
                                count[v] += 1;
                            }
                        }
                    }
                }
                let removed: Vec<usize> = (0..n).filter(|&v| alive[v] && count[v] < k).collect();
                if removed.is_empty() {
                    break;
                }
                for v in removed {
                    alive[v] = false;
                }
            }
            for v in 0..n {
                if alive[v] {
                    core[v] = k;
                }
            }
            k += 1;
        }
        core
    }

    #[test]
    fn directed_k_cores_match_brute_force() {
        for (name, n, arcs) in [
            ("path", 3, vec![(0, 1), (1, 2)]),
            (
                "cycle_and_star",
                6,
                vec![(0, 1), (1, 2), (2, 0), (3, 0), (4, 0), (5, 0), (0, 0)],
            ),
            ("random_sparse", 40, random_arcs(40, 80, 1)),
            ("random_dense", 40, random_arcs(40, 500, 2)),
        ] {
            let g = graph(name, "general", n, &arcs);
            for degree in [CoreDegree::In, CoreDegree::Out, CoreDegree::Total] {
                let cores = DirectedKCores::new(&g, degree).unwrap();
                let got: Vec<usize> = cores.k_cores().iter().map(|c| c.to_usize()).collect();
                assert_eq!(got, brute_force(n, &arcs, degree), "{name}, {degree:?}");
            }
        }
    }

    #[test]
    fn directed_k_cores_of_undirected_graphs_are_the_usual_ones() {
        use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
        let mut edges: Vec<_> = random_arcs(40, 200, 6)
            .into_iter()
            .filter(|(u, v)| u < v)
            .collect();
        edges.sort();
        edges.dedup();
        let g = graph("undirected", "symmetric", 40, &edges);
        let usual: Vec<usize> = AlgoBatageljZaversnik::new(g.to_undirected().unwrap())
            .unwrap()
            .k_cores()
            .iter()
            .map(|&c| c as usize)
            .collect();
        // in- and out-cores are the usual cores; total degree cores count each edge once per arc
        // (twice), as networkx's `core_number` of the symmetric digraph does
        for (degree, factor) in [
            (CoreDegree::In, 1),
            (CoreDegree::Out, 1),
            (CoreDegree::Total, 2),
        ] {
            let cores = DirectedKCores::new(&g, degree).unwrap();
            let got: Vec<usize> = cores.k_cores().iter().map(|c| c.to_usize()).collect();
            let want: Vec<usize> = usual.iter().map(|c| factor * c).collect();
            assert_eq!(got, want, "{degree:?}");
        }
    }
}
