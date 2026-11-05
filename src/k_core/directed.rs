use crate::graph;
use crate::graph::*;
use crate::k_core::count::CoreCount;
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
/// Computed in `O(|V| + |E|)` by Batagelj and Zaversnik's bucket peeling (["An O(m) Algorithm
/// for Cores Decomposition of Networks"](https://doi.org/10.48550/arXiv.cs/0310049)), peeling by
/// in-degree a view of the graph: peeling a node lowers the in-degree of its out-neighbours, so
/// in-cores peel the graph, out-cores its transpose and total degree cores its underlying
/// undirected graph with an edge per arc ([`UndirectedMode::Each`]).
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
        // the view first, so that failing to build it leaves no result file behind
        enum View<'v, N: graph::N, E: graph::E, Ix: graph::IndexType> {
            Directed(&'v DiGraph<N, E, Ix>),
            Undirected(&'v Graph<N, E, Ix>),
        }
        let view = match degree {
            CoreDegree::In => View::Directed(g),
            CoreDegree::Out => View::Directed(g.transpose()?),
            // in plus out degree: every arc counts, both ways too
            CoreDegree::Total => View::Undirected(g.to_undirected_with(UndirectedMode::Each)?),
        };
        let out_fn = g.build_cache_filename(CacheFile::KCoreDirected(degree), None)?;
        let k_cores = SharedSliceMut::<Cx>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut cores = Self { g, degree, k_cores };
        let peeled = match view {
            View::Directed(v) => cores.peel(v),
            View::Undirected(v) => cores.peel(v),
        };
        if let Err(e) = peeled {
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

    /// Peels `p` (a view of the graph, with the same nodes) by in-degree: removing a node lowers
    /// the in-degree of its out-neighbours.
    fn peel<Dir: Directedness>(
        &mut self,
        p: &GraphMemoryMap<N, E, Ix, Dir>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = p.size();
        if node_count == 0 {
            return Ok(());
        }
        let index_ptr = SharedSlice::<Ix>::new(p.offsets_ptr(), p.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(p.neighbours_ptr(), p.width());
        let degree = self.degree;
        let helper = |seq| {
            self.g
                .build_cache_filename(CacheFile::KCoreDirected(degree), Some(seq))
        };
        let helpers = [helper(0)?, helper(1)?, helper(2)?];
        let mut degree_mem = SharedSliceMut::<Ix>::abst_mem_mut(&helpers[0], node_count, true)?;
        let mut node_mem = SharedSliceMut::<Ix>::abst_mem_mut(&helpers[1], node_count, true)?;
        let mut pos_mem = SharedSliceMut::<Ix>::abst_mem_mut(&helpers[2], node_count, true)?;
        let (degree, node, pos) = (
            degree_mem.as_mut_slice(),
            node_mem.as_mut_slice(),
            pos_mem.as_mut_slice(),
        );
        let core = self.k_cores.as_mut_slice();

        // in-degrees, without self-loops
        let mut max_degree = 0usize;
        for u in 0..node_count {
            for e in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                let v = neighbours_ptr.get(e).index();
                if v != u {
                    degree[v] = Ix::new(degree[v].index() + 1);
                    max_degree = max_degree.max(degree[v].index());
                }
            }
        }

        if max_degree > Cx::MAX {
            let v = (0..node_count)
                .find(|&v| degree[v].index() == max_degree)
                .unwrap_or_default();
            let e = crate::k_core::count::check_degree::<Cx>(v, max_degree).unwrap_err();
            drop((degree_mem, node_mem, pos_mem));
            for f in &helpers {
                let _ = std::fs::remove_file(f);
            }
            return Err(e.to_string().into());
        }

        // nodes sorted by degree; bins[d]: where nodes of degree d start
        let mut bins = vec![0usize; max_degree + 2];
        for d in degree.iter() {
            bins[d.index() + 1] += 1;
        }
        for d in 0..=max_degree {
            bins[d + 1] += bins[d];
        }
        for v in 0..node_count {
            let d = degree[v].index();
            pos[v] = Ix::new(bins[d]);
            node[bins[d]] = Ix::new(v);
            bins[d] += 1;
        }
        for d in (1..=max_degree).rev() {
            bins[d] = bins[d - 1];
        }
        bins[0] = 0;

        // peel nodes in order of current degree
        for i in 0..node_count {
            let v = node[i].index();
            let degree_v = degree[v].index();
            core[v] = Cx::from_usize(degree_v);
            for e in index_ptr.get(v).index()..index_ptr.get(v + 1).index() {
                let u = neighbours_ptr.get(e).index();
                let degree_u = degree[u].index();
                if u == v || degree_u <= degree_v {
                    continue;
                }
                // move u to the front of its bin, then shrink the bin past it
                let (u_pos, front) = (pos[u].index(), bins[degree_u]);
                let w = node[front].index();
                if u != w {
                    node[u_pos] = Ix::new(w);
                    pos[w] = Ix::new(u_pos);
                    node[front] = Ix::new(u);
                    pos[u] = Ix::new(front);
                }
                bins[degree_u] += 1;
                degree[u] = Ix::new(degree_u - 1);
            }
        }

        self.k_cores.flush_async()?;
        drop((degree_mem, node_mem, pos_mem));
        for f in &helpers {
            let _ = std::fs::remove_file(f);
        }
        Ok(())
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
