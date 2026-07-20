use crate::graph;
use crate::graph::cache::AccumulatorAtomic;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::Ordering;
use std::mem::ManuallyDrop;
use std::path::Path;

/// The weakly connected components of a [`GraphMemoryMap`] instance: its nodes partitioned by
/// whether they're linked by a path ignoring arc directions. For undirected graphs, these are
/// its connected components.
///
/// Computed with a lock-free union-find (linking roots to smaller roots, with path halving)
/// over every arc, in parallel. Components are numbered by their smallest node.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct ConnectedComponents<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph whose components are computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the component of each node.
    components: AbstractedProceduralMemoryMut<Ix>,
    /// Number of components.
    count: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ConnectedComponents<'a, N, E, Ix> {
    /// Computes the weakly connected components of the graph (its connected components, if
    /// it's undirected).
    ///
    /// The resulting component of each node is stored in memory (in a memmapped file).
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance whose components are to be computed.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::ConnectedComponents, None)?;
        let components = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut cc = Self {
            g,
            components,
            count: 0,
        };
        cc.compute()?;
        Ok(cc)
    }

    /// Searches for a previously cached result, and if not found computes the weakly connected
    /// components of the graph.
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let c_fn = g.build_cache_filename(CacheFile::ConnectedComponents, None)?;
        if Path::new(&c_fn).exists()
            && let Ok(components) = AbstractedProceduralMemoryMut::<Ix>::from_file_name(&c_fn)
            && components.as_slice().len() == g.size()
        {
            // components are numbered in order: the last one is the largest number
            let count = components.as_slice().iter().map(|c| c.index() + 1).max().unwrap_or(0);
            return Ok(Self {
                g,
                components,
                count,
            });
        }
        Self::new(g)
    }

    /// Returns the component of a given node.
    pub fn component(&self, node: usize) -> usize {
        assert!(node < self.g.size());
        self.components.get(node).index()
    }

    /// Returns a slice containing the component of each node.
    pub fn components(&self) -> &[Ix] {
        self.components.as_slice()
    }

    /// Returns the number of components.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the number of nodes of each component.
    pub fn sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.count];
        for c in self.components() {
            sizes[c.index()] += 1;
        }
        sizes
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        this.g.cleanup_cache(CacheFile::ConnectedComponents)?;
        let out_fn = this.g.build_cache_filename(CacheFile::ConnectedComponents, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.count = label_components(self.g, &self.components, CacheFile::ConnectedComponents)?;
        Ok(())
    }
}

/// Labels every node with its weakly connected component (numbered by smallest node), using a
/// scratch file of `scratch`'s, and returns the number of components.
pub(super) fn label_components<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    g: &GraphMemoryMap<N, E, Ix>,
    components_mem: &AbstractedProceduralMemoryMut<Ix>,
    scratch: CacheFile,
) -> Result<usize, Box<dyn std::error::Error>> {
    let node_count = g.size();
    if node_count == 0 {
        return Ok(0);
    }
    let edge_count = g.width();
    let index_ptr = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
    let neighbours_ptr = SharedSlice::<Ix>::new(g.neighbours_ptr(), edge_count);
    // an undirected graph stores each edge both ways: one of them is enough
    let directed = g.is_directed()?;

    let p_fn = g.build_cache_filename(scratch, Some(0))?;
    let parent = SharedSliceMut::<Ix::Atomic>::abst_mem_mut(&p_fn, node_count, true)?;

    let threads = g.thread_num();
    let node_load = node_count.div_ceil(threads);
    let in_parallel = |f: &(dyn Fn(usize, usize) + Sync)| -> Result<(), Box<dyn std::error::Error>> {
        thread::scope(|scope| {
            for tid in 0..threads {
                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                scope.spawn(move |_| f(begin, end));
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })
    };

    // every node its own root
    let parents = parent.shared_slice();
    in_parallel(&|begin, end| {
        for u in begin..end {
            // node ids fit in `Ix`
            let _ = parents.get(u).checked_store(u, Ordering::Relaxed);
        }
    })?;

    // roots only ever link to smaller roots, and path halving only replaces a parent by
    // its own (smaller or equal) parent: parents never exceed their nodes, so there are no
    // cycles, and a component's root is its smallest node
    let find = |mut x: usize| -> usize {
        loop {
            let p = parents.get(x).load_widened(Ordering::Acquire);
            if p == x {
                return x;
            }
            let grandparent = parents.get(p).load_widened(Ordering::Acquire);
            if grandparent != p {
                parents.get(x).compare_exchange_widened(p, grandparent, Ordering::AcqRel);
            }
            x = p;
        }
    };
    in_parallel(&|begin, end| {
        for u in begin..end {
            for e in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                let v = neighbours_ptr.get(e).index();
                if v == u || (!directed && v > u) {
                    continue;
                }
                loop {
                    let (ru, rv) = (find(u), find(v));
                    if ru == rv {
                        break;
                    }
                    let (high, low) = if ru > rv { (ru, rv) } else { (rv, ru) };
                    if parents.get(high).compare_exchange_widened(high, low, Ordering::AcqRel) {
                        break;
                    }
                }
            }
        }
    })?;

    // number components by their smallest node, i.e. their root: roots precede their nodes
    let mut components = components_mem.shared_slice();
    let mut count = 0usize;
    for u in 0..node_count {
        let root = find(u);
        let component = if root == u {
            count += 1;
            count - 1
        } else {
            components.get(root).index()
        };
        *components.get_mut(u) = Ix::new(component);
    }

    components_mem.flush_async()?;
    drop(parent);
    let _ = std::fs::remove_file(&p_fn);
    Ok(count)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, graph_with_threads, label, random_arcs};

    fn expected(n: usize, arcs: &[(usize, usize)]) -> Vec<usize> {
        // reachability of the underlying undirected graph
        let both: Vec<_> = arcs.iter().flat_map(|&(u, v)| [(u, v), (v, u)]).collect();
        let reach = crate::components::test_utils::reachability(n, &both);
        label(n, |u, v| reach[u][v])
    }

    #[test]
    fn weak_components_match_brute_force() {
        for (name, n, arcs) in [
            ("isolated", 4, vec![(1, 1)]),
            ("path", 5, vec![(0, 1), (2, 1), (3, 4)]),
            ("cycle_and_tail", 6, vec![(0, 1), (1, 2), (2, 0), (2, 3), (5, 4)]),
            ("random_sparse", 60, random_arcs(60, 45, 3)),
            ("random_dense", 60, random_arcs(60, 400, 5)),
        ] {
            let g = graph(name, "general", n, &arcs);
            let cc = ConnectedComponents::new(&g).unwrap();
            let got: Vec<usize> = cc.components().iter().map(|c| c.index()).collect();
            let want = expected(n, &arcs);
            assert_eq!(got, want, "{name}");
            assert_eq!(cc.count(), want.iter().max().map_or(0, |m| m + 1), "{name}");
            assert_eq!(cc.sizes().iter().sum::<usize>(), n, "{name}");
        }
    }

    /// Many threads racing on a large graph, against a sequential union-find.
    #[test]
    fn weak_components_under_contention() {
        let n = 20_000;
        for seed in 0..4 {
            let arcs = random_arcs(n, 16_000, seed);
            let mut parent: Vec<usize> = (0..n).collect();
            fn find(parent: &mut [usize], mut x: usize) -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                x
            }
            for &(u, v) in &arcs {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                parent[ru.max(rv)] = ru.min(rv);
            }
            // roots are smallest nodes: number them in order
            let mut number = vec![usize::MAX; n];
            let mut count = 0;
            let want: Vec<usize> = (0..n)
                .map(|u| {
                    let root = find(&mut parent, u);
                    if number[root] == usize::MAX {
                        number[root] = count;
                        count += 1;
                    }
                    number[root]
                })
                .collect();

            let g = graph_with_threads(&format!("contention_{seed}"), "general", n, &arcs, 16);
            let cc = ConnectedComponents::new(&g).unwrap();
            let got: Vec<usize> = cc.components().iter().map(|c| c.index()).collect();
            assert_eq!(got, want, "seed {seed}");
        }
    }

    #[test]
    fn components_of_undirected_graphs() {
        let edges = [(0, 1), (1, 2), (3, 4), (5, 5)];
        let g = graph("undirected", "symmetric", 7, &edges);
        let cc = ConnectedComponents::new(&g).unwrap();
        assert_eq!(cc.components(), &[0, 0, 0, 1, 1, 2, 3]);
        assert_eq!(cc.count(), 4);
        assert_eq!(cc.sizes(), vec![3, 2, 1, 1]);
        let cached = ConnectedComponents::get_or_compute(&g).unwrap();
        assert_eq!(cached.count(), 4);
    }
}
