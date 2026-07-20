use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;
use std::path::Path;

/// The strongly connected components of a [`GraphMemoryMap`] instance: its nodes partitioned by
/// whether they're reachable from each other along arcs. For undirected graphs, these are its
/// connected components.
///
/// Computed with an iterative version of Tarjan's algorithm (["Depth-First Search and Linear
/// Graph Algorithms"](https://doi.org/10.1137/0201010)), in `O(|V| + |E|)`, following arcs out
/// of nodes only (undirected graphs' are their weak ones, found in parallel instead). Components
/// are numbered by their smallest node.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct StronglyConnectedComponents<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph whose components are computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the component of each node.
    components: AbstractedProceduralMemoryMut<Ix>,
    /// Number of components.
    count: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> StronglyConnectedComponents<'a, N, E, Ix> {
    /// Computes the strongly connected components of the graph.
    ///
    /// The resulting component of each node is stored in memory (in a memmapped file).
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance whose components are to be computed.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::StronglyConnectedComponents, None)?;
        let components = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut scc = Self {
            g,
            components,
            count: 0,
        };
        scc.compute()?;
        Ok(scc)
    }

    /// Searches for a previously cached result, and if not found computes the strongly
    /// connected components of the graph.
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let c_fn = g.build_cache_filename(CacheFile::StronglyConnectedComponents, None)?;
        if Path::new(&c_fn).exists()
            && let Ok(components) = AbstractedProceduralMemoryMut::<Ix>::from_file_name(&c_fn)
            && components.as_slice().len() == g.size()
        {
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
        this.g.cleanup_cache(CacheFile::StronglyConnectedComponents)?;
        let out_fn = this.g.build_cache_filename(CacheFile::StronglyConnectedComponents, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // undirected graphs' strong components are their weak ones: found in parallel
        if !self.g.is_directed()? {
            self.count = super::weak::label_components(
                self.g,
                &self.components,
                CacheFile::StronglyConnectedComponents,
            )?;
            return Ok(());
        }
        let node_count = self.g.size();
        if node_count == 0 {
            return Ok(());
        }
        let edge_count = self.g.width();
        let index_ptr = SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        let helper = |seq| self.g.build_cache_filename(CacheFile::StronglyConnectedComponents, Some(seq));
        let helpers = [helper(0)?, helper(1)?, helper(2)?, helper(3)?, helper(4)?];
        // discovery order (1-based: 0 is undiscovered) and lowest discovery order reachable
        let mut order = SharedSliceMut::<usize>::abst_mem_mut(&helpers[0], node_count, true)?;
        let mut low = SharedSliceMut::<usize>::abst_mem_mut(&helpers[1], node_count, true)?;
        // Tarjan's stack, and whether a node is on it
        let mut stack = SharedSliceMut::<Ix>::abst_mem_mut(&helpers[2], node_count, true)?;
        let mut on_stack = SharedSliceMut::<bool>::abst_mem_mut(&helpers[3], node_count, true)?;
        // the depth first search's path: each node with its next arc to follow
        let mut path = SharedSliceMut::<(Ix, Ix)>::abst_mem_mut(&helpers[4], node_count, true)?;
        let (order, low, stack, on_stack, path) = (
            order.as_mut_slice(),
            low.as_mut_slice(),
            stack.as_mut_slice(),
            on_stack.as_mut_slice(),
            path.as_mut_slice(),
        );
        let mut components = self.components.shared_slice();

        let (mut discovered, mut stack_len, mut count) = (0usize, 0usize, 0usize);
        for source in 0..node_count {
            if order[source] != 0 {
                continue;
            }
            let mut path_len = 0usize;
            // numbers a node, pushes it on the stack, and continues the search from it
            macro_rules! discover {
                ($v:expr) => {{
                    let v = $v;
                    discovered += 1;
                    order[v] = discovered;
                    low[v] = discovered;
                    stack[stack_len] = Ix::new(v);
                    stack_len += 1;
                    on_stack[v] = true;
                    path[path_len] = (Ix::new(v), *index_ptr.get(v));
                    path_len += 1;
                }};
            }
            discover!(source);

            while path_len > 0 {
                let (v, next) = path[path_len - 1];
                let (v, next) = (v.index(), next.index());
                if next < index_ptr.get(v + 1).index() {
                    path[path_len - 1].1 = Ix::new(next + 1);
                    let w = neighbours_ptr.get(next).index();
                    if order[w] == 0 {
                        discover!(w);
                    } else if on_stack[w] {
                        low[v] = low[v].min(order[w]);
                    }
                    continue;
                }
                // v is done: it roots a component if nothing on the stack below it is reachable
                path_len -= 1;
                if low[v] == order[v] {
                    loop {
                        stack_len -= 1;
                        let w = stack[stack_len].index();
                        on_stack[w] = false;
                        *components.get_mut(w) = Ix::new(count);
                        if w == v {
                            break;
                        }
                    }
                    count += 1;
                }
                if path_len > 0 {
                    let parent = path[path_len - 1].0.index();
                    low[parent] = low[parent].min(low[v]);
                }
            }
        }

        // renumber components by their smallest node (Tarjan's finds them in reverse
        // topological order)
        let renumber = &mut order[..count];
        renumber.fill(usize::MAX);
        let mut next = 0;
        for u in 0..node_count {
            let c = components.get(u).index();
            if renumber[c] == usize::MAX {
                renumber[c] = next;
                next += 1;
            }
            *components.get_mut(u) = Ix::new(renumber[c]);
        }
        self.count = count;

        self.components.flush_async()?;
        for f in &helpers {
            let _ = std::fs::remove_file(f);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::components::test_utils::{graph, label, random_arcs, reachability};

    #[test]
    fn strong_components_match_brute_force() {
        // a long path: the search is iterative, so no stack overflow
        let long_path: Vec<_> = (0..5000).map(|u| (u, u + 1)).collect();
        for (name, n, arcs) in [
            ("isolated", 4, vec![(1, 1)]),
            ("path", 4, vec![(0, 1), (1, 2), (2, 3)]),
            ("cycle_and_tail", 6, vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (5, 4)]),
            ("two_way", 3, vec![(0, 1), (1, 0), (2, 1)]),
            ("random_sparse", 60, random_arcs(60, 70, 3)),
            ("random_dense", 60, random_arcs(60, 200, 5)),
            ("long_path", 5001, long_path),
        ] {
            let g = graph(name, "general", n, &arcs);
            let scc = StronglyConnectedComponents::new(&g).unwrap();
            let got: Vec<usize> = scc.components().iter().map(|c| c.index()).collect();
            let want = if n > 100 {
                (0..n).collect() // a path: every node on its own
            } else {
                let reach = reachability(n, &arcs);
                label(n, |u, v| reach[u][v] && reach[v][u])
            };
            assert_eq!(got, want, "{name}");
            assert_eq!(scc.count(), want.iter().max().map_or(0, |m| m + 1), "{name}");
        }
    }

    #[test]
    fn strong_components_of_undirected_graphs_are_connected_components() {
        let g = graph("undirected", "symmetric", 6, &[(0, 1), (1, 2), (3, 4)]);
        let scc = StronglyConnectedComponents::new(&g).unwrap();
        assert_eq!(scc.components(), &[0, 0, 0, 1, 1, 2]);
        assert_eq!(scc.sizes(), vec![3, 2, 1]);
    }
}
