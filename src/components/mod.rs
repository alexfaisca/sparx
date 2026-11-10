//! Connected components: [weakly](weak::ConnectedComponents) connected components (the
//! connected components of undirected graphs) and [strongly](strong::StronglyConnectedComponents)
//! connected components (the same, for undirected graphs).
//!
//! Both number components `0..count()` by their smallest node: component `0` is node `0`'s,
//! component `1` that of the smallest node outside it, and so on.
pub mod biconnected;
pub mod strong;
pub mod weak;

use crate::graph;
use crate::shared_slice::*;
use std::path::Path;

/// The number of nodes of each of `count` components, in a memmapped file (`file`): reused if
/// it holds `count` sizes, counted from `components` otherwise.
///
/// Sizes are `usize`s rather than `Ix`s: a component may have as many nodes as the graph, one
/// more than the largest node id.
pub(super) fn component_sizes<Ix: graph::IndexType>(
    components: &[Ix],
    count: usize,
    file: &str,
    reuse: bool,
) -> Result<AbstractedProceduralMemoryMut<usize>, Box<dyn std::error::Error>> {
    if reuse
        && Path::new(file).exists()
        && let Ok(sizes) = AbstractedProceduralMemoryMut::<usize>::from_file_name(file)
        && sizes.as_slice().len() == count
    {
        return Ok(sizes);
    }
    let mut sizes = SharedSliceMut::<usize>::abst_mem_mut(file, count, true)?;
    let s = sizes.as_mut_slice();
    for c in components {
        s[c.index()] += 1;
    }
    sizes.flush_async()?;
    Ok(sizes)
}

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};
    use crate::graph::{DiGraph, Directed};
    use std::io::Write;

    /// A graph loaded as a [`DiGraph`] (which any graph can be: see `to_undirected` for the
    /// undirected graph of a symmetric one).
    pub(crate) fn graph(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
    ) -> TestGraph<(), (), usize, Directed> {
        graph_with_threads(name, header, n, edges, 3)
    }

    pub(crate) fn graph_with_threads(
        name: &str,
        header: &str,
        n: usize,
        edges: &[(usize, usize)],
        threads: u8,
    ) -> TestGraph<(), (), usize, Directed> {
        let dir = unique_test_dir("components_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern {header}").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let id = format!("components_{name}_{}", unique_run_tag());
        let graph =
            DiGraph::from_file(&path, Some(id), Some(threads)).expect("parse should succeed");
        TestGraph::new(graph, Some(dir))
    }

    /// Numbers the classes of `same` (an equivalence relation) by their smallest node.
    pub(crate) fn label(n: usize, same: impl Fn(usize, usize) -> bool) -> Vec<usize> {
        let mut labels = vec![usize::MAX; n];
        let mut count = 0;
        for u in 0..n {
            if labels[u] == usize::MAX {
                for (v, label) in labels.iter_mut().enumerate().skip(u) {
                    if *label == usize::MAX && same(u, v) {
                        *label = count;
                    }
                }
                count += 1;
            }
        }
        labels
    }

    /// `reach[u][v]`: whether `v` is reachable from `u` along `arcs`.
    pub(crate) fn reachability(n: usize, arcs: &[(usize, usize)]) -> Vec<Vec<bool>> {
        let mut reach = vec![vec![false; n]; n];
        for (u, row) in reach.iter_mut().enumerate() {
            row[u] = true;
        }
        for &(u, v) in arcs {
            reach[u][v] = true;
        }
        for k in 0..n {
            // row k doesn't change while going through k
            let via_k = reach[k].clone();
            for row in reach.iter_mut() {
                if row[k] {
                    for (r, &v) in row.iter_mut().zip(&via_k) {
                        *r |= v;
                    }
                }
            }
        }
        reach
    }

    /// Random arcs over `n` nodes, with self-loops and parallel arcs.
    pub(crate) fn random_arcs(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        (0..m).map(|_| (next() % n, next() % n)).collect()
    }
}
