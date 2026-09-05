pub mod batagelj_zaversnik;
pub mod liu_et_al;

#[cfg(test)]
use _verify::verify_k_cores;

#[cfg(test)]
mod _verify {
    use crate::{
        graph::{self, GraphMemoryMap},
        shared_slice::AbstractedProceduralMemoryMut,
    };

    use std::collections::BTreeSet;

    /// Each node's coreness by a simple sequential reference: repeatedly take a node of minimum
    /// (current) degree, whose coreness is that degree, and decrement its neighbours' degrees
    /// (once per neighbour list entry, i.e. per parallel edge) where higher. A node's degree is
    /// its neighbour list's length, self-loops and parallel edges included --- the convention the
    /// decompositions follow.
    pub(super) fn reference_coreness<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        graph: &GraphMemoryMap<N, E, Ix>,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let node_count = graph.size();
        let mut degree: Vec<usize> = (0..node_count)
            .map(|v| graph.neighbours(v).map(|n| n.count()))
            .collect::<Result<_, _>>()?;
        let mut queue: BTreeSet<(usize, usize)> =
            (0..node_count).map(|v| (degree[v], v)).collect();
        let mut coreness = vec![0; node_count];
        while let Some((deg_v, v)) = queue.pop_first() {
            coreness[v] = deg_v;
            for u in graph.neighbours(v)? {
                if degree[u] > deg_v && queue.remove(&(degree[u], u)) {
                    degree[u] -= 1;
                    queue.insert((degree[u], u));
                }
            }
        }
        Ok(coreness)
    }

    /// Verifies `node_coreness` (one value per node) is `graph`'s k-core decomposition.
    #[allow(dead_code)]
    pub(super) fn verify_k_cores<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        graph: &GraphMemoryMap<N, E, Ix>,
        node_coreness: AbstractedProceduralMemoryMut<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let coreness = node_coreness.as_slice();
        if coreness.len() != graph.size() {
            return Err(format!(
                "k-cores invalid: graph has {} nodes but coreness has length {}",
                graph.size(),
                coreness.len()
            )
            .into());
        }
        let expected = reference_coreness(graph)?;
        match (0..graph.size()).find(|&v| coreness[v] as usize != expected[v]) {
            Some(v) => Err(format!(
                "k-cores invalid: node {v} has coreness {}, expected {}",
                coreness[v], expected[v]
            )
            .into()),
            None => Ok(()),
        }
    }
}
