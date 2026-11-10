pub mod astar;
pub mod bfs;
pub mod bfs_visit;
pub(crate) mod delta_stepping;
pub mod dfs;
pub mod dfs_visit;
pub mod dijkstra;
pub mod hierholzer;
pub mod shortest_path_visit;

#[cfg(test)]
use _verify::verify_trails;

#[cfg(test)]
mod _verify {
    use crate::{
        graph::{self, GraphMemoryMap},
        shared_slice::AbstractedProceduralMemoryMut,
    };

    #[allow(dead_code)]
    pub(super) fn verify_trails<N: graph::N, E: graph::E, Ix: graph::IndexType>(
        graph: &GraphMemoryMap<N, E, Ix>,
        euler_trails: AbstractedProceduralMemoryMut<Ix>,
        euler_index: AbstractedProceduralMemoryMut<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // trails length == graph width
        if euler_trails.len() != graph.width() {
            return Err(format!(
                "euler trail invalid: graph width is {} but trail has length {}",
                graph.width(),
                euler_trails.len()
            )
            .into());
        }

        // brief trail_index check
        if euler_index.is_empty() {
            if graph.width() == 0 {
                if euler_trails.is_empty() {
                    return Ok(());
                } else {
                    return Err(format!(
                        "euler trail invalid: graph width is 0 but trail has length {}",
                        euler_trails.len()
                    )
                    .into());
                }
            } else {
                return Err("euler trail invalid: graph width > 0 but trail has length 0".into());
            }
        } else if graph.width() == 0 {
            return Err(format!(
            "euler trail invalid: graph width is 0 but index is not empty and trail has length {}",
            euler_trails.len()
        )
        .into());
        }

        // whether each edge was used by a trail
        let mut check = vec![false; graph.width()];
        let k = graph.width();

        let trails: Vec<(usize, usize)> = euler_index
            .as_slice()
            .iter()
            .enumerate()
            .map(|(idx, end)| {
                if idx == 0 {
                    (0, end.index())
                } else {
                    (euler_index.get(idx - 1).index(), end.index())
                }
            })
            .collect();

        let edge_indexes = graph.offsets_ptr();
        let mut j = 0;
        for (begin, end) in trails {
            for curr_node in begin..end {
                let next_node = if curr_node == end - 1 {
                    begin
                } else {
                    curr_node + 1
                };
                let c_node = euler_trails.get(curr_node).index();
                let n_node = euler_trails.get(next_node).index();
                let c_neighbours = graph.neighbours(c_node)?;
                let mut found = false;
                for (idx, dest_node) in c_neighbours.enumerate() {
                    if dest_node == n_node {
                        let b_idx = unsafe { edge_indexes.add(c_node).read() }.index();
                        let e_idx = b_idx + idx;
                        if !check[e_idx] {
                            // print!(
                            //     "{e_idx} (for {c_node} between ({} {}))-> ",
                            //     unsafe { edge_indexes.add(*c_node).read() },
                            //     unsafe { edge_indexes.add(*c_node + 1).read() }
                            // );
                            // println!("{}", unsafe { graph.edges_ptr().add(e_idx).read() });
                            check[e_idx] = true;
                            found = true;
                            break;
                        }
                    }
                }
                // println!("{c_node} : {n_node}");
                // println!();
                if !found {
                    return Err(format!("invalid euler trail couldn't find unvisited edge from {c_node} to {n_node} (idx: b -> {curr_node}, e -> {next_node}, trail_end -> {end}/{j})").into());
                }
                j += 1;
            }
        }

        let mut i = 0;
        for used in check.iter() {
            if !used {
                i += 1;
            }
        }
        if i != 0 {
            return Err(format!(
                "invalid euler trail found unvisited edge at idx {i} of {} ({k} {j})",
                graph.width()
            )
            .into());
        }

        Ok(())
    }
}
