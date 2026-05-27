pub mod bfs;
pub mod bfs_visit;
pub mod dfs;
pub mod dfs_visit;
pub mod hierholzer;

#[cfg(test)]
use _verify::verify_trails;

#[cfg(test)]
mod _verify {
    use crate::{
        graph::{
            self, GraphMemoryMap,
            cache::utils::{FileType, H, cache_file_name},
        },
        shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut},
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

        let edge_checker = SharedSliceMut::<bool>::abst_mem_mut(
            cache_file_name("", &FileType::Test(H::H), None)?.as_str(),
            graph.width(),
            true,
        )?;
        let mut check = edge_checker.shared_slice();
        let mut k = 0;
        for e_idx in 0..graph.width() {
            *check.get_mut(e_idx) = false;
            k += 1;
        }

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
                        if !*check.get(e_idx) {
                            // print!(
                            //     "{e_idx} (for {c_node} between ({} {}))-> ",
                            //     unsafe { edge_indexes.add(*c_node).read() },
                            //     unsafe { edge_indexes.add(*c_node + 1).read() }
                            // );
                            // println!("{}", unsafe { graph.edges_ptr().add(e_idx).read() });
                            *check.get_mut(e_idx) = true;
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
        for e_idx in 0..graph.width() {
            if !*check.get(e_idx) {
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
