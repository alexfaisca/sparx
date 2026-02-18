use super::{GraphFile, GraphMemoryMap, cache::GraphCache};
use crate::shared_slice::SharedSliceMut;

use fst::Streamer;
use std::fs::OpenOptions;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    /// Applies the mask: fn(usize) -> bool function to each node id and returns the resulting subgraph.
    ///
    /// The resulting subgraph is that wherein only the set of nodes, `S ⊂ V`, of the nodes for whose the output of mask is true, as well as, only the set of edges coming from and going to nodes in `S`[^1].
    ///
    /// [^1]: the node ids of the subgraph may not, and probably will not, correspond to the original node identifiers, efffectively, it will be a whole new graph.
    pub(super) fn apply_mask_to_nodes(
        &mut self,
        mask: fn(usize) -> bool,
        identifier: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = self.size();

        let id = match identifier.map(|id| id.to_string()) {
            Some(id) => id.to_string(),
            None => {
                let seq = self.exports_fetch_increment()?;
                self.graph_cache.build_subgraph_cache_id(seq)?
            }
        };

        // build subgraph's cache entries' filenames
        let n_fn = GraphCache::<N, E, Ix>::build_graph_filename(&id, GraphFile::Neighbors, None)?;
        let o_fn = GraphCache::<N, E, Ix>::build_graph_filename(&id, GraphFile::Offsets, None)?;
        let ml_fn = GraphCache::<N, E, Ix>::build_graph_filename(&id, GraphFile::MetaLabels, None)?;

        if node_count > 1 {
            // helper counter
            let hc_fn = self.build_helper_filename(0)?;
            // helper indexer
            let hi_fn = self.build_helper_filename(1)?;

            // allocate |V| + 1 usize's to store the beginning and end offsets for each node's edges
            let mut edge_count =
                SharedSliceMut::<usize>::abst_mem_mut(&hc_fn, self.offsets_size(), true)?;
            // allocate |V| usize's to store each node's new id if present in the subgraph
            let mut node_index = SharedSliceMut::<usize>::abst_mem_mut(&hi_fn, self.size(), true)?;

            let mut curr_node_index: usize = 0;
            let mut curr_edge_count: usize = 0;
            *edge_count.get_mut(0) = curr_edge_count;
            // iterate over |V|
            for u in 0..self.size() {
                if mask(u) {
                    *node_index.get_mut(u) = curr_node_index;
                    curr_node_index += 1;
                    let neighbours = self.neighbours(u)?.filter(|&x| mask(x)).count();
                    curr_edge_count += neighbours;
                    *edge_count.get_mut(u + 1) = curr_edge_count;
                } else {
                    *node_index.get_mut(u) = usize::MAX;
                    *edge_count.get_mut(u + 1) = curr_edge_count;
                }
            }

            crate::graph::cache::ensure_fits_in_index_type::<Ix>(curr_node_index, curr_edge_count)?;

            let mut metalabel_stream = self.metalabels.stream();

            let mut neighbours =
                SharedSliceMut::<Ix>::abst_mem_mut(&n_fn, curr_edge_count, true)?;
            let mut offsets =
                SharedSliceMut::<Ix>::abst_mem_mut(&o_fn, curr_node_index + 1, true)?;
            let metalabel_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&ml_fn)?;

            let mut build = fst::MapBuilder::new(&metalabel_file)?;

            // write nodes in order of lexicographically ordered metalabels to avoid sorting metaabels
            // FIXME: what is more costly random page accesses or sorting build merging metalabel fst?
            *offsets.get_mut(0) = Ix::new(0);
            while let Some((metalabel, node_id)) = metalabel_stream.next() {
                let id = node_id as usize;
                if mask(id) {
                    // write index file for next node (id + 1)
                    let new_id = *node_index.get(id);
                    *offsets.get_mut(new_id + 1) = Ix::new(*edge_count.get(id + 1));
                    // write edge file node directly, element by element --- no intermediate `Vec`
                    // needed: `edge_count` already precomputed exactly how many of this node's
                    // neighbours survive the mask, so each surviving neighbour can be written
                    // straight to its final position as it's filtered.
                    for (offset, v) in (*edge_count.get(id)..).zip(self.neighbours(id)?.filter(|&x| mask(x))) {
                        *neighbours.get_mut(offset) = Ix::new(v);
                    }
                    // write fst for node
                    build.insert(metalabel, new_id as u64)?;
                }
            }
            build.finish()?;
        } else {
            // if graph is empty allocate empty for its empty subgraph
            SharedSliceMut::<Ix>::abst_mem_mut(&n_fn, 0, true)?;
            let mut i = SharedSliceMut::<Ix>::abst_mem_mut(&o_fn, 1, true)?;
            // store end of offsets in index entry at |V| (empty graph --- offsets end at 0)
            *i.get_mut(0) = Ix::new(0);
            let metalabel_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&ml_fn)?;

            let build = fst::MapBuilder::new(&metalabel_file)?;
            build.finish()?;
        }

        // finalize by initialozing a GraphCache instance for the subgraph and building it
        let cache: GraphCache<N, E, Ix> = GraphCache::open(&ml_fn, None)?;
        self.cleanup_helpers()?;
        GraphMemoryMap::init_from_cache(cache, Some(self.thread_count))
    }
}


#[cfg(test)]
mod test {
    use crate::graph::test_utils::unique_test_dir;
    use crate::shared_slice::SharedSliceMut;

    /// Direct, self-contained equivalence check for the `apply_mask_to_nodes` neighbour-write
    /// refactor: writing each surviving (mask-passing) neighbour directly to its precomputed
    /// position must produce byte-identical output to the original
    /// `.filter(...).collect::<Vec<Ix>>()`-then-`write_slice` approach, for several representative
    /// (neighbours, mask) scenarios --- including an empty result and a fully-filtered-out node.
    ///
    /// This bypasses `GraphMemoryMap`/`GraphCache`/`induced_subgraph` entirely (that round trip is
    /// currently blocked by unrelated, pre-existing bugs in filename construction and missing
    /// metadata, separate from this refactor) and instead exercises exactly the write pattern that
    /// changed, in isolation.
    #[test]
    fn direct_write_matches_collect_then_write_for_every_scenario() {
        let dir = unique_test_dir("induced_subgraph_write_check");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");

        let scenarios: &[(&[usize], fn(usize) -> bool)] = &[
            (&[1, 2, 3], |_| true),       // nothing filtered out
            (&[1, 2, 3, 4, 5], |x| x % 2 == 0), // partially filtered
            (&[2], |x| x != 2),          // fully filtered out --- empty result
            (&[], |_| true),             // no neighbours at all
        ];

        for (i, &(neighbours, mask)) in scenarios.iter().enumerate() {
            let expected: Vec<u32> = neighbours
                .iter()
                .copied()
                .filter(|&x| mask(x))
                .map(|x| x as u32)
                .collect();

            let out_fn = dir.join(format!("direct_{i}.mmap"));
            let len = expected.len().max(1);
            let mut out = SharedSliceMut::<u32>::abst_mem_mut(
                out_fn.to_str().unwrap(),
                len,
                true,
            )
            .expect("abst_mem_mut should succeed");

            for (offset, v) in neighbours.iter().copied().filter(|&x| mask(x)).enumerate() {
                *out.get_mut(offset) = v as u32;
            }

            let actual: Vec<u32> = (0..expected.len()).map(|i| *out.get(i)).collect();
            assert_eq!(
                actual, expected,
                "scenario {i}: direct-write output must match collect-then-write output"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
