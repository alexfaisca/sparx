use super::{
    AccumulatorAtomic, GraphCache, MultithreadedParserIndexBounds,
    utils::{FileType, H, cache_file_name},
};
use crate::shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut};

use crossbeam::thread;
use num_cpus::get_physical;
use portable_atomic::Ordering;
use std::path::Path;

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: super::IndexType> GraphCache<N, E, Ix> {
    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed to have file extension .lz4, if provided in compressed form using LZ4, or .txt, if provided in plaintext form. Furthermore, the file contents must follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `path` --- input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch`--- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to **NOT** storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub(super) fn from_node_edge_file_impl<P: AsRef<Path>, D: AccumulatorAtomic>(
        nodes_path: P,
        edges_path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;

        let nodes_path_str =
            nodes_path
                .as_ref()
                .to_str()
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error getting nodes path str from {:?}",
                        nodes_path.as_ref()
                    )
                    .into()
                })?;
        let edges_path_str =
            edges_path
                .as_ref()
                .to_str()
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error getting edges path str from {:?}",
                        edges_path.as_ref()
                    )
                    .into()
                })?;

        // parse optional inputs && fallback to defaults for the Nones found
        let id = id.unwrap_or(nodes_path_str.to_string());
        let batching = Some(batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b));
        let in_fst = in_fst.unwrap_or(|_id: usize| -> bool { false });

        let (input_nodes, nodes_tmp_path) = Self::read_input_file(nodes_path_str)?;
        let (input_edges, edges_tmp_path) = Self::read_input_file(edges_path_str)?;

        // init cache
        let mut cache = Self::init_with_id(&id, batching)?;

        // parse and cache input
        cache.parallel_parse_node_edge::<D>(&input_nodes[..], &input_edges[..], in_fst)?;

        // if a tmp file was created delete it
        if let Some(p) = nodes_tmp_path {
            std::fs::remove_file(p)?;
        }
        if let Some(p) = edges_tmp_path {
            std::fs::remove_file(p)?;
        }

        // make cache readonly (for now only serves to allow clone() on instances)
        cache.finish(nodes_path.as_ref())?;

        Ok(cache)
    }

    fn parse_node_edge_direction(dir: &str) -> Result<u64, Box<dyn std::error::Error>> {
        // List upper/lower case variatons out to avoid allocating &str's.
        match dir {
            "FF" => Ok(0u64),
            "FR" => Ok(1u64),
            "RF" => Ok(2u64),
            "RR" => Ok(3u64),
            "fF" => Ok(0u64),
            "fR" => Ok(1u64),
            "rF" => Ok(2u64),
            "rR" => Ok(3u64),
            "Ff" => Ok(0u64),
            "Fr" => Ok(1u64),
            "Rf" => Ok(2u64),
            "Rr" => Ok(3u64),
            "ff" => Ok(0u64),
            "fr" => Ok(1u64),
            "rf" => Ok(2u64),
            "rr" => Ok(3u64),
            _ => Err(format!("error ubknown edge direction annotation (supported are 'FF', 'FR', 'RF', 'RR' & their respective upper/lower case variatons): EdgeAnnotation {{dir: '{dir}'}}").into()),
        }
    }

    fn parse_node_edge_thread_bounds(
        input: &[u8],
        threads: usize,
    ) -> Result<MultithreadedParserIndexBounds, Box<dyn std::error::Error>> {
        let thread_load = input.len().div_ceil(threads);

        // figure out thread bounds
        let mut bounds = vec![(0usize, 0usize); threads];
        let mut previous_end = 0usize;

        (0..threads).try_for_each(|tid| -> Result<(), Box<dyn std::error::Error>> {
            let mut begin = previous_end;
            if begin != input.len() && begin != 0 && input[begin] != b'\n' {
                return Err(format!("error getting threads bounds for thread {tid}: input[{{thread_begin: {begin}}}] = {} (should equal {})", stringify!(input[begin]), stringify!('\n')).into());
            }
            if begin < input.len() - 1 && input[begin] == b'\n' {
                begin += 1
            }
            let mut end = std::cmp::min((tid + 1) * thread_load, input.len());

            // find beginning of next node entry after end of slice (marked by '\n')
            while end < input.len() && input[end] != b'\n' {
                end += 1;
            }
            previous_end = end;

            if end < input.len() - 1 && input[end] == b'\n'  {
                end += 1;
            }
            bounds[tid] = (begin, end);
            Ok(())
        })?;

        Ok(bounds.into_boxed_slice())
    }

    fn parse_node_edge_max_node_id(
        input: &[u8],
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let mut begin = input.len().saturating_sub(2); // skip last b'\n'
        let mut last_space = begin;

        while begin > 0 && input[begin] != b'\n' {
            if input[begin].is_ascii_whitespace() {
                last_space = begin;
            }
            begin -= 1;
        }

        // only possible if input.len() == 0 or something went wrong
        // in either case, file is considered empty
        if begin == input.len() || input[begin] != b'\n' && begin != 0 {
            return Ok(None);
        }
        if input[begin] != b'\n' && begin == 0 {
        } else {
            begin += 1;
        }

        let it = input[begin..last_space].iter().copied();
        let mut acc: usize = 0;
        let mut saw_digit = false;
        for b in it {
            if !b.is_ascii_digit() {
                return Err(format!("error parsing nodes input file's last node index: {b} ({}) is not a valid digit", b).into());
            }
            saw_digit = true;
            let d = (b - b'0') as usize;
            acc = acc
                .checked_mul(10)
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error parsing nodes input file's last node index: {acc} * 10 overflowed"
                    )
                    .into()
                })?;
            acc = acc
                .checked_add(d)
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error parsing nodes input file's last node index: {acc} + {d} overflowed"
                    )
                    .into()
                })?;
        }
        // get max node id
        if !saw_digit {
            return Err("error parsing nodes input file's last node index: not one valid ascii digit was found".into());
        }

        Ok(Some(acc))
    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `max_edges` - Ascribes a maximum number of edges for a node. If None is provided, defaults to `16`.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer is not to be included in the graph's metalabel fst or true, vice-versa.
    ///
    /// # Returns
    ///
    /// Empty Ok().
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * In the input slice no k-mer sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    fn parallel_parse_node_edge<D: AccumulatorAtomic>(
        &mut self,
        input_nodes: &[u8],
        input_edges: &[u8],
        _in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = (get_physical() * 2).max(1);

        let _nodes_bounds = Self::parse_node_edge_thread_bounds(input_nodes, threads)?;
        let edges_bounds = Self::parse_node_edge_thread_bounds(input_edges, threads)?;

        let max_id = match Self::parse_node_edge_max_node_id(input_nodes)? {
            Some(id) => id,
            // not one id was found --- input file is empty, so graph cache has empty files
            None => return Ok(()),
        };

        let node_count = match max_id.overflowing_add(1) {
            (_, true) => {
                return Err(format!(
                    "error getting nodes edges input file's node count: {max_id} + 1 overflowed"
                )
                .into());
            }
            (r, false) => r,
        };
        let offsets_size = match node_count.overflowing_add(1) {
            (_, true) => return Err(format!(
                "error getting nodes edges input file's offset size: {node_count} + 1 overflowed"
            )
            .into()),
            (r, false) => r,
        };

        let d_fn = cache_file_name(&self.offsets_filename, &FileType::Helper(H::H), Some(1))?;
        let degrees =
            SharedSliceMut::<D>::abst_mem_mut(&d_fn, offsets_size, true)?;

        // get node degrees. `D::increment` is unchecked (no per-edge branch, see its
        // documentation) --- overflow is instead caught below, once, by comparing
        // `finalize_ix_offsets`'s computed total against `edges_emitted` (tallied practically
        // for free: each thread counts its own lines locally and returns the count once, on
        // join, not once per edge).
        let edges_emitted = thread::scope(|s| -> Result<usize, Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = edges_bounds[tid];
                let input = &input_edges[thread_bounds.0..thread_bounds.1];
                let degrees = degrees.shared_slice();

                handles.push(s.spawn(
                    move |_| -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
                        let mut local_edges = 0usize;
                        // this assumes UTF-8 but avoids full conversion
                        for line in input.split(|&b| b == b'\n') {
                            if line.is_empty() {
                                continue;
                            }

                            // convert each line to str temporarily
                            let line_str = std::str::from_utf8(line)?;

                            let node = line_str.split_whitespace().collect::<Vec<&str>>();
                            let mut node = node.iter().peekable();

                            let id: usize = node.next().unwrap().parse()?;
                            degrees.get(id).increment(Ordering::Relaxed);
                            local_edges += 1;
                        }
                        Ok(local_edges)
                    },
                ));
            });
            // check for errors and sum each thread's local edge count
            let mut total_edges = 0usize;
            for (tid, r) in handles.into_iter().enumerate() {
                total_edges += r
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!(
                            "error getting ggcat input file's node degrees (thread {tid}): {:?}",
                            e
                        )
                        .into()
                    })?;
            }
            Ok(total_edges)
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        // prefix sum degrees to get offsets, persisting them, Ix-width, into `self.offsets_file`
        let (_degrees_owner, offsets_s, sum) =
            self.finalize_ix_offsets(degrees, offsets_size, Some(edges_emitted))?;

        let neighbors =
            AbstractedProceduralMemoryMut::<Ix>::from_file(&self.neighbors_file, sum)?;

        let counters_fn =
            cache_file_name(&self.offsets_filename, &FileType::Helper(H::H), Some(0))?;
        let counters = SharedSliceMut::<D>::abst_mem_mut(&counters_fn, node_count, true)?;
        // write edges. Every edge independently claims its own, unique output position via
        // `counters`' atomic fetch-add --- regardless of which thread/chunk it was parsed from,
        // or whether a node's edges happen to be split across a thread boundary. This used to be
        // special-cased (the first distinct node id seen by each thread was queued, then flushed
        // to a *statically computed* position range assumed to be exactly this thread's share of
        // that node's segment) to preserve each node's edges' original file order even when
        // split across threads --- but that queue-flush raced with any other thread's
        // `fetch_add`-based writes to the very same node (nothing synchronized the two
        // mechanisms), so, depending on scheduling, two edges could land on the same position
        // (silently dropping one) or the "assumed" position range could be wrong entirely for a
        // node spanning more than two threads' chunks. Confirmed as a real, reproducible data
        // race (not a test-infra artifact): simply re-running the affected round-trip test
        // in complete isolation (no other tests involved) still intermittently produced wrong
        // neighbour sets. Preserving per-node original file order was never actually load-bearing
        // here anyway: `sort_edges` (below) re-sorts every node's neighbour list immediately
        // after this, making any such ordering purely incidental.
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = edges_bounds[tid];
                let input = &input_edges[thread_bounds.0..thread_bounds.1];
                let offsets = offsets_s;
                let counters = counters.shared_slice();
                let mut neighbors = neighbors.shared_slice();

                handles.push(s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for line in input.split(|&b| b == b'\n') {
                            if line.is_empty() {
                                continue;
                            }

                            // convert each line to str temporarily
                            let line_str = std::str::from_utf8(line)?;

                            let node = line_str.split_whitespace().collect::<Vec<&str>>();
                            let mut node = node.iter().peekable();

                            let orig_id: usize = node.next().unwrap().parse()?;
                            let dest_id: usize = node.next().unwrap().parse()?;
                            let dir = node.next().unwrap();
                            Self::parse_node_edge_direction(dir).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error in thread {tid}: {e}").into()
                                },
                            )?;

                            *neighbors.get_mut(
                                D::widen(*offsets.get(orig_id))
                                    + counters.get(orig_id).fetch_add_one_widened(Ordering::Relaxed),
                            ) = Ix::new(dest_id);
                        }
                        Ok(())
                    },
                ));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                let () = r
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!(
                            "error getting node degrees from ggcat file (thread {tid}): {:?}",
                            e
                        )
                        .into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        Self::sort_edges::<D>(offsets_s, neighbors.shared_slice(), threads)?;
        neighbors.flush_async()?;
        drop(neighbors);

        drop(_degrees_owner);
        std::fs::remove_file(&counters_fn)?;
        std::fs::remove_file(&d_fn)?;

        self.merge_fsts(&[])?;

        // Cleanup temp batch files (not necessary because that is done in method finish())
        // for batch_file in batches {
        //     let _ = std::fs::remove_file(batch_file);
        // }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::GraphCache;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use crate::graph::{GraphMemoryMap, label::VoidLabel};
    use std::io::Write;

    fn write_tiny_nodes_edges(dir: &std::path::Path) -> std::path::PathBuf {
        let nodes_path = dir.join("unitig_annotated_tiny.nodes");
        let edges_path = dir.join("unitig_tiny.edges");

        let mut nodes_f = std::fs::File::create(&nodes_path).expect("error creating nodes fixture");
        for id in 0..4 {
            writeln!(nodes_f, "{id} 0 0").unwrap();
        }
        nodes_f.flush().unwrap();

        let mut edges_f = std::fs::File::create(&edges_path).expect("error creating edges fixture");
        for (orig, dest) in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)] {
            writeln!(edges_f, "{orig} {dest} ff").unwrap();
        }
        edges_f.flush().unwrap();

        nodes_path
    }

    /// Regression test for the `Ix`-generic cache pipeline: parsing the same `.nodes`/`.edges`
    /// input with both `usize` and `u32` as the index type must produce identical graphs (same
    /// node/edge counts and neighbour lists), and the resulting cache's metadata must faithfully
    /// reflect the narrower index width.
    #[test]
    fn nodes_edges_parser_round_trips_for_usize_and_u32_ix() {
        let dir = unique_test_dir("nodes_edges_ix_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let nodes_path = write_tiny_nodes_edges(&dir);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &nodes_path,
            Some(format!("nodes_edges_ix_test_usize_{run_tag}")),
            None,
            None,
        )
        .expect("usize nodes/edges parse should succeed");
        let graph_usize =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_usize, Some(1))
                .expect("usize graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &nodes_path,
            Some(format!("nodes_edges_ix_test_u32_{run_tag}")),
            None,
            None,
        )
        .expect("u32 nodes/edges parse should succeed");
        let graph_u32 =
            GraphMemoryMap::<VoidLabel, VoidLabel, u32>::init_from_cache(cache_u32, Some(1))
                .expect("u32 graph init should succeed");

        assert_eq!(graph_usize.size(), graph_u32.size());
        assert_eq!(graph_usize.width(), graph_u32.width());
        for u in 0..graph_usize.size() {
            let mut n_usize: Vec<usize> = graph_usize.neighbours(u).unwrap().collect();
            let mut n_u32: Vec<usize> = graph_u32.neighbours(u).unwrap().collect();
            n_usize.sort();
            n_u32.sort();
            assert_eq!(
                n_usize, n_u32,
                "node {u} neighbours differ between usize and u32 Ix"
            );
        }

        let meta_u32 = graph_u32.metadata().expect("error reading u32 metadata");
        assert_eq!(meta_u32.index_size, std::mem::size_of::<u32>());

        std::fs::remove_dir_all(&dir).ok();
    }
}
