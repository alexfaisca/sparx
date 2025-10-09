use super::{
    GraphCache, MultithreadedParserIndexBounds,
    utils::{FileType, H, cache_file_name},
};
use crate::shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut};

use crossbeam::thread;
use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};
use std::{path::Path, sync::Arc};

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
    pub(super) fn from_node_edge_file_impl<P: AsRef<Path>>(
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
        cache.parallel_parse_node_edge(&input_nodes[..], &input_edges[..], in_fst)?;

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
    fn parallel_parse_node_edge(
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

        let offsets = AbstractedProceduralMemoryMut::<AtomicUsize>::from_file(
            &self.offsets_file,
            offsets_size,
        )?;

        // get node degrees
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = edges_bounds[tid];
                let input = &input_edges[thread_bounds.0..thread_bounds.1];
                let offsets = offsets.shared_slice();

                handles.push(s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
                            offsets.get(id).add(1, Ordering::Relaxed);
                        }
                        Ok(())
                    },
                ));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!(
                            "error getting ggcat input file's node degrees (thread {tid}): {:?}",
                            e
                        )
                        .into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        let mut offsets_s = unsafe {
            offsets.shared_slice().cast::<usize>().ok_or_else(
                || -> Box<dyn std::error::Error> { "error getting non atomic slice".into() },
            )?
        };

        // prefix sum degrees to get offsets
        let mut sum: usize = 0;
        for u in 0..node_count {
            let degree: usize = *offsets_s.get(u);
            *offsets_s.get_mut(u) = sum;
            sum += degree;
        }
        *offsets_s.get_mut(node_count) = sum;
        offsets.flush()?;

        let neighbors =
            AbstractedProceduralMemoryMut::<usize>::from_file(&self.neighbors_file, sum)?;
        let _batch_num = Arc::new(AtomicUsize::new(0));
        let _batch_size = self.batch.map_or(Self::DEFAULT_BATCHING_SIZE, |s| s);

        let counters_fn =
            cache_file_name(&self.offsets_filename, &FileType::Helper(H::H), Some(0))?;
        let counters = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&counters_fn, node_count, true)?;
        // write edges
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
                        let mut input_lines = input.split(|&b| b == b'\n');

                        // parse first node's edges: guarantee edges are stored in ascending order even
                        // when a node's parsing is split between threads
                        let mut first_node = None;
                        let mut queue = Vec::with_capacity(256);
                        for line in &mut input_lines {
                            if line.is_empty() {
                                continue;
                            }
                            let line_str = std::str::from_utf8(line)?;

                            let node = line_str.split_whitespace().collect::<Vec<&str>>();
                            let mut node = node.iter().peekable();

                            let orig_id: usize = node.next().unwrap().parse()?;
                            let dest_id: usize = node.next().unwrap().parse()?;
                            let dir = node.next().unwrap();
                            let dir = Self::parse_node_edge_direction(dir).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error in thread {tid}: {e}").into()
                                },
                            )?;
                            if let Some(id) = first_node {
                                if orig_id == id {
                                    queue.push((dest_id, dir));
                                } else {
                                    // *edges.get_mut(
                                    //     offsets.get(orig_id).load(Ordering::Relaxed)
                                    //         + counters.get(orig_id).fetch_add(1, Ordering::Relaxed),
                                    // ) = Edge::new(dest_id as u64, dir);
                                    *neighbors.get_mut(
                                        *offsets.get(orig_id)
                                            + counters.get(orig_id).fetch_add(1, Ordering::Relaxed),
                                    ) = dest_id;
                                    break;
                                }
                            } else {
                                first_node = Some(orig_id);
                                queue.push((dest_id, dir));
                            }
                        }
                        // now write edges of first node taking into account the missing ones
                        if let Some(id) = first_node {
                            let _begin = *offsets.get(id + 1) - queue.len();
                            for (offset, (dest, _dir)) in (_begin..*offsets.get(id + 1)).zip(queue)
                            {
                                // *edges.get_mut(offset) = Edge::new(dest as u64, dir);
                                *neighbors.get_mut(offset) = dest;
                            }
                        }
                        // parse rest of the thread's edges
                        for line in input_lines {
                            if line.is_empty() {
                                continue;
                            }

                            // convert each line to str temporarily
                            let line_str = std::str::from_utf8(line)?;

                            let node = line_str.split_whitespace().collect::<Vec<&str>>();
                            let mut node = node.iter().peekable();

                            let orig_id: usize = node.next().unwrap().parse()?;
                            let dest_id: usize = node.next().unwrap().parse()?;
                            let _dir = node.next().unwrap();

                            *neighbors.get_mut(
                                *offsets.get(orig_id)
                                    + counters.get(orig_id).fetch_add(1, Ordering::Relaxed),
                            ) = dest_id;

                            continue;
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

        Self::sort_edges(offsets_s, neighbors.shared_slice(), threads)?;
        neighbors.flush_async()?;
        drop(offsets);
        drop(neighbors);

        std::fs::remove_file(&counters_fn)?;

        self.merge_fsts(&[])?;

        // Cleanup temp batch files (not necessary because that is done in method finish())
        // for batch_file in batches {
        //     let _ = std::fs::remove_file(batch_file);
        // }
        self.graph_bytes = sum;
        self.index_bytes = offsets_size;

        Ok(())
    }
}
