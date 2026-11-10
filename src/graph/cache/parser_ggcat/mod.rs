mod producers;

use super::{
    AccumulatorAtomic, GraphCache,
    utils::{FileType, H},
};
use crate::shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut};

use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

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
    pub(super) fn from_ggcat_file_impl<P: AsRef<Path>, D: AccumulatorAtomic>(
        path: P,
        id: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;
        Self::refuse_weights("GGCAT")?;

        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error getting path str from {:?}", path.as_ref()).into()
            })?;
        // parse optional inputs && fallback to defaults for the Nones found
        let id = id.unwrap_or(path_str.to_string());
        let batching = Some(Self::DEFAULT_BATCHING_SIZE);
        let (input, tmp_path) = Self::read_input_file(path, &id, 0)?;

        // init cache
        let mut cache = Self::init_with_id(&id, batching)?;

        // parse and cache input
        cache.parallel_parse_ggcat_bytes_mmap::<D>(&input[..])?;

        // if a tmp file was created delete it
        if let Some(p) = tmp_path {
            std::fs::remove_file(p)?;
        }
        Ok(cache)
    }

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
    pub(super) fn from_ggcat_file_with_fst_impl<P: AsRef<Path>, D: AccumulatorAtomic>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;
        Self::refuse_weights("GGCAT")?;

        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error getting path str from {:?}", path.as_ref()).into()
            })?;
        // parse optional inputs && fallback to defaults for the Nones found
        let id = id.unwrap_or(path_str.to_string());
        let batching = Some(batch.unwrap_or(Self::DEFAULT_BATCHING_SIZE));
        let in_fst = in_fst.unwrap_or(|_id: usize| -> bool { false });
        let (input, tmp_path) = Self::read_input_file(path, &id, 0)?;

        // init cache
        let mut cache = Self::init_with_id(&id, batching)?;

        // parse and cache input
        cache.parallel_parse_with_fst_ggcat_bytes_mmap::<D>(&input[..], in_fst)?;

        // if a tmp file was created delete it
        if let Some(p) = tmp_path {
            std::fs::remove_file(p)?;
        }

        Ok(cache)
    }

    /// Creates a [`GraphCache`] instance from an existing cached graph.
    ///
    /// # Arguments
    ///
    /// * `filename` --- filename of one of the graph's cache files[^1].
    /// * `batch` --- size of input chunking for fst rebuild[^2][^3].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^4].
    ///
    /// [^1]: may be from any of the graph's cached files, the graph's cache id is then extracted from the filename and all necessary filename's necessary for the [`GraphCache`] instance are inferred from the cache id.
    /// [^2]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^3]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^4]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub(super) fn rebuild_fst_from_ggcat_file_impl<P: AsRef<Path>>(
        &mut self,
        path: P,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batching = batch.unwrap_or(Self::DEFAULT_BATCHING_SIZE);
        let in_fst = in_fst.unwrap_or(|_id: usize| -> bool { true });

        // self.parallel_fst_from_ggcat_with_reader(path, batching, in_fst, get_physical())?;
        let (input, tmp) = Self::read_input_file(path, &self.cache_id()?, 0)?;

        self.parallel_parse_fst_ggcat_bytes_mmap(
            &input[..],
            batching,
            in_fst,
            num_cpus::get_physical(),
        )?;

        if let Some(p) = tmp {
            std::fs::remove_file(p)?;
        }
        // self.fst_from_ggcat_bytes(input.as_slice(), batching, in_fst)?;

        Ok(())
    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
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
    fn parallel_parse_ggcat_bytes_mmap<D: AccumulatorAtomic>(
        &mut self,
        input: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = (get_physical() * 2).max(1);

        let max_id = match Self::parse_ggcat_max_node_id(input)? {
            Some(id) => id,
            // not one id was found --- input file is empty, so graph cache has empty files
            None => return Ok(()),
        };

        let node_count = match max_id.overflowing_add(1) {
            (_, true) => {
                return Err(format!(
                    "error getting ggcat input file's node count: {max_id} + 1 overflowed"
                )
                .into());
            }
            (r, false) => r,
        };
        let offsets_size = match node_count.overflowing_add(1) {
            (_, true) => {
                return Err(format!(
                    "error getting ggcat input file's offset size: {node_count} + 1 overflowed"
                )
                .into());
            }
            (r, false) => r,
        };

        let d_fn = Self::init_cache_file_from_id_or_random(
            &self.cache_id()?,
            FileType::Helper(H::H),
            Some(1),
        )?;
        let degrees = SharedSliceMut::<D>::abst_mem_mut(&d_fn, offsets_size, true)?;

        // parse node degrees. a single node's degree exceeding `D`'s width is reported as an
        // `AccumulatorOverflow`, never silently wrapped --- see `AccumulatorAtomic`'s
        // documentation.
        let overflowed = portable_atomic::AtomicBool::new(false);
        Self::parallel_edge_counter_parse_ggcat_bytes_mmap_with(
            input,
            {
                let degrees = degrees.shared_slice();
                let overflowed = &overflowed;
                move |id, edge_count| {
                    if degrees
                        .get(id)
                        .checked_store(edge_count, Ordering::Relaxed)
                        .is_err()
                    {
                        overflowed.store(true, Ordering::Relaxed);
                    }
                }
            },
            threads,
        )?;
        if overflowed.load(Ordering::Relaxed) {
            return Err(super::AccumulatorOverflow.into());
        }

        // prefix sum degrees to get node offsets, persisting them, Ix-width, into `self.offsets_file`
        let (_degrees_owner, offsets, sum) =
            self.finalize_ix_offsets(degrees, offsets_size, None)?;

        let neighbors = AbstractedProceduralMemoryMut::<Ix>::from_file(&self.neighbors_file, sum)?;
        let edgelabels = AbstractedProceduralMemoryMut::<E>::from_file(&self.edgelabel_file, sum)?;
        let nodelabels =
            AbstractedProceduralMemoryMut::<N>::from_file(&self.nodelabel_file, offsets_size)?;

        if !N::is_labeled() && !E::is_labeled() {
            Self::parallel_no_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    move |node_id, neighbors_slice| {
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node {node_id}");
                    }
                },
                threads,
            )?;
        } else if !E::is_labeled() {
            Self::parallel_node_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut nodelabels = nodelabels.shared_slice();
                    move |node_id, neighbors_slice, node_label| {
                        *nodelabels.get_mut(node_id) = node_label;
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node ");
                    }
                },
                threads,
            )?;
            nodelabels.flush_async()?;
        } else if !N::is_labeled() {
            Self::parallel_edge_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    move |node_id, neighbors_slice, edgelabels_slice| {
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                        edgelabels
                            .write_slice(D::widen(*offsets.get(node_id)), edgelabels_slice)
                            .expect("error failed to write edgelabels slice for node");
                    }
                },
                threads,
            )?;
            edgelabels.flush_async()?;
        } else {
            Self::parallel_node_edge_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut nodelabels = nodelabels.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    move |node_id, neighbors_slice, node_label, edgelabels_slice| {
                        *nodelabels.get_mut(node_id) = node_label;
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                        edgelabels
                            .write_slice(D::widen(*offsets.get(node_id)), edgelabels_slice)
                            .expect("error failed to write edgelabels slice for node");
                    }
                },
                threads,
            )?;
            nodelabels.flush_async()?;
            edgelabels.flush_async()?;
        }

        neighbors.flush_async()?;
        drop(neighbors);
        drop(_degrees_owner);
        std::fs::remove_file(&d_fn)?;

        std::fs::remove_file(&self.metalabel_filename)?;

        // build empty fst
        self.merge_fsts(&[])?;

        // Cleanup temp batch files (not necessary because that is done in method finish())
        // for batch_file in batches {
        //     let _ = std::fs::remove_file(batch_file);
        // }

        Ok(())
    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance with metalabels.
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
    fn parallel_parse_with_fst_ggcat_bytes_mmap<D: AccumulatorAtomic>(
        &mut self,
        input: &[u8],
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = (get_physical() * 2).max(1);

        let max_id = match Self::parse_ggcat_max_node_id(input)? {
            Some(id) => id,
            // not one id was found --- input file is empty, so graph cache has empty files
            None => return Ok(()),
        };

        let node_count = match max_id.overflowing_add(1) {
            (_, true) => {
                return Err(format!(
                    "error getting ggcat input file's node count: {max_id} + 1 overflowed"
                )
                .into());
            }
            (r, false) => r,
        };
        let offsets_size = match node_count.overflowing_add(1) {
            (_, true) => {
                return Err(format!(
                    "error getting ggcat input file's offset size: {node_count} + 1 overflowed"
                )
                .into());
            }
            (r, false) => r,
        };

        let d_fn = Self::init_cache_file_from_id_or_random(
            &self.cache_id()?,
            FileType::Helper(H::H),
            Some(1),
        )?;
        let degrees = SharedSliceMut::<D>::abst_mem_mut(&d_fn, offsets_size, true)?;

        // parse node degrees. a single node's degree exceeding `D`'s width is reported as an
        // `AccumulatorOverflow`, never silently wrapped --- see `AccumulatorAtomic`'s
        // documentation.
        let overflowed = portable_atomic::AtomicBool::new(false);
        Self::parallel_edge_counter_parse_ggcat_bytes_mmap_with(
            input,
            {
                let degrees = degrees.shared_slice();
                let overflowed = &overflowed;
                move |id, edge_count| {
                    if degrees
                        .get(id)
                        .checked_store(edge_count, Ordering::Relaxed)
                        .is_err()
                    {
                        overflowed.store(true, Ordering::Relaxed);
                    }
                }
            },
            threads,
        )?;
        if overflowed.load(Ordering::Relaxed) {
            return Err(super::AccumulatorOverflow.into());
        }

        // prefix sum degrees to get node offsets, persisting them, Ix-width, into `self.offsets_file`
        let (_degrees_owner, offsets, sum) =
            self.finalize_ix_offsets(degrees, offsets_size, None)?;

        let neighbors = AbstractedProceduralMemoryMut::<Ix>::from_file(&self.neighbors_file, sum)?;
        let edgelabels = AbstractedProceduralMemoryMut::<E>::from_file(&self.edgelabel_file, sum)?;
        let nodelabels =
            AbstractedProceduralMemoryMut::<N>::from_file(&self.nodelabel_file, offsets_size)?;

        let batch_num = Arc::new(AtomicUsize::new(0));
        let batch_size = self.batch.unwrap_or(Self::DEFAULT_BATCHING_SIZE);
        let max_batches = node_count.div_ceil(threads) + threads;
        let h_fn = Self::init_cache_file_from_id_or_random(
            &self.cache_id()?,
            FileType::Helper(H::H),
            Some(0),
        )?;
        let batch_path_bufs_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&h_fn)?;
        let batch_path_bufs =
            AbstractedProceduralMemoryMut::<String>::from_file(&batch_path_bufs_file, max_batches)?;
        let in_fst = { move |u| in_fst(u) };

        if !N::is_labeled() && !E::is_labeled() {
            Self::parallel_meta_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    move |node_id, neighbors_slice| {
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                    }
                },
                in_fst,
                {
                    let cache = &self;
                    let batch_num = batch_num.clone();
                    let mut batch_path_bufs = batch_path_bufs.shared_slice();
                    move |c_b| {
                        let batch = batch_num.fetch_add(1, Ordering::Relaxed);
                        let tmp_fst = cache.build_batch_fst(c_b, batch).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error building fst batch: {e}").into()
                            },
                        )?;
                        *batch_path_bufs.get_mut(batch) = tmp_fst
                            .to_str()
                            .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                                "error getting fst batch pathbuf str".into()
                            })?
                            .to_string();
                        Ok(())
                    }
                },
                batch_size,
                threads,
            )?;
        } else if !E::is_labeled() {
            Self::parallel_node_meta_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut nodelabels = nodelabels.shared_slice();
                    move |node_id, neighbors_slice, node_label| {
                        *nodelabels.get_mut(node_id) = node_label;
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                    }
                },
                in_fst,
                {
                    let cache = &self;
                    let batch_num = batch_num.clone();
                    let mut batch_path_bufs = batch_path_bufs.shared_slice();
                    move |c_b| {
                        let batch = batch_num.fetch_add(1, Ordering::Relaxed);
                        let tmp_fst = cache.build_batch_fst(c_b, batch).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error building fst batch: {e}").into()
                            },
                        )?;
                        *batch_path_bufs.get_mut(batch) = tmp_fst
                            .to_str()
                            .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                                "error getting fst batch pathbuf str".into()
                            })?
                            .to_string();
                        Ok(())
                    }
                },
                batch_size,
                threads,
            )?;
            nodelabels.flush_async()?;
        } else if !N::is_labeled() {
            Self::parallel_edge_meta_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    move |node_id, neighbors_slice, edgelabels_slice| {
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                        edgelabels
                            .write_slice(D::widen(*offsets.get(node_id)), edgelabels_slice)
                            .expect("error failed to write edge labels slice for node");
                    }
                },
                in_fst,
                {
                    let cache = &self;
                    let batch_num = batch_num.clone();
                    let mut batch_path_bufs = batch_path_bufs.shared_slice();
                    move |c_b| {
                        let batch = batch_num.fetch_add(1, Ordering::Relaxed);
                        let tmp_fst = cache.build_batch_fst(c_b, batch).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error building fst batch: {e}").into()
                            },
                        )?;
                        *batch_path_bufs.get_mut(batch) = tmp_fst
                            .to_str()
                            .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                                "error getting fst batch pathbuf str".into()
                            })?
                            .to_string();
                        Ok(())
                    }
                },
                batch_size,
                threads,
            )?;
            edgelabels.flush_async()?;
        } else {
            Self::parallel_full_labels_parse_ggcat_bytes_mmap_with(
                input,
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut nodelabels = nodelabels.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    move |node_id, neighbors_slice, node_label, edgelabels_slice| {
                        *nodelabels.get_mut(node_id) = node_label;
                        neighbors
                            .write_slice(D::widen(*offsets.get(node_id)), neighbors_slice)
                            .expect("error failed to write neigh slice for node");
                        edgelabels
                            .write_slice(D::widen(*offsets.get(node_id)), edgelabels_slice)
                            .expect("error failed to write edge labels slice for node");
                    }
                },
                in_fst,
                {
                    let cache = &self;
                    let batch_num = batch_num.clone();
                    let mut batch_path_bufs = batch_path_bufs.shared_slice();
                    move |c_b| {
                        let batch = batch_num.fetch_add(1, Ordering::Relaxed);
                        let tmp_fst = cache.build_batch_fst(c_b, batch).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error building fst batch: {e}").into()
                            },
                        )?;
                        *batch_path_bufs.get_mut(batch) = tmp_fst
                            .to_str()
                            .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                                "error getting fst batch pathbuf str".into()
                            })?
                            .to_string();
                        Ok(())
                    }
                },
                batch_size,
                threads,
            )?;
            nodelabels.flush_async()?;
            edgelabels.flush_async()?;
        }

        drop(_degrees_owner);
        std::fs::remove_file(&d_fn)?;

        // if graph cache was used to build a graph then the graph's fst holds meatlabel_filename
        // open, hence, it must be removed so that the new fst may be built
        std::fs::remove_file(&self.metalabel_filename)?;

        // merge all batch FSTs into option
        let mut tmp_fsts = Vec::new();
        batch_path_bufs
            .slice(0, batch_num.load(Ordering::Relaxed))
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error getting pathbuf strings as slice".into()
            })?
            .iter()
            .try_for_each(|s| -> Result<(), Box<dyn std::error::Error>> {
                tmp_fsts.push(PathBuf::from_str(s)?);
                Ok(())
            })?;
        self.merge_fsts(tmp_fsts.as_slice())?;

        // Cleanup temp batch files
        for batch_file in tmp_fsts {
            std::fs::remove_file(batch_file)?;
        }
        // Cleanup temp batch filenames file
        std::fs::remove_file(&h_fn)?;

        Ok(())
    }

    fn parallel_parse_fst_ggcat_bytes_mmap(
        &mut self,
        input: &[u8],
        batch_size: usize,
        in_fst: fn(usize) -> bool,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.readonly {
            return Err("error cache must be readonly to build fst in parallel".into());
        }

        let batch_num = Arc::new(AtomicUsize::new(0));
        let max_batches = self.index_bytes.saturating_sub(1).div_ceil(threads) + threads;
        let h_fn = Self::init_cache_file_from_id_or_random(
            &self.cache_id()?,
            FileType::Helper(H::H),
            Some(0),
        )?;
        let batch_path_bufs_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&h_fn)?;
        let batch_path_bufs =
            AbstractedProceduralMemoryMut::<String>::from_file(&batch_path_bufs_file, max_batches)?;
        let in_fst = { move |u| in_fst(u) };

        Self::parallel_only_meta_labels_parse_ggcat_bytes_mmap_with(
            input,
            in_fst,
            {
                let cache = &self;
                let batch_num = batch_num.clone();
                let mut batch_path_bufs = batch_path_bufs.shared_slice();
                move |c_b| {
                    let batch = batch_num.fetch_add(1, Ordering::Relaxed);
                    let tmp_fst = cache.build_batch_fst(c_b, batch).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error building fst batch: {e}").into()
                        },
                    )?;
                    *batch_path_bufs.get_mut(batch) = tmp_fst
                        .to_str()
                        .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            "error getting fst batch pathbuf str".into()
                        })?
                        .to_string();
                    Ok(())
                }
            },
            batch_size,
            threads,
        )?;

        // if graph cache was used to build a graph then the graph's fst holds meatlabel_filename
        // open, hence, it must be removed so that the new fst may be built
        std::fs::remove_file(&self.metalabel_filename)?;

        // merge all batch FSTs into option
        let mut tmp_fsts = Vec::new();
        batch_path_bufs
            .slice(0, batch_num.load(Ordering::Relaxed))
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error getting pathbuf strings as slice".into()
            })?
            .iter()
            .try_for_each(|s| -> Result<(), Box<dyn std::error::Error>> {
                tmp_fsts.push(PathBuf::from_str(s)?);
                Ok(())
            })?;
        self.merge_fsts(tmp_fsts.as_slice())?;

        // Cleanup temp batch files
        for batch_file in tmp_fsts {
            std::fs::remove_file(batch_file)?;
        }
        // Cleanup temp batch filenames file
        std::fs::remove_file(&h_fn)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::GraphCache;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use std::io::Write;

    fn write_tiny_ggcat(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating ggcat fixture");
        for (id, dest) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            writeln!(f, ">{id} 31 1 x:x:{dest}").unwrap();
            writeln!(f, "ACGT").unwrap();
        }
        f.flush().unwrap();
    }

    /// Regression test for the `Ix`-generic cache pipeline: parsing the same GGCAT-format input
    /// with both `usize` and `u32` as the index type must produce identical graphs (same
    /// node/edge counts and neighbour lists), and the resulting cache's metadata must faithfully
    /// reflect the narrower index width.
    #[test]
    fn ggcat_parser_round_trips_for_usize_and_u32_ix() {
        let dir = unique_test_dir("ggcat_ix_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("tiny.txt");
        write_tiny_ggcat(&path);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("ggcat_ix_test_usize_{run_tag}")),
            None,
            None,
        )
        .expect("usize ggcat parse should succeed");
        let graph_usize = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_usize,
            Some(1),
        )
        .expect("usize graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some(format!("ggcat_ix_test_u32_{run_tag}")),
            None,
            None,
        )
        .expect("u32 ggcat parse should succeed");
        let graph_u32 =
            crate::graph::DiGraph::<VoidLabel, VoidLabel, u32>::init_from_cache(cache_u32, Some(1))
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

    /// Concurrent parses of the same compressed input (under different cache ids) used to share
    /// the helper file its contents were decompressed into, corrupting each other: they must all
    /// succeed, with identical graphs.
    #[test]
    fn concurrent_parses_of_the_same_compressed_input() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ggcat/random_graph_9_5.lz4"
        ));
        let graphs: Vec<(usize, usize, Vec<usize>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let path = &path;
                    scope.spawn(move || {
                        let (graph, _cache) =
                            crate::graph::test_utils::open_fixture(path, Some(2)).unwrap();
                        let neighbours = (0..graph.size())
                            .flat_map(|u| graph.neighbours(u).unwrap())
                            .collect();
                        (graph.size(), graph.width(), neighbours)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(graphs[0].1 > 0);
        assert!(graphs.iter().all(|g| *g == graphs[0]));
    }
}
