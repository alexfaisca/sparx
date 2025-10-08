mod producers;

use crate::{
    graph::cache::utils::{FileType, H, cache_file_name},
    shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut},
};

use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};
use std::path::Path;

use super::GraphCache;

impl<N: super::N, E: super::E, Ix: super::IndexType> GraphCache<N, E, Ix> {
    /// Parses a [`MatrixMarket`](https://math.nist.gov/MatrixMarket/formats.html) file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed have file extension .mtx and must be of type matrix coordinate pattern/integer symmetric/skew-symmetric/general.
    ///
    /// # Arguments
    ///
    /// * `path` - input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch` --- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, as a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub(super) fn from_mtx_file_impl<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;

        // init cache
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error getting path str from {:?}", path.as_ref()).into()
            })?;
        let id = id.unwrap_or(path_str.to_string());
        let mut cache = Self::init_with_id(&id, batch)?;

        let threads = (get_physical() * 2).max(1);

        let (node_count, _, _) = Self::parse_mtx_header(path.as_ref())?;

        let h_fn = cache_file_name(&cache.offsets_filename(), &FileType::Helper(H::H), Some(0))?;
        let offset_size = match node_count.overflowing_add(1) {
            (_, true) => {
                return Err(
                    format!("error calculating offsets size: {node_count} + 1 overflowed").into(),
                );
            }
            (r, false) => r,
        };

        cache.index_bytes = offset_size;
        let offsets = AbstractedProceduralMemoryMut::<AtomicUsize>::from_file(
            &cache.offsets_file,
            offset_size,
        )?;
        let counters = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&h_fn, node_count, true)?;

        // accumulate node degrees on index
        {
            Self::parallel_no_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let offsets = offsets.shared_slice().clone();
                    move |u, _| {
                        offsets.get(u).add(1, Ordering::Relaxed);
                    }
                },
                threads,
            )?;
        }
        // build offset vector from degrees
        let mut sum = 0;
        let mut max_degree = 0;
        // this works because after aloc memmaped files are zeroed (so index[nr] = 0)
        let mut offsets_s = unsafe {
            offsets.shared_slice().cast::<usize>().ok_or_else(
                || -> Box<dyn std::error::Error> { "error getting non atomic slice".into() },
            )?
        };

        // let mut deg = vec![0; 512];
        for u in 0..offset_size {
            let deg_u = *offsets_s.get(u);
            if deg_u > max_degree {
                max_degree = deg_u;
            }
            // deg[deg_u] += 1;
            *offsets_s.get_mut(u) = sum;
            sum += deg_u;
        }

        // println!("{:?} max deg {max_degree}", deg);
        // if max_degree >= u8::MAX as usize {
        //     return Err(format!("Error graph has a max_degree of {max_degree} which, unforturnately, is bigger than {}, our current maximum supported size. If you feel a mistake has been made or really need this feature, please contact the developer team. We sincerely apologize.", u8::MAX).into());
        // }

        cache.graph_bytes = sum;
        let neighbors =
            AbstractedProceduralMemoryMut::<usize>::from_file(&cache.neighbors_file, sum)?;
        let edgelabels = AbstractedProceduralMemoryMut::<E>::from_file(&cache.edgelabel_file, sum)?;

        if E::is_labeled() {
            Self::parallel_edge_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    let counters = counters.shared_slice();
                    move |u, v, w| {
                        *neighbors
                            .get_mut(*offsets_s.get(u) + counters.get(u).load(Ordering::Relaxed)) =
                            v;
                        *edgelabels.get_mut(
                            *offsets_s.get(u) + counters.get(u).fetch_add(1, Ordering::Relaxed),
                        ) = E::new(w);
                    }
                },
                threads,
            )?;
            Self::sort_edges_with_labels(
                offsets_s,
                neighbors.shared_slice(),
                edgelabels.shared_slice(),
                threads,
            )?;
            edgelabels.flush()?;
        } else {
            Self::parallel_no_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let mut neighbors = neighbors.shared_slice();
                    let counters = counters.shared_slice();
                    move |u, v| {
                        *neighbors.get_mut(
                            *offsets_s.get(u) + counters.get(u).fetch_add(1, Ordering::Relaxed),
                        ) = v;
                    }
                },
                threads,
            )?;
            Self::sort_edges(offsets_s, neighbors.shared_slice(), threads)?;
        }

        std::fs::remove_file(&h_fn)?;

        cache.merge_fsts(&[])?;
        Ok(cache)
    }
}

type MTXHeader = (usize, usize, usize);
