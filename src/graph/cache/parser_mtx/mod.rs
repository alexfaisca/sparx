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

        let d_fn = cache_file_name(&cache.offsets_filename(), &FileType::Helper(H::H), Some(1))?;
        let degrees = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&d_fn, offset_size, true)?;
        let counters = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&h_fn, node_count, true)?;

        // accumulate node degrees on index
        {
            Self::parallel_no_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let degrees = degrees.shared_slice().clone();
                    move |u, _| {
                        degrees.get(u).add(1, Ordering::Relaxed);
                    }
                },
                threads,
            )?;
        }
        // build offset vector from degrees, persisting it, Ix-width, into `cache.offsets_file`
        let (_degrees_owner, offsets_s, sum) = cache.finalize_ix_offsets(degrees, offset_size)?;

        let neighbors =
            AbstractedProceduralMemoryMut::<Ix>::from_file(&cache.neighbors_file, sum)?;
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
                            Ix::new(v);
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
                        ) = Ix::new(v);
                    }
                },
                threads,
            )?;
            Self::sort_edges(offsets_s, neighbors.shared_slice(), threads)?;
        }

        drop(_degrees_owner);
        std::fs::remove_file(&h_fn)?;
        std::fs::remove_file(&d_fn)?;

        cache.merge_fsts(&[])?;
        Ok(cache)
    }
}

type MTXHeader = (usize, usize, usize);

#[cfg(test)]
mod test {
    use super::GraphCache;
    use crate::graph::{GraphMemoryMap, label::VoidLabel};
    use std::io::Write;

    fn write_tiny_mtx(path: &std::path::Path) {
        let mut f = std::fs::File::create(path).expect("error creating tiny mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "4 4 6").unwrap();
        writeln!(f, "1 2").unwrap();
        writeln!(f, "2 3").unwrap();
        writeln!(f, "3 4").unwrap();
        writeln!(f, "4 1").unwrap();
        writeln!(f, "1 3").unwrap();
        writeln!(f, "2 4").unwrap();
        f.flush().unwrap();
    }

    /// Regression test for the `Ix`-generic cache pipeline: parsing the same `.mtx` input with
    /// both `usize` and `u32` as the index type must produce identical graphs (same node/edge
    /// counts and neighbour lists), and the resulting cache's metadata must faithfully reflect
    /// the narrower index width.
    #[test]
    fn mtx_parser_round_trips_for_usize_and_u32_ix() {
        let dir = std::env::temp_dir().join(format!(
            "sparx_mtx_ix_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("tiny.mtx");
        write_tiny_mtx(&path);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some("mtx_ix_test_usize".to_string()),
            None,
            None,
        )
        .expect("usize mtx parse should succeed");
        let graph_usize =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_usize, Some(1))
                .expect("usize graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some("mtx_ix_test_u32".to_string()),
            None,
            None,
        )
        .expect("u32 mtx parse should succeed");
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
        let meta_usize = graph_usize
            .metadata()
            .expect("error reading usize metadata");
        assert_eq!(meta_usize.index_size, std::mem::size_of::<usize>());

        // sanity check that algorithms operating directly on the CSR offsets/neighbors arrays
        // (not just the higher level iterator API exercised above) also agree between Ix widths
        use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
        let k_cores_usize = AlgoBatageljZaversnik::new(&graph_usize)
            .expect("usize k-core computation should succeed")
            .k_cores()
            .to_vec();
        let k_cores_u32 = AlgoBatageljZaversnik::new(&graph_u32)
            .expect("u32 k-core computation should succeed")
            .k_cores()
            .to_vec();
        assert_eq!(
            k_cores_usize, k_cores_u32,
            "k-core numbers differ between usize and u32 Ix"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for opening a cache with a mismatched `Ix` width (must be rejected, not
    /// silently misinterpreted), and for the `convert_index_type` cache-conversion utility: the
    /// converted cache's `dataset_name` must have `_<new index size>` appended, it must be
    /// assigned a distinct `cache_id`, and it must be fully usable with the requested `Ix`.
    #[test]
    fn mtx_cache_rejects_mismatched_ix_and_converts_correctly() {
        let dir = std::env::temp_dir().join(format!(
            "sparx_mtx_convert_ix_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("tiny.mtx");
        write_tiny_mtx(&path);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some("mtx_convert_ix_test".to_string()),
            None,
            None,
        )
        .expect("usize mtx parse should succeed");

        // opening the very same cache file with a mismatched Ix must be rejected
        let mismatched = GraphCache::<VoidLabel, VoidLabel, u32>::open(
            cache_usize.offsets_filename(),
            None,
        );
        assert!(
            mismatched.is_err(),
            "opening a cache with a mismatched Ix width should fail, not silently misinterpret data"
        );

        let original_meta = cache_usize.metadata().expect("error reading metadata");

        let cache_u32 = cache_usize
            .convert_index_type::<u32>()
            .expect("conversion to u32 should succeed");
        let converted_meta = cache_u32.metadata().expect("error reading converted metadata");

        assert_eq!(
            converted_meta.dataset_name,
            format!("{}_{}", original_meta.dataset_name, std::mem::size_of::<u32>())
        );
        assert_ne!(
            converted_meta.cache_id, original_meta.cache_id,
            "converted cache must get a new cache_id"
        );
        assert_eq!(converted_meta.index_size, std::mem::size_of::<u32>());
        assert_eq!(converted_meta.nodes, original_meta.nodes);
        assert_eq!(converted_meta.edges, original_meta.edges);

        let graph_usize =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_usize, Some(1))
                .expect("usize graph init should succeed");
        let graph_u32 =
            GraphMemoryMap::<VoidLabel, VoidLabel, u32>::init_from_cache(cache_u32, Some(1))
                .expect("converted u32 graph init should succeed");

        assert_eq!(graph_usize.size(), graph_u32.size());
        assert_eq!(graph_usize.width(), graph_u32.width());
        for u in 0..graph_usize.size() {
            let mut n_usize: Vec<usize> = graph_usize.neighbours(u).unwrap().collect();
            let mut n_u32: Vec<usize> = graph_u32.neighbours(u).unwrap().collect();
            n_usize.sort();
            n_u32.sort();
            assert_eq!(
                n_usize, n_u32,
                "node {u} neighbours differ after conversion"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for the `Ix`-overflow safety guard: attempting to build (or convert to) a
    /// cache with a `u8` index type on a graph with more than 256 nodes must fail loudly with a
    /// clear error, rather than silently truncating node/edge ids (`Ix::new` is a truncating
    /// numeric cast, so without this guard a 300-node graph would wrap node id 256 down to 0,
    /// corrupting the CSR representation without any indication).
    #[test]
    fn mtx_parser_rejects_graph_too_large_for_u8_ix() {
        let dir = std::env::temp_dir().join(format!(
            "sparx_mtx_overflow_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("overflow.mtx");

        // 300 nodes, far beyond u8::MAX (255); a single edge is enough to trigger the node-count
        // check regardless of edge-count considerations.
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        use std::io::Write;
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "300 300 1").unwrap();
        writeln!(f, "1 2").unwrap();
        f.flush().unwrap();

        let result = GraphCache::<VoidLabel, VoidLabel, u8>::from_file(
            &path,
            Some("mtx_overflow_test_u8".to_string()),
            None,
            None,
        );
        assert!(
            result.is_err(),
            "building a 300-node graph with a u8 Ix must fail, not silently truncate"
        );

        // the same graph must build fine with a wide-enough Ix
        let ok = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some("mtx_overflow_test_u32".to_string()),
            None,
            None,
        );
        assert!(ok.is_ok(), "300 nodes should fit comfortably in u32");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Companion to [`mtx_parser_rejects_graph_too_large_for_u8_ix`]: the same overflow guard
    /// must also apply when *narrowing* an existing cache via `convert_index_type`, not just when
    /// building a cache from scratch.
    #[test]
    fn convert_index_type_rejects_graph_too_large_for_target_ix() {
        let dir = std::env::temp_dir().join(format!(
            "sparx_convert_overflow_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join("overflow.mtx");

        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        use std::io::Write;
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "300 300 1").unwrap();
        writeln!(f, "1 2").unwrap();
        f.flush().unwrap();

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some("convert_overflow_test_u32".to_string()),
            None,
            None,
        )
        .expect("300 nodes should fit comfortably in u32");

        let result = cache_u32.convert_index_type::<u8>();
        assert!(
            result.is_err(),
            "converting a 300-node graph down to a u8 Ix must fail, not silently truncate"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
