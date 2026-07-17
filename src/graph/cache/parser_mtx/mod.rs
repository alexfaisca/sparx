mod producers;

use crate::{
    graph::cache::utils::{FileType, H, cache_file_name},
    shared_slice::{AbstractedProceduralMemoryMut, SharedSliceMut},
};

use num_cpus::get_physical;
use portable_atomic::Ordering;
use std::path::Path;

use super::{AccumulatorAtomic, GraphCache};

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
    pub(super) fn from_mtx_file_impl<P: AsRef<Path>, D: AccumulatorAtomic>(
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
        let degrees = SharedSliceMut::<D>::abst_mem_mut(&d_fn, offset_size, true)?;
        let counters = SharedSliceMut::<D>::abst_mem_mut(&h_fn, node_count, true)?;

        // accumulate node degrees on index. `D::increment` is unchecked (no per-edge branch,
        // see its documentation) --- overflow is instead caught below, once, by comparing
        // `finalize_ix_offsets`'s computed total against `edges_emitted` (tallied practically
        // for free: each parsing thread accumulates its own local count and flushes it to a
        // shared atomic exactly once, when it finishes, not once per edge).
        let edges_emitted = Self::parallel_no_labels_parse_mtx_with(
            path.as_ref(),
            {
                let degrees = degrees.shared_slice().clone();
                move |u, _| {
                    degrees.get(u).increment(Ordering::Relaxed);
                }
            },
            threads,
        )?;
        // build offset vector from degrees, persisting it, Ix-width, into `cache.offsets_file`
        let (_degrees_owner, offsets_s, sum) =
            cache.finalize_ix_offsets(degrees, offset_size, Some(edges_emitted))?;

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
                        // safe, unchecked: a per-node position counter can never exceed that
                        // node's own degree, which `checked_increment` above already validated
                        // fits `D`'s width.
                        *neighbors.get_mut(
                            D::widen(*offsets_s.get(u)) + counters.get(u).load_widened(Ordering::Relaxed),
                        ) = Ix::new(v);
                        *edgelabels.get_mut(
                            D::widen(*offsets_s.get(u))
                                + counters.get(u).fetch_add_one_widened(Ordering::Relaxed),
                        ) = E::new(w);
                    }
                },
                threads,
            )?;
            Self::sort_edges_with_labels::<D>(
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
                            D::widen(*offsets_s.get(u))
                                + counters.get(u).fetch_add_one_widened(Ordering::Relaxed),
                        ) = Ix::new(v);
                    }
                },
                threads,
            )?;
            Self::sort_edges::<D>(offsets_s, neighbors.shared_slice(), threads)?;
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
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
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
        let dir = unique_test_dir("mtx_ix_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("tiny.mtx");
        write_tiny_mtx(&path);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("mtx_ix_test_usize_{run_tag}")),
            None,
            None,
        )
        .expect("usize mtx parse should succeed");
        let graph_usize =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_usize, Some(1))
                .expect("usize graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some(format!("mtx_ix_test_u32_{run_tag}")),
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

    /// Regression test for the `AccumulatorAtomic`-generic degree/edge-position accumulation
    /// pipeline: parsing the same `.mtx` input with `AtomicUsize` (default), `AtomicU32`, and
    /// `AtomicU16` as the accumulator type --- with `Ix` fixed to `usize` --- must produce
    /// identical graphs (same neighbour lists and k-core numbers), regardless of which width was
    /// used internally during construction.
    #[test]
    fn mtx_parser_round_trips_across_accumulators() {
        use portable_atomic::{AtomicU16, AtomicU32};

        let dir = unique_test_dir("mtx_accumulator_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("tiny.mtx");
        write_tiny_mtx(&path);

        let cache_default = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("mtx_accumulator_test_default_{run_tag}")),
            None,
            None,
        )
        .expect("default (AtomicUsize) accumulator mtx parse should succeed");
        let graph_default = GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_default,
            Some(1),
        )
        .expect("default accumulator graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<
            _,
            AtomicU32,
        >(
            &path,
            Some(format!("mtx_accumulator_test_u32_{run_tag}")),
            None,
            None,
        )
        .expect("AtomicU32 accumulator mtx parse should succeed");
        let graph_u32 =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_u32, Some(1))
                .expect("AtomicU32 accumulator graph init should succeed");

        let cache_u16 = GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<
            _,
            AtomicU16,
        >(
            &path,
            Some(format!("mtx_accumulator_test_u16_{run_tag}")),
            None,
            None,
        )
        .expect("AtomicU16 accumulator mtx parse should succeed");
        let graph_u16 =
            GraphMemoryMap::<VoidLabel, VoidLabel, usize>::init_from_cache(cache_u16, Some(1))
                .expect("AtomicU16 accumulator graph init should succeed");

        assert_eq!(graph_default.size(), graph_u32.size());
        assert_eq!(graph_default.size(), graph_u16.size());
        assert_eq!(graph_default.width(), graph_u32.width());
        assert_eq!(graph_default.width(), graph_u16.width());
        for u in 0..graph_default.size() {
            let mut n_default: Vec<usize> = graph_default.neighbours(u).unwrap().collect();
            let mut n_u32: Vec<usize> = graph_u32.neighbours(u).unwrap().collect();
            let mut n_u16: Vec<usize> = graph_u16.neighbours(u).unwrap().collect();
            n_default.sort();
            n_u32.sort();
            n_u16.sort();
            assert_eq!(
                n_default, n_u32,
                "node {u} neighbours differ between AtomicUsize and AtomicU32 accumulators"
            );
            assert_eq!(
                n_default, n_u16,
                "node {u} neighbours differ between AtomicUsize and AtomicU16 accumulators"
            );
        }

        use crate::k_core::batagelj_zaversnik::AlgoBatageljZaversnik;
        let k_cores_default = AlgoBatageljZaversnik::new(&graph_default)
            .expect("default accumulator k-core computation should succeed")
            .k_cores()
            .to_vec();
        let k_cores_u32 = AlgoBatageljZaversnik::new(&graph_u32)
            .expect("AtomicU32 accumulator k-core computation should succeed")
            .k_cores()
            .to_vec();
        let k_cores_u16 = AlgoBatageljZaversnik::new(&graph_u16)
            .expect("AtomicU16 accumulator k-core computation should succeed")
            .k_cores()
            .to_vec();
        assert_eq!(
            k_cores_default, k_cores_u32,
            "k-core numbers differ between AtomicUsize and AtomicU32 accumulators"
        );
        assert_eq!(
            k_cores_default, k_cores_u16,
            "k-core numbers differ between AtomicUsize and AtomicU16 accumulators"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A single node's degree exceeding the chosen `AccumulatorAtomic` width (`AtomicU16` here,
    /// so `u16::MAX = 65535`) must be reported as a clean error --- never silently wrapped into
    /// corrupted offsets/neighbours. Uses a star graph (one hub node with `u16::MAX + 10`
    /// distinct neighbours) to cheaply exceed the width without a pathologically large fixture.
    #[test]
    fn mtx_parser_reports_accumulator_overflow_instead_of_corrupting() {
        use portable_atomic::AtomicU16;

        let dir = unique_test_dir("mtx_accumulator_overflow_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("star.mtx");

        let hub_degree = u16::MAX as usize + 10;
        let node_count = hub_degree + 1; // hub (node 1) + `hub_degree` leaves
        {
            let mut f = std::fs::File::create(&path).expect("error creating star mtx fixture");
            writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
            writeln!(f, "{node_count} {node_count} {hub_degree}").unwrap();
            for leaf in 2..=node_count {
                writeln!(f, "1 {leaf}").unwrap();
            }
            f.flush().unwrap();
        }

        let result = GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<
            _,
            AtomicU16,
        >(
            &path,
            Some(format!(
                "mtx_accumulator_overflow_test_{run_tag}"
            )),
            None,
            None,
        );
        assert!(
            result.is_err(),
            "parsing a hub node with degree > u16::MAX using an AtomicU16 accumulator should \
             fail loudly, not silently wrap/corrupt"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for opening a cache with a mismatched `Ix` width (must be rejected, not
    /// silently misinterpreted), and for the `convert_index_type` cache-conversion utility: the
    /// converted cache's `dataset_name` must have `_<new index size>` appended, it must be
    /// assigned a distinct `cache_id`, and it must be fully usable with the requested `Ix`.
    #[test]
    fn mtx_cache_rejects_mismatched_ix_and_converts_correctly() {
        let dir = unique_test_dir("mtx_convert_ix_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("tiny.mtx");
        write_tiny_mtx(&path);

        let cache_usize = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("mtx_convert_ix_test_{run_tag}")),
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
        let dir = unique_test_dir("mtx_overflow_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
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
            Some(format!("mtx_overflow_test_u8_{run_tag}")),
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
            Some(format!("mtx_overflow_test_u32_{run_tag}")),
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
        let dir = unique_test_dir("convert_overflow_test");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let run_tag = unique_run_tag();
        let path = dir.join("overflow.mtx");

        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        use std::io::Write;
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "300 300 1").unwrap();
        writeln!(f, "1 2").unwrap();
        f.flush().unwrap();

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some(format!("convert_overflow_test_u32_{run_tag}")),
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
