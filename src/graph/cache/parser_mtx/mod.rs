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
    /// Input file is assumed have file extension .mtx and must be a sparse (`coordinate`) square
    /// matrix, of `pattern`, `integer`, `real`/`double` or `complex` entries, `general`,
    /// `symmetric`, `skew-symmetric` or `hermitian` (see [`MtxHeader`]). Values are only read into
    /// integer edge labels; unlabelled graphs ignore them.
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

        let node_count = Self::parse_mtx_header(path.as_ref())?.nodes;

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

        let neighbors = AbstractedProceduralMemoryMut::<Ix>::from_file(&cache.neighbors_file, sum)?;
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
                        // one claim for both: other threads may be placing edges of `u` too
                        let pos = D::widen(*offsets_s.get(u))
                            + counters.get(u).fetch_add_one_widened(Ordering::Relaxed);
                        *neighbors.get_mut(pos) = Ix::new(v);
                        *edgelabels.get_mut(pos) = E::new(w);
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
        // mirrored entries make symmetric inputs symmetric: no need to check
        if Self::mtx_header_is_symmetric(path.as_ref())? {
            cache.symmetric = Some(true);
        }

        Ok(cache)
    }
}

/// The kind of values a MatrixMarket file's entries carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MtxField {
    /// None: entries are positions only.
    Pattern,
    Integer,
    /// `real` or `double`.
    Real,
    Complex,
}

impl std::fmt::Display for MtxField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MtxField::Pattern => "pattern",
            MtxField::Integer => "integer",
            MtxField::Real => "real",
            MtxField::Complex => "complex",
        })
    }
}

/// How a MatrixMarket file's entries stand for the matrix's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MtxSymmetry {
    /// Each entry is one arc.
    General,
    /// Each off-diagonal entry `(i, j)` is also `(j, i)`: with the same value...
    Symmetric,
    /// ... with the negated value...
    SkewSymmetric,
    /// ... or with the conjugated value.
    Hermitian,
}

impl MtxSymmetry {
    /// Whether each off-diagonal entry stands for an arc both ways.
    pub(super) fn mirrored(self) -> bool {
        self != MtxSymmetry::General
    }
}

/// A MatrixMarket file's header: its banner and size line.
#[derive(Clone, Copy, Debug)]
pub(super) struct MtxHeader {
    pub(super) field: MtxField,
    pub(super) symmetry: MtxSymmetry,
    /// Rows, which are as many as columns: the graph's nodes.
    pub(super) nodes: usize,
    /// Entries listed.
    pub(super) entries: usize,
    /// Where the entries start, in bytes.
    pub(super) data_offset: u64,
}

#[cfg(test)]
mod test {
    use super::GraphCache;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use crate::graph::{IndexType, label::VoidLabel};
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

    /// Weighted edges of one node, parsed by every thread at once, each keep their own target and
    /// weight (the slot of an edge's target and of its weight were once claimed separately, so
    /// that concurrent edges of a node could overwrite each other's targets).
    #[test]
    fn concurrently_parsed_weighted_edges_keep_their_targets() {
        use crate::graph::E as _;

        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(transparent)]
        struct Weight(usize);
        unsafe impl bytemuck::Zeroable for Weight {}
        unsafe impl bytemuck::Pod for Weight {}
        impl crate::graph::E for Weight {
            fn new(v: usize) -> Self {
                Weight(v)
            }
            fn is_labeled() -> bool {
                true
            }
        }
        assert!(Weight::is_labeled());

        let dir = unique_test_dir("mtx_weighted_race");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        // node 1's edges, to every other node, are spread over the whole file (and so over
        // every parsing thread's chunk), between edges of other nodes
        let n = 4000;
        let mut mtx = format!(
            "%%MatrixMarket matrix coordinate integer general\n{n} {n} {}\n",
            2 * (n - 1)
        );
        for v in 2..=n {
            mtx += &format!("1 {v} {}\n{v} 1 {}\n", 10 * v, 10 * v + 1);
        }
        let path = dir.join("hub.mtx");
        std::fs::write(&path, mtx).unwrap();

        for run in 0..5 {
            let cache = GraphCache::<VoidLabel, Weight, usize>::from_file(
                &path,
                Some(format!("mtx_weighted_race_{run}_{}", unique_run_tag())),
                None,
                None,
            )
            .unwrap();
            let graph = crate::graph::DiGraph::init_from_cache(cache, Some(1)).unwrap();
            for u in 0..graph.size() {
                for (pos, v) in graph.index_node(u).zip(graph.neighbours(u).unwrap()) {
                    // targets are 0-based, weights were written for 1-based ids
                    let expected = match (u, v) {
                        (0, v) if v > 0 => 10 * (v + 1),
                        (u, 0) if u > 0 => 10 * (u + 1) + 1,
                        _ => panic!("run {run}: no edge {u} -> {v} in the input"),
                    };
                    assert_eq!(
                        graph.edge_label(pos),
                        Weight(expected),
                        "run {run}: edge {u} -> {v}"
                    );
                }
            }
            let mut targets: Vec<usize> = graph.neighbours(0).unwrap().collect();
            targets.sort_unstable();
            assert_eq!(targets, (1..n).collect::<Vec<_>>(), "run {run}");
        }
    }

    /// A failed parse leaves no partial cache entry behind (and a missing input is named).
    #[test]
    fn failed_parses_leave_no_cache_files() {
        let dir = unique_test_dir("mtx_failed_parse");
        let _fixture = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let entry_files = |id: &str| -> Vec<std::path::PathBuf> {
            let hash = crate::graph::cache::utils::id_from_filename(id).unwrap();
            glob::glob(&format!(
                "{}*{hash}.*",
                crate::graph::cache::utils::CACHE_DIR
            ))
            .unwrap()
            .flatten()
            .collect()
        };

        let missing = dir.join("missing.mtx");
        let id = format!("mtx_missing_{}", unique_run_tag());
        let err = GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &missing,
            Some(id.clone()),
            None,
            None,
        )
        .expect_err("a missing input should fail");
        assert!(err.to_string().contains("missing.mtx"), "{err}");
        assert_eq!(entry_files(&id), Vec::<std::path::PathBuf>::new());

        // an edge to a node past the header's node count
        let malformed = dir.join("malformed.mtx");
        let mut f = std::fs::File::create(&malformed).unwrap();
        writeln!(
            f,
            "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n9 1"
        )
        .unwrap();
        drop(f);
        let id = format!("mtx_malformed_{}", unique_run_tag());
        GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
            &malformed,
            Some(id.clone()),
            None,
            None,
        )
        .expect_err("a malformed input should fail");
        assert_eq!(entry_files(&id), Vec::<std::path::PathBuf>::new());
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
        let graph_usize = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_usize,
            Some(1),
        )
        .expect("usize graph init should succeed");

        let cache_u32 = GraphCache::<VoidLabel, VoidLabel, u32>::from_file(
            &path,
            Some(format!("mtx_ix_test_u32_{run_tag}")),
            None,
            None,
        )
        .expect("u32 mtx parse should succeed");
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
        let meta_usize = graph_usize
            .metadata()
            .expect("error reading usize metadata");
        assert_eq!(meta_usize.index_size, std::mem::size_of::<usize>());

        // sanity check that algorithms operating directly on the CSR offsets/neighbors arrays
        // (not just the higher level iterator API exercised above) also agree between Ix widths
        // (BFS rather than an undirected-only algorithm: the fixture is a directed graph)
        use crate::trails::bfs::BFSDists;
        let distances_usize: Vec<usize> = BFSDists::new(&graph_usize, 0)
            .expect("usize bfs should succeed")
            .get_distances()
            .iter()
            .map(|d| d.index())
            .collect();
        let distances_u32: Vec<usize> = BFSDists::new(&graph_u32, 0)
            .expect("u32 bfs should succeed")
            .get_distances()
            .iter()
            .map(|d| d.index())
            .collect();
        assert_eq!(
            distances_usize, distances_u32,
            "bfs distances differ between usize and u32 Ix"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for the `AccumulatorAtomic`-generic degree/edge-position accumulation
    /// pipeline: parsing the same `.mtx` input with `AtomicUsize` (default), `AtomicU32`, and
    /// `AtomicU16` as the accumulator type --- with `Ix` fixed to `usize` --- must produce
    /// identical graphs (same neighbour lists and BFS distances), regardless of which width was
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
        let graph_default = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_default,
            Some(1),
        )
        .expect("default accumulator graph init should succeed");

        let cache_u32 =
            GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<_, AtomicU32>(
                &path,
                Some(format!("mtx_accumulator_test_u32_{run_tag}")),
                None,
                None,
            )
            .expect("AtomicU32 accumulator mtx parse should succeed");
        let graph_u32 = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_u32,
            Some(1),
        )
        .expect("AtomicU32 accumulator graph init should succeed");

        let cache_u16 =
            GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<_, AtomicU16>(
                &path,
                Some(format!("mtx_accumulator_test_u16_{run_tag}")),
                None,
                None,
            )
            .expect("AtomicU16 accumulator mtx parse should succeed");
        let graph_u16 = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_u16,
            Some(1),
        )
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

        // (BFS rather than an undirected-only algorithm: the fixture is a directed graph)
        use crate::trails::bfs::BFSDists;
        let distances =
            |g: &crate::graph::DiGraph<VoidLabel, VoidLabel, usize>, what: &str| -> Vec<usize> {
                BFSDists::new(g, 0)
                    .unwrap_or_else(|e| panic!("{what} accumulator bfs should succeed: {e}"))
                    .get_distances()
                    .to_vec()
            };
        let distances_default = distances(&graph_default, "default");
        assert_eq!(
            distances_default,
            distances(&graph_u32, "AtomicU32"),
            "bfs distances differ between AtomicUsize and AtomicU32 accumulators"
        );
        assert_eq!(
            distances_default,
            distances(&graph_u16, "AtomicU16"),
            "bfs distances differ between AtomicUsize and AtomicU16 accumulators"
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

        let result =
            GraphCache::<VoidLabel, VoidLabel, usize>::from_file_with_accumulator::<_, AtomicU16>(
                &path,
                Some(format!("mtx_accumulator_overflow_test_{run_tag}")),
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
        let mismatched =
            GraphCache::<VoidLabel, VoidLabel, u32>::open(cache_usize.offsets_filename(), None);
        assert!(
            mismatched.is_err(),
            "opening a cache with a mismatched Ix width should fail, not silently misinterpret data"
        );

        let original_meta = cache_usize.metadata().expect("error reading metadata");

        let cache_u32 = cache_usize
            .convert_index_type::<u32>()
            .expect("conversion to u32 should succeed");
        let converted_meta = cache_u32
            .metadata()
            .expect("error reading converted metadata");

        assert_eq!(
            converted_meta.dataset_name,
            format!(
                "{}_{}",
                original_meta.dataset_name,
                std::mem::size_of::<u32>()
            )
        );
        assert_ne!(
            converted_meta.cache_id, original_meta.cache_id,
            "converted cache must get a new cache_id"
        );
        assert_eq!(converted_meta.index_size, std::mem::size_of::<u32>());
        assert_eq!(converted_meta.nodes, original_meta.nodes);
        assert_eq!(converted_meta.edges, original_meta.edges);

        let graph_usize = crate::graph::DiGraph::<VoidLabel, VoidLabel, usize>::init_from_cache(
            cache_usize,
            Some(1),
        )
        .expect("usize graph init should succeed");
        let graph_u32 =
            crate::graph::DiGraph::<VoidLabel, VoidLabel, u32>::init_from_cache(cache_u32, Some(1))
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

    /// A file declaring another number of entries than it has fails, every time: the parser
    /// used to stop at a racy running total, so its counting and placing passes could read
    /// different entries, leaving unfilled (and, for `symmetric` files, unreciprocated) arcs.
    #[test]
    fn mtx_entry_count_must_match_the_header() {
        use crate::graph::{DiGraph, Graph};
        let dir = unique_test_dir("mtx_nnz");
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let entries: Vec<(usize, usize)> = (0..20_000)
            .map(|_| (1 + next() % 200, 1 + next() % 200))
            .collect();
        for (name, declared, kind) in [
            ("under", 3, "symmetric"),
            ("over", 30_000, "symmetric"),
            ("under_general", 3, "general"),
        ] {
            let path = dir.join(format!("{name}.mtx"));
            let mut text =
                format!("%%MatrixMarket matrix coordinate pattern {kind}\n200 200 {declared}\n");
            for (i, j) in &entries {
                text += &format!("{i} {j}\n");
            }
            std::fs::write(&path, text).unwrap();
            for trial in 0..5 {
                let id = format!("mtx_nnz_{name}_{trial}_{}", unique_run_tag());
                let err = Graph::<VoidLabel, VoidLabel, usize>::from_file(
                    &path,
                    Some(id.clone()),
                    Some(4),
                )
                .err()
                .unwrap_or_else(|| panic!("{name}, trial {trial}: parsed"));
                assert!(err.to_string().contains("declares"), "{name}: {err}");
                assert!(
                    DiGraph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(4))
                        .is_err()
                );
            }
        }
        // the files' 20 000 entries, declared right, parse
        let path = dir.join("right.mtx");
        let mut text =
            "%%MatrixMarket matrix coordinate pattern symmetric\n200 200 20000\n".to_string();
        for (i, j) in &entries {
            text += &format!("{i} {j}\n");
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("mtx_nnz_right_{}", unique_run_tag());
        let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(&path, Some(id), Some(4)).unwrap();
        assert!(g.is_symmetric().unwrap());
        assert_eq!(
            g.width(),
            2 * entries.iter().filter(|(i, j)| i != j).count()
                + entries.iter().filter(|(i, j)| i == j).count()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    struct Label(usize);
    unsafe impl bytemuck::Zeroable for Label {}
    unsafe impl bytemuck::Pod for Label {}
    impl crate::graph::E for Label {
        fn new(v: usize) -> Self {
            Label(v)
        }
        fn is_labeled() -> bool {
            true
        }
    }

    fn write(dir: &std::path::Path, name: &str, text: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{name}.mtx"));
        std::fs::write(&path, text).unwrap();
        path
    }

    fn arcs<Dir: crate::graph::Directedness>(
        g: &crate::graph::GraphMemoryMap<VoidLabel, VoidLabel, usize, Dir>,
    ) -> Vec<(usize, usize)> {
        (0..g.size())
            .flat_map(|u| g.neighbours(u).unwrap().map(move |v| (u, v)))
            .collect()
    }

    /// Real, double and complex files load their structure (values ignored) like the same
    /// pattern files; hermitian and skew-symmetric ones are mirrored like symmetric ones.
    #[test]
    fn mtx_real_complex_and_hermitian_inputs_load_their_structure() {
        use crate::graph::{DiGraph, Graph};
        let dir = unique_test_dir("mtx_fields");
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        let tag = unique_run_tag();
        // entries (1-based) and, per field, how a value is written
        let entries = [(2, 1), (3, 1), (3, 3), (4, 2)];
        let file = |field: &str, symmetry: &str, value: &dyn Fn(usize) -> String| {
            let mut text = format!(
                "%%MatrixMarket matrix coordinate {field} {symmetry}\n% a comment\n\n4 4 {}\n",
                entries.len()
            );
            for (k, (i, j)) in entries.iter().enumerate() {
                text += &format!("{i} {j}{}\n", value(k));
            }
            text
        };
        let none = |_: usize| String::new();
        let real = |k: usize| [" -1.5", " 2.0e-3", " 0", " 7"][k].to_string();
        let complex = |k: usize| [" 1.0 -2.0", " 0.5 0", " 3 0", " -1e2 4"][k].to_string();
        for (symmetry, kind) in [("general", "general"), ("symmetric", "symmetric")] {
            let reference = write(
                &dir,
                &format!("pattern_{symmetry}"),
                &file("pattern", symmetry, &none),
            );
            let reference = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
                &reference,
                Some(format!("mtx_fields_ref_{kind}_{tag}")),
                Some(2),
            )
            .unwrap();
            for (field, value) in [
                ("real", &real as &dyn Fn(usize) -> String),
                ("double", &real),
                ("complex", &complex),
                ("integer", &|k: usize| format!(" {}", k + 1)),
            ] {
                let path = write(
                    &dir,
                    &format!("{field}_{symmetry}"),
                    &file(field, symmetry, value),
                );
                let g = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
                    &path,
                    Some(format!("mtx_fields_{field}_{kind}_{tag}")),
                    Some(2),
                )
                .unwrap_or_else(|e| panic!("{field} {symmetry}: {e}"));
                assert_eq!(arcs(&g), arcs(&reference), "{field} {symmetry}");
                assert_eq!(
                    g.is_symmetric().unwrap(),
                    symmetry != "general",
                    "{field} {symmetry}"
                );
            }
        }
        // hermitian (complex) and skew-symmetric files are mirrored: undirected graphs
        let symmetric = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
            &write(&dir, "pattern_sym", &file("pattern", "symmetric", &none)),
            Some(format!("mtx_fields_sym_{tag}")),
            Some(2),
        )
        .unwrap();
        for (name, text) in [
            ("hermitian", file("complex", "hermitian", &complex)),
            ("skew", file("real", "skew-symmetric", &real)),
            (
                "skew_int",
                file("integer", "skew-symmetric", &|k: usize| {
                    format!(" {}", k + 1)
                }),
            ),
        ] {
            let path = write(&dir, name, &text);
            let g = Graph::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                Some(format!("mtx_fields_{name}_{tag}")),
                Some(2),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(arcs(&g), arcs(&symmetric), "{name}");
            assert_eq!(g.metadata().unwrap().symmetric, Some(true), "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Edge labels are non-negative integers: other values are refused, with what to do instead.
    #[test]
    fn mtx_values_labels_cannot_hold_are_refused() {
        use crate::graph::DiGraph;
        let dir = unique_test_dir("mtx_label_values");
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        let tag = unique_run_tag();
        for (name, text, expected) in [
            (
                "real",
                "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 2 0.5\n",
                "unlabelled",
            ),
            (
                "complex",
                "%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 1 1\n",
                "unlabelled",
            ),
            (
                "skew",
                "%%MatrixMarket matrix coordinate integer skew-symmetric\n2 2 1\n2 1 3\n",
                "skew-symmetric",
            ),
            (
                "negative",
                "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 2 -3\n",
                "non-negative integer",
            ),
        ] {
            let path = write(&dir, name, text);
            let id = format!("mtx_label_values_{name}_{tag}");
            let err =
                DiGraph::<VoidLabel, Label, usize>::from_file(&path, Some(id.clone()), Some(2))
                    .err()
                    .unwrap_or_else(|| panic!("{name}: loaded"));
            assert!(err.to_string().contains(expected), "{name}: {err}");
            let hashed = crate::graph::cache::utils::id_from_filename(&id).unwrap();
            let left: Vec<_> = glob::glob(&format!(
                "{}*{hashed}.*",
                crate::graph::cache::utils::CACHE_DIR
            ))
            .unwrap()
            .flatten()
            .collect();
            assert!(left.is_empty(), "{name}: left {left:?}");
            // the same files load unlabelled
            DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                Some(format!("{id}_unlabelled")),
                Some(2),
            )
            .unwrap_or_else(|e| panic!("{name}, unlabelled: {e}"));
        }
        // integer values are kept
        let path = write(
            &dir,
            "integer",
            "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 2 5\n2 1 7\n",
        );
        let g = DiGraph::<VoidLabel, Label, usize>::from_file(
            &path,
            Some(format!("mtx_label_values_int_{tag}")),
            Some(2),
        )
        .unwrap();
        assert_eq!((g.edge_label(0), g.edge_label(1)), (Label(5), Label(7)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Files that aren't sparse square matrices are refused, saying why.
    #[test]
    fn mtx_headers_are_checked() {
        use crate::graph::DiGraph;
        let dir = unique_test_dir("mtx_headers");
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = crate::graph::test_utils::CacheGuard::dir(dir.clone());
        let tag = unique_run_tag();
        for (name, text, expected) in [
            (
                "array",
                "%%MatrixMarket matrix array real general\n2 2\n1\n0\n0\n1\n",
                "coordinate",
            ),
            (
                "rectangular",
                "%%MatrixMarket matrix coordinate pattern general\n2 3 1\n1 3\n",
                "square",
            ),
            (
                "field",
                "%%MatrixMarket matrix coordinate boolean general\n2 2 1\n1 2\n",
                "unknown MatrixMarket field",
            ),
            (
                "symmetry",
                "%%MatrixMarket matrix coordinate pattern upper\n2 2 1\n1 2\n",
                "unknown MatrixMarket symmetry",
            ),
            (
                "banner",
                "%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n",
                "%%MatrixMarket",
            ),
            (
                "size",
                "%%MatrixMarket matrix coordinate pattern general\n2 two 1\n1 2\n",
                "columns",
            ),
        ] {
            let path = write(&dir, name, text);
            let err = DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
                &path,
                Some(format!("mtx_headers_{name}_{tag}")),
                Some(2),
            )
            .err()
            .unwrap_or_else(|| panic!("{name}: loaded"));
            assert!(err.to_string().contains(expected), "{name}: {err}");
        }
        // case and spacing don't matter
        let path = write(
            &dir,
            "case",
            "%%MatrixMarket  MATRIX Coordinate REAL   General\n2 2 1\n1 2 1.0\n",
        );
        DiGraph::<VoidLabel, VoidLabel, usize>::from_file(
            &path,
            Some(format!("mtx_headers_case_{tag}")),
            Some(2),
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
