use crate::{
    centralities::exact::{ExactClosenessCentrality, ExactHarmonicCentrality, ExactLinCentrality},
    graph::{
        self, GraphMemoryMap,
        cache::{
            GraphCache,
            utils::{
                CACHE_DIR, EXACT_VALUE_CACHE_DIR, FileType, H, cache_file_name_from_id,
                id_from_filename, pers_cache_file_name,
            },
        },
    },
};

use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::path::{Path, PathBuf};

type CacheRecords = OnceCell<GraphCache<(), (), usize>>;
type TestCache = DashMap<String, CacheRecords>;
type ExactValueCache = DashMap<String, OnceCell<String>>;

static TEST_CACHE: OnceCell<TestCache> = OnceCell::new();
static EXACT_VALUE_CACHE: OnceCell<ExactValueCache> = OnceCell::new();

#[cfg(test)]
thread_local! {
    /// Ids of the graph caches created on this thread, one list per live fixture guard (see
    /// [`CacheGuard::dir`]), innermost last.
    static GUARDED_CACHE_IDS: std::cell::RefCell<Vec<Vec<String>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
/// Records a graph cache's id (as in its files' names) for the innermost live fixture guard on
/// this thread, if any.
pub(crate) fn register_cache_id(id: String) {
    GUARDED_CACHE_IDS.with(|ids| {
        if let Some(innermost) = ids.borrow_mut().last_mut() {
            innermost.push(id);
        }
    });
}

#[cfg(test)]
/// Removes, when dropped (also when a test fails), cached graphs' files --- their own, those of
/// every algorithm run on them and their exact values, all named after the graph's unique id, and
/// the graphs derived from them --- and a test's fixture directory.
pub(crate) struct CacheGuard {
    ids: Vec<String>,
    fixture_dir: Option<PathBuf>,
    /// Whether this guard collects the ids of the graphs created on its thread while it lives.
    collects: bool,
}

#[cfg(test)]
impl CacheGuard {
    /// Guards `graph`'s cached files, and `fixture_dir` if given.
    pub(crate) fn new<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>(
        graph: &GraphMemoryMap<N, E, Ix, Dir>,
        fixture_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            ids: graph.graph_id().into_iter().collect(),
            fixture_dir,
            collects: false,
        }
    }

    /// Guards a test's fixture directory and every graph cache created on this thread while the
    /// guard lives (by the test, or the helpers it calls).
    pub(crate) fn dir(fixture_dir: PathBuf) -> Self {
        GUARDED_CACHE_IDS.with(|ids| ids.borrow_mut().push(Vec::new()));
        Self {
            ids: Vec::new(),
            fixture_dir: Some(fixture_dir),
            collects: true,
        }
    }
}

#[cfg(test)]
impl Drop for CacheGuard {
    fn drop(&mut self) {
        if self.collects {
            let collected = GUARDED_CACHE_IDS.with(|ids| ids.borrow_mut().pop());
            self.ids.extend(collected.into_iter().flatten());
        }
        for id in &self.ids {
            // graphs derived from it (transposes, undirected views), named after their own ids
            let _ = crate::graph::cache::drop_derived_of(id);
            // and exact values of it: named after its (unique) id, no other test reads them
            for dir in ["", EXACT_VALUE_CACHE_DIR] {
                if let Ok(files) = glob::glob(&format!("{CACHE_DIR}{dir}*{id}.*")) {
                    for file in files.flatten() {
                        let _ = std::fs::remove_file(file);
                    }
                }
            }
        }
        if let Some(dir) = &self.fixture_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

#[cfg(test)]
/// A test graph, removing its cached files and fixture directory when dropped (see
/// [`CacheGuard`]); derefs to its [`GraphMemoryMap`].
pub(crate) struct TestGraph<
    N: graph::N = (),
    E: graph::E = (),
    Ix: graph::IndexType = usize,
    Dir: graph::Directedness = graph::Undirected,
> {
    graph: GraphMemoryMap<N, E, Ix, Dir>,
    // dropped after the graph (whose files it removes)
    _guard: CacheGuard,
}

#[cfg(test)]
impl<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness>
    TestGraph<N, E, Ix, Dir>
{
    pub(crate) fn new(graph: GraphMemoryMap<N, E, Ix, Dir>, fixture_dir: Option<PathBuf>) -> Self {
        let _guard = CacheGuard::new(&graph, fixture_dir);
        Self { graph, _guard }
    }
}

#[cfg(test)]
impl<N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: graph::Directedness> std::ops::Deref
    for TestGraph<N, E, Ix, Dir>
{
    type Target = GraphMemoryMap<N, E, Ix, Dir>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

#[cfg(test)]
/// Parses a (committed, read only) fixture graph under a unique id, so that tests running
/// algorithms on the same fixture concurrently never share cached files, with a guard removing
/// those files (but not the fixture) when the test is done.
pub(crate) fn open_fixture(
    path: &Path,
    threads: Option<u8>,
) -> Result<(GraphMemoryMap, CacheGuard), Box<dyn std::error::Error>> {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let graph =
        GraphMemoryMap::from_file(path, Some(format!("{name}_{}", unique_run_tag())), threads)?;
    let guard = CacheGuard::new(&graph, None);
    Ok((graph, guard))
}

pub(crate) fn mem_cache() -> &'static TestCache {
    TEST_CACHE.get_or_init(DashMap::new)
}

pub(crate) fn exact_value_cache() -> &'static ExactValueCache {
    EXACT_VALUE_CACHE.get_or_init(DashMap::new)
}

/// Returns a short, genuinely unique tag (`"<pid>_<thread-id>_<counter>"`), suitable for use as
/// (part of) a cache id string, test-fixture directory name, or anywhere else uniqueness across
/// concurrently-running tests matters. See [`unique_test_dir`]'s doc comment for why
/// pid+thread-id alone is not sufficient.
#[allow(dead_code)]
pub(crate) fn unique_run_tag() -> String {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!(
        "{}_{:?}_{id}",
        std::process::id(),
        std::thread::current().id()
    )
}

/// Returns a genuinely unique scratch directory for a test fixture file.
///
/// `std::process::id()` + `std::thread::current().id()` alone is *not* sufficient: cargo test's
/// worker thread pool reuses OS threads across different test functions, so two different tests
/// running on the same worker thread could derive the exact same deterministic path and race on
/// (or silently reuse a stale cache for) each other's fixture file --- this was a real, confirmed
/// source of cross-test flakiness when running the suite with its default parallelism. An
/// atomic, monotonically increasing counter makes every call unique regardless of thread reuse
/// or execution order.
#[allow(dead_code)]
pub(crate) fn unique_test_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sparx_{tag}_{}", unique_run_tag()))
}

#[allow(dead_code)]
fn cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(CACHE_DIR))
}

#[allow(dead_code)]
fn exact_value_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir_str = CACHE_DIR.to_string() + EXACT_VALUE_CACHE_DIR;
    Ok(PathBuf::from(dir_str))
}

#[allow(dead_code)]
fn cache_file_for(graph_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let filename = graph_path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error getting filename for dataset path {:?}", graph_path).into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "error getting filename for dataset path as str {:?}",
                graph_path
            )
            .into()
        })?;
    Ok(cache_file_name_from_id(
        &FileType::Edges(H::H),
        &id_from_filename(filename)?,
        None,
    ))
}

#[allow(dead_code)]
fn graph_id_from_dataset_file_name(
    data_filename: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let file_name = data_filename
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- empty path".into()
        })?;

    let _parent_dir = data_filename.parent().unwrap_or_else(|| Path::new(""));
    id_from_filename(file_name)
}

#[allow(dead_code)]
pub(crate) fn get_or_init_dataset_cache_entry(
    graph_path: &Path,
) -> Result<GraphCache<(), (), usize>, Box<dyn std::error::Error>> {
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;

    // Get or create the per-key OnceLock
    let test_entry = mem_cache().entry(cache_id).or_try_insert_with(
        || -> Result<CacheRecords, Box<dyn std::error::Error>> {
            let lock = OnceCell::new();
            Ok(lock)
        },
    )?;
    let cache = test_entry.get_or_try_init(
        || -> Result<GraphCache<(), (), usize>, Box<dyn std::error::Error>> {
            let cache_filename = cache_file_for(graph_path)?;
            eprintln!("check if file with fn {cache_filename} exists");
            if Path::new(&cache_filename).exists() {
                eprintln!("yes, found it {:?}", graph_path);
                GraphCache::<(), (), usize>::open(&cache_filename, None).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!(
                            "error getting `GraphCache` instance for path {:?}: {:?}",
                            graph_path, e
                        )
                        .into()
                    },
                )
            } else {
                eprintln!("no, built it {:?}", graph_path);
                let file_name = graph_path
                    .file_name()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error invalid file path --- path not found".into()
                    })?
                    .to_str()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error invalid file path --- empty path".into()
                    })?;

                GraphCache::<(), (), usize>::from_file(
                    graph_path,
                    Some(file_name.to_string()),
                    None,
                    None,
                )
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!(
                        "error creating `GraphCache` instance for path {:?}: {:?}",
                        graph_path, e
                    )
                    .into()
                })
            }
        },
    )?;

    Ok(cache.clone())
}

#[allow(dead_code)]
pub(crate) fn get_or_init_dataset_exact_value<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    graph_path: &Path,
    graph: &GraphMemoryMap<N, E, Ix>,
    value_type: FileType,
) -> Result<String, Box<dyn std::error::Error>> {
    if value_type != FileType::ExactClosenessCentrality(H::H)
        && value_type != FileType::ExactHarmonicCentrality(H::H)
        && value_type != FileType::ExactLinCentrality(H::H)
    {
        return Err(format!(
            "error exact value files may only be of type {}, {} or {}",
            FileType::ExactClosenessCentrality(H::H),
            FileType::ExactHarmonicCentrality(H::H),
            FileType::ExactLinCentrality(H::H),
        )
        .into());
    }
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;
    let e_fn = pers_cache_file_name(&graph.cache_fst_filename(), &value_type, None)?;

    // Get or create the per-key OnceLock
    let test_entry = exact_value_cache()
        .entry(cache_id.clone())
        .or_try_insert_with(|| -> Result<OnceCell<String>, Box<dyn std::error::Error>> {
            let lock = OnceCell::new();
            Ok(lock)
        })?;
    test_entry
        .get_or_try_init(|| -> Result<String, Box<dyn std::error::Error>> {
            // values are kept across runs, by the dataset's path: those of another version of it
            // (with another number of nodes), or of an interrupted run, are computed again
            let complete = std::fs::metadata(&e_fn)
                .is_ok_and(|m| m.len() == (graph.size() * std::mem::size_of::<f64>()) as u64);
            if complete {
                eprintln!("found exact vals for {:?} at {e_fn}", graph_path);
                Ok(e_fn)
            } else {
                let _ = std::fs::remove_file(&e_fn);
                // return Err(format!("no entry {e_fn}").into());
                eprintln!("built exact vals for {:?}", graph_path);
                eprintln!("built in {:?} {e_fn}", exact_value_dir());
                match value_type {
                    FileType::ExactClosenessCentrality(H::H) => {
                        println!("compute closeness centrality");
                        ExactClosenessCentrality::<_, _, _, f64>::new(graph, Some(true))?;
                    }
                    FileType::ExactHarmonicCentrality(H::H) => {
                        println!("compute harmonic centrality");
                        ExactHarmonicCentrality::<_, _, _, f64>::new(graph, Some(true))?;
                    }
                    FileType::ExactLinCentrality(H::H) => {
                        println!("compute lin's centrality");
                        ExactLinCentrality::<_, _, _, f64>::new(graph)?;
                    }
                    _ => {}
                };
                Ok(e_fn)
            }
        })
        .cloned()
}

#[allow(dead_code)]
#[cfg(feature = "bench")]
/// Build (or open) a graph for a given dataset path.
pub fn load_graph<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    dataset: P,
) -> Result<GraphMemoryMap<N, E, Ix>, Box<dyn std::error::Error>> {
    let cache = GraphCache::<N, E, Ix>::from_file(dataset, None, None, None)?;
    GraphMemoryMap::init_from_cache(cache, Some(16))
}

// #[allow(dead_code)]
// #[cfg(feature = "bench")]
// /// Build (or open) a graph for a given dataset path.
// pub fn load_graph<EdgeType, Edge, P: AsRef<Path>>(
//     dataset: P,
// ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>>
// where
//     EdgeType: GenericEdgeType,
//     Edge: GenericEdge<EdgeType>,
// {
//     let cache = get_or_init_dataset_cache_entry(dataset.as_ref())?;
//     GraphMemoryMap::init(cache, Some(16))
// }

#[allow(dead_code)]
#[cfg(feature = "bench")]
/// Build (or open) a graph for a given dataset path with a given `suggested threads` number.
pub fn load_graph_with_threads<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    dataset: P,
    threads: u8,
) -> Result<GraphMemoryMap<N, E, Ix>, Box<dyn std::error::Error>> {
    let cache = GraphCache::<N, E, Ix>::from_file(dataset, None, None, None)?;
    GraphMemoryMap::init_from_cache(cache, Some(threads))
}

#[allow(dead_code)]
#[cfg(feature = "bench")]
pub static DATASETS: &[(&str, &str)] = &[
    ("ggcat_1_5", "datasets/graphs/graph_1_5.lz4"),
    ("ggcat_2_5", "datasets/graphs/graph_2_5.lz4"),
    ("ggcat_3_5", "datasets/graphs/graph_3_5.lz4"),
    ("ggcat_4_5", "datasets/graphs/graph_4_5.lz4"),
    // ("ggcat_5_5", "datasets/graphs/graph_5_5.lz4"),
    // ("ggcat_6_5", "datasets/graphs/graph_6_5.lz4"),
    // ("ggcat_7_5", "datasets/graphs/graph_7_5.lz4"),
    // ("ggcat_8_5", "datasets/graphs/graph_8_5.lz4"),
    // ("ggcat_9_5", "datasets/graphs/graph_9_5.lz4"),
    ("ggcat_1_10", "datasets/graphs/graph_1_10.lz4"),
    ("ggcat_2_10", "datasets/graphs/graph_2_10.lz4"),
    ("ggcat_3_10", "datasets/graphs/graph_3_10.lz4"),
    ("ggcat_4_10", "datasets/graphs/graph_4_10.lz4"),
    // ("ggcat_5_10", "datasets/graphs/graph_5_10.lz4"),
    // ("ggcat_6_10", "datasets/graphs/graph_6_10.lz4"),
    // ("ggcat_7_10", "datasets/graphs/graph_7_10.lz4"),
    // ("ggcat_8_10", "datasets/graphs/graph_8_10.lz4"),
    // ("ggcat_9_10", "datasets/graphs/graph_9_10.lz4"),
    // ("ggcat_8_15", "datasets/graphs/graph_8_15.lz4"),
    // ("ggcat_9_15", "datasets/graphs/graph_9_15.lz4"),
    // ("kmer_V2a", "../proteic_dbgs/1/kmer_V2a/kmer_V2a.mtx"),
    // ("kmer_A2a", "../proteic_dbgs/2/kmer_A2a/kmer_A2a.mtx"),
    // ("kmer_V1r", "../proteic_dbgs/3/kmer_V1r/kmer_V1r.mtx"),
];
