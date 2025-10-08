use crate::{
    centralities::exact::{ExactHarmonicCentrality, ExactLinCentrality},
    graph::{
        self, GraphMemoryMap,
        cache::{
            GraphCache,
            utils::{
                CACHE_DIR, EXACT_VALUE_CACHE_DIR, FileType, H, cache_file_name_from_id,
                id_from_filename, pers_cache_file_name,
            },
        },
        label::VoidLabel,
    },
    shared_slice::SharedSliceMut,
    trails::bfs::BFSDists,
};

use crossbeam::thread;
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::path::{Path, PathBuf};

type CacheRecord = OnceCell<GraphCache<VoidLabel, VoidLabel, usize>>;
type TestCache = DashMap<String, CacheRecord>;
type ExactValueCache = DashMap<String, OnceCell<String>>;

static TEST_CACHE: OnceCell<TestCache> = OnceCell::new();
static EXACT_VALUE_CACHE: OnceCell<ExactValueCache> = OnceCell::new();

pub(crate) fn mem_cache() -> &'static TestCache {
    TEST_CACHE.get_or_init(DashMap::new)
}

pub(crate) fn exact_value_cache() -> &'static ExactValueCache {
    EXACT_VALUE_CACHE.get_or_init(DashMap::new)
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
) -> Result<GraphCache<VoidLabel, VoidLabel, usize>, Box<dyn std::error::Error>> {
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;

    // Get or create the per-key OnceCell
    let test_entry = mem_cache().entry(cache_id).or_try_insert_with(
        || -> Result<CacheRecord, Box<dyn std::error::Error>> {
            let lock = OnceCell::new();
            Ok(lock)
        },
    )?;
    let cache = test_entry.get_or_try_init(
        || -> Result<GraphCache<VoidLabel, VoidLabel, usize>, Box<dyn std::error::Error>> {
            let cache_filename = cache_file_for(graph_path)?;
            eprintln!("check if file with fn {cache_filename} exists");
            if Path::new(&cache_filename).exists() {
                eprintln!("yes, found it {:?}", graph_path);
                GraphCache::<VoidLabel, VoidLabel, usize>::open(&cache_filename, None).map_err(
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

                GraphCache::<VoidLabel, VoidLabel, usize>::from_file(
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

#[cfg(feature = "bench")]
pub fn get_or_init_dataset_exact_closeness<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    graph_path: &Path,
    graph: &GraphMemoryMap<N, E, Ix>,
) -> Result<String, Box<dyn std::error::Error>> {
    get_or_init_dataset_exact_value(graph_path, graph, FileType::ExactClosenessCentrality(H::H))
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

    // Get or create the per-key OnceCell
    let test_entry = exact_value_cache()
        .entry(cache_id.clone())
        .or_try_insert_with(|| -> Result<OnceCell<String>, Box<dyn std::error::Error>> {
            let lock = OnceCell::new();
            Ok(lock)
        })?;
    test_entry
        .get_or_try_init(|| -> Result<String, Box<dyn std::error::Error>> {
            if Path::new(&e_fn).exists() {
                eprintln!("found exact vals for {:?} at {e_fn}", graph_path);
                Ok(e_fn)
            } else {
                // return Err(format!("no entry {e_fn}").into());
                eprintln!("built exact vals for {:?}", graph_path);
                eprintln!("built in {:?} {e_fn}", exact_value_dir());
                match value_type {
                    FileType::ExactClosenessCentrality(H::H) => {
                        println!("compute closeness centrality");
                        let node_count = graph.size();
                        let e = SharedSliceMut::<f64>::abst_mem_mut(&e_fn, node_count, true)?;
                        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                            let mut handles = vec![];
                            let node_load = node_count.div_ceil(50);
                            for tid in 0..50 {
                                let graph = graph.clone();
                                let mut e = e.shared_slice();
                                let begin = (node_load * tid).min(node_count);
                                let end = (begin + node_load).min(node_count);
                                handles.push(scope.spawn(
                                    move |_| -> Result<
                                        (),
                                        Box<dyn std::error::Error + Send + Sync>,
                                    > {
                                        for u in begin..end {
                                            if u % 1000 == 0 {
                                                println!("reached {u} of {node_count}");
                                            }
                                            let bfs = BFSDists::new_t(&graph, u, tid).map_err(
                                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                                    format!("error in BFS for {u}: {:?}", e).into()
                                                },
                                            )?;
                                            if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                                println!("found isolated node at {u}");
                                                *e.get_mut(u) = 0.0;
                                            } else {
                                                *e.get_mut(u) =
                                                    bfs.recheable() as f64 / bfs.total_distances();
                                            }
                                        }
                                        Ok(())
                                    },
                                ));
                            }
                            // check for errors
                            for (tid, r) in handles.into_iter().enumerate() {
                                r.join()
                                    .map_err(|e| -> Box<dyn std::error::Error> {
                                        format!("error in thread {tid}: {:?}", e).into()
                                    })?
                                    .map_err(|e| -> Box<dyn std::error::Error> {
                                        format!("error: {:?}", e).into()
                                    })?;
                            }
                            Ok(())
                        })
                        .map_err(
                            |e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() },
                        )??;
                        // ExactClosenessCentrality::new(graph, Some(true))?;
                    }
                    FileType::ExactHarmonicCentrality(H::H) => {
                        println!("compute harmonic centrality");
                        ExactHarmonicCentrality::new(graph, Some(true))?;
                    }
                    FileType::ExactLinCentrality(H::H) => {
                        println!("compute lin's centrality");
                        ExactLinCentrality::new(graph)?;
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
