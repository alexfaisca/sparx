use glob::glob;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    fmt::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

#[cfg(not(any(test, feature = "bench")))]
pub static CACHE_DIR: &str = "./.cache/";
#[cfg(any(test, feature = "bench"))]
pub static CACHE_DIR: &str = "./.test_cache/";

#[cfg(any(test, feature = "bench"))]
pub static EXACT_VALUE_CACHE_DIR: &str = "exact_values/";

pub(super) fn apply_permutation_in_place<T, U>(
    perm: &mut [usize], // perm[i] = destination index for element currently at i
    data1: &mut [T],
    data2: &mut [U],
) {
    debug_assert!(perm.len() <= data1.len());
    debug_assert!(data1.len() <= data2.len());
    debug_assert!(perm.iter().all(|&p| p < perm.len())); // valid indices

    for i in 0..perm.len() {
        while perm[i] != i {
            let k = perm[i];

            // apply same swap to both data arrays
            data1.swap(i, k);
            data2.swap(i, k);

            // CRUCIAL: keep perm consistent with the data by swapping it too
            perm.swap(i, k);
            // After the swap, perm[j] might now be j; loop continues until it is.
        }
    }
}

/// Create a stable 256-bit hex id from a filename (path & extension are ignored).
///
/// - Deterministic across runs and machines.
/// - Uses SHA-256.
/// - Returns a 64-char lowercase hex string.
///
/// # Arguments
///
/// * `key` --- the key to be hashed in order to obtain the hexadecimal id.
///
#[allow(dead_code)]
pub fn id_from_filename(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();

    // hex-encode.
    let mut result = String::with_capacity(digest.len() * 2);
    digest
        .iter()
        .try_for_each(|b| -> Result<(), Box<dyn std::error::Error>> {
            write!(&mut result, "{:02x}", b)?;
            Ok(())
        })?;
    Ok(result)
}

#[allow(dead_code)]
pub fn graph_id_from_cache_file_name(
    cache_filename: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = Path::new(cache_filename.as_str());

    let file_name = path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- empty path".into()
        })?;

    // extract id from filename
    let re = Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).map_err(
        |e| -> Box<dyn std::error::Error> {
            format!("error analyzing file name (regex failed): {e}").into()
        },
    )?;

    let caps = re
        .captures(file_name)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 1)".into()
        })?;

    let id = caps
        .get(1)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 2)".into()
        })?
        .as_str();

    Ok(id.to_string())
}

#[allow(dead_code)]
fn graph_id_and_dir_from_cache_file_name(
    cache_filename: &str,
    _target_type: &FileType,
) -> Result<(String, PathBuf), Box<dyn std::error::Error>> {
    let path = Path::new(cache_filename);

    let file_name = path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path empty".into()
        })?;

    let parent_dir = path.parent().unwrap_or_else(|| Path::new(""));

    // extract id from filename
    let re = Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).map_err(
        |e| -> Box<dyn std::error::Error> {
            format!("error analyzing file name (regex failed): {e}").into()
        },
    )?;

    let caps = re
        .captures(file_name)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 1)".into()
        })?;

    let id = caps
        .get(1)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 2)".into()
        })?
        .as_str()
        .to_string();

    #[cfg(any(test, feature = "bench"))]
    // check if parent_dir needs tweaking for specific target type
    if *_target_type == FileType::Test(H::H) {
        return Ok((id, PathBuf::from(CACHE_DIR)));
    } else if *_target_type == FileType::ExactClosenessCentrality(H::H)
        || *_target_type == FileType::ExactHarmonicCentrality(H::H)
        || *_target_type == FileType::ExactLinCentrality(H::H)
    {
        return Ok((id, PathBuf::from(parent_dir).join(EXACT_VALUE_CACHE_DIR)));
    }

    Ok((id, PathBuf::from(parent_dir)))
}

#[allow(dead_code)]
pub fn pers_cache_file_name(
    original_filename: &str,
    target_type: &FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename, target_type)?;
    let new_filename =
        pers_file_name_from_id_and_sequence_for_type(target_type, &id, sequence_number);
    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

#[allow(dead_code)]
pub fn toml_cache_file_name(
    original_filename: &str,
    target_type: &FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    // For now only human readable file
    assert!(*target_type == FileType::CacheMetadata(H::H));
    let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename, target_type)?;
    let new_filename =
        toml_file_name_from_id_and_sequence_for_type(target_type, &id, sequence_number);
    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

#[allow(dead_code)]
pub fn cache_file_name(
    original_filename: &str,
    target_type: &FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename, target_type)?;
    let new_filename = file_name_from_id_and_sequence_for_type(target_type, &id, sequence_number);
    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

pub fn id_for_subgraph_export(id: String, sequence_number: Option<usize>) -> String {
    // this isn't a filename --- it's an id to be used in filenames
    match sequence_number {
        Some(i) => format!("{}<{}>{}", "inducedsubgraph", i, id),
        None => format!("{}{}", "inducedsubgraph", id),
    }
}

/// Remove all cached `.tmp` files for a given `id` and [`FileType`] in the cache directory.
///
/// [`FileType`]: ./enum.FileType.html#
#[allow(dead_code)]
pub(crate) fn cleanup_cache(
    id: &str,
    target_type: &FileType,
) -> Result<(), Box<dyn std::error::Error>> {
    let match_entries_to =
        CACHE_DIR.to_string() + suffix_for_file_type(target_type) + "*" + id + ".tmp";
    let cache_entries = glob(&match_entries_to).map_err(|e| -> Box<dyn std::error::Error> {
        format!("error cleaning up cache for entries with name {match_entries_to}: {e}").into()
    })?;
    for entry in cache_entries {
        std::fs::remove_file(entry.map_err(|e| -> Box<dyn std::error::Error> {
            format!("error cleaning up cache entry: {e}").into()
        })?)?;
    }
    Ok(())
}

/// Remove all cached `.tmp` in the cache directory.
///
#[allow(dead_code)]
pub(crate) fn remove_tmp_files_from_cache() -> Result<(), Box<dyn std::error::Error>> {
    let cache_entries = glob((CACHE_DIR.to_string() + "/*.tmp").as_str()).map_err(
        |e| -> Box<dyn std::error::Error> { format!("error cleaning up cache: {e}").into() },
    )?;
    for entry in cache_entries {
        std::fs::remove_file(entry.map_err(|e| -> Box<dyn std::error::Error> {
            format!("error cleaning up cache entry: {e}").into()
        })?)?;
    }
    Ok(())
}

/// Hides file types from users.
///
/// Short for "Hidden".
#[derive(Clone, Debug, PartialEq)]
pub enum H {
    H,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code, private_interfaces, clippy::upper_case_acronyms)]
pub enum FileType {
    /// Only member visible to users
    General,
    CacheMetadata(H),
    Edges(H),
    Index(H),
    NodeLabel(H),
    EdgeLabel(H),
    MetaLabel(H),
    Helper(H),
    BFS(H),
    DFS(H),
    EulerIndex(H),
    EulerTrail(H),
    KCoreBZ(H),
    KCoreLEA(H),
    Triangles(H),
    KTrussBEA(H),
    KTrussPKT(H),
    ClusteringCoefficient(H),
    EdgeReciprocal(H),
    EdgeOver(H),
    HyperBall(H),
    HyperBallDistances(H),
    HyperBallInvDistances(H),
    HyperBallClosenessCentrality(H),
    HyperBallHarmonicCentrality(H),
    HyperBallLinCentrality(H),
    GVELouvain(H),
    ExactClosenessCentrality(H),
    ExactHarmonicCentrality(H),
    ExactLinCentrality(H),
    #[cfg(any(test, feature = "bench"))]
    Test(H),
}

pub fn cache_file_name_from_id(
    target_type: &FileType,
    id: &str,
    sequence_number: Option<usize>,
) -> String {
    CACHE_DIR.to_string()
        + file_name_from_id_and_sequence_for_type(target_type, id, sequence_number).as_str()
}

pub fn cache_metadata_file_name_from_id(
    target_type: &FileType,
    id: &str,
    sequence_number: Option<usize>,
) -> String {
    CACHE_DIR.to_string()
        + toml_file_name_from_id_and_sequence_for_type(target_type, id, sequence_number).as_str()
}

fn toml_file_name_from_id_and_sequence_for_type(
    target: &FileType,
    id: &str,
    seq: Option<usize>,
) -> String {
    match seq {
        None => format!("{}_{}.{}", suffix_for_file_type(target), id, "toml"),
        // sequenced files are temporary
        Some(i) => format!("{}_{}_{}.{}", suffix_for_file_type(target), i, id, "toml"),
    }
}

fn pers_file_name_from_id_and_sequence_for_type(
    target: &FileType,
    id: &str,
    seq: Option<usize>,
) -> String {
    match seq {
        None => format!("{}_{}.{}", suffix_for_file_type(target), id, "mmap"),
        // sequenced files are temporary
        Some(i) => format!("{}_{}_{}.{}", suffix_for_file_type(target), i, id, "mmap"),
    }
}

fn file_name_from_id_and_sequence_for_type(
    target: &FileType,
    id: &str,
    seq: Option<usize>,
) -> String {
    match seq {
        None => format!("{}_{}.{}", suffix_for_file_type(target), id, "mmap"),
        // sequenced files are temporary
        Some(i) => format!("{}_{}_{}.{}", suffix_for_file_type(target), i, id, "tmp"),
    }
}

fn suffix_for_file_type(target_type: &FileType) -> &'static str {
    static SUFFIX_FOR_GENERAL: &str = "miscelanious";
    static SUFFIX_FOR_CACHE_METADATA: &str = "metadata";
    static SUFFIX_FOR_EDGES: &str = "edges";
    static SUFFIX_FOR_INDEX: &str = "index";
    static SUFFIX_FOR_NODE_LABEL: &str = "nodelabels";
    static SUFFIX_FOR_EDGE_LABEL: &str = "edgelabels";
    static SUFFIX_FOR_META_LABEL: &str = "fst";
    static SUFFIX_FOR_HELPER: &str = "helper";
    static SUFFIX_FOR_BFS: &str = "bfs";
    static SUFFIX_FOR_DFS: &str = "dfs";
    static SUFFIX_FOR_EULER_INDEX: &str = "eulerindex";
    static SUFFIX_FOR_EULER_TRAIL: &str = "eulertrail";
    static SUFFIX_FOR_KCORE_BZ: &str = "kcoresbz";
    static SUFFIX_FOR_KCORE_LEA: &str = "kcoreslea";
    static SUFFIX_FOR_TRIANGLES: &str = "triangles";
    static SUFFIX_FOR_KTRUSS_BEA: &str = "ktrusslea";
    static SUFFIX_FOR_KTRUSS_PKT: &str = "ktrusspkt";
    static SUFFIX_FOR_CLUSTERING_COEFFICIENT: &str = "clusteringcoefficient";
    static SUFFIX_FOR_EDGE_RECIPROCAL: &str = "edgereciprocal";
    static SUFFIX_FOR_EDGE_OVER: &str = "edgeover";
    static SUFFIX_FOR_HYPERBALL: &str = "hyperball";
    static SUFFIX_FOR_HYPERBALL_DISTANCES: &str = "hyperballdistances";
    static SUFFIX_FOR_HYPERBALL_INV_DISTANCES: &str = "hyperballinvdistances";
    static SUFFIX_FOR_HYPERBALL_CLOSENESS_CENTRALITY: &str = "hyperballcloseness";
    static SUFFIX_FOR_HYPERBALL_HARMONIC_CENTRALITY: &str = "hyperballharmonic";
    static SUFFIX_FOR_HYPERBALL_LIN_CENTRALITY: &str = "hyperballlin";
    static SUFFIX_FOR_GVE_LOUVAIN: &str = "louvain";
    static SUFFIX_FOR_EXACT_CLOSENESS_CENTRALITY: &str = "exactclosenesscentrality";
    static SUFFIX_FOR_EXACT_HARMONIC_CENTRALITY: &str = "exactharmoniccentrality";
    static SUFFIX_FOR_EXACT_LIN_CENTRALITY: &str = "exactlincentrality";
    #[cfg(any(test, feature = "bench"))]
    static SUFFIX_FOR_TEST: &str = "test";

    match target_type {
        FileType::General => SUFFIX_FOR_GENERAL,
        FileType::CacheMetadata(_) => SUFFIX_FOR_CACHE_METADATA,
        FileType::Edges(_) => SUFFIX_FOR_EDGES,
        FileType::Index(_) => SUFFIX_FOR_INDEX,
        FileType::NodeLabel(_) => SUFFIX_FOR_NODE_LABEL,
        FileType::EdgeLabel(_) => SUFFIX_FOR_EDGE_LABEL,
        FileType::MetaLabel(_) => SUFFIX_FOR_META_LABEL,
        FileType::Helper(_) => SUFFIX_FOR_HELPER,
        FileType::BFS(_) => SUFFIX_FOR_BFS,
        FileType::DFS(_) => SUFFIX_FOR_DFS,
        FileType::EulerIndex(_) => SUFFIX_FOR_EULER_INDEX,
        FileType::EulerTrail(_) => SUFFIX_FOR_EULER_TRAIL,
        FileType::KCoreBZ(_) => SUFFIX_FOR_KCORE_BZ,
        FileType::KCoreLEA(_) => SUFFIX_FOR_KCORE_LEA,
        FileType::Triangles(_) => SUFFIX_FOR_TRIANGLES,
        FileType::KTrussBEA(_) => SUFFIX_FOR_KTRUSS_BEA,
        FileType::KTrussPKT(_) => SUFFIX_FOR_KTRUSS_PKT,
        FileType::ClusteringCoefficient(_) => SUFFIX_FOR_CLUSTERING_COEFFICIENT,
        FileType::EdgeReciprocal(_) => SUFFIX_FOR_EDGE_RECIPROCAL,
        FileType::EdgeOver(_) => SUFFIX_FOR_EDGE_OVER,
        FileType::HyperBall(_) => SUFFIX_FOR_HYPERBALL,
        FileType::HyperBallDistances(_) => SUFFIX_FOR_HYPERBALL_DISTANCES,
        FileType::HyperBallInvDistances(_) => SUFFIX_FOR_HYPERBALL_INV_DISTANCES,
        FileType::HyperBallClosenessCentrality(_) => SUFFIX_FOR_HYPERBALL_CLOSENESS_CENTRALITY,
        FileType::HyperBallHarmonicCentrality(_) => SUFFIX_FOR_HYPERBALL_HARMONIC_CENTRALITY,
        FileType::HyperBallLinCentrality(_) => SUFFIX_FOR_HYPERBALL_LIN_CENTRALITY,
        FileType::GVELouvain(_) => SUFFIX_FOR_GVE_LOUVAIN,
        FileType::ExactClosenessCentrality(_) => SUFFIX_FOR_EXACT_CLOSENESS_CENTRALITY,
        FileType::ExactHarmonicCentrality(_) => SUFFIX_FOR_EXACT_HARMONIC_CENTRALITY,
        FileType::ExactLinCentrality(_) => SUFFIX_FOR_EXACT_LIN_CENTRALITY,
        #[cfg(any(test, feature = "bench"))]
        FileType::Test(_) => SUFFIX_FOR_TEST,
    }
}

/// Given a path to `unitig_{name}.edges`,
/// return the companion `unitig_annotated_{name}.nodes`.
#[cfg(feature = "nodes_edges")]
pub(super) fn edges_to_nodes<P: AsRef<Path>>(edges_file: P) -> Option<PathBuf> {
    let path = edges_file.as_ref();

    let parent = path.parent();
    let stem = path.file_stem()?.to_str()?;
    let ext = path.extension()?.to_str()?;

    if ext != "edges" || !stem.starts_with("unitig_") {
        return None;
    }

    let name = &stem["unitig_".len()..];
    if name.is_empty() {
        return None;
    }

    let new_name = format!("unitig_annotated_{}.nodes", name);
    Some(parent.map_or_else(
        || PathBuf::from(new_name.clone()),
        |dir| dir.join(new_name.clone()),
    ))
}

/// Given a path to `unitig_annotated_{name}.nodes`,
/// return the companion `unitig_{name}.edges`.
#[cfg(feature = "nodes_edges")]
pub(super) fn nodes_to_edges<P: AsRef<Path>>(nodes_file: P) -> Option<PathBuf> {
    let path = nodes_file.as_ref();

    let parent = path.parent();
    let stem = path.file_stem()?.to_str()?;
    let ext = path.extension()?.to_str()?;

    if ext != "nodes" || !stem.starts_with("unitig_annotated_") {
        return None;
    }

    let name = &stem["unitig_annotated_".len()..];
    if name.is_empty() {
        return None;
    }

    let new_name = format!("unitig_{}.edges", name);
    Some(parent.map_or_else(
        || PathBuf::from(new_name.clone()),
        |dir| dir.join(new_name.clone()),
    ))
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FileType::General => "Miscelanious",
            FileType::CacheMetadata(_) => "Metadata",
            FileType::Edges(_) => "Edges",
            FileType::Index(_) => "Index",
            FileType::NodeLabel(_) => "NodeLabel",
            FileType::EdgeLabel(_) => "EdgeLabel",
            FileType::MetaLabel(_) => "MetaLabel",
            FileType::Helper(_) => "Helper",
            FileType::BFS(_) => "BFS",
            FileType::DFS(_) => "DFS",
            FileType::EulerIndex(_) => "EulerIndex",
            FileType::EulerTrail(_) => "EulerTrail",
            FileType::KCoreBZ(_) => "KCoreBatageljZaversnik",
            FileType::KCoreLEA(_) => "KCoreLiuEtAl",
            FileType::Triangles(_) => "Triangles",
            FileType::KTrussBEA(_) => "KTrussBurkhardtEtAl",
            FileType::KTrussPKT(_) => "KTrussPKT",
            FileType::ClusteringCoefficient(_) => "ClusteringCoefficient",
            FileType::EdgeReciprocal(_) => "EdgeReciprocal",
            FileType::EdgeOver(_) => "EdgeOver",
            FileType::HyperBall(_) => "HyperBall",
            FileType::HyperBallDistances(_) => "HyperBallDistances",
            FileType::HyperBallInvDistances(_) => "HyperBallInverseDistances",
            FileType::HyperBallClosenessCentrality(_) => "HyperBallClosenessCentrality",
            FileType::HyperBallHarmonicCentrality(_) => "HyperBallHarmonicCentrality",
            FileType::HyperBallLinCentrality(_) => "HyperBallLinCentrality",
            FileType::GVELouvain(_) => "Louvain",
            FileType::ExactClosenessCentrality(_) => "ExactClosenessCentrality",
            FileType::ExactHarmonicCentrality(_) => "ExactHarmonicCentrality",
            FileType::ExactLinCentrality(_) => "ExactLinCentrality",
            #[cfg(any(test, feature = "bench"))]
            FileType::Test(_) => "Test",
        };
        write!(f, "FileType {{{}}}", s)
    }
}

/// Ensure `path` exists and is writable. If it's read-only, make it writable.
///
/// Returns:
/// - Ok(true)  -> file exists and is writable now (already or after change)
/// - Ok(false) -> file does not exist
/// - Err(e)    -> other I/O error (permissions, ACLs, etc.)
pub(super) fn ensure_file_writable(
    path: impl AsRef<Path>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let path = path.as_ref();

    // Does it exist?
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };

    // If it's a directory, treat as error (adjust if you want to allow dirs)
    if meta.is_dir() {
        return Err("path refers to a directory, not a file".into());
    }

    #[cfg(not(unix))]
    {
        let mut perms = meta.permissions();
        if !perms.readonly() {
            return Ok(true);
        }
        // Clear the read-only attribute on Windows/other.
        perms.set_readonly(false);
        std::fs::set_permissions(path, perms)?;
        return Ok(true);
    }

    #[cfg(unix)]
    {
        let perms = meta.permissions();
        let mode = perms.mode();
        let user_w = 0o200;
        if (mode & user_w) != 0 {
            return Ok(true); // already user-writable
        }
        let new_mode = mode | user_w;
        let mut new_perms = perms;
        new_perms.set_mode(new_mode);
        std::fs::set_permissions(path, new_perms)?;
        Ok(true)
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;
    use std::fs::{self, File};
    #[allow(unused_imports)]
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    // ---- helpers -------------------------------------------------------------

    /// Ensure CACHE_DIR exists before filesystem tests.
    #[allow(dead_code)]
    fn ensure_cache_dir() -> PathBuf {
        let p = Path::new(CACHE_DIR);
        fs::create_dir_all(p).unwrap();
        p.to_path_buf()
    }

    #[allow(dead_code)]
    fn touch<P: AsRef<Path>>(p: P) {
        File::create(p).expect("failed to create file");
    }

    #[test]
    fn test_apply_permutations() {
        apply_permutation_in_place::<usize, f64>(&mut [], &mut [], &mut []);

        let mut idx0 = [2, 1, 0];
        let mut data01: [usize; 3] = [0, 1, 2];
        let mut data02: [f64; 3] = [2., 1., 0.];
        apply_permutation_in_place(&mut idx0, &mut data01, data02.as_mut());
        assert!(idx0 == [0, 1, 2]);
        assert!(data01 == [2, 1, 0]);
        assert!(data02 == [0., 1., 2.]);

        let mut idx1 = [5, 4, 0, 1, 2, 6, 3];
        let mut data11: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
        let mut data12: [f64; 7] = [6., 5., 4., 3., 2., 1., 0.];
        apply_permutation_in_place(&mut idx1, &mut data11, data12.as_mut());
        assert!(idx1 == [0, 1, 2, 3, 4, 5, 6]);
        assert!(data11 == [2, 3, 4, 6, 1, 0, 5]);
        assert!(data12 == [4., 3., 2., 0., 5., 6., 1.]);

        let mut idx2 = [3, 4, 0, 1, 2, 6, 5];
        let mut data21: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut data22: [f64; 10] = [6., 5., 4., 3., 2., 1., 0., 7., 8., 9.];
        apply_permutation_in_place(&mut idx2, &mut data21, &mut data22);
        assert!(idx2 == [0, 1, 2, 3, 4, 5, 6]);
        assert!(data21 == [2, 3, 4, 0, 1, 6, 5, 7, 8]);
        assert!(data22 == [4., 3., 2., 6., 5., 0., 1., 7., 8., 9.]);
    }

    #[test]
    fn test_id_from_filename() {
        let got = id_from_filename("").unwrap();
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let s = "The king’s palace is an easy place to enter but hard to leave.";
        let got = id_from_filename(s).unwrap();
        assert_eq!(
            got,
            "50cac1edb3fb760fb223f423fddb4011b7162499b4dab5af99dc1c2eb8e99724"
        );

        let s = "Bare is the back of a brotherless man.";
        let got = id_from_filename(s).unwrap();
        assert_eq!(
            got,
            "4e9b2b6e09882e1fee49a2129bd427d4a5549240291f4dbe497f44565841a843"
        );

        let s = "There is a great difference in what men are born into the world for and what they become known for.";
        let got = id_from_filename(s).unwrap();
        assert_eq!(
            got,
            "e0714d960fff0f743c33188dd4858075afdc0516489b9395ca789ef5cf2cff76"
        );

        let s = "There’s no slaying a man destined to live.";
        let got = id_from_filename(s).unwrap();
        assert_eq!(
            got,
            "0f0d93692e456ed51534419aca2f6f2337a058baa81c5d968cccd7099fcb31e7"
        );
    }

    #[test]
    fn test_graph_id_from_cache_filename() {
        let p = "/data/cache/edges_foo123.bin".to_string();
        let id = graph_id_from_cache_file_name(p).unwrap();
        assert_eq!(id, "foo123");
        let p = "/some/dir/neighbors_my_graph_42.mtx".to_string();
        let id = graph_id_from_cache_file_name(p).unwrap();
        assert_eq!(id, "42");
        // Windows? Not allowed! muahahahha :3
        let p = r"C:\cache\OFFSETS_theID123.DAT".to_string();
        assert!(graph_id_from_cache_file_name(p).is_err());
        // lacks the required "<prefix>_<id>.<ext>" pattern
        let p = "file.txt".to_string();
        assert!(graph_id_from_cache_file_name(p).is_err());
        let p = "edges_id".to_string();
        assert!(graph_id_from_cache_file_name(p).is_err());
        // id contains '-' which is not matched by \w+
        let p = "edges_my-id.bin".to_string();
        assert!(graph_id_from_cache_file_name(p).is_err())
    }

    #[test]
    fn test_graph_id_and_dir_from_cache_filename() {
        let p = "/data/cache/edges_aaabbb123.bin";
        let (id, dir) = graph_id_and_dir_from_cache_file_name(p, &FileType::Edges(H::H)).unwrap();

        assert_eq!(id, "aaabbb123");

        let expected_dir = Path::new(p)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();
        assert_eq!(dir, expected_dir);
        let p = "OFFSETS_TheID123.DAT";
        let (id, dir) = graph_id_and_dir_from_cache_file_name(p, &FileType::Index(H::H)).unwrap();

        assert_eq!(id, "TheID123");
        assert_eq!(dir, PathBuf::from("")); // function uses Path::new("")

        let p =
            "/any/dir/NEIGHBORS_123aef543ba2352612acdefddfe3253315435aaa9090982035808abbeeecc.bin";
        let (id, dir) =
            graph_id_and_dir_from_cache_file_name(p, &FileType::NodeLabel(H::H)).unwrap();

        assert_eq!(
            id,
            "123aef543ba2352612acdefddfe3253315435aaa9090982035808abbeeecc"
        );
        let expected_dir = Path::new(p)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();
        assert_eq!(dir, expected_dir);

        let p =
            "/any/dir/NEIGHBORS_1ebac462355920468dededed35820358105814305820851058bcadefedecc.bin";
        let (id, dir) = graph_id_and_dir_from_cache_file_name(p, &FileType::Test(H::H)).unwrap();
        assert_eq!(
            id,
            "1ebac462355920468dededed35820358105814305820851058bcadefedecc"
        );
        assert_eq!(dir, Path::new(CACHE_DIR));
    }

    // ---- pers_cache_file_name ------------------------------------------------

    #[test]
    fn test_pers_name_edges_no_seq() {
        // original filename matches "<prefix>_<id>.<ext>"
        let orig = "/some/dir/EDGES_myId.mmap";
        let got = pers_cache_file_name(orig, &FileType::Edges(H::H), None).unwrap();
        // pers_* always ends with ".mmap"
        assert_eq!(got, "/some/dir/edges_myId.mmap");
    }

    #[test]
    fn test_pers_name_index_with_seq() {
        let orig = "/d/OFFSETS_idX.mmap";
        let got = pers_cache_file_name(orig, &FileType::Index(H::H), Some(7)).unwrap();
        // pers_* with seq => "<suffix>_<seq>_<id>.mmap"
        assert_eq!(got, "/d/index_7_idX.mmap");
    }

    // ---- cache_file_name -----------------------------------------------------

    #[test]
    fn test_cache_name_index_no_seq_mmap() {
        let orig = "/d/OFFSETS_idX.whatever";
        let got = cache_file_name(orig, &FileType::Index(H::H), None).unwrap();
        // no seq => ".mmap"
        assert_eq!(got, "/d/index_idX.mmap");
    }

    #[test]
    fn test_cache_name_index_with_seq_tmp() {
        let orig = "/d/OFFSETS_idX.whatever";
        let got = cache_file_name(orig, &FileType::Index(H::H), Some(3)).unwrap();
        // seq => ".tmp"
        assert_eq!(got, "/d/index_3_idX.tmp");
    }

    // ---- id_for_subgraph_export ---------------------------------------------

    #[test]
    fn test_id_for_subgraph_no_seq() {
        let got = id_for_subgraph_export("abc".to_string(), None);
        assert_eq!(got, "inducedsubgraphabc");
    }

    #[test]
    fn test_id_for_subgraph_with_seq() {
        let got = id_for_subgraph_export("xyz".to_string(), Some(5));
        assert_eq!(got, "inducedsubgraph<5>xyz");
    }

    // ---- cleanup_cache (removes matching *.tmp in CACHE_DIR) -----------------

    #[test]
    fn test_cleanup_cache_removes_only_matching_id_and_suffix() {
        let cache_root = ensure_cache_dir();

        // Build filenames that match the glob pattern used by cleanup_cache:
        //   CACHE_DIR + suffix_for_file_type(target) + "*" + id + ".tmp"
        let id_a = "idA_unique";
        let id_b = "idB_unique";

        let f1 = cache_root.join(format!(
            "{}_0_{}.tmp",
            suffix_for_file_type(&FileType::Index(H::H)),
            id_a
        ));
        let f2 = cache_root.join(format!(
            "{}_5_{}.tmp",
            suffix_for_file_type(&FileType::Index(H::H)),
            id_a
        ));
        let f3 = cache_root.join(format!(
            "{}_1_{}.tmp",
            suffix_for_file_type(&FileType::Edges(H::H)),
            id_b
        )); // different suffix+id

        touch(&f1);
        touch(&f2);
        touch(&f3);

        // Act: remove only Index/H files for idA
        cleanup_cache(id_a, &FileType::Index(H::H)).unwrap();

        assert!(!f1.exists(), "f1 should be deleted");
        assert!(!f2.exists(), "f2 should be deleted");
        assert!(f3.exists(), "f3 (different id/suffix) must remain");

        // cleanup leftover
        let _ = fs::remove_file(&f3);
    }

    // ---- remove_tmp_files_from_cache (removes all *.tmp in CACHE_DIR) -------

    #[test]
    fn test_remove_tmp_files_from_cache_removes_all_tmp() {
        let cache_root = ensure_cache_dir();

        let g1 = cache_root.join("index_9_idZ.tmp");
        let g2 = cache_root.join("helper_2_idZ.tmp");
        let keep = cache_root.join("edges_1_idZ.mmap"); // should remain

        touch(&g1);
        touch(&g2);
        touch(&keep);

        remove_tmp_files_from_cache().unwrap();

        assert!(!g1.exists(), "tmp file 1 should be gone");
        assert!(!g2.exists(), "tmp file 2 should be gone");
        assert!(keep.exists(), "non-tmp file must remain");

        let _ = fs::remove_file(&keep);
    }

    // ---- cache_file_name_from_id sanity (helper used elsewhere) --------------

    #[test]
    fn test_cache_file_name_from_id_formats_correctly() {
        let got = cache_file_name_from_id(&FileType::Edges(H::H), "fooBar", Some(12));
        // Expect: CACHE_DIR + "<suffix>_<seq>_<id>.tmp"
        let expected_tail = format!(
            "{}_{}_{}.tmp",
            suffix_for_file_type(&FileType::Edges(H::H)),
            12,
            "fooBar"
        );
        assert!(got.starts_with(CACHE_DIR));
        assert!(got.ends_with(&expected_tail), "got = {}", got);
    }

    /// Make a unique path under the OS temp dir without external deps.
    #[allow(dead_code)]
    fn unique_temp_path(name_hint: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        p.push(format!("{}_{}_{}", name_hint, pid, nanos));
        p
    }

    #[test]
    fn returns_false_for_missing_file() {
        let p = unique_temp_path("missing_file");
        assert!(!p.exists());
        let r = ensure_file_writable(&p).unwrap();
        assert!(!r, "should return Ok(false) for non-existent path");
    }

    #[test]
    fn ok_true_for_already_writable_file() {
        let p = unique_temp_path("writable_file.txt");
        {
            let mut f = fs::File::create(&p).unwrap();
            writeln!(f, "hello").unwrap();
        }
        // File starts writable; function should be a no-op that returns true.
        let r = ensure_file_writable(&p).unwrap();
        assert!(r, "existing writable file should return Ok(true)");

        // Clean up
        fs::remove_file(&p).ok();
    }

    #[test]
    fn makes_readonly_file_writable() {
        let p = unique_temp_path("readonly_file.txt");
        {
            let mut f = fs::File::create(&p).unwrap();
            writeln!(f, "data").unwrap();
        }

        // Make it read-only
        let mut perms = fs::metadata(&p).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // remove all write bits (user/group/other)
            perms.set_mode(perms.mode() & !0o222);
        }
        #[cfg(not(unix))]
        {
            perms.set_readonly(true);
        }
        fs::set_permissions(&p, perms).unwrap();

        // Verify that writing now fails
        let can_write_now = fs::OpenOptions::new().write(true).open(&p).is_ok();
        assert!(!can_write_now, "precondition: file should be read-only");

        // Call the function: it should flip to writable and return Ok(true)
        let r = ensure_file_writable(&p).unwrap();
        assert!(r, "should return Ok(true) after making file writable");

        // Now writing should succeed
        let mut wf = fs::OpenOptions::new().write(true).open(&p).unwrap();
        writeln!(wf, "more").unwrap();

        // Clean up
        fs::remove_file(&p).ok();
    }
}
