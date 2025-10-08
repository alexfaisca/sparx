#[allow(unused_imports)]
use sparkx::centralities::hyper_ball::*;
#[allow(unused_imports)]
use sparkx::communities::gve_louvain::AlgoGVELouvain;
#[allow(unused_imports)]
use sparkx::communities::{approx_dirichlet_hkpr::*, hk_relax::*};
use sparkx::graph;
#[allow(unused_imports)]
use sparkx::graph::{E, GraphMemoryMap, IndexType, N, label::VoidLabel};
#[allow(unused_imports)]
use sparkx::k_core::{batagelj_zaversnik::*, liu_et_al::*};
#[allow(unused_imports)]
use sparkx::k_truss::{burkhardt_et_al::*, clustering_coefficient::*, pkt::*};
use sparkx::shared_slice::SharedSliceMut;
#[allow(unused_imports)]
use sparkx::trails::hierholzer::*;

use clap::{ArgAction, Parser};
use static_assertions::const_assert;
use std::fmt::Display;
use std::fs::{OpenOptions, create_dir_all, metadata};
use std::io::{self, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Checout matchtigs
// https://github.com/algbio/matchtigs

static BITS: usize = std::mem::size_of::<usize>();
static BITS_U64: usize = std::mem::size_of::<u64>();

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

#[derive(Parser)]
#[command(
    name = "'SparkX'",
    version = "0.1",
    long_about = "Very pretentious name, for a library so small. I know... ('~')\n The contents of this crate should be a good start for a solid tool though. :)"
)]
struct ProgramArgs {
    /// Enable debugging mode
    #[arg(short, long)]
    debug: bool,

    /// Enable graph memory mapping mode
    #[arg(short, long, default_value_t = true)]
    mmap: bool,

    /// Program thread number, default is 1
    #[arg(short, long)]
    threads: Option<u8>,

    /// Enable or disable verbose output.
    ///
    /// Examples:
    ///   (not provided)   -> false
    ///   -v / --verbose   -> true
    ///   -v=true          -> true
    ///   -v=false         -> false
    #[arg(short, long, num_args(0..=1), default_value_t = false, default_missing_value = "true", action = ArgAction::Set)]
    verbose: bool,

    /// Output file(s) identifier (filenames are generated using this as the graph id)
    #[arg(short, long)]
    output_id: Option<String>,

    /// Required — input file (.txt, .lz4, .mmap are accepted)
    #[arg(short, long, required = true)]
    file: String,

    /// Executes a given cache profiling target flow.
    #[arg(short)]
    error_target: Option<u64>,

    /// Executes a given cache profiling target flow.
    #[arg(short)]
    cache_target: Option<u64>,
}

fn main() {
    let args = ProgramArgs::parse();
    let _mmap: bool = args.mmap;
    let _verbose: bool = args.verbose;
    let _debug: bool = args.debug;

    // Set BRUIJNX_VERBOSE environment varioble (if = "1" then program operates in verbose mode
    // else program operates in non-verbose mode).
    unsafe { std::env::set_var("BRUIJNX_VERBOSE", if args.verbose { "1" } else { "0" }) };

    if BITS < BITS_U64 {
        panic!("error program can't operate on 64-bit systems");
    }

    #[cfg(feature = "bench")]
    if let Some(error_target) = args.error_target {
        if error_target == 0 {
            println!("proceeding into error_target {error_target}");
            hyperball_profile::<(), (), usize, _>(
                args.file.clone(),
                args.threads,
                args.output_id.clone(),
            )
            .expect("hyperball profile should succeed");
        } else if error_target == 1 {
            hk_relax_profile::<(), (), usize, _>(
                args.file.clone(),
                args.threads,
                args.output_id.clone(),
            )
            .expect("hk-relax profile should succeed");
        } else {
            general_profile(args.file.clone(), args.threads)
                .expect("general profile should succeed");
        }
        return;
    }
    if let Some(cache_target) = args.cache_target {
        println!("going into mem_target {cache_target}");
        match cache_target {
            0 => {
                cache_profile_0::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 0 shouldn't fail");
            }
            1 => {
                cache_profile_1::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 1 shouldn't fail");
            }
            2 => {
                cache_profile_2::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 2 shouldn't fail");
            }
            3 => {
                pages_profile_0::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 3 shouldn't fail");
            }
            4 => {
                pages_profile_1::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 4 shouldn't fail");
            }
            5 => {
                pages_profile_2::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 5 shouldn't fail");
            }
            6 => {
                pages_profile_3::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 6 shouldn't fail");
            }
            7 => {
                pages_profile_4::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 7 shouldn't fail");
            }
            8 => {
                pages_profile_5::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 8 shouldn't fail");
            }
            9 => {
                pages_profile_6::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 9 shouldn't fail");
            }
            _ => {
                pages_profile_7::<(), (), usize, _>(
                    args.file.clone(),
                    args.threads,
                    args.output_id,
                )
                .expect("target 10 shouldn't fail");
            }
        }
    } else {
        sandbox_parse::<(), (), usize, _>(args.file.clone(), args.threads, args.output_id)
            .expect("error couldn't parse file");
    }
}

// Experimentar simples, depois rustworkx, depois métodos mais eficientes
#[expect(dead_code)]
fn sandbox_open<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    data_path: String,
    threads: Option<u8>,
) -> Result<GraphMemoryMap<N, E, Ix>, Box<dyn std::error::Error>> {
    let graph_mmaped = GraphMemoryMap::<N, E, Ix>::open(&data_path, None, threads)?;

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */
    metalabel_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    Ok(graph_mmaped)
}

fn metalabel_search<N: graph::N, E: graph::E, Ix: graph::IndexType>(
    graph: &GraphMemoryMap<N, E, Ix>,
) {
    loop {
        print!("Enter something (press <ENTER> to quit): ");
        std::io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        match graph.node_id_from_metalabel(input) {
            Ok(val) => println!("Value for key {} is {}", input, val),
            Err(e) => println!("Key not found: {e}"),
        }
    }
}

fn sandbox_parse<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<GraphMemoryMap<N, E, Ix>, Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!("metadata:\n{}", graph_mmaped.metadata()?);

    let time = Instant::now();
    let mut _burkhardt_et_al = AlgoBurkhardtEtAl::new(&graph_mmaped)?;
    println!("k-truss burkhardt et al {:?}", time.elapsed());
    println!();
    _burkhardt_et_al.drop_cache()?;

    // let time = Instant::now();
    // let mut _pkt = AlgoPKT::new(&graph_mmaped)?;
    // println!("k-truss pkt {:?}", time.elapsed());
    // println!();
    // _pkt.drop_cache()?;

    // for i in 0..1 {
    //     let time = Instant::now();
    //     let mut _pkt = AlgoPKT::new(&graph_mmaped)?;
    //     for (idx, t) in _pkt.k_trusses().iter().enumerate() {
    //         if *t != b[idx] {
    //             println!("{idx} should be {} but is {t}", b[idx]);
    //         }
    //     }
    //     println!("k-truss pkt {:?}", time.elapsed());
    //     println!();
    //     _pkt.drop_cache()?;
    // }

    //
    // let time = Instant::now();
    // let mut _bz = AlgoBatageljZaversnik::new(&graph_mmaped)?;
    // println!("k-core batagelj zaversnik {:?}", time.elapsed());
    // _bz.drop_cache()?;

    // let time = Instant::now();
    // let mut _liu_et_al = AlgoLiuEtAl::new(&graph_mmaped)?;
    // println!("k-core liu et al {:?}", time.elapsed());
    // println!();
    // _liu_et_al.drop_cache()?;

    // let time = Instant::now();
    // let mut _louvain = AlgoGVELouvain::new(&graph_mmaped)?;
    // println!("found {} communities", _louvain.community_count());
    // println!("partition modularity {} ", _louvain.partition_modularity());
    // println!("louvain finished in {:?}", time.elapsed());
    // println!(
    //     "partition modularity according to graph_mmaped method {}",
    //     graph_mmaped.modularity(_louvain.communities(), _louvain.community_count())?
    // );
    // _louvain.drop_cache()?;
    // println!();
    //
    // let time = Instant::now();
    // let mut hyperball = HyperBallInner::<_, _, _, Precision6, 6>::new(&graph_mmaped)?;
    // println!("hyperball {:?}", time.elapsed());
    // hyperball.drop_cache()?;

    // _louvain.coalesce_isolated_nodes()?;
    // println!("found {} communities", _louvain.community_count());
    // println!("partition modularity {} ", _louvain.partition_modularity());
    // println!("louvain finished in {:?}", time.elapsed());
    // println!(
    //     "partition modularity according to graph_mmaped method {}",
    //     graph_mmaped.modularity(_louvain.communities(), _louvain.community_count())?
    // );
    // println!();
    // _louvain.drop_cache()?;

    // let time = Instant::now();
    // graph_cache.rebuild_fst_from_ggcat_file(path, None, None)?;
    // println!("cache fst rebuilt {:?}", time.elapsed());

    // let time = Instant::now();
    // let graph_mmaped: GraphMemoryMap<N, E, Ix> =
    //     GraphMemoryMap::<EdgeType, Edge>::init(graph_cache.clone(), threads)?;
    // println!("graph initialized {:?}", time.elapsed());
    // label_search(&graph_mmaped);

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */

    // metalabel_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */
    //

    // let e_fn = cache_file_name(
    //     &graph_mmaped.cache_index_filename(),
    //     FileType::ExactClosenessCentrality(H::H),
    //     None,
    // )?;
    // let node_count = graph_mmaped.size();
    // println!("compute closeness centrality {e_fn}",);
    // // let file = File::open(&e_fn)?;
    // // let vals = unsafe { MmapOptions::new().map(&file)? };
    // // let mut z = 0;
    // // for i in 0..node_count {
    // //     let val = unsafe { (vals.as_ptr() as *const f64).add(i).read() };
    // //     if val == 0. {
    // //         z += 1;
    // //     }
    // // }
    // // println!("found {z} zeroes");
    // let mut e = SharedSliceMut::<f64>::abst_mem_mut(&e_fn, node_count, true)?;
    // for u in 0..node_count {
    //     if u % 1000 == 0 {
    //         println!("reached {u} of {node_count}");
    //     }
    //     let bfs = BFSDists::new(&graph_mmaped, u)?;
    //     if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
    //         println!("found isolated node at {u}");
    //         *e.get_mut(u) = 0.0;
    //     } else {
    //         *e.get_mut(u) = bfs.recheable() as f64 / bfs.total_distances();
    //     }
    // }
    // e.flush()?;
    // let mut i = 0;
    // loop {
    //     if i >= graph_mmaped.size() {
    //         break;
    //     }
    //     let time = Instant::now();
    //     let mut bfs = BFSDists::new(&graph_mmaped, i)?;
    //     println!(
    //         "source {i} -> bfs reacheable {} total_dist {}",
    //         bfs.recheable(),
    //         bfs.total_distances()
    //     );
    //     println!("bfs computed in {:?}", time.elapsed());
    //     println!();
    //     i += 10340;
    //     bfs.drop_cache()?;
    // }

    // let time = Instant::now();
    // let mut _dfs = DFS::new(&graph_mmaped, 0)?;
    // println!("dfs computed in {:?}", time.elapsed());
    //
    // let time = Instant::now();
    // let mut _euler_trail = AlgoHierholzer::new(&graph_mmaped)?;
    // println!("found {} euler trails", _euler_trail.trail_number());
    // println!("euler trail built {:?}", time.elapsed());
    // println!();
    // _euler_trail.drop_cache()?;

    // let time = Instant::now();
    // hyperball.compute_harmonic_centrality(None)?;
    // println!("harmonic centrality {:?}", time.elapsed());
    // println!();
    //

    //
    //
    // let time = Instant::now();
    // let conductivity = ClusteringCoefficient::new(&graph_mmaped)?;
    // println!(
    //     "graph transitivity {:?}",
    //     conductivity.get_graph_transitivity()
    // );
    // println!(
    //     "average clustering coefficient {:?}",
    //     conductivity.get_average_clustering_coefficient()
    // );
    // println!("clustering coefficient finished in {:?}", time.elapsed());
    // println!();

    // let g = GraphMemoryMap::<N, E, Ix>::open(
    //     "./.cache/metadata_5013817c269e634c07b3adc5178df93b89f991de411b2cf94e6a3d33b80f063f.toml",
    //     None,
    //     Some(32),
    // )?;

    // println!("graph metadata is {:?}", g.metadata()?);
    // drop(g);

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    // let mut i = 0;
    // while i < graph_mmaped.size().map_or(0, |s| s) {
    //     let time = Instant::now();
    //     let _approx_dirichlet_hkpr =
    //         ApproxDirHKPR::new(&graph_mmaped, 0.008, i, 100000, 4000000, 0.05)?;
    //     let _ = _approx_dirichlet_hkpr.compute()?;
    //     println!("ApproxDirichletHeatKernelK {:?}", time.elapsed());
    //     i += 12934600;
    // }
    //
    // let b = graph_mmaped.apply_mask_to_nodes(|u| -> bool { u % 2 == 0 }, Some("evennodes"))?;
    // println!("graph even nodes {:?}", b);

    // let time = Instant::now();
    // let a = graph_mmaped.export_petgraph_stripped()?;
    // println!("rustworkx_core export {:?} {:?}", a, time.elapsed());
    // let time = Instant::now();
    // use rustworkx_core::centrality::betweenness_centrality;
    // println!(
    //     "rustworkx_core export {:?} {:?}",
    //     betweenness_centrality(&a, false, true, 50),
    //     time.elapsed()
    // );
    Ok(graph_mmaped)
}

fn cache_profile_0<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _pkt = AlgoPKT::new(&graph_mmaped)?;
    println!("k-truss pkt {:?}", time.elapsed());
    println!();
    _pkt.drop_cache()?;

    let time = Instant::now();
    let mut _burkhardt_et_al = AlgoBurkhardtEtAl::new(&graph_mmaped)?;
    println!("k-truss burkhardt et al {:?}", time.elapsed());
    println!();
    _burkhardt_et_al.drop_cache()?;

    let time = Instant::now();
    let mut _bz = AlgoBatageljZaversnik::new(&graph_mmaped)?;
    println!("k-core batagelj zaversnik {:?}", time.elapsed());
    _bz.drop_cache()?;

    let time = Instant::now();
    let mut _liu_et_al = AlgoLiuEtAl::new(&graph_mmaped)?;
    println!("k-core liu et al {:?}", time.elapsed());
    println!();
    _liu_et_al.drop_cache()?;

    let time = Instant::now();
    let mut _louvain = AlgoGVELouvain::new(&graph_mmaped)?;
    println!("found {} communities", _louvain.community_count());
    println!("partition modularity {} ", _louvain.partition_modularity());
    println!("louvain finished in {:?}", time.elapsed());
    println!(
        "partition modularity according to graph_mmaped method {}",
        graph_mmaped.modularity(_louvain.communities(), _louvain.community_count())?
    );
    println!();
    _louvain.drop_cache()?;

    let time = Instant::now();
    let mut hyperball = HyperBallInner::<_, _, _, Precision6, 6>::new(&graph_mmaped)?;
    println!("hyperball {:?}", time.elapsed());
    hyperball.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn cache_profile_1<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _hierholzers = AlgoHierholzer::new(&graph_mmaped)?;
    println!(
        "found {} hierholzer's euler trails",
        _hierholzers.trail_number()
    );
    println!("hierholzer's euler trail built {:?}", time.elapsed());
    println!();
    _hierholzers.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn cache_profile_2<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let mut i = 0;
    let size = graph_mmaped.size();
    while i < size {
        let time = Instant::now();
        let hk_relax = HKRelax::new(&graph_mmaped, 45., 0.01, vec![i], None, None)?;
        let _ = hk_relax.compute()?;
        println!("HKRelax {:?}", time.elapsed());
        i += size / 50;
    }

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_0<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_1<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _hierholzers = AlgoHierholzer::new(&graph_mmaped)?;
    println!(
        "found {} hierholzer's euler trails",
        _hierholzers.trail_number()
    );
    println!("hierholzer's euler trail built {:?}", time.elapsed());
    println!();
    _hierholzers.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_2<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _bz = AlgoBatageljZaversnik::new(&graph_mmaped)?;
    println!("k-core batagelj zaversnik {:?}", time.elapsed());
    _bz.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_3<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _liu_et_al = AlgoLiuEtAl::new(&graph_mmaped)?;
    println!("k-core liu et al {:?}", time.elapsed());
    println!();
    _liu_et_al.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_4<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _burkhardt_et_al = AlgoBurkhardtEtAl::new(&graph_mmaped)?;
    println!("k-truss burkhardt et al {:?}", time.elapsed());
    println!();
    _burkhardt_et_al.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_5<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _pkt = AlgoPKT::new(&graph_mmaped)?;
    println!("k-truss pkt {:?}", time.elapsed());
    println!();
    _pkt.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_6<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _louvain = AlgoGVELouvain::new(&graph_mmaped)?;
    println!("found {} communities", _louvain.community_count());
    println!("partition modularity {} ", _louvain.partition_modularity());
    println!("louvain finished in {:?}", time.elapsed());
    println!(
        "partition modularity according to graph_mmaped method {}",
        graph_mmaped.modularity(_louvain.communities(), _louvain.community_count())?
    );
    println!();
    _louvain.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

fn pages_profile_7<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut hyperball = HyperBallInner::<_, _, _, Precision6, 6>::new(&graph_mmaped)?;
    println!("hyperball {:?}", time.elapsed());
    hyperball.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

#[cfg(feature = "bench")]
fn hyperball_profile<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion

    use sparkx::{
        shared_slice::AbstractedProceduralMemory, test_common::get_or_init_dataset_exact_closeness,
        utils,
    };
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;

    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let avg = [1];
    let mut avg_fn = vec![];
    let mut avg_arr = vec![];
    for i in avg {
        let h_fn = graph_mmaped.build_cache_filename(graph::CacheFile::General, Some(i))?;
        avg_fn.push(h_fn.clone());
        avg_arr.push(SharedSliceMut::<f64>::abst_mem_mut(
            &h_fn,
            graph_mmaped.size(),
            true,
        )?);
    }

    fn bucket_for(i: usize) -> usize {
        match i {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=4 => 3,
            5..=9 => 4,
            10..=19 => 5,
            20..=29 => 6,
            30..=39 => 7,
            40..=49 => 8,
            50..=59 => 9,
            60..=69 => 10,
            70..=79 => 11,
            80..=89 => 12,
            _ => 13, // 90..=99
        }
    }

    // for i in 0..100 {
    let time = Instant::now();
    let mut hyperball = HyperBallInner::<_, _, _, Precision4, 6>::new(&graph_mmaped)?;
    println!("hyperball {:?}", time.elapsed());
    {
        let c = hyperball.compute_closeness_centrality(true)?;
        let b = bucket_for(0);
        (0..graph_mmaped.size()).for_each(|u| {
            *avg_arr[b].get_mut(u) += c[u];
        });
    }

    let e_fn = get_or_init_dataset_exact_closeness(path.as_ref(), &graph_mmaped)?;
    let exact = AbstractedProceduralMemory::<f64>::from_file_name(&e_fn)?;

    let mut zeroes = 0;
    (0..graph_mmaped.size()).for_each(|i| {
        if *exact.get(i) == 0. {
            zeroes += 1;
        }
    });
    println!("found {zeroes} zeroes");

    let e_mae = utils::mae(avg_arr[0].as_slice(), exact.as_slice());
    let e_mape = utils::mape(avg_arr[0].as_slice(), exact.as_slice());
    let rho = utils::spearman_rho(avg_arr[0].as_slice(), exact.as_slice());
    println!("{e_mae} | {e_mape} | {rho}");

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

#[cfg(feature = "bench")]
fn hk_relax_profile<N: graph::N, E: graph::E, Ix: graph::IndexType, P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion

    use rand::Rng;
    use std::path::PathBuf;

    let out_file: PathBuf = "results/hkpr_all.csv".into();
    let dataset = path
        .as_ref()
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error couldn.t get filename for {:?}", path.as_ref()).into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error couldn.t get filename strfor {:?}", path.as_ref()).into()
        })?;
    let runs_per_params = 10000;

    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<N, E, Ix> =
        GraphMemoryMap::<N, E, Ix>::from_file(path.as_ref(), id, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );

    let mut conds = vec![];
    let mut results = vec![];
    for (t, eps) in [
        (1., 0.01),
        (5., 0.01),
        (10., 0.01),
        (15., 0.01),
        (20., 0.01),
        (25., 0.01),
        (30., 0.01),
        (35., 0.01),
        (40., 0.01),
        (1., 0.001),
        (5., 0.001),
        (10., 0.001),
        (15., 0.001),
        (20., 0.001),
        (25., 0.001),
        (30., 0.001),
        (35., 0.001),
        (40., 0.001),
        (1., 0.0001),
        (5., 0.0001),
        (10., 0.0001),
        (15., 0.0001),
        (20., 0.0001),
        (25., 0.0001),
        (30., 0.0001),
        (35., 0.0001),
        (40., 0.0001),
        (1., 0.00001),
        (5., 0.00001),
        (10., 0.00001),
        (15., 0.00001),
        (20., 0.00001),
        (25., 0.00001),
        (30., 0.00001),
        (35., 0.00001),
        (40., 0.00001),
        (1., 0.000001),
        (5., 0.000001),
        (10., 0.000001),
        (15., 0.000001),
        (20., 0.000001),
        (25., 0.000001),
        (30., 0.000001),
        (35., 0.000001),
        (40., 0.000001),
    ] {
        println!("{t} :: {eps}");
        let mut i = 0;
        loop {
            if i == runs_per_params {
                break;
            }
            let s = rand::rng().random_range(0..graph_mmaped.size());

            let hk_relax = HKRelax::new(&graph_mmaped, t, eps, vec![s], None, None)?;
            let time = Instant::now();
            let c = hk_relax.compute()?;
            let elapsed = time.elapsed();
            if c.size < 10 || c.width < 10 {
                continue;
            }
            results.push(HkprRecord {
                dataset,
                t,
                eps,
                seed: s as u64,
                conductance: c.conductance,
                cluster_size: c.size as u64,
                volume: c.width as u64,
                runtime_secs: elapsed.as_micros(),
            });
            println!("{i} ---> {}", c.conductance);
            conds.push(c.conductance);
            i += 1;
        }
    }

    append_records(&out_file, &results)?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    Ok(())
}

#[cfg(feature = "bench")]
pub fn general_profile<P: AsRef<Path>>(
    path: P,
    threads: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    static RUNS: usize = 15;
    let dataset = filename_only(&path)?;
    let time = Instant::now();
    let mut graph: GraphMemoryMap<(), (), usize> =
        GraphMemoryMap::<(), (), usize>::from_file(path.as_ref(), None, threads)?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph.size(),
        graph.width(),
        time.elapsed()
    );
    let threads = threads.unwrap_or(1).max(1) as usize;

    // 1) K-core — Batagelj–Zaversnik (50 runs)
    for _ in 0..RUNS {
        let t = Instant::now();
        AlgoBatageljZaversnik::new(&graph)?;
        let us = t.elapsed().as_micros();
        append_kcore(
            &KCoreRecord {
                dataset,
                algo: "batagelj-zaversnik",
                runtime_micros: us,
                threads,
            },
            defaults::KCORE,
        )?;
    }

    // // 2) K-core — Liu et al. (51 runs)
    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     AlgoLiuEtAl::new(&graph)?;
    //     let us = t.elapsed().as_micros();
    //     append_kcore(
    //         &KCoreRecord {
    //             dataset,
    //             algo: "liu",
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::KCORE,
    //     )?;
    // }

    // 3) K-truss — Burkhardt (50 runs)
    for _ in 0..RUNS {
        let t = Instant::now();
        AlgoBurkhardtEtAl::new(&graph)?;
        let us = t.elapsed().as_micros();
        append_ktruss(
            &KTrussRecord {
                dataset,
                algo: "burkhardt",
                runtime_micros: us,
                threads,
            },
            defaults::KTRUSS,
        )?;
    }

    // // 4) K-truss — PKT (50 runs)
    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     AlgoPKT::new(&graph)?;
    //     let us = t.elapsed().as_micros();
    //     append_ktruss(
    //         &KTrussRecord {
    //             dataset,
    //             algo: "pkt",
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::KTRUSS,
    //     )?;
    // }

    // // 5) Louvain (50 runs) — summary + per-pass rows (linked by run_id)
    // for _ in 0..RUNS {
    //     let run_id = louvain_new_run_id();
    //     let mut pass_rows: Vec<LouvainPassRecord> = Vec::new();
    //     let total_start = Instant::now();
    //     let mut pass_idx: u32 = 0;
    //
    //     let l = AlgoGVELouvain::new(&graph)?;
    //     let total_us = total_start.elapsed().as_micros();
    //     for &(iters, coms, elapsed) in l.get_iters().iter() {
    //         pass_rows.push(LouvainPassRecord {
    //             run_id,
    //             pass_idx,
    //             iters_for_pass: iters,
    //             coms_in_pass: coms,
    //             runtime_micros: elapsed,
    //         });
    //         pass_idx += 1;
    //     }
    //
    //     append_louvain_summary(
    //         &LouvainSummaryRecord {
    //             run_id,
    //             dataset,
    //             modularity: l.partition_modularity(),
    //             runtime_micros: total_us,
    //             levels: l.get_iters().len(),
    //             passes_total: Some(pass_idx),
    //             threads,
    //         },
    //         defaults::LOUV_SUM,
    //     )?;
    //     append_louvain_passes_bulk(&pass_rows, defaults::LOUV_PASS)?;
    // }

    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     let h = HyperBall4::new(&graph)?;
    //     let us = t.elapsed().as_micros();
    //     append_hyperball(
    //         &HyperBallRecord {
    //             dataset,
    //             precision_p: 4,
    //             iterations: h.get_iters(),
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::HYPERBALL,
    //     )?;
    // }
    //
    // println!("going for hyp 6");
    //
    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     let h = HyperBall6::new(&graph)?;
    //     let us = t.elapsed().as_micros();
    //     println!("+1");
    //     append_hyperball(
    //         &HyperBallRecord {
    //             dataset,
    //             precision_p: 6,
    //             iterations: h.get_iters(),
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::HYPERBALL,
    //     )?;
    // }
    // println!("going for hyp 8");
    //
    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     let h = HyperBall8::new(&graph).unwrap();
    //     println!("+1");
    //     let us = t.elapsed().as_micros();
    //     append_hyperball(
    //         &HyperBallRecord {
    //             dataset,
    //             precision_p: 8,
    //             iterations: h.get_iters(),
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::HYPERBALL,
    //     )?;
    // }
    // println!("going for hyp 10");
    //
    // for _ in 0..RUNS {
    //     let t = Instant::now();
    //     let h = HyperBall10::new(&graph).unwrap();
    //     let us = t.elapsed().as_micros();
    //     append_hyperball(
    //         &HyperBallRecord {
    //             dataset,
    //             precision_p: 10,
    //             iterations: h.get_iters(),
    //             runtime_micros: us,
    //             threads,
    //         },
    //         defaults::HYPERBALL,
    //     )?;
    //     println!("+1");
    // }

    println!("droping");
    graph.drop_cache()?;
    println!("dropped");

    Ok(())
}

/*
 * Helper Structs & Functions
 * */

#[derive(Debug)]
struct HkprRecord<'a> {
    dataset: &'a str,
    t: f64,
    eps: f64,
    seed: u64,
    conductance: f64,
    cluster_size: u64,
    volume: u64,
    runtime_secs: u128,
}

#[derive(Debug)]
pub struct KCoreRecord<'a> {
    pub dataset: &'a str,
    pub algo: &'a str,
    pub runtime_micros: u128,
    pub threads: usize,
}

#[derive(Debug)]
pub struct KTrussRecord<'a> {
    pub dataset: &'a str,
    pub algo: &'a str,
    pub runtime_micros: u128,
    pub threads: usize,
}

#[derive(Debug)]
pub struct LouvainSummaryRecord<'a> {
    pub run_id: u128,
    pub dataset: &'a str,
    pub modularity: f64,
    pub runtime_micros: u128,
    // Optional extras:
    pub levels: usize,
    pub passes_total: Option<u32>,
    pub threads: usize,
}

#[derive(Debug)]
pub struct LouvainPassRecord {
    pub run_id: u128,
    pub pass_idx: u32,
    pub iters_for_pass: usize,
    pub coms_in_pass: usize,
    pub runtime_micros: u128,
}

#[derive(Debug)]
pub struct HyperBallRecord<'a> {
    pub dataset: &'a str,
    pub precision_p: u32,
    pub iterations: usize,
    pub runtime_micros: u128,
    pub threads: usize,
}

/// Escape a field for CSV according to RFC 4180-ish rules.
/// If it contains a comma, quote, or newline, wrap in quotes and double internal quotes.
fn csv_escape(field: &str) -> String {
    let needs_quotes =
        field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r');
    if !needs_quotes {
        return field.to_string();
    }
    let mut s = String::with_capacity(field.len() + 2);
    s.push('"');
    for ch in field.chars() {
        if ch == '"' {
            s.push('"'); // double the quote
        }
        s.push(ch);
    }
    s.push('"');
    s
}

/// Convert a record to a CSV line (no trailing newline).
fn record_to_csv_line(rec: &HkprRecord<'_>) -> String {
    // Numeric fields don’t need escaping.
    // For dataset we call csv_escape in case it has commas/spaces/etc.
    format!(
        "{},{},{},{},{},{},{},{}",
        csv_escape(rec.dataset),
        rec.t,
        rec.eps,
        rec.seed,
        rec.conductance,
        rec.cluster_size,
        rec.volume,
        rec.runtime_secs
    )
}

/// Append header if file is empty; then append one line.
fn _append_record(csv_path: impl AsRef<Path>, rec: &HkprRecord<'_>) -> io::Result<()> {
    let path = csv_path.as_ref();
    if let Some(dir) = path.parent() {
        create_dir_all(dir)?;
    }

    // Detect empty file (or non-existent).
    let is_empty = match metadata(path) {
        Ok(m) => m.len() == 0,
        Err(_) => true,
    };

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    if is_empty {
        // Write header
        writeln!(
            file,
            "dataset,t,eps,seed,conductance,cluster_size,volume,runtime_secs"
        )?;
    }

    // Write the record line
    writeln!(file, "{}", record_to_csv_line(rec))?;
    Ok(())
}

/// Append many records efficiently (header once if needed).
fn append_records(csv_path: impl AsRef<Path>, recs: &[HkprRecord<'_>]) -> io::Result<()> {
    let path = csv_path.as_ref();
    if let Some(dir) = path.parent() {
        create_dir_all(dir)?;
    }
    let is_empty = match metadata(path) {
        Ok(m) => m.len() == 0,
        Err(_) => true,
    };

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    if is_empty {
        writeln!(
            file,
            "dataset,t,eps,seed,conductance,cluster_size,volume,runtime_secs"
        )?;
    }

    for rec in recs {
        writeln!(file, "{}", record_to_csv_line(rec))?;
    }
    Ok(())
}

fn ensure_header_and_open(path: &Path, header: &str) -> io::Result<std::fs::File> {
    if let Some(dir) = path.parent() {
        create_dir_all(dir)?;
    }
    let is_empty = match metadata(path) {
        Ok(m) => m.len() == 0,
        Err(_) => true,
    };
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if is_empty {
        writeln!(file, "{header}")?;
    }
    Ok(file)
}

pub fn append_kcore(rec: &KCoreRecord<'_>, csv_path: impl AsRef<Path>) -> io::Result<()> {
    let path = csv_path.as_ref();
    let header = "dataset,algo,runtime_micros,threads";
    let mut f = ensure_header_and_open(path, header)?;
    writeln!(
        f,
        "{}, {}, {}, {}",
        csv_escape(rec.dataset),
        csv_escape(rec.algo),
        rec.runtime_micros,
        rec.threads,
    )?;
    Ok(())
}

pub fn append_ktruss(rec: &KTrussRecord<'_>, csv_path: impl AsRef<Path>) -> io::Result<()> {
    let path = csv_path.as_ref();
    let header = "dataset,algo,runtime_micros,threads";
    let mut f = ensure_header_and_open(path, header)?;
    writeln!(
        f,
        "{}, {}, {}, {}",
        csv_escape(rec.dataset),
        csv_escape(rec.algo),
        rec.runtime_micros,
        rec.threads,
    )?;
    Ok(())
}

pub fn append_louvain_summary(
    rec: &LouvainSummaryRecord<'_>,
    csv_path: impl AsRef<Path>,
) -> io::Result<()> {
    let path = csv_path.as_ref();
    let header = "run_id,dataset,modularity,runtime_micros,levels,passes_total,threads";
    let mut f = ensure_header_and_open(path, header)?;
    writeln!(
        f,
        "{},{},{},{},{},{},{}",
        rec.run_id,
        csv_escape(rec.dataset),
        rec.modularity,
        rec.runtime_micros,
        rec.levels,
        rec.passes_total.map(|x| x.to_string()).unwrap_or_default(),
        rec.threads,
    )?;
    Ok(())
}

pub fn append_louvain_pass(rec: &LouvainPassRecord, csv_path: impl AsRef<Path>) -> io::Result<()> {
    let path = csv_path.as_ref();
    let header = "run_id,pass_idx,iters_for_pass,coms_in_pass,runtime_micros";
    let mut f = ensure_header_and_open(path, header)?;
    writeln!(
        f,
        "{},{},{},{},{}",
        rec.run_id, rec.pass_idx, rec.iters_for_pass, rec.coms_in_pass, rec.runtime_micros
    )?;
    Ok(())
}

pub fn append_louvain_passes_bulk(
    recs: &[LouvainPassRecord],
    csv_path: impl AsRef<Path>,
) -> io::Result<()> {
    if recs.is_empty() {
        return Ok(());
    }
    let path = csv_path.as_ref();
    let header = "run_id,pass_idx,iters_for_pass,coms_in_pass,runtime_micros";
    let mut f = ensure_header_and_open(path, header)?;
    for r in recs {
        writeln!(
            f,
            "{},{},{},{},{}",
            r.run_id, r.pass_idx, r.iters_for_pass, r.coms_in_pass, r.runtime_micros
        )?;
    }
    Ok(())
}

pub fn append_hyperball(rec: &HyperBallRecord<'_>, csv_path: impl AsRef<Path>) -> io::Result<()> {
    let path = csv_path.as_ref();
    let header = "dataset,precision_p,iterations,runtime_micros,threads";
    let mut f = ensure_header_and_open(path, header)?;
    writeln!(
        f,
        "{},{},{},{},{}",
        csv_escape(rec.dataset),
        rec.precision_p,
        rec.iterations,
        rec.runtime_micros,
        rec.threads,
    )?;
    Ok(())
}

pub mod defaults {
    pub const KCORE: &str = "results/kcore.csv";
    pub const KTRUSS: &str = "results/ktruss.csv";
    pub const LOUV_SUM: &str = "results/louvain.csv";
    pub const LOUV_PASS: &str = "results/louvain_passes.csv";
    pub const HYPERBALL: &str = "results/hyperball.csv";
}

fn next_run_id() -> u128 {
    static mut CTR: u64 = 0;
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u128;
    let pid = std::process::id() as u128;
    let c = unsafe {
        CTR = CTR.wrapping_add(1);
        CTR as u128
    };
    (secs << 32) ^ (pid << 16) ^ c
}

pub fn louvain_new_run_id() -> u128 {
    next_run_id()
}

// Small helper to get a &str dataset label from a path
fn filename_only<P: AsRef<Path>>(p: P) -> Result<&'static str, Box<dyn std::error::Error>> {
    // SAFETY: We return a leaked &'static str to avoid lifetime plumbing.
    // If you prefer no leak, thread the String around instead.
    let name = p
        .as_ref()
        .file_name()
        .ok_or("missing filename")?
        .to_str()
        .ok_or("filename not UTF-8")?;
    Ok(Box::leak(name.to_string().into_boxed_str()))
}

#[derive(Debug)]
enum ParsingError<T: std::error::Error> {
    Some(T),
}

impl<T: std::error::Error> From<T> for ParsingError<T> {
    fn from(e: T) -> Self {
        ParsingError::<T>::Some(e)
    }
}

impl<T: std::error::Error> Display for ParsingError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsingError::Some(e) => write!(f, "Other {{{}}}", e),
        }
    }
}
