# SparX

High-performance **Rust** library for large-scale **static sparse graphs analytics** on commodity hardware. Built on a memory-mapped core. It includes implementations of:

- **k-core decomposition** (Batagelj–Zaversnik sequential; Liu et al. parallel)
- **k-truss decomposition** (Burkhardt et al. sequential; PKT parallel)
- **Louvain** (GVE-Louvain variant)
- **HyperBall** (HLL++ distance/centrality estimation)
- **Heat Kernel PageRank** (HK-relax --- SSPR/MSPR)
- Classic traversals/utilities (BFS/DFS, Hierholzer’s Euler trails, clustering coefficient, transitivity, triangle counting, etc.)

Comes with CSV **benchmark writers** and **plotting scripts** to visualize runtime scaling versus graph size.

---

## Table of contents

- [Features](#features)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Data formats](#data-formats)
- [Quick start](#quick-start)
  - [Manually Linking *SparX*](#manually-linking-sparx)
  - [Load & inspect](#loading--inspecting-a-graph)
  - [Run algorithms](#running-algorithms)
  - [Narrow index types & accumulators](#narrow-index-types--accumulators)
  - [Configurable edge/community weight precision](#configurable-edgecommunity-weight-precision)
- [Reproducibility](#reproducibility)
  - [Memory Benchmarks](#memory-benchmarks)
  - [Wall-Time Benchmarks](#wall-time-benchmarks)

---

## Features

- Zero-copy, **memory-mapped** graph store: `GraphMemoryMap<N, E, Ix>`
- Pluggable index types (`usize`/`u32`/`u16`/`u8`) and optional node/edge labels
- **Narrow-`Ix` throughout**: every algorithm's internal scratch arrays (not just the graph's own
  CSR storage) use the graph's chosen `Ix` width, so picking e.g. `u32` on a sub-4-billion-edge
  graph reduces memory traffic for every traversal, decomposition, and centrality in the crate,
  not just storage size
- **Configurable accumulator width** during parsing (`AccumulatorAtomic`: `AtomicUsize`/`AtomicU32`/
  `AtomicU16`/`AtomicU8`), independent of the final `Ix` --- lets the (hot, streamed-through-once)
  degree/edge-position accumulation itself run narrower, not just the resulting CSR
- Cache-entry **`Ix` conversion**: re-encode an existing cache at a different index width without
  re-parsing the original dataset
- **Configurable edge/community weight precision** (`Weight`: `f32`/`f64`) for GVE-Louvain,
  HyperBall, and the exact centralities, independent of `Ix`
- Cache files + rich **metadata** (`Display` implemented)
- Algorithms expose summaries and can **drop** local caches
- Minimal, dependency-free **CSV appenders** to `results/`
- Ready-to-use **plotting** scripts (with Matplotlib) for reproducibility

---

## Pre-requisites

- **Rust ≥ 1.87**

If you don’t already have Rust installed or need to update to at least version 1.87, the recommended way is via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installing Rust, restart your shell and verify:

```bash
rustc --version
```

If the installation was successful, a version number ≥ 1.87 should be printed.

## Installation

To install the *SparX* library the following steps must be followed.
Firstly, clone the library's GitHub repository (if you want to use the [OOTB Datasets](#ootb-datasets) made available via `build_datasets.sh` it is advisable you clone `sparx` to the same directory you have cloned `GGCAT` into if you have previously done so):

```bash
git clone git@github.com:alexfaisca/sparx.git
```

In macOS, in the first installation, after cloning the repository, Apple's tool licensing must be accepted. You may do so by running:

```bash
sudo xcodebuild -license
```

After this, you may change directory into the repository and build the library (recommended in release mode):

```bash
cd sparx
cargo build --release
```

The library has an assortment of non-default features. As an example, `feature = "rayon"` may be enabled by running the following command:

```bash
cargo build --features rayon --release
```

# Data Formats

Input: **GGCAT Output Files** (`.lz4`), **Matrix Market** (`.mtx`) and cached internal formats (auto-created on first load).

## OOTB Datasets

A script to build datasets out-of-the-box is provided for reproducibility and sandboxing purposes: `build_datasets.sh`. It builds all the datasets used for testing and benchmarking the library. It may be run in the following manner:

```bash
bash build_datasets.sh
```

It is possible to benchmark the library with datasets outside of those built with `build_datasets.sh`, to guarantee all benching related tools', e.g. plotting scripts, preconditions entry consisting of `<dataset_name>,<V>,<E>` should be added to `datasets_size.csv` (`<dataset_name>` should be the dataset's filename without file extension):

```csv
dataset,V,E
kmer_V1r,222222,3333333
kmer_V2a,55042369,117217600
...
my_custom_dataset,123123,123123123
```

# Quick Start

To follow this guide it is advisable you first run the `build_datasets.sh` script. This way you will be able to follow along and use the provided code examples as they come.

## Manually Linking *SparX*

As *SparX* is not yet an indexed crate its usage requires manual linkage.

### Creating a Rust Project

To create a Rust project run:

```bash
cargo new myapp
cd myapp
```

This creates a folder `myapp/` with the usual Rust project layout.

```arduino
parentdir/
  myapp/           # new crate
    src/           # source code
    Cargo.toml     # cargo configuration
```

### Linking *SparX*

#### 1. Installing *SparX* directly from Git.

  To install *SparX* directly from Git, simply add it to the `[dependencies]` section in `myapp/Cargo.toml`:

```toml
 [dependencies]
  # commit or tag of the version you want to use should be in rev
  sparx = { git = "https://github.com/alexfaisca/sparx.git", rev = "v0.1.0" }
  # optionally, select features to be enabled:
  # sparx = { git = "...", tag = "v0.1.0", features = ["mtx","ggcat","rayon"] }
```

#### 2. Installing and linking *SparX* directly inside a Rust project.

  Install *SparX* as shown in [Installation](#installation) directly inside `myapp/`, after which, your project layout will look something like:

```arduino
parentdir/
  myapp/           # new crate
    sparx/         # sparx library
    src/           # source code
    Cargo.toml     # cargo configuration
```

  Define *SparX* as a member of your workspace and add it to your project's dependencies in `myapp/Cargo.toml`:

```toml
[workspace]
  members = [".", "sparx"]

[dependencies]
  sparx = { path = "./sparx" }
  # optionally, select features to be enabled:
  # sparx = { path = "./sparx", features = ["mtx","ggcat","rayon"] }
```


#### 3. Installing and linking *SparX* in a workspace.

  If your library already has a top-level `Cargo.toml` with a `[workspace]`, in similar layout to the one shown bellow:

 ```arduino
workspace/
  myapp/           # new crate
    src/           # source code
    Cargo.toml     # cargo configuration
  Cargo.toml       # workspace file
 ```

  Install *SparX* as shown in [Installation](#installation) next to `myapp/`, after which, your project layout will look something like:

 ```arduino
workspace/
  myapp/           # new crate
    src/           # source code
    Cargo.toml     # cargo configuration
  sparx/           # sparx library
  Cargo.toml       # workspace file
 ```

  Add *SparX* to the members of the workspace in `workspace/Cargo.toml`:

```toml
[workspace]
  members = ["myapp", "sparx"]
```

  And then, add it to your project's dependencies in `myapp/Cargo.toml`:

```toml
[dependencies]
  sparx = { path = "../sparx" }
  # optionally, select features to be enabled:
  # sparx = { path = "../sparx", features = ["mtx","ggcat","rayon"] }
```

## Loading & Inspecting a Graph

```rust
use sparx::graph::GraphMemoryMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // path to the dataset from which a graph is to be built
    let path = "./datasets/graphs/kmer_V2a.mtx";
    // build the graph --- graph is mutable so calling drop_cache() becomes possible
    let mut g: GraphMemoryMap<(), (), usize> = GraphMemoryMap::from_file(path, None, Some(32))?;
    println!("graph built (|V|={:?}, |E|={})", g.size(), g.width());
    println!("metadata:\n{}", g.metadata()?);
    // drop graph's cache entry
    g.drop_cache()?;
    Ok(())
}

```

## Running Algorithms

```rust
use sparx::centralities::hyper_ball::*;
use sparx::communities::gve_louvain::*;
use sparx::communities::hk_relax::*;
use sparx::graph;
use sparx::graph::{E, GraphMemoryMap, IndexType, N, label::VoidLabel};
use sparx::k_core::{batagelj_zaversnik::*, liu_et_al::*};
use sparx::k_truss::{burkhardt_et_al::*, clustering_coefficient::*, pkt::*};
use sparx::trails::hierholzer::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // path to the dataset from which a graph is to be built
    let path = "./datasets/graphs/kmer_V2a.mtx";
    // build the graph
    let mut g: GraphMemoryMap<(), (), usize> =
        GraphMemoryMap::from_file(path, None, Some(32))?;


    let mut pkt = AlgoPKT::new(&g)?;
    // get trussness of edge 0
    let kt = pkt.trussness(0);
    pkt.drop_cache()?;

    let mut burkhardt_et_al = AlgoBurkhardtEtAl::new(&g)?;
    // get trussness of all edges
    let kts = burkhardt_et_al.k_trusses();
    burkhardt_et_al.drop_cache()?;

    let mut bz = AlgoBatageljZaversnik::new(&g)?;
    // get coreness of node 0
    let c = bz.coreness(0);
    bz.drop_cache()?;

    let mut liu_et_al = AlgoLiuEtAl::new(&g)?;
    // get coreness of all nodes
    let cs = liu_et_al.k_cores();
    liu_et_al.drop_cache()?;

    let mut louvain = AlgoGVELouvain::new(&g)?;
    let number_communities_found = louvain.community_count();
    let partition_modularity = louvain.partition_modularity();
    // get community of node 0
    let com_0 = louvain.node_community(0);
    //get all communities
    let coms = louvain.communities();
    louvain.drop_cache()?;
    println!();

    // hyperball
    let mut hyperball = HyperBall4::new(&g)?;
    // get estimate of total distance from node 0 to all nodes in the graph
    let g_dist_0 = hyperball.get_geodesic_distance(0);
    // get estimates of total harmonic distances for all nodes in the graph
    let hg_dists = hyperball.harmonic_geodesic_distances();
    hyperball.drop_cache()?;
    // hyperball with manual precision and word-size configuration
    let mut hyperball_inner = HyperBallInner::<_, _, _, Precision6, 6>::new(&g)?;
    //get estimates of closeness centrality for all nodes in the graph, normalized by reacheable set
    let r_closeness = hyperball_inner.get_or_compute_closeness_centrality(true)?;
    //get estimates of closeness centrality for all nodes in the graph, normalized by graph size
    let t_harmonic = hyperball_inner.get_or_compute_harmonic_centrality(false)?;
    hyperball_inner.drop_cache()?;

    // drop graph's cache entry
    g.drop_cache()?;
    Ok(())
}

```

## Narrow Index Types & Accumulators

`GraphMemoryMap<N, E, Ix>`'s `Ix` type parameter (`usize`, `u32`, `u16`, or `u8`) controls the
width of *every* node/edge index stored throughout the crate --- not just the graph's own CSR
offsets/neighbours arrays, but every algorithm's internal scratch buffers too (community
assignments, coreness/trussness bookkeeping, traversal stacks/queues, Euler trail indices, etc.).
Choosing a narrower `Ix` that still comfortably fits the graph's node/edge counts reduces memory
traffic everywhere those buffers are streamed through.

```rust
use sparx::graph::GraphMemoryMap;
use portable_atomic::AtomicU32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "./datasets/graphs/kmer_V2a.mtx";

    // build directly with a narrower Ix (valid as long as the graph's node/edge counts fit
    // u32::MAX --- this is checked, and parsing fails loudly otherwise)
    let g: GraphMemoryMap<(), (), u32> = GraphMemoryMap::from_file(path, None, Some(32))?;

    // the width-matched accumulator used for the (hot, streamed-once) degree/edge-position
    // accumulation during parsing is independent of Ix, and defaults to AtomicUsize for safety
    // --- opt into a narrower one explicitly once you know it fits, for reduced memory traffic
    // during parsing itself:
    let g_fast: GraphMemoryMap<(), (), u32> =
        GraphMemoryMap::from_file_with_accumulator::<_, AtomicU32>(path, None, Some(32))?;

    // convert an already-built cache (any Ix) to a narrower one without re-parsing the original
    // dataset --- yields a new, independent cache entry; fails loudly (rather than silently
    // truncating) if the graph's node/edge counts don't actually fit the target width
    let g_default: GraphMemoryMap<(), (), usize> = GraphMemoryMap::from_file(path, None, Some(32))?;
    let g_u32: GraphMemoryMap<(), (), u32> = g_default.convert_index_type::<u32>()?;

    println!("{} {} {}", g.size(), g_fast.size(), g_u32.size());
    Ok(())
}

```

## Configurable Edge/Community Weight Precision

GVE-Louvain, HyperBall, and the exact centralities (`ExactClosenessCentrality`,
`ExactHarmonicCentrality`, `ExactLinCentrality`) take an additional `W: Weight` type parameter
(`f32` or `f64`, defaulting to `f64`), independent of `Ix`, controlling the precision of their
stored edge/community weights and accumulated distances. Narrower `f32` storage reduces memory
traffic on large graphs; numerically sensitive comparisons (e.g. GVE-Louvain's modularity-gain
decision) are always computed in `f64` internally regardless of `W`, so narrowing `W` doesn't
sacrifice correctness.

```rust
use sparx::communities::gve_louvain::AlgoGVELouvain;
use sparx::graph::GraphMemoryMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "./datasets/graphs/kmer_V2a.mtx";
    let g: GraphMemoryMap<(), (), usize> = GraphMemoryMap::from_file(path, None, Some(32))?;

    // f64 weights (default) --- turbofish needed here since Rust doesn't infer defaulted
    // generic parameters from a bare `::new(...)` call
    let louvain_f64 = AlgoGVELouvain::<_, _, _, f64>::new(&g)?;

    // f32 weights --- halves the memory traffic of the k/sigma/edge-weight accumulators
    let louvain_f32 = AlgoGVELouvain::<_, _, _, f32>::new(&g)?;

    println!(
        "{} {}",
        louvain_f64.partition_modularity(),
        louvain_f32.partition_modularity()
    );
    Ok(())
}

```

# Reproducibility

## Memory Benchmarks

To obtain our memory benchmarking results we made use of a custom `cargo` target which launches the process and records the memory usage frame by frame using `valgrind --tool=massif`. To reproduce our results, on a Linux machine, with an `x86` architecture CPU, first build the `cargo` profile by running:

```bash
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=-sha" cargo build --profile bench_cache
```

If your machine runs Linux on an `arm64` CPU, try running the equivalent architecture command:

```bash
RUSTFLAGS="-C target-cpu=armv8-a -C target-feature=-crypto,-sve,-sve2" cargo build --profile bench_cache
```

However, the authors did not test the benchmarking tools work on an `arm` CPU and no guarantees are given.

The authors also cannot guarantee reproducibility outside of Linux machines, as Valgrind might not work on other platforms.

To bench the heap memory usage of a given dataset `graph.mtx`, in directory `datasets/graphs/` at the library's base directory, run:

```bash
cargo cache --tool=massif --dataset=./datasets/graphs/graph.mtx -t target_profile
```

And to bench the total memory usage, including memory mapped files, run:
```bash
GLIBC_TUNABLES=glibc.malloc.arena_max=1  cargo cache --tool=massif-pages --dataset=./datasets/graphs/graph.mtx -t target_profile
```


### `target_profile` Options

| Value | Profile Name                          | Main Algorithms or Actions Included                                                                 |
|-------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **0** | cache_profile_0 | Runs all heap independent algorithms with `--pages-as-heap=no` to verify heap independence. |
| **1** | cache_profile_1 | Runs **Hierholzer's Algorithm** with `--pages-as-heap=no` to measure heap memory usage. |
| **2** | cache_profile_2 | Runs **HKRelax** for multiple seeds with `t = 45` and `ε = 0.01` with `--pages-as-heap=no` to measure heap memory usage. |
| **3** | pages_profile_0 | Builds a graph from the given dataset with `--pages-as-heap=yes` to measure total memory usage. |
| **4** | pages_profile_1 | Runs **Hierholzer's Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **5** | pages_profile_2 | Computes the **k-cores** of a graph using **Batagelj & Zaversnik's Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **6** | pages_profile_3 | Computes the **k-cores** of a graph using **Liu et al.'s Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **7** | pages_profile_4 | Computes the **k-trusses** of a graph using **Burkhardt et al.'s Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **8** | pages_profile_5 | Computes the **k-trusses** of a graph using **PKT** with `--pages-as-heap=yes` to measure total memory usage. |
| **9** | pages_profile_6 | Computes the **Louvain partition** of a graph using **GVELouvain** with `--pages-as-heap=yes` to measure total memory usage. |
| **10** | pages_profile_7 | Runs **HyperBall** with precision 6 with `--pages-as-heap=yes` to measure total memory usage. |



## Wall-Time Benchmarks

For our wall-time benchmarks we made use of runtime target profiles enabled with `feature = "bench"`, they may be deployed for a given dataset `graph.mtx`, in directory `datasets/graphs/` at the library's base directory by running:
```bash
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph.mtx -e target_profile
```


### `target_profile` Options

| Value | Profile Name                          | Main Algorithms or Actions Included                                                                 |
|-------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **0** | hyperball_profile | **HyperBall** validation --- the exact closeness centrality of every node of the graph is calculated and the Mean Absolute Error, Mean Absolute Percentual Error and Spearman's Rho of HyperBall's closeness centrality approximation are measured and logged. |
| **1** | hk_relax_profile | **HK-Relax** validation --- the algorithm is run 10'000 times with variable `t` and `ε` parameters for random seed nodes in single--source mode, and the resulting communities are then filtered, parametrized and logged. |
| **2** | general_profile | Runs all other algorithms multiple times, and parametrizes and logs each run. |

### Thread Count

To bench wall-time with different thread counts the `-t16` can be altered to change the number of threads to be used in parallel algorithms:

    1. If `-t0` or `-t1` are supplied, the program runs single-threaded.
    2. If `-tx`, where `x > 1`, is supplied, the program runs with `x` threads.
