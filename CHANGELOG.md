# Changelog

All notable changes to `sparx-graph` (the Rust crate) and its Python bindings.

## Unreleased

### Added

* `shared_slice::MemoryPolicy` (`Auto`, `Ram`, `Disk`): where an algorithm keeps the arrays it
  rewrites as it runs, with `available_memory` and `fits_in_memory` (at most half the memory
  available). `HyperBallInner::new_with_memory`, and `hyperball(memory=...)` in Python.

### Changed

* *HyperBall* keeps its counters, accumulators and scratch arrays in RAM when they take at most
  half the memory available (`MemoryPolicy::Auto`, the default), writing its cached results once,
  at the end. They were memory mapped from the graph's cache, so that every iteration's changes
  were written back to the disk: once they outgrew the kernel's dirty page limits, runs were
  bound by the disk's speed (58 minutes rather than 50 seconds on a 21.9M node graph, on a slow
  disk). On the disk, counters are now only rewritten when they changed in an iteration or the
  one before, rather than every iteration.
* Exact betweenness's threads are capped to one, rather than unbounded, when how much memory is
  available is unknown.

### Fixed

* Biconnectivity was wrong on graphs with as many nodes as their `Ix` can number (`Ix::MAX + 1`:
  256 at `u8`), whose last node was taken for the search trees' "no parent" marker.
* The unweighted clustering coefficients and transitivity of multigraphs counted triangles once
  per combination of parallel edges, but possible ones once per pair of neighbours. They're now
  the simple graph's (networkx's of `nx.Graph(multigraph)`). Results on graphs whose parallel
  edges close no triangle, the bundled de Bruijn graphs among them, are unchanged.
* `export_petgraph_stripped` and `export_rustworkx_stripped` return the error of a node whose
  neighbours can't be read, rather than printing it and exporting the graph without its arcs.
* A cache entry's metadata (`metadata()`, and its file) counted `|V| + 1` nodes (its offsets).
  Entries cached (and kept) before still show `|V| + 1`: the count is only written once.
* The (unweighted, undirected) average clustering coefficient is the same whatever the number
  of threads, as the directed and weighted ones (its last bits depended on how nodes were shared
  out among threads).
* The documentation of exact betweenness in the Python bindings promised results the same
  whatever the number of threads, which only KADABRA's are (exact ones may differ in their last
  bits, as the Rust documentation says); the sweep cut's targets (`hk_relax`, `hkpr`,
  `sweep_cut`) are documented as they work: the sweep stops past twice them, and only prefixes
  of conductance below 1 count (without one, the community is empty). `threads` is documented
  (at most 255, 1 by default). `DiGraph.numeric_assortativity` is over arcs, not edges each way.
  Betweenness counts parallel arcs as different paths, unlike networkx (now said so), and
  `DirectedClusteringCoefficient` counts them, unlike `ClusteringCoefficient`.
* The benchmarking CLI reports the crate's version (was a fixed `0.1`), and its help describes
  `-f`'s input formats and `-e` correctly.
* Documentation: the Python type stubs only accept `direction="both"` on `Graph.label_propagation`
  (as the bindings); the README's Python example called `Graph.k_truss(kind=...)`, a `DiGraph`'s;
  the README's testing section claimed more coverage than the tests have; the shortest paths
  speedup in its summary is 47x, as in the benchmarks.

### Testing

* CI tests the `rayon`, `petgraph`, `rustworkx` and `iter_with_offset` features (whose tests
  never ran), builds each alone, and documents them; GPU tests fail in CI when they find no
  adapter, rather than passing unrun.
* Tests of the petgraph and rustworkx exports, of `GraphMemoryMap`'s shorthands (`k_core`,
  `k_truss`, `louvain`, ...), and of triangle counts, clustering coefficients, HyperBall,
  HK-Relax, components and exact centralities across thread counts; the README's GPU example is
  compiled as a doctest. Betweenness tests remove their files even when they fail. Exact
  betweenness and KADABRA on multigraphs, clustering coefficients on the multigraph fixture.
* Python tests of the bindings' untested options: sweep cut targets, `hkpr`'s
  `target_conductance`, `label_propagation`'s `max_iterations`, spectral options (including the
  Bethe Hessian embedding), `id`, `metadata` and the thread limit, and `Graph`'s weight parsing;
  and of the methods `test_edge_cases.py` didn't call on degenerate graphs (spectral embeddings,
  sweep cuts, numeric and attribute assortativity, landmarks, `shortest_paths`, directed
  triangles, `to_undirected`).

### Changed

* With the `bench` feature, `sparx::test_common` is gone: its one public helper is
  `sparx::graph::test_utils::get_or_init_dataset_exact_closeness`, alongside the tests' others.
  It computes the values with `ExactClosenessCentrality` (the same, checked against the BFS
  formula it used), and the exact harmonic and Lin values are written where it says they are.
  The unimplemented, unused `degree_dist_impl` is removed.

## 0.2.1 — 2026-11-13

### Added

* `sparx::graph` re-exports the parsing accumulators: the `AccumulatorAtomic` trait,
  `AccumulatorOverflow` and the atomic counters (`AtomicU8` to `AtomicUsize`), so that choosing an
  accumulator for `from_file_with_accumulator` needs no `portable-atomic` dependency.
* `BFSDists::reachable`.
* Examples: `shortest_paths` (Dijkstra/Δ-stepping, shortest path trees, A* with landmarks) and
  `components` (connected, biconnected and strongly connected components, assortativity).

### Changed

* The examples share one layout: `[graph] [threads]` arguments, a bundled fixture by default, and
  the graph's cache entry dropped even when a run fails.
* The README's Rust examples are complete programs, compiled as doctests; it gained a testing
  section.

### Deprecated

* `BFSDists::recheable`, renamed `reachable`.

### Fixed

* `ApproxDirHKPR::compute` failed whenever a random walk's sampled length was 0, which small heat
  kernel times (high target conductances) made near certain.
* `ApproxDirHKPR` finds the empty community (as HK-Relax) on a single node graph or from a seed
  without edges, rather than failing on a degenerate sample count, and refuses targets giving no
  positive heat kernel time with the reason. The Python `hkpr`'s default target volume is at least 1.
* The undirected average clustering coefficient of a graph without nodes is 0 (was NaN), as the
  directed and the weighted ones.

## 0.2.0 — 2026-11-11

### Added

* **Directed graphs:** `DiGraph` (`GraphMemoryMap<_, _, _, Directed>`), with in-arcs alongside
  out-arcs, `view`s following arcs in either `Direction`, and directed variants of k-cores (in-,
  out- and total degree), k-trusses (cycle and feed-forward triangles), triangle counting,
  Fagiolo's clustering, strongly connected components, Louvain and Leiden (Leicht and Newman's
  directed modularity, multithreaded) and label propagation.
* **Edge weights:** `Weighted<T>` edge labels (any integer or float type), read from Matrix Market
  values.
  PageRank, Louvain and Leiden, clustering (Onnela's and Fagiolo's coefficients), degree
  assortativity (strengths), exact closeness, harmonic and Lin centralities, HK-Relax, HKPR and
  sweep cuts use them. Unweighted graphs run the same code as before (dispatch is resolved at
  compile time).
* **Weighted shortest paths:** Dijkstra's algorithm and parallel Δ-stepping (`DijkstraDists`,
  `ShortestPathEngine`; distances identical bit for bit across engines), optional predecessors,
  A* with ALT landmarks (persistable in the cache), bidirectional search, and visitor-style
  `DijkstraVisit` / `astar_visit` with BGL-style hooks (`ShortestPathVisitor`).
* **Betweenness centrality** (`centralities::betweenness`): exact (`ExactBetweennessCentrality`:
  Brandes' algorithm in parallel over sources, within each biconnected component on undirected
  graphs) or estimated by KADABRA's adaptive sampling (in parallel, balanced bidirectional BFS,
  every node within `epsilon` or the top `k` told apart; the same whatever the number of
  threads), on directed and weighted graphs too.
* **Label propagation** (`communities::label_propagation`): semi-synchronous (Jones–Plassmann
  colouring) with GVE-LPA pruning; deterministic whatever the number of threads; `LpaDirection`
  on directed graphs.
  `LpaOptions::exploration_ends_early` (`exploration="auto"` in Python) ends exploring at the
  first pass that moves at most `tolerance` of the nodes: up to 2.5x faster where exploring has
  stopped paying, for a slightly lower modularity, and the same communities where it pays.
  `LpaEngine::Asynchronous` (`engine="asynchronous"` in Python): nodes move in turn, seeing
  labels as they are (as NetworKit's PLP), without coloring: 1.7x to 3.1x as fast on graphs of
  millions of nodes, the communities depending on the threads' timing.
* **Spectral clustering** (`communities::spectral`): a compressive engine (KPM and Chebyshev
  low-pass filtering of random signals) and an exact one (Chebyshev-filtered subspace iteration
  with locking), on the regularized Laplacian or the Bethe Hessian, with automatic choice of `k`;
  `SpectralEmbedding` returns the eigenpairs.
* **Leiden's refinement** for Louvain (`Refinement::Leiden`, `new_leiden`), and splitting of
  internally disconnected communities.
* **PageRank** with pull and partition-centric engines.
* **Biconnectivity:** biconnected components, articulation points, bridges and two-edge-connected
  components.
* **Assortativity:** degree (Pearson and Spearman), numeric and attribute coefficients.
* **GPU** acceleration (`gpu` feature, wgpu) for triangle counting, HyperBall and degree
  assortativity.
* Genericity over the index width (`u32`/`u64`) and over the width of counts and coreness values
  across k-cores, k-trusses, HyperBall, traversals, Hierholzer's algorithm and centralities.
* Memory-mapped scratch and output memory throughout (connected components, triangle counts,
  traversal stacks, directed Louvain), so working sets live in the cache rather than the heap.
* HyperBall skips nodes whose neighbours didn't change; SIMD HyperLogLog unions; batched BFS
  for exact centralities.
* `sparx::label!` declares node and edge label types (integer newtypes) in a line each: derives,
  `#[repr(transparent)]`, compile-time checked `Pod`/`Zeroable`, and the `N`/`E` impls, without
  the declaring crate depending on `bytemuck`.
* **Python bindings** (`sparx-graph` on PyPI): `sparx.Graph` and `sparx.DiGraph` with the
  algorithms above, returning `numpy` arrays with the GIL released.
* Performance-regression checks (`tests/perf_regression.rs`, run in CI), including the parallel
  speedup of label propagation and the exact spectral engine's convergence.

### Changed

* Louvain's initial tolerance is `1e-6` (it was `0.01`), which stopped local moving too early on
  sparse graphs; ties are broken in a fixed order, so single-threaded runs repeat exactly.
* Liu et al.'s algorithm is the default k-core decomposition; directed k-cores (in-, out- and
  total degree) are decomposed by it too, in parallel (they were sequential).
* Directed clustering coefficients are computed in parallel after counting triangles (that
  part was sequential), with an average that's the same whatever the threads.
* Approximate Dirichlet HKPR takes a seed (`with_random_seed`, default `0x5eed`), and sweep cuts
  break ties by node id, so results repeat.
* **Breaking:** `SharedSlice`, `SharedSliceMut` and `SharedQueueMut` are `Send`/`Sync` only for
  element types that allow it (`T: Sync`, and `T: Send + Sync` for the mutable ones).
* **Breaking:** the `sparx-graph-derive` crate is gone, and with it the `GenericEdge`,
  `GenericEdgeType`, `N` and `E` derive macros re-exported by `sparx::graph` (unused, or no-ops),
  the `GenericEdge` and `GenericEdgeType` traits they implemented (unused too)
  and `#[sparkx_label]`: `sparx-graph` is a single crate, publishable on its own. Label types are
  declared by `sparx::label!` (see Added), or with plain derives (`bytemuck`'s `Pod` and
  `Zeroable` among them).
* **Breaking:** the `into_boxed_slice` methods of the shared slice views were removed: they
  freed memory-mapped memory as if it were heap memory.

* Faster, with the same results: PageRank's `Auto` engine pulls the first iteration before
  building the partition-centric bins (graphs converging at once: up to 10x); label propagation's
  coloring is counter-based Jones and Plassmann's, `O(|E|)` work rather than a pass over the
  remaining nodes' arcs per round (3.3x), and its passes count labels without weights on
  unweighted graphs (1.2x to 5x in all); modularity is computed in parallel, in memory callers
  can lend (`Graph::modularity_with`). K-core decomposition (Liu et al.'s peeling, undirected and
  directed) is 1.1x to 3.1x faster: its threads push peeled nodes in batches (each node had
  updated two counters all threads contended for), take the frontier in chunks as they finish,
  and bound each local search (one could peel most of a near-regular graph while the other
  threads waited). Shortest paths on unweighted graphs are found by a parallel BFS, whose levels
  of a few nodes one thread searches alone (sharing each cost two barriers: on graphs of long
  paths, tens of thousands of them).

### Fixed

* Parallel arcs of a labeled (or weighted) graph came out of parallel parsing in whatever order
  the threads reached them, so that their labels' order, and sums over them (weighted
  PageRank's ranks, for one), could differ from run to run in their last bits: they're ordered
  by their labels now.
* GGCAT graphs with edge labels: neighbour lists were permuted by the inverse of their sort order,
  so they came out unsorted (labels stayed paired with their neighbours).
* Thread-safety and cache-naming bugs in exact centralities; a data race in PKT; contention on
  atomics in parallel loops.
* `BFSDists`' count of reachable nodes.
