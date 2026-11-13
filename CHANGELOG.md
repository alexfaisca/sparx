# Changelog

All notable changes to `sparx-graph` (the Rust crate) and its Python bindings.

## 0.2.0 — 2026-11-11

### Added

* **Directed graphs:** `DiGraph` (`GraphMemoryMap<_, _, _, Directed>`), with in-arcs alongside
  out-arcs, `view`s following arcs in either `Direction`, and directed variants of k-cores (in-,
  out- and total degree), k-trusses (cycle and feed-forward triangles), triangle counting,
  Fagiolo's clustering, strongly connected components, Louvain and Leiden (Leicht and Newman's
  directed modularity, multithreaded) and label propagation.
* **Edge weights:** `Weighted<T>` edge labels (`f64` or `f32`), read from Matrix Market values.
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
