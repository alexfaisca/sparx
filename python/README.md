# sparx (Python)

Python bindings of [*SparX*](https://github.com/alexfaisca/sparx): memory-mapped analytics for
large static sparse graphs (de Bruijn graphs in particular). Graphs are parsed once into a
memory-mapped compressed sparse row cache, and algorithms run in parallel with the GIL released,
returning `numpy` arrays.

```sh
pip install sparx-graph
```

Wheels are built for Linux (x86_64, aarch64) and macOS (Apple silicon, Intel), for Python 3.9 and
later.

```python
import sparx

with sparx.Graph("graph.mtx", index="u32") as g:  # undirected
    print(g)                                   # <sparx.Graph |V|=... |E|=... index=u32>
    offsets, targets = g.csr()                 # edge e of node u: offsets[u] <= e < offsets[u+1]

    coreness = g.k_core()                      # per node
    trussness = g.k_truss(algorithm="pkt")     # per edge, in csr() order
    triangles = g.triangles(width="u32")       # per edge

    cc = g.clustering()                        # {"local", "average", "transitivity"}
    louvain = g.louvain()                      # {"communities", "count", "modularity"}
    leiden = g.leiden()                        # the same, refined (GVE-Leiden)
    hb = g.hyperball(precision=8)              # {"closeness", "harmonic", "lin", ...} estimates
    closeness = g.closeness()                  # exact
    ranks = g.pagerank()                       # alpha=0.85, tol=1e-6 (in total, see below)
    dist = g.bfs(0)                            # -1 where unreachable
    bcc = g.biconnectivity()                   # {"components", "heads", "articulation_points",
                                               #  "bridges", "two_edge_components", "two_edge_count"}
    blocks = g.biconnected_components()        # node arrays, as networkx's biconnected_components
    r = g.degree_assortativity()               # Newman's (networkx's degree_assortativity_coefficient)
    rho = g.degree_assortativity(method="spearman")  # of the degrees' ranks
    g.numeric_assortativity(values)            # a number per node
    g.attribute_assortativity(labels)          # an integer category per node

    lpa = g.label_propagation()                # {"communities", "count", "modularity", ...}
    sc = g.spectral_clustering(16)             # 16 clusters (engine="exact", k=None: chosen)
    emb = g.spectral_embedding(8)              # {"eigenvalues", "vectors", "residuals", ...}
    q = g.modularity(lpa["communities"])       # of any partition
    local = g.hk_relax([0])                    # a local community: {"nodes", "conductance", ...}
    bc = g.betweenness()                       # {"betweenness", "exact", "samples", ...}
    hubs = g.betweenness(engine="kadabra", epsilon=0.001, k=10)["top"]  # sampled, top 10

with sparx.Graph("roads.mtx", weighted=True, weight_type="f32") as w:  # Matrix Market values
    w.edge_weights()                           # in csr() order
    far = w.dijkstra(0, engine="delta")        # weighted distances, in parallel (Δ-stepping)
    w.build_landmarks(8)                       # for A*
    path = w.shortest_path(0, 9)               # {"distance", "nodes", "expanded"}
    paths = w.shortest_paths([0, 1], [9, 5], paths=False)  # distances only, in parallel

with sparx.DiGraph("web.mtx") as d:            # directed
    d.in_degree(0), d.in_neighbours(0)         # and degree/neighbours: out-arcs
    cycles = d.directed_triangles("cycle")     # per arc, directed cycles only
    flow = d.k_truss("feedforward")            # trusses of feed-forward triangles
    cores = d.k_core(degree="in")              # "in", "out" or "total" (in + out: networkx's core_number)
    cc = d.clustering()                        # Fagiolo's: {"local", "average"}
    communities = d.louvain()                  # by directed modularity
    refined = d.leiden()                       # the same, with Leiden's refinement
    spread = d.label_propagation(direction="in")  # labels spreading along arcs
    reach = d.bfs(0, direction="in")           # "out", "in" or "both" (ignoring directions)
    r = d.degree_assortativity(x="out", y="in")  # any of "out" and "in" at either end
    scc = d.components(kind="strong")          # {"components", "count", "sizes"}
    u = d.to_undirected()                      # a sparx.Graph, valid while d is open
    m = d.to_undirected(mode="each")           # "collapse" (default), "each" (per arc) or "mutual"
    trussness = u.k_truss()
```

* `sparx.Graph` is undirected: loading a directed dataset (one with an arc `u -> v` without a
  reverse `v -> u`) raises `RuntimeError`. `sparx.DiGraph` loads any dataset; `is_symmetric`
  tells whether its arcs all have reverses.
* `DiGraph.to_undirected(mode=...)` is the underlying undirected graph: `"collapse"` (the
  default) has one edge per pair of nodes linked by an arc either way (networkx's
  `DiGraph.to_undirected()`), `"each"` one edge per arc, reciprocal and parallel arcs becoming
  parallel edges (networkx's `MultiDiGraph.to_undirected()`), and `"mutual"` one edge per pair
  linked both ways, self-loops included (`to_undirected(reciprocal=True)`). `direction="both"`
  runs on the collapsed graph.
* `Graph.to_directed()` and `DiGraph.to_undirected()` return graphs mapping their parent's cache,
  which the parent owns: they're usable while it's open, and raise `RuntimeError` after.
* Inputs: Matrix Market (`.mtx`), [ggcat](https://github.com/algbio/ggcat) graphs (`.lz4`) and
  nodes/edges files.
* `weighted=True` reads a Matrix Market file's values as edge weights, `weight_type="f64"` (the
  default) or `"f32"` (half the memory; `weight_type` reads it back) (`pattern` files' edges weigh
  one; ggcat and nodes/edges files carry no weights, and fail). `weight_parsing` is
  `"fast"` (the default: no per-value validation, malformed values give unspecified weights) or
  `"checked"` (malformed, out of range or non-finite values raise). `is_weighted`, `weights(u)`
  and `edge_weights()` (in `csr()` order) read them back; unweighted graphs' weigh one. A kept
  weighted graph reopens with `open(cache_file, weighted=True)` (and its `weight_type`).
* `modularity(communities)` is a given partition's (each node's label): Newman's, or on a `DiGraph`
  Leicht and Newman's directed modularity; weighted on weighted graphs. On a `Graph`,
  `hk_relax(seeds)` (Kloster and Gleich's heat kernel diffusion) and `hkpr(seed)` (Chung and
  Simpson's approximate Dirichlet heat kernel PageRank, seeded by `random_seed`) find a local
  community by a sweep cut, `sweep_cut(nodes, heat)` sweeps any diffusion: `{"nodes", "heat",
  "size", "volume", "width", "conductance"}`.
* On weighted graphs, algorithms follow networkx's `weight="weight"`: PageRank (transitions in
  proportion to weight), Louvain and Leiden (weighted modularity), clustering (Onnela's and
  Fagiolo's weighted coefficients; transitivity stays unweighted, as networkx's) and degree
  assortativity (of strengths), and exact `closeness`, `harmonic` and `lin` (weighted
  distances, networkx's `distance="weight"`; with `direction="both"`, two nodes are as far
  apart as their lightest arc). `dijkstra(source)` gives weighted distances (`inf` where
  unreachable; hop counts on unweighted graphs), while `bfs` always counts hops. Weights must be
  finite and non-negative, and a `DiGraph`'s Louvain needs integer weights (it compares gains
  exactly). HyperBall (it counts hops), k-cores, k-trusses and components ignore weights.
* Shortest paths: `dijkstra(source, engine=..., predecessors=False)` runs Dijkstra's algorithm
  (`"sequential"`) or Δ-stepping on the graph's threads (`"delta"`, `delta=` its bucket width;
  `"auto"` picks it on large graphs), with the same distances bit for bit. With
  `predecessors=True` it also returns each node's parent in a shortest path tree (`-1` for the
  source and unreachable nodes), found only then, the same for every engine.
  `shortest_path(s, t)` and `shortest_paths(sources, targets)` (in parallel) give
  `{"distance", "nodes", "expanded"}` dicts (`None` without a path; no `"nodes"` with
  `paths=False`, which keeps less memory), by A*: guided by landmarks once `build_landmarks(k)`
  has kept some in the graph's cache (they persist with `keep_cache=True`; `landmarks` lists
  them, `drop_landmarks()` removes them), else bidirectional Dijkstra. `heuristic=` and
  `bidirectional=` override that: bidirectional search pays without landmarks (hundreds of
  times fewer nodes on small-world graphs), one-directional with them (on road-like graphs).
* `DiGraph.to_undirected(merge=...)` weighs an edge merged from several arcs (`"collapse"` and
  `"mutual"`: every arc between two nodes, both ways) by their `"sum"` (the default), `"min"` or
  `"max"`, the same from both ends.
* `index` (`"u32"` or `"u64"`) is the width of node and edge indices. `width` (`"u8"` to `"u64"`)
  is the width of triangle counts, trussness and coreness values; a value that doesn't fit, such
  as a node of degree 256 in a `"u8"` `k_core`, fails with an error rather than overflowing.
* Undirected edges are stored once in each direction; per-edge results follow `csr()`.
* `biconnectivity` and `biconnected_components` are on `sparx.Graph` only. In `"components"`,
  each node is labelled with one biconnected component it's in and doesn't head, and `"heads"`
  gives each component's head (the node it shares with the component above it). Isolated nodes,
  and nodes heading every component they're in, are labelled -1. `biconnected_components()` lists every component's
  nodes, heads included. `"bridges"` is an `(n, 2)` array of sorted pairs, and parallel edges
  are never bridges.
* Assortativity coefficients match networkx's, multigraphs included. `method="spearman"`
  correlates the degrees' ranks rather than the degrees themselves: on graphs with heavy-tailed
  degrees, Pearson's coefficient shrinks as they grow, while Spearman's doesn't. Degree and
  attribute coefficients are exact, and the same whatever the number of threads; a coefficient
  is `NaN` where it's undefined (for example, all degrees equal). `attribute_assortativity`
  takes integer labels: for others, pass `numpy.unique(labels, return_inverse=True)[1]`.
* `label_propagation()` finds communities by semi-synchronous label propagation (colors of
  non-adjacent nodes moving in turn): `{"communities", "count", "modularity", "moves",
  "colors"}`, the same whatever the threads, connected and numbered by their least node. On a
  `DiGraph`, `direction="in"` has labels spread along arcs (`"out"` against them, `"both"`, the
  default, ignores directions) and the modularity is directed. Faster than `leiden()`, and as
  good or better on clear structure; `leiden()`'s modularity is higher on graphs without it.
  `exploration="auto"` ends exploring early, once a pass moves at most `tolerance` of the
  nodes: faster where exploring has stopped paying, for a slightly lower modularity.
* `spectral_clustering(k)` clusters by spectral clustering, into `{"communities", "count",
  "modularity", "lambda_k"}`, the same whatever the threads. `engine="compressive"` (the default)
  filters random signals by a low-pass graph filter; `engine="exact"` finds the eigenvectors
  themselves (Chebyshev-filtered subspace iteration, with locking; `"eigenvalues"` and
  `"converged"` in the dict), and with `k=None` chooses `k` by the largest eigengap among
  `max_k`. `matrix="laplacian"` (the default, regularized by the mean degree; any graph) or
  `"bethe"` (the Bethe Hessian: unweighted graphs, weaker clusters). A `DiGraph` is clustered by
  its symmetrization (`A + A^T`). `spectral_embedding(count)` returns the eigenpairs themselves:
  `{"eigenvalues", "vectors" (n × count), "residuals", "iterations", "converged"}`. On planted
  partitions it finds clusters Leiden misses; on graphs without a few clear clusters (de Bruijn
  graphs), Leiden's modularity is higher.
* `betweenness()` is networkx's `betweenness_centrality` (`normalized=True` by default,
  `endpoints=False`): exact by Brandes' algorithm (`engine="exact"`), or estimated by KADABRA's
  adaptive sampling (`engine="kadabra"`): every node within `epsilon` (a fraction of ordered
  pairs: `epsilon * n / (n - 2)` normalized) with probability `1 - delta`, or, with `k`, until
  the `k` most central nodes (`"top"`) are told apart, which takes far fewer samples. `"auto"`
  (the default) is exact when `|V| |E| <= 1e10`. Results are the same whatever the threads
  (`seed` seeds KADABRA); weights must be positive.
* `louvain()` and `leiden()` communities are connected: any community Louvain leaves
  internally disconnected is split into its parts, which only raises modularity. `leiden()`
  refines each pass's communities into connected sub-communities before aggregating them
  (GVE-Leiden), finding better communities on some graphs for some more time.
* `pagerank`'s `tol` bounds the ranks' total change, while networkx's is multiplied by the
  number of nodes (which, ranks summing to 1, graphs of millions of nodes meet at once):
  networkx's results are those of `tol=nx_tol * len(g)`. Edges are unweighted (`weight=None`)
  unless the graph is weighted.
* `pagerank(engine=...)` picks how ranks are computed, with the same results, bit for bit:
  `"pull"` (fastest while the ranks fit in the CPU's cache; undirected graphs only),
  `"partition"` (partition-centric, fastest once they don't) or `"auto"` (the default: `"pull"`
  for undirected graphs of up to a million nodes or so, `"partition"` otherwise).
* The cache lives in `./.cache/` (relative to the working directory) and is dropped when the
  graph is closed or garbage collected. With `keep_cache=True` it stays, and
  `sparx.Graph.open(g.cache_file)` reopens the graph without parsing it again.
* With GPU support (built with the `gpu` feature: `maturin build --features gpu`), `triangles`,
  `hyperball` and `degree_assortativity` take `gpu=True`, and run on any Vulkan, Metal or DX12 GPU.

## Building from source

```sh
pip install maturin
cd python && maturin develop --release
pytest tests
```
