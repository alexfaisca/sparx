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

with sparx.Graph("graph.mtx", index="u32") as g:
    print(g)                                   # <sparx.Graph |V|=... |E|=... index=u32>
    offsets, targets = g.csr()                 # edge e of node u: offsets[u] <= e < offsets[u+1]

    coreness = g.k_core()                      # per node
    trussness = g.k_truss(algorithm="pkt")     # per edge, in csr() order
    triangles = g.triangles(width="u32")       # per edge
    cycles = g.directed_triangles("cycle")     # per edge, directed cycles only
    flow = g.k_truss(kind="feedforward")       # trusses of feed-forward triangles

    cc = g.clustering()                        # {"local", "average", "transitivity"}
    louvain = g.louvain()                      # {"communities", "count", "modularity"}
    hb = g.hyperball(precision=8)              # {"closeness", "harmonic", "lin", ...} estimates
    closeness = g.closeness()                  # exact
    dist = g.bfs(0)                            # -1 where unreachable
```

* Inputs: Matrix Market (`.mtx`), [ggcat](https://github.com/algbio/ggcat) graphs (`.lz4`) and
  nodes/edges files.
* `index` (`"u32"` or `"u64"`) is the width of node and edge indices; `width` (`"u8"` to `"u64"`)
  that of triangle counts and trussness values, which fail with an error rather than overflow.
* Undirected edges are stored once in each direction; per-edge results follow `csr()`.
* The cache lives in `./.cache/` (relative to the working directory) and is dropped when the
  graph is closed or garbage collected. With `keep_cache=True` it stays, and
  `sparx.Graph.open(g.cache_file)` reopens the graph without parsing it again.
* With GPU support (built with the `gpu` feature: `maturin build --features gpu`), `triangles`
  and `hyperball` take `gpu=True`, and run on any Vulkan, Metal or DX12 GPU.

## Building from source

```sh
pip install maturin
cd python && maturin develop --release
pytest tests
```
