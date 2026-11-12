#!/usr/bin/env python3
"""Wall-time benchmarks of sparx's algorithms, against NetworKit's where it has an equivalent.

Run from anywhere (paths are relative to the repository's root), with sparx's Python package
(`maturin develop --release` in `python/`, or a built `_sparx` on `PYTHONPATH`) and, for the
comparisons, `networkit` installed:

    python3 scripts/bench/bench.py                       # every dataset, both libraries
    python3 scripts/bench/bench.py --datasets graph_8_15 --libraries sparx --algorithms louvain,leiden
    python3 scripts/bench/bench.py --list                # the datasets and algorithms

Each (dataset, library) pair runs in a process of its own (so that one's memory doesn't weigh on
the other's, and a run past `--timeout` is stopped), appending a row per run to a CSV in
`results/bench/` (as it goes: a stopped or crashed run keeps the rows before it). Both libraries
see the same graph: sparx parses the dataset (any format: MatrixMarket, GGCAT's `.lz4`), and
NetworKit's graph is built from sparx's CSR (each undirected edge once; NetworKit's copy without
self-loops, which its k-core decomposition refuses). Algorithms run with every thread
(`--threads`), best of `--reps` runs (one, for runs past a minute); algorithms past a dataset's
size caps (see ALGORITHMS) are recorded as skipped. Summarize with `summarize.py`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "datasets" / "graphs"
RESULTS = ROOT / "results" / "bench"
FIELDS = [
    "timestamp", "library", "version", "dataset", "nodes", "edges", "threads", "algorithm",
    "run", "seconds", "status", "result",
]


# ------------------------------------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    name: str
    file: str
    # how it's made if missing: `None` for the ones `scripts/datasets/build_datasets.sh` builds
    generate: Optional[Callable[[Path], None]] = None

    @property
    def path(self) -> Path:
        return GRAPHS / self.file


def write_mtx(path: Path, n: int, edges) -> None:
    import numpy as np

    edges = np.asarray(edges, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    lo, hi = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    pairs = np.unique(lo * n + hi)
    lo, hi = pairs // n, pairs % n
    with open(path, "w") as f:
        f.write(f"%%MatrixMarket matrix coordinate pattern symmetric\n{n} {n} {len(pairs)}\n")
        np.savetxt(f, np.stack([hi + 1, lo + 1], axis=1), fmt="%d")


def barabasi_albert(n: int, m: int, seed: int):
    """Preferential attachment: each new node links to `m` earlier ones, by degree."""
    import numpy as np

    rng = np.random.default_rng(seed)
    targets = np.empty(2 * n * m, dtype=np.int64)
    targets[: 2 * m] = np.repeat(np.arange(m), 2)
    filled, edges = 2 * m, []
    for v in range(m, n):
        chosen = targets[rng.integers(0, filled, m)]
        edges.extend((v, int(u)) for u in chosen)
        targets[filled : filled + m] = chosen
        targets[filled + m : filled + 2 * m] = v
        filled += 2 * m
    return edges


def generated(kind: str, n: int, seed: int) -> Callable[[Path], None]:
    def make(path: Path) -> None:
        import numpy as np

        if kind == "ba":
            edges = barabasi_albert(n, 4, seed)
        else:
            rng = np.random.default_rng(seed)
            edges = rng.integers(0, n, (4 * n, 2))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mtx(path, n, edges)

    return make


DATASETS = [
    # small: for the algorithms whose cost grows with |V| |E| (exact centralities, betweenness)
    Dataset("ba_100k", "bench_ba_100k.mtx", generated("ba", 100_000, 1)),
    Dataset("er_100k", "bench_er_100k.mtx", generated("er", 100_000, 2)),
    # medium
    Dataset("graph_4_10", "graph_4_10.lz4"),
    Dataset("graph_8_10", "graph_8_10.lz4"),
    Dataset("synth_dna_k10", "synth_dna_k10_edges8_4M.mtx"),
    # large
    Dataset("synth_dna_k11", "synth_dna_k11_edges33_5M.mtx"),
    Dataset("graph_8_15", "graph_8_15.lz4"),
    Dataset("synth_prot20", "synth_prot20_k5_edges128M.mtx"),
    Dataset("graph_9_15", "graph_9_15.lz4"),
    Dataset("synth_prot23", "synth_prot23_k5_edges296M.mtx"),
]


# ------------------------------------------------------------------------------------------------
# Algorithms
# ------------------------------------------------------------------------------------------------

INF = float("inf")


@dataclass(frozen=True)
class Algorithm:
    name: str
    sparx: Optional[Callable[[Any], str]]
    networkit: Optional[Callable[[Any], str]]
    # applies to graphs of at most these many nodes and (undirected) edges
    max_nodes: float = INF
    max_edges: float = INF
    # NetworKit's own edge cap, where it's much slower than sparx
    networkit_max_edges: float = INF


def nk():
    import networkit

    return networkit


def nk_run(algorithm, summary=lambda a: ""):
    algorithm.run()
    return summary(algorithm)


def communities(result) -> str:
    return f"{result['count']} communities, modularity {result['modularity']:.4f}"


def nk_partition(algorithm_class, **kwargs):
    def run(G):
        a = algorithm_class(G, **kwargs)
        a.run()
        p = a.getPartition()
        q = nk().community.Modularity().getQuality(p, G)
        return f"{p.numberOfSubsets()} communities, modularity {q:.4f}"

    return run


def nk_pagerank(G):
    a = nk().centrality.PageRank(G, damp=0.85, tol=1e-6)
    # sparx's stopping rule: the ranks' total (L1) change
    a.norm = nk().centrality.Norm.L1_NORM
    a.run()
    return f"{a.numberOfIterations()} iterations"


SMALL, SPECTRAL_EXACT, MEDIUM = 110_000, 4_200_000, 75_000_000

ALGORITHMS = [
    Algorithm(
        "components",
        lambda g: f"{g.components()['count']} components",
        lambda G: nk_run(
            # (parallel, as sparx's; NetworKit's `ConnectedComponents` is sequential)
            nk().components.ParallelConnectedComponents(G),
            lambda a: f"{a.numberOfComponents()} components",
        ),
    ),
    Algorithm(
        "biconnectivity",
        lambda g: f"{len(g.biconnectivity()['heads'])} biconnected components",
        lambda G: nk_run(
            nk().components.BiconnectedComponents(G),
            lambda a: f"{a.numberOfComponents()} biconnected components",
        ),
    ),
    Algorithm(
        "k_core",
        lambda g: f"max coreness {int(g.k_core(width='u32').max())}",
        lambda G: nk_run(
            nk().centrality.CoreDecomposition(G),
            lambda a: f"max coreness {int(a.maxCoreNumber())}",
        ),
    ),
    Algorithm("triangles", lambda g: f"{int(g.triangles(width='u32').sum()) // 6} triangles", None),
    Algorithm(
        "clustering",
        lambda g: f"average {g.clustering()['average']:.5f}",
        lambda G: nk_run(
            nk().centrality.LocalClusteringCoefficient(G, turbo=True),
            lambda a: f"average {sum(a.scores()) / G.numberOfNodes():.5f}",
        ),
    ),
    Algorithm("k_truss", lambda g: f"max trussness {int(g.k_truss(width='u32').max())}", None),
    Algorithm("louvain", lambda g: communities(g.louvain()), nk_partition(lambda G: nk().community.PLM(G, refine=False))),
    Algorithm("leiden", lambda g: communities(g.leiden()), nk_partition(lambda G: nk().community.ParallelLeiden(G))),
    Algorithm(
        "label_propagation",
        lambda g: communities(g.label_propagation()),
        nk_partition(lambda G: nk().community.PLP(G)),
    ),
    # exploration ending early (`exploration="auto"`): compared with NetworKit's same PLP run (see
    # `summarize.py`'s COMPARED_WITH), which isn't run twice
    Algorithm(
        "label_propagation_auto",
        lambda g: communities(g.label_propagation(exploration="auto")),
        None,
    ),
    # the asynchronous engine (as NetworKit's PLP), with each exploration
    Algorithm(
        "label_propagation_async",
        lambda g: communities(g.label_propagation(engine="asynchronous")),
        None,
    ),
    Algorithm(
        "label_propagation_async_auto",
        lambda g: communities(g.label_propagation(engine="asynchronous", exploration="auto")),
        None,
    ),
    Algorithm("pagerank", lambda g: f"max rank {g.pagerank().max():.3e}", nk_pagerank),
    Algorithm(
        "bfs",
        lambda g: f"{int((g.bfs(SOURCE) >= 0).sum())} reached",
        lambda G: nk_run(
            nk().distance.BFS(G, SOURCE, storePaths=False),
            lambda a: f"{sum(d < 1e300 for d in a.getDistances())} reached",
        ),
    ),
    Algorithm(
        "sssp",
        lambda g: f"{int((g.dijkstra(SOURCE, engine='auto') < float('inf')).sum())} reached",
        lambda G: nk_run(
            nk().distance.Dijkstra(G, SOURCE, storePaths=False),
            lambda a: f"{sum(d < 1e300 for d in a.getDistances())} reached",
        ),
    ),
    Algorithm("degree_assortativity", lambda g: f"r = {g.degree_assortativity():.5f}", None),
    Algorithm(
        "hyperball",
        lambda g: f"mean harmonic {g.hyperball(precision=8)['harmonic'].mean():.4e}",
        None,
        max_edges=MEDIUM,
    ),
    Algorithm(
        "spectral",
        lambda g: f"{g.spectral_clustering(16)['count']} clusters",
        None,
        max_edges=MEDIUM,
    ),
    Algorithm(
        "spectral_exact",
        lambda g: f"{g.spectral_clustering(16, engine='exact')['count']} clusters",
        None,
        max_nodes=SPECTRAL_EXACT,
    ),
    Algorithm(
        "kadabra",
        lambda g: f"{g.betweenness(engine='kadabra', epsilon=0.01)['samples']} samples",
        lambda G: nk_run(
            nk().centrality.KadabraBetweenness(G, 0.01, 0.1),
            lambda a: f"{a.getNumberOfIterations()} samples",
        ),
        max_edges=MEDIUM,
        # 19 minutes on graph_4_10's 1.8M edges, stopped after 2 h 45 min on graph_8_10's 2.1M
        # (sparx: 0.4 and 0.6 s)
        networkit_max_edges=2_000_000,
    ),
    Algorithm(
        "betweenness",
        lambda g: f"max {g.betweenness(engine='exact')['betweenness'].max():.4e}",
        lambda G: nk_run(
            nk().centrality.Betweenness(G, normalized=True),
            lambda a: f"max {max(a.scores()):.4e}",
        ),
        max_nodes=SMALL,
    ),
    Algorithm(
        "closeness",
        lambda g: f"max {g.closeness().max():.4e}",
        lambda G: nk_run(
            nk().centrality.Closeness(G, True, nk().centrality.ClosenessVariant.GENERALIZED),
            lambda a: f"max {max(a.scores()):.4e}",
        ),
        max_nodes=SMALL,
    ),
    Algorithm(
        "harmonic",
        lambda g: f"max {g.harmonic().max():.4e}",
        lambda G: nk_run(
            nk().centrality.HarmonicCloseness(G, normalized=True),
            lambda a: f"max {max(a.scores()):.4e}",
        ),
        max_nodes=SMALL,
    ),
]


# ------------------------------------------------------------------------------------------------
# Workers: one (dataset, library) pair each
# ------------------------------------------------------------------------------------------------

class Recorder:
    def __init__(self, out: Path, library: str, version: str, dataset: str, threads: int):
        self.out, self.base = out, dict(library=library, version=version, dataset=dataset, threads=threads)
        self.nodes = self.edges = 0

    def row(self, algorithm: str, run: int, seconds: float, status: str, result: str = "") -> None:
        with open(self.out, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(
                dict(
                    self.base,
                    timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                    nodes=self.nodes,
                    edges=self.edges,
                    algorithm=algorithm,
                    run=run,
                    seconds=f"{seconds:.6f}",
                    status=status,
                    result=result,
                )
            )
        print(f"  {self.base['library']:9} {algorithm:21} run {run}: {status} {seconds:9.3f}s  {result}", flush=True)


def timed(f):
    start = time.perf_counter()
    result = f()
    return time.perf_counter() - start, result


def run_algorithms(rec: Recorder, graph, library: str, names: list[str], reps: int) -> None:
    for a in ALGORITHMS:
        if a.name not in names:
            continue
        f = getattr(a, library)
        if f is None:
            continue
        if rec.nodes > a.max_nodes or rec.edges > a.max_edges:
            rec.row(a.name, 0, 0.0, "skipped", "past the size caps")
            continue
        if library == "networkit" and rec.edges > a.networkit_max_edges:
            rec.row(a.name, 0, 0.0, "skipped", "past NetworKit's size cap (hours)")
            continue
        # (a run the timeout stops leaves only this row)
        rec.row(a.name, -1, 0.0, "started")
        for run in range(reps):
            try:
                seconds, result = timed(lambda: f(graph))
            except Exception as e:  # recorded, and the next algorithm runs
                rec.row(a.name, run, 0.0, "error", f"{type(e).__name__}: {e}"[:300])
                break
            rec.row(a.name, run, seconds, "ok", result)
            if seconds > 60:
                break


# where BFS and SSSP start: the least node of the largest component (node 0 can be isolated, as
# on graph_8_15, which would time a search of one node), the same for both libraries
SOURCE = 0


def source_of(g) -> int:
    """The least node of `g`'s largest (connected) component, by sparx (not timed)."""
    import numpy as np

    c = g.components()
    largest = int(np.argmax(c["sizes"]))
    return int(np.flatnonzero(c["components"] == largest)[0])


def worker(args) -> None:
    global SOURCE
    import sparx

    dataset = next(d for d in DATASETS if d.name == args.dataset)
    if args.library == "sparx":
        version = importlib.metadata.version("sparx-graph") if has_dist("sparx-graph") else "dev"
        rec = Recorder(args.out, "sparx", version, dataset.name, args.threads)
        seconds, g = timed(lambda: sparx.Graph(dataset.path, index="u32", threads=args.threads))
        with g:
            rec.nodes, rec.edges = g.num_nodes, g.num_edges // 2
            SOURCE = source_of(g)
            rec.row("load", 0, seconds, "ok", "parsed, cache built")
            run_algorithms(rec, g, "sparx", args.algorithms, args.reps)
    else:
        import networkit
        import numpy as np
        import scipy.sparse as sp

        networkit.setNumberOfThreads(args.threads)
        rec = Recorder(args.out, "networkit", networkit.__version__, dataset.name, args.threads)
        with sparx.Graph(dataset.path, index="u32", threads=args.threads) as g:
            offsets, targets = g.csr()
            rec.nodes, rec.edges = g.num_nodes, g.num_edges // 2
            SOURCE = source_of(g)
        n = len(offsets) - 1
        # each undirected edge once (from its smaller end), as NetworKit's undirected graphs take
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
        keep = targets >= rows
        targets = targets[keep].astype(np.int64)
        counts = np.bincount(rows[keep], minlength=n)
        del rows, keep
        indptr = np.concatenate([[0], np.cumsum(counts)])
        upper = sp.csr_matrix((np.ones(len(targets)), targets, indptr), shape=(n, n))
        del targets
        seconds, G = timed(lambda: networkit.graph.GraphFromCsr(upper, directed=False))
        del upper
        G.removeSelfLoops()
        rec.row("load", 0, seconds, "ok", "built from sparx's CSR (not comparable with parsing)")
        run_algorithms(rec, G, "networkit", args.algorithms, args.reps)


def has_dist(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="all", help="comma-separated names, or 'all'")
    p.add_argument("--libraries", default="sparx,networkit")
    p.add_argument("--algorithms", default="all", help="comma-separated names, or 'all'")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--reps", type=int, default=3, help="runs per algorithm (best is kept)")
    p.add_argument("--timeout", type=float, default=4 * 3600, help="seconds per (dataset, library)")
    p.add_argument("--out", type=Path, help="CSV to append to (default: a new one in results/bench)")
    p.add_argument("--workdir", type=Path, default=RESULTS / "work", help="where caches go")
    p.add_argument("--list", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dataset", help=argparse.SUPPRESS)
    p.add_argument("--library", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.algorithms = (
        [a.name for a in ALGORITHMS] if args.algorithms == "all" else args.algorithms.split(",")
    )
    if args.worker:
        worker(args)
        return
    if args.list:
        for d in DATASETS:
            print(f"dataset {d.name:15} {d.path}{'' if d.path.exists() else '  (missing)'}")
        for a in ALGORITHMS:
            libs = [lib for lib in ("sparx", "networkit") if getattr(a, lib)]
            caps = ", ".join(
                f"{k} <= {v:,.0f}" for k, v in (("nodes", a.max_nodes), ("edges", a.max_edges)) if v < INF
            )
            print(f"algorithm {a.name:21} {'+'.join(libs):16} {caps}")
        return

    names = [d.name for d in DATASETS] if args.datasets == "all" else args.datasets.split(",")
    unknown = set(names) - {d.name for d in DATASETS} | set(args.algorithms) - {a.name for a in ALGORITHMS}
    if unknown:
        sys.exit(f"unknown datasets or algorithms: {', '.join(sorted(unknown))} (see --list)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    # absolute: the workers run in `--workdir`
    out = (args.out or RESULTS / f"bench_{dt.datetime.now():%Y%m%d_%H%M%S}.csv").resolve()
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"results: {out}\nmachine: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} CPUs; {args.threads} threads", flush=True)
    for name in names:
        dataset = next(d for d in DATASETS if d.name == name)
        if not dataset.path.exists():
            if dataset.generate is None:
                print(f"{name}: {dataset.path} missing (see scripts/datasets/build_datasets.sh)")
                continue
            print(f"{name}: generating {dataset.path}", flush=True)
            dataset.generate(dataset.path)
        for library in args.libraries.split(","):
            print(f"{name} ({library})", flush=True)
            command = [
                sys.executable, __file__, "--worker", "--dataset", name, "--library", library,
                "--algorithms", ",".join(args.algorithms), "--threads", str(args.threads),
                "--reps", str(args.reps), "--out", str(out),
            ]
            try:
                done = subprocess.run(command, cwd=args.workdir, timeout=args.timeout)
                if done.returncode != 0:
                    print(f"{name} ({library}): exited with {done.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"{name} ({library}): stopped after {args.timeout:.0f}s", flush=True)


if __name__ == "__main__":
    main()
