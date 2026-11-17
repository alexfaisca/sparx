#!/usr/bin/env python3
"""Times sparx's HyperBall against Boldi and Vigna's reference implementations: WebGraph's
`it.unimi.dsi.webgraph.algo.HyperBall` (Java, `webgraph`) and its Rust port's (webgraph-rs's
`webgraph-algo`, `webgraph-rs`).

Both run on the same undirected graph, with as many registers per counter (`2^log2m`: sparx's
`Precision<log2m>`) and threads, until no counter changes, timed from the counters'
initialisation to the last iteration (the graph's loading aside). sparx runs
`examples/hyperball_bench.rs`, with its iteration cap raised to its largest (1024); WebGraph
`scripts/bench/webgraph/HyperBallBench.java`, on the graph stored as a BVGraph (parallel edges
merged: they change no ball), after `--warmups` untimed runs in the same JVM (its JIT compiler
warmed up, as in a long-lived JVM); webgraph-rs `scripts/bench/webgraph-rs/` on the same BVGraph
(its `.ef` built by webgraph-cli's `webgraph build ef`), on a rayon pool of as many threads,
with its `HyperLogLog` counters (registers packed as the Java ones: `webgraph-rs`) or its
`HyperLogLog8` ones (a byte per register, merged with SIMD instructions, as sparx's:
`webgraph-rs8`). Each
run's iterations are recorded: runs that stop at sparx's cap haven't converged, and aren't
comparable.

Setup: `scripts/bench/webgraph/fetch_jars.sh` (WebGraph's jars, from Maven Central), Java 11+,
sparx's Python bindings (`pip install ./python`, to export the graphs) and, for webgraph-rs,
cargo (its crates and `webgraph-cli` are built on the first run, under
`scripts/bench/webgraph-rs/`). Results are appended
to `--csv` (default `results/hyperball_webgraph.csv`), and a summary printed: each
implementation's best time of `--repeats`.

    python scripts/bench/hyperball_webgraph.py datasets/graphs/bench_er_100k.mtx \\
        tests/fixtures/ggcat/random_graph_9_5.lz4 --log2m 8 --threads 16
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEBGRAPH = ROOT / "scripts" / "bench" / "webgraph"
WEBGRAPH_RS = ROOT / "scripts" / "bench" / "webgraph-rs"
WEBGRAPH_RS_BIN = WEBGRAPH_RS / "target" / "release" / "hyperball-webgraph-rs"
SPARX_BIN = ROOT / "target" / "release" / "examples" / "hyperball_bench"
WEBGRAPH_CLI = WEBGRAPH_RS / "cli" / "bin" / "webgraph"
IMPLS = ("sparx", "webgraph", "webgraph-rs", "webgraph-rs8")
SPARX_MAX_ITERATIONS = 1024
FIELDS = ["timestamp", "graph", "nodes", "arcs", "impl", "log2m", "threads", "run", "seconds", "iterations",
          "write_seconds", "in_ram"]


def open_csv(path: Path):
    """`path`, appended to, its header (and rows) brought up to `FIELDS` if written by an earlier
    version of this script."""
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                rows = list(reader)
                with open(path, "w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    new_file = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    return f, writer


def run_json(cmd: list[str], cwd: Path | None = None) -> dict:
    """Runs `cmd`, returning the JSON its last line of output holds."""
    out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def simple_csr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The graph's CSR (by sparx's parser), parallel arcs merged (a BVGraph has none)."""
    import sparx

    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="sparx_export_"))  # (sparx's cache, dropped on close)
    try:
        with sparx.Graph(str(path)) as g:
            offsets, targets = (np.asarray(a, dtype=np.int64) for a in g.csr())
            n = g.num_nodes
    finally:
        os.chdir(cwd)
    sources = np.repeat(np.arange(n), np.diff(offsets))
    keep = np.ones(len(targets), dtype=bool)
    keep[1:] = (targets[1:] != targets[:-1]) | (sources[1:] != sources[:-1])
    degrees = np.bincount(sources[keep], minlength=n)
    return np.concatenate([[0], np.cumsum(degrees)]), targets[keep]


def java(*args: str, heap: str) -> list[str]:
    return ["java", f"-Xmx{heap}", "-cp", str(WEBGRAPH / "lib" / "*"), str(WEBGRAPH / "HyperBallBench.java"), *args]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graphs", nargs="+", type=Path, help="undirected graphs (.mtx, ggcat .lz4)")
    parser.add_argument("--log2m", type=int, nargs="+", default=[8], help="registers per counter, 2^log2m (4 to 16)")
    parser.add_argument("--threads", type=int, nargs="+", default=[16])
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per implementation")
    parser.add_argument("--warmups", type=int, default=2, help="untimed WebGraph runs first, in the same JVM")
    parser.add_argument("--heap", default="16g", help="the JVM's maximum heap")
    parser.add_argument("--impls", nargs="+", choices=IMPLS, default=list(IMPLS), help="implementations to run")
    parser.add_argument("--sparx-memory", nargs="+", choices=["auto", "ram", "disk"], default=["auto"],
                        help="where sparx keeps its working memory (several: each is run, as `sparx-<memory>`)")
    parser.add_argument("--write-cache", action="store_true",
                        help="have sparx also write its results to the graph's cache, timed apart (write_seconds)")
    parser.add_argument("--work", type=Path, default=ROOT / "results" / "hyperball_webgraph")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "hyperball_webgraph.csv")
    args = parser.parse_args()
    args.graphs = [g.resolve() for g in args.graphs]

    if not any((WEBGRAPH / "lib").glob("webgraph-*.jar")):
        sys.exit("no WebGraph jars: run scripts/bench/webgraph/fetch_jars.sh (Java's also converts the graphs)")
    subprocess.run(["cargo", "build", "--release", "--features", "bench", "--example", "hyperball_bench"], cwd=ROOT, check=True)
    if any(i.startswith("webgraph-rs") for i in args.impls):
        subprocess.run(["cargo", "build", "--release"], cwd=WEBGRAPH_RS, check=True)
        if not WEBGRAPH_CLI.exists():
            # (the release matching webgraph 0.7.0)
            subprocess.run(["cargo", "install", "--locked", "webgraph-cli", "--version", "=0.5.0",
                            "--root", str(WEBGRAPH_RS / "cli")], check=True)
    args.work.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    f, writer = open_csv(args.csv)
    with f:

        for graph in args.graphs:
            name = graph.name.split(".")[0]
            work = args.work / name
            work.mkdir(exist_ok=True)
            basename = work / "bvgraph"
            if not Path(f"{basename}.graph").exists():
                offsets, targets = simple_csr(graph)
                offsets.astype("<i8").tofile(work / "csr.offsets")
                targets.astype("<i4").tofile(work / "csr.targets")
                del offsets, targets
                subprocess.run(java("convert", str(work / "csr.offsets"), str(work / "csr.targets"), str(basename), heap=args.heap), check=True)
                (work / "csr.offsets").unlink()
                (work / "csr.targets").unlink()
            if any(i.startswith("webgraph-rs") for i in args.impls) and not Path(f"{basename}.ef").exists():
                # webgraph-rs reads the offsets in 4 byte words: Java's are padded to a whole one
                # with zeros, which Java's reader, reading n + 1 offsets, never gets to
                offsets = Path(f"{basename}.offsets")
                with open(offsets, "ab") as padded:
                    padded.write(b"\0" * (-offsets.stat().st_size % 4))
                subprocess.run([str(WEBGRAPH_CLI), "build", "ef", str(basename)], check=True)
            for log2m in args.log2m:
                for threads in args.threads:
                    prefix = work / f"p{log2m}"
                    for r in range(args.repeats):
                        runs = [("sparx", m) for m in args.sparx_memory] if "sparx" in args.impls else []
                        runs += [(i, None) for i in args.impls if i != "sparx"]
                        for impl, memory in runs:
                            if impl == "sparx":
                                out = run_json([str(SPARX_BIN), str(graph), str(threads), str(log2m), str(prefix), memory, *(["write"] if args.write_cache else [])], cwd=ROOT)
                                if len(args.sparx_memory) > 1:
                                    impl = f"sparx-{memory}"
                            elif impl == "webgraph":
                                out = run_json(java("run", str(basename), str(log2m), str(threads), "0", str(prefix), str(args.warmups), heap=args.heap))
                            else:
                                counters = "hll8" if impl == "webgraph-rs8" else "hll"
                                out = run_json([str(WEBGRAPH_RS_BIN), str(basename), str(log2m), str(threads), "0", str(prefix), counters])
                            row = dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"), graph=name,
                                       nodes=out["nodes"], arcs=out["arcs"], impl=impl, log2m=log2m, threads=threads,
                                       run=r, seconds=out["seconds"], iterations=out["iterations"],
                                       write_seconds=out.get("write_seconds", ""), in_ram=out.get("in_ram", ""))
                            writer.writerow(row)
                            f.flush()
                            rows.append(row)
                            print(f"{name} log2m={log2m} threads={threads} {impl}: {out['seconds']:.3f} s, "
                                  f"{out['iterations']} iterations", flush=True)

    summarize(rows)
    return 0


def summarize(rows: list[dict]):
    """Each implementation's best time, its speedup over sparx's (`x`: theirs over sparx's), and
    iterations."""
    impls = list(dict.fromkeys(r["impl"] for r in sorted(rows, key=lambda r: not r["impl"].startswith("sparx"))))
    sparx = next((i for i in impls if i.startswith("sparx")), None)
    keys = sorted({(r["graph"], r["log2m"], r["threads"]) for r in rows},
                  key=lambda k: (next(r["arcs"] for r in rows if r["graph"] == k[0]), k))
    header = f"\n{'graph':<28} {'nodes':>10} {'log2m':>5} {'thr':>3}"
    for impl in impls:
        header += f" {impl + ' s':>15}" + (f" {'(x)':>6}" if impl != sparx else "")
    print(header + f"  {'iterations':<16}")
    for graph, log2m, threads in keys:
        mine = [r for r in rows if (r["graph"], r["log2m"], r["threads"]) == (graph, log2m, threads)]
        best = {i: min((r["seconds"] for r in mine if r["impl"] == i), default=float("nan")) for i in impls}
        iterations = [str(next((r["iterations"] for r in mine if r["impl"] == i), "-")) for i in impls]
        line = f"{graph:<28} {mine[0]['nodes']:>10} {log2m:>5} {threads:>3}"
        for impl in impls:
            line += f" {best[impl]:>15.3f}"
            if impl != sparx:
                line += f" {best[impl] / best.get(sparx, float('nan')):>5.1f}x"
        capped = " (sparx capped)" if any(it == str(SPARX_MAX_ITERATIONS) for i, it in zip(impls, iterations) if i.startswith("sparx")) else ""
        print(line + f"  {' / '.join(iterations)}{capped}")


if __name__ == "__main__":
    raise SystemExit(main())
