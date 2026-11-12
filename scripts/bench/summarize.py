#!/usr/bin/env python3
"""Summarizes `bench.py`'s CSVs: each (dataset, algorithm)'s best time per library, and sparx's
speedup over NetworKit, as a Markdown table (to stdout, or `--out`).

    python3 scripts/bench/summarize.py results/bench/bench_*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# sparx's algorithms NetworKit's timings of another stand for (one NetworKit algorithm compared
# with several sparx modes)
COMPARED_WITH = {
    "label_propagation_auto": "label_propagation",
    "label_propagation_async": "label_propagation",
    "label_propagation_async_auto": "label_propagation",
}


def fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms" if seconds >= 0.01 else f"{seconds * 1000:.1f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--latest",
        action="store_true",
        help="each (dataset, algorithm, library)'s rows from the last CSV that has any (later "
        "CSVs re-running some algorithms after changes), rather than the best of all",
    )
    args = p.parse_args()

    best: dict[tuple[str, str, str], float] = {}
    status: dict[tuple[str, str, str], str] = {}
    sizes: dict[str, tuple[int, int]] = {}
    datasets: list[str] = []
    algorithms: list[str] = []
    threads = set()
    rows = []
    for i, path in enumerate(args.csvs):
        with open(path) as f:
            rows += [(i, row) for row in csv.DictReader(f)]
    if args.latest:
        last: dict[tuple[str, str, str], int] = {}
        for i, row in rows:
            last[(row["dataset"], row["algorithm"], row["library"])] = i
        rows = [(i, r) for i, r in rows if last[(r["dataset"], r["algorithm"], r["library"])] == i]
    for _, row in rows:
        key = (row["dataset"], row["algorithm"], row["library"])
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
        if row["algorithm"] not in algorithms:
            algorithms.append(row["algorithm"])
        sizes[row["dataset"]] = (int(row["nodes"]), int(row["edges"]))
        threads.add(row["threads"])
        if row["status"] == "ok":
            best[key] = min(best.get(key, float("inf")), float(row["seconds"]))
        elif row["status"] == "started":
            # stopped by the timeout, unless a later row says otherwise
            status.setdefault(key, "timeout")
        else:
            status[key] = row["status"]

    # each mode right after the algorithm it's compared with
    for mode, base in COMPARED_WITH.items():
        if mode in algorithms and base in algorithms:
            algorithms.remove(mode)
            algorithms.insert(algorithms.index(base) + 1, mode)

    def cell(key) -> str:
        if key in best:
            return fmt_time(best[key])
        return status.get(key, "")

    lines = [
        f"Best wall time of each algorithm ({', '.join(sorted(threads))} threads); "
        "`speedup` is NetworKit's time over sparx's (above 1: sparx faster).",
        "",
    ]
    for dataset in datasets:
        n, m = sizes[dataset]
        lines += [f"### {dataset} ({n:,} nodes, {m:,} edges)", "",
                  "| algorithm | sparx | NetworKit | speedup |", "|---|---|---|---|"]
        for algorithm in algorithms:
            s = (dataset, algorithm, "sparx")
            k = (dataset, COMPARED_WITH.get(algorithm, algorithm), "networkit")
            if s not in best and s not in status and k not in best and k not in status:
                continue
            speedup = f"{best[k] / best[s]:.2f}x" if s in best and k in best and algorithm != "load" else ""
            lines.append(f"| {algorithm} | {cell(s)} | {cell(k)} | {speedup} |")
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
