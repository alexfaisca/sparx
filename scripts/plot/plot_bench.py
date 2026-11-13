#!/usr/bin/env python3
"""Plots `scripts/bench/bench.py`'s results for the README: each figure in a light and a dark
variant (`docs/bench/<name>-light.png`, `-dark.png`), picked by GitHub with `<picture>`.

    python3 scripts/plot/plot_bench.py results/bench/bench_full.csv results/bench/bench_*.csv

Later CSVs win: each (dataset, algorithm, library)'s timings are taken from the last CSV that has
any (as `summarize.py --latest`), so CSVs re-running some algorithms after changes give the
current picture. Figures:

* `speedup`: NetworKit's time over sparx's, every compared algorithm on every dataset, on a
  diverging scale (blue: sparx faster, red: NetworKit faster, gray: even);
* `walltime`: both libraries' times on the two largest graphs (by nodes, by edges), log scale;
* `sparx-only`: the algorithms NetworKit has no counterpart of, on every dataset they ran on.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# the reference palette's slots (validated light and dark: see the dataviz method)
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e4e3df",
        sparx="#2a78d6", networkit="#eb6834", cool="#2a78d6", warm="#e34948", mid="#f0efec",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8f8e88", grid="#33332f",
        sparx="#3987e5", networkit="#d95926", cool="#3987e5", warm="#e66767", mid="#383835",
    ),
}

# compared algorithms, in the figures' order (sparx's name -> NetworKit's timings it's compared
# with, and the label)
COMPARED = [
    ("components", "components", "connected components"),
    ("biconnectivity", "biconnectivity", "biconnected components"),
    ("k_core", "k_core", "k-cores"),
    ("clustering", "clustering", "clustering coefficients"),
    ("louvain", "louvain", "Louvain"),
    ("leiden", "leiden", "Leiden"),
    ("label_propagation", "label_propagation", "label propagation (semi-sync.)"),
    ("label_propagation_async", "label_propagation", "label propagation (async.)"),
    ("pagerank", "pagerank", "PageRank"),
    ("bfs", "bfs", "BFS"),
    ("sssp", "sssp", "shortest paths (unweighted)"),
    ("kadabra", "kadabra", "betweenness (KADABRA)"),
    ("betweenness", "betweenness", "betweenness (exact)"),
    ("closeness", "closeness", "closeness (exact)"),
    ("harmonic", "harmonic", "harmonic (exact)"),
]
SPARX_ONLY = [
    ("triangles", "triangle counts"),
    ("k_truss", "k-trusses"),
    ("hyperball", "HyperBall"),
    ("spectral", "spectral clustering"),
    ("spectral_exact", "spectral clustering (exact)"),
    ("degree_assortativity", "degree assortativity"),
]


def load(paths: list[Path]):
    """Best time of each (dataset, algorithm, library), from the last CSV that has it; the
    datasets' sizes and order."""
    rows = []
    for i, path in enumerate(paths):
        with open(path) as f:
            rows += [(i, r) for r in csv.DictReader(f)]
    last: dict[tuple[str, str, str], int] = {}
    for i, r in rows:
        last[(r["dataset"], r["algorithm"], r["library"])] = i
    best: dict[tuple[str, str, str], float] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for i, r in rows:
        key = (r["dataset"], r["algorithm"], r["library"])
        sizes[r["dataset"]] = (int(r["nodes"]), int(r["edges"]))
        if last[key] == i and r["status"] == "ok":
            best[key] = min(best.get(key, math.inf), float(r["seconds"]))
    datasets = sorted(sizes, key=lambda d: (sizes[d][1], sizes[d][0]))
    return best, sizes, datasets


def human(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def fmt_tick(s: float, _=None) -> str:
    if s < 1:
        return f"{s * 1000:g} ms"
    return f"{s:g} s" if s < 120 else f"{s / 60:g} min"


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9)


def luminance(hex_color: str) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def speedup(best, sizes, datasets, t, out: Path) -> None:
    rows = [(s, n, label) for s, n, label in COMPARED
            if any((d, s, "sparx") in best and (d, n, "networkit") in best for d in datasets)]
    cmap = LinearSegmentedColormap.from_list("diverging", [t["warm"], t["mid"], t["cool"]])
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(rows) + 2.2), facecolor=t["surface"])
    style(ax, t)
    # log10 of the speedup, clamped to 0.1x..10x for the colors (the cells say the value)
    limit = 1.0
    for y, (s, n, _) in enumerate(rows):
        for x, d in enumerate(datasets):
            a, b = best.get((d, s, "sparx")), best.get((d, n, "networkit"))
            if a is None or b is None:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=t["surface"], lw=0))
                ax.text(x, y, "–", ha="center", va="center", color=t["muted"], fontsize=9)
                continue
            ratio = b / a
            v = max(-limit, min(limit, math.log10(ratio)))
            fill = matplotlib.colors.to_hex(cmap((v + limit) / (2 * limit)))
            # 2px surface gap between cells
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=fill, ec=t["surface"], lw=2))
            label = f"{ratio:.0f}×" if ratio >= 99.5 else f"{ratio:.1f}×" if ratio >= 9.95 else f"{ratio:.2f}×"
            # the text token reading best on the cell
            ink = max((t["ink"], t["surface"]), key=lambda c: contrast(c, fill))
            ax.text(x, y, label, ha="center", va="center", color=ink, fontsize=8.5)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [f"{d}\n{human(sizes[d][0])} nodes\n{human(sizes[d][1])} edges" for d in datasets],
        fontsize=8.5, color=t["ink2"],
    )
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, _, label in rows], color=t["ink"], fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.015, "NetworKit's time ÷ sparx's (best of 3 to 5 runs, one past a minute; 32 threads). "
             "Blue: sparx faster; red: NetworKit faster; colors saturate at 10×; – : not run "
             "(size caps).", color=t["ink2"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def walltime(best, sizes, datasets, t, out: Path) -> None:
    largest = [max(datasets, key=lambda d: sizes[d][0]), max(datasets, key=lambda d: sizes[d][1])]
    if largest[0] == largest[1]:
        largest = largest[:1]
    fig, axes = plt.subplots(1, len(largest), figsize=(6.2 * len(largest), 6.4),
                             facecolor=t["surface"], sharey=True)
    axes = [axes] if len(largest) == 1 else axes
    rows = [(s, n, label) for s, n, label in COMPARED
            if any((d, s, "sparx") in best and (d, n, "networkit") in best for d in largest)]
    h = 0.36
    for ax, d in zip(axes, largest):
        style(ax, t)
        ax.set_xscale("log")
        ax.grid(axis="x", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for y, (s, n, _) in enumerate(rows):
            for off, lib, key in ((-h / 2 - 0.02, "sparx", s), (h / 2 + 0.02, "networkit", n)):
                v = best.get((d, key, lib))
                if v is not None:
                    ax.barh(y + off, v, height=h, color=t[lib], lw=0)
        ax.set_title(f"{d}  ({human(sizes[d][0])} nodes, {human(sizes[d][1])} edges)",
                     color=t["ink"], fontsize=10.5, loc="left")
        ax.set_xlabel("wall time (s, log scale; shorter is faster)", color=t["ink2"], fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_tick))
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for _, _, label in rows], color=t["ink"], fontsize=9.5)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["sparx"]), plt.Rectangle((0, 0), 1, 1, color=t["networkit"])]
    fig.legend(handles, ["sparx", "NetworKit"], loc="upper right", ncol=2, frameon=False,
               labelcolor=t["ink"], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def sparx_only(best, sizes, datasets, t, out: Path) -> None:
    algorithms = [(a, label) for a, label in SPARX_ONLY if any((d, a, "sparx") in best for d in datasets)]
    cols = 3
    rows = math.ceil(len(algorithms) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows), facecolor=t["surface"])
    axes = list(axes.flat)
    for ax, (a, label) in zip(axes, algorithms):
        style(ax, t)
        ran = [d for d in datasets if (d, a, "sparx") in best]
        ax.barh(range(len(ran)), [best[(d, a, "sparx")] for d in ran], height=0.6, color=t["sparx"], lw=0)
        ax.set_xscale("log")
        ax.grid(axis="x", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(range(len(ran)))
        ax.set_yticklabels([f"{d} ({human(sizes[d][1])})" for d in ran], color=t["ink2"], fontsize=8)
        ax.set_ylim(len(ran) - 0.5, -0.5)
        ax.set_title(label, color=t["ink"], fontsize=10, loc="left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_tick))
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[len(algorithms):]:
        ax.set_visible(False)
    fig.text(0.01, 0.01, "sparx's wall time (log scale; edge counts in parentheses), 32 threads. "
             "Datasets past an algorithm's size caps aren't run.", color=t["ink2"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "bench")
    args = p.parse_args()
    best, sizes, datasets = load(args.csvs)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        speedup(best, sizes, datasets, t, args.out / f"speedup-{name}.png")
        walltime(best, sizes, datasets, t, args.out / f"walltime-{name}.png")
        sparx_only(best, sizes, datasets, t, args.out / f"sparx-only-{name}.png")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
