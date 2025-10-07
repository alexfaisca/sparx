#!/usr/bin/env python3
# plot_massif.py
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install matplotlib")
import argparse, csv, math, re
from pathlib import Path

# Filenames like "3_graph_0_5.massif.out" → target=3, dataset="graph_0_5"
FNAME_RE = re.compile(r"^(\d+)_(.+)\.massif\.out$")
SNAP_RE  = re.compile(r"^snapshot=(\d+)")
KV_RE    = re.compile(r"^(mem_heap_B|mem_heap_extra_B|mem_stacks_B|mem_heapB|mem_heap_extraB|mem_stackB|time)=(\d+)$")

def norm_key(s: str) -> str:
    return s.lower().replace("_", "").replace("stacks", "stack")

def parse_massif_peak(path: Path) -> int:
    """Return peak of heap+extra+stacks in bytes for a massif.out file."""
    cur = {}
    peak = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = SNAP_RE.match(line)
            if m:
                if {"mem_heap_B","mem_heap_extra_B","mem_stacks_B"} <= cur.keys():
                    total = cur["mem_heap_B"] + cur["mem_heap_extra_B"] + cur["mem_stacks_B"]
                    peak = max(peak, total)
                cur = {}
                continue
            m = KV_RE.match(line)
            if m:
                k, v = m.group(1), int(m.group(2))
                nk = norm_key(k)
                if nk == "memheapb":
                    cur["mem_heap_B"] = v
                elif nk == "memheapextrab":
                    cur["mem_heap_extra_B"] = v
                elif nk == "memstackb":
                    cur["mem_stacks_B"] = v
    if {"mem_heap_B","mem_heap_extra_B","mem_stacks_B"} <= cur.keys():
        total = cur["mem_heap_B"] + cur["mem_heap_extra_B"] + cur["mem_stacks_B"]
        peak = max(peak, total)
    return peak

def human_bytes(n: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB"]
    v = float(n); i = 0
    while v >= 1024 and i < len(units)-1:
        v /= 1024.0; i += 1
    return f"{v:.2f} {units[i]}"

def load_sizes_csv(csv_path: Path):
    """
    Expect header: dataset,V,E
    dataset: string key that must match the dataset extracted from filename (the bit after N_)
    V,E: integers
    """
    sizes = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ds = row["dataset"].strip()
            V = int(row["V"].replace("_","").replace(",",""))
            E = int(row["E"].replace("_","").replace(",",""))
            sizes[ds] = (V, E)
    return sizes

def compute_x(V: int, E: int, mode: str) -> float:
    if mode == "V":
        return V
    if mode == "E":
        return V
    if mode == "VE":
        return V + E
    elif mode == "VElogE":
        # natural log; use --log-base to change if you want
        return V + E * math.log(max(E, 1))
    else:
        raise ValueError("unknown x-mode: " + mode)

def collect_files(inputs, pattern):
    files = set()
    pats = [p.strip() for p in pattern.split(",") if p.strip()]
    for root in inputs:
        P = Path(root)
        if P.is_file():
            files.add(P.resolve())
        elif P.is_dir():
            for pat in pats:
                for f in P.rglob(pat):
                    if f.is_file():
                        files.add(f.resolve())
        else:
            print(f"[warn] path not found: {root}")
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(description="Scatter plots of Massif peaks per target (N_*.massif.out).")
    ap.add_argument("inputs", nargs="+", help="massif files or directories")
    ap.add_argument("--pattern", default="*.massif.out", help="glob pattern(s) for directories (comma-separated)")
    ap.add_argument("--sizes", required=True, help="CSV with columns: dataset,V,E")
    ap.add_argument("--x", choices=["V", "E", "VE","VElogE"], default="VE", help="x-axis formula")
    ap.add_argument("-o", "--outdir", default="massif_scatter", help="output directory")
    ap.add_argument("--logx", action="store_true", help="log scale x-axis")
    ap.add_argument("--logy", action="store_true", help="log scale y-axis")
    ap.add_argument("--labels", action="store_true", help="label points with dataset names")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sizes = load_sizes_csv(Path(args.sizes))
    files = collect_files(args.inputs, args.pattern)

    # group: target → list of dict rows
    groups = {i: [] for i in range(0, 11)}  # 0..10
    skipped = 0
    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            skipped += 1
            continue
        tgt = int(m.group(1))
        dataset_full = m.group(2)
        dataset  = dataset_full.split('.', 1)[0]
        peak = parse_massif_peak(f)

        if dataset not in sizes:
            print(f"[warn] no size entry for dataset '{dataset}', skipping {f.name}")
            continue

        V, E = sizes[dataset]
        x = compute_x(V, E, args.x)
        groups.setdefault(tgt, []).append({
            "dataset": dataset, "file": f.name, "V": V, "E": E, "x": x, "peak": peak
        })

    if skipped:
        print(f"[info] skipped {skipped} files that did not match 'N_*.massif.out'")

    # one scatter per target
    for tgt in sorted(groups.keys()):
        rows = groups[tgt]
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        ys = [r["peak"] for r in rows]
        labels = [r["dataset"] for r in rows]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(xs, ys, s=18)
        xlab = "|V| + |E|" if args.x == "VE" else ("|V| + |E| · ln |E|" if args.x == "VElogE" else ("|V|" if args.x == "V" else "E"))
        ax.set_title(f"Massif Peaks — Profile {tgt}")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Peak Allocated Bytes")
        if args.logx: ax.set_xscale("log")
        if args.logy: ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.5)

        if args.labels:
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)

        fig.tight_layout()
        png = outdir / f"target_{tgt}_scatter_{args.x}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)

        # per-target CSV
        csv_path = outdir / f"target_{tgt}_scatter_{args.x}.csv"
        with csv_path.open("w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["dataset","file","V","E","x","peak_bytes","peak_h"])
            for r in rows:
                w.writerow([r["dataset"], r["file"], r["V"], r["E"], r["x"], r["peak"], human_bytes(r["peak"])])
        print(f"[ok] target {tgt}: {len(rows)} points → {png}  (table: {csv_path})")

    print(f"Done. Output in: {outdir}")

if __name__ == "__main__":
    main()
