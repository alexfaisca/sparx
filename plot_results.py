#!/usr/bin/env python3
# plot_thread_scaling.py

# Use non-interactive backend if no display (safe on servers)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install matplotlib")

import numpy as np
import csv
import os
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"

# Nice, consistent colors per thread count
THREAD_PALETTE = {
    1:  "#1f77b4",
    2:  "#ff7f0e",
    4:  "#2ca02c",
    8:  "#d62728",
    16: "#9467bd",
    32: "#8c564b",
    64: "#e377c2",
}

# ----------------------- utilities -----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _norm_dataset_name(s: str) -> str:
    """
    Normalize dataset key so that:
    './datasets/kmer_V2a.mtx' -> 'kmer_V2a'
    'kmer_V2a.mtx' -> 'kmer_V2a'
    'kmer_V2a' -> 'kmer_V2a'
    """
    base = os.path.basename(s)
    name, _ext = os.path.splitext(base)
    return name

def load_sizes_mapping():
    """
    Read a CSV with header: dataset,V,E (comma-separated).
    Searches common locations/names:
      results/datasets_sizes.csv
      results/datasets_size.csv
      datasets_sizes.csv
      datasets_size.csv
    Returns: dict { normalized_dataset_name: (V, E) }
    """
    candidates = [
        os.path.join("results", "datasets_sizes.csv"),
        os.path.join("results", "datasets_size.csv"),
        "datasets_sizes.csv",
        "datasets_size.csv",
    ]
    sizes = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        print(f"[INFO] Using dataset sizes file: {path}")
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            # Expect columns exactly: dataset, V, E
            for row in r:
                ds = (row.get("dataset") or "").strip()
                V  = (row.get("V") or "").replace(",", "").replace("_", "").strip()
                E  = (row.get("E") or "").replace(",", "").replace("_", "").strip()
                if not ds or not V or not E:
                    continue
                try:
                    v = int(V); e = int(E)
                except ValueError:
                    continue
                # skip zero placeholders (remove if you want them included)
                # if v == 0 and e == 0:
                #     continue
                sizes[_norm_dataset_name(ds)] = (v, e)
        break
    if not sizes:
        print("[WARN] No dataset sizes file found; rows without inline n_nodes/n_edges will be skipped.")
    return sizes

def read_csv_if_exists(path):
    if not os.path.exists(path):
        print(f"[INFO] Missing file: {path} (skipping)")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def get_threads(row):
    t = row.get("threads")
    if t is None or t == "":
        return 1
    try:
        return int(t)
    except Exception:
        return 1

def to_seconds(row):
    # accept runtime_micros or runtime_secs
    if "runtime_micros" in row and row["runtime_micros"]:
        return int(row["runtime_micros"]) / 1_000_000.0
    if "runtime_secs" in row and row["runtime_secs"]:
        return float(row["runtime_secs"])
    raise ValueError("row lacks runtime_micros/runtime_secs")

def ci95(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    return 1.96 * (stdev(vals) / sqrt(n))

def get_size_for_row(row, sizes_map):
    # prefer inline if provided
    n_str = row.get("n_nodes")
    m_str = row.get("n_edges")
    if n_str and m_str:
        try:
            return int(n_str), int(m_str)
        except Exception:
            pass
    ds = row.get("dataset")
    if not ds:
        return None
    return sizes_map.get(_norm_dataset_name(ds))

def aggregate_by(points, sizes_map, t, extra_keys):
    """
    Group rows by (group_id=tuple(extra_keys values), dataset, threads).
    Compute mean runtime and 95% CI.
    Returns: dict { group_id -> list of (x=V+E, y_mean, y_ci, dataset, threads, count) }
    """
    buckets = defaultdict(list)
    for row in points:
        ds = row.get("dataset", "")
        size = get_size_for_row(row, sizes_map)
        if size is None:
            if ds:
                print(f"[WARN] No size for dataset '{ds}', skipping row.")
            continue
        V, E = size
        threads = get_threads(row)
        try:
            secs = to_seconds(row)
        except Exception:
            continue

        gid = tuple((row.get(k) or "") for k in extra_keys) if extra_keys else ("ALL",)
        x = 0
        if t == 0:
            x = V + E
        elif t == 1:
            if V != 0:
                x = V * np.log(V) + E
            else:
                x = E
        elif t == 2:
            x = E ** 1.5
        else:
            x = E
        buckets[(gid, _norm_dataset_name(ds), threads, x)].append(secs)

    series = defaultdict(list)
    for (gid, ds_norm, threads, x), vals in buckets.items():
        ym = mean(vals)
        yci = ci95(vals)
        series[gid].append((x, ym, yci, ds_norm, threads, len(vals)))
    # sort each series by x
    for gid in series:
        series[gid].sort(key=lambda t: t[0])
    return series

def plot_series(series, t, title, filename_stub, y_label="runtime (s)", logx=True, logy=True):
    ensure_dir(PLOTS_DIR)

    for gid, pts in series.items():
        from collections import defaultdict
        by_thr = defaultdict(list)
        for x, y, ci, ds, thr, count in pts:
            by_thr[thr].append((x, y, ci, ds, count))

        fig, ax = plt.subplots(figsize=(8, 5))

        # One line per thread count; let Matplotlib pick default colors
        for thr, items in sorted(by_thr.items(), key=lambda kv: kv[0]):
            items.sort(key=lambda t: t[0])  # sort by x for a clean line
            xs = [it[0] for it in items]
            ys = [it[1] for it in items]
            es = [it[2] for it in items]

            # Draw errorbar without specifying color -> uses default cycle
            cont = ax.errorbar(xs, ys, yerr=es, fmt='o-', capsize=3, alpha=0.95,
                               label=f"{thr} threads")

            # Get the line color Matplotlib chose
            line = cont.lines[0] if hasattr(cont, "lines") else cont[0]
            color = line.get_color()

            # Shaded 95% CI band if we have variance
            if any(e > 0 for e in es):
                xs_arr = np.array(xs, dtype=float)
                low = np.array([y - e for y, e in zip(ys, es)])
                high = np.array([y + e for y, e in zip(ys, es)])
                ax.fill_between(xs_arr, low, high, alpha=0.15, color=color, linewidth=0)

        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        if t == 0:
            ax.set_xlabel("|V| + |E|")
        elif t == 1:
            ax.set_xlabel("|V| * log(|V|) + |E|")
        elif t == 2:
            ax.set_xlabel("|E|^1.5")
        else:
            ax.set_xlabel("|E|")
        ax.set_ylabel(y_label)

        pretty_gid = ", ".join([g for g in gid if g]) if gid else ""
        full_title = title if pretty_gid in ("ALL", "", None) else f"{title} — {pretty_gid}"
        ax.set_title(full_title)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(title="Threads", loc='best')
        fig.tight_layout()

        suffix = "" if pretty_gid in ("ALL", "", None) else "_" + "_".join(
            s.replace("/", "_") for s in pretty_gid.split(", ")
        ).strip()
        outpath = os.path.join(PLOTS_DIR, f"{filename_stub}{suffix}.png")
        fig.savefig(outpath, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok] wrote {outpath}")

# ----------------------- main flow -----------------------

def main():
    ensure_dir(PLOTS_DIR)
    sizes_map = load_sizes_mapping()

    # K-CORE
    kcore_rows = read_csv_if_exists(os.path.join(RESULTS_DIR, "kcore.csv"))
    if kcore_rows:
        kc_series = aggregate_by(kcore_rows, sizes_map, 3, extra_keys=["algo"])
        plot_series(kc_series, 3, "K-core runtime", "kcore_runtime")

    # K-TRUSS
    ktr_rows = read_csv_if_exists(os.path.join(RESULTS_DIR, "ktruss.csv"))
    if ktr_rows:
        kt_series = aggregate_by(ktr_rows, sizes_map, 2, extra_keys=["algo"])
        plot_series(kt_series, 2, "K-truss runtime", "ktruss_runtime")

    # LOUVAIN
    louv_rows = read_csv_if_exists(os.path.join(RESULTS_DIR, "louvain.csv"))
    if louv_rows:
        lv_series = aggregate_by(louv_rows, sizes_map, 1, extra_keys=[])
        plot_series(lv_series, 1, "Louvain runtime", "louvain_runtime")

    # HYPERBALL
    hb_rows = read_csv_if_exists(os.path.join(RESULTS_DIR, "hyperball.csv"))
    if hb_rows:
        # one figure per precision_p
        hb_series = aggregate_by(hb_rows, sizes_map, 3, extra_keys=["precision_p"])
        plot_series(hb_series, 3, "HyperBall runtime", "hyperball_runtime")

    print(f"✓ Plots saved in: {os.path.abspath(PLOTS_DIR)}")

if __name__ == "__main__":
    main()
