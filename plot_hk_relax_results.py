#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot HKPR evaluation results (Matplotlib only).

Input CSV schema (one row per run):
    dataset,t,eps,seed,conductance,cluster_size,volume,runtime_secs

Usage:
    python plot_hkpr_eval.py --csv hkpr_results.csv --dataset graph1
    # omit --dataset to plot all datasets (separately)
"""

import argparse
import os
from typing import Optional, Iterable, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install numpy")

try:
    import pandas as pd
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install pandas")

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install matplotlib")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# 95% confidence interval
def ci95(arr: np.ndarray) -> Tuple[float, float]:
    """Return half-width of 95% CI using normal approx: 1.96 * std/sqrt(n)."""
    n = len(arr)
    if n <= 1:
        return (0.0, 0.0)
    hw = 1.96 * np.std(arr, ddof=1) / np.sqrt(n)
    return (-hw, hw)

def group_stats(df: pd.DataFrame, value_col: str, by: Iterable[str]):
    g = df.groupby(list(by))[value_col]
    stats = g.agg(['mean', 'count', 'std']).reset_index()
    # 95% CI half width
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count']).replace(0, np.nan))
    stats['ci'] = stats['ci'].fillna(0.0)
    return stats

def plot_mean_ci_vs_t(df: pd.DataFrame, value_col: str, title: str, ylabel: str,
                      outpath: str, ylog: bool = False):
    """
    Overlay mean ± 95% CI vs t for each eps (Matplotlib default colors).
    """
    stats = group_stats(df, value_col=value_col, by=['eps', 't'])
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (eps, sub) in enumerate(stats.groupby('eps', sort=True)):
        sub = sub.sort_values('t')
        ax.plot(sub['t'], sub['mean'], marker='o', label=f"eps={eps}")
        ax.fill_between(sub['t'], sub['mean'] - sub['ci'], sub['mean'] + sub['ci'], alpha=0.2)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

    if ylog:
        ax.set_yscale('log')
    ax.legend(title="ε")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_boxplots_conductance(df: pd.DataFrame, outdir: str, dataset: str):
    """
    Boxplots of conductance across seeds per t, one subplot per eps.
    """
    eps_vals = sorted(df['eps'].unique())
    n = len(eps_vals)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4.5*rows), squeeze=False, sharey=True)

    for idx, eps in enumerate(eps_vals):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = df[df['eps'] == eps]
        # Ensure consistent t order
        t_vals = sorted(sub['t'].unique())
        data = [sub[sub['t'] == t]['conductance'].values for t in t_vals]
        ax.boxplot(data, tick_labels=[str(t) for t in t_vals], showfliers=False)
        ax.set_title(f"{dataset} — Conductance by t (eps={eps})")
        ax.set_xlabel("t")
        if c == 0:
            ax.set_ylabel("Conductance")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Hide any empty subplots
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"res_hk_relax_{dataset}_conductance_boxplots.png"),
                dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_runtime_scatter(df: pd.DataFrame, outpath: str):
    """
    Scatter: conductance vs runtime (colored by eps, marker by t bucket).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, sub in df.groupby('eps', sort=True):
        ax.scatter(sub['runtime_secs'], sub['conductance'], alpha=0.7, label=f"eps={eps}")
    ax.set_xlabel("Runtime (μs)")
    ax.set_ylabel("Conductance")
    ax.set_title("Conductance vs Runtime")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="ε")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)


def make_plots_for_dataset(df: pd.DataFrame, dataset: str, outdir: str) -> None:
    dfd = df[df['dataset'] == dataset].copy()
    if dfd.empty:
        print(f"[WARN] No rows for dataset={dataset}")
        return

    # Core plots
    plot_mean_ci_vs_t(
        dfd, value_col='conductance',
        title=f"{dataset} — conductance vs t (mean ± 95% CI)",
        ylabel="conductance",
        outpath=os.path.join(outdir, f"res_hk_relax_{dataset}_conductance_vs_t.png"),
        ylog=True  # often helpful
    )

    plot_boxplots_conductance(dfd, outdir, dataset)

    plot_mean_ci_vs_t(
        dfd, value_col='cluster_size',
        title=f"{dataset} — cluster size vs t (mean ± 95% CI)",
        ylabel="cluster size",
        outpath=os.path.join(outdir, f"res_hk_relax_{dataset}_cluster_size_vs_t.png"),
        ylog=False
    )

    plot_mean_ci_vs_t(
        dfd, value_col='runtime_secs',
        title=f"{dataset} — runtime vs t (mean ± 95% CI)",
        ylabel="runtime (μs)",
        outpath=os.path.join(outdir, f"res_hk_relax_{dataset}_runtime_vs_t.png"),
        ylog=False
    )

    plot_runtime_scatter(
        dfd,
        outpath=os.path.join(outdir, f"res_hk_relax_{dataset}_conductance_vs_runtime.png")
    )


def main():
    ap = argparse.ArgumentParser(description="Plot HKPR evaluation results (Matplotlib only).")
    ap.add_argument("--csv", required=True, help="Path to hkpr_results.csv")
    ap.add_argument("--dataset", default=None, help="Dataset name; if omitted, plots per dataset found")
    ap.add_argument("--outdir", default="plots", help="Directory to save plots")
    ap.add_argument("--min_size", type=int, default=10, help="Minimum cluster size to keep (safety filter)")
    ap.add_argument("--min_volume", type=int, default=10, help="Minimum cluster volume to keep (safety filter)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # load data
    df = pd.read_csv(args.csv)

    # check columns for sanity T.T
    required = {"dataset","t","eps","seed","conductance","cluster_size","volume","runtime_secs"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {sorted(missing)}")

    # safety filter (no trivial runs allowed)
    df = df[(df["cluster_size"] >= args.min_size) & (df["volume"] >= args.min_volume)].copy()
    if df.empty:
        raise SystemExit("No rows left after filtering for min_size/min_volume.")

    # sort for nice legends
    df['t'] = df['t'].astype(float)
    df['eps'] = df['eps'].astype(float)

    if args.dataset is not None:
        make_plots_for_dataset(df, args.dataset, args.outdir)
    else:
        for ds in sorted(df['dataset'].unique()):
            make_plots_for_dataset(df, ds, args.outdir)

    print(f"Plots saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
