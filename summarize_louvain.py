#!/usr/bin/env python3
"""
Summarize Louvain runs across datasets.

Inputs
------
1) louvain.csv with columns:
   run_id,dataset,modularity,runtime_micros,levels,passes_total,threads
2) louvain_passes.csv with columns:
   run_id,pass_idx,iters_for_pass,coms_in_pass,runtime_micros

Output
------
Pipe-separated table to stdout:
<dataset> | <levels> | <total_passes> | <passes_at_first_level> | <modularity>
"""

import sys
import pandas as pd

# --------- config / paths ----------
LOUVAIN_CSV = "./results/louvain.csv"
PASSES_CSV = "./results/louvain_passes.csv"

if len(sys.argv) >= 2:
    LOUVAIN_CSV = sys.argv[1]
if len(sys.argv) >= 3:
    PASSES_CSV = sys.argv[2]

# --------- load ----------
lv = pd.read_csv(LOUVAIN_CSV)
lp = pd.read_csv(PASSES_CSV)

# Keep only the columns we need from louvain.csv
lv = lv[["run_id", "dataset", "modularity", "levels"]].copy()

# ---------- totals per run_id from louvain_passes.csv ----------
# Sum iters_for_pass over all passes for each run_id
sum_per_run = (
    lp.groupby("run_id", as_index=False)["iters_for_pass"]
      .sum()
      .rename(columns={"iters_for_pass": "total_passes_run"})
)

# First-level (pass_idx == 0) iters per run_id
first_level_per_run = (
    lp[lp["pass_idx"] == 0][["run_id", "iters_for_pass"]]
      .rename(columns={"iters_for_pass": "passes_at_first_level_run"})
)

# Attach dataset to per-run stats
sum_per_run = sum_per_run.merge(lv[["run_id", "dataset"]], on="run_id", how="left")
first_level_per_run = first_level_per_run.merge(
    lv[["run_id", "dataset"]], on="run_id", how="left"
)

# ---------- aggregate per dataset ----------
# From louvain.csv: average levels and modularity per dataset
agg_lv = (
    lv.groupby("dataset", as_index=False)
      .agg(levels=("levels", "mean"),
           modularity=("modularity", "mean"))
)

# From passes: average total_passes and passes_at_first_level per dataset
agg_passes_total = (
    sum_per_run.groupby("dataset", as_index=False)
               .agg(total_passes=("total_passes_run", "mean"))
)

agg_passes_first = (
    first_level_per_run.groupby("dataset", as_index=False)
                       .agg(passes_at_first_level=("passes_at_first_level_run", "mean"))
)

# Merge everything
out = (agg_lv
       .merge(agg_passes_total, on="dataset", how="left")
       .merge(agg_passes_first, on="dataset", how="left")
       [["dataset", "levels", "total_passes", "passes_at_first_level", "modularity"]]
       .sort_values("dataset")
)

# Pretty print
print("dataset | levels | total_passes | passes_at_first_level | modularity")
for _, r in out.iterrows():
    # print levels/total/passes as floats to allow non-integers after averaging
    print(f"{r['dataset']} | "
          f"{r['levels']:.2f} | "
          f"{r['total_passes']:.2f} | "
          f"{r['passes_at_first_level']:.2f} | "
          f"{r['modularity']:.6f}")
