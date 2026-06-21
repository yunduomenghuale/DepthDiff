"""Paired statistical tests of DepthDiff vs each baseline on per-cell Pearson
(matching the house-style significance table). Uses figure_data.npz per dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATASETS = {"PBMC3K": "PBMC3K", "PBMC68K": "PBMC68K", "Pancreas": "PANCREAS"}
BASELINES = [("raw", "Raw"), ("knn", "kNN"), ("sae", "AE"), ("cvae", "cVAE"), ("mlp", "MLP")]


def cohen_d_paired(a, b):
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


rows = []
for ds, folder in DATASETS.items():
    fd = np.load(ROOT / folder / "results" / "figure_data.npz")
    # pool per-cell Pearson across the three depths
    dd = np.concatenate([fd[f"percell_depthdiff_{p}"] for p in (25, 50, 75)])
    for key, label in BASELINES:
        bl = np.concatenate([fd[f"percell_{key}_{p}"] for p in (25, 50, 75)])
        t, p = stats.ttest_rel(dd, bl)
        rows.append({
            "dataset": ds, "comparison": f"DepthDiff vs {label}", "n_cells": len(dd),
            "mean_DepthDiff": round(float(dd.mean()), 3), "mean_baseline": round(float(bl.mean()), 3),
            "t": round(float(t), 2), "p_value": f"{p:.2e}", "cohen_d": round(float(cohen_d_paired(dd, bl)), 3),
        })

df = pd.DataFrame(rows)
out = ROOT / "paper_stats.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f"Saved {out}")
