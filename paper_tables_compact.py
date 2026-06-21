"""Compact cross-condition averaged tables (technical + biological), DiffFormer-style.
Averages each metric over all 9 (dataset x depth) conditions per method."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = {"PBMC3K": "PBMC3K", "PBMC68K": "PBMC68K", "Pancreas": "PANCREAS"}
METHODS = [  # display, file, key
    ("Raw", "metrics.csv", "raw_low_depth"),
    ("kNN", "metrics_extra.csv", "knn_supervised"),
    ("AE", "metrics.csv", "sae"),
    ("cVAE", "metrics.csv", "cvae"),
    ("MLP", "metrics.csv", "mlp"),
    ("scVI", "metrics_scvi_real.csv", "scvi_real"),
    ("DCA", "metrics_dca_real.csv", "dca_real"),
    ("DepthDiff", "metrics.csv", "depthdiff"),
]
TECH = [("rmse", "RMSE", False), ("pearson", "Pearson", True)]
BIO = [("deg_top100_overlap", "DEG overlap", True), ("logfc_spearman", "logFC-S", True),
       ("marker_spearman", "marker-S", True)]


def avg_table(metrics):
    rows = {}
    for disp, fname, key in METHODS:
        vals = {m: [] for m, _, _ in metrics}
        for folder in DATASETS.values():
            df = pd.read_csv(ROOT / folder / "results" / fname)
            sub = df[df["method"] == key]
            for m, _, _ in metrics:
                vals[m].extend(sub[m].tolist())
        rows[disp] = {m: np.mean(vals[m]) for m, _, _ in metrics}
    # markdown
    head = "| 方法 | " + " | ".join(lbl for _, lbl, _ in metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    best = {}
    for m, _, hi in metrics:
        col = {d: rows[d][m] for d in rows}
        best[m] = (max if hi else min)(col.values())
    lines = [head, sep]
    for disp, _, _ in METHODS:
        cells = []
        for m, _, _ in metrics:
            v = rows[disp][m]
            cells.append(f"**{v:.3f}**" if abs(v - best[m]) < 1e-9 else f"{v:.3f}")
        lines.append(f"| {disp} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


print("### 技术指标（跨 9 条件均值）\n")
print(avg_table(TECH))
print("\n\n### 生物信号指标（跨 9 条件均值）\n")
print(avg_table(BIO))
