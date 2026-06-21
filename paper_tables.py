"""Aggregate all method metrics into consolidated markdown tables for the paper."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ["PBMC3K", "PBMC68K", "PANCREAS"]
# display name -> source file, internal method key
METHODS = [
    ("raw", "metrics.csv", "raw_low_depth"),
    ("kNN", "metrics_extra.csv", "knn_supervised"),
    ("AE", "metrics.csv", "sae"),
    ("cVAE", "metrics.csv", "cvae"),
    ("MLP", "metrics.csv", "mlp"),
    ("scVI", "metrics_scvi_real.csv", "scvi_real"),
    ("DCA", "metrics_dca_real.csv", "dca_real"),
    ("DepthDiff", "metrics.csv", "depthdiff"),
]
DEPTHS = [0.25, 0.5, 0.75]


def load(dataset):
    cache = {}
    for _, fname, _ in METHODS:
        p = ROOT / dataset / "results" / fname
        if p.exists() and fname not in cache:
            cache[fname] = pd.read_csv(p)
    return cache


def value(cache, fname, key, depth, metric):
    df = cache.get(fname)
    if df is None:
        return None
    row = df[(df["method"] == key) & (df["depth"].round(2) == round(depth, 2))]
    return None if row.empty else float(row.iloc[0][metric])


def table(metric, higher_better=True):
    head = "| 数据集 | 深度 | " + " | ".join(m[0] for m in METHODS) + " |"
    sep = "|" + "---|" * (len(METHODS) + 2)
    lines = [head, sep]
    for ds in DATASETS:
        cache = load(ds)
        for d in DEPTHS:
            vals = [value(cache, f, k, d, metric) for _, f, k in METHODS]
            best = (max if higher_better else min)(v for v in vals if v is not None)
            cells = []
            for v in vals:
                if v is None:
                    cells.append("—")
                elif abs(v - best) < 1e-9:
                    cells.append(f"**{v:.3f}**")
                else:
                    cells.append(f"{v:.3f}")
            lines.append(f"| {ds} | {int(d*100)}% | " + " | ".join(cells) + " |")
    return "\n".join(lines)


print("### Pearson（表达重构，越高越好；粗体=该行最优）\n")
print(table("pearson", higher_better=True))
print("\n\n### DEG top-100 overlap（差异表达信号，越高越好）\n")
print(table("deg_top100_overlap", higher_better=True))
print("\n\n### RMSE（越低越好）\n")
print(table("rmse", higher_better=False))
