"""MAGIC baseline (van Dijk et al. 2018) -- unsupervised graph-diffusion imputation.

Runs the official magic-impute package on the log1p(CP10k) low-depth matrix (the
same space as the high-depth target). Like scVI/DCA, MAGIC is UNSUPERVISED (never
sees paired high-depth) and outputs DENSE smoothed values, so it is reported
separately from the supervised main comparison (see paper section 4.2). Metrics
reuse evaluate.py and the identical KMeans reference labels for comparability.

Run in the gcn_cdm env (magic-impute already installed there).
"""
import warnings

import numpy as np
import pandas as pd
import magic
from sklearn.cluster import KMeans

import config
from evaluate import (
    deg_overlap,
    logfc_metrics,
    marker_metrics,
    module_score_pearson,
    prediction_cluster_metrics,
)
from utils import load_processed, pearson_flat, rmse, set_seed

warnings.filterwarnings("ignore")
METHOD = "magic"


def main():
    set_seed()
    data = load_processed()
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    rows = []
    for depth in config.DEPTHS:
        pct = int(round(depth * 100))
        x_low = data.lows[depth].astype(np.float32)  # all cells, log1p(CP10k)
        op = magic.MAGIC(random_state=config.SEED, verbose=0)
        Y = np.asarray(op.fit_transform(x_low), dtype=np.float32)
        pred = Y[data.test_idx]
        cm = prediction_cluster_metrics(labels, pred)
        marker_spearman, marker_overlap = marker_metrics(target, pred, labels)
        logfc_pearson, logfc_spearman, deg_direction = logfc_metrics(target, pred, labels)
        rows.append({
            "depth": depth,
            "method": METHOD,
            "rmse": rmse(pred, target),
            "pearson": pearson_flat(pred, target),
            "cluster_ari": cm["cluster_ari"],
            "cluster_nmi": cm["cluster_nmi"],
            "marker_spearman": marker_spearman,
            "marker_top50_overlap": marker_overlap,
            "deg_top100_overlap": deg_overlap(target, pred, labels),
            "logfc_pearson": logfc_pearson,
            "logfc_spearman": logfc_spearman,
            "deg_direction_consistency": deg_direction,
            "module_score_pearson": module_score_pearson(target, pred, labels),
        })
        print(f"depth {pct}% done | pred zeros% {float((pred == 0).mean()):.3f}")

    df = pd.DataFrame(rows).round(3)
    out = config.RESULTS_DIR / "metrics_magic.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
