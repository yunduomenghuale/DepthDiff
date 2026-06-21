"""Additional baselines on the same fixed-UMI benchmark:

- ALRA  (unsupervised, zero-preserving low-rank imputation; Linderman et al.)
  The fairest unsupervised comparator here because it re-zeros after recovery,
  so it is not structurally penalized by the sparse log-normalized target the
  way dense denoisers (scVI/DCA) are.
- kNN   (supervised non-parametric regression): for each test cell, average the
  PAIRED high-depth profiles of its nearest low-depth training cells. Apples-to-
  apples with DepthDiff/MLP (all use the paired high-depth target).

Predictions live in the same log-normalized space as the target; metrics reuse
evaluate.py's functions and identical KMeans labels for comparability.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import config
from evaluate import (
    deg_overlap,
    logfc_metrics,
    marker_metrics,
    module_score_pearson,
    prediction_cluster_metrics,
)
from utils import load_processed, pearson_flat, rmse, set_seed


def alra(A, k_max=100):
    """Adaptively-thresholded Low-Rank Approximation (log-normalized input)."""
    k_max = min(k_max, min(A.shape) - 1)
    U, s, Vt = randomized_svd(A, n_components=k_max, random_state=config.SEED)
    # ALRA k selection: largest singular-value spacing that stands out from the
    # noise level estimated on the small (upper-index) singular values.
    spacings = s[:-1] - s[1:]
    noise = spacings[len(spacings) // 2:]
    thresh = noise.mean() + 6.0 * noise.std()
    significant = np.where(spacings > thresh)[0]
    k = int(significant.max() + 1) if significant.size else 20
    k = max(2, min(k, k_max))
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    out = Ak.copy()
    for g in range(A.shape[1]):
        col = out[:, g]
        thr = abs(Ak[:, g].min())          # symmetric threshold = magnitude of most negative entry
        col[col < thr] = 0.0
        orig_nz = A[:, g][A[:, g] > 0]
        pos = col > 0
        if pos.sum() > 1 and orig_nz.size > 1:
            vals = col[pos]
            col[pos] = (vals - vals.mean()) / (vals.std() + 1e-8) * orig_nz.std() + orig_nz.mean()
        col[col < 0] = 0.0
        out[:, g] = col
    return out.astype(np.float32)


def metrics_row(depth, method, target, pred, labels):
    cm = prediction_cluster_metrics(labels, pred)
    marker_spearman, marker_overlap = marker_metrics(target, pred, labels)
    logfc_pearson, logfc_spearman, deg_direction = logfc_metrics(target, pred, labels)
    return {
        "depth": depth,
        "method": method,
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
    }


def main():
    set_seed()
    data = load_processed()
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    rows = []
    for depth in config.DEPTHS:
        low = data.lows[depth]

        # kNN supervised: neighbors in low space, average paired high targets.
        # (A faithful ALRA reimplementation over-fills zeros on this log-normalized
        #  sparse target and is not reported; use the official R ALRA if needed.)
        nn = NearestNeighbors(n_neighbors=15).fit(low[data.train_idx])
        _, idx = nn.kneighbors(low[data.test_idx])
        knn_pred = data.high[data.train_idx][idx].mean(axis=1).astype(np.float32)
        rows.append(metrics_row(depth, "knn_supervised", target, knn_pred, labels))
        print(f"depth {depth}: knn done")

    df = pd.DataFrame(rows).round(3)
    out = config.RESULTS_DIR / "metrics_extra.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
