"""Official scvi-tools scVI baseline on the same fixed-UMI benchmark.

Unlike the lightweight ``scvi_like`` reimplementation in models.py, this trains
a real scVI model (scvi-tools) on the RAW low-depth UMI counts, with the
sequencing-depth ratio supplied as a continuous covariate so a single unified
model spans all depths (matching the DepthDiff / baseline setup).

Predictions are scVI's normalized expression scaled to TARGET_SUM and log1p'd,
so they live in the same space as the log-normalized high-depth target. Metrics
reuse evaluate.py's functions and the identical KMeans reference labels, so the
numbers are directly comparable to the other methods in results/metrics.csv.

Run in an environment that has scvi-tools (e.g. the dedicated scvi_bench env).
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse
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

METHOD = "scvi_real"
N_LATENT = 10


def depth_pct(depth: float) -> int:
    return int(round(depth * 100))


def build_adata(raw_low, cell_index):
    """Stack the given cells' raw low-depth counts across all depths.

    Row (depth_i, cell j) lands at i * len(cell_index) + j.
    """
    blocks, depth_cov = [], []
    for depth in config.DEPTHS:
        blocks.append(raw_low[depth_pct(depth)][cell_index])
        depth_cov.append(np.full(len(cell_index), float(depth), dtype=np.float32))
    adata = ad.AnnData(sparse.vstack(blocks).tocsr().astype(np.float32))
    adata.layers["counts"] = adata.X.copy()
    adata.obs["depth"] = np.concatenate(depth_cov)
    return adata


def main():
    set_seed()
    scvi.settings.seed = config.SEED
    data = load_processed()
    n_cells = data.high.shape[0]
    all_idx = np.arange(n_cells)
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    raw_low = {
        depth_pct(d): sparse.load_npz(config.PROCESSED_DIR / f"raw_low_{depth_pct(d)}.npz")
        for d in config.DEPTHS
    }

    # Transductive setup: scVI sees all cells' low-depth counts (unsupervised),
    # which only advantages the baseline, making the comparison conservative.
    adata = build_adata(raw_low, all_idx)
    scvi.model.SCVI.setup_anndata(adata, layer="counts", continuous_covariate_keys=["depth"])
    model = scvi.model.SCVI(adata, n_latent=N_LATENT)
    # Large datasets give many steps/epoch, so fewer epochs suffice; rely on
    # scVI's validation early stopping in all cases.
    max_epochs = 200 if adata.n_obs <= 30000 else 100
    model.train(max_epochs=max_epochs, early_stopping=True)

    normalized = model.get_normalized_expression(
        adata, library_size=config.TARGET_SUM, return_numpy=True
    )

    rows = []
    for i, depth in enumerate(config.DEPTHS):
        test_rows = i * n_cells + data.test_idx
        pred = np.log1p(normalized[test_rows]).astype(np.float32)
        cm = prediction_cluster_metrics(labels, pred)
        marker_spearman, marker_overlap = marker_metrics(target, pred, labels)
        logfc_pearson, logfc_spearman, deg_direction = logfc_metrics(target, pred, labels)
        rows.append(
            {
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
            }
        )

    df = pd.DataFrame(rows).round(3)
    out = config.RESULTS_DIR / "metrics_scvi_real.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
