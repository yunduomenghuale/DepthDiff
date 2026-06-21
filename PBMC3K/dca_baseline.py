"""Faithful DCA baseline: a negative-binomial denoising autoencoder.

The original DCA package is TF1-based and not installable on a modern stack, so
this is a faithful PyTorch reimplementation of its core model: an autoencoder
that reconstructs the observed counts under a negative-binomial likelihood with
a per-gene dispersion. Like the official DCA (and unlike DepthDiff/MLP) it is
UNSUPERVISED -- it only sees low-depth counts and never the high-depth target.

A single unified model spans all depths (depth supplied as an input feature).
The denoised mean fractions are scaled to TARGET_SUM and log1p'd so predictions
live in the same space as the log-normalized high-depth target, and metrics
reuse evaluate.py's functions and identical KMeans labels for comparability.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.cluster import KMeans
from torch import nn

import config
from evaluate import (
    deg_overlap,
    logfc_metrics,
    marker_metrics,
    module_score_pearson,
    prediction_cluster_metrics,
)
from utils import get_device, load_processed, pearson_flat, rmse, set_seed

METHOD = "dca_real"


class NBAutoencoder(nn.Module):
    def __init__(self, n_genes, hidden_dim=512, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, n_genes)        # -> softmax fractions
        self.log_theta = nn.Parameter(torch.zeros(n_genes))    # per-gene dispersion

    def forward(self, x_log, depth, size_factor):
        z = self.encoder(torch.cat([x_log, depth], dim=1))
        h = self.decoder(torch.cat([z, depth], dim=1))
        frac = torch.softmax(self.mean_head(h), dim=1)         # sums to 1 per cell
        mu = frac * size_factor                                # NB mean in count space
        theta = self.log_theta.exp().clamp(1e-4, 1e4)
        return mu, theta, frac


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative-binomial negative log-likelihood (mean mu, dispersion theta)."""
    t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
    t2 = (theta + x) * torch.log1p(mu / (theta + eps)) + x * (torch.log(theta + eps) - torch.log(mu + eps))
    return (t1 + t2).sum(dim=1).mean()


def stack_depths(raw_low, cell_idx):
    """Stack the given cells' raw low counts across all depths (sparse) + depth vector."""
    blocks, depth_cov = [], []
    for depth in config.DEPTHS:
        blocks.append(raw_low[int(round(depth * 100))][cell_idx])
        depth_cov.append(np.full(len(cell_idx), float(depth), dtype=np.float32))
    return sparse.vstack(blocks).tocsr(), np.concatenate(depth_cov)


def iter_batches(x_sparse, depth_arr, device, shuffle):
    """Yield (x_log, depth, size_factor, x_counts) by fast batch row-slicing."""
    n = x_sparse.shape[0]
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, config.BATCH_SIZE):
        rows = order[start : start + config.BATCH_SIZE]
        x = torch.tensor(x_sparse[rows].toarray(), dtype=torch.float32, device=device)
        sf = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        depth_t = torch.tensor(depth_arr[rows], device=device).view(-1, 1)
        yield torch.log1p(x), depth_t, sf, x


def main():
    set_seed()
    device = get_device()
    data = load_processed()
    n_genes = data.high.shape[1]
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    raw_low = {
        int(round(d * 100)): sparse.load_npz(config.PROCESSED_DIR / f"raw_low_{int(round(d * 100))}.npz").tocsr()
        for d in config.DEPTHS
    }

    train_x, train_d = stack_depths(raw_low, data.train_idx)
    val_x, val_d = stack_depths(raw_low, data.val_idx)

    model = NBAutoencoder(n_genes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for x_log, depth_t, sf, x in iter_batches(train_x, train_d, device, shuffle=True):
            if x.size(0) < 2:  # BatchNorm needs >1 sample in train mode
                continue
            mu, theta, _ = model(x_log, depth_t, sf)
            loss = nb_nll(x, mu, theta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            vtot = vn = 0.0
            for x_log, depth_t, sf, x in iter_batches(val_x, val_d, device, shuffle=False):
                mu, theta, _ = model(x_log, depth_t, sf)
                vtot += nb_nll(x, mu, theta).item() * x.size(0)
                vn += x.size(0)
            val = vtot / vn
        print(f"epoch {epoch:03d}/{config.EPOCHS} val_nb_nll={val:.4f}")
        if val < best_val:
            best_val, best_state, patience = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    rows = []
    with torch.no_grad():
        for depth in config.DEPTHS:
            low = raw_low[int(round(depth * 100))][data.test_idx].toarray().astype(np.float32)
            preds = []
            for start in range(0, low.shape[0], config.BATCH_SIZE):
                xb = torch.tensor(low[start : start + config.BATCH_SIZE], device=device)
                sf = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
                depth_t = torch.full((xb.size(0), 1), float(depth), device=device)
                _, _, frac = model(torch.log1p(xb), depth_t, sf)
                preds.append((frac * config.TARGET_SUM).cpu().numpy())
            pred = np.log1p(np.concatenate(preds, axis=0)).astype(np.float32)
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
    out = config.RESULTS_DIR / "metrics_dca_real.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
