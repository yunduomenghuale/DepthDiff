"""Generate distribution-level data for the redesigned figures (run per dataset).

Saves results/figure_data.npz with:
  - percell_<method>_<pct>: per-cell Pearson (pred vs high-depth target) over test cells
  - pred_<method>_50, target_50: prediction / target matrices at 50% depth
    (only for small datasets, used for density + per-gene logFC scatter)
Covers raw + supervised methods (kNN, AE, cVAE, MLP, DepthDiff); the official
unsupervised scVI/DCA appear only in the aggregate score heatmap.
"""
import numpy as np
import magic
from sklearn.neighbors import NearestNeighbors

import config
from diffusion import make_diffusion
from evaluate import load_baselines, load_depthdiff, predict_baseline, predict_depthdiff
from utils import get_device, load_processed, set_seed

MAX_CELLS_FOR_MATRIX = 3000  # save full pred matrices only for small datasets


def percell_pearson(pred, target):
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    num = (pc * tc).sum(axis=1)
    den = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1)) + 1e-12
    return (num / den).astype(np.float32)


def main():
    set_seed()
    device = get_device()
    data = load_processed()
    n_genes = data.high.shape[1]
    target = data.high[data.test_idx]
    dd = load_depthdiff(n_genes, device)
    diffusion = make_diffusion(device)
    baselines = load_baselines(n_genes, device)  # sae, cvae, mlp

    save = {}
    for depth in config.DEPTHS:
        pct = int(round(depth * 100))
        x_low = data.lows[depth][data.test_idx]
        preds = {"raw": x_low.astype(np.float32)}
        preds["depthdiff"] = predict_depthdiff(dd, diffusion, x_low, depth, device)
        for name, model in baselines.items():
            preds[name] = predict_baseline(name, model, x_low, depth, device)
        nn = NearestNeighbors(n_neighbors=15).fit(data.lows[depth][data.train_idx])
        _, idx = nn.kneighbors(x_low)
        preds["knn"] = data.high[data.train_idx][idx].mean(axis=1).astype(np.float32)
        preds["magic"] = np.asarray(
            magic.MAGIC(random_state=config.SEED, verbose=0).fit_transform(x_low), dtype=np.float32)

        for name, pred in preds.items():
            save[f"percell_{name}_{pct}"] = percell_pearson(pred, target)
            save[f"pred_{name}_{pct}"] = pred.astype(np.float32)
        save[f"target_{pct}"] = target.astype(np.float32)
        print(f"depth {pct}% done ({len(preds)} methods)")

    out = config.RESULTS_DIR / "figure_data.npz"
    np.savez_compressed(out, **save)
    print(f"Saved {out} ({len(save)} arrays)")


if __name__ == "__main__":
    main()
