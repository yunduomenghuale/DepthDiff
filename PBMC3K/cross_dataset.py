"""Cross-dataset generalization: train DepthDiff on PBMC68K, apply to unseen
PBMC3K test cells (and vice-versa as a within-dataset upper bound), on the
shared HVG panel. Tests whether the learned low->high mapping transfers to an
independent dataset/donors -- a real generalization check.

Run from the PBMC3K folder with the gcn_cdm env.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

import config
from diffusion import make_diffusion
from evaluate import deg_overlap, logfc_metrics, predict_depthdiff
from models import make_depthdiff
from train_depthdiff import train_one_epoch, validate
from utils import (
    DepthPairDataset,
    EarlyStopping,
    depth_to_name,
    get_device,
    load_processed,
    pearson_flat,
    rmse,
    set_seed,
)

ROOT = config.ROOT.parent
NPZ = {"PBMC3K": "pbmc3k_fixed_umi_pairs.npz", "PBMC68K": "pbmc68k_fixed_umi_pairs.npz"}


def load_dataset(folder):
    d = np.load(ROOT / folder / "data" / "processed" / NPZ[folder], allow_pickle=True)
    lows = {float(x): d[depth_to_name(float(x))].astype(np.float32) for x in config.DEPTHS}
    return {
        "high": d["high"].astype(np.float32), "lows": lows,
        "train": d["train_idx"], "val": d["val_idx"], "test": d["test_idx"],
        "genes": d["gene_names"].astype(str),
    }


def reindex(ds, common):
    pos = {g: i for i, g in enumerate(ds["genes"])}
    idx = np.array([pos[g] for g in common])
    return ds["high"][:, idx], {d: ds["lows"][d][:, idx] for d in ds["lows"]}


def train_model(high, lows, train_idx, val_idx, n_genes, device):
    set_seed()
    model = make_depthdiff(n_genes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    diffusion = make_diffusion(device)
    tl = DataLoader(DepthPairDataset(high, lows, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    vl = DataLoader(DepthPairDataset(high, lows, val_idx), batch_size=config.BATCH_SIZE, shuffle=False)
    stopper, best = EarlyStopping(config.PATIENCE), None
    for _ in range(1, config.EPOCHS + 1):
        train_one_epoch(model, diffusion, tl, opt, device)
        if stopper.step(validate(model, diffusion, vl, device)):
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if stopper.should_stop:
            break
    if best:
        model.load_state_dict(best)
    return model, diffusion


def evaluate_on(model, diffusion, high_t, lows_t, test_idx, labels, device, tag):
    target = high_t[test_idx]
    rows = []
    for depth in config.DEPTHS:
        pred = predict_depthdiff(model, diffusion, lows_t[depth][test_idx], depth, device)
        _, lf_s, _ = logfc_metrics(target, pred, labels)
        rows.append({"depth": depth, "method": tag, "rmse": round(rmse(pred, target), 3),
                     "pearson": round(pearson_flat(pred, target), 3),
                     "deg": deg_overlap(target, pred, labels), "logfc_s": round(lf_s, 3)})
    return rows


def main():
    set_seed()
    device = get_device()
    d3, d68 = load_dataset("PBMC3K"), load_dataset("PBMC68K")
    common = [g for g in d3["genes"] if g in set(d68["genes"])]
    print(f"shared HVGs: {len(common)} (PBMC3K {len(d3['genes'])}, PBMC68K {len(d68['genes'])})")

    high3, lows3 = reindex(d3, common)
    high68, lows68 = reindex(d68, common)
    n_genes = len(common)

    target3 = high3[d3["test"]]
    labels3 = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target3)

    # raw baseline on PBMC3K test (common genes)
    rows = []
    for depth in config.DEPTHS:
        pred = lows3[depth][d3["test"]]
        _, lf_s, _ = logfc_metrics(target3, pred, labels3)
        rows.append({"depth": depth, "method": "raw", "rmse": round(rmse(pred, target3), 3),
                     "pearson": round(pearson_flat(pred, target3), 3),
                     "deg": deg_overlap(target3, pred, labels3), "logfc_s": round(lf_s, 3)})

    print("training within-dataset model (PBMC3K -> PBMC3K)...")
    m_in, diff = train_model(high3, lows3, d3["train"], d3["val"], n_genes, device)
    rows += evaluate_on(m_in, diff, high3, lows3, d3["test"], labels3, device, "DepthDiff (within 3K->3K)")

    print("training cross-dataset model (PBMC68K -> PBMC3K)...")
    m_cross, diff = train_model(high68, lows68, d68["train"], d68["val"], n_genes, device)
    rows += evaluate_on(m_cross, diff, high3, lows3, d3["test"], labels3, device, "DepthDiff (cross 68K->3K)")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["depth", "method"])
    out = config.RESULTS_DIR / "cross_dataset.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
