"""Orthogonal validation via CITE-seq (GSE100866 CBMC, 13 antibodies).

Protein (ADT) is an independent measurement modality never used in training.
We simulate low sequencing depth on the RNA, enhance it with DepthDiff, and test
whether the RNA-protein correlation (degraded by low depth) is recovered toward
the high-depth level. Recovery of an orthogonal protein signal is strong evidence
that the simulation is realistic and the model restores genuine biological signal.

Run from the PBMC3K folder with the gcn_cdm env.
"""
import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats
from torch.utils.data import DataLoader

import config
from diffusion import make_diffusion
from models import make_depthdiff
from train_depthdiff import train_one_epoch, validate
from evaluate import predict_depthdiff
from utils import DepthPairDataset, EarlyStopping, get_device, set_seed

DATA = config.ROOT.parent / "citeseq_data"
ADT_CSV = DATA / "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz"
RNA_CSV = DATA / "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz"
PAIRS = {  # protein -> gene
    "CD3": "CD3E", "CD4": "CD4", "CD8": "CD8A", "CD14": "CD14", "CD16": "FCGR3A",
    "CD19": "CD19", "CD56": "NCAM1", "CD11c": "ITGAX", "CCR7": "CCR7",
}
DEPTHS = [0.25, 0.50, 0.75]


def normalize_log1p(counts):
    lib = counts.sum(1, keepdims=True); lib[lib == 0] = 1
    return np.log1p(counts / lib * config.TARGET_SUM).astype(np.float32)


def main():
    set_seed()
    device = get_device()
    rng = np.random.default_rng(config.SEED)

    adt = pd.read_csv(ADT_CSV, index_col=0)
    rna = pd.read_csv(RNA_CSV, index_col=0)
    rna = rna[adt.columns]  # align cells

    # keep predominantly-human cells, then human genes only
    human = rna.index.str.startswith("HUMAN_")
    mouse = rna.index.str.startswith("MOUSE_")
    hfrac = rna[human].sum(0) / (rna[human].sum(0) + rna[mouse].sum(0) + 1e-9)
    keep = (hfrac > 0.9).values
    rna_h = rna[human].iloc[:, keep]
    adt = adt.iloc[:, keep]
    genes = rna_h.index.str.replace("HUMAN_", "", regex=False).to_numpy()
    counts = rna_h.to_numpy().T.astype(np.float32)  # cells x genes
    protein = adt.to_numpy().T.astype(np.float32)   # cells x 13
    print(f"cells={counts.shape[0]} human_genes={counts.shape[1]} kept_human_frac>0.9")

    # gene QC + HVG, force-include marker genes
    gene_ok = (counts > 0).sum(0) >= 10
    log_all = normalize_log1p(counts)
    disp = log_all.var(0)
    disp[~gene_ok] = -1
    hvg = set(np.argsort(-disp)[:2000].tolist())
    gpos = {g: i for i, g in enumerate(genes)}
    for g in PAIRS.values():
        if g in gpos:
            hvg.add(gpos[g])
    hvg = np.array(sorted(hvg))
    genes_hvg = genes[hvg]
    high_counts = counts[:, hvg]
    print(f"HVG features={len(hvg)} (markers present: {[g for g in PAIRS.values() if g in set(genes_hvg)]})")

    high = normalize_log1p(high_counts)
    lows = {d: normalize_log1p(rng.binomial(high_counts.astype(int), d).astype(np.float32)) for d in DEPTHS}

    n = high.shape[0]
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    train_idx, val_idx, test_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    # train one depth-conditioned DepthDiff
    model = make_depthdiff(len(hvg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    diffusion = make_diffusion(device)
    tl = DataLoader(DepthPairDataset(high, lows, train_idx, depths=DEPTHS), batch_size=config.BATCH_SIZE, shuffle=True)
    vl = DataLoader(DepthPairDataset(high, lows, val_idx, depths=DEPTHS), batch_size=config.BATCH_SIZE, shuffle=False)
    stopper, best = EarlyStopping(config.PATIENCE), None
    for _ in range(config.EPOCHS):
        train_one_epoch(model, diffusion, tl, opt, device)
        if stopper.step(validate(model, diffusion, vl, device)):
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if stopper.should_stop:
            break
    if best:
        model.load_state_dict(best)

    prot_test = protein[test_idx]
    valid_pairs = [(p, g) for p, g in PAIRS.items() if g in set(genes_hvg) and p in adt.index]
    gcol = {g: i for i, g in enumerate(genes_hvg)}
    pcol = {p: i for i, p in enumerate(adt.index)}

    def mean_corr(expr):
        cs = [stats.spearmanr(expr[:, gcol[g]], prot_test[:, pcol[p]]).correlation for p, g in valid_pairs]
        return float(np.nanmean(cs))

    rows = []
    for d in DEPTHS:
        xlow = lows[d][test_idx]
        enh = predict_depthdiff(model, diffusion, xlow, d, device)
        c_high = mean_corr(high[test_idx]); c_low = mean_corr(xlow); c_enh = mean_corr(enh)
        rec = (c_enh - c_low) / (c_high - c_low + 1e-9) * 100
        rows.append({"depth": d, "corr_high": round(c_high, 3), "corr_low": round(c_low, 3),
                     "corr_enhanced": round(c_enh, 3), "recovery_%": round(rec, 1)})
        print(rows[-1])

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "citeseq_validation.csv"
    df.to_csv(out, index=False)
    print(f"pairs used: {[p for p,_ in valid_pairs]}")
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
