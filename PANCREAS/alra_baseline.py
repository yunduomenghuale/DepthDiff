"""ALRA baseline (Linderman et al. 2022) -- unsupervised, zero-preserving low-rank
imputation, using the OFFICIAL R implementation (KlugerLab/ALRA, alra_official.R).

Unlike MAGIC/scVI/DCA, ALRA preserves zeros (it thresholds the low-rank
reconstruction per gene), so it is a fairer unsupervised comparison on the sparse
reconstruction metric. Still unsupervised (never sees paired high-depth) -> reported
with the other unsupervised methods (see paper section 4.2).

Pipeline: export low-depth raw counts (test cells) -> Rscript runs ALRA
(normalize_data = log1p(CP10k), same space as target) -> read imputed -> metrics.
Run in gcn_cdm; needs R (rsvd, Matrix) available.
"""
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite, mmread
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
METHOD = "alra"
REPO = Path(__file__).resolve().parents[1]
RSCRIPT = r"C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe"
# ASCII-only scratch dir: R / scipy cannot reliably open the CJK repo path.
TMP = Path("C:/alra_tmp")


def main():
    set_seed()
    data = load_processed()
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    TMP.mkdir(exist_ok=True)
    shutil.copy(REPO / "alra_official.R", TMP / "alra_official.R")
    shutil.copy(REPO / "alra_runner.R", TMP / "alra_runner.R")

    pcts = []
    for depth in config.DEPTHS:
        pct = int(round(depth * 100)); pcts.append(str(pct))
        raw = sparse.load_npz(config.PROCESSED_DIR / f"raw_low_{pct}.npz").tocsr()[data.test_idx]
        with open(TMP / f"_alra_in_{pct}.mtx", "wb") as fh:
            mmwrite(fh, sparse.csr_matrix(raw), field="integer")

    subprocess.run([RSCRIPT, str(TMP / "alra_runner.R"), str(TMP / "alra_official.R"),
                    str(TMP), ",".join(pcts)], check=True)

    rows = []
    for depth in config.DEPTHS:
        pct = int(round(depth * 100))
        with open(TMP / f"_alra_out_{pct}.mtx", "rb") as fh:
            pred = np.asarray(mmread(fh).todense(), dtype=np.float32)
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
        for f in (TMP / f"_alra_in_{pct}.mtx", TMP / f"_alra_out_{pct}.mtx"):
            f.unlink(missing_ok=True)

    df = pd.DataFrame(rows).round(3)
    out = config.RESULTS_DIR / "metrics_alra.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
