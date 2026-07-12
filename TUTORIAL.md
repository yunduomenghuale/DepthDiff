# DepthDiff Tutorial

This tutorial walks through (1) reproducing a dataset benchmark end to end and
(2) applying a trained DepthDiff model to your own low-depth matrix.

All commands are run from inside a dataset directory (e.g. `PBMC3K/`), because the
scripts import the local `config.py` and read/write `checkpoints/`, `data/`, and
`results/` relative to that directory.

---

## 1. Reproduce a benchmark end to end

```bash
cd PBMC3K
pip install -r requirements.txt

# one command: download -> simulate low depth -> train -> evaluate
python run_all.py
```

Or run the stages individually:

```bash
python data_processing.py     # download PBMC3K, QC, HVG selection, fixed-UMI low-depth pairs
python train_depthdiff.py     # train the depth-conditioned DepthDiff model
python train_baselines.py     # train MLP / cVAE / DCA-like AE with the same split & budget
python evaluate.py            # evaluate Raw, DepthDiff, and baselines on the held-out test set
```

Outputs:

- `checkpoints/fixed_umi_depthdiff_best.pt` — trained DepthDiff weights
- `results/metrics.csv` — RMSE, Pearson, DEG overlap, logFC/marker Spearman, ARI/NMI, …

Optional extras (PBMC3K): `python ablation.py`, `python cross_dataset.py`,
`python citeseq_validation.py`. Optional baselines: `python magic_baseline.py`
(needs `magic-impute`).

---

## 2. Enhance your own low-depth data

DepthDiff operates on a **fixed gene panel** in **log1p(CP10k)** space. Your input
matrix must therefore use the same genes, in the same order, as the panel the model
was trained on. That panel is saved during preprocessing at
`data/processed/selected_genes.csv` (2,000 highly variable genes).

Prepare `x_low` as a `float32` array of shape `(n_cells, 2000)`:
normalize each cell to 10,000 counts (CP10k), apply `log1p`, and subset/reorder the
columns to match `selected_genes.csv`. Genes that are not in the training panel are
outside the model's output space and cannot be enhanced by a model trained on that
panel (retrain on your own panel if needed — see the paper, Section 3.8).

Save the following as `enhance.py` inside a dataset directory and run it:

```python
import numpy as np
import torch

import config
from utils import get_device
from diffusion import make_diffusion
from evaluate import load_depthdiff, predict_depthdiff

device = get_device()

# x_low: (n_cells, 2000) float32, log1p(CP10k), columns matching selected_genes.csv
x_low = np.load("my_low_depth.npy").astype("float32")
n_genes = x_low.shape[1]

model = load_depthdiff(n_genes, device)      # loads checkpoints/fixed_umi_depthdiff_best.pt
diffusion = make_diffusion(device)

depth = 0.5                                   # sequencing-depth ratio in (0, 1]
x_high = predict_depthdiff(model, diffusion, x_low, depth, device)

np.save("my_enhanced.npy", x_high)
print("enhanced matrix:", x_high.shape)
```

Notes:

- `depth` is the estimated fraction of the target (high) depth that your data
  represents; values around 0.25–0.75 were used in the paper.
- Inference is a single forward pass per cell and is fast (≈65,000 cells/s on an
  RTX 4060; see the runtime benchmark below).

---

## 3. Reproduce the inference-runtime benchmark

```bash
cd PBMC68K
python bench_runtime.py
```

This times single-step DepthDiff, full reverse-diffusion sampling, cVAE, MLP, and
MAGIC on the PBMC68K test set (Table 5 in the paper). It requires the processed
data and trained checkpoints to be present (run `python run_all.py` first).
