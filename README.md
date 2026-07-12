# DepthDiff

**Restoring low-depth single-cell RNA-seq signals via diffusion denoising.**

DepthDiff is a sequencing-depth-conditioned deep-learning framework that enhances
low-depth scRNA-seq expression profiles toward paired high-depth references. It is
trained with a **diffusion-based denoising objective** but, unlike conventional
diffusion models, performs a **single forward pass at inference** (no iterative
reverse sampling), making it fast and scalable to atlas-sized data. The low-depth
profile `x_low` and the sequencing-depth ratio `d` are used as conditions, and the
model predicts the residual toward the high-depth profile.

This repository contains the full, reproducible pipeline (data processing,
training, baselines, evaluation) and the result tables for three datasets.

---

## Quick Start

```bash
# 1. clone
git clone https://github.com/yunduomenghuale/DepthDiff.git
cd DepthDiff

# 2. install dependencies (conda recommended)
conda create -n depthdiff python=3.10 -y
conda activate depthdiff
pip install -r PBMC3K/requirements.txt
# optional: only needed to reproduce the MAGIC baseline
pip install magic-impute

# 3. run the full pipeline on the smallest dataset (PBMC3K)
cd PBMC3K
python run_all.py
```

`run_all.py` runs the whole workflow end to end and needs **no command-line
arguments**: it downloads the public PBMC3K data, builds the fixed-UMI low-depth
benchmark, trains DepthDiff and all baselines, and writes metrics to `results/`.
Random seeds are fixed in `config.py`. On a single consumer GPU the PBMC3K run
finishes in a few minutes.

For a step-by-step walkthrough and how to enhance **your own** low-depth matrix,
see **[TUTORIAL.md](TUTORIAL.md)**.

---

## Repository structure

| Path | Contents |
|---|---|
| `PBMC3K/`, `PBMC68K/`, `PANCREAS/` | Self-contained pipeline + result metrics for each dataset |
| `PBMC68K/bench_runtime.py` | Reproduces the inference-runtime benchmark (Table 5 in the paper) |
| `alra_official.R`, `alra_runner.R` | ALRA baseline (R) |
| `TUTORIAL.md` | Hands-on quick-start and custom-data inference example |

Each dataset directory contains the same scripts (shown for `PBMC3K/`):

- **Pipeline:** `data_processing.py` → `train_depthdiff.py` / `train_baselines.py` → `evaluate.py` (or one-shot `run_all.py`)
- **Model & shared code:** `models.py`, `diffusion.py`, `config.py`, `utils.py`
- **Baselines:** `train_baselines.py` (MLP / cVAE / DCA-like AE), `magic_baseline.py`, `extra_baselines.py` (kNN), `scvi_baseline.py`, `dca_baseline.py`, `alra_baseline.py`
- **Extended experiments:** `ablation.py`, `cross_dataset.py` (PBMC3K only), `citeseq_validation.py` (PBMC3K only)

Each dataset also has its own `README.md` describing the benchmark setup, model
details, and the full list of evaluation metrics.

---

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1 (CUDA build recommended)
- numpy, scipy, pandas, scikit-learn, matplotlib, tqdm (see `requirements.txt`)
- Optional: `magic-impute` (MAGIC baseline), `scvi-tools` (official scVI baseline)

A single consumer GPU (e.g. NVIDIA RTX 4060) is sufficient for all experiments.

---

## Data and checkpoints

To keep the repository lightweight, large **regenerable** files are not committed
and are rebuilt locally by the scripts above:

- `*/data/raw/` — raw public datasets (PBMC3K / PBMC68K from 10x Genomics; pancreas from GEO: GSE84133; CITE-seq from GEO: GSE100866), downloaded by `data_processing.py`
- `*/data/processed/*.npz` — paired high/low-depth matrices
- `*/checkpoints/*.pt` — trained model weights

The result tables (`results/*.csv`) and small processing summaries are included so
results can be inspected without rerunning the pipeline.

---

## Citation

If you use DepthDiff, please cite the associated paper (details to be added upon
publication).
