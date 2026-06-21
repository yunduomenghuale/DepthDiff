# PBMC68K DepthDiff Experiment

This folder contains the PBMC68K benchmark used for low-depth single-cell RNA-seq enhancement.

## Final Setup

- Dataset: Fresh 68k PBMCs Donor A 10x Genomics count matrix.
- Benchmark: fixed-UMI low-depth simulation.
- Depths: 25%, 50%, and 75% of the HVG-space pseudo-high-depth median library size.
- Main model: unified depth-conditioned DepthDiff with a residual MLP denoiser.
- Baselines: raw low-depth, MLP, DCA-like autoencoder, and scVI-like VAE.
- Fair comparison: DepthDiff and all trainable baselines are trained as unified depth-conditioned models.

## Files

- `data_processing.py`: downloads PBMC68K, applies QC, selects HVGs, and creates the fixed-UMI benchmark.
- `train_depthdiff.py`: trains one unified DepthDiff checkpoint for all depths.
- `train_baselines.py`: trains one unified checkpoint per baseline with the same split, epochs, patience, and optimizer.
- `evaluate.py`: evaluates raw low-depth, DepthDiff, and baselines on held-out cells.
- `run_all.py`: runs the final workflow in order.
- `config.py`, `models.py`, `utils.py`: shared configuration, model definitions, and utilities.
- `requirements.txt`: Python dependencies.

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

Or step by step:

```bash
python data_processing.py
python train_depthdiff.py
python train_baselines.py
python evaluate.py
```

All scripts run without parameters. Random seeds are fixed in `config.py`.

## Outputs

- `data/processed/pbmc68k_fixed_umi_pairs.npz`: paired high-depth and low-depth matrices.
- `data/processed/qc_metrics.csv`: per-cell QC metrics after filtering.
- `data/processed/qc_summary.csv`: before/after QC summary.
- `data/processed/fixed_umi_depth_simulation_summary.csv`: library size, detected genes, and zero fraction for each depth.
- `data/processed/selected_genes.csv`: selected HVG names.
- `checkpoints/fixed_umi_depthdiff_best.pt`: best unified DepthDiff checkpoint.
- `checkpoints/fixed_umi_baseline_*_best.pt`: best unified baseline checkpoints.
- `results/metrics_fixed_umi.csv`: final benchmark metrics.
- `results/metrics.csv`: copy of the latest metrics.
- `results/*.png`: summary figures.

## Benchmark

The filtered PBMC68K count matrix is treated as pseudo-high-depth data. Each cell is downsampled to fixed UMI targets corresponding to 25%, 50%, and 75% of the pseudo-high-depth median library size in HVG space.

The simulation also includes gene-specific capture variation and low-expression dropout to better approximate realistic low-depth sparsity.

## DepthDiff

DepthDiff is an SR3-style **conditional denoising diffusion model**. The low-depth
profile `x_low` is the conditioning observation, and the model learns the reverse
diffusion process that turns Gaussian noise into the high-depth profile, conditioned
on `x_low` and the sequencing-depth ratio `d`. A single depth-conditioned model is
shared across all depths.

Forward process (training) — noise is added to the diffusion target `x0`
(`x0 = x_high - x_low` by default, i.e. the residual towards high depth):

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
eps_theta(x_t, x_low, t, d) -> predicted noise
```

Reverse process (inference) — a real reverse-diffusion chain is run from pure
noise, conditioned on `x_low` and `d`:

```text
x_T ~ N(0, I)
for t = T-1 ... 0:  x_{t-1} = step(eps_theta(x_t, x_low, t, d))
pred_high = clamp(x_low + x0_hat, 0)
```

Sampling uses DDPM ancestral sampling by default (DDIM is available via
`config.SAMPLER`). Point metrics use the posterior mean over
`config.EVAL_NUM_SAMPLES` reverse-diffusion samples.

The training objective combines:

- the standard DDPM noise-prediction loss (depth-weighted across 25/50/75%)
- a cell-wise correlation preservation loss on the reconstructed `x0`
- a high-variance gene preservation loss on the reconstructed `x0`

For fairness, all trainable baselines are also trained as unified depth-conditioned models over the same low-depth/high-depth pairs.

## Metrics

`evaluate.py` reports:

- `rmse`
- `pearson`
- `cluster_ari`
- `cluster_nmi`
- `marker_spearman`
- `marker_top50_overlap`
- `deg_top100_overlap`
- `logfc_pearson`
- `logfc_spearman`
- `deg_direction_consistency`
- `module_score_pearson`
