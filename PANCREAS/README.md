# PANCREAS DepthDiff Benchmark

This folder runs the same fixed-UMI low-depth enhancement benchmark used for PBMC3K/PBMC68K on the Baron human pancreas scRNA-seq dataset.

Dataset:

- Baron et al. human pancreas inDrop scRNA-seq
- GEO accession: GSE84133
- Downloaded automatically from GEO as `GSE84133_RAW.tar`
- The script reads the human `*_umifm_counts.csv.gz` count matrices and builds paired low-depth/high-depth data.

Pipeline:

```bash
cd ~/Desktop/biology/PANCREAS
python data_processing.py
python train_depthdiff.py
python train_baselines.py
python evaluate.py
```

Or run everything:

```bash
python run_all.py
```

Benchmark settings:

- Depths: 25%, 50%, 75%
- Low-depth simulation: fixed-UMI downsampling with train-only parameter estimation
- Target: high-depth log-normalized expression
- Main model: unified depth-conditioned DepthDiff
- Baselines: MLP, DCA-like autoencoder, scVI-like VAE, raw low-depth reference

Main outputs:

- Processed pairs: `data/processed/pancreas_fixed_umi_pairs.npz`
- Depth simulation summary: `data/processed/fixed_umi_depth_simulation_summary.csv`
- Checkpoints: `checkpoints/`
- Metrics: `results/metrics_fixed_umi.csv`
