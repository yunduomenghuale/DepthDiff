from pathlib import Path

SEED = 2026
DATASET_NAME = "PBMC68K"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

PBMC68K_URLS = [
    (
        "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
        "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    ),
    (
        "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
        "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    ),
]
PBMC68K_TAR = RAW_DIR / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
PBMC68K_EXTRACTED = RAW_DIR

BENCHMARK_NAME = "fixed_umi"
PROCESSED_NPZ = PROCESSED_DIR / "pbmc68k_fixed_umi_pairs.npz"
QC_METRICS_CSV = PROCESSED_DIR / "qc_metrics.csv"
QC_SUMMARY_CSV = PROCESSED_DIR / "qc_summary.csv"
FIXED_UMI_DEPTH_SUMMARY_CSV = PROCESSED_DIR / "fixed_umi_depth_simulation_summary.csv"
DEPTH_SUMMARY_CSV = FIXED_UMI_DEPTH_SUMMARY_CSV
SELECTED_GENES_CSV = PROCESSED_DIR / "selected_genes.csv"

DEPTHS = [0.25, 0.50, 0.75]
N_TOP_GENES = 2000
TARGET_SUM = 10000.0
HVG_MIN_CELLS = 20
HVG_N_BINS = 20
MIN_GENES_PER_CELL = 200
MAX_GENES_PER_CELL_QUANTILE = 0.995
MIN_COUNTS_PER_CELL = 500
MAX_COUNTS_PER_CELL_QUANTILE = 0.995
MIN_CELLS_PER_GENE = 3
MAX_MITO_FRACTION = 0.15
GENE_CAPTURE_SIGMA = 0.25
LOW_EXPRESSION_DROPOUT_STRENGTH = 0.35
LOW_EXPRESSION_DROPOUT_MIDPOINT = 0.15
FIXED_UMI_TARGET_RATIOS = DEPTHS
FIXED_UMI_GENE_DROPOUT_STRENGTH = 0.15

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

BATCH_SIZE = 128
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

HIDDEN_DIM = 1024
LATENT_DIM = 64
DIFFUSION_STEPS = 50
SAMPLING_STEPS = 50
BETA_SCHEDULE = "cosine"         # "cosine" (alpha_bar_T ~ 0 at any T) or "linear"
BETA_START = 1e-4
BETA_END = 0.02
PREDICTION_TYPE = "x0"           # "epsilon" (predict noise) or "x0" (predict clean target)
DIFFUSION_TARGET = "residual"    # diffuse (x_high - x_low); set "x_high" for direct
SAMPLER = "single_step"          # "single_step" (default, 1 forward), "ddpm", or "ddim"
DDIM_ETA = 0.0                   # 0 = deterministic DDIM, only used when SAMPLER == "ddim"
EVAL_NUM_SAMPLES = 1             # samples to average; 1 for single-step inference

DEPTHDIFF_METHOD_NAME = "depthdiff"
DEPTH_LOSS_WEIGHTS = {
    0.25: 2.0,
    0.50: 1.5,
    0.75: 1.0,
}
AUX_X0_LOSS = True               # apply corr/var losses on the reconstructed x0
CORRELATION_LOSS_WEIGHT = 0.10
VARIANCE_LOSS_WEIGHT = 0.05
TOP_VARIANCE_FRACTION = 0.20

DEVICE = "cuda"
