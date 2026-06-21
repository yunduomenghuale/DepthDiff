import tarfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.model_selection import train_test_split

import config
from utils import depth_to_name, ensure_dirs, set_seed


def find_10x_matrix_dir():
    candidates = []
    for matrix_path in config.RAW_DIR.rglob("matrix.mtx"):
        folder = matrix_path.parent
        if (folder / "genes.tsv").exists() and (folder / "barcodes.tsv").exists():
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError(f"Could not find matrix.mtx/genes.tsv/barcodes.tsv under {config.RAW_DIR}")
    return sorted(candidates, key=lambda path: str(path))[0]


def download_pbmc68k() -> None:
    ensure_dirs()
    try:
        matrix_dir = find_10x_matrix_dir()
        print(f"Found extracted PBMC68K data at {matrix_dir}")
        return
    except FileNotFoundError:
        pass
    if not config.PBMC68K_TAR.exists():
        last_error = None
        for url in config.PBMC68K_URLS:
            try:
                print(f"Downloading PBMC68K from {url}")
                urllib.request.urlretrieve(url, config.PBMC68K_TAR)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if config.PBMC68K_TAR.exists():
                    config.PBMC68K_TAR.unlink()
                print(f"Download failed from {url}: {exc}")
        if last_error is not None:
            raise last_error
    print("Extracting PBMC68K archive")
    with tarfile.open(config.PBMC68K_TAR, "r:gz") as tar:
        tar.extractall(config.RAW_DIR)


def normalize_log1p(counts: sparse.spmatrix) -> np.ndarray:
    counts = counts.astype(np.float32).tocsr()
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    totals[totals == 0] = 1.0
    scale = config.TARGET_SUM / totals
    norm = sparse.diags(scale).dot(counts)
    norm.data = np.log1p(norm.data)
    return norm.toarray().astype(np.float32)


def compute_qc_metrics(counts: sparse.csr_matrix, gene_names: np.ndarray, barcodes: np.ndarray) -> pd.DataFrame:
    total_counts = np.asarray(counts.sum(axis=1)).reshape(-1)
    n_genes = np.asarray((counts > 0).sum(axis=1)).reshape(-1)
    mito_mask = np.char.upper(gene_names.astype(str)).astype(str)
    mito_mask = np.char.startswith(mito_mask, "MT-")
    if mito_mask.any():
        mito_counts = np.asarray(counts[:, mito_mask].sum(axis=1)).reshape(-1)
    else:
        mito_counts = np.zeros(counts.shape[0], dtype=np.float32)
    pct_mito = mito_counts / np.maximum(total_counts, 1.0)
    return pd.DataFrame(
        {
            "barcode": barcodes,
            "total_counts": total_counts,
            "n_genes": n_genes,
            "mito_fraction": pct_mito,
        }
    )


def save_qc_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    rows = []
    for label, df in [("before_qc", before), ("after_qc", after)]:
        rows.append(
            {
                "stage": label,
                "n_cells": len(df),
                "median_total_counts": df["total_counts"].median(),
                "median_detected_genes": df["n_genes"].median(),
                "mean_mito_fraction": df["mito_fraction"].mean(),
                "median_mito_fraction": df["mito_fraction"].median(),
            }
        )
    pd.DataFrame(rows).round(4).to_csv(config.QC_SUMMARY_CSV, index=False)


def filter_cells_and_genes(counts: sparse.csr_matrix, gene_names: np.ndarray, barcodes: np.ndarray):
    qc_before = compute_qc_metrics(counts, gene_names, barcodes)
    max_genes = qc_before["n_genes"].quantile(config.MAX_GENES_PER_CELL_QUANTILE)
    max_counts = qc_before["total_counts"].quantile(config.MAX_COUNTS_PER_CELL_QUANTILE)
    keep_cells = (
        (qc_before["n_genes"] >= config.MIN_GENES_PER_CELL)
        & (qc_before["n_genes"] <= max_genes)
        & (qc_before["total_counts"] >= config.MIN_COUNTS_PER_CELL)
        & (qc_before["total_counts"] <= max_counts)
        & (qc_before["mito_fraction"] <= config.MAX_MITO_FRACTION)
    ).to_numpy()
    counts = counts[keep_cells].tocsr()
    barcodes = barcodes[keep_cells]
    qc_after_cells = compute_qc_metrics(counts, gene_names, barcodes)
    save_qc_summary(qc_before, qc_after_cells)
    qc_after_cells.round(4).to_csv(config.QC_METRICS_CSV, index=False)

    cells_per_gene = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    keep_genes = cells_per_gene >= config.MIN_CELLS_PER_GENE
    counts = counts[:, keep_genes].tocsr()
    gene_names = gene_names[keep_genes]
    return counts, gene_names, barcodes


def make_unique(names: np.ndarray) -> np.ndarray:
    seen = {}
    unique = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name}-{seen[name]}")
    return np.array(unique)


def select_highly_variable_genes(counts: sparse.csr_matrix, n_top: int) -> np.ndarray:
    counts = counts.astype(np.float32).tocsr()
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    totals[totals == 0] = 1.0
    scale = config.TARGET_SUM / totals
    log_norm = sparse.diags(scale).dot(counts).tocsr()
    log_norm.data = np.log1p(log_norm.data)
    means = np.asarray(log_norm.sum(axis=0)).reshape(-1) / log_norm.shape[0]
    square_sums = np.asarray(log_norm.power(2).sum(axis=0)).reshape(-1)
    variances = square_sums / log_norm.shape[0] - means**2
    dispersion = variances / (means + 1e-6)
    detected_cells = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    valid = (means > 0) & (detected_cells >= config.HVG_MIN_CELLS)
    scores = np.full(counts.shape[1], -np.inf, dtype=np.float32)
    if valid.sum() < n_top:
        valid = means > 0

    valid_idx = np.where(valid)[0]
    valid_means = means[valid_idx]
    valid_dispersion = dispersion[valid_idx]
    order = np.argsort(valid_means)
    ranked_idx = valid_idx[order]
    ranked_dispersion = valid_dispersion[order]
    bins = np.array_split(np.arange(len(ranked_idx)), config.HVG_N_BINS)
    for bin_positions in bins:
        if len(bin_positions) == 0:
            continue
        gene_idx = ranked_idx[bin_positions]
        bin_disp = ranked_dispersion[bin_positions]
        bin_mean = bin_disp.mean()
        bin_std = bin_disp.std() + 1e-6
        scores[gene_idx] = (bin_disp - bin_mean) / bin_std
    n_top = min(n_top, counts.shape[1])
    return np.argsort(-scores)[:n_top]


def gene_capture_probabilities(counts: sparse.csr_matrix, rng: np.random.Generator) -> np.ndarray:
    gene_means = np.asarray(counts.mean(axis=0)).reshape(-1)
    scaled = np.log1p(gene_means)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
    low_expression_dropout = config.LOW_EXPRESSION_DROPOUT_STRENGTH / (
        1.0 + np.exp((scaled - config.LOW_EXPRESSION_DROPOUT_MIDPOINT) * 12.0)
    )
    gene_effect = rng.lognormal(mean=0.0, sigma=config.GENE_CAPTURE_SIGMA, size=counts.shape[1])
    gene_effect = gene_effect / gene_effect.mean()
    return np.clip((1.0 - low_expression_dropout) * gene_effect, 0.02, 1.0).astype(np.float32)


def fixed_umi_counts(counts: sparse.csr_matrix, target_umi: int, seed: int, gene_prob: np.ndarray) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    counts = counts.tocsr()
    rows = []
    cols = []
    vals = []
    for row_idx in range(counts.shape[0]):
        start, end = counts.indptr[row_idx], counts.indptr[row_idx + 1]
        row_cols = counts.indices[start:end]
        row_vals = counts.data[start:end].astype(np.int64)
        if row_vals.size == 0:
            continue

        thinned = rng.binomial(row_vals, gene_prob[row_cols]).astype(np.int64)
        keep = thinned > 0
        row_cols = row_cols[keep]
        thinned = thinned[keep]
        total = int(thinned.sum())
        if total == 0:
            continue

        sample_n = min(int(target_umi), total)
        molecules = np.repeat(row_cols, thinned)
        sampled_cols = rng.choice(molecules, size=sample_n, replace=False)
        unique_cols, sampled_vals = np.unique(sampled_cols, return_counts=True)
        rows.extend([row_idx] * len(unique_cols))
        cols.extend(unique_cols.tolist())
        vals.extend(sampled_vals.astype(np.float32).tolist())

    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=counts.shape,
        dtype=np.float32,
    )


def count_summary(name: str, counts: sparse.csr_matrix, depth: float | None = None) -> dict:
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    detected = np.asarray((counts > 0).sum(axis=1)).reshape(-1)
    zero_fraction = 1.0 - counts.nnz / float(counts.shape[0] * counts.shape[1])
    row = {
        "matrix": name,
        "depth": depth if depth is not None else 1.0,
        "n_cells": counts.shape[0],
        "n_genes": counts.shape[1],
        "mean_library_size": totals.mean(),
        "median_library_size": np.median(totals),
        "mean_detected_genes": detected.mean(),
        "median_detected_genes": np.median(detected),
        "zero_fraction": zero_fraction,
    }
    return row


def make_split(n_cells: int):
    all_idx = np.arange(n_cells)
    train_idx, temp_idx = train_test_split(
        all_idx,
        train_size=config.TRAIN_FRACTION,
        random_state=config.SEED,
        shuffle=True,
    )
    val_relative = config.VAL_FRACTION / (1.0 - config.TRAIN_FRACTION)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_relative,
        random_state=config.SEED,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def save_benchmark_payload(path, high, gene_names, barcodes, train_idx, val_idx, test_idx, lows):
    payload = {
        "high": high,
        "gene_names": gene_names,
        "barcodes": barcodes,
        "depths": np.array(config.DEPTHS, dtype=np.float32),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    for depth, low_counts in lows.items():
        payload[depth_to_name(depth)] = normalize_log1p(low_counts)
    np.savez_compressed(path, **payload)


def save_raw_counts(high_counts, lows) -> None:
    """Save raw (un-normalized) HVG counts for count-based baselines (scVI/DCA).

    Stored separately as scipy sparse matrices so the main log-normalized
    benchmark payload and all existing loaders stay unchanged.
    """
    sparse.save_npz(
        config.PROCESSED_DIR / "raw_high.npz", high_counts.tocsr().astype(np.float32)
    )
    for depth, low in lows.items():
        sparse.save_npz(
            config.PROCESSED_DIR / f"raw_low_{int(round(depth * 100))}.npz",
            low.tocsr().astype(np.float32),
        )


def preprocess() -> None:
    set_seed()
    download_pbmc68k()
    print("Reading 10x matrix without scanpy/anndata")
    matrix_dir = find_10x_matrix_dir()
    matrix_path = matrix_dir / "matrix.mtx"
    genes_path = matrix_dir / "genes.tsv"
    barcodes_path = matrix_dir / "barcodes.tsv"

    # Pass an opened binary stream so non-ASCII (e.g. CJK) paths work on Windows;
    # scipy's fast_matrix_market C++ reader cannot open Unicode paths directly.
    with open(matrix_path, "rb") as matrix_file:
        raw_matrix = mmread(matrix_file).T.tocsr().astype(np.float32)
    genes = pd.read_csv(genes_path, sep="\t", header=None)
    gene_names = make_unique(genes.iloc[:, 1].astype(str).to_numpy())
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()

    counts, gene_names, barcodes = filter_cells_and_genes(raw_matrix, gene_names, barcodes)
    full_filtered_summary = count_summary("full_filtered_counts", counts)
    train_idx_raw, val_idx_raw, test_idx_raw = make_split(counts.shape[0])
    hvg_idx = select_highly_variable_genes(counts[train_idx_raw], config.N_TOP_GENES)
    counts = counts[:, hvg_idx].tocsr()
    gene_names = gene_names[hvg_idx]
    pd.DataFrame({"gene": gene_names}).to_csv(config.SELECTED_GENES_CSV, index=False)

    print(f"Cells: {counts.shape[0]}, genes: {counts.shape[1]}")
    high = normalize_log1p(counts)
    train_idx, val_idx, test_idx = train_idx_raw, val_idx_raw, test_idx_raw

    train_counts = counts[train_idx].tocsr()
    hvg_library = np.asarray(train_counts.sum(axis=1)).reshape(-1)
    median_hvg_library = float(np.median(hvg_library))
    gene_effect_rng = np.random.default_rng(config.SEED + 9000)
    gene_effect = gene_capture_probabilities(train_counts, gene_effect_rng)
    gene_prob = np.clip(
        1.0 - config.FIXED_UMI_GENE_DROPOUT_STRENGTH * (1.0 - gene_effect / gene_effect.max()),
        0.05,
        1.0,
    )
    fixed_summary = [full_filtered_summary, count_summary("hvg_high_depth", counts)]
    fixed_lows = {}
    for depth in config.FIXED_UMI_TARGET_RATIOS:
        target_umi = max(1, int(round(median_hvg_library * depth)))
        low_counts = fixed_umi_counts(counts, target_umi, seed=config.SEED + 10000 + int(depth * 1000), gene_prob=gene_prob)
        fixed_lows[depth] = low_counts
        row = count_summary(depth_to_name(depth), low_counts, depth)
        row["target_umi_per_cell"] = target_umi
        fixed_summary.append(row)
        print(f"Prepared fixed-UMI low-depth matrix for {depth:.0%} target_umi={target_umi}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fixed_summary).round(4).to_csv(config.FIXED_UMI_DEPTH_SUMMARY_CSV, index=False)

    save_benchmark_payload(config.PROCESSED_NPZ, high, gene_names, barcodes, train_idx, val_idx, test_idx, fixed_lows)
    save_raw_counts(counts, fixed_lows)
    print(f"Saved fixed-UMI data to {config.PROCESSED_NPZ}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    preprocess()
