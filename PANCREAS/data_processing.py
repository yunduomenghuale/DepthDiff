import gzip
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

import config
from utils import depth_to_name, ensure_dirs, set_seed


def download_pancreas() -> None:
    ensure_dirs()
    if any(config.RAW_DIR.glob("*human*_umifm_counts.csv.gz")):
        print(f"Found extracted pancreas count files in {config.RAW_DIR}")
        return
    if not config.PANCREAS_TAR.exists():
        last_error = None
        for url in config.PANCREAS_URLS:
            try:
                print(f"Downloading pancreas Baron GSE84133 from {url}")
                urllib.request.urlretrieve(url, config.PANCREAS_TAR)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if config.PANCREAS_TAR.exists():
                    config.PANCREAS_TAR.unlink()
                print(f"Download failed from {url}: {exc}")
        if last_error is not None:
            raise last_error
    print("Extracting pancreas archive")
    with tarfile.open(config.PANCREAS_TAR, "r:*") as tar:
        tar.extractall(config.RAW_DIR)


def read_one_baron_csv(path):
    print(f"Reading {path.name}")
    with gzip.open(path, "rt") as handle:
        df = pd.read_csv(handle)
    if df.shape[1] <= 3:
        raise ValueError(f"Unexpected Baron pancreas table shape in {path}: {df.shape}")

    gene_names = df.columns[3:].astype(str).to_numpy()
    count_df = df.iloc[:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cell_ids = df.iloc[:, 0].astype(str).to_numpy()
    barcodes = df.iloc[:, 1].astype(str).to_numpy() if df.shape[1] > 1 else cell_ids
    cell_names = np.array(
        [f"{path.stem}_{cell_id}_{barcode}" for cell_id, barcode in zip(cell_ids, barcodes)],
        dtype=object,
    )
    counts = sparse.csr_matrix(count_df.to_numpy(dtype=np.float32))
    return counts, gene_names, cell_names


def read_pancreas_counts():
    files = sorted(config.RAW_DIR.glob("*human*_umifm_counts.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No Baron human pancreas count files under {config.RAW_DIR}")

    matrices = []
    gene_lists = []
    cell_lists = []
    for path in files:
        counts, genes, cells = read_one_baron_csv(path)
        matrices.append(counts)
        gene_lists.append(genes)
        cell_lists.append(cells)

    first_genes = gene_lists[0]
    if all(np.array_equal(first_genes, genes) for genes in gene_lists):
        counts = sparse.vstack(matrices).tocsr()
        gene_names = first_genes
    else:
        common = set(first_genes.tolist())
        for genes in gene_lists[1:]:
            common &= set(genes.tolist())
        gene_names = np.array([gene for gene in first_genes if gene in common])
        aligned = []
        for counts_i, genes_i in zip(matrices, gene_lists):
            lookup = {gene: idx for idx, gene in enumerate(genes_i)}
            aligned.append(counts_i[:, [lookup[gene] for gene in gene_names]])
        counts = sparse.vstack(aligned).tocsr()

    barcodes = np.concatenate(cell_lists).astype(str)
    gene_names = make_unique(gene_names.astype(str))
    return counts, gene_names, barcodes


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
    mito_mask = np.char.startswith(mito_mask, "MT-") | np.char.startswith(mito_mask, "MT.")
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
    return {
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
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "depths": np.array(config.DEPTHS, dtype=np.float32),
    }
    for depth, low in lows.items():
        payload[depth_to_name(depth)] = low
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
    ensure_dirs()
    download_pancreas()
    raw_counts, gene_names, barcodes = read_pancreas_counts()
    print(f"Raw cells: {raw_counts.shape[0]}, genes: {raw_counts.shape[1]}")

    filtered_counts, gene_names, barcodes = filter_cells_and_genes(raw_counts, gene_names, barcodes)
    train_idx, val_idx, test_idx = make_split(filtered_counts.shape[0])

    hvg_idx = select_highly_variable_genes(filtered_counts[train_idx], config.N_TOP_GENES)
    hvg_counts = filtered_counts[:, hvg_idx].tocsr()
    gene_names = gene_names[hvg_idx]
    high = normalize_log1p(hvg_counts)
    print(f"Cells: {high.shape[0]}, genes: {high.shape[1]}")

    rng = np.random.default_rng(config.SEED)
    gene_prob = gene_capture_probabilities(hvg_counts[train_idx], rng)
    train_library = np.asarray(hvg_counts[train_idx].sum(axis=1)).reshape(-1)
    target_base = float(np.median(train_library))

    summaries = [
        count_summary("full_filtered_counts", filtered_counts, 1.0),
        count_summary("hvg_high_depth", hvg_counts, 1.0),
    ]
    lows = {}
    raw_lows = {}
    for depth in config.DEPTHS:
        target_umi = max(1, int(round(target_base * depth)))
        low_counts = fixed_umi_counts(
            hvg_counts,
            target_umi=target_umi,
            seed=config.SEED + int(depth * 1000),
            gene_prob=gene_prob,
        )
        low = normalize_log1p(low_counts)
        lows[float(depth)] = low
        raw_lows[float(depth)] = low_counts
        row = count_summary(f"depth_{depth_to_name(depth)}", low_counts, depth)
        row["target_umi_per_cell"] = target_umi
        summaries.append(row)
        print(f"Prepared fixed-UMI low-depth matrix for {int(depth * 100)}% target_umi={target_umi}")

    save_benchmark_payload(config.PROCESSED_NPZ, high, gene_names, barcodes, train_idx, val_idx, test_idx, lows)
    save_raw_counts(hvg_counts, raw_lows)
    pd.DataFrame(summaries).round(4).to_csv(config.FIXED_UMI_DEPTH_SUMMARY_CSV, index=False)
    pd.DataFrame({"gene": gene_names}).to_csv(config.SELECTED_GENES_CSV, index=False)
    print(f"Saved processed data to {config.PROCESSED_NPZ}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    preprocess()
