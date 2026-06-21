# Runs official ALRA on exported low-depth count matrices.
# Usage: Rscript alra_runner.R <alra_official.R> <results_dir> <pct1,pct2,...>
# Reads  <results_dir>/_alra_in_<pct>.mtx  (cells x genes raw counts)
# Writes <results_dir>/_alra_out_<pct>.mtx (cells x genes imputed, log1p(CP10k) space)
suppressMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
resdir <- args[2]
pcts <- strsplit(args[3], ",")[[1]]
set.seed(2026)
for (pct in pcts) {
  A <- as.matrix(readMM(file.path(resdir, sprintf("_alra_in_%s.mtx", pct))))
  A_norm <- normalize_data(A)            # log(CP10k + 1), same space as target
  res <- alra(A_norm, k = 0)             # k=0 -> choose_k automatically
  out <- res[[3]]                        # A_norm_rank_k_cor_sc (zero-preserving)
  writeMM(Matrix(out, sparse = TRUE),
          file.path(resdir, sprintf("_alra_out_%s.mtx", pct)))
  cat(sprintf("pct %s done (%d x %d)\n", pct, nrow(out), ncol(out)))
}
