"""Wall-clock inference runtime benchmark on the PBMC68K test set (reviewer request).

Measures end-to-end enhancement time at 50% depth for:
  - DepthDiff single-step (default)
  - DepthDiff full reverse sampling (50-step DDPM)
  - cVAE, MLP (supervised baselines)
  - MAGIC (unsupervised, transductive on the full matrix)
GPU timings use cuda.synchronize(); neural nets are averaged over repeats.
"""
import time
import numpy as np
import torch

import config
from utils import load_processed, get_device, set_seed
from diffusion import make_diffusion
from evaluate import load_depthdiff, load_baselines, predict_depthdiff, predict_baseline

set_seed()
device = get_device()
use_cuda = torch.cuda.is_available()
data = load_processed()
n_genes = data.high.shape[1]
depth = 0.50
x_low_test = data.lows[depth][data.test_idx].astype(np.float32)
n_test = x_low_test.shape[0]
print(f"PBMC68K test cells @ {int(depth*100)}% depth: {n_test} | genes: {n_genes} | device: {device}")

dd = load_depthdiff(n_genes, device)
diffusion = make_diffusion(device)
baselines = load_baselines(n_genes, device)


def timeit(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


results = {}

config.SAMPLER = "single_step"; config.EVAL_NUM_SAMPLES = 1
results["DepthDiff (single-step)"] = timeit(
    lambda: predict_depthdiff(dd, diffusion, x_low_test, depth, device), repeats=5)

config.SAMPLER = "ddpm"
results["DepthDiff (full reverse, 50-step DDPM)"] = timeit(
    lambda: predict_depthdiff(dd, diffusion, x_low_test, depth, device), repeats=3)
config.SAMPLER = "single_step"

if "cvae" in baselines:
    results["cVAE"] = timeit(
        lambda: predict_baseline("cvae", baselines["cvae"], x_low_test, depth, device), repeats=5)
if "mlp" in baselines:
    results["MLP"] = timeit(
        lambda: predict_baseline("mlp", baselines["mlp"], x_low_test, depth, device), repeats=5)

# MAGIC is transductive: it must process the whole matrix to produce test predictions.
import magic  # noqa: E402
x_low_full = data.lows[depth].astype(np.float32)
t0 = time.perf_counter()
op = magic.MAGIC(random_state=config.SEED, verbose=0)
_ = op.fit_transform(x_low_full)
results["MAGIC (full matrix, CPU)"] = (time.perf_counter() - t0, 0.0)
print(f"MAGIC processed full matrix: {x_low_full.shape[0]} cells")

print("\n=== Inference runtime on PBMC68K test set (50% depth) ===")
for k, (m, s) in results.items():
    print(f"{k:42s} {m:9.3f} s  (+/- {s:.3f})")

ss = results["DepthDiff (single-step)"][0]
fr = results["DepthDiff (full reverse, 50-step DDPM)"][0]
print(f"\nSingle-step speedup vs full reverse: {fr / ss:.1f}x")
print(f"Single-step throughput: {n_test / ss:,.0f} cells/s")
