"""Ablation study for DepthDiff design choices, on the same fixed-UMI benchmark.

Three axes:
  1. Prediction parameterization: x0 vs epsilon.
  2. Noise schedule: cosine vs linear.
  3. Inference: full reverse-diffusion sampling vs a single forward step
     (predict x0 once from pure noise conditioned on x_low -- i.e. no iterative
     refinement), using the same trained x0+cosine model.

All variants share identical architecture, data split, epochs, optimizer and
seed; metrics reuse evaluate.py's functions and identical KMeans labels.
Reported values are means over the three depths (25/50/75%).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

import config
from diffusion import make_diffusion
from evaluate import deg_overlap, logfc_metrics, predict_depthdiff
from models import make_depthdiff
from train_depthdiff import train_one_epoch, validate
from utils import (
    DepthPairDataset,
    EarlyStopping,
    get_device,
    load_processed,
    pearson_flat,
    rmse,
    set_seed,
)

VARIANTS = [("x0", "cosine"), ("epsilon", "cosine"), ("x0", "linear"), ("epsilon", "linear")]


def train_variant(data, device, n_genes):
    set_seed()  # identical init/order for every variant
    model = make_depthdiff(n_genes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    diffusion = make_diffusion(device)
    tl = DataLoader(DepthPairDataset(data.high, data.lows, data.train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    vl = DataLoader(DepthPairDataset(data.high, data.lows, data.val_idx), batch_size=config.BATCH_SIZE, shuffle=False)
    stopper = EarlyStopping(config.PATIENCE)
    best = None
    for _ in range(1, config.EPOCHS + 1):
        train_one_epoch(model, diffusion, tl, opt, device)
        v = validate(model, diffusion, vl, device)
        if stopper.step(v):
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if stopper.should_stop:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, diffusion


@torch.no_grad()
def single_step_predict(model, diffusion, x_low, depth, device, n_genes):
    """One forward step: predict x0 from pure noise conditioned on x_low (no chain)."""
    model.eval()
    preds = []
    for s in range(0, x_low.shape[0], config.BATCH_SIZE):
        low = torch.from_numpy(x_low[s : s + config.BATCH_SIZE]).to(device)
        depths = torch.full((low.size(0),), float(depth), device=device)
        x = torch.randn(low.size(0), n_genes, device=device)
        t = torch.full((low.size(0),), diffusion.num_steps - 1, device=device, dtype=torch.long)
        _, x0 = diffusion.model_eps_x0(model, x, low, t, depths)
        preds.append(diffusion.high_from_target(low, x0).clamp(min=0).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def score(target, pred, labels):
    _, logfc_spearman, _ = logfc_metrics(target, pred, labels)
    return {
        "rmse": rmse(pred, target),
        "pearson": pearson_flat(pred, target),
        "deg": deg_overlap(target, pred, labels),
        "logfc_s": logfc_spearman,
    }


def mean_over_depths(per_depth):
    keys = per_depth[0].keys()
    return {k: float(np.nanmean([d[k] for d in per_depth])) for k in keys}


def main():
    set_seed()
    device = get_device()
    data = load_processed()
    n_genes = data.high.shape[1]
    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)

    rows = []
    for pred_type, schedule in VARIANTS:
        config.PREDICTION_TYPE = pred_type
        config.BETA_SCHEDULE = schedule
        config.SAMPLER = "ddpm"
        config.EVAL_NUM_SAMPLES = 5
        model, diffusion = train_variant(data, device, n_genes)

        per_depth = [
            score(target, predict_depthdiff(model, diffusion, data.lows[d][data.test_idx], d, device), labels)
            for d in config.DEPTHS
        ]
        row = {"parameterization": pred_type, "schedule": schedule, "inference": "reverse(50, N=5)"}
        row.update(mean_over_depths(per_depth))
        rows.append(row)
        print(f"done {pred_type}+{schedule} reverse: {row}")

        if pred_type == "x0" and schedule == "cosine":
            per_depth = [
                score(target, single_step_predict(model, diffusion, data.lows[d][data.test_idx], d, device, n_genes), labels)
                for d in config.DEPTHS
            ]
            row = {"parameterization": "x0", "schedule": "cosine", "inference": "single-step"}
            row.update(mean_over_depths(per_depth))
            rows.append(row)
            print(f"done x0+cosine single-step: {row}")

    df = pd.DataFrame(rows).round(3)
    out = config.RESULTS_DIR / "ablation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
