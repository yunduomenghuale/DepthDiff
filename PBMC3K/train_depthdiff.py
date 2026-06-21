import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from diffusion import make_diffusion
from models import make_depthdiff
from utils import DepthPairDataset, EarlyStopping, ensure_dirs, get_device, load_processed, save_checkpoint, set_seed


def depth_weights(depths):
    weights = torch.ones_like(depths)
    for depth, weight in config.DEPTH_LOSS_WEIGHTS.items():
        weights = torch.where(
            torch.isclose(depths, torch.tensor(float(depth), device=depths.device)),
            torch.full_like(weights, float(weight)),
            weights,
        )
    return weights


def sample_correlation_loss(pred, target):
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered.pow(2).sum(dim=1) + 1e-8)
        * (target_centered.pow(2).sum(dim=1) + 1e-8)
    )
    corr = numerator / denominator
    return 1.0 - corr.mean()


def variance_preservation_loss(pred, target):
    gene_var = target.var(dim=0, unbiased=False)
    n_top = max(1, int(target.size(1) * config.TOP_VARIANCE_FRACTION))
    top_idx = torch.topk(gene_var, k=n_top).indices
    pred_std = pred[:, top_idx].std(dim=0, unbiased=False)
    target_std = target[:, top_idx].std(dim=0, unbiased=False)
    return F.mse_loss(pred_std, target_std)


def diffusion_loss(model, diffusion, x_low, x_high, depths, noise=None, timesteps=None):
    """Depth-weighted DDPM simple loss plus auxiliary x0 signal losses.

    The base objective is ``||eps - eps_theta||^2`` for epsilon-prediction or
    ``||x0 - x0_theta||^2`` for x0-prediction (``config.PREDICTION_TYPE``).
    The clean target is reconstructed and the biology-preserving terms
    (cell-wise correlation, high-variance gene preservation) are applied on the
    reconstructed high-depth profile.
    """
    x0 = diffusion.target_from_pair(x_low, x_high)
    if timesteps is None:
        timesteps = torch.randint(0, config.DIFFUSION_STEPS, (x0.size(0),), device=x0.device)
    if noise is None:
        noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, timesteps, noise)
    eps_pred, x0_pred = diffusion.model_eps_x0(model, x_t, x_low, timesteps, depths)

    if config.PREDICTION_TYPE == "x0":
        per_sample = F.mse_loss(x0_pred, x0, reduction="none").mean(dim=1)
    else:
        per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=1)
    weights = depth_weights(depths)
    loss = (per_sample * weights).sum() / weights.sum()

    if config.AUX_X0_LOSS:
        pred_high = diffusion.high_from_target(x_low, x0_pred)
        loss = (
            loss
            + config.CORRELATION_LOSS_WEIGHT * sample_correlation_loss(pred_high, x_high)
            + config.VARIANCE_LOSS_WEIGHT * variance_preservation_loss(pred_high, x_high)
        )
    return loss


def train_one_epoch(model, diffusion, loader, optimizer, device):
    model.train()
    total = 0.0
    for x_low, x_high, depths in tqdm(loader, desc="train", leave=False):
        x_low = x_low.to(device)
        x_high = x_high.to(device)
        depths = depths.to(device)
        loss = diffusion_loss(model, diffusion, x_low, x_high, depths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        total += loss.item() * x_high.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, diffusion, loader, device):
    """Validation loss with a fixed noise/timestep draw for comparable curves."""
    model.eval()
    gen = torch.Generator(device=device)
    total = 0.0
    for x_low, x_high, depths in tqdm(loader, desc="val", leave=False):
        x_low = x_low.to(device)
        x_high = x_high.to(device)
        depths = depths.to(device)
        gen.manual_seed(config.SEED)
        timesteps = torch.randint(
            0, config.DIFFUSION_STEPS, (x_high.size(0),), device=device, generator=gen
        )
        noise = torch.randn(x_high.shape, device=device, generator=gen)
        loss = diffusion_loss(model, diffusion, x_low, x_high, depths, noise, timesteps)
        total += loss.item() * x_high.size(0)
    return total / len(loader.dataset)


def main():
    set_seed()
    ensure_dirs()
    data = load_processed()
    device = get_device()
    n_genes = data.high.shape[1]
    train_ds = DepthPairDataset(data.high, data.lows, data.train_idx)
    val_ds = DepthPairDataset(data.high, data.lows, data.val_idx)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = make_depthdiff(n_genes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    diffusion = make_diffusion(device)
    stopper = EarlyStopping(config.PATIENCE)
    best_path = config.CHECKPOINT_DIR / f"{config.BENCHMARK_NAME}_{config.DEPTHDIFF_METHOD_NAME}_best.pt"

    print(f"Training unified depth-conditioned DepthDiff (diffusion) for depths={config.DEPTHS}")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer, device)
        val_loss = validate(model, diffusion, val_loader, device)
        print(f"epoch {epoch:03d}/{config.EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if stopper.step(val_loss):
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_loss,
                extra={
                    "n_genes": n_genes,
                    "method": config.DEPTHDIFF_METHOD_NAME,
                    "benchmark": config.BENCHMARK_NAME,
                    "depths": config.DEPTHS,
                    "backbone": "residual_mlp",
                    "prediction_type": config.PREDICTION_TYPE,
                    "diffusion_target": config.DIFFUSION_TARGET,
                    "diffusion_steps": config.DIFFUSION_STEPS,
                    "depth_conditioned": True,
                    "depth_loss_weights": config.DEPTH_LOSS_WEIGHTS,
                },
            )
            print(f"Saved best checkpoint to {best_path}")
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
