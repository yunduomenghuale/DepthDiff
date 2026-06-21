import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models import make_baseline, vae_loss
from utils import (
    DepthPairDataset,
    EarlyStopping,
    ensure_dirs,
    get_device,
    load_processed,
    save_checkpoint,
    set_seed,
)


BASELINES = ["mlp", "sae", "cvae"]


def model_loss(name, output, target):
    if name == "cvae":
        recon, mu, logvar = output
        return vae_loss(recon, target, mu, logvar)
    return F.mse_loss(output, target)


def train_one_epoch(name, model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x_low, x_high, depths in tqdm(loader, desc=f"{name} train", leave=False):
        x_low = x_low.to(device)
        x_high = x_high.to(device)
        depths = depths.to(device)
        output = model(x_low, depths)
        loss = model_loss(name, output, x_high)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        total += loss.item() * x_high.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(name, model, loader, device):
    model.eval()
    total = 0.0
    for x_low, x_high, depths in tqdm(loader, desc=f"{name} val", leave=False):
        x_low = x_low.to(device)
        x_high = x_high.to(device)
        depths = depths.to(device)
        output = model(x_low, depths)
        loss = model_loss(name, output, x_high)
        total += loss.item() * x_high.size(0)
    return total / len(loader.dataset)


def train_baseline(name, data, device):
    n_genes = data.high.shape[1]
    train_ds = DepthPairDataset(data.high, data.lows, data.train_idx)
    val_ds = DepthPairDataset(data.high, data.lows, data.val_idx)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = make_baseline(name, n_genes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    stopper = EarlyStopping(config.PATIENCE)
    best_path = config.CHECKPOINT_DIR / f"{config.BENCHMARK_NAME}_baseline_{name}_best.pt"

    print(f"Training unified baseline: {name} depths={config.DEPTHS}")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(name, model, train_loader, optimizer, device)
        val_loss = validate(name, model, val_loader, device)
        print(f"{name} epoch {epoch:03d}/{config.EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if stopper.step(val_loss):
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_loss,
                extra={
                    "n_genes": n_genes,
                    "baseline": name,
                    "benchmark": config.BENCHMARK_NAME,
                    "depths": config.DEPTHS,
                    "depth_conditioned": True,
                },
            )
            print(f"Saved best checkpoint to {best_path}")
        if stopper.should_stop:
            print(f"{name} early stopping at epoch {epoch}")
            break


def main():
    set_seed()
    ensure_dirs()
    data = load_processed()
    device = get_device()
    for name in BASELINES:
        set_seed(config.SEED)
        train_baseline(name, data, device)


if __name__ == "__main__":
    main()
