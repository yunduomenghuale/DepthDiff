import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config


def ensure_dirs() -> None:
    for path in [config.RAW_DIR, config.PROCESSED_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def depth_to_name(depth: float) -> str:
    return f"depth_{int(round(depth * 100))}"


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.float().view(-1, 1)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=values.device).float()
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    angles = values * freqs.view(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


@dataclass
class PBMCData:
    high: np.ndarray
    lows: dict
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    gene_names: np.ndarray


def load_processed(path: Path = config.PROCESSED_NPZ) -> PBMCData:
    data = np.load(path, allow_pickle=True)
    lows = {float(d): data[depth_to_name(float(d))].astype(np.float32) for d in config.DEPTHS}
    return PBMCData(
        high=data["high"].astype(np.float32),
        lows=lows,
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
        gene_names=data["gene_names"],
    )


class DepthPairDataset(Dataset):
    def __init__(self, high: np.ndarray, lows: dict, indices: np.ndarray, depths=None):
        self.high = high
        self.lows = lows
        self.indices = np.asarray(indices)
        self.depths = list(config.DEPTHS if depths is None else depths)

    def __len__(self) -> int:
        return len(self.indices) * len(self.depths)

    def __getitem__(self, item: int):
        cell_pos = item // len(self.depths)
        depth_pos = item % len(self.depths)
        idx = self.indices[cell_pos]
        depth = float(self.depths[depth_pos])
        return (
            torch.from_numpy(self.lows[depth][idx]),
            torch.from_numpy(self.high[idx]),
            torch.tensor(depth, dtype=torch.float32),
        )


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.count = 0

    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.count >= self.patience


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, epoch: int, val_loss: float, extra=None) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "config": {
            "seed": config.SEED,
            "benchmark": config.BENCHMARK_NAME,
            "processed_npz": str(config.PROCESSED_NPZ),
            "depths": config.DEPTHS,
            "epochs": config.EPOCHS,
            "patience": config.PATIENCE,
            "optimizer": "AdamW",
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
