import torch
from torch import nn
import torch.nn.functional as F

import config
from utils import sinusoidal_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class DepthDiff(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        cond_dim = 64
        self.cond_dim = cond_dim
        self.input = nn.Sequential(
            nn.Linear(n_genes * 2 + cond_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, n_genes)

    def forward(self, x_t, x_low, timesteps, depths):
        t_emb = sinusoidal_embedding(timesteps, self.cond_dim)
        d_emb = sinusoidal_embedding(depths * 1000.0, self.cond_dim)
        h = torch.cat([x_t, x_low, t_emb, d_emb], dim=1)
        h = self.input(h)
        h = self.blocks(h)
        return self.output(h)


def make_depthdiff(n_genes: int) -> nn.Module:
    return DepthDiff(n_genes)


class DirectMLP(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x_low, depths):
        return self.net(torch.cat([x_low, depths.view(-1, 1)], dim=1))


class SupervisedAE(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = config.HIDDEN_DIM, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x_low, depths):
        z = self.encoder(torch.cat([x_low, depths.view(-1, 1)], dim=1))
        return self.decoder(torch.cat([z, depths.view(-1, 1)], dim=1))


class ConditionalVAE(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = config.HIDDEN_DIM, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(self, x_low, depths):
        h = self.encoder(torch.cat([x_low, depths.view(-1, 1)], dim=1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x_low, depths):
        mu, logvar = self.encode(x_low, depths)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(torch.cat([z, depths.view(-1, 1)], dim=1))
        return recon, mu, logvar


def vae_loss(recon, target, mu, logvar, beta: float = 1e-3):
    mse = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl


def make_baseline(name: str, n_genes: int) -> nn.Module:
    if name == "mlp":
        return DirectMLP(n_genes)
    if name == "sae":
        return SupervisedAE(n_genes)
    if name == "cvae":
        return ConditionalVAE(n_genes)
    raise ValueError(f"Unknown baseline: {name}")
