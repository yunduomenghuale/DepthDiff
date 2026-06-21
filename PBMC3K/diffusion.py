"""Conditional Gaussian diffusion for depth-conditioned scRNA-seq enhancement.

DepthDiff is formulated as an SR3-style conditional denoising diffusion model.
The low-depth profile ``x_low`` plays the role of the conditioning "low-resolution"
observation, and the model learns the reverse process that turns Gaussian noise
into the (residual towards the) high-depth profile, conditioned on ``x_low`` and
the sequencing-depth ratio ``d``.

Two parameterisations of the diffusion target are supported (see
``config.DIFFUSION_TARGET``):

- ``"residual"`` (default): diffuse ``r = x_high - x_low`` and reconstruct
  ``x_high = x_low + r``. This keeps the residual-correction prior of the
  original design while remaining a proper diffusion process.
- ``"x_high"``: diffuse the high-depth profile directly.

The network always predicts the noise ``epsilon`` (``config.PREDICTION_TYPE``).
"""

import torch

import config


class GaussianDiffusion:
    def __init__(self, num_steps, beta_start, beta_end, device):
        self.num_steps = num_steps
        self.device = device
        schedule = getattr(config, "BETA_SCHEDULE", "cosine")
        if schedule == "cosine":
            # Nichol & Dhariwal cosine schedule: ensures alpha_bar[T-1] ~ 0 even for
            # small num_steps, so sampling from N(0, I) matches the forward endpoint.
            s = 0.008
            steps = torch.arange(num_steps + 1, device=device, dtype=torch.float64)
            f = torch.cos(((steps / num_steps) + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
            alpha_bars = (f / f[0])[1:]
            alpha_bars_prev = (f / f[0])[:-1]
            betas = (1.0 - alpha_bars / alpha_bars_prev).clamp(max=0.999)
            alphas = 1.0 - betas
            self.betas = betas.float()
            self.alphas = alphas.float()
            self.alpha_bars = alpha_bars.float()
            self.alpha_bars_prev = alpha_bars_prev.float()
        else:
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            self.betas = betas
            self.alphas = alphas
            self.alpha_bars = alpha_bars
            self.alpha_bars_prev = torch.cat(
                [torch.ones(1, device=device), alpha_bars[:-1]]
            )

    def q_sample(self, x0, t, noise):
        """Forward process: add noise to the diffusion target x0 at timestep t."""
        a_bar = self.alpha_bars[t].view(-1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """Recover the clean diffusion target from x_t and a predicted noise."""
        a_bar = self.alpha_bars[t].view(-1, 1)
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)

    def predict_eps_from_x0(self, x_t, t, x0):
        """Recover the noise implied by a predicted clean target."""
        a_bar = self.alpha_bars[t].view(-1, 1)
        return (x_t - torch.sqrt(a_bar) * x0) / torch.sqrt(1.0 - a_bar)

    def model_eps_x0(self, model, x_t, x_low, t, depths):
        """Return (eps, x0) from the network under either parameterisation.

        ``config.PREDICTION_TYPE`` selects whether the network output is the
        noise ("epsilon") or the clean diffusion target ("x0").
        """
        out = model(x_t, x_low, t.float(), depths)
        if config.PREDICTION_TYPE == "x0":
            return self.predict_eps_from_x0(x_t, t, out), out
        return out, self.predict_x0_from_eps(x_t, t, out)

    @staticmethod
    def target_from_pair(x_low, x_high):
        """Diffusion target x0 for a (low, high) pair under the chosen mode."""
        if config.DIFFUSION_TARGET == "residual":
            return x_high - x_low
        return x_high

    @staticmethod
    def high_from_target(x_low, x0):
        """Map a (denoised) diffusion target back to a high-depth profile."""
        if config.DIFFUSION_TARGET == "residual":
            return x_low + x0
        return x0

    @torch.no_grad()
    def ddpm_sample(self, model, x_low, depths, n_genes):
        """Ancestral DDPM reverse sampling conditioned on x_low and depth."""
        batch = x_low.size(0)
        x = torch.randn(batch, n_genes, device=self.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((batch,), step, device=self.device, dtype=torch.long)
            _, x0 = self.model_eps_x0(model, x, x_low, t, depths)
            beta = self.betas[t].view(-1, 1)
            alpha = self.alphas[t].view(-1, 1)
            a_bar = self.alpha_bars[t].view(-1, 1)
            a_bar_prev = self.alpha_bars_prev[t].view(-1, 1)
            coef_x0 = torch.sqrt(a_bar_prev) * beta / (1.0 - a_bar)
            coef_xt = torch.sqrt(alpha) * (1.0 - a_bar_prev) / (1.0 - a_bar)
            mean = coef_x0 * x0 + coef_xt * x
            if step > 0:
                var = beta * (1.0 - a_bar_prev) / (1.0 - a_bar)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return self.high_from_target(x_low, x)

    @torch.no_grad()
    def ddim_sample(self, model, x_low, depths, n_genes, steps, eta=0.0):
        """Deterministic (eta=0) / stochastic DDIM reverse sampling."""
        batch = x_low.size(0)
        x = torch.randn(batch, n_genes, device=self.device)
        timeline = torch.linspace(
            self.num_steps - 1, 0, steps, device=self.device
        ).round().long()
        for i, step in enumerate(timeline):
            t = torch.full((batch,), int(step), device=self.device, dtype=torch.long)
            eps, x0 = self.model_eps_x0(model, x, x_low, t, depths)
            a_bar = self.alpha_bars[t].view(-1, 1)
            if i < len(timeline) - 1:
                a_bar_prev = self.alpha_bars[timeline[i + 1]].view(-1, 1)
            else:
                a_bar_prev = torch.ones_like(a_bar)
            sigma = eta * torch.sqrt(
                (1.0 - a_bar_prev) / (1.0 - a_bar) * (1.0 - a_bar / a_bar_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma ** 2, min=0.0)) * eps
            x = torch.sqrt(a_bar_prev) * x0 + dir_xt
            if eta > 0 and i < len(timeline) - 1:
                x = x + sigma * torch.randn_like(x)
        return self.high_from_target(x_low, x)

    @torch.no_grad()
    def single_step_sample(self, model, x_low, depths, n_genes):
        """Single forward step: predict the clean target once from pure noise,
        conditioned on x_low. No iterative reverse chain.

        Ablations (see paper) show this matches or beats full reverse sampling on
        all reconstruction / biological-signal / variance metrics at a fraction
        of the cost, so it is the default inference for DepthDiff.
        """
        batch = x_low.size(0)
        x = torch.randn(batch, n_genes, device=self.device)
        t = torch.full((batch,), self.num_steps - 1, device=self.device, dtype=torch.long)
        _, x0 = self.model_eps_x0(model, x, x_low, t, depths)
        return self.high_from_target(x_low, x0)

    @torch.no_grad()
    def sample(self, model, x_low, depths, n_genes):
        """Dispatch to the configured sampler."""
        if config.SAMPLER == "single_step":
            return self.single_step_sample(model, x_low, depths, n_genes)
        if config.SAMPLER == "ddim":
            return self.ddim_sample(
                model, x_low, depths, n_genes, config.SAMPLING_STEPS, config.DDIM_ETA
            )
        return self.ddpm_sample(model, x_low, depths, n_genes)


def make_diffusion(device) -> GaussianDiffusion:
    return GaussianDiffusion(
        config.DIFFUSION_STEPS, config.BETA_START, config.BETA_END, device
    )
