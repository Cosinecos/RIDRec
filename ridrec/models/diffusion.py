from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DiffusionTrainOutput:
    loss: torch.Tensor
    refined_latent: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EpsilonDenoiser(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(timesteps)
        x = torch.cat([z_t, condition, t], dim=-1)
        return self.net(x)


class ConditionalDiffusionRefiner(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        time_dim: int,
        hidden_dim: int,
        steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        beta_tilde = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("beta_tilde", beta_tilde)
        self.denoiser = EpsilonDenoiser(latent_dim, condition_dim, time_dim, hidden_dim)

    def training_step(self, z0: torch.Tensor, condition: torch.Tensor) -> DiffusionTrainOutput:
        batch_size = z0.size(0)
        t = torch.randint(0, self.steps, (batch_size,), device=z0.device)
        eps = torch.randn_like(z0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)
        z_t = alpha_bar_t.sqrt() * z0 + (1.0 - alpha_bar_t).sqrt() * eps
        eps_pred = self.denoiser(z_t, t, condition)
        loss = F.mse_loss(eps_pred, eps)
        refined = (z_t - (1.0 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt().clamp_min(1e-8)
        return DiffusionTrainOutput(loss=loss, refined_latent=refined)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_trajectories: int) -> torch.Tensor:
        batch_size, condition_dim = condition.shape
        device = condition.device
        cond = condition.unsqueeze(1).expand(batch_size, num_trajectories, condition_dim).reshape(batch_size * num_trajectories, condition_dim)
        z = torch.randn(batch_size * num_trajectories, self.denoiser.net[-1].out_features, device=device)
        for step in reversed(range(self.steps)):
            t = torch.full((z.size(0),), step, dtype=torch.long, device=device)
            eps_pred = self.denoiser(z, t, cond)
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            coeff = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = (z - coeff * eps_pred) / torch.sqrt(alpha_t)
            if step > 0:
                noise = torch.randn_like(z)
                z = mean + torch.sqrt(self.beta_tilde[step]) * noise
            else:
                z = mean
        return z.view(batch_size, num_trajectories, -1)
