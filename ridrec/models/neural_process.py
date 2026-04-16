from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NPOutput:
    prior_mu: torch.Tensor
    prior_logvar: torch.Tensor
    posterior_mu: torch.Tensor | None
    posterior_logvar: torch.Tensor | None
    z0: torch.Tensor
    kl: torch.Tensor


class NPLatentModule(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        context_input_dim = embedding_dim + embedding_dim + context_dim
        target_input_dim = context_input_dim + embedding_dim
        self.prior_net = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(target_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.fallback_latent = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        r: torch.Tensor,
        target_emb: torch.Tensor | None,
        use_np: bool,
        training: bool,
    ) -> NPOutput:
        context_input = torch.cat([h, c, r], dim=-1)
        if not use_np:
            z0 = self.fallback_latent(context_input)
            zeros = torch.zeros_like(z0)
            return NPOutput(
                prior_mu=z0,
                prior_logvar=zeros,
                posterior_mu=None,
                posterior_logvar=None,
                z0=z0,
                kl=torch.zeros((), device=z0.device),
            )

        prior_mu, prior_logvar = self._split_stats(self.prior_net(context_input))
        if training and target_emb is not None:
            posterior_input = torch.cat([context_input, target_emb], dim=-1)
            posterior_mu, posterior_logvar = self._split_stats(self.posterior_net(posterior_input))
            std = torch.exp(0.5 * posterior_logvar)
            eps = torch.randn_like(std)
            z0 = posterior_mu + eps * std
            kl_per_dim = 0.5 * (
                prior_logvar
                - posterior_logvar
                + (torch.exp(posterior_logvar) + (posterior_mu - prior_mu).pow(2)) / torch.exp(prior_logvar)
                - 1.0
            )
            kl = kl_per_dim.sum(dim=-1).mean()
            return NPOutput(
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                posterior_mu=posterior_mu,
                posterior_logvar=posterior_logvar,
                z0=z0,
                kl=kl,
            )

        return NPOutput(
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=None,
            posterior_logvar=None,
            z0=prior_mu,
            kl=torch.zeros((), device=prior_mu.device),
        )
