from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ridrec.config import ModelConfig
from ridrec.data import SessionBatch
from ridrec.models.diffusion import ConditionalDiffusionRefiner
from ridrec.models.encoder import GRUSessionEncoder
from ridrec.models.neural_process import NPLatentModule
from ridrec.models.retrieval import RetrievalMemoryBank


@dataclass
class RIDRecOutput:
    logits: torch.Tensor
    rank_loss: torch.Tensor | None
    kl_loss: torch.Tensor
    diff_loss: torch.Tensor
    retrieval_used_ratio: float


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RIDRec(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=0)
        self.momentum_item_embedding = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=0)
        self.session_encoder = GRUSessionEncoder(config.embedding_dim, config.encoder_hidden_dim, config.encoder_dropout)
        self.momentum_session_encoder = GRUSessionEncoder(config.embedding_dim, config.encoder_hidden_dim, config.encoder_dropout)
        self.retrieval_bank = RetrievalMemoryBank(
            embedding_dim=config.embedding_dim,
            context_dim=config.context_dim,
            memory_size=config.memory_size,
            topk=config.topk_retrieval,
            temperature=config.similarity_temperature,
        )
        self.np_module = NPLatentModule(
            embedding_dim=config.embedding_dim,
            context_dim=config.context_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.projector_hidden_dim,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
        )
        self.condition_builder = MLP(
            config.embedding_dim + config.embedding_dim + config.context_dim + config.latent_dim + config.latent_dim,
            config.projector_hidden_dim,
            config.condition_dim,
        )
        self.diffusion = ConditionalDiffusionRefiner(
            latent_dim=config.latent_dim,
            condition_dim=config.condition_dim,
            time_dim=config.time_dim,
            hidden_dim=config.denoiser_hidden_dim,
            steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self.interest_projector = MLP(
            config.embedding_dim + config.embedding_dim + config.latent_dim,
            config.predictor_hidden_dim,
            config.embedding_dim,
        )
        self.trajectory_attention = nn.Linear(config.embedding_dim, config.aggregator_hidden_dim)
        self.trajectory_score = nn.Linear(config.aggregator_hidden_dim, 1)
        self.register_buffer("train_steps", torch.zeros(1, dtype=torch.long))
        self._initialize_momentum_modules()

    def _initialize_momentum_modules(self) -> None:
        self.momentum_item_embedding.load_state_dict(self.item_embedding.state_dict())
        self.momentum_session_encoder.load_state_dict(self.session_encoder.state_dict())
        for p in self.momentum_item_embedding.parameters():
            p.requires_grad = False
        for p in self.momentum_session_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        m = self.config.momentum
        for p, p_m in zip(self.item_embedding.parameters(), self.momentum_item_embedding.parameters()):
            p_m.data.mul_(m).add_(p.data, alpha=1.0 - m)
        for p, p_m in zip(self.session_encoder.parameters(), self.momentum_session_encoder.parameters()):
            p_m.data.mul_(m).add_(p.data, alpha=1.0 - m)

    def encode_sessions(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(items)
        return self.session_encoder(emb, lengths)

    @torch.no_grad()
    def encode_sessions_momentum(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.momentum_item_embedding(items)
        return self.momentum_session_encoder(emb, lengths)

    def retrieval_enabled(self) -> bool:
        return self.config.use_retrieval and self.retrieval_bank.size >= self.config.warmup_min_size

    def _build_condition(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        r: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        cond_input = torch.cat([h, c, r, prior_mu, prior_logvar], dim=-1)
        return self.condition_builder(cond_input)

    def _aggregate_trajectories(self, h: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return self.interest_projector(torch.cat([h, c, z], dim=-1))
        batch_size, num_trajectories, latent_dim = z.shape
        h_rep = h.unsqueeze(1).expand(batch_size, num_trajectories, h.size(-1))
        c_rep = c.unsqueeze(1).expand(batch_size, num_trajectories, c.size(-1))
        interest = self.interest_projector(torch.cat([h_rep, c_rep, z], dim=-1))
        attn_hidden = torch.tanh(self.trajectory_attention(interest))
        scores = self.trajectory_score(attn_hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("br,brd->bd", weights, interest)

    def _score_items(self, user_repr: torch.Tensor) -> torch.Tensor:
        logits = user_repr @ self.item_embedding.weight.transpose(0, 1)
        logits[:, 0] = -1e9
        return logits

    def forward(self, batch: SessionBatch) -> RIDRecOutput:
        h = self.encode_sessions(batch.items, batch.lengths)
        retrieval = self.retrieval_bank(
            current_repr=h,
            session_ids=batch.session_ids,
            item_embedding=self.item_embedding,
            enabled=self.retrieval_enabled(),
        )
        c = retrieval.context
        r = retrieval.retrieved_summary
        target_emb = self.item_embedding(batch.targets)
        np_out = self.np_module(
            h=h,
            c=c,
            r=r,
            target_emb=target_emb,
            use_np=self.config.use_np,
            training=self.training,
        )
        condition = self._build_condition(h, c, r, np_out.prior_mu, np_out.prior_logvar)

        if self.training:
            if self.config.use_diffusion:
                diff_out = self.diffusion.training_step(np_out.z0, condition)
                z_rank = diff_out.refined_latent
                diff_loss = diff_out.loss
            else:
                z_rank = np_out.z0
                diff_loss = torch.zeros((), device=h.device)
            user_repr = self._aggregate_trajectories(h, c, z_rank)
            logits = self._score_items(user_repr)
            rank_loss = F.cross_entropy(logits, batch.targets)
            self.train_steps += 1
            return RIDRecOutput(
                logits=logits,
                rank_loss=rank_loss,
                kl_loss=np_out.kl,
                diff_loss=diff_loss,
                retrieval_used_ratio=retrieval.used_retrieval.float().mean().item(),
            )

        if self.config.use_diffusion:
            z_samples = self.diffusion.sample(condition, self.config.reverse_trajectories)
            user_repr = self._aggregate_trajectories(h, c, z_samples)
        else:
            user_repr = self._aggregate_trajectories(h, c, np_out.prior_mu)
        logits = self._score_items(user_repr)
        return RIDRecOutput(
            logits=logits,
            rank_loss=None,
            kl_loss=torch.zeros((), device=h.device),
            diff_loss=torch.zeros((), device=h.device),
            retrieval_used_ratio=retrieval.used_retrieval.float().mean().item(),
        )
