from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RetrievalOutput:
    context: torch.Tensor
    retrieved_summary: torch.Tensor
    used_retrieval: torch.Tensor


class RetrievalMemoryBank(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        memory_size: int,
        topk: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.topk = topk
        self.temperature = temperature
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.summary_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.fallback_context = nn.Parameter(torch.zeros(embedding_dim))
        self.fallback_summary = nn.Parameter(torch.zeros(context_dim))
        self.register_buffer("feature_queue", torch.zeros(memory_size, embedding_dim))
        self.register_buffer("session_queue", torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer("target_queue", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_size", torch.zeros(1, dtype=torch.long))

    @property
    def size(self) -> int:
        return int(self.queue_size.item())

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, session_ids: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = features.size(0)
        ptr = int(self.queue_ptr.item())
        if batch_size >= self.memory_size:
            features = features[-self.memory_size :]
            session_ids = session_ids[-self.memory_size :]
            targets = targets[-self.memory_size :]
            batch_size = self.memory_size
        end = ptr + batch_size
        if end <= self.memory_size:
            self.feature_queue[ptr:end] = features.detach()
            self.session_queue[ptr:end] = session_ids.detach()
            self.target_queue[ptr:end] = targets.detach()
        else:
            first = self.memory_size - ptr
            second = end - self.memory_size
            self.feature_queue[ptr:] = features[:first].detach()
            self.feature_queue[:second] = features[first:].detach()
            self.session_queue[ptr:] = session_ids[:first].detach()
            self.session_queue[:second] = session_ids[first:].detach()
            self.target_queue[ptr:] = targets[:first].detach()
            self.target_queue[:second] = targets[first:].detach()
        self.queue_ptr[0] = end % self.memory_size
        self.queue_size[0] = min(self.memory_size, self.size + batch_size)

    def forward(
        self,
        current_repr: torch.Tensor,
        session_ids: torch.Tensor,
        item_embedding: nn.Embedding,
        enabled: bool,
    ) -> RetrievalOutput:
        batch_size = current_repr.size(0)
        if not enabled or self.size == 0:
            context = self.fallback_context.unsqueeze(0).expand(batch_size, -1)
            summary = self.fallback_summary.unsqueeze(0).expand(batch_size, -1)
            used = torch.zeros(batch_size, dtype=torch.bool, device=current_repr.device)
            return RetrievalOutput(context=context, retrieved_summary=summary, used_retrieval=used)

        bank_features = self.feature_queue[: self.size]
        bank_sessions = self.session_queue[: self.size]
        bank_targets = self.target_queue[: self.size]

        q = F.normalize(self.query_proj(current_repr), dim=-1)
        k = F.normalize(self.key_proj(bank_features), dim=-1)
        sims = q @ k.transpose(0, 1)
        valid_mask = bank_sessions.unsqueeze(0).ne(session_ids.unsqueeze(1))
        sims = sims.masked_fill(~valid_mask, float("-inf"))

        topk = min(self.topk, self.size)
        top_values, top_indices = sims.topk(topk, dim=1)
        selected_mask = torch.isfinite(top_values)
        used_retrieval = selected_mask.any(dim=1)

        safe_logits = top_values / self.temperature
        safe_logits = safe_logits.masked_fill(~selected_mask, -1e9)
        attn = torch.softmax(safe_logits, dim=1)
        attn = attn * selected_mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)

        neighbor_features = bank_features[top_indices]
        neighbor_targets = bank_targets[top_indices]
        context = torch.einsum("bk,bkd->bd", attn, neighbor_features)

        target_emb = item_embedding(neighbor_targets.clamp_min(0))
        summary_input = torch.cat([neighbor_features, target_emb], dim=-1)
        summary_vectors = self.summary_proj(summary_input)
        summary = torch.einsum("bk,bkd->bd", attn, summary_vectors)

        fallback_context = self.fallback_context.unsqueeze(0).expand(batch_size, -1)
        fallback_summary = self.fallback_summary.unsqueeze(0).expand(batch_size, -1)
        context = torch.where(used_retrieval.unsqueeze(1), context, fallback_context)
        summary = torch.where(used_retrieval.unsqueeze(1), summary, fallback_summary)
        return RetrievalOutput(context=context, retrieved_summary=summary, used_retrieval=used_retrieval)
