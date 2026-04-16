from __future__ import annotations

import torch
from torch import nn


class GRUSessionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, embedded_items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded_items = self.dropout(embedded_items)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_items,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.out(hidden[-1])
