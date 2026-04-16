from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ridrec.utils import load_jsonl


@dataclass
class SessionBatch:
    session_ids: torch.Tensor
    items: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device | str) -> "SessionBatch":
        return SessionBatch(
            session_ids=self.session_ids.to(device),
            items=self.items.to(device),
            lengths=self.lengths.to(device),
            targets=self.targets.to(device),
        )


class SessionDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.samples = load_jsonl(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        return {
            "session_id": int(row["session_id"]),
            "items": [int(x) for x in row["items"]],
            "target": int(row["target"]),
        }


def collate_sessions(batch: list[dict]) -> SessionBatch:
    lengths = torch.tensor([len(x["items"]) for x in batch], dtype=torch.long)
    max_len = lengths.max().item()
    items = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, row in enumerate(batch):
        seq = torch.tensor(row["items"], dtype=torch.long)
        items[i, : seq.size(0)] = seq
    session_ids = torch.tensor([x["session_id"] for x in batch], dtype=torch.long)
    targets = torch.tensor([x["target"] for x in batch], dtype=torch.long)
    return SessionBatch(session_ids=session_ids, items=items, lengths=lengths, targets=targets)
