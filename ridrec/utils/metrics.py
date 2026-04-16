from __future__ import annotations

from collections import defaultdict

import torch


class RankingMeter:
    def __init__(self, ks: list[int], num_items: int) -> None:
        self.ks = sorted(ks)
        self.num_items = num_items
        self.total = 0
        self.hr_sum = defaultdict(float)
        self.mrr_sum = defaultdict(float)
        self.coverage_sets = {k: set() for k in self.ks}

    @torch.no_grad()
    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        max_k = max(self.ks)
        topk = scores.topk(max_k, dim=1).indices
        self.total += targets.size(0)
        for k in self.ks:
            pred_k = topk[:, :k]
            hits = pred_k.eq(targets.unsqueeze(1))
            self.hr_sum[k] += hits.any(dim=1).float().sum().item()
            rank_positions = hits.float().argmax(dim=1) + 1
            reciprocal = torch.where(hits.any(dim=1), 1.0 / rank_positions.float(), torch.zeros_like(rank_positions, dtype=torch.float))
            self.mrr_sum[k] += reciprocal.sum().item()
            self.coverage_sets[k].update(pred_k.reshape(-1).tolist())

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for k in self.ks:
            results[f"hr@{k}"] = 100.0 * self.hr_sum[k] / max(self.total, 1)
            results[f"mrr@{k}"] = 100.0 * self.mrr_sum[k] / max(self.total, 1)
            covered = {x for x in self.coverage_sets[k] if x != 0}
            results[f"cov@{k}"] = 100.0 * len(covered) / max(self.num_items, 1)
        return results
