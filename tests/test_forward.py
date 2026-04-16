from __future__ import annotations

import torch

from ridrec.config import ModelConfig
from ridrec.data import SessionBatch
from ridrec.models import RIDRec



def test_forward_shapes() -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = ModelConfig(num_items=50, warmup_min_size=0, diffusion_steps=5, reverse_trajectories=2)
    model = RIDRec(config)
    batch = SessionBatch(
        session_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        items=torch.tensor([[1, 2, 3], [4, 5, 0], [6, 7, 8]], dtype=torch.long),
        lengths=torch.tensor([3, 2, 3], dtype=torch.long),
        targets=torch.tensor([4, 6, 9], dtype=torch.long),
    )
    model.train()
    out = model(batch)
    assert out.logits.shape == (3, 51)
    assert out.rank_loss is not None
    model.update_momentum_encoder()
    with torch.no_grad():
        keys = model.encode_sessions_momentum(batch.items, batch.lengths)
        model.retrieval_bank.enqueue(keys, batch.session_ids, batch.targets)
    model.eval()
    eval_out = model(batch)
    assert eval_out.logits.shape == (3, 51)
