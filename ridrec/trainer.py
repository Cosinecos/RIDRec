from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ridrec.config import ExperimentConfig, load_metadata, save_resolved_config
from ridrec.data import SessionDataset, collate_sessions
from ridrec.models import RIDRec
from ridrec.utils import RankingMeter, write_json


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        metadata = load_metadata(config.train.metadata_path)
        if config.model.num_items <= 0:
            config.model.num_items = int(metadata["num_items"])
        self.model = RIDRec(config.model).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, self.output_dir / "resolved_config.yaml")
        self.metadata = metadata

    def _make_loader(self, path: str, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        dataset = SessionDataset(path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_sessions,
            pin_memory=torch.cuda.is_available(),
        )

    def train(self) -> dict[str, float]:
        train_loader = self._make_loader(
            self.config.train.train_path,
            self.config.train.batch_size,
            True,
            self.config.train.num_workers,
        )
        valid_loader = self._make_loader(
            self.config.train.valid_path,
            self.config.eval.batch_size,
            False,
            self.config.eval.num_workers,
        )
        best_metric = float("-inf")
        best_state = None
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.train.epochs + 1):
            self.model.train()
            running = {"loss": 0.0, "rank": 0.0, "kl": 0.0, "diff": 0.0, "retrieval": 0.0}
            progress = tqdm(train_loader, desc=f"train epoch {epoch}")
            for step, batch in enumerate(progress, start=1):
                batch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(batch)
                loss = out.rank_loss + self.config.train.lambda_kl * out.kl_loss + self.config.train.lambda_diff * out.diff_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                self.model.update_momentum_encoder()
                with torch.no_grad():
                    keys = self.model.encode_sessions_momentum(batch.items, batch.lengths)
                    self.model.retrieval_bank.enqueue(keys, batch.session_ids, batch.targets)
                running["loss"] += loss.item()
                running["rank"] += out.rank_loss.item()
                running["kl"] += out.kl_loss.item()
                running["diff"] += out.diff_loss.item()
                running["retrieval"] += out.retrieval_used_ratio
                if step % self.config.train.log_every == 0 or step == len(train_loader):
                    denom = step
                    progress.set_postfix(
                        loss=f"{running['loss'] / denom:.4f}",
                        rank=f"{running['rank'] / denom:.4f}",
                        kl=f"{running['kl'] / denom:.4f}",
                        diff=f"{running['diff'] / denom:.4f}",
                        retrieval=f"{running['retrieval'] / denom:.3f}",
                    )

            valid_metrics = self.evaluate_loader(valid_loader)
            record = {"epoch": epoch, **valid_metrics}
            history.append(record)
            metric_value = valid_metrics.get(self.config.checkpoint_metric, float("-inf"))
            if metric_value > best_metric:
                best_metric = metric_value
                best_state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": valid_metrics,
                }
                torch.save(best_state, self.output_dir / "best.pt")
            if not self.config.train.save_best_only:
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": valid_metrics,
                    },
                    self.output_dir / f"epoch_{epoch}.pt",
                )

        write_json(self.output_dir / "history.json", {"history": history})
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        return best_state["metrics"]

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = RankingMeter(self.config.eval.ks, self.config.model.num_items)
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = batch.to(self.device)
            out = self.model(batch)
            meter.update(out.logits, batch.targets)
        return meter.compute()

    @torch.no_grad()
    def evaluate_checkpoint(self, checkpoint_path: str | Path, test_path: str | None = None) -> dict[str, float]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        path = test_path or self.config.eval.test_path
        loader = self._make_loader(path, self.config.eval.batch_size, False, self.config.eval.num_workers)
        metrics = self.evaluate_loader(loader)
        write_json(self.output_dir / "test_metrics.json", metrics)
        return metrics
