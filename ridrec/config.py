from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    num_items: int = 0
    embedding_dim: int = 100
    latent_dim: int = 100
    context_dim: int = 100
    encoder_hidden_dim: int = 100
    projector_hidden_dim: int = 200
    predictor_hidden_dim: int = 200
    aggregator_hidden_dim: int = 128
    denoiser_hidden_dim: int = 256
    condition_dim: int = 128
    time_dim: int = 128
    encoder_dropout: float = 0.1
    similarity_temperature: float = 0.2
    topk_retrieval: int = 10
    memory_size: int = 50000
    momentum: float = 0.995
    warmup_min_size: int = 1000
    diffusion_steps: int = 50
    reverse_trajectories: int = 4
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    use_retrieval: bool = True
    use_np: bool = True
    use_diffusion: bool = True
    context_ratio: float = 0.1


@dataclass
class TrainConfig:
    train_path: str = "data/processed/train.jsonl"
    valid_path: str = "data/processed/valid.jsonl"
    metadata_path: str = "data/processed/metadata.json"
    batch_size: int = 100
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    lambda_kl: float = 0.1
    lambda_diff: float = 1.0
    log_every: int = 50
    save_best_only: bool = True


@dataclass
class EvalConfig:
    test_path: str = "data/processed/test.jsonl"
    batch_size: int = 256
    num_workers: int = 0
    ks: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class ExperimentConfig:
    experiment_name: str = "ridrec_default"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs/ridrec_default"
    checkpoint_metric: str = "mrr@20"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_config(data: dict[str, Any]) -> ExperimentConfig:
    model = ModelConfig(**data.get("model", {}))
    train = TrainConfig(**data.get("train", {}))
    eval_cfg = EvalConfig(**data.get("eval", {}))
    top = {k: v for k, v in data.items() if k not in {"model", "train", "eval"}}
    return ExperimentConfig(model=model, train=train, eval=eval_cfg, **top)


def load_config(*paths: str | Path) -> ExperimentConfig:
    merged: dict[str, Any] = asdict(ExperimentConfig())
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            patch = yaml.safe_load(f) or {}
        merged = _deep_update(merged, patch)
    return _to_config(merged)


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=True)


def load_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
