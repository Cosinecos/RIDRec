# RIDRec

PyTorch implementation of RIDRec: Retrieval-Enhanced Intent Diffusion for Session-Based Painting Recommendation.

This repository implements a trainable, evaluable, and ablation-ready version of RIDRec. The overall framework consists of three core components:

- Cross-session retrieval and weighted fusion
- Neural Process-based latent intent modeling
- Conditional diffusion-based latent refinement

It also preserves the key training and evaluation characteristics of the paper:

- FIFO session memory bank
- Momentum encoder for retrieval keys
- Full-ranking evaluation over the entire item set
- HR@K / MRR@K / COV@K
- Support for w/o Retrieval / w/o NP / w/o Diffusion ablations

## 1. Project Features

- Training pipeline for next-item prediction in anonymous session-based painting recommendation
- Cross-session retrieval based on a FIFO memory bank
- Momentum session encoder and momentum item embedding
- Top-K retrieval, attention fusion, and fallback context
- NP prior / posterior latent modeling
- KL regularization
- Conditional diffusion denoising module
- Aggregation over multiple reverse trajectories
- Full-ranking metric computation
- Generic raw CSV preprocessing script
- Ready-to-run training and evaluation scripts
- Config-based ablations for retrieval / NP / diffusion

## 2. Repository Structure

```text
ridrec_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── ablations/
│       ├── wo_diffusion.yaml
│       ├── wo_np.yaml
│       └── wo_retrieval.yaml
├── scripts/
│   ├── prepare_sessions.py
│   ├── train.py
│   └── evaluate.py
├── ridrec/
│   ├── __init__.py
│   ├── config.py
│   ├── trainer.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── retrieval.py
│   │   ├── neural_process.py
│   │   ├── diffusion.py
│   │   └── ridrec.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── metrics.py
│       └── random.py
└── tests/
    └── test_forward.py