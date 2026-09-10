# RIDRec: Retrieval-Enhanced Intent Diffusion for Anonymous Short-Session Recommendation

RIDRec is a session-based recommendation framework designed for anonymous short-session scenarios. It combines cross-session retrieval, latent intent modeling, and diffusion-based intent refinement to improve next-item recommendation under limited behavioral observations.

## Background

Session-based recommendation predicts the next item based only on interactions observed in the current session.

This setting is particularly challenging for short and anonymous sessions. A short session contains limited behavioral evidence, while the absence of long-term user profiles makes it difficult to identify the user’s underlying intent. Moreover, similar interaction sequences may correspond to different purposes, creating uncertainty in session representation.

RIDRec addresses these challenges by retrieving relevant information from other sessions and modeling session intent as a latent distribution rather than a single deterministic representation.

## Method Overview

RIDRec mainly consists of the following components.

### 1. Session Representation

The interactions in the current session are encoded into a session representation that summarizes the user’s recent behavioral context.

This representation provides the basic information used for cross-session retrieval and subsequent intent modeling.

### 2. Cross-Session Retrieval

RIDRec retrieves sessions with behavioral patterns related to the current session from a session memory.

The retrieved sessions provide complementary contextual evidence that may not be available in a short target session. Their representations are aggregated according to their relevance to form a retrieval-enhanced context.

This mechanism allows RIDRec to use collaborative information from other sessions while preserving the characteristics of the current session.

### 3. Latent Intent Modeling

Instead of representing session intent as a fixed vector, RIDRec adopts a Neural-Process-style latent-variable formulation.

The latent distribution captures uncertainty in the user’s underlying intent, which is especially important when only a small number of interactions are observed. Retrieved cross-session information is incorporated into the intent distribution as additional contextual evidence.

### 4. Diffusion-Based Intent Refinement

RIDRec further introduces a conditional diffusion process to refine the latent intent representation.

Starting from a noisy latent variable, the reverse diffusion process gradually reconstructs an intent representation conditioned on the current session and the retrieved context. This process enables the model to represent multiple plausible intents and reduce the influence of uncertainty in short sessions.

### 5. Next-Item Prediction

The refined intent representation is combined with the current session representation to estimate matching scores over candidate items.

The model then ranks candidate items according to these scores and predicts the item most likely to be selected next.

## Main Contributions

1. A retrieval-enhanced framework that supplements limited short-session observations with relevant cross-session information.

2. A latent-variable formulation that represents uncertain session intent as a distribution instead of a deterministic vector.

3. A conditional diffusion mechanism that progressively refines latent intent representations for next-item recommendation.

## Data Preprocessing

The repository retains a general preprocessing utility for session-based recommendation datasets.

The input should be a CSV file containing the following columns:

```text
session_id,item_id,timestamp
```

The preprocessing procedure includes:

- sorting interactions by session and timestamp;
- filtering short sessions;
- filtering low-frequency items;
- splitting sessions chronologically into training, validation, and test sets;
- removing validation and test items that do not appear in the training set;
- remapping item identifiers;
- constructing prefix–target examples for next-item prediction.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

### Running Preprocessing

```bash
python -m scripts.prepare_sessions \
  --input path/to/interactions.csv \
  --output-dir path/to/processed_data \
  --session-col session_id \
  --item-col item_id \
  --time-col timestamp
```

Optional preprocessing parameters include:

```text
--min-session-length
--min-item-support
--train-ratio
--valid-ratio
```

The processed files are saved in the specified output directory:

```text
train.jsonl
valid.jsonl
test.jsonl
metadata.json
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{liu2026ridrec,
  title     = {RIDRec: Retrieval-Enhanced Intent Diffusion for Anonymous Short-Session Recommendation},
  author    = {Liu, Peilin and Ji, Zhiquan and Yan, Gang},
  booktitle = {Proceedings of the 2026 ACM SIGIR International Conference on the Theory of Information Retrieval},
  year      = {2026},
  doi       = {10.1145/3805713.3820428}
}
```

> **Note:** The complete model implementation and detailed reproduction instructions are currently being organized and will be released in this repository soon.
