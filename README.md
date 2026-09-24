# Two-Stage Generative Recommender (E-Commerce)

A portfolio project implementing a modern, research-grounded recommendation system:
**generative candidate retrieval (Semantic IDs, TIGER-style) → learned-to-rank reranking**,
served via FastAPI and a React demo UI.

## Why this architecture

Real-world recommenders (Amazon, YouTube, Meta) don't use one model — they use a
**two-stage pipeline**:

1. **Retrieval** — cheaply narrow millions of items down to a few hundred candidates
   per user.
2. **Ranking** — a more expensive, accurate model scores and orders those candidates.

This project implements both stages, plus baselines to benchmark against:

| Stage | Model | Reference |
|---|---|---|
| Baseline | Popularity | — |
| Baseline | Matrix Factorization (ALS, implicit feedback) | classic CF |
| Retrieval | RQ-VAE Semantic ID tokenizer + autoregressive Transformer | [TIGER, Google DeepMind, NeurIPS 2023](https://arxiv.org/abs/2305.05065) |
| Retrieval/Ranking inspiration | HSTU generative sequential transducer | [Meta, "Actions Speak Louder Than Words," 2024](https://github.com/facebookresearch/generative-recommenders) |
| Ranking | Feature-based reranker (collaborative + content signals) | industrial two-tower/DCN pattern |

Research grounding is documented per-stage in `docs/` as it's built, so each piece can be
explained in an interview: what problem it solves, why this method vs. alternatives, and
what its measured impact was (offline metrics).

## Project structure

```
backend/
  app/          FastAPI service (serving layer)
  models/       model implementations (baseline, RQ-VAE, retrieval transformer, ranker)
  scripts/      training / data-prep scripts
data/
  raw/          downloaded dataset (not committed)
  processed/    cleaned interactions, sessions, splits (not committed)
notebooks/      exploration notebooks
frontend/       React demo app
docs/           architecture notes + paper references per component
```

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) — **5-core**
`Video_Games` subset: 94,762 users, 25,612 items, 814,586 interactions, every user/item
guaranteed ≥5 interactions by construction. Real product titles, real timestamps (1999–2023).
This is the same dataset family used by the TIGER/HSTU-style papers this architecture is
based on, so results are directly comparable to published benchmarks.

Full rationale — including why the raw (non-5-core) reviews and an earlier UCI Online Retail
placeholder were both tried and rejected first — is in `docs/01_dataset.md`. Short version:
raw Amazon reviews are 93.9% one-time reviewers (no history to model from); 5-core filtering
fixes that by construction.

## Evaluation methodology

Time-based train/val/test split (train on the past, predict the future — no leakage from
random splits). Metrics: Recall@K, NDCG@K, MAP@K, compared across every model tier so the
value of each added stage of complexity is measurable, not assumed.

## Status

Build in progress — see task list. Each stage is built, explained, and evaluated before
moving to the next.


## Tutorial notebooks (run each section on Colab)

| section | notebook |
|---|---|
| Data, split, popularity & ALS baselines, metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_01_data_and_baselines.ipynb) |
| Semantic IDs (Sentence-T5 + RQ-VAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_02_semantic_ids.ipynb) |
| Generative retrieval (Transformer + constrained beam search) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_03_generative_retrieval.ipynb) |
| Ranking (LightGBM LambdaMART, with the injection bug) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_04_ranking.ipynb) |
| Evaluation: time split vs leave-one-out, SASRec | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanmigutierrez/generative-recommendation-engine/blob/main/notebooks/tutorial_05_evaluation.ipynb) |

The notebooks download pre-computed artifacts from the [v1.0-artifacts release](https://github.com/juanmigutierrez/generative-recommendation-engine/releases/tag/v1.0-artifacts); see .
