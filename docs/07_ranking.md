# Step 7 — Ranking / Reranking Stage

## Why this step exists

Retrieval (Steps 4-6) has one job: narrow the whole 25,612-item catalog down to a short,
plausible list, cheaply. It's not supposed to be precise -- ALS scores every item with one
matrix multiplication, the Transformer generates candidates token-by-token. Neither can
afford to pull in signals like price or how recently an item has been trending, for every
item, every time. Ranking is the second half of the two-stage pattern: take the ~100
candidates retrieval already produced, and spend a little more compute per candidate to
reorder them properly.

## Why ALS as the candidate source here, not the Transformer

The architecture diagram treats "retrieval" as one abstract stage. In this project, both
ALS (Step 4) and the Semantic ID Transformer (Step 6) are retrieval models. Training a
ranker needs candidates for thousands of users, and the Transformer's constrained beam
search is a forward pass per generation step -- Step 6 deliberately evaluated it on only a
2,000-user sample because it doesn't scale further inside this sandbox's time budget. ALS
generates top-100 candidates for 5,000 users, batched, in under 2 seconds -- an actual
matmul over the whole catalog, which is what "cheap, wide retrieval" is supposed to mean.
So ALS is the candidate source here, used honestly as a stand-in for "whatever retrieval
stage a production system picks" -- the ranking logic itself (features, model, evaluation
methodology) doesn't care which retrieval stage produced the candidates.

## Features

Seven features per (user, item) candidate pair, computed in
`backend/models/ranking_features.py`, entirely vectorized (no python loops over users or
items -- that's what keeps candidate generation + feature assembly for thousands of users
under 5 seconds):

| feature | what it captures |
|---|---|
| `als_score` | retrieval's own confidence, carried forward so the ranker can learn how much to trust it |
| `item_popularity_log` | log1p of how often this item was bought overall in train |
| `item_price_imputed` | price, imputed where missing |
| `has_price` | 1 if the price was actually observed, 0 if imputed |
| `item_recency_norm` | how recently, on average, this item has been bought -- trending vs. stale |
| `content_sim` | cosine similarity between the item's Sentence-T5 embedding (Step 5) and the user's profile embedding (mean of their train-history item embeddings) |
| `user_n_interactions_log` | log1p of the user's train history length -- a rough proxy for how much signal exists to personalize with |

**The missing-price handling, done the way `docs/01_dataset.md` said it would be:** 24.7%
of interaction rows have no price (delisted items, third-party sellers, prices not
captured at crawl time -- a genuine source-data gap, not a bug). The original plan was
category-median imputation, but this dataset has no clean category taxonomy to group by,
so imputation here is a simpler, honestly-scoped version: global median price, plus a
`has_price` flag so the model can still distinguish "actually cheap" from "we don't know."

**`content_sim` is the direct link back to Steps 5-6:** it's the same Sentence-T5 embedding
space the RQ-VAE quantizes into Semantic IDs, just used continuously here instead of
discretized. A user whose history is all gaming peripherals gets a high `content_sim` score
on other gaming peripherals, independent of whether ALS happened to surface them.

## Model: LightGBM, LambdaMART

`backend/scripts/train_ranker.py`. Learning-to-rank, not plain classification --
[`LGBMRanker`](https://lightgbm.readthedocs.io/) with `objective="lambdarank"`, the
standard choice for production ranking stages (this is the same family of method used at
Bing, and widely in industry recsys rankers). Rows are grouped per user (LightGBM needs
contiguous groups to know which candidates compete against which), and the objective
directly optimizes NDCG rather than treating every row as an independent classification.

**Training data:** 12,000 users sampled from the ~26,838 users who are both warm (have
train history) and have at least one val interaction. Each user's ALS top-100 candidates
become training rows; a candidate is labeled 1 if it's in that user's val_targets, 0
otherwise. Val -- not train -- supplies the labels here, the same reasoning as any
two-stage model family: retrieval was already fit on train, so the second stage needs a
different slice of held-out signal to learn from, or it would just be re-learning what ALS
already knows.

**Positive injection:** in 27,401 cases, a user's true next-purchased item wasn't in ALS's
own top-100 -- retrieval missed it. Those get added to the candidate set anyway (with a
neutral placeholder `als_score`), because otherwise the ranker would never see what a real
positive example looks like for that user at all. 1,227,401 total training rows, 29,344
positives (2.4%).

**Feature importances** (gain-based split count, from the trained model):

| feature | importance |
|---|---|
| `als_score` | 1516 |
| `user_n_interactions_log` | 1361 |
| `item_popularity_log` | 1205 |
| `item_recency_norm` | 842 |
| `content_sim` | 670 |
| `item_price_imputed` | 353 |
| `has_price` | 53 |

Retrieval's own score dominates, as expected -- it's already a strong single signal. But
`content_sim` and `item_recency_norm` both pull real weight, meaning the ranker is
genuinely using information ALS didn't have, not just rediscovering ALS's ranking.

## Evaluation: the fair comparison

The only honest way to measure a reranker is on the *same candidate set*, reordered or
not. Reranking can never add an item that wasn't already retrieved -- recall@100 is fixed
the moment ALS generates the shortlist. So the real question is whether reordering moves
true positives up into the top 10/20.

Two samples, both excluded from ranker training:
- **val (held-out):** 5,000 users sampled from the ~14,838 eligible val users *not* used to
  train the ranker.
- **test:** 5,000 users with test-period purchases, never touched during training at all --
  this is the number that matters most.

| split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|
| val | als candidates (unranked) | 0.0159 | 0.0240 | 0.0097 | 0.0123 |
| val | als + reranked | 0.0141 | 0.0234 | 0.0088 | 0.0116 |
| test | als candidates (unranked) | 0.0050 | 0.0081 | 0.0029 | 0.0038 |
| test | **als + reranked** | **0.0057** | **0.0094** | **0.0030** | **0.0041** |

![ALS candidates vs. reranked, val and test](../data/processed/ranking_comparison.png)

Another honest mixed result, in the same spirit as Step 6's: on the held-out val sample,
reranking is slightly *worse* than leaving ALS's own order alone. On test -- the sample
that never touched training in any way -- reranking wins on every metric, recall@20 up
14% relatively (0.0081 → 0.0094). Two things worth saying plainly rather than picking
whichever number flatters more: the val drop could be the ranker slightly overfitting to
the 12,000-user training population's specific patterns; and with only 5,000 users and a
2-3% positive rate per split, these differences sit close to the noise floor of the sample
size -- a larger eval sample would be the natural next thing to run to know if the test
improvement is real signal or a lucky draw, not something to treat as a settled result on
5,000 users.

## A concrete example

User 1510's recent train history: a Razer gaming keypad, a PS4 console bundle, an Xbox
controller, a Redragon gaming mouse -- a clear "PC/console gaming peripherals" buyer. Their
actual next test-period purchase was a **Logitech G502 HERO gaming mouse**.

ALS's own ranking buried it at position 98 out of 100 candidates -- present in the
shortlist, but nowhere a user would see it. The reranker moved it to **position 1**. The
feature that did the most work: `content_sim` for this item was 0.89 -- almost as similar
as two embeddings get -- because a gaming mouse sits right in the middle of the embedding
neighborhood defined by this user's keypad, controller, and other mouse purchases. ALS's
score alone (0.216, unremarkable among the 100 candidates) had no way to know that; the
content signal from Step 5 did.

## What's next

Step 8 formalizes the evaluation this project has been doing piecemeal, stage by stage,
into one script that runs every model tier (popularity → ALS → Transformer retrieval →
ALS + ranking) against the same val/test sets and produces a single comparison table --
the actual "here's what each layer bought us" evidence for the write-up.

## Outputs

- `backend/models/ranking_features.py` -- shared feature-building code (item features, user
  profile embeddings, candidate assembly).
- `backend/scripts/train_ranker.py`, `evaluate_ranking.py` -- training and evaluation
  pipelines.
- `data/processed/ranker_model.pkl` -- trained LightGBM ranker.
- `data/processed/ranker_train_meta.json` -- training config, sample sizes, feature
  importances.
- `data/processed/ranking_eval_results.json` -- recall@K/NDCG@K, unranked vs. reranked, on
  held-out val and test samples.
- `data/processed/ranking_comparison.png` -- results plot.
