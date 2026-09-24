# Step 10 — Paper-comparable evaluation (leave-one-out protocol)

## Why this step exists

`docs/09_audit_sources_and_results.md` found that the reason nothing beat popularity in the
Step 8 table was mostly the *task*, not the models: the global time split asks a model to
predict which games a user reviews in 2021-2023 from ~4 reviews written before Nov 2019,
with a median gap of 1,397 days and 72% of test target items never seen in training. TIGER,
SASRec and essentially every sequential-recommendation paper evaluate something different:
**leave-one-out next-item prediction**, where the context ends immediately before the
target and the target almost always exists in training. The two protocols answer different
questions and their numbers are an order of magnitude apart, so this step adds the paper's
protocol as a second evaluation track, keeps the time split as the "production-realistic"
one, and adds the baseline the papers actually compare against (SASRec).

## Protocol (`backend/scripts/split_data_loo.py`, `run_loo_evaluation.py`)

Per user, interactions sorted by time and de-duplicated by item:

    i_1 ... i_{n-2}   -> training history
    i_{n-1}           -> val target   (context = i_1 .. i_{n-2})
    i_n               -> test target  (context = i_1 .. i_{n-1})

- 94,762 users; median training history 4 items, mean 6.6, p95 17.
- Cold targets: 0.17% (val), 0.07% (test) -- versus 32% / 72% on the time split.
- Every model's output has the user's own context items filtered out (the paper setting).
- Metrics: Recall@K and NDCG@K, K in {5, 10, 20}. With one target per user, Recall@K is
  HitRate@K, the number SASRec reports; TIGER reports Recall@5/10 and NDCG@5/10.
- Evaluated on **all 94,762 users** on both splits (no sampling).

## Models on this track

| tier | script | notes |
|---|---|---|
| popularity | `run_loo_evaluation.py` | fit on LOO train rows |
| ALS | `run_loo_evaluation.py` | same hyper-parameters as Step 4, fit on LOO train rows |
| **SASRec** | `train_sasrec.py`, `models/sasrec.py` | Kang & McAuley 2018 -- same causal Transformer backbone as the Semantic-ID model but over raw item IDs, scored by dot product with the item table. 2 layers, d=64, dropout 0.2, 20-item context, full-softmax CE (the "SASRec+" variant). 36 epochs on a Colab GPU, final loss 6.48. |
| **Semantic-ID transformer** | `train_transformer.py --loo`, `build_semantic_sequences.py --loo --max-items 20 --user-buckets 2000` | TIGER-like configuration: 4 layers, d=128, d_ff=512, dropout 0.1, hashed user-ID token (2,000 buckets) prepended, 20-item context, constrained beam search (width 30, then filtered to 20). 45 epochs on a Colab GPU, final loss 2.16 per Semantic-ID digit. Decoder-only rather than TIGER's T5 encoder-decoder. |

Training was done with `notebooks/colab_loo_training.ipynb` (T4 GPU; both trainers resume
from the CPU checkpoints that were produced first).

## Results (all 94,762 users per split)

| split | model | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|---|---|
| val | popularity | 0.0151 | 0.0268 | 0.0415 | 0.0099 | 0.0136 | 0.0173 |
| val | als | 0.0537 | 0.0833 | 0.1241 | 0.0350 | 0.0446 | 0.0548 |
| val | **sasrec** | **0.0630** | **0.0958** | **0.1405** | **0.0415** | **0.0521** | **0.0633** |
| val | transformer (Semantic ID) | 0.0482 | 0.0765 | 0.1147 | 0.0315 | 0.0406 | 0.0502 |
| test | popularity | 0.0137 | 0.0249 | 0.0379 | 0.0091 | 0.0126 | 0.0158 |
| test | als | 0.0372 | 0.0569 | 0.0846 | 0.0243 | 0.0307 | 0.0376 |
| test | **sasrec** | **0.0520** | **0.0792** | **0.1168** | **0.0347** | **0.0434** | **0.0529** |
| test | transformer (Semantic ID) | 0.0443 | 0.0690 | 0.1031 | 0.0292 | 0.0372 | 0.0457 |

For reference, TIGER's paper (Amazon Beauty 2014, same protocol): SASRec Recall@10 0.0605 /
NDCG@10 0.0318, TIGER Recall@10 0.0648 / NDCG@10 0.0384. Video_Games is a different
category and dataset year, so the absolute numbers are not directly comparable; the
comparison that is valid is *within this table*.

## Reading the results

1. **The generative model works.** On the paper's protocol the Semantic-ID transformer gets
   Recall@10 of 0.077 / 0.069 (val / test) -- 2.8x the popularity floor and in the same
   range as the numbers TIGER publishes. The Step 8 table, where it looked no better than
   popularity, was measuring a task that no model in this project can do well, not a broken
   implementation.
2. **It does not beat SASRec here.** SASRec is ahead on every metric, by roughly 13-20%
   relative (val Recall@10 0.096 vs 0.077; test 0.079 vs 0.069). TIGER reports the opposite
   ordering (+7-17% over SASRec) on Beauty/Sports/Toys. Plausible reasons, in order of how
   cheap they are to test:
   - **Undertraining and beam search.** Loss was still falling at epoch 45 (2.16, from
     4.16 at epoch 1); TIGER trains ~200k steps. Beam search with cumulative log-probability
     also favours frequent Semantic-ID prefixes (Step 9 diagnostics showed the old model
     generating only ~500 distinct items across 8,000 slots); a length- or
     frequency-normalised beam score, or a larger beam, is worth trying.
   - **Semantic ID quality.** The RQ-VAE reaches only 28% level-1 codebook utilisation and
     a 25% prefix-collision rate (`rqvae_metrics.json`). The Semantic ID is the entire
     input to this model, so a weak first-level partition caps what it can learn. TIGER
     uses a deeper encoder (768-512-256-128-32) and embeds title + brand + category + price,
     not the title alone.
   - **Architecture.** Decoder-only vs TIGER's T5 encoder-decoder, and a 4/128 model rather
     than TIGER's 4+4/128/1024.
   - **Dataset.** Video_Games has a heavier head (consoles, blockbuster titles) than Beauty;
     an ID-based model can memorise that head directly, whereas the content-derived model has
     to reach it through shared codes.
3. **ALS is a strong baseline under this protocol** (0.083 val), which again says the
   Step 8 ordering was a property of the time split, not of the models.
4. **Val > test for every model** because the test context includes the val item (one more
   item of history) but the test target is, on average, further from the bulk of the
   user's history -- the usual LOO pattern.

## Ranker fix (carried over from the audit)

The Step 7 ranker was retrained without positive injection (`train_ranker.py`): only
users whose true val item was actually among ALS's top-100 candidates contribute training
groups (2,347 of 18,000 sampled), and the users sampled for ranker training are saved to
`ranker_train_users.npy` and excluded from every val evaluation. On held-out users the
reranker now roughly doubles ALS on the time split instead of being a wash:

| split | model | recall@10 | recall@20 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|
| val (held out) | als, ALS order | 0.0138 | 0.0221 | 0.0088 | 0.0114 |
| val (held out) | als, reranked | **0.0271** | **0.0384** | **0.0201** | **0.0235** |
| test | als, ALS order | 0.0046 | 0.0072 | 0.0031 | 0.0039 |
| test | als, reranked | **0.0101** | **0.0128** | **0.0069** | **0.0077** |

## What to try next, in order

1. Train the Semantic-ID model to a loss plateau (another 50-100 epochs on GPU) and
   re-run the LOO table -- the cheapest test of whether the SASRec gap is training budget.
2. Beam search: width 50 with per-item length normalisation; measure distinct-items
   coverage alongside recall.
3. Retrain the RQ-VAE with richer item text (title + store + category + price from the
   raw metadata) and the deeper TIGER encoder; check level-1 utilisation and collision rate
   before retraining the transformer.
4. Rerank SASRec / transformer candidates with the fixed LightGBM ranker under LOO (the
   ranker currently sits on ALS candidates only).

## Outputs

- `backend/scripts/split_data_loo.py`, `run_loo_evaluation.py`, `train_sasrec.py`,
  `backend/models/sasrec.py`; `--loo` flags in `build_semantic_sequences.py` and
  `train_transformer.py`; dropout + `forward_hidden` in `backend/models/transformer.py`;
  vectorised beam search in `evaluate_retrieval.py` (identical output, ~10x faster).
- `notebooks/colab_loo_training.ipynb` -- GPU training + evaluation.
- `data/processed/loo_*.parquet`, `loo_split_stats.json`, `sasrec_checkpoint.pkl`,
  `transformer_checkpoint_loo.pkl`, `loo_evaluation_results.json`.
