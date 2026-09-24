# Audit: are the cited sources actually implemented, and why doesn't the generative stack beat popularity?

Written after reading every file in `backend/models/` and `backend/scripts/`, the docs, and
re-running diagnostics against the trained checkpoints (`backend/scripts/diagnose_retrieval.py`).

## 1. Source-by-source: what the code really does vs. what the paper does

| source | claimed role | verdict |
|---|---|---|
| **TIGER** (Rajput et al. 2023) | Semantic IDs + generative retrieval | **Partially.** RQ-VAE matches (3 levels x 256 codes, latent 32, beta=0.25, k-means codebook init, dedup digit). The generative model does **not**: TIGER is a T5 **encoder-decoder** (4+4 layers, d_model 128, d_ff 1024, 6 heads, dropout 0.1, ~200k steps, batch 256, a hashed **user-ID token** prepended to the sequence, beam width 10). Ours is a 2-layer decoder-only, d_model 64, d_ff 128, no dropout, no user token, ~1,670 optimizer steps (10 epochs x 167 batches). Most importantly TIGER's **evaluation protocol is leave-one-out next-item prediction** (last item = test, second-to-last = val, full prior history up to 20 items as context). Ours is a global time split predicting *every* item in a 2-year window -- a different task (see section 2). TIGER also reports on Beauty / Sports / Toys (2014 dump), never Video_Games, and embeds title + price + brand + category, not title alone. |
| **HSTU** (Zhai et al. 2024) | "other half of the framing" | **Not implemented.** Nothing from HSTU is in the code: no pointwise aggregated attention, no relative-time bias, no action/item interleaving, no M-FALCON. It is cited only for the "retrieval as generation" idea. The blog/docs should say it's *motivation*, not something built. |
| **VQ-VAE** (van den Oord et al. 2017) | quantization + straight-through | **Yes.** `rqvae.forward` has the STE (`z + stop_gradient(z_q - z)`), codebook loss and beta-weighted commitment loss exactly as in the paper. |
| **Vaswani et al. 2017** | Transformer | **Yes** (pre-norm GPT-2-style variant, causal + pad mask, weight tying). Faithful, just tiny. |
| **Sentence-T5** (Ni et al. 2021) | item embeddings | **Yes**, `sentence-t5-base` via sentence-transformers. Input is the product title only. |
| **Hu, Koren & Volinsky 2008** | ALS baseline | **Yes**, via `implicit` with `c_ui = alpha * count` (alpha=40). Note `implicit` adds the +1 internally, so the formulation is right. |
| **LightGBM / LambdaMART** (Ke et al. 2017) | reranker | **The library is used, but the training data is broken** -- see section 3. |

So the citation list is honest for RQ-VAE, VQ-VAE, Sentence-T5, ALS and the Transformer
block. It overstates TIGER (architecture and protocol differ) and HSTU (not implemented at
all), and the blog line "results here are roughly comparable to published numbers" is not
true -- the protocol makes them incomparable.

## 2. Why every model scores ~0.01-0.02: the split, not the models

Measured on `train/val/test.parquet`:

| | val | test |
|---|---|---|
| train items per warm user (median) | 4 | 4 |
| gap from user's last train interaction to first eval interaction (median) | **673 days** | **1,397 days** |
| eval rows whose item never appears in train ("cold items") | 31.9% | **71.8%** |
| eval target items in train's top-100 by popularity | 4.5% | 1.7% |

The test task is: from ~4 reviews written before Nov 2019, predict which video games the
user will review 2021-2023 -- where 72% of those games did not exist in the training data.
No collaborative model (popularity, ALS, ALS+ranker) can *ever* retrieve a cold item, so
their test recall ceiling is ~28% of the target mass, and in practice they retrieve stale
2019 titles. This is why everything collapses to the same floor and why test is so much
worse than val. TIGER's leave-one-out protocol has ~0 cold items and a context that ends
immediately before the target; the paper's Recall@10 of 0.06 and our 0.007 are answers to
different questions.

**Diagnostic (`diagnose_retrieval.py loo`)**: scoring the *same* checkpoint on a
TIGER-style next-item task (predict a user's last train item from the preceding ones,
1,000 users) gives transformer Recall@10 = **0.059**, NDCG@10 = 0.031 vs. popularity
0.038 / 0.019 -- in the paper's ballpark and clearly ahead of popularity. (This is
optimistic because the last item was seen during training; a clean version needs
retraining with the last item held out, but it shows the model *works* under the paper's
protocol.) With every user given someone else's history, Recall@10 drops to 0.017, so the
model is genuinely conditioning on the context, not just emitting popular IDs.

## 3. Real bugs / weaknesses found

1. **Reranker learned the positive-injection artifact (train_ranker.py).** 27,401 of 29,344
   training positives (93%) were *injected* val items that ALS never retrieved. They were
   given `als_score = min score in the list` and, being mostly cold items, `popularity = 0`.
   Probing the trained booster with all other features fixed:
   `als_score` 0.1 -> +0.39, 0.5 -> -4.40; train count 0-5 -> +6.5, 100 -> -4.6.
   The ranker learned "the lowest-ALS-score, never-seen item is the positive" -- the exact
   opposite of what it should do on real ALS candidates. Fix: do not inject positives with
   fake features (train only on candidates actually retrieved, or inject with feature values
   marked missing/NaN so LightGBM can treat them as such, or use a candidate generator with
   better recall@100 so injection is rare).
2. **Val results for `als + ranking` in Step 8 are leaked.** The ranker was trained with
   val-period labels for 12,000 warm users; 715 of the 2,000 users in the Step 8 val sample
   (44% of its warm users) are ranker-training users. Val numbers for that tier are not
   held-out. Test is clean.
3. **Transformer is badly undertrained and has collapsed diversity.** Loss 2.76 and still
   falling ~0.06/epoch after 10 epochs (~1.7k steps vs TIGER's 200k). Across 8,000
   recommendation slots on val the beam search produced only **492 distinct items**; five
   items account for >10% of all slots. Only 0.2% of generated items are cold, even though
   33% (val) / 72% (test) of targets are -- the one advantage Semantic IDs should give is not
   being exercised. Causes: too few steps, no dropout, beam search on cumulative log-prob
   (which favours frequent prefixes), no user token, context limited to 9 items.
4. **Beam search does not filter already-seen items** (popularity and ALS do). Small effect
   (2.8% of slots) but inconsistent.
5. **Level-1 codebook utilisation is 28%** (73/256 codes) and 25% of items collide on the
   3-digit prefix -- weak first-level partition; TIGER-style deeper encoder
   (768-512-256-128-32) and a collision-avoidance loss/longer training would help.
6. **Item text is title only.** TIGER concatenates title, brand, category, price; adding
   the metadata already in `Video_Games_meta.parquet` is cheap.

## 4. What to change to get paper-comparable results

1. Add a **leave-one-out protocol** alongside the time split: per user, last item = test,
   second-to-last = val, train on the rest, context = up to 20 previous items. Report both
   tables; keep the time split as the "production-realistic" one and say explicitly that it
   is a harder, different task (and that the popularity floor there is ~0.01).
2. Retrain the ranker without fake-feature injection, and evaluate it on users disjoint from
   its training users.
3. Train the transformer much longer (target loss plateau, not 10 epochs), add dropout 0.1,
   prepend a user-ID token, use context length 20 items, and consider length-normalised
   beam scores or temperature to restore diversity.
4. Replace the blog's "comparable to published numbers" sentence with the protocol
   difference, and describe HSTU as inspiration rather than implementation.


## Status (after Step 10)

- Sections 2 and 4.1: done -- `docs/10_loo_protocol.md`. Under the leave-one-out protocol
  the Semantic-ID transformer reaches Recall@10 0.077 / 0.069 (val / test), 2.8x popularity
  and in TIGER's published range; SASRec is ~15% ahead of it on this dataset.
- Bug 1 and 2 (ranker): fixed in `train_ranker.py` / `evaluate_ranking.py` /
  `run_full_evaluation.py`; the reranker now doubles ALS on held-out users.
- Bug 3 (undertraining): the LOO model was trained 45 GPU epochs (loss 4.16 -> 2.16, still
  falling slowly); beam search is now vectorised so a full-population eval is feasible.
- Bug 4 (seen-item filtering in beam search): done in `run_loo_evaluation.py`.
- Bug 5 (RQ-VAE utilisation) and 6 (title-only text): still open -- the main remaining
  suspects for the SASRec gap.
- Blog: comparability sentence rewritten; HSTU labelled as framing, not implementation;
  SASRec added to the sources.
